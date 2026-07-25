# Robit Face-First Tracking Design

## Summary

Replace RF-DETR as Robit's primary gaze sensor with OpenCV YuNet face
detection. YuNet runs on CPU against frames already supplied by the shared
camera broker and publishes face boxes plus five facial landmarks. The
tracking controller aims at the midpoint between the detected eyes rather than
estimating face height from a general person box.

RF-DETR remains temporarily available as a low-rate person-reacquisition
fallback. It cannot directly override a fresh face target, and it does not run
continuously while a face is stable. The existing vision-language model remains
responsible for semantic scene and object understanding.

The cutover is gated by a shadow evaluation on actual Robit camera frames.
YuNet is the selected candidate, but it must beat the current tracker on
detection retention, aim-point stability, false positives, latency, and compute
load before it is permitted to control the head.

## Goals

- Keep Robit's visual attention centered on a visible face.
- Reduce GPU contention and detector latency.
- Preserve calm gaze through brief occlusion or head turns.
- Retain rough person reacquisition when no face is visible.
- Avoid face recognition, identity embeddings, or stored biometric data.
- Keep all routine frames memory-only and use the shared frame broker.
- Preserve the browser-facing tracking API where practical.

## Non-Goals

- Recognizing who a person is.
- Inferring emotion, attention, or intent from facial landmarks.
- Replacing the vision-language model for object or scene understanding.
- Increasing robot camera acquisition above the existing 2 FPS ceiling.
- Allowing raw detector output to control actuators directly.
- Fine-tuning a face detector before the stock models are evaluated on Robit's
  camera.

## Model Selection

### Selected primary detector: OpenCV YuNet

YuNet is a small ONNX face detector supported by OpenCV's
`FaceDetectorYN`. It returns a face box, confidence, and five landmarks: both
eyes, nose tip, and both mouth corners. The published model is approximately
338 KB. Its paper reports 1.6 ms inference on an Intel i7-12700K at 320x320,
and the OpenCV model documentation describes detection support for faces around
10x10 through 300x300 pixels.

YuNet is preferred because Robit's source frames are low resolution and the
required output is a stable facial aim point. CPU execution also avoids
competing with speech and language/vision workloads on the GPU.

References:

- <https://docs.opencv.org/master/d0/dd4/tutorial_dnn_face.html>
- <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- <https://link.springer.com/article/10.1007/s11633-023-1423-y>

### Evaluated alternatives

- **MediaPipe BlazeFace:** viable runner-up with six facial points and a
  streaming mode, but its 128x128 detector input is less compelling for
  Robit's small faces. It remains the fallback candidate if YuNet underperforms
  on profile faces in the shadow evaluation.
- **SCRFD:** strong difficult-face accuracy, but the supplied InsightFace model
  weights have non-commercial-research licensing restrictions and add a larger
  runtime surface.
- **MediaPipe Pose Landmarker:** useful for body landmarks, but performs more
  work than gaze requires and is optimized around a sufficiently visible body.
- **RF-DETR Nano:** remains useful for rough person geometry, but is not suited
  to facial gaze because it returns only general person boxes.

## Architecture

### Detector boundary

Keep detection isolated from PC Brain behind a local sidecar interface. The
sidecar receives a frame ID, capture timestamp, rotated JPEG bytes, and score
threshold. It returns:

```json
{
  "frame_id": "opaque-frame-id",
  "captured_at": "ISO-8601 timestamp",
  "backend": "opencv/yunet/cpu",
  "latency_ms": 4.2,
  "queue_ms": 0.1,
  "faces": [
    {
      "confidence": 0.94,
      "bounding_box": [0.30, 0.18, 0.58, 0.56],
      "right_eye": [0.38, 0.31],
      "left_eye": [0.50, 0.31],
      "nose": [0.44, 0.39],
      "right_mouth": [0.39, 0.47],
      "left_mouth": [0.49, 0.47]
    }
  ]
}
```

All coordinates are normalized to the rotated frame. Results for a different
frame ID or timestamp are rejected. The sidecar does not acquire camera frames,
retain images, recognize identities, or issue actuator commands.

The detector interface is typed around faces rather than preserving RF-DETR's
person-only response. RF-DETR remains behind a separate person-detector
adapter, so either detector can be changed without rewriting the tracking
controller.

### Tracking controller

The tracking controller is the sole owner of target association, temporal
estimation, gaze publication, and body-pivot authorization.

For a face target:

1. Derive the measured gaze point from the midpoint between the two eye
   landmarks.
2. Reject malformed landmark geometry and faces below the configured minimum
   pixel size.
3. Acquire after two consistent detections within two seconds.
4. Associate by temporal position, box overlap, scale change, and predicted
   motion. Do not use face embeddings.
5. Require three consistently superior detections before switching faces.
6. Feed the gaze point into the existing bounded alpha-beta estimator.
7. Publish desired gaze through the actuator broker using the existing
   deadband, coalescing, and rate limits.

The detector never sends individual servo steps and never authorizes movement
by itself.

### Face-first state flow

The controller tracks both a state and a target source:

```text
face_searching
  -> face_acquiring
  -> face_tracking
  -> face_grace
  -> body_reacquiring
  -> body_fallback
```

- **face_searching:** YuNet runs at 2 FPS. No stale target can move the body.
- **face_acquiring:** two consistent face detections are required.
- **face_tracking:** YuNet runs at 2 FPS initially and may drop to 1 FPS after
  the gaze is stable. Face geometry is authoritative.
- **face_grace:** preserve the last reasonable gaze for up to 2.5 seconds.
  Prediction is capped at 0.5 seconds and does not count as a detection.
- **body_reacquiring:** after 2.5 seconds without a face, RF-DETR may run at
  1 FPS against the same shared frames. YuNet continues searching.
- **body_fallback:** a confirmed person target may guide rough upper-body gaze
  and one bounded repositioning pivot. It cannot replace a fresh face target.
- When the face detector returns two consistent observations, ownership returns
  to face tracking and all body-based estimator state is discarded.
- After five seconds with neither a face nor a person, issue one neutral head
  target and remain in face searching without repeated body search movement.

Frames captured during a pivot or settling interval update neither the
face-gaze estimator nor body-movement authorization. The next post-settling
frame establishes a new screen-space estimate.

### Workload behavior

- YuNet runs on CPU and is outside the GPU workload queue.
- RF-DETR runs only during body reacquisition or controlled shadow evaluation.
- Active voice may still reduce RF-DETR cadence, but does not need to pause
  YuNet.
- Explicit VLM requests retain their existing workload priority and consume the
  latest shared frame rather than opening a new camera poller.
- Only one rotated JPEG variant is produced for both detectors.

## Failure Handling

- If YuNet fails to load, tracking reports degraded face detection and may use
  RF-DETR fallback without claiming facial focus.
- If RF-DETR is unavailable, Robit holds the last face gaze through the grace
  period, then returns to neutral without body searching.
- Detector timeouts contain the detector name, exception class, stage, frame
  ID, queue time, and inference time.
- Stale results are logged and ignored.
- Invalid landmark geometry is treated as a missed face, not as a controller
  fault.
- Reconnection never replays a gaze or body command derived from an expired
  frame.

## Public Status and Observability

Extend `/tracking/status` with:

- `target_source`: `face`, `person_fallback`, or `none`
- `face_box`, normalized landmark coordinates, and face confidence
- face target age and estimator confidence
- YuNet cadence, queue latency, and inference latency
- fallback detector state and activation reason
- counts for face acquisitions, face losses, source changes, head commands,
  pivots, stale results, and invalid landmarks

At 1 Hz, `tracking.sample` records normalized geometry, target source,
association result, desired/actual head position, and detector timing. Routine
image bytes are never journaled.

## Shadow Evaluation

Before YuNet can control actuators, run it in shadow mode alongside the current
tracker on 300 to 500 real Robit frames. Include:

- frontal, profile, looking down, and briefly occluded faces
- near, typical-room, and far distances
- seated and standing positions
- bright, dim, and backlit scenes
- a second person entering and crossing the frame
- empty-room and face-like-object negative scenes
- frames immediately before and after head or body movement

YuNet and BlazeFace full-range may both be evaluated in the same harness. No
shadow detector result may issue an actuator command.

## Acceptance Criteria

YuNet becomes the primary detector only if the Robit-camera evaluation shows:

- at least 95% face retention after acquisition while a usable face remains
  visible
- reacquisition within two detector frames after a face returns
- fewer than 1% false-positive frames in the negative set and no persistent
  false target lasting two frames
- normalized eye-midpoint jitter below 0.03 p95 for a stationary person
- CPU inference latency below 20 ms p95 at the production frame size
- no GPU allocation by the face detector process
- no additional robot camera requests beyond the shared 2 FPS ceiling
- no more than four tracking head targets per second
- no body pivot while a stable face is centered
- no duplicate pivot caused by pre-pivot estimator state

The physical acceptance run must then retain a stationary face at least 95% of
the time for ten minutes and complete a thirty-minute voice-plus-tracking run
without camera, detector, or control-channel failure.

If YuNet misses the thresholds but BlazeFace passes them, BlazeFace becomes the
primary face detector without changing the controller contract. If neither
passes, keep RF-DETR active and collect failure-specific frames before
considering fine-tuning.

## Rollout and Rollback

Introduce a configuration-selectable detector mode:

- `shadow`: current RF-DETR control with YuNet measurement only
- `face_first`: YuNet control with RF-DETR fallback
- `person_only`: previous RF-DETR behavior for rollback

Rollout proceeds from shadow evaluation to an attended ten-minute physical
test, then the thirty-minute mixed workload test. A configuration rollback must
not require firmware changes.

