# Calibrated Angular Gaze Controller Design

## Summary

Replace Robit's repeated fixed-gain head steps with one calculated absolute
head target derived from the detected eye midpoint, the camera field of view,
and the ESP's reported actual head position. The firmware continues to own
velocity- and acceleration-limited servo motion, so the PC can request the
correct destination immediately without making the physical movement abrupt.

This changes only face-gaze publication. YuNet detection, temporal face
association, manual leases, actuator coalescing, pivot settling, and calm
face-loss behavior remain intact.

## Root Cause

The current controller converts normalized image error into servo steps with:

```text
servo_step = image_error * 40
```

Each step is capped at 18 degrees. Stable face detections run at 1 FPS, so an
under-sized correction necessarily waits roughly another second for every
residual step. Temporal smoothing further reduces the early correction.

The camera already supplies enough geometry for a better estimate. For a
62-degree horizontal field of view, a face at normalized `x = 0.90` is about
25.7 camera degrees from center, while the current linear rule requests only
16 servo degrees.

## Angular Controller

For a fresh face detection, calculate:

```text
error_x = eye_midpoint_x - 0.5
error_y = eye_midpoint_y - 0.5

camera_angle_x = atan(2 * error_x * tan(horizontal_fov / 2))
camera_angle_y = atan(2 * error_y * tan(vertical_fov / 2))

pan_correction  = degrees(camera_angle_x) * pan_gain * pan_sign
tilt_correction = degrees(camera_angle_y) * tilt_gain * tilt_sign
```

The absolute command is based on actual telemetry:

```text
target_pan  = actual_pan  + pan_correction
target_tilt = actual_tilt + tilt_correction
```

The control channel records a monotonic receive timestamp whenever hello state
or telemetry arrives. Actual telemetry is fresh when the channel is connected,
`pan_actual` and `tilt_actual` are valid integers, and that timestamp is no more
than one second old. Otherwise, use the controller's last
acknowledged/synchronized head position. Never compound a new correction from
an unacknowledged desired target.

Clamp the final target to the existing servo ranges: pan `55..135`, tilt
`35..115`. Do not apply the old 18-degree incremental step cap. The maximum
camera correction is naturally bounded by half the configured field of view,
and the absolute servo clamp remains authoritative.

## Calibration and Defaults

Add these settings:

- `ROBIT_TRACKING_CAMERA_HORIZONTAL_FOV_DEGREES=62`
- `ROBIT_TRACKING_CAMERA_VERTICAL_FOV_DEGREES=48.5`
- `ROBIT_TRACKING_PAN_ANGLE_GAIN=1.0`
- `ROBIT_TRACKING_TILT_ANGLE_GAIN=1.0`
- `ROBIT_TRACKING_ANGULAR_DEADBAND_DEGREES=2.0`

The default vertical field of view is the 4:3 pinhole equivalent of the
existing 62-degree horizontal field of view. Gains permit physical calibration
without changing controller code. Signs continue to use the existing
`ROBIT_TRACKING_PAN_SIGN` and `ROBIT_TRACKING_TILT_SIGN`.

All field-of-view values must be finite and between 1 and 179 degrees. Gains
must be finite and greater than zero. The angular deadband must be finite and
non-negative.

## Temporal Behavior and Fluidity

The first command after the required two-frame face acquisition uses the fresh
eye midpoint directly rather than the lagging smoothed center. Later frames
calculate a residual angular correction. Prediction remains capped at 0.5
seconds, but prediction cannot make an initial correction larger than the
fresh geometric correction.

No command is emitted when both angular errors are within the configured
two-degree deadband. Tracking publication remains capped at 4 Hz by the
actuator broker.

The PC publishes the absolute destination immediately. The ESP's existing
50 Hz acceleration- and velocity-limited interpolation controls the physical
trajectory, preserving fluid motion. Manual control retains its three-second
lease and forces fresh two-frame face acquisition afterward.

Frames captured or processed during a body pivot or settling interval remain
observation-only and cannot update the gaze estimator or publish a head target.

## Body Pivot Interaction

A newly calculated head target must be allowed to move before body-pivot
authorization. Body pivot confirmation uses persistent residual image error or
measured actual head offset after the head correction, not the size of the
just-issued desired target.

The existing 1.5-second confirmation, bounded pivot duration, one-second
settling period, and five-second opposite-direction block remain unchanged.

## Status and Diagnostics

Extend tracking samples and `/tracking/status` with:

- fresh normalized eye midpoint
- horizontal and vertical angular error
- actual head position used as the calculation origin
- calculated absolute head target
- residual angular error on the next fresh detection
- whether actual telemetry or acknowledged fallback position was used

These are numeric values only. No image bytes are logged or journaled.

## Tests

Unit tests cover:

- centered eyes producing no command
- a face at `x = 0.90` producing approximately 25.7 degrees of pan correction
- symmetric left/right and up/down corrections
- absolute targets originating from actual telemetry
- acknowledged-position fallback when telemetry is unavailable
- servo-range clamping
- configurable calibration gains and signs
- angular deadband behavior
- initial acquisition bypassing lagging temporal smoothing
- residual correction after the first move
- no estimator or command update during pivot settling
- manual override requiring fresh acquisition
- the actuator broker retaining its 4 Hz tracking cap

Existing face association, shadow isolation, loss behavior, endpoint, and full
PC Brain suites remain regression gates.

## Acceptance

In an attended test with a stationary visible face:

- acquisition still requires two consistent detections
- the first post-acquisition head command is issued from fresh eye geometry
- the face enters the angular deadband within one primary head movement plus at
  most one residual correction
- no unnecessary body pivot occurs while the face can be centered within the
  servo range
- physical head movement remains acceleration-limited and visually fluid
- tracking publishes no more than four head targets per second
