# On-Demand RF-DETR Person Search Design

## Summary

Replace continuous YuNet/RF-DETR tracking with one explicit, bounded command:
"look at me." Robit searches for at most ten seconds using RF-DETR, stops on
the first valid person detection, and remains facing that general direction
with his head at a fixed head-height pose.

This is a hard cleanup. YuNet, face landmarks, angular gaze correction,
detector comparison modes, and the continuous tracking controller are removed
rather than disabled.

## Behavior

The deterministic text and voice phrases "look at me" and "find me" start one
person search. Only one search may run at a time.

At the start of a search, Robit:

1. Cancels any previous search.
2. Commands the head to pan `90` and tilt `75`.
3. Waits for the head command to complete and for a fresh camera frame.
4. Runs RF-DETR once on that frame.

If RF-DETR returns any person at or above the configured person confidence,
the first returned person wins. Robit sends an ordinary stop command, remains
at the current body orientation and fixed head-height pose, and reports that
he found someone. He does not center the person, publish a corrective head
target, or continue tracking.

If no person is detected, Robit performs one `350` ms right pivot at speed
`140`, waits `500` ms after the bounded movement ends, obtains a frame captured
after settling, and tries RF-DETR again. This repeats until a person is found
or ten seconds have elapsed.

On timeout, Robit sends an ordinary stop command, leaves his head at the
head-height pose, and reports that he could not find anyone. Manual head,
drive, and stop commands cancel an active search before taking control.

## Architecture

### PC Brain

Replace `PersonTrackingService` with a small `PersonSearchService`. It owns
only:

- idle, positioning, detecting, turning, found, timed-out, cancelled, and
  faulted states;
- one active search task;
- the ten-second deadline;
- fresh-frame RF-DETR requests;
- bounded body pivots and settling;
- cancellation generation;
- compact status and command counters.

The service does not maintain a target between frames, estimate velocity,
associate people, predict motion, calculate gaze angles, or run a background
detection loop.

The existing frame broker remains the only source of robot-camera frames.
Search requests ask for a frame newer than the completed head or body command,
so a frame captured during movement cannot authorize success or another turn.

The actuator broker remains the only source of ESP commands. Search pivots use
bounded movement durations. An ordinary stop supersedes movement after a
detection, timeout, cancellation, or fault.

### Tracking Sidecar

The sidecar becomes RF-DETR person-only:

- remove YuNet initialization and inference;
- remove face model asset downloading;
- remove `shadow`, `face_only`, and `person_only` mode selection;
- remove face boxes and landmarks from the response schema;
- retain the existing JPEG request, frame identity checks, person confidence,
  queue timing, inference timing, backend status, and isolated RF-DETR
  dependency.

If OpenCV is no longer used anywhere in the sidecar after cleanup, remove its
dependency and model files.

### Commands and APIs

Deterministic text and realtime voice routing recognize "look at me" and "find
me" and invoke the same search operation. The model does not need to infer a
sequence of turns.

Expose:

- `POST /tracking/look-at-me` to start and await a search;
- `GET /tracking/status` for current state, elapsed time, remaining time,
  attempts, last detection confidence, last error, and command counts.

Remove `/tracking/start`. Keep `/tracking/stop` only as a cancellation alias
for an active search; it does not manage a persistent tracking mode. Existing
manual robot endpoints remain unchanged.

## Configuration

Keep only person-search settings:

- RF-DETR base URL and request timeout;
- person confidence;
- search timeout, default `10` seconds;
- head-height pan, default `90`;
- head-height tilt, default `75`;
- right-pivot speed, default `140`;
- pivot duration, default `350` ms;
- post-command settling interval, default `500` ms.

Remove YuNet confidence, minimum face size, detector mode, camera field of
view, angular gains, angular deadband, continuous detector cadences, face
association timers, and continuous pivot timers.

## Failures and Cancellation

A detector error, camera timeout, rejected actuator command, or control-channel
disconnect ends the search as `faulted`, sends an ordinary stop when the
control channel permits it, and reports the exception class and stage.

Cancellation is generation-based. Results from an older camera request,
detector request, head command, or turn cannot mutate the new search or issue
another actuator command.

Search confirmation is spoken only after the final stop acknowledgement.
Failures and timeouts are reported as failures or timeouts, never as success.

## Observability

Do not store image bytes. Journal one event for search start and one terminal
event for found, timeout, cancelled, or faulted. Each detection attempt emits
a bounded numeric sample containing frame ID, attempt number, person
count, best confidence, detector latency, camera latency, and current search
state.

Health reports RF-DETR availability and latency. Tracking status no longer
contains face geometry, angular calculations, estimator confidence, target
age, or shadow comparison statistics.

## Tests

Tests cover:

- head-height positioning precedes the first detection;
- the first fresh frame containing any person ends the search;
- detection at the edge of the frame causes no centering correction;
- a miss causes one bounded right pivot and a post-settling fresh frame;
- a person found after a pivot causes stop and no further turn;
- timeout occurs within the ten-second bound and stops movement;
- manual head, drive, and stop cancel the search;
- stale results from cancelled searches cannot issue commands;
- only one search runs at a time;
- detector, camera, actuator, and disconnect failures are reported accurately;
- no background detector requests occur while idle;
- the sidecar returns person-only results and has no YuNet/OpenCV model path;
- deterministic text and voice phrases invoke the same operation;
- status and journal samples contain no image bytes.

## Acceptance

From idle, saying "look at me" causes Robit to place his head at the configured
head height and search to the right. He stops on the first RF-DETR person
detection without attempting to center or follow that person. The operation
finishes within ten seconds, remains idle afterward, and sends no further
camera, detector, head, or body tracking commands until another explicit
search request.
