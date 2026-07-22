# Robit Architecture

## Recommendation

Build Robit as a distributed system:

```text
Browser / phone
      |
      v
PC brain service  <---- camera / mic / speakers / LLM / tools
      |
      v
ESP robot firmware <---- motors / servos / sensors
```

The ESP should not do vision, LLM calls, speech, or long-running decision logic. It should only do deterministic hardware control and report status. Your PC has the CPU, memory, networking, and storage for everything else.

## Why This Split

The current sketch is a good proof of concept: it starts a Wi-Fi access point, serves a web page, drives the motors, and controls the pan/tilt head. The problem is that it mixes UI, networking, and hardware control in one file. That is fine for testing, but it will get painful once you add camera streaming, autonomous behaviors, and LLM tool calling.

The next version should keep firmware small:

- receive commands
- clamp unsafe values
- stop motors if commands stop arriving
- expose status
- optionally read simple sensors later

The PC brain should handle:

- camera stream ingestion
- object/person detection
- speech-to-text and text-to-speech
- LLM conversation state
- tool calling
- route planning or behavior scripts
- logs and debugging UI

## Networking

Support both modes:

1. **Station mode**: Robit joins your home/dev Wi-Fi. This is the normal mode when the PC brain controls it.
2. **Fallback access point**: Robit creates `Robit-Control` if it cannot join Wi-Fi. This is useful for setup and recovery.

The firmware scaffold implements that pattern.

## Control API

Keep the robot API boring:

```text
GET /cmd?move=forward
GET /cmd?move=reverse
GET /cmd?move=left
GET /cmd?move=right
GET /cmd?move=stop
GET /speed?value=180
GET /servo?pan=90
GET /servo?tilt=90
GET /status
```

Later, if you need lower latency, add WebSocket or UDP commands. Start with HTTP because it is easy to debug.

## Camera

The camera is the built-in XIAO ESP32S3 Sense camera. Firmware serializes all
`/capture` and `/stream` acquisition through one mutex and enforces a five-second
minimum capture interval (0.2 FPS). The PC brain's in-memory `FrameBroker`
fetches that frame once and shares it with both the browser and perception pipeline.
Routine frames are never written to disk.

## Structured Vision

`VisionService` rotates camera frames into their displayed orientation, rejects
blurred or unchanged 160x120 previews, and runs background awareness no more
than once every five seconds while conversation is idle. Valid VLM JSON becomes
a short-lived `SceneSnapshot`; snapshots from the last minute form `WorldState`.
The service sends selected frames to the existing Gemma 4 E4B llama.cpp server,
requests JSON-schema output, and disables itself cleanly if `/v1/models` does
not advertise multimodal image support. The shared server runs with reasoning
disabled so short voice and scene responses do not spend their latency budget
on hidden reasoning tokens.

Text and Voice can request a fresh read-only inspection through the PC brain.
The latest unexpired validated snapshot is also injected into every Text and
Voice model turn as live visual context. Awareness refreshes that context when
the scene changes and carries it forward across pixel-equivalent frames without
rerunning the VLM.
The same turn cannot use a vision result for movement or head control, and the
emergency stop remains available. `GET /perception/latest` is the browser's
scene-state feed and `POST /perception/query` forces an explicit inspection.

Gate 5 is functionally available but not accepted as complete. Live testing has
shown roughly 2.6-second fresh-inference latency, generic scene descriptions,
occasional truncated structured output, and stale visual claims from realtime
dialogue history. The gateway now removes superseded visual dialogue when it
restores a session, makes current snapshots override earlier descriptions, and
cancels speculative speech before answering an explicit visual question from a
fresh validated result. These mitigations preserve correctness but do not solve
the remaining latency and description-quality limits. The 0.2 FPS camera cap
also intentionally permits up to five seconds of ambient-view delay. Corpus,
schema-validity, voice-concurrency, physical traffic, and retention acceptance
tests remain required before the gate is closed.

## LLM Tool Calling

The LLM should call PC-side tools like:

- `drive(move, duration_ms)`
- `set_head(pan, tilt)`
- `look_for(object_name)`
- `speak(text)`
- `stop()`

Those tools should call the robot API or local camera/speech modules. The LLM should not directly hit firmware endpoints; keep a safety layer in the PC brain.

## Safety Defaults

The firmware should:

- stop motors on boot
- stop motors on unknown command
- stop motors after a heartbeat timeout
- clamp servo angles
- clamp motor speed

The PC brain should:

- rate-limit movement commands
- prefer short movement durations
- expose a hard stop command
- log autonomous actions

## Build Roadmap

1. **Firmware cleanup**
   - Move credentials to `config.h`
   - Add `/status`
   - Add command timeout
   - Keep fallback AP mode

2. **Manual PC control**
   - Run `pc_brain`
   - Proxy manual commands to the robot
   - Confirm drive/head controls work

3. **Vision**
   - Validate shared E4B p95 latency, schema compliance, and grounding
   - Verify that two llama.cpp slots preserve foreground voice responsiveness
   - Fall back from 140 to 70 visual tokens before considering another model

4. **Voice**
   - Add push-to-talk or wake-word later
   - Start with a simple text chat interface

5. **LLM tools**
   - Add a tool layer in the PC brain
   - Start with safe movement and head-control tools
   - Add vision query tools after camera is reliable

6. **Autonomy**
   - Add scripted behaviors first
   - Let the LLM choose among those scripts
   - Add guardrails before free-form movement
