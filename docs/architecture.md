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

The camera being used is the built in esp32 xiao seeed sense camera.

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
   - Add a USB/IP camera stream to the PC brain
   - Display the stream in the web UI
   - Add snapshot capture endpoint

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
