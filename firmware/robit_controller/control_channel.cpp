#include "control_channel.h"

#include <WiFi.h>

#if __has_include("config.h")
#include "config.h"
#else
#include "config.example.h"
#endif

#include "eyes.h"
#include "motors.h"
#include "robot_state.h"
#include "servos.h"

namespace {
constexpr size_t MAX_CONTROL_MESSAGE_BYTES = 512;

#ifndef ROBIT_CONTROL_TCP_PORT
#define ROBIT_CONTROL_TCP_PORT 82
#endif
#ifndef ROBIT_CONTROL_TELEMETRY_INTERVAL_MS
#define ROBIT_CONTROL_TELEMETRY_INTERVAL_MS 500
#endif
#ifndef ROBIT_CONTROL_HEARTBEAT_TIMEOUT_MS
#define ROBIT_CONTROL_HEARTBEAT_TIMEOUT_MS 3000
#endif

WiFiServer controlServer(ROBIT_CONTROL_TCP_PORT);
WiFiClient controlClient;
String inputBuffer;
String activeSession;
unsigned long lastAcceptedSequence = 0;
unsigned long lastControlMessageAt = 0;
unsigned long lastTelemetryAt = 0;
bool handshakeComplete = false;
bool telemetryDirty = true;
String lastObservedMovement = "";
String lastObservedEyes = "";
int lastObservedPanTarget = -1;
int lastObservedTiltTarget = -1;

String jsonEscape(const String& value) {
  String escaped;
  escaped.reserve(value.length());
  for (size_t index = 0; index < value.length(); index++) {
    const char valueCharacter = value[index];
    if (valueCharacter == '"' || valueCharacter == '\\') escaped += '\\';
    escaped += valueCharacter;
  }
  return escaped;
}

String jsonStringValue(const String& source, const String& key, const String& fallback = "") {
  const String needle = "\"" + key + "\"";
  const int keyIndex = source.indexOf(needle);
  if (keyIndex < 0) return fallback;
  const int colonIndex = source.indexOf(':', keyIndex + needle.length());
  if (colonIndex < 0) return fallback;
  const int startQuote = source.indexOf('"', colonIndex + 1);
  if (startQuote < 0) return fallback;
  const int endQuote = source.indexOf('"', startQuote + 1);
  if (endQuote < 0) return fallback;
  return source.substring(startQuote + 1, endQuote);
}

long jsonLongValue(const String& source, const String& key, long fallback) {
  const String needle = "\"" + key + "\"";
  const int keyIndex = source.indexOf(needle);
  if (keyIndex < 0) return fallback;
  const int colonIndex = source.indexOf(':', keyIndex + needle.length());
  if (colonIndex < 0) return fallback;
  int start = colonIndex + 1;
  while (start < source.length() && isspace(source[start])) start++;
  int end = start;
  while (end < source.length() && (isdigit(source[end]) || source[end] == '-')) end++;
  if (end == start) return fallback;
  return source.substring(start, end).toInt();
}

unsigned long jsonUnsignedLongValue(
  const String& source,
  const String& key,
  unsigned long fallback
) {
  const String needle = "\"" + key + "\"";
  const int keyIndex = source.indexOf(needle);
  if (keyIndex < 0) return fallback;
  const int colonIndex = source.indexOf(':', keyIndex + needle.length());
  if (colonIndex < 0) return fallback;
  int start = colonIndex + 1;
  while (start < source.length() && isspace(source[start])) start++;
  int end = start;
  while (end < source.length() && isdigit(source[end])) end++;
  if (end == start) return fallback;
  return strtoul(source.substring(start, end).c_str(), nullptr, 10);
}

void sendLine(const String& payload) {
  if (!controlClient || !controlClient.connected()) return;
  controlClient.print(payload);
  controlClient.print('\n');
}

String compactStateJson() {
  String json = "{";
  json += "\"movement\":\"" + jsonEscape(robotState.movement) + "\",";
  json += "\"speed\":" + String(robotState.motorSpeed) + ",";
  json += "\"pan_actual\":" + String(getActualPanAngle()) + ",";
  json += "\"tilt_actual\":" + String(getActualTiltAngle()) + ",";
  json += "\"pan_target\":" + String(getTargetPanAngle()) + ",";
  json += "\"tilt_target\":" + String(getTargetTiltAngle()) + ",";
  json += "\"eyes\":\"" + jsonEscape(robotState.eyeExpression) + "\",";
  json += "\"wifi_rssi\":" + String(WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0) + ",";
  json += "\"camera\":" + String(robotState.cameraEnabled ? "true" : "false") + ",";
  json += "\"fault\":" + String(isBrainHeartbeatFaultActive() ? "true" : "false");
  json += "}";
  return json;
}

void sendAck(unsigned long sequence, const String& status, const String& detail = "") {
  String json = "{\"v\":1,\"type\":\"ack\",\"session\":\"" + jsonEscape(activeSession) + "\"";
  json += ",\"seq\":" + String(sequence);
  json += ",\"status\":\"" + jsonEscape(status) + "\"";
  if (detail.length() > 0) json += ",\"detail\":\"" + jsonEscape(detail) + "\"";
  json += ",\"state\":" + compactStateJson() + "}";
  sendLine(json);
}

void sendTelemetry() {
  String json = "{\"v\":1,\"type\":\"telemetry\",\"session\":\"" + jsonEscape(activeSession) + "\"";
  json += ",\"last_seq\":" + String(lastAcceptedSequence);
  json += ",\"state\":" + compactStateJson() + "}";
  sendLine(json);
  lastTelemetryAt = millis();
  telemetryDirty = false;
  lastObservedMovement = robotState.movement;
  lastObservedEyes = robotState.eyeExpression;
  lastObservedPanTarget = getTargetPanAngle();
  lastObservedTiltTarget = getTargetTiltAngle();
}

void resetConnection(bool stopBody) {
  if (stopBody) stopMotors();
  if (controlClient) controlClient.stop();
  inputBuffer = "";
  activeSession = "";
  lastAcceptedSequence = 0;
  lastControlMessageAt = 0;
  handshakeComplete = false;
  telemetryDirty = true;
}

void acceptNewClient() {
  if (controlClient && controlClient.connected()) return;
  WiFiClient candidate = controlServer.available();
  if (!candidate) return;
  resetConnection(true);
  controlClient = candidate;
  controlClient.setNoDelay(true);
  lastControlMessageAt = millis();
}

bool validateCommandEnvelope(const String& line, unsigned long& sequence) {
  if (!handshakeComplete) return false;
  if (jsonLongValue(line, "v", 0) != 1) return false;
  if (jsonStringValue(line, "session") != activeSession) return false;
  const unsigned long rawSequence = jsonUnsignedLongValue(line, "seq", 0);
  const long ttlMs = jsonLongValue(line, "ttl_ms", -1);
  const unsigned long expiresAtMs = jsonUnsignedLongValue(
    line,
    "expires_at_ms",
    0
  );
  if (rawSequence == 0 || ttlMs < 1 || ttlMs > 5000) return false;
  sequence = rawSequence;
  if (
    expiresAtMs == 0 ||
    static_cast<long>(millis() - expiresAtMs) >= 0
  ) {
    Serial.printf("[CONTROL][WARN] expired command seq=%lu\n", sequence);
    sendAck(sequence, "expired");
    return false;
  }
  if (sequence <= lastAcceptedSequence) {
    Serial.printf(
      "[CONTROL][WARN] stale command seq=%lu last_seq=%lu\n",
      sequence,
      lastAcceptedSequence
    );
    sendAck(sequence, "stale");
    return false;
  }
  return true;
}

void handleHello(const String& line) {
  const String session = jsonStringValue(line, "session");
  if (jsonLongValue(line, "v", 0) != 1 || session.length() == 0) {
    sendLine("{\"v\":1,\"type\":\"hello_ack\",\"status\":\"invalid\"}");
    return;
  }
  activeSession = session;
  lastAcceptedSequence = 0;
  lastControlMessageAt = millis();
  handshakeComplete = true;
  recordBrainHeartbeat();
  String json = "{\"v\":1,\"type\":\"hello_ack\",\"status\":\"ready\",\"session\":\"";
  json += jsonEscape(activeSession);
  json += "\",\"protocol\":1,\"firmware\":\"robit-control-v1\"";
  json += ",\"uptime_ms\":" + String(millis());
  json += ",\"capabilities\":[\"head_target\",\"drive\",\"stop\",\"eyes\",\"telemetry\"]";
  json += ",\"state\":" + compactStateJson() + "}";
  sendLine(json);
  telemetryDirty = true;
}

void handleCommand(const String& line) {
  const String type = jsonStringValue(line, "type");
  if (type == "hello") {
    handleHello(line);
    return;
  }

  unsigned long sequence = 0;
  if (!validateCommandEnvelope(line, sequence)) {
    if (handshakeComplete && sequence == 0) sendAck(0, "invalid", "invalid envelope");
    return;
  }

  bool accepted = true;
  if (type == "head_target") {
    const int pan = constrain(jsonLongValue(line, "pan", getTargetPanAngle()), PAN_MIN, PAN_MAX);
    const int tilt = constrain(jsonLongValue(line, "tilt", getTargetTiltAngle()), TILT_MIN, TILT_MAX);
    setHeadPosition(pan, tilt);
  } else if (type == "drive") {
    const String direction = jsonStringValue(line, "direction");
    const int speed = constrain(jsonLongValue(line, "speed", robotState.motorSpeed), 0, 255);
    const int durationMs = constrain(jsonLongValue(line, "duration_ms", 0), 0, MAX_MOVEMENT_DURATION_MS);
    if (
      direction != "forward" && direction != "reverse" &&
      direction != "left" && direction != "right" && direction != "stop"
    ) {
      accepted = false;
    } else {
      commandMovement(direction, speed, static_cast<unsigned long>(durationMs));
    }
  } else if (type == "stop") {
    stopMotors();
  } else if (type == "eyes") {
    const String expression = jsonStringValue(line, "expression");
    const int durationMs = constrain(jsonLongValue(line, "duration_ms", 0), 0, 10000);
    accepted = setEyeExpression(expression, static_cast<unsigned long>(durationMs));
  } else if (type != "ping") {
    accepted = false;
  }

  if (!accepted) {
    sendAck(sequence, "invalid", "unsupported command or value");
    return;
  }

  lastAcceptedSequence = sequence;
  lastControlMessageAt = millis();
  recordBrainHeartbeat();
  telemetryDirty = true;
  sendAck(sequence, "accepted");
}
}

void initializeControlChannel() {
  controlServer.begin();
  Serial.printf(
    "[CONTROL] Persistent TCP control listening on port %u\n",
    ROBIT_CONTROL_TCP_PORT
  );
}

void updateControlChannel() {
  acceptNewClient();
  if (!controlClient || !controlClient.connected()) {
    if (handshakeComplete) resetConnection(true);
    return;
  }

  while (controlClient.available()) {
    const char next = static_cast<char>(controlClient.read());
    if (next == '\r') continue;
    if (next == '\n') {
      if (inputBuffer.length() > 0) handleCommand(inputBuffer);
      inputBuffer = "";
      continue;
    }
    if (inputBuffer.length() >= MAX_CONTROL_MESSAGE_BYTES) {
      inputBuffer = "";
      sendAck(0, "invalid", "message too large");
      continue;
    }
    inputBuffer += next;
  }

  const unsigned long now = millis();
  if (
    robotState.movement != lastObservedMovement ||
    robotState.eyeExpression != lastObservedEyes ||
    getTargetPanAngle() != lastObservedPanTarget ||
    getTargetTiltAngle() != lastObservedTiltTarget
  ) {
    telemetryDirty = true;
  }
  if (
    handshakeComplete &&
    now - lastControlMessageAt > ROBIT_CONTROL_HEARTBEAT_TIMEOUT_MS
  ) {
    resetConnection(true);
    return;
  }
  if (
    handshakeComplete &&
    (
      telemetryDirty ||
      now - lastTelemetryAt >= ROBIT_CONTROL_TELEMETRY_INTERVAL_MS
    )
  ) {
    sendTelemetry();
  }
}
