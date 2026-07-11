#include "wifi_control.h"

#include <WebServer.h>
#include <WiFi.h>

#if __has_include("config.h")
#include "config.h"
#else
#include "config.example.h"
#endif

#include "motors.h"
#include "robot_state.h"
#include "servos.h"

namespace {
WebServer server(80);

String jsonEscape(const String& value) {
  String escaped;
  escaped.reserve(value.length());
  for (size_t i = 0; i < value.length(); i++) {
    const char c = value[i];
    if (c == '"' || c == '\\') escaped += '\\';
    escaped += c;
  }
  return escaped;
}

String statusJson() {
  String json = "{";
  json += "\"ok\":true,";
  json += "\"mode\":\"" + String(robotState.apFallback ? "ap" : "sta") + "\",";
  json += "\"ip\":\"" + jsonEscape(getRobotIp()) + "\",";
  json += "\"movement\":\"" + jsonEscape(robotState.movement) + "\",";
  json += "\"move\":\"" + jsonEscape(robotState.movement) + "\",";
  json += "\"speed\":" + String(robotState.motorSpeed) + ",";
  json += "\"pan\":" + String(robotState.panAngle) + ",";
  json += "\"tilt\":" + String(robotState.tiltAngle) + ",";
  json += "\"eyes\":\"" + jsonEscape(robotState.eyeExpression) + "\",";
  json += "\"camera\":" + String(robotState.cameraEnabled ? "true" : "false");
  json += "}";
  return json;
}

int queryInt(const String& name, int fallback) {
  if (!server.hasArg(name)) return fallback;
  return server.arg(name).toInt();
}

String body() {
  return server.hasArg("plain") ? server.arg("plain") : "";
}

String jsonStringValue(const String& source, const String& key, const String& fallback = "") {
  const String needle = "\"" + key + "\"";
  int keyIndex = source.indexOf(needle);
  if (keyIndex < 0) return fallback;
  int colonIndex = source.indexOf(':', keyIndex + needle.length());
  if (colonIndex < 0) return fallback;
  int startQuote = source.indexOf('"', colonIndex + 1);
  if (startQuote < 0) return fallback;
  int endQuote = source.indexOf('"', startQuote + 1);
  if (endQuote < 0) return fallback;
  return source.substring(startQuote + 1, endQuote);
}

int jsonIntValue(const String& source, const String& key, int fallback) {
  const String needle = "\"" + key + "\"";
  int keyIndex = source.indexOf(needle);
  if (keyIndex < 0) return fallback;
  int colonIndex = source.indexOf(':', keyIndex + needle.length());
  if (colonIndex < 0) return fallback;
  int start = colonIndex + 1;
  while (start < source.length() && isspace(source[start])) start++;
  int end = start;
  while (end < source.length() && (isdigit(source[end]) || source[end] == '-')) end++;
  if (end == start) return fallback;
  return source.substring(start, end).toInt();
}

void sendJson(int status, const String& payload) {
  server.send(status, "application/json", payload);
}

void handleRoot() {
  server.send(200, "text/plain", "Robit controller online. Use /status, /cmd, /speed, /servo, or /api/status.");
}

void handleStatus() {
  sendJson(200, statusJson());
}

void handleCmd() {
  const String move = server.arg("move");
  commandMovement(move, -1, 0);
  server.send(200, "text/plain", "OK");
}

void handleSpeed() {
  if (server.hasArg("value")) {
    setMotorSpeed(server.arg("value").toInt());
  }
  server.send(200, "text/plain", "OK");
}

void handleServo() {
  if (server.hasArg("pan")) setPanAngle(server.arg("pan").toInt());
  if (server.hasArg("tilt")) setTiltAngle(server.arg("tilt").toInt());
  server.send(200, "text/plain", "OK");
}

void handleApiMove() {
  const String payload = body();
  const String direction = server.hasArg("direction")
    ? server.arg("direction")
    : jsonStringValue(payload, "direction", "stop");
  const int speed = queryInt("speed", jsonIntValue(payload, "speed", -1));
  const int durationMs = queryInt("duration_ms", jsonIntValue(payload, "duration_ms", 0));

  if (
    direction != "forward" &&
    direction != "reverse" &&
    direction != "left" &&
    direction != "right" &&
    direction != "stop"
  ) {
    sendJson(400, "{\"ok\":false,\"error\":\"unknown direction\"}");
    return;
  }

  commandMovement(direction, speed, durationMs);
  sendJson(200, statusJson());
}

void handleApiHead() {
  const String payload = body();
  const int pan = queryInt("pan", jsonIntValue(payload, "pan", robotState.panAngle));
  const int tilt = queryInt("tilt", jsonIntValue(payload, "tilt", robotState.tiltAngle));
  const int panDelta = queryInt("pan_delta", jsonIntValue(payload, "pan_delta", 0));
  const int tiltDelta = queryInt("tilt_delta", jsonIntValue(payload, "tilt_delta", 0));

  if (panDelta != 0 || tiltDelta != 0) {
    moveHeadRelative(panDelta, tiltDelta);
  } else {
    setHeadPosition(pan, tilt);
  }
  sendJson(200, statusJson());
}

void handleEmergencyStop() {
  emergencyStopMotors();
  sendJson(200, statusJson());
}
}

void initializeWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ROBIT_STA_SSID, ROBIT_STA_PASSWORD);

  const unsigned long startedAt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startedAt < WIFI_CONNECT_TIMEOUT_MS) {
    delay(250);
  }

  if (WiFi.status() == WL_CONNECTED) {
    robotState.wifiConnected = true;
    robotState.apFallback = false;
    Serial.print("[WIFI] Connected: ");
    Serial.println(WiFi.localIP());
    return;
  }

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ROBIT_AP_SSID, ROBIT_AP_PASSWORD);
  robotState.wifiConnected = true;
  robotState.apFallback = true;
  Serial.print("[WIFI] Fallback AP started: ");
  Serial.println(WiFi.softAPIP());
}

void updateWifi() {
  robotState.wifiConnected = robotState.apFallback || WiFi.status() == WL_CONNECTED;
}

void initializeHttpServer() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/cmd", HTTP_GET, handleCmd);
  server.on("/speed", HTTP_GET, handleSpeed);
  server.on("/servo", HTTP_GET, handleServo);
  server.on("/api/status", HTTP_GET, handleStatus);
  server.on("/api/move", HTTP_ANY, handleApiMove);
  server.on("/api/head", HTTP_ANY, handleApiHead);
  server.on("/api/emergency-stop", HTTP_ANY, handleEmergencyStop);
  server.begin();
  Serial.println("[HTTP] Control server started on port 80");
}

void updateHttpServer() {
  server.handleClient();
}

String getRobotIp() {
  return robotState.apFallback ? WiFi.softAPIP().toString() : WiFi.localIP().toString();
}

bool isAccessPointMode() {
  return robotState.apFallback;
}
