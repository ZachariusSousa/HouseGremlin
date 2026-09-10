#include "wifi_control.h"

#include <WebServer.h>
#include <WiFi.h>
#include <ESPmDNS.h>

#if __has_include("config.h")
#include "config.h"
#else
#include "config.example.h"
#endif

#include "camera.h"
#include "control_channel.h"
#include "eyes.h"
#include "motors.h"
#include "robot_state.h"
#include "servos.h"
#include "telemetry_serialization.h"

namespace {
WebServer server(80);
bool mdnsStarted = false;
unsigned long lastWifiReconnectAttemptAt = 0;
constexpr unsigned long WIFI_RECONNECT_INTERVAL_MS = 5000;

#ifndef ROBIT_HOSTNAME
#define ROBIT_HOSTNAME "robit"
#endif
#ifndef ROBIT_CONTROL_TCP_PORT
#define ROBIT_CONTROL_TCP_PORT 82
#endif

robit::telemetry::HttpStatusSnapshot httpStatusSnapshot() {
  robit::telemetry::HttpStatusSnapshot status;
  robit::telemetry::StateSnapshot& state = status.state;
  state.movement = robotState.movement.c_str();
  state.speed = robotState.motorSpeed;
  state.pan = robotState.panAngle;
  state.tilt = robotState.tiltAngle;
  state.pan_actual = getActualPanAngle();
  state.tilt_actual = getActualTiltAngle();
  state.pan_target = getTargetPanAngle();
  state.tilt_target = getTargetTiltAngle();
  state.eyes = robotState.eyeExpression.c_str();
  state.wifi_rssi = WiFi.status() == WL_CONNECTED ? WiFi.RSSI() : 0;
  state.camera_enabled = robotState.cameraEnabled;
  state.heartbeat_armed = isBrainHeartbeatArmed();
  state.heartbeat_fault = isBrainHeartbeatFaultActive();
  state.uptime_ms = millis();
  state.wifi_mode = robotState.apFallback ? "ap" : "sta";
  state.heap_free_bytes = ESP.getFreeHeap();
  state.heap_min_free_bytes = ESP.getMinFreeHeap();
  state.heap_total_bytes = ESP.getHeapSize();
  state.psram_free_bytes = ESP.getFreePsram();
  state.psram_total_bytes = ESP.getPsramSize();
  state.control_last_receive_age_ms = controlLastReceiveAgeMs();
  status.ip = getRobotIp().c_str();
  status.hostname = (String(ROBIT_HOSTNAME) + ".local").c_str();
  return status;
}

String statusJson() {
  const std::string json = robit::telemetry::httpStatusJson(
    httpStatusSnapshot()
  );
  return String(json.c_str());
}

void sendJson(int status, const String& payload) {
  server.send(status, "application/json", payload);
}

void handleRoot() {
  server.send(200, "text/plain", "Robit controller online. Control uses TCP port 82; HTTP provides status and camera only.");
}

void handleStatus() {
  sendJson(200, statusJson());
}

void handleCameraStreamRedirect() {
  const String host = getRobotIp();
  server.sendHeader("Location", "http://" + host + ":81/stream");
  server.send(302, "text/plain", "Camera stream is on port 81");
}

void handleCameraCaptureRedirect() {
  const String host = getRobotIp();
  server.sendHeader("Location", "http://" + host + ":81/capture");
  server.send(302, "text/plain", "Camera capture is on port 81");
}

void startMdns() {
  if (robotState.apFallback) return;
  if (MDNS.begin(ROBIT_HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    MDNS.addService("robit-camera", "tcp", 81);
    MDNS.addService("robit-control", "tcp", ROBIT_CONTROL_TCP_PORT);
    mdnsStarted = true;
    Serial.print("[WIFI] mDNS started: http://");
    Serial.print(ROBIT_HOSTNAME);
    Serial.println(".local");
  } else {
    mdnsStarted = false;
    Serial.println("[WIFI][ERROR] mDNS start failed");
  }
}
}

void initializeWifi() {
  // Robit is a latency-sensitive, mains/battery-powered controller. Favor a
  // responsive radio and stable multi-tasked camera/control traffic over Wi-Fi
  // power saving.
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
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
    startMdns();
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
  if (robotState.apFallback) {
    robotState.wifiConnected = true;
    return;
  }

  robotState.wifiConnected = WiFi.status() == WL_CONNECTED;
  if (robotState.wifiConnected) return;

  const unsigned long now = millis();
  if (now - lastWifiReconnectAttemptAt < WIFI_RECONNECT_INTERVAL_MS) return;
  lastWifiReconnectAttemptAt = now;
  Serial.println("[WIFI][WARN] Station disconnected; requesting reconnect");
  WiFi.reconnect();
}

void initializeHttpServer() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/api/status", HTTP_GET, handleStatus);
  server.on("/camera", HTTP_GET, []() { handleCameraPage(server); });
  // Camera acquisition runs only in the dedicated port-81 task. Never call
  // esp_camera_fb_get() from the main control HTTP loop: a stalled sensor must
  // not make status or control-channel servicing unresponsive.
  server.on("/camera/capture", HTTP_GET, handleCameraCaptureRedirect);
  server.on("/camera/stream", HTTP_GET, handleCameraStreamRedirect);
  server.begin();
  Serial.println("[HTTP] Control server started on port 80");
  initializeCameraServer();
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
