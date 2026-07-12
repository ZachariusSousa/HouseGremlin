#include "camera.h"

#include <WiFi.h>
#include "esp_camera.h"
#if __has_include(<esp_arduino_version.h>)
#include <esp_arduino_version.h>
#endif

#ifndef ESP_ARDUINO_VERSION_MAJOR
#define ESP_ARDUINO_VERSION_MAJOR 3
#endif

#include "robot_state.h"

namespace {
WebServer cameraServer(81);
TaskHandle_t cameraServerTaskHandle = nullptr;

// Seeed Studio XIAO ESP32S3 Sense OV2640 camera pin map.
constexpr int PWDN_GPIO_NUM = -1;
constexpr int RESET_GPIO_NUM = -1;
constexpr int XCLK_GPIO_NUM = 10;
constexpr int SIOD_GPIO_NUM = 40;
constexpr int SIOC_GPIO_NUM = 39;
constexpr int Y9_GPIO_NUM = 48;
constexpr int Y8_GPIO_NUM = 11;
constexpr int Y7_GPIO_NUM = 12;
constexpr int Y6_GPIO_NUM = 14;
constexpr int Y5_GPIO_NUM = 16;
constexpr int Y4_GPIO_NUM = 18;
constexpr int Y3_GPIO_NUM = 17;
constexpr int Y2_GPIO_NUM = 15;
constexpr int VSYNC_GPIO_NUM = 38;
constexpr int HREF_GPIO_NUM = 47;
constexpr int PCLK_GPIO_NUM = 13;

constexpr char STREAM_BOUNDARY[] = "123456789000000000000987654321";

void handleStream() {
  WiFiClient client = cameraServer.client();
  client.print(
    String("HTTP/1.1 200 OK\r\n") +
    "Access-Control-Allow-Origin: *\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=" + STREAM_BOUNDARY + "\r\n\r\n"
  );

  while (client.connected()) {
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[CAMERA][ERROR] Frame capture failed");
      delay(50);
      continue;
    }

    client.print("--");
    client.print(STREAM_BOUNDARY);
    client.print("\r\nContent-Type: image/jpeg\r\nContent-Length: ");
    client.print(fb->len);
    client.print("\r\n\r\n");
    client.write(fb->buf, fb->len);
    client.print("\r\n");
    esp_camera_fb_return(fb);

    vTaskDelay(pdMS_TO_TICKS(30));
  }
}

void handleStreamStatus() {
  cameraServer.send(200, "application/json", "{\"ok\":true,\"stream\":\"/stream\"}");
}

void cameraServerTask(void*) {
  for (;;) {
    cameraServer.handleClient();
    vTaskDelay(pdMS_TO_TICKS(5));
  }
}
}

bool initializeCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
#else
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
#endif
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.frame_size = FRAMESIZE_QVGA;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.jpeg_quality = 14;
  config.fb_count = psramFound() ? 2 : 1;

  esp_err_t error = esp_camera_init(&config);
  if (error != ESP_OK) {
    robotState.cameraEnabled = false;
    Serial.printf("[CAMERA][ERROR] Initialization failed: 0x%x\n", error);
    return false;
  }

  sensor_t* sensor = esp_camera_sensor_get();
  if (sensor) {
    sensor->set_framesize(sensor, FRAMESIZE_QVGA);
  }

  robotState.cameraEnabled = true;
  Serial.println("[CAMERA] Initialized");
  return true;
}

void initializeCameraServer() {
  if (!robotState.cameraEnabled) {
    Serial.println("[CAMERA] Stream server skipped because camera is disabled");
    return;
  }
  if (cameraServerTaskHandle != nullptr) {
    return;
  }

  cameraServer.on("/", HTTP_GET, handleStreamStatus);
  cameraServer.on("/stream", HTTP_GET, handleStream);
  cameraServer.begin();
  Serial.println("[CAMERA] Stream server started on port 81");
  xTaskCreatePinnedToCore(
    cameraServerTask,
    "camera_http",
    8192,
    nullptr,
    1,
    &cameraServerTaskHandle,
    0
  );
}

void updateCamera() {
}

void handleCameraPage(WebServer& server) {
  const char html[] =
    "<!doctype html><html><head><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
    "<title>Robit Camera</title>"
    "<style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}"
    "main{display:grid;min-height:100vh;place-items:center;padding:16px}"
    "img{max-width:100%;height:auto;border:1px solid #444}</style></head>"
    "<body><main><img src=\"http://\" id=\"stream\" alt=\"Robit camera stream\"></main>"
    "<script>document.getElementById('stream').src='http://'+location.hostname+':81/stream';</script>"
    "</body></html>";
  server.send(200, "text/html", html);
}

void handleCameraCapture(WebServer& server) {
  if (!robotState.cameraEnabled) {
    server.send(503, "application/json", "{\"ok\":false,\"error\":\"camera disabled\"}");
    return;
  }

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(503, "application/json", "{\"ok\":false,\"error\":\"capture failed\"}");
    return;
  }

  WiFiClient client = server.client();
  client.print("HTTP/1.1 200 OK\r\n");
  client.print("Access-Control-Allow-Origin: *\r\n");
  client.print("Content-Type: image/jpeg\r\n");
  client.print("Content-Length: ");
  client.print(fb->len);
  client.print("\r\n\r\n");
  client.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);
}
