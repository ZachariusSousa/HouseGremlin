#include "camera.h"
#include "motors.h"
#include "robot_state.h"
#include "servos.h"
#include "wifi_control.h"

void setup() {
  Serial.begin(115200);
  delay(50);
  Serial.println("[BOOT] Robit starting");

  initializeState();
  initializeMotors();
  initializeServos();
  initializeCamera();
  initializeWifi();
  initializeHttpServer();
}

void loop() {
  updateWifi();
  updateHttpServer();
  updateMotors();
  updateServos();
  updateCamera();
  enforceSafetyTimeouts();
}
