#include "camera.h"
#include "control_channel.h"
#include "eyes.h"
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
  initializeEyes();
  initializeCamera();
  initializeWifi();
  initializeControlChannel();
  initializeHttpServer();
}

void loop() {
  updateWifi();
  updateControlChannel();
  updateHttpServer();
  updateMotors();
  updateServos();
  updateEyes();
  updateCamera();
  enforceSafetyTimeouts();
}
