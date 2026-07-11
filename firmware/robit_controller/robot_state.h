#pragma once

#include <Arduino.h>

struct RobotState {
  String movement;
  int motorSpeed;
  int panAngle;
  int tiltAngle;
  String eyeExpression;
  bool cameraEnabled;
  bool wifiConnected;
  bool apFallback;
  unsigned long lastMovementCommandAt;
  unsigned long movementStopAt;
  unsigned long lastLlmCommandAt;
  unsigned long emergencyStopUntil;
};

extern RobotState robotState;

void initializeState();
void enforceSafetyTimeouts();
