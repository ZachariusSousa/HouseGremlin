#include "robot_state.h"

#include "motors.h"

RobotState robotState;

void initializeState() {
  robotState.movement = "stop";
  robotState.motorSpeed = 180;
  robotState.panAngle = 90;
  robotState.tiltAngle = 90;
  robotState.eyeExpression = "neutral";
  robotState.cameraEnabled = false;
  robotState.wifiConnected = false;
  robotState.apFallback = false;
  robotState.lastMovementCommandAt = 0;
  robotState.movementStopAt = 0;
  robotState.lastLlmCommandAt = 0;
  robotState.emergencyStopUntil = 0;
}

void enforceSafetyTimeouts() {
  const unsigned long now = millis();
  if (robotState.movement != "stop" && now - robotState.lastMovementCommandAt > COMMAND_TIMEOUT_MS) {
    stopMotors();
  }
  if (robotState.movement != "stop" && robotState.movementStopAt > 0 && now >= robotState.movementStopAt) {
    stopMotors();
  }
}
