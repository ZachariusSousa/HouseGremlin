#include "motors.h"

#include "pins.h"
#include "robot_state.h"

namespace {
void setDirection(bool leftForward, bool leftReverse, bool rightForward, bool rightReverse) {
  digitalWrite(LEFT_FWD_PIN, leftForward ? HIGH : LOW);
  digitalWrite(LEFT_REV_PIN, leftReverse ? HIGH : LOW);
  digitalWrite(RIGHT_FWD_PIN, rightForward ? HIGH : LOW);
  digitalWrite(RIGHT_REV_PIN, rightReverse ? HIGH : LOW);
}

bool movementBlocked() {
  return robotState.emergencyStopUntil > millis();
}
}

bool initializeMotors() {
  pinMode(LEFT_FWD_PIN, OUTPUT);
  pinMode(LEFT_REV_PIN, OUTPUT);
  pinMode(RIGHT_FWD_PIN, OUTPUT);
  pinMode(RIGHT_REV_PIN, OUTPUT);
  pinMode(MOTOR_PWM_PIN, OUTPUT);
  stopMotors();
  Serial.println("[MOTOR] Initialized");
  return true;
}

void setMotorSpeed(int speed) {
  robotState.motorSpeed = constrain(speed, 0, 255);
  if (robotState.movement != "stop") {
    analogWrite(MOTOR_PWM_PIN, robotState.motorSpeed);
  }
}

void stopMotors() {
  analogWrite(MOTOR_PWM_PIN, 0);
  setDirection(false, false, false, false);
  robotState.movement = "stop";
  robotState.movementStopAt = 0;
}

void emergencyStopMotors() {
  stopMotors();
  robotState.emergencyStopUntil = millis() + EMERGENCY_STOP_HOLD_MS;
}

void moveForward() {
  if (movementBlocked()) return;
  analogWrite(MOTOR_PWM_PIN, robotState.motorSpeed);
  setDirection(true, false, true, false);
  robotState.movement = "forward";
}

void moveReverse() {
  if (movementBlocked()) return;
  analogWrite(MOTOR_PWM_PIN, robotState.motorSpeed);
  setDirection(false, true, false, true);
  robotState.movement = "reverse";
}

void turnLeft() {
  if (movementBlocked()) return;
  analogWrite(MOTOR_PWM_PIN, robotState.motorSpeed);
  setDirection(false, true, true, false);
  robotState.movement = "left";
}

void turnRight() {
  if (movementBlocked()) return;
  analogWrite(MOTOR_PWM_PIN, robotState.motorSpeed);
  setDirection(true, false, false, true);
  robotState.movement = "right";
}

void commandMovement(const String& direction, int speed, unsigned long durationMs) {
  if (speed >= 0) {
    setMotorSpeed(speed);
  }

  robotState.lastMovementCommandAt = millis();
  robotState.movementStopAt = 0;
  if (durationMs > 0) {
    robotState.movementStopAt = millis() + min(durationMs, MAX_MOVEMENT_DURATION_MS);
  }

  if (direction == "forward") moveForward();
  else if (direction == "reverse") moveReverse();
  else if (direction == "left") turnLeft();
  else if (direction == "right") turnRight();
  else stopMotors();
}

void updateMotors() {
}
