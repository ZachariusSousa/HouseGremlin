#pragma once

#include <Arduino.h>

constexpr unsigned long COMMAND_TIMEOUT_MS = 750;
constexpr unsigned long MAX_MOVEMENT_DURATION_MS = 1000;

bool initializeMotors();
void moveForward();
void moveReverse();
void turnLeft();
void turnRight();
void stopMotors();
void setMotorSpeed(int speed);
void commandMovement(const String& direction, int speed, unsigned long durationMs);
void updateMotors();
