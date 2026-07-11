#pragma once

#include <Arduino.h>

constexpr int PAN_MIN = 55;
constexpr int PAN_MAX = 135;
constexpr int TILT_MIN = 35;
constexpr int TILT_MAX = 115;

bool initializeServos();
void setPanAngle(int angle);
void setTiltAngle(int angle);
void setHeadPosition(int pan, int tilt);
void moveHeadRelative(int panDelta, int tiltDelta);
void centerHead();
void updateServos();
