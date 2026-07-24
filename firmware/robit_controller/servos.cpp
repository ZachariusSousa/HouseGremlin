#include "servos.h"

#include <Adafruit_PWMServoDriver.h>
#include <Wire.h>

#if __has_include("config.h")
#include "config.h"
#else
#include "config.example.h"
#endif

#include "pins.h"
#include "robot_state.h"

namespace {
Adafruit_PWMServoDriver pwm(SERVO_PCA9685_ADDRESS);
int currentPanAngle = 90;
int currentTiltAngle = 90;
int targetPanAngle = 90;
int targetTiltAngle = 90;
unsigned long lastServoUpdateAt = 0;

#ifndef PAN_SERVO_MIN_PULSE
#define PAN_SERVO_MIN_PULSE 80
#endif

#ifndef PAN_SERVO_MAX_PULSE
#define PAN_SERVO_MAX_PULSE 620
#endif

#ifndef PAN_SERVO_CENTER_TRIM_DEGREES
#define PAN_SERVO_CENTER_TRIM_DEGREES 0
#endif

#ifndef PAN_SERVO_INVERT
#define PAN_SERVO_INVERT 0
#endif

#ifndef HEAD_SERVO_MAX_STEP_DEGREES
#define HEAD_SERVO_MAX_STEP_DEGREES 3
#endif

#ifndef HEAD_SERVO_UPDATE_INTERVAL_MS
#define HEAD_SERVO_UPDATE_INTERVAL_MS 20
#endif

#ifndef TILT_SERVO_MIN_PULSE
#define TILT_SERVO_MIN_PULSE 80
#endif

#ifndef TILT_SERVO_MAX_PULSE
#define TILT_SERVO_MAX_PULSE 620
#endif

#ifndef TILT_SERVO_CENTER_TRIM_DEGREES
#define TILT_SERVO_CENTER_TRIM_DEGREES 0
#endif

#ifndef TILT_SERVO_INVERT
#define TILT_SERVO_INVERT 0
#endif

int calibratedAngle(int angle, int trimDegrees, bool invert) {
  int calibrated = constrain(angle + trimDegrees, 0, 180);
  return invert ? 180 - calibrated : calibrated;
}

int angleToPulse(int angle, int minPulse, int maxPulse, int trimDegrees, bool invert) {
  return map(calibratedAngle(angle, trimDegrees, invert), 0, 180, minPulse, maxPulse);
}

void writeServo(uint8_t channel, int angle, int minPulse, int maxPulse, int trimDegrees, bool invert) {
  pwm.setPWM(channel, 0, angleToPulse(angle, minPulse, maxPulse, trimDegrees, invert));
}

void writePanServo(int angle) {
  writeServo(
    SERVO_PAN_CHANNEL,
    angle,
    PAN_SERVO_MIN_PULSE,
    PAN_SERVO_MAX_PULSE,
    PAN_SERVO_CENTER_TRIM_DEGREES,
    PAN_SERVO_INVERT != 0
  );
}

void writeTiltServo(int angle) {
  writeServo(
    SERVO_TILT_CHANNEL,
    angle,
    TILT_SERVO_MIN_PULSE,
    TILT_SERVO_MAX_PULSE,
    TILT_SERVO_CENTER_TRIM_DEGREES,
    TILT_SERVO_INVERT != 0
  );
}

int stepToward(int current, int target) {
  const int distance = abs(target - current);
  const int maximumStep = max(1, HEAD_SERVO_MAX_STEP_DEGREES);
  const int stepSize = distance > 20 ? maximumStep : (distance > 8 ? min(2, maximumStep) : 1);
  if (current < target) return min(current + stepSize, target);
  if (current > target) return max(current - stepSize, target);
  return current;
}
}

bool initializeServos() {
  Wire.begin(SERVO_SDA_PIN, SERVO_SCL_PIN);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(250);
  currentPanAngle = targetPanAngle = robotState.panAngle = 90;
  currentTiltAngle = targetTiltAngle = robotState.tiltAngle = 90;
  writePanServo(currentPanAngle);
  writeTiltServo(currentTiltAngle);
  Serial.println("[SERVO] PCA9685 initialized at 0x40");
  return true;
}

void setPanAngle(int angle) {
  targetPanAngle = constrain(angle, PAN_MIN, PAN_MAX);
  robotState.panAngle = targetPanAngle;
}

void setTiltAngle(int angle) {
  targetTiltAngle = constrain(angle, TILT_MIN, TILT_MAX);
  robotState.tiltAngle = targetTiltAngle;
}

void setHeadPosition(int pan, int tilt) {
  setPanAngle(pan);
  setTiltAngle(tilt);
}

void moveHeadRelative(int panDelta, int tiltDelta) {
  setHeadPosition(robotState.panAngle + panDelta, robotState.tiltAngle + tiltDelta);
}

void centerHead() {
  setHeadPosition(90, 90);
}

void updateServos() {
  const unsigned long now = millis();
  if (now - lastServoUpdateAt < HEAD_SERVO_UPDATE_INTERVAL_MS) return;
  lastServoUpdateAt = now;

  const int nextPan = stepToward(currentPanAngle, targetPanAngle);
  const int nextTilt = stepToward(currentTiltAngle, targetTiltAngle);
  if (nextPan != currentPanAngle) {
    currentPanAngle = nextPan;
    writePanServo(currentPanAngle);
  }
  if (nextTilt != currentTiltAngle) {
    currentTiltAngle = nextTilt;
    writeTiltServo(currentTiltAngle);
  }
}
