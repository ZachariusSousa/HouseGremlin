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

#ifndef PAN_SERVO_STEP_DEGREES
#define PAN_SERVO_STEP_DEGREES 2
#endif

#ifndef PAN_SERVO_STEP_DELAY_MS
#define PAN_SERVO_STEP_DELAY_MS 4
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

void movePanServo(int fromAngle, int toAngle) {
  const int stepSize = max(1, PAN_SERVO_STEP_DEGREES);
  if (fromAngle == toAngle) {
    writePanServo(toAngle);
    return;
  }

  const int direction = toAngle > fromAngle ? 1 : -1;
  for (int angle = fromAngle; angle != toAngle; angle += direction * stepSize) {
    if ((direction > 0 && angle > toAngle) || (direction < 0 && angle < toAngle)) break;
    writePanServo(angle);
    delay(PAN_SERVO_STEP_DELAY_MS);
  }
  writePanServo(toAngle);
}
}

bool initializeServos() {
  Wire.begin(SERVO_SDA_PIN, SERVO_SCL_PIN);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(250);
  centerHead();
  Serial.println("[SERVO] PCA9685 initialized at 0x40");
  return true;
}

void setPanAngle(int angle) {
  const int targetAngle = constrain(angle, PAN_MIN, PAN_MAX);
  const int previousAngle = constrain(robotState.panAngle, PAN_MIN, PAN_MAX);
  robotState.panAngle = targetAngle;
  movePanServo(previousAngle, targetAngle);
}

void setTiltAngle(int angle) {
  robotState.tiltAngle = constrain(angle, TILT_MIN, TILT_MAX);
  writeTiltServo(robotState.tiltAngle);
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
}
