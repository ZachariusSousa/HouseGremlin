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
float currentPanPosition = 90.0f;
float currentTiltPosition = 90.0f;
int targetPanAngle = 90;
int targetTiltAngle = 90;
float panVelocityDegreesPerSecond = 0.0f;
float tiltVelocityDegreesPerSecond = 0.0f;
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

#ifndef HEAD_PAN_MAX_SPEED_DPS
#define HEAD_PAN_MAX_SPEED_DPS 90.0f
#endif

#ifndef HEAD_TILT_MAX_SPEED_DPS
#define HEAD_TILT_MAX_SPEED_DPS 70.0f
#endif

#ifndef HEAD_SERVO_ACCELERATION_DPS2
#define HEAD_SERVO_ACCELERATION_DPS2 240.0f
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

float easedStepToward(
  float current,
  int target,
  float& velocityDegreesPerSecond,
  float maximumSpeedDegreesPerSecond,
  float deltaSeconds
) {
  const float distance = static_cast<float>(target - current);
  if (fabsf(distance) < 0.01f) {
    velocityDegreesPerSecond = 0.0f;
    return target;
  }

  const float direction = distance > 0.0f ? 1.0f : -1.0f;
  const float brakingSpeed = sqrtf(
    2.0f * HEAD_SERVO_ACCELERATION_DPS2 * fabsf(distance)
  );
  const float desiredSpeed = direction * min(maximumSpeedDegreesPerSecond, brakingSpeed);
  const float maximumVelocityChange = HEAD_SERVO_ACCELERATION_DPS2 * deltaSeconds;
  if (velocityDegreesPerSecond < desiredSpeed) {
    velocityDegreesPerSecond = min(
      velocityDegreesPerSecond + maximumVelocityChange,
      desiredSpeed
    );
  } else {
    velocityDegreesPerSecond = max(
      velocityDegreesPerSecond - maximumVelocityChange,
      desiredSpeed
    );
  }

  const float step = velocityDegreesPerSecond * deltaSeconds;
  if (fabsf(step) >= fabsf(distance)) {
    velocityDegreesPerSecond = 0.0f;
    return target;
  }
  return current + step;
}
}

bool initializeServos() {
  Wire.begin(SERVO_SDA_PIN, SERVO_SCL_PIN);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(250);
  currentPanAngle = targetPanAngle = robotState.panAngle = 90;
  currentTiltAngle = targetTiltAngle = robotState.tiltAngle = 90;
  currentPanPosition = 90.0f;
  currentTiltPosition = 90.0f;
  panVelocityDegreesPerSecond = 0.0f;
  tiltVelocityDegreesPerSecond = 0.0f;
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
  const unsigned long elapsedMs = lastServoUpdateAt == 0
    ? HEAD_SERVO_UPDATE_INTERVAL_MS
    : now - lastServoUpdateAt;
  lastServoUpdateAt = now;
  const float deltaSeconds = min(elapsedMs, 100UL) / 1000.0f;

  currentPanPosition = easedStepToward(
    currentPanPosition,
    targetPanAngle,
    panVelocityDegreesPerSecond,
    HEAD_PAN_MAX_SPEED_DPS,
    deltaSeconds
  );
  currentTiltPosition = easedStepToward(
    currentTiltPosition,
    targetTiltAngle,
    tiltVelocityDegreesPerSecond,
    HEAD_TILT_MAX_SPEED_DPS,
    deltaSeconds
  );
  const int nextPanAngle = roundf(currentPanPosition);
  const int nextTiltAngle = roundf(currentTiltPosition);
  if (nextPanAngle != currentPanAngle) {
    currentPanAngle = nextPanAngle;
    writePanServo(currentPanAngle);
  }
  if (nextTiltAngle != currentTiltAngle) {
    currentTiltAngle = nextTiltAngle;
    writeTiltServo(currentTiltAngle);
  }
}

int getActualPanAngle() {
  return currentPanAngle;
}

int getActualTiltAngle() {
  return currentTiltAngle;
}

int getTargetPanAngle() {
  return targetPanAngle;
}

int getTargetTiltAngle() {
  return targetTiltAngle;
}
