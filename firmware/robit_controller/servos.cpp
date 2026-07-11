#include "servos.h"

#include <Adafruit_PWMServoDriver.h>
#include <Wire.h>

#include "pins.h"
#include "robot_state.h"

namespace {
constexpr int SERVOMIN = 80;
constexpr int SERVOMAX = 620;
Adafruit_PWMServoDriver pwm(SERVO_PCA9685_ADDRESS);

int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setServo(uint8_t channel, int angle) {
  pwm.setPWM(channel, 0, angleToPulse(constrain(angle, 0, 180)));
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
  robotState.panAngle = constrain(angle, PAN_MIN, PAN_MAX);
  setServo(SERVO_PAN_CHANNEL, robotState.panAngle);
}

void setTiltAngle(int angle) {
  robotState.tiltAngle = constrain(angle, TILT_MIN, TILT_MAX);
  setServo(SERVO_TILT_CHANNEL, robotState.tiltAngle);
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
