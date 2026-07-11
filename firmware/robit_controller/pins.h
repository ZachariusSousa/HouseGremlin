#pragma once

#include <Arduino.h>

// Tracks. Both motor-driver PWM inputs are soldered to D4.
constexpr uint8_t LEFT_FWD_PIN = D0;
constexpr uint8_t LEFT_REV_PIN = D1;
constexpr uint8_t RIGHT_FWD_PIN = D2;
constexpr uint8_t RIGHT_REV_PIN = D3;
constexpr uint8_t MOTOR_PWM_PIN = D4;

// PCA9685 servo controller.
constexpr uint8_t SERVO_SDA_PIN = D6;
constexpr uint8_t SERVO_SCL_PIN = D7;
constexpr uint8_t SERVO_PCA9685_ADDRESS = 0x40;
constexpr uint8_t SERVO_PAN_CHANNEL = 0;
constexpr uint8_t SERVO_TILT_CHANNEL = 1;

// Reserved for the eye OLED bus in the next milestone.
constexpr uint8_t EYES_SDA_PIN = D5;
constexpr uint8_t EYES_SCL_PIN = D8;
constexpr uint8_t EYES_OLED_ADDRESS = 0x3C;
