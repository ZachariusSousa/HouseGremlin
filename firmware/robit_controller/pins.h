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

// Eye OLEDs share the PCA9685 I2C bus. Their addresses do not conflict with
// each other or with the servo controller at 0x40.
constexpr uint8_t EYES_SDA_PIN = SERVO_SDA_PIN;
constexpr uint8_t EYES_SCL_PIN = SERVO_SCL_PIN;
constexpr uint8_t EYES_LEFT_OLED_ADDRESS = 0x3C;
constexpr uint8_t EYES_RIGHT_OLED_ADDRESS = 0x3D;
