#pragma once

// Copy this file to config.h and edit it for your network.
// config.h is ignored by git so Wi-Fi credentials stay local.

#define ROBIT_STA_SSID "YourWiFiName"
#define ROBIT_STA_PASSWORD "YourWiFiPassword"

#define ROBIT_AP_SSID "Robit-Control"
#define ROBIT_AP_PASSWORD "12345678"

// Station-mode mDNS name. Try http://robit.local after the robot joins Wi-Fi.
#define ROBIT_HOSTNAME "robit"

// Servo calibration. Copy these to config.h and tune per physical servo.
// If pan struggles or does not land on exact angles, tune pan first and leave tilt alone.
#define PAN_SERVO_MIN_PULSE 80
#define PAN_SERVO_MAX_PULSE 620
#define PAN_SERVO_CENTER_TRIM_DEGREES 0
#define PAN_SERVO_INVERT 0
#define PAN_SERVO_STEP_DEGREES 2
#define PAN_SERVO_STEP_DELAY_MS 4

#define TILT_SERVO_MIN_PULSE 80
#define TILT_SERVO_MAX_PULSE 620
#define TILT_SERVO_CENTER_TRIM_DEGREES 0
#define TILT_SERVO_INVERT 0
