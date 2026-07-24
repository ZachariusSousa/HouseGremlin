#pragma once

// Copy this file to config.h and edit it for your network.
// config.h is ignored by git so Wi-Fi credentials stay local.

#define ROBIT_STA_SSID "YourWiFiName"
#define ROBIT_STA_PASSWORD "YourWiFiPassword"

#define ROBIT_AP_SSID "Robit-Control"
#define ROBIT_AP_PASSWORD "12345678"

// Station-mode mDNS name. Try http://robit.local after the robot joins Wi-Fi.
#define ROBIT_HOSTNAME "robit"

// Global camera acquisition ceiling shared by /capture and /stream.
// The PC brain controls its idle polling interval separately and may request
// fresher frames while tracking. Five FPS keeps tracking responsive while
// QVGA JPEG frames remain small.
#define ROBIT_CAMERA_MAX_FPS 5

// Servo calibration. Copy these to config.h and tune per physical servo.
// If pan struggles or does not land on exact angles, tune pan first and leave tilt alone.
#define PAN_SERVO_MIN_PULSE 80
#define PAN_SERVO_MAX_PULSE 620
#define PAN_SERVO_CENTER_TRIM_DEGREES 0
#define PAN_SERVO_INVERT 0
// Head commands set a target. The firmware eases both axes toward that target
// at 50 Hz so low-rate camera detections do not look like servo snaps.
#define HEAD_SERVO_MAX_STEP_DEGREES 3
#define HEAD_SERVO_UPDATE_INTERVAL_MS 20

#define TILT_SERVO_MIN_PULSE 80
#define TILT_SERVO_MAX_PULSE 620
#define TILT_SERVO_CENTER_TRIM_DEGREES 0
#define TILT_SERVO_INVERT 0
