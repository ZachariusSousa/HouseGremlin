#pragma once

// Copy this file to config.h and edit it for your network.
// config.h is ignored by git so Wi-Fi credentials stay local.

#define ROBIT_STA_SSID "YourWiFiName"
#define ROBIT_STA_PASSWORD "YourWiFiPassword"

#define ROBIT_AP_SSID "Robit-Control"
#define ROBIT_AP_PASSWORD "12345678"

// Station-mode mDNS name. Try http://robit.local after the robot joins Wi-Fi.
#define ROBIT_HOSTNAME "robit"

// Persistent PC control channel (protocol v1).
#define ROBIT_CONTROL_TCP_PORT 82
#define ROBIT_CONTROL_HEARTBEAT_TIMEOUT_MS 3000
#define ROBIT_CONTROL_TELEMETRY_INTERVAL_MS 500

// Global camera acquisition ceiling shared by /capture and /stream. The PC
// frame broker is the only poller and never requests more than two FPS.
#define ROBIT_CAMERA_MAX_FPS 2

// Servo calibration. Copy these to config.h and tune per physical servo.
// If pan struggles or does not land on exact angles, tune pan first and leave tilt alone.
#define PAN_SERVO_MIN_PULSE 80
#define PAN_SERVO_MAX_PULSE 620
#define PAN_SERVO_CENTER_TRIM_DEGREES 0
#define PAN_SERVO_INVERT 0
// Head commands set a target. The firmware follows it with acceleration and
// velocity limits at 50 Hz so sparse targets still produce fluid motion.
#define HEAD_PAN_MAX_SPEED_DPS 90.0f
#define HEAD_TILT_MAX_SPEED_DPS 70.0f
#define HEAD_SERVO_ACCELERATION_DPS2 240.0f
#define HEAD_SERVO_UPDATE_INTERVAL_MS 20

#define TILT_SERVO_MIN_PULSE 80
#define TILT_SERVO_MAX_PULSE 620
#define TILT_SERVO_CENTER_TRIM_DEGREES 0
#define TILT_SERVO_INVERT 0
