#pragma once

#include <Arduino.h>

constexpr unsigned long WIFI_CONNECT_TIMEOUT_MS = 12000;

void initializeWifi();
void updateWifi();
void initializeHttpServer();
void updateHttpServer();
String getRobotIp();
bool isAccessPointMode();
