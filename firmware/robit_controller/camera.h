#pragma once

#include <Arduino.h>
#include <WebServer.h>

bool initializeCamera();
void initializeCameraServer();
void updateCamera();
void handleCameraPage(WebServer& server);
void handleCameraCapture(WebServer& server);
