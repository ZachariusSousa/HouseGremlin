#pragma once

#include <Arduino.h>

bool initializeEyes();
void updateEyes();
bool isEyeExpressionSupported(const String& expression);
bool setEyeExpression(const String& expression, unsigned long durationMs = 0);
bool recordBrainHeartbeat();
bool isBrainHeartbeatArmed();
bool isBrainHeartbeatFaultActive();
