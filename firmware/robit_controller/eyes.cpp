#include "eyes.h"

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>

#include "pins.h"
#include "robot_state.h"

namespace {
constexpr int EYES_DISPLAY_WIDTH = 128;
constexpr int EYES_DISPLAY_HEIGHT = 64;
constexpr int EYES_BAR_MARGIN = 10;
constexpr int EYES_STROKE_WIDTH = 9;
constexpr unsigned long EYES_ANIMATION_FRAME_MS = 180;
constexpr unsigned long BRAIN_HEARTBEAT_TIMEOUT_MS = 12000;

// The OLEDs share the primary I2C bus with the PCA9685. The servo controller
// initializes Wire on D6/D7 before the eyes start, so do not reinitialize it.
Adafruit_SSD1306 leftEyeDisplay(EYES_DISPLAY_WIDTH, EYES_DISPLAY_HEIGHT, &Wire, -1);
Adafruit_SSD1306 rightEyeDisplay(EYES_DISPLAY_WIDTH, EYES_DISPLAY_HEIGHT, &Wire, -1);
bool leftEyeReady = false;
bool rightEyeReady = false;
bool eyesReady = false;
unsigned long expressionExpiresAt = 0;
unsigned long lastAnimationFrameAt = 0;
uint8_t animationFrame = 0;
bool heartbeatArmed = false;
bool heartbeatFaultActive = false;
unsigned long lastBrainHeartbeatAt = 0;

void thickLine(
  Adafruit_SSD1306& display,
  int x1,
  int y1,
  int x2,
  int y2,
  int thickness = EYES_STROKE_WIDTH
) {
  const int half = thickness / 2;
  for (int offset = -half; offset <= half; ++offset) {
    display.drawLine(x1, y1 + offset, x2, y2 + offset, SSD1306_WHITE);
  }
}

void drawBar(Adafruit_SSD1306& display, int y = 32, int height = 12, int xOffset = 0) {
  display.fillRoundRect(
    EYES_BAR_MARGIN + xOffset,
    y - (height / 2),
    EYES_DISPLAY_WIDTH - (EYES_BAR_MARGIN * 2),
    height,
    height / 2,
    SSD1306_WHITE
  );
}

void drawWave(Adafruit_SSD1306& display) {
  constexpr int xs[] = {12, 29, 46, 64, 82, 99, 116};
  constexpr int ys[] = {34, 26, 34, 26, 34, 26, 34};
  for (size_t i = 1; i < sizeof(xs) / sizeof(xs[0]); ++i) {
    thickLine(display, xs[i - 1], ys[i - 1], xs[i], ys[i], 7);
  }
}

void drawChevron(Adafruit_SSD1306& display, bool pointsRight) {
  const int outerX = pointsRight ? 34 : 94;
  const int pointX = pointsRight ? 88 : 40;
  thickLine(display, outerX, 16, pointX, 32);
  thickLine(display, pointX, 32, outerX, 48);
}

void drawExpression(Adafruit_SSD1306& display, bool isLeft, const String& expression) {
  display.clearDisplay();

  if (expression == "neutral") {
    drawBar(display);
  } else if (expression == "angry") {
    thickLine(display, 18, isLeft ? 18 : 46, 110, isLeft ? 46 : 18);
  } else if (expression == "cute") {
    drawChevron(display, isLeft);
  } else if (expression == "concerned") {
    thickLine(display, 18, isLeft ? 46 : 18, 110, isLeft ? 18 : 46);
  } else if (expression == "content") {
    drawWave(display);
  } else if (expression == "happy") {
    thickLine(display, 22, 45, 64, 20);
    thickLine(display, 64, 20, 106, 45);
  } else if (expression == "startled") {
    display.fillCircle(64, 32, 22, SSD1306_WHITE);
    display.fillCircle(64, 32, 14, SSD1306_BLACK);
  } else if (expression == "sleepy") {
    drawBar(display, 43, 8);
  } else if (expression == "curious") {
    const int radius = isLeft ? 20 : 13;
    display.fillCircle(64, 32, radius, SSD1306_WHITE);
    display.fillCircle(64, 32, radius - 7, SSD1306_BLACK);
  } else if (expression == "confused") {
    if (isLeft) drawWave(display);
    else drawBar(display);
  } else if (expression == "suspicious") {
    if (isLeft) display.fillRoundRect(50, 27, 68, 10, 5, SSD1306_WHITE);
    else display.fillRoundRect(10, 27, 68, 10, 5, SSD1306_WHITE);
  } else if (expression == "wink") {
    if (isLeft) drawBar(display);
    else drawChevron(display, false);
  } else if (expression == "fault") {
    thickLine(display, 25, 12, 103, 52, 10);
    thickLine(display, 25, 52, 103, 12, 10);
  } else if (expression == "listening") {
    const int height = 8 + ((animationFrame % 3) * 3);
    drawBar(display, 32, height);
  } else if (expression == "thinking") {
    const int offset = static_cast<int>(animationFrame % 5) * 4 - 8;
    drawBar(display, 32, 10, offset);
  } else if (expression == "speaking") {
    const int height = animationFrame % 2 == 0 ? 8 : 16;
    drawBar(display, 32, height);
  } else {
    drawBar(display);
  }

  display.display();
}

void renderCurrentExpression() {
  if (leftEyeReady) drawExpression(leftEyeDisplay, true, robotState.eyeExpression);
  if (rightEyeReady) drawExpression(rightEyeDisplay, false, robotState.eyeExpression);
}

String canonicalExpression(String expression) {
  expression.trim();
  expression.toLowerCase();
  if (expression == "embarrassed") return "cute";
  if (expression == "scared") return "concerned";
  if (expression == "relaxed") return "content";
  if (expression == "excited") return "happy";
  if (expression == "surprised") return "startled";
  if (expression == "error") return "fault";
  return expression;
}

bool isAnimatedExpression(const String& expression) {
  return expression == "listening" || expression == "thinking" || expression == "speaking";
}

void fillDisplay(Adafruit_SSD1306& display) {
  display.clearDisplay();
  display.fillScreen(SSD1306_WHITE);
  display.display();
}

void showWiringTest() {
  if (leftEyeReady) fillDisplay(leftEyeDisplay);
  if (rightEyeReady) fillDisplay(rightEyeDisplay);
  delay(750);
}
}

bool isEyeExpressionSupported(const String& requestedExpression) {
  const String expression = canonicalExpression(requestedExpression);
  return expression == "neutral" || expression == "angry" || expression == "cute" ||
    expression == "concerned" || expression == "content" || expression == "happy" ||
    expression == "startled" || expression == "sleepy" || expression == "curious" ||
    expression == "confused" || expression == "suspicious" || expression == "wink" ||
    expression == "fault" || expression == "listening" || expression == "thinking" ||
    expression == "speaking";
}

bool setEyeExpression(const String& requestedExpression, unsigned long durationMs) {
  const String expression = canonicalExpression(requestedExpression);
  if (!eyesReady || !isEyeExpressionSupported(expression)) return false;
  if (heartbeatFaultActive && expression != "fault") return false;

  robotState.eyeExpression = expression;
  expressionExpiresAt = durationMs > 0 ? millis() + durationMs : 0;
  animationFrame = 0;
  lastAnimationFrameAt = millis();
  renderCurrentExpression();
  return true;
}

bool recordBrainHeartbeat() {
  const bool recovered = heartbeatFaultActive;
  lastBrainHeartbeatAt = millis();
  heartbeatArmed = true;
  heartbeatFaultActive = false;
  if (recovered) {
    setEyeExpression("neutral", 0);
    Serial.println("[EYES] PC brain heartbeat recovered");
  }
  return recovered;
}

bool isBrainHeartbeatArmed() {
  return heartbeatArmed;
}

bool isBrainHeartbeatFaultActive() {
  return heartbeatFaultActive;
}

bool initializeEyes() {
  leftEyeReady = leftEyeDisplay.begin(
    SSD1306_SWITCHCAPVCC,
    EYES_LEFT_OLED_ADDRESS,
    true,
    false
  );
  rightEyeReady = rightEyeDisplay.begin(
    SSD1306_SWITCHCAPVCC,
    EYES_RIGHT_OLED_ADDRESS,
    true,
    false
  );
  eyesReady = leftEyeReady || rightEyeReady;

  if (!leftEyeReady) {
    Serial.println("[EYES][ERROR] Left SSD1306 not found at 0x3C on shared D6/D7 bus");
  }
  if (!rightEyeReady) {
    Serial.println("[EYES][ERROR] Right SSD1306 not found at 0x3D on shared D6/D7 bus");
  }
  if (!eyesReady) return false;

  showWiringTest();
  setEyeExpression("neutral", 0);
  Serial.println("[EYES] OLED expressions active on shared D6/D7 bus at 0x3C/0x3D");
  return leftEyeReady && rightEyeReady;
}

void updateEyes() {
  if (!eyesReady) return;

  const unsigned long now = millis();
  if (
    heartbeatArmed &&
    !heartbeatFaultActive &&
    now - lastBrainHeartbeatAt > BRAIN_HEARTBEAT_TIMEOUT_MS
  ) {
    heartbeatFaultActive = true;
    setEyeExpression("fault", 0);
    Serial.println("[EYES][FAULT] PC brain heartbeat timed out");
    return;
  }
  if (expressionExpiresAt > 0 && static_cast<long>(now - expressionExpiresAt) >= 0) {
    setEyeExpression("neutral", 0);
    return;
  }

  if (isAnimatedExpression(robotState.eyeExpression) && now - lastAnimationFrameAt >= EYES_ANIMATION_FRAME_MS) {
    lastAnimationFrameAt = now;
    animationFrame++;
    renderCurrentExpression();
  }
}
