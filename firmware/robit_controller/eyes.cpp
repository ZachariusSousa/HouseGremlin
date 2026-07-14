#include "eyes.h"

#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Wire.h>

#include "pins.h"

namespace {
constexpr int EYES_DISPLAY_WIDTH = 128;
constexpr int EYES_DISPLAY_HEIGHT = 64;
constexpr int EYES_BAR_HEIGHT = 12;
constexpr int EYES_BAR_MARGIN = 10;

TwoWire eyesWire(1);
Adafruit_SSD1306 eyesDisplay(EYES_DISPLAY_WIDTH, EYES_DISPLAY_HEIGHT, &eyesWire, -1);
bool eyesReady = false;

void showWiringTest() {
  // Briefly illuminate every pixel, then settle on a thick horizontal eye line.
  // Displays sharing this bus and address intentionally mirror the same image.
  eyesDisplay.clearDisplay();
  eyesDisplay.fillScreen(SSD1306_WHITE);
  eyesDisplay.display();
  delay(750);

  eyesDisplay.clearDisplay();
  const int barY = (EYES_DISPLAY_HEIGHT - EYES_BAR_HEIGHT) / 2;
  eyesDisplay.fillRoundRect(
    EYES_BAR_MARGIN,
    barY,
    EYES_DISPLAY_WIDTH - (EYES_BAR_MARGIN * 2),
    EYES_BAR_HEIGHT,
    EYES_BAR_HEIGHT / 2,
    SSD1306_WHITE
  );
  eyesDisplay.display();
}
}

bool initializeEyes() {
  eyesWire.begin(EYES_SDA_PIN, EYES_SCL_PIN);
  eyesReady = eyesDisplay.begin(SSD1306_SWITCHCAPVCC, EYES_OLED_ADDRESS, true, false);
  if (!eyesReady) {
    Serial.println("[EYES][ERROR] SSD1306 not found at 0x3C on D5/D8");
    return false;
  }

  showWiringTest();
  Serial.println("[EYES] OLED wiring test active on D5/D8 at 0x3C");
  return true;
}

void updateEyes() {
  if (!eyesReady) return;
}
