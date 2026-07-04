#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#if __has_include("config.h")
#include "config.h"
#else
#include "config.example.h"
#endif

WebServer server(80);
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

#define SDA_PIN D6
#define SCL_PIN D7

#define PAN_CH 0
#define TILT_CH 1

const int PAN_MIN = 55;
const int PAN_MAX = 135;
const int TILT_MIN = 35;
const int TILT_MAX = 115;

#define SERVOMIN 80
#define SERVOMAX 620

const int LEFT_FWD = D0;
const int LEFT_REV = D1;
const int RIGHT_FWD = D2;
const int RIGHT_REV = D3;
const int LEFT_PWM = D4;
const int RIGHT_PWM = D5;

const unsigned long COMMAND_TIMEOUT_MS = 750;
const unsigned long WIFI_CONNECT_TIMEOUT_MS = 12000;

int panAngle = 90;
int tiltAngle = 90;
int motorSpeed = 180;
String lastMove = "stop";
unsigned long lastMoveAt = 0;
bool apFallback = false;

int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setServo(int channel, int angle) {
  pwm.setPWM(channel, 0, angleToPulse(constrain(angle, 0, 180)));
}

void setPan(int angle) {
  panAngle = constrain(angle, PAN_MIN, PAN_MAX);
  setServo(PAN_CH, panAngle);
}

void setTilt(int angle) {
  tiltAngle = constrain(angle, TILT_MIN, TILT_MAX);
  setServo(TILT_CH, tiltAngle);
}

void setSpeed(int spd) {
  motorSpeed = constrain(spd, 0, 255);
  analogWrite(LEFT_PWM, motorSpeed);
  analogWrite(RIGHT_PWM, motorSpeed);
}

void stopRobot() {
  analogWrite(LEFT_PWM, 0);
  analogWrite(RIGHT_PWM, 0);
  digitalWrite(LEFT_FWD, LOW);
  digitalWrite(LEFT_REV, LOW);
  digitalWrite(RIGHT_FWD, LOW);
  digitalWrite(RIGHT_REV, LOW);
  lastMove = "stop";
}

void forward() {
  setSpeed(motorSpeed);
  digitalWrite(LEFT_FWD, HIGH);
  digitalWrite(LEFT_REV, LOW);
  digitalWrite(RIGHT_FWD, HIGH);
  digitalWrite(RIGHT_REV, LOW);
  lastMove = "forward";
}

void reverse() {
  setSpeed(motorSpeed);
  digitalWrite(LEFT_FWD, LOW);
  digitalWrite(LEFT_REV, HIGH);
  digitalWrite(RIGHT_FWD, LOW);
  digitalWrite(RIGHT_REV, HIGH);
  lastMove = "reverse";
}

void left() {
  setSpeed(motorSpeed);
  digitalWrite(LEFT_FWD, LOW);
  digitalWrite(LEFT_REV, HIGH);
  digitalWrite(RIGHT_FWD, HIGH);
  digitalWrite(RIGHT_REV, LOW);
  lastMove = "left";
}

void right() {
  setSpeed(motorSpeed);
  digitalWrite(LEFT_FWD, HIGH);
  digitalWrite(LEFT_REV, LOW);
  digitalWrite(RIGHT_FWD, LOW);
  digitalWrite(RIGHT_REV, HIGH);
  lastMove = "right";
}

void handleRoot() {
  server.send(200, "text/plain", "Robit controller online. Use /status, /cmd, /speed, or /servo.");
}

void handleStatus() {
  String json = "{";
  json += "\"ok\":true,";
  json += "\"mode\":\"" + String(apFallback ? "ap" : "sta") + "\",";
  json += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
  json += "\"move\":\"" + lastMove + "\",";
  json += "\"speed\":" + String(motorSpeed) + ",";
  json += "\"pan\":" + String(panAngle) + ",";
  json += "\"tilt\":" + String(tiltAngle);
  json += "}";
  server.send(200, "application/json", json);
}

void handleCmd() {
  String move = server.arg("move");
  lastMoveAt = millis();

  if (move == "forward") forward();
  else if (move == "reverse") reverse();
  else if (move == "left") left();
  else if (move == "right") right();
  else stopRobot();

  server.send(200, "text/plain", "OK");
}

void handleSpeed() {
  if (server.hasArg("value")) {
    setSpeed(server.arg("value").toInt());
  }
  server.send(200, "text/plain", "OK");
}

void handleServo() {
  if (server.hasArg("pan")) setPan(server.arg("pan").toInt());
  if (server.hasArg("tilt")) setTilt(server.arg("tilt").toInt());
  server.send(200, "text/plain", "OK");
}

void startWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ROBIT_STA_SSID, ROBIT_STA_PASSWORD);

  unsigned long startedAt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startedAt < WIFI_CONNECT_TIMEOUT_MS) {
    delay(250);
  }

  if (WiFi.status() == WL_CONNECTED) {
    apFallback = false;
    Serial.print("Robit joined WiFi: ");
    Serial.println(WiFi.localIP());
    return;
  }

  WiFi.mode(WIFI_AP);
  WiFi.softAP(ROBIT_AP_SSID, ROBIT_AP_PASSWORD);
  apFallback = true;
  Serial.print("Robit fallback AP started: ");
  Serial.println(WiFi.softAPIP());
}

void setup() {
  Serial.begin(115200);

  pinMode(LEFT_FWD, OUTPUT);
  pinMode(LEFT_REV, OUTPUT);
  pinMode(RIGHT_FWD, OUTPUT);
  pinMode(RIGHT_REV, OUTPUT);
  pinMode(LEFT_PWM, OUTPUT);
  pinMode(RIGHT_PWM, OUTPUT);

  stopRobot();

  Wire.begin(SDA_PIN, SCL_PIN);
  pwm.begin();
  pwm.setPWMFreq(50);
  delay(500);
  setPan(90);
  setTilt(90);

  startWifi();

  server.on("/", handleRoot);
  server.on("/status", handleStatus);
  server.on("/cmd", handleCmd);
  server.on("/speed", handleSpeed);
  server.on("/servo", handleServo);
  server.begin();

  Serial.println("Robit HTTP controller started");
}

void loop() {
  server.handleClient();

  if (lastMove != "stop" && millis() - lastMoveAt > COMMAND_TIMEOUT_MS) {
    stopRobot();
  }
}
