#include <cstdint>
#include <iostream>

#include "telemetry_serialization.h"

int main() {
  robit::telemetry::StateSnapshot state;
  state.movement = "reverse";
  state.speed = 255;
  state.pan_actual = 135;
  state.tilt_actual = 115;
  state.pan_target = 135;
  state.tilt_target = 115;
  state.eyes = "concerned";
  state.wifi_rssi = -128;
  state.camera_enabled = true;
  state.heartbeat_fault = true;
  state.uptime_ms = UINT32_MAX;
  state.wifi_mode = "ap";
  state.heap_free_bytes = UINT32_MAX;
  state.heap_min_free_bytes = UINT32_MAX;
  state.heap_total_bytes = UINT32_MAX;
  state.psram_free_bytes = 0;
  state.psram_total_bytes = 0;
  state.heartbeat_armed = true;
  state.control_last_receive_age_ms = UINT32_MAX;

  robit::telemetry::HttpStatusSnapshot http;
  http.state = state;
  http.ip = "192.168.1.9";
  http.hostname = "robit.local";

  robit::telemetry::HttpStatusSnapshot escaped_http = http;
  escaped_http.state.movement = "move\"\\path";
  escaped_http.state.eyes = "eyes\"\\path";
  escaped_http.ip = "10.0.0.1\"\\path";
  escaped_http.hostname = "robit\"\\node.local";

  const std::string tcp = robit::telemetry::telemetryPacketJson(
    std::string(48, 's'), UINT32_MAX, state
  );
  std::cout << "{\"tcp_bytes\":" << tcp.size();
  std::cout << ",\"tcp\":" << tcp;
  std::cout << ",\"legacy\":"
            << robit::telemetry::legacyStateJson(state);
  std::cout << ",\"http\":" << robit::telemetry::httpStatusJson(http);
  std::cout << ",\"escaped_http\":"
            << robit::telemetry::httpStatusJson(escaped_http);
  std::cout << "}\n";
  return 0;
}
