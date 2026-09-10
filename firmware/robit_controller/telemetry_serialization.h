#pragma once

#include <cstdint>
#include <string>

namespace robit {
namespace telemetry {

struct StateSnapshot {
  std::string movement;
  int speed = 0;
  int pan = 0;
  int tilt = 0;
  int pan_actual = 0;
  int tilt_actual = 0;
  int pan_target = 0;
  int tilt_target = 0;
  std::string eyes;
  int wifi_rssi = 0;
  bool camera_enabled = false;
  bool heartbeat_armed = false;
  bool heartbeat_fault = false;
  std::uint32_t uptime_ms = 0;
  std::string wifi_mode;
  std::uint32_t heap_free_bytes = 0;
  std::uint32_t heap_min_free_bytes = 0;
  std::uint32_t heap_total_bytes = 0;
  std::uint32_t psram_free_bytes = 0;
  std::uint32_t psram_total_bytes = 0;
  std::uint32_t control_last_receive_age_ms = 0;
};

struct HttpStatusSnapshot {
  bool ok = true;
  StateSnapshot state;
  std::string ip;
  std::string hostname;
};

std::string legacyStateJson(const StateSnapshot& snapshot);
std::string telemetryStateJson(const StateSnapshot& snapshot);
std::string telemetryPacketJson(
  const std::string& session,
  std::uint32_t last_sequence,
  const StateSnapshot& snapshot
);
std::string httpStatusJson(const HttpStatusSnapshot& snapshot);

}  // namespace telemetry
}  // namespace robit
