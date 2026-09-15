#include "telemetry_serialization.h"

namespace robit {
namespace telemetry {
namespace {

std::string jsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.length());
  for (char character : value) {
    if (character == '"' || character == '\\') escaped += '\\';
    escaped += character;
  }
  return escaped;
}

const char* jsonBool(bool value) {
  return value ? "true" : "false";
}

std::string jsonString(const std::string& value) {
  return "\"" + jsonEscape(value) + "\"";
}

}  // namespace

std::string legacyStateJson(const StateSnapshot& snapshot) {
  std::string json = "{";
  json += "\"movement\":" + jsonString(snapshot.movement) + ",";
  json += "\"speed\":" + std::to_string(snapshot.speed) + ",";
  json += "\"pan_actual\":" + std::to_string(snapshot.pan_actual) + ",";
  json += "\"tilt_actual\":" + std::to_string(snapshot.tilt_actual) + ",";
  json += "\"pan_target\":" + std::to_string(snapshot.pan_target) + ",";
  json += "\"tilt_target\":" + std::to_string(snapshot.tilt_target) + ",";
  json += "\"eyes\":" + jsonString(snapshot.eyes) + ",";
  json += "\"wifi_rssi\":" + std::to_string(snapshot.wifi_rssi) + ",";
  json += std::string("\"camera\":") + jsonBool(snapshot.camera_enabled) + ",";
  json += std::string("\"fault\":") + jsonBool(snapshot.heartbeat_fault);
  return json + "}";
}

std::string telemetryStateJson(const StateSnapshot& snapshot) {
  std::string json = legacyStateJson(snapshot);
  json.pop_back();
  json += ",\"u\":" + std::to_string(snapshot.uptime_ms);
  json += ",\"wm\":" + jsonString(snapshot.wifi_mode);
  json += ",\"hf\":" + std::to_string(snapshot.heap_free_bytes);
  json += ",\"hm\":" + std::to_string(snapshot.heap_min_free_bytes);
  json += ",\"ht\":" + std::to_string(snapshot.heap_total_bytes);
  json += ",\"pf\":" + std::to_string(snapshot.psram_free_bytes);
  json += ",\"pt\":" + std::to_string(snapshot.psram_total_bytes);
  json += ",\"ha\":" + std::string(jsonBool(snapshot.heartbeat_armed));
  json += ",\"ca\":" + std::to_string(snapshot.control_last_receive_age_ms);
  return json + "}";
}

std::string telemetryPacketJson(
  const std::string& session,
  std::uint32_t last_sequence,
  const StateSnapshot& snapshot
) {
  std::string json = "{\"v\":1,\"type\":\"telemetry\",\"session\":";
  json += jsonString(session);
  json += ",\"last_seq\":" + std::to_string(last_sequence);
  json += ",\"state\":" + telemetryStateJson(snapshot);
  return json + "}";
}

std::string httpStatusJson(const HttpStatusSnapshot& snapshot) {
  const StateSnapshot& state = snapshot.state;
  std::string json = "{";
  json += "\"ok\":" + std::string(jsonBool(snapshot.ok)) + ",";
  json += "\"mode\":" + jsonString(state.wifi_mode) + ",";
  json += "\"wifi_mode\":" + jsonString(state.wifi_mode) + ",";
  json += "\"ip\":" + jsonString(snapshot.ip) + ",";
  json += "\"hostname\":" + jsonString(snapshot.hostname) + ",";
  json += "\"wifi_rssi\":" + std::to_string(state.wifi_rssi) + ",";
  json += "\"uptime_ms\":" + std::to_string(state.uptime_ms) + ",";
  json += "\"heap_free_bytes\":" + std::to_string(state.heap_free_bytes) + ",";
  json += "\"heap_min_free_bytes\":" + std::to_string(state.heap_min_free_bytes) + ",";
  json += "\"heap_total_bytes\":" + std::to_string(state.heap_total_bytes) + ",";
  json += "\"psram_free_bytes\":" + std::to_string(state.psram_free_bytes) + ",";
  json += "\"psram_total_bytes\":" + std::to_string(state.psram_total_bytes) + ",";
  json += "\"movement\":" + jsonString(state.movement) + ",";
  json += "\"move\":" + jsonString(state.movement) + ",";
  json += "\"speed\":" + std::to_string(state.speed) + ",";
  json += "\"pan\":" + std::to_string(state.pan) + ",";
  json += "\"tilt\":" + std::to_string(state.tilt) + ",";
  json += "\"pan_actual\":" + std::to_string(state.pan_actual) + ",";
  json += "\"tilt_actual\":" + std::to_string(state.tilt_actual) + ",";
  json += "\"pan_target\":" + std::to_string(state.pan_target) + ",";
  json += "\"tilt_target\":" + std::to_string(state.tilt_target) + ",";
  json += "\"eyes\":" + jsonString(state.eyes) + ",";
  json += "\"brain_heartbeat_armed\":" + std::string(jsonBool(state.heartbeat_armed)) + ",";
  json += "\"brain_heartbeat_fault\":" + std::string(jsonBool(state.heartbeat_fault)) + ",";
  json += "\"heartbeat_armed\":" + std::string(jsonBool(state.heartbeat_armed)) + ",";
  json += "\"heartbeat_fault\":" + std::string(jsonBool(state.heartbeat_fault)) + ",";
  json += "\"control_last_receive_age_ms\":";
  json += std::to_string(state.control_last_receive_age_ms) + ",";
  json += "\"camera\":" + std::string(jsonBool(state.camera_enabled)) + ",";
  json += "\"camera_enabled\":" + std::string(jsonBool(state.camera_enabled));
  return json + "}";
}

}  // namespace telemetry
}  // namespace robit
