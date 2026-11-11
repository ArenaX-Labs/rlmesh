// rlmesh::Robot: the device robot over zenoh-pico, on the board's clock and
// randomness.

#include <ESPmDNS.h>
#include <Preferences.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RLMesh.h"
#include "rlmesh_transport.h"

namespace rlmesh {

namespace {
// poll() sleeps until the robot is next due or something arrives, at most
// this long, so a sketch's loop() never spins on its core: with zenoh-pico
// running, a core that never blocks leaves the radio dropping most of what it
// receives (a classic ESP32 on Wi-Fi, measured: 30 to 70% of pings lost,
// joins failing for ever). The cap leaves a sketch's own work (buttons, the
// provisioning port) waiting no longer than this.
const uint32_t kMaxWaitUs = 5000;

// The transport's input wakes the task that polls: it is the one that began.
void wakeLoop(void* task) { xTaskNotifyGive(static_cast<TaskHandle_t>(task)); }

// The NVS namespace and key of the e-stop latch.
const char* const kLatchSpace = "rlmesh-estop";
const char* const kLatchKey = "latched";

bool storedLatch() {
  Preferences nvs;
  if (!nvs.begin(kLatchSpace, true)) return false;
  bool latched = nvs.getBool(kLatchKey, false);
  nvs.end();
  return latched;
}

void storeLatch(bool latched) {
  Preferences nvs;
  if (!nvs.begin(kLatchSpace, false)) return;
  nvs.putBool(kLatchKey, latched);
  nvs.end();
}
}  // namespace

bool Robot::begin(const char* address, const char* name, const char* robotJson,
                  uint32_t heartbeatMs) {
  rlmesh_device_robot_config_t config = {address, nullptr,   nullptr,     0,
                                         name,    robotJson, heartbeatMs, &esp32::fillRandom};
  return start(config);
}

bool Robot::begin(const Provisioning& provisioning, const char* robotJson, uint32_t heartbeatMs) {
  end();
  if (!provisioning.provisioned()) return false;
  rlmesh_device_robot_config_t config = {
      nullptr, provisioning.key_, provisioning.record_, provisioning.recordLen_,
      nullptr, robotJson,         heartbeatMs,          &esp32::fillRandom};
  return start(config);
}

bool Robot::start(const rlmesh_device_robot_config_t& config) {
  end();
  // A hostname: letters, digits and '-'.
  snprintf(mdnsName_, sizeof mdnsName_, "rlmesh-%s", config.name != nullptr ? config.name : "");
  if (config.device_key != nullptr) {
    uint8_t publicKey[32];
    rlmesh_device_key_public(config.device_key, publicKey);
    // The device key gives each copy of the firmware its own hostname.
    snprintf(mdnsName_, sizeof mdnsName_, "rlmesh-%02x%02x%02x%02x%02x%02x%02x%02x", publicKey[0],
             publicKey[1], publicKey[2], publicKey[3], publicKey[4], publicKey[5], publicKey[6],
             publicKey[7]);
  }
  for (char* c = mdnsName_; *c != '\0'; ++c) {
    if (!isalnum(static_cast<unsigned char>(*c))) *c = '-';
  }
  if (rlmesh_device_robot_new(&config, &robot_) != RLMESH_DEVICE_OK) {
    robot_ = nullptr;
    return false;
  }
  // A latch from before a reboot still holds, until a local clear.
  if (storedLatch()) {
    rlmesh_device_robot_estop(robot_, "latched before a reboot", esp32::nowUs());
    persisted_ = 1;
  }
  zenoh_ = static_cast<rlmesh_zenoh_t*>(malloc(rlmesh_zenoh_size()));
  if (zenoh_ == nullptr || rlmesh_zenoh_open(zenoh_, robot_, &esp32::nowUs) != 0) {
    free(zenoh_);
    zenoh_ = nullptr;
    end();
    return false;
  }
  rlmesh_zenoh_on_input(zenoh_, &wakeLoop, xTaskGetCurrentTaskHandle());
  return true;
}

void Robot::poll() {
  if (zenoh_ == nullptr) return;
  rlmesh_zenoh_poll(zenoh_);
  discover();
  persistLatch();
  // Whole ticks only: a shorter wait would round to none, or up past it.
  uint64_t due = rlmesh_device_robot_next_due(robot_, esp32::nowUs());
  TickType_t ticks =
      TickType_t((due < kMaxWaitUs ? due : kMaxWaitUs) / (1000 * portTICK_PERIOD_MS));
  if (ticks > 0) ulTaskNotifyTake(pdTRUE, ticks);
}

void Robot::persistLatch() {
  int8_t latched = estopLatched() ? 1 : 0;
  if (latched == persisted_) return;
  storeLatch(latched == 1);
  persisted_ = latched;
}

// Every stored home address failed: ask mDNS for `_rlmesh._tcp`, and keep only
// the instance whose TXT id is the stored host id. At most
// every 30 s: a query blocks for about a second.
void Robot::discover() {
  const char* id = rlmesh_device_robot_home_id(robot_);
  if (id == nullptr || id[0] == '\0' || !status().home_unreachable) return;
  if (lastDiscovery_ != 0 && millis() - lastDiscovery_ < 30000) return;
  lastDiscovery_ = millis();
  // Its own hostname: the home's id is the home's (`{id}.local`).
  if (!mdns_) mdns_ = MDNS.begin(mdnsName_);
  int found = MDNS.queryService("rlmesh", "tcp");
  for (int i = 0; i < found; ++i) {
    if (MDNS.txt(i, "id") != id) continue;
    char address[64];
    snprintf(address, sizeof address, "%s:%u", MDNS.address(i).toString().c_str(),
             unsigned(MDNS.port(i)));
    rlmesh_device_robot_add_home(robot_, address, esp32::nowUs());
  }
}

void Robot::end() {
  if (zenoh_ != nullptr) {
    rlmesh_zenoh_close(zenoh_);
    free(zenoh_);
    zenoh_ = nullptr;
  }
  rlmesh_device_robot_free(robot_);
  robot_ = nullptr;
}

int Robot::part(const char* path) const {
  return robot_ != nullptr ? rlmesh_device_robot_part(robot_, path) : -1;
}

TickResult Robot::tick(int part, rlmesh_device_tick_t& out) {
  if (robot_ == nullptr || part < 0 || part >= RLMESH_DEVICE_NO_PART)
    return TickResult::InvalidArgument;
  return static_cast<TickResult>(
      rlmesh_device_robot_tick(robot_, uint8_t(part), esp32::nowUs(), &out));
}

void Robot::handoverDone(int part) {
  if (robot_ != nullptr && part >= 0 && part < RLMESH_DEVICE_NO_PART) {
    rlmesh_device_robot_handover_done(robot_, uint8_t(part));
  }
}

int Robot::publish(int part, const void* value, size_t len) {
  if (robot_ == nullptr || part < 0 || part >= RLMESH_DEVICE_NO_PART) {
    return RLMESH_DEVICE_ERR_ARG;
  }
  uint64_t now = esp32::nowUs();
  return rlmesh_device_robot_publish(robot_, uint8_t(part), now, static_cast<const uint8_t*>(value),
                                     len, now);
}

void Robot::estop(const char* reason) {
  if (robot_ != nullptr) rlmesh_device_robot_estop(robot_, reason, esp32::nowUs());
}

void Robot::clearEstop() {
  if (robot_ != nullptr) rlmesh_device_robot_clear_estop(robot_);
  persistLatch();
}

rlmesh_device_robot_status_t Robot::status() const {
  rlmesh_device_robot_status_t s = {};
  if (robot_ != nullptr) rlmesh_device_robot_status(robot_, &s);
  return s;
}

uint32_t Robot::transportFailures() const {
  return zenoh_ != nullptr ? rlmesh_zenoh_failures(zenoh_) : 0;
}

const char* Robot::lastRefusal() const { return rlmesh_device_refusal_name(status().last_refusal); }

uint64_t Robot::refusals(uint8_t code) const {
  return robot_ != nullptr ? rlmesh_device_robot_refusals(robot_, code) : 0;
}

void Robot::printRefusals(Print& out) const {
  rlmesh_device_robot_status_t s = status();
  const char* last = rlmesh_device_refusal_name(s.last_refusal);
  out.printf("  refused %llu chunks, last %s:", (unsigned long long)s.chunks_refused,
             last != nullptr ? last : "none");
  for (uint8_t code = 1; code <= RLMESH_DEVICE_REFUSALS; ++code) {
    uint64_t n = refusals(code);
    if (n != 0) out.printf(" %s %llu", rlmesh_device_refusal_name(code), (unsigned long long)n);
  }
  out.printf("; %llu over %lu attaches:", (unsigned long long)refusalsTotal(0),
             (unsigned long)attaches());
  for (uint8_t code = 1; code <= RLMESH_DEVICE_REFUSALS; ++code) {
    uint64_t n = refusalsTotal(code);
    if (n != 0) out.printf(" %s %llu", rlmesh_device_refusal_name(code), (unsigned long long)n);
  }
  out.println();
}

uint64_t Robot::refusalsTotal(uint8_t code) const {
  return robot_ != nullptr ? rlmesh_device_robot_refusals_total(robot_, code) : 0;
}

uint32_t Robot::attaches() const {
  return robot_ != nullptr ? rlmesh_device_robot_attaches(robot_) : 0;
}

uint64_t Robot::dropped(uint8_t what, uint8_t code) const {
  return robot_ != nullptr ? rlmesh_device_robot_dropped(robot_, what, code) : 0;
}

void Robot::printDropped(Print& out) const {
  static const char* const kWhat[] = {"views", "replies", "chunks", "other"};
  out.print("  dropped:");
  bool any = false;
  for (uint8_t what = 0; what < 4; ++what) {
    for (uint8_t code = 1; code <= RLMESH_DEVICE_DROPPED; ++code) {
      uint64_t n = dropped(what, code);
      if (n == 0) continue;
      out.printf(" %s %s %llu", kWhat[what], rlmesh_device_dropped_name(code),
                 (unsigned long long)n);
      any = true;
    }
  }
  if (!any) out.print(" none");
  out.println();
}

void Robot::viewCounts(uint64_t& sealed, uint64_t& signedViews, uint64_t& droppedViews) const {
  sealed = signedViews = droppedViews = 0;
  if (robot_ != nullptr) {
    rlmesh_device_robot_view_counts(robot_, &sealed, &signedViews, &droppedViews);
  }
}

}  // namespace rlmesh
