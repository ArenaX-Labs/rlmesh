// rlmesh::Robot: the device robot over zenoh-pico, on the board's clock and
// randomness.

#include <ESPmDNS.h>
#include <Preferences.h>
#include <ctype.h>
#include <esp_wifi.h>
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

uint32_t Robot::transportTasks() const { return rlmesh_zenoh_tasks(); }

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
  // Samples the board did not send: no heap room, or (on a link slower than
  // it publishes) replaced by a newer one while waiting for the link.
  uint32_t replaced = 0, crowded = 0;
  if (zenoh_ != nullptr) rlmesh_zenoh_send_counts(zenoh_, &replaced, &crowded);
  out.printf("; unsent %llu, send queue replaced %lu dropped %lu\n",
             (unsigned long long)(robot_ != nullptr ? rlmesh_device_robot_unsent(robot_) : 0),
             (unsigned long)replaced, (unsigned long)crowded);
  // Each part's streams: what the sketch and the edges offered, what went
  // out, and the rate the delivery plan allows (every sample, or a rate).
  rlmesh_device_publishing_t p;
  for (int part = 0; robot_ != nullptr && part < RLMESH_DEVICE_NO_PART &&
                     rlmesh_device_robot_publishing(robot_, uint8_t(part), &p) == RLMESH_DEVICE_OK;
       ++part) {
    out.printf("  part %d published: state %llu of %llu at ", part,
               (unsigned long long)p.state_sent, (unsigned long long)p.state_offered);
    if (p.state_hz > 0) {
      out.printf("%.1f Hz", p.state_hz);
    } else {
      out.print("every sample");
    }
    out.printf(", applied %llu of %llu at ", (unsigned long long)p.applied_sent,
               (unsigned long long)p.applied_offered);
    if (p.applied_hz > 0) {
      out.printf("%.1f Hz\n", p.applied_hz);
    } else {
      out.println("every sample");
    }
  }
}

bool Robot::chunkTiming(int part, rlmesh_device_chunk_timing_t& out) const {
  out = {};
  if (robot_ == nullptr || part < 0 || part >= RLMESH_DEVICE_NO_PART) return false;
  return rlmesh_device_robot_chunk_timing(robot_, uint8_t(part), &out) == RLMESH_DEVICE_OK;
}

void Robot::printChunkTiming(Print& out, int part) const {
  rlmesh_device_chunk_timing_t t;
  if (!chunkTiming(part, t)) return;
  out.printf(
      "  chunks received %llu accepted %llu lost %lu late max %.1f ms gap max %.1f ms, by ms:",
      (unsigned long long)t.received, (unsigned long long)t.accepted, (unsigned long)t.lost,
      t.late_max_us / 1000.0, t.gap_max_us / 1000.0);
  static const char* const kBuckets[RLMESH_DEVICE_LATE_BUCKETS] = {
      "<1", "1-2", "2-4", "4-8", "8-16", "16-32", "32-64", ">=64"};
  for (int i = 0; i < RLMESH_DEVICE_LATE_BUCKETS; ++i) {
    if (t.late_ms[i] != 0) out.printf(" %s %lu", kBuckets[i], (unsigned long)t.late_ms[i]);
  }
  out.println();
}

void Robot::useDatagram(bool on) {
  if (robot_ != nullptr) rlmesh_device_robot_use_datagram(robot_, on ? 1 : 0);
}

bool Robot::datagramStats(rlmesh_device_datagram_stats_t& out) const {
  out = {};
  return robot_ != nullptr && rlmesh_device_robot_datagram_stats(robot_, &out) == RLMESH_DEVICE_OK;
}

void Robot::printDatagram(Print& out) const {
  // The radio, and whether the board's traffic waits on the access point's
  // beacons (102.4 ms): in power save the board's frames do, whichever path
  // they take; an access point that buffers for a station it takes to be
  // asleep lines up what the board receives; a send path held up by the
  // radio lines up the sends themselves (0: not at all, 1: every one).
  wifi_ps_type_t ps = WIFI_PS_NONE;
  if (esp_wifi_get_ps(&ps) == ESP_OK) {
    out.printf("  wifi power save %s", ps == WIFI_PS_NONE        ? "off"
                                       : ps == WIFI_PS_MIN_MODEM ? "on (min modem)"
                                                                 : "on (max modem)");
    wifi_ap_record_t ap;
    if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
      out.printf(", channel %u %s, rssi %d", ap.primary,
                 ap.second == WIFI_SECOND_CHAN_NONE ? "HT20" : "HT40", ap.rssi);
    }
    wifi_config_t config;
    if (esp_wifi_get_config(WIFI_IF_STA, &config) == ESP_OK) {
      out.printf(", listen interval %u", config.sta.listen_interval);
    }
    float sends = 0, receives = 0;
    uint32_t sent = 0, received = 0;
    if (zenoh_ != nullptr) rlmesh_zenoh_beacon_lock(zenoh_, &sends, &sent, &receives, &received);
    out.printf("; on the beacon: sends %.2f of %lu, receives %.2f of %lu\n", sends,
               (unsigned long)sent, receives, (unsigned long)received);
  }
  // The loop, since the last print: what arrives while it is away from
  // poll() waits, and a chunk that waits past its validity is refused (on
  // UDP) or replaced by a newer one (lost). Printing at 115200 baud with no
  // serial transmit buffer holds it about 87 us a character.
  if (zenoh_ != nullptr) {
    uint32_t longest = 0, held = 0;
    rlmesh_zenoh_poll_gaps(zenoh_, &longest, &held);
    out.printf("  loop: longest between polls %.1f ms, %lu over 50 ms\n", longest / 1000.0,
               (unsigned long)held);
  }
  rlmesh_device_datagram_stats_t d;
  if (!datagramStats(d)) {
    out.println("  realtime parts over tcp");
    return;
  }
  out.printf(
      "  realtime parts over %s: sent %llu, received %llu, lost %llu, reordered %llu, "
      "unopened %llu, fallbacks %lu, returns %lu, on udp %llu s, on tcp %llu s\n",
      d.active ? "udp" : "tcp", (unsigned long long)d.sent, (unsigned long long)d.received,
      (unsigned long long)d.lost, (unsigned long long)d.reordered, (unsigned long long)d.unopened,
      (unsigned long)d.fallbacks, (unsigned long)d.returns,
      (unsigned long long)(d.datagram_ms / 1000), (unsigned long long)(d.session_ms / 1000));
}

void Robot::viewCounts(uint64_t& sealed, uint64_t& signedViews, uint64_t& droppedViews) const {
  sealed = signedViews = droppedViews = 0;
  if (robot_ != nullptr) {
    rlmesh_device_robot_view_counts(robot_, &sealed, &signedViews, &droppedViews);
  }
}

}  // namespace rlmesh
