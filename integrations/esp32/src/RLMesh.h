// RLMesh: an ESP32 on Wi-Fi as an RLMesh robot. The session's
// protocol and crypto are RLMesh's Rust core, linked as a precompiled
// library (`rlmesh_device.h`); this wraps it with the board's clock and
// randomness, and zenoh-pico (`rlmesh_zenoh.h`) for its network.

#ifndef RLMESH_H_
#define RLMESH_H_

#include <sdkconfig.h>

#if !defined(CONFIG_IDF_TARGET_ESP32) && !defined(CONFIG_IDF_TARGET_ESP32C6) && \
    !defined(CONFIG_IDF_TARGET_ESP32S3)
#error "RLMesh runs on the classic ESP32 (ESP32-WROOM-32) ESP32-C6 and ESP32-S3"
#endif

#include <Print.h>
#include <Stream.h>

#include "rlmesh_device.h"

// The transport (rlmesh_zenoh.h): opaque here, so a sketch never sees
// zenoh-pico's headers.
struct rlmesh_zenoh_t;

namespace rlmesh {
namespace esp32 {

// The board's microsecond clock (esp_timer).
uint64_t nowUs();

// The hardware RNG. It is cryptographic only with the radio on (Wi-Fi or
// Bluetooth): start Wi-Fi before anything draws keys.
int fillRandom(uint8_t* buf, size_t len);

// What the session's crypto costs on this board (rlmesh_device_timing):
// about a second, blocking. Run it from setup(), never from a control loop.
inline int timing(rlmesh_device_timing_t& out) {
  return rlmesh_device_timing(&nowUs, &fillRandom, &out);
}

}  // namespace esp32

// The board's identity and its home host, kept in NVS:
// a device key it makes on first boot, and the record `rlmesh devices
// provision` writes over USB.
class Provisioning {
 public:
  Provisioning() = default;
  Provisioning(const Provisioning&) = delete;
  Provisioning& operator=(const Provisioning&) = delete;
  ~Provisioning();

  // Loads the device key, making it on first boot, and the named record.
  // Start Wi-Fi first: the RNG is cryptographic only with the
  // radio on. False if NVS fails.
  bool begin();
  // Answers the host on `port` (the USB serial port): call it every loop.
  // True on the call that stored a new record.
  bool poll(Stream& port);
  bool provisioned() const { return recordLen_ > 0; }
  // The public key the host enrolls.
  void publicKey(uint8_t out[32]) const;

 private:
  friend class Robot;
  uint8_t key_[32] = {};
  uint8_t record_[252] = {};
  size_t recordLen_ = 0;
  rlmesh_device_provision_t* responder_ = nullptr;
};

// The C ABI's tick result, including the reason no tick was produced.
enum class TickResult : int {
  Ready = 1,
  Idle = 0,
  InvalidArgument = RLMESH_DEVICE_ERR_ARG,
  NotReady = RLMESH_DEVICE_ERR_NOT_READY,
};

// The board as a robot in a session, over Wi-Fi. Start Wi-Fi first.
class Robot {
 public:
  Robot() = default;
  Robot(const Robot&) = delete;
  Robot& operator=(const Robot&) = delete;
  ~Robot() { end(); }

  // Joins the session at `address`, as `rlmesh host` prints it
  // ("192.168.1.20:7447#secret.hostkey"), as robot `name` with the map
  // `robotJson` (authored JSON, copied). Returns false on an address
  // without its key, or a map the host would refuse; joining itself goes on
  // in poll().
  bool begin(const char* address, const char* name, const char* robotJson,
             uint32_t heartbeatMs = 50);
  // Joins the home host `provisioning` holds, as its own device (session
  // "default" on the record's first address), using the stored robot name.
  // False when not provisioned.
  bool begin(const Provisioning& provisioning, const char* robotJson, uint32_t heartbeatMs = 50);
  // Runs the robot: call it every loop().
  void poll();
  void end();

  // The index of the part at `path` in the map, which tick() and publish()
  // take, or -1.
  int part(const char* path) const;
  // One control tick of `part`'s command group, at the part's control rate:
  // Ready with `out` filled (valid until the next call) when there is an
  // action to dispatch, a safe behaviour to run, or a handover. It also
  // publishes what was applied.
  TickResult tick(int part, rlmesh_device_tick_t& out);
  // The robot switched its control mode for a handover: the group may
  // dispatch again.
  void handoverDone(int part);
  // Publishes a sample of `part`, its dense value captured now:
  // RLMESH_DEVICE_OK, or _ERR_NOT_READY before the robot runs.
  int publish(int part, const void* value, size_t len);
  template <typename T>
  int publish(int part, const T& value) {
    return publish(part, &value, sizeof value);
  }

  // Latches this robot's e-stop (an e-stop pin): every part stops at once,
  // with or without the host, and the host is told. Only clearEstop()
  // clears it. A latch (this one, or a session e-stop the board received)
  // is kept in NVS: the board comes back latched after a reboot.
  void estop(const char* reason = "e-stop pin");
  // Clears the e-stop latch, whoever set it: call it only from a local
  // action on the board, with the e-stop released.
  void clearEstop();
  bool estopLatched() const { return status().estop != 0; }

  rlmesh_device_robot_status_t status() const;
  bool running() const { return status().phase == 4; }
  // Actions the transport failed to carry out.
  uint32_t transportFailures() const;
  // Network tasks running now (the transport's and zenoh-pico's, one per
  // open session): flat across re-attaches, or each one is leaking a stack.
  uint32_t transportTasks() const;
  // Why the edges last refused a command chunk ("stale_token"), or nullptr
  // while none has been refused this attach. status().chunks_refused counts
  // them.
  const char* lastRefusal() const;
  // Chunks refused this attach for one reason, by its code (1 to
  // RLMESH_DEVICE_REFUSALS; rlmesh_device_refusal_name names them).
  uint64_t refusals(uint8_t code) const;
  // Prints the refusals on one line, with a count per reason.
  void printRefusals(Print& out) const;
  // What the robot received and dropped over every attach, keeping what it
  // held: `what` 0 its views, 1 replies, 2 chunks that did not open, 3
  // anything else; `code` 1 to RLMESH_DEVICE_DROPPED
  // (rlmesh_device_dropped_name names them).
  uint64_t dropped(uint8_t what, uint8_t code) const;
  // Prints what was dropped on one line, by what and why.
  void printDropped(Print& out) const;
  // Sealed views taken, signed views taken, and views dropped, over every
  // attach.
  void viewCounts(uint64_t& sealed, uint64_t& signedViews, uint64_t& droppedViews) const;
  // Chunks refused for one reason over every attach, this one included
  // (code 0: every reason).
  uint64_t refusalsTotal(uint8_t code) const;
  // Attaches the host accepted: one more each time the robot joined again.
  uint32_t attaches() const;
  // How `part`'s command chunks arrived this attach (received, accepted,
  // lost, lateness on arrival by bucket, the longest gap): false before the
  // robot runs or for a part that takes no commands.
  bool chunkTiming(int part, rlmesh_device_chunk_timing_t& out) const;
  // Prints one part's chunk timing on one line.
  void printChunkTiming(Print& out, int part) const;
  // Whether the robot takes the host's datagram path (UDP) when the host
  // offers it (`rlmesh host --board-data udp`): true, the default, follows the
  // host; false keeps every part on TCP. Call it after begin(), before the
  // first poll(); it holds from the next attach on.
  void useDatagram(bool on);
  // The datagram path this attach (active, sent, received, lost, reordered,
  // unopened, fallbacks): false when the host offered none.
  bool datagramStats(rlmesh_device_datagram_stats_t& out) const;
  // Prints the datagram path's counters on one line, or the path in use.
  void printDatagram(Print& out) const;

 private:
  bool start(const rlmesh_device_robot_config_t& config);
  // Looks for the home over mDNS when no stored address answers.
  void discover();
  // Keeps the e-stop latch in NVS, writing only when it changes.
  void persistLatch();
  int8_t persisted_ = -1;
  uint32_t lastDiscovery_ = 0;
  bool mdns_ = false;
  char mdnsName_[40] = {0};

  rlmesh_device_robot_t* robot_ = nullptr;
  struct rlmesh_zenoh_t* zenoh_ = nullptr;
};

}  // namespace rlmesh

#endif  // RLMESH_H_
