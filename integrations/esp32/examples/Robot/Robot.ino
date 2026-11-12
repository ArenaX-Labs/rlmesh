// The board joins an RLMesh session over Wi-Fi under its provisioned name: two hobby
// servos (pan on GPIO 18, tilt on 19) driven by whoever holds the lease.
// Power the servos separately, not from the board's 3.3 V.
//
// Its home is the `rlmesh host` it was provisioned for. With the board on
// USB, on the host's computer (close the Serial Monitor first):
//
//   rlmesh devices provision --port /dev/ttyUSB0 --name eye-3
//
// It joins the host's session with its own device key as soon as it is
// provisioned, and on every boot after. Every eye uses this same firmware;
// its name comes from USB provisioning. The firmware holds no secret.

#include <RLMesh.h>
#include <WiFi.h>

#if defined(ARDUINO_PARTITION_default)
#error "RLMesh examples: select Tools > Partition Scheme > Minimal SPIFFS and keep the bundled partitions.csv"
#endif

const char* kSsid = "your-wifi";
const char* kPassword = "your-password";

const char kMap[] =
    "{\"eyes\":{\"kind\":\"joints\",\"names\":[\"pan\",\"tilt\"],\"fields\":[\"pos\"],"
    "\"hz\":50,\"command\":\"position\",\"safe\":\"hold\","
    "\"limits\":{\"pos\":[[-0.6,0.6],[-0.4,0.4]]}}}";

const int kPanPin = 18;
const int kTiltPin = 19;
// An e-stop button, or -1 for none: normally closed, between this pin and
// GND, so a press or a cut wire reads high. It latches the board's e-stop;
// hold the BOOT button for 2 s, with the e-stop released, to clear it.
const int kEstopPin = -1;
#if defined(CONFIG_IDF_TARGET_ESP32C6)
const int kBootPin = 9;
#else
const int kBootPin = 0;
#endif
const uint32_t kPeriodUs = 20000;  // the map's 50 Hz

const char* const kPhases[] = {"reading the host",
                               "handshake",
                               "attaching",
                               "declaring parts",
                               "running",
                               "retrying",
                               "waiting for a run (lobby)",
                               "stopped: another board has its name"};
const char* const kFaults[] = {
    "",
    "no reply from the host",
    "the handshake failed: the address's key isn't this host's",
    "refused by the host",
    "lost the host",
    "internal error",
};

rlmesh::Provisioning provisioning;
rlmesh::Robot robot;
int eyes = -1;
uint32_t bootHeldSince = 0;
float target[2] = {0, 0};  // radians: pan, tilt
uint8_t lastPhase = 255;
uint32_t lastReport = 0;
uint8_t lastRefusal = 0;
uint64_t lastRefused = 0;
uint32_t lastRefusalReport = 0;
uint64_t lastDropped = 0;
uint32_t lastDroppedCheck = 0;
uint64_t nextTick = 0;

// A hobby servo: 1500 us centred, 1000 us either way over about 90 degrees.
void writeServo(int pin, float rad) {
  float us = 1500.0f + rad * (1000.0f / 1.5708f);
  ledcWrite(pin, uint32_t(us * 16384.0f / 20000.0f));
}

void setup() {
  // Room for a status report: with none, each print waits for the port (87 us
  // a character at 115200 baud), and the loop with it; a report of a few
  // hundred characters then holds commands that arrive meanwhile past their
  // life.
  Serial.setTxBufferSize(2048);
  Serial.begin(115200);
  delay(1000);
  ledcAttach(kPanPin, 50, 14);
  ledcAttach(kTiltPin, 50, 14);
  writeServo(kPanPin, 0);
  writeServo(kTiltPin, 0);
  if (kEstopPin >= 0) pinMode(kEstopPin, INPUT_PULLUP);
  pinMode(kBootPin, INPUT_PULLUP);

  WiFi.mode(WIFI_STA);
  WiFi.begin(kSsid, kPassword);
  Serial.printf("Wi-Fi: joining %s", kSsid);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print('.');
  }
  Serial.printf(" %s\n", WiFi.localIP().toString().c_str());
  // After Wi-Fi: a first boot draws the device key from the RNG.
  if (!provisioning.begin()) Serial.println("NVS failed: no device key");
  join();
}

void join() {
  // A latch outlives a new home: the next robot starts latched.
  bool latched = robot.estopLatched();
  bool ok = robot.begin(provisioning, kMap);
  if (latched) robot.estop("e-stop latched before a new home");
  if (!ok) {
    Serial.println("not provisioned: rlmesh devices provision --port <this port> --name eye-3");
  }
  eyes = robot.part("eyes");
}

void report() {
  rlmesh_device_robot_status_t s = robot.status();
  if (s.phase != lastPhase) {
    lastPhase = s.phase;
    Serial.printf("%8lu ms  %s%s", (unsigned long)millis(), s.in_run ? "run: " : "",
                  kPhases[s.phase < 8 ? s.phase : 5]);
    if (s.fault != 0 && s.fault < 6) {
      Serial.printf(" (last: %s", kFaults[s.fault]);
      if (s.fault == 3) Serial.printf(", status %u", s.refused_status);
      Serial.print(')');
    }
    Serial.println();
  }
  // Why commands are refused: at once when the reason changes, and at most
  // every 5 s while they keep coming.
  if (s.last_refusal != lastRefusal ||
      (s.chunks_refused != lastRefused && millis() - lastRefusalReport > 5000)) {
    lastRefusal = s.last_refusal;
    lastRefused = s.chunks_refused;
    lastRefusalReport = millis();
    robot.printRefusals(Serial);
  }
  // What the board could not take (a view too large for it, a reply that
  // did not open): checked every 5 s, printed when it grew.
  if (millis() - lastDroppedCheck > 5000) {
    lastDroppedCheck = millis();
    uint64_t dropped = 0;
    for (uint8_t what = 0; what < 4; ++what) {
      for (uint8_t code = 1; code <= RLMESH_DEVICE_DROPPED; ++code) {
        dropped += robot.dropped(what, code);
      }
    }
    if (dropped != lastDropped) {
      lastDropped = dropped;
      robot.printDropped(Serial);
    }
  }
  if (robot.running() && millis() - lastReport > 5000) {
    lastReport = millis();
    Serial.printf("  mode %u, epoch %llu, %llu chunks, %llu echoes, e-stop %u, heap %lu\n", s.mode,
                  (unsigned long long)s.epoch, (unsigned long long)s.chunks_accepted,
                  (unsigned long long)s.echoes, s.estop, (unsigned long)ESP.getFreeHeap());
    uint64_t sealed, signedViews, droppedViews;
    robot.viewCounts(sealed, signedViews, droppedViews);
    Serial.printf("  views: %llu sealed, %llu signed, %llu dropped\n", (unsigned long long)sealed,
                  (unsigned long long)signedViews, (unsigned long long)droppedViews);
    robot.printChunkTiming(Serial, eyes);
    robot.printDatagram(Serial);
  }
}

// The e-stop button, every loop: it latches at once, session or not. BOOT
// held for 2 s clears it, with the e-stop released.
void estopButton() {
  bool pressed = kEstopPin >= 0 && digitalRead(kEstopPin) == HIGH;
  if (pressed && !robot.estopLatched()) {
    robot.estop("e-stop button");
    Serial.println("e-stop latched");
  }
  bool boot = digitalRead(kBootPin) == LOW;
  if (!boot) {
    bootHeldSince = 0;
  } else if (bootHeldSince == 0) {
    bootHeldSince = millis();
  } else if (millis() - bootHeldSince > 2000 && robot.estopLatched()) {
    if (pressed) {
      Serial.println("release the e-stop button first");
    } else {
      robot.clearEstop();
      Serial.println("e-stop cleared: disarm and arm the session to drive again");
    }
    bootHeldSince = 0;
  }
}

void loop() {
  // The host provisions over this port at any time; a new home is joined
  // at once.
  if (provisioning.poll(Serial)) {
    Serial.println("provisioned");
    join();
  }
  robot.poll();
  estopButton();
  report();
  uint64_t now = rlmesh::esp32::nowUs();
  if (eyes < 0 || !robot.running() || now < nextTick) return;
  nextTick = now + kPeriodUs;
  rlmesh_device_tick_t t;
  if (robot.tick(eyes, t) == rlmesh::TickResult::Ready && t.has_action && t.value_len == sizeof target) {
    memcpy(target, t.value, sizeof target);
    writeServo(kPanPin, target[0]);
    writeServo(kTiltPin, target[1]);
  }
  // A safe tick with no action: the declared behaviour is "hold", so the
  // servos keep their last target. A "stop" part would detach them here.
  // No position sensors: report the last target as the state.
  robot.publish(eyes, target);
}
