// The board joins an RLMesh session over Wi-Fi as robot "eyes": two hobby
// servos (pan on GPIO 18, tilt on 19) driven by whoever holds the lease. Run
// a host on the computer, listening on its LAN address:
//
//   rlmesh host --listen 192.168.1.20:7447
//
// and paste the address it prints below. Open the Serial Monitor at 115200.
// Power the servos separately, not from the board's 3.3 V.

#include <RLMeshESP32.h>
#include <WiFi.h>

const char* kSsid = "your-wifi";
const char* kPassword = "your-password";
const char* kAddress = "192.168.1.20:7447#paste-the-address-rlmesh-host-printed";

const char kMap[] =
    "{\"eyes\":{\"kind\":\"joints\",\"names\":[\"pan\",\"tilt\"],\"fields\":[\"pos\"],"
    "\"hz\":50,\"command\":\"position\",\"safe\":\"hold\","
    "\"limits\":{\"pos\":[[-0.6,0.6],[-0.4,0.4]]}}}";

const int kPanPin = 18;
const int kTiltPin = 19;
const uint32_t kPeriodUs = 20000;  // the map's 50 Hz

const char* const kPhases[] = {"reading the host", "handshake", "attaching",
                               "declaring parts",  "running",   "retrying"};
const char* const kFaults[] = {
    "",
    "no reply from the host",
    "the host's state is unsigned",
    "the handshake failed: the address's key isn't this host's",
    "refused by the host",
    "lost the host",
    "internal error",
};

rlmesh::Robot robot;
int eyes = -1;
float target[2] = {0, 0};  // radians: pan, tilt
uint8_t lastPhase = 255;
uint32_t lastReport = 0;
uint64_t nextTick = 0;

// A hobby servo: 1500 us centred, 1000 us either way over about 90 degrees.
void writeServo(int pin, float rad) {
  float us = 1500.0f + rad * (1000.0f / 1.5708f);
  ledcWrite(pin, uint32_t(us * 16384.0f / 20000.0f));
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  ledcAttach(kPanPin, 50, 14);
  ledcAttach(kTiltPin, 50, 14);
  writeServo(kPanPin, 0);
  writeServo(kTiltPin, 0);

  WiFi.mode(WIFI_STA);
  WiFi.begin(kSsid, kPassword);
  Serial.printf("Wi-Fi: joining %s", kSsid);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print('.');
  }
  Serial.printf(" %s\n", WiFi.localIP().toString().c_str());
  if (!robot.begin(kAddress, "eyes", kMap)) {
    Serial.println("check kAddress: it is what `rlmesh host` printed, ending in #key");
    return;
  }
  eyes = robot.part("eyes");
}

void report() {
  rlmesh_device_robot_status_t s = robot.status();
  if (s.phase != lastPhase) {
    lastPhase = s.phase;
    Serial.printf("%8lu ms  %s", (unsigned long)millis(), kPhases[s.phase < 6 ? s.phase : 5]);
    if (s.fault != 0 && s.fault < 7) {
      Serial.printf(" (last: %s", kFaults[s.fault]);
      if (s.fault == 4) Serial.printf(", status %u", s.refused_status);
      Serial.print(')');
    }
    Serial.println();
  }
  if (robot.running() && millis() - lastReport > 5000) {
    lastReport = millis();
    Serial.printf("  mode %u, epoch %llu, %llu chunks, %llu echoes, heap %lu\n", s.mode,
                  (unsigned long long)s.epoch, (unsigned long long)s.chunks_accepted,
                  (unsigned long long)s.echoes, (unsigned long)ESP.getFreeHeap());
  }
}

void loop() {
  robot.poll();
  report();
  uint64_t now = rlmesh::esp32::nowUs();
  if (eyes < 0 || !robot.running() || now < nextTick) return;
  nextTick = now + kPeriodUs;
  rlmesh_device_tick_t t;
  if (robot.tick(eyes, t) && t.has_action && t.value_len == sizeof target) {
    memcpy(target, t.value, sizeof target);
    writeServo(kPanPin, target[0]);
    writeServo(kTiltPin, target[1]);
  }
  // A safe tick with no action: the declared behaviour is "hold", so the
  // servos keep their last target. A "stop" part would detach them here.
  // No position sensors: report the last target as the state.
  robot.publish(eyes, target);
}
