// The board joins an RLMesh session over Wi-Fi as robot "eyes": a pan/tilt
// pair, declared to the host. Run a host on the computer, listening on its
// LAN address:
//
//   rlmesh host --listen 192.168.1.20:7447
//
// and paste the address it prints below. Open the Serial Monitor at 115200.
// Commands and samples come next (work in progress): so far the board
// joins, declares its parts and heartbeats.

#include <RLMeshESP32.h>
#include <WiFi.h>

const char* kSsid = "your-wifi";
const char* kPassword = "your-password";
const char* kAddress = "192.168.1.20:7447#paste-the-address-rlmesh-host-printed";

const char kMap[] =
    "{\"eyes\":{\"kind\":\"joints\",\"names\":[\"pan\",\"tilt\"],\"fields\":[\"pos\"],"
    "\"hz\":50,\"command\":\"position\",\"safe\":\"hold\","
    "\"limits\":{\"pos\":[[-0.6,0.6],[-0.4,0.4]]}}}";

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
uint8_t lastPhase = 255;
uint32_t lastReport = 0;

void setup() {
  Serial.begin(115200);
  delay(1000);
  WiFi.mode(WIFI_STA);
  WiFi.begin(kSsid, kPassword);
  Serial.printf("Wi-Fi: joining %s", kSsid);
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
    Serial.print('.');
  }
  Serial.printf(" %s\n", WiFi.localIP().toString().c_str());
  if (!robot.begin(kAddress, "eyes", kMap)) {
    Serial.println("can't reach the host: check kAddress (it ends in #key)");
  }
}

void loop() {
  robot.poll();
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
    Serial.printf("  generation %lu, epoch %llu, mode %u, %llu echoes, heap %lu\n",
                  (unsigned long)s.generation, (unsigned long long)s.epoch, s.mode,
                  (unsigned long long)s.echoes, (unsigned long)ESP.getFreeHeap());
  }
  delay(1);
}
