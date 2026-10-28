// What the session's crypto costs on this board: the attach handshake, a
// signed authority state, and the chunk and sample of each control tick.
// Open the Serial Monitor at 115200.

#include <RLMeshESP32.h>
#include <WiFi.h>

// One control tick at 100 Hz.
const uint32_t kTickUs = 10000;

void row(const char* what, uint32_t us) {
  Serial.printf("  %-34s %8lu us\n", what, (unsigned long)us);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  // The hardware RNG is cryptographic only with the radio on.
  WiFi.mode(WIFI_STA);

  uint32_t v = rlmesh_device_version();
  Serial.printf("RLMeshESP32 timing (library %lu.%lu.%lu, %lu MHz)\n", (unsigned long)(v >> 16),
                (unsigned long)(v >> 8 & 0xFF), (unsigned long)(v & 0xFF),
                (unsigned long)getCpuFrequencyMhz());
  rlmesh_device_timing_t t;
  int rc = rlmesh::esp32::timing(t);
  if (rc != RLMESH_DEVICE_OK) {
    Serial.printf("timing failed: %d\n", rc);
    return;
  }
  row("handshake: msg 1", t.handshake_start_us);
  row("handshake: msg 2", t.handshake_finish_us);
  row("authority state: verify (1 KiB)", t.state_verify_us);
  row("chunk: open, signed (256 B)", t.chunk_open_sign_us);
  row("chunk: open, sealed (256 B)", t.chunk_open_seal_us);
  row("sample: seal, signed (64 B)", t.sample_seal_sign_us);
  uint32_t tick = t.chunk_open_sign_us + t.sample_seal_sign_us;
  Serial.printf("one 100 Hz tick (chunk + sample, signed): %lu us, %lu.%lu%% of a core\n",
                (unsigned long)tick, (unsigned long)(tick * 100 / kTickUs),
                (unsigned long)(tick * 1000 / kTickUs % 10));
  Serial.printf("heap: %lu free, %lu lowest\n", (unsigned long)ESP.getFreeHeap(),
                (unsigned long)ESP.getMinFreeHeap());
  // loopTask's stack is 8 KB.
  Serial.printf("stack: %lu bytes never used\n", (unsigned long)uxTaskGetStackHighWaterMark(NULL));
}

void loop() {}
