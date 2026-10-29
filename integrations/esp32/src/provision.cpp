// rlmesh::Provisioning: the device key and the provisioning record in NVS
// (Preferences), and the tether spec §14 responder on a serial port.

#include <Preferences.h>
#include <string.h>

#include "RLMeshESP32.h"

namespace rlmesh {

namespace {

const char kNamespace[] = "rlmesh";
const char kKey[] = "key";
const char kRecord[] = "record";

#if defined(CONFIG_IDF_TARGET_ESP32)
const uint8_t kArch = 4;  // Xtensa
#else
const uint8_t kArch = 3;  // RISC-V
#endif

}  // namespace

Provisioning::~Provisioning() {
  rlmesh_device_provision_free(responder_);
  memset(key_, 0, sizeof key_);
}

bool Provisioning::begin(const char* name) {
  rlmesh_device_provision_free(responder_);
  responder_ = nullptr;
  Preferences nvs;
  if (!nvs.begin(kNamespace, false)) return false;
  if (nvs.getBytesLength(kKey) != sizeof key_ ||
      nvs.getBytes(kKey, key_, sizeof key_) != sizeof key_) {
    // First boot: a new device. Each put is one NVS commit, which keeps the
    // old value if power fails mid-write.
    esp32::fillRandom(key_, sizeof key_);
    if (nvs.putBytes(kKey, key_, sizeof key_) != sizeof key_) {
      nvs.end();
      return false;
    }
  }
  recordLen_ = nvs.getBytesLength(kRecord);
  if (recordLen_ > sizeof record_ || nvs.getBytes(kRecord, record_, recordLen_) != recordLen_ ||
      !rlmesh_device_record_valid(record_, recordLen_)) {
    recordLen_ = 0;
  }
  nvs.end();
  return rlmesh_device_provision_new(name, key_, kArch, record_, recordLen_, &responder_) ==
         RLMESH_DEVICE_OK;
}

void Provisioning::publicKey(uint8_t out[32]) const { rlmesh_device_key_public(key_, out); }

bool Provisioning::poll(Stream& port) {
  bool stored = false;
  while (responder_ != nullptr && port.available() > 0) {
    const uint8_t* data = nullptr;
    size_t len = 0;
    int step = rlmesh_device_provision_push(responder_, uint8_t(port.read()), &data, &len);
    if (step == RLMESH_DEVICE_PROVISION_STORE) {
      bool ok = len <= sizeof record_;
      if (ok) {
        Preferences nvs;
        ok = nvs.begin(kNamespace, false) && nvs.putBytes(kRecord, data, len) == len;
        nvs.end();
      }
      if (ok) {
        memcpy(record_, data, len);
        recordLen_ = len;
        stored = true;
      }
      rlmesh_device_provision_stored(responder_, ok ? 1 : 0, &data, &len);
      step = RLMESH_DEVICE_PROVISION_REPLY;
    }
    if (step == RLMESH_DEVICE_PROVISION_REPLY) {
      port.write(data, len);
      port.flush();
    }
  }
  return stored;
}

}  // namespace rlmesh
