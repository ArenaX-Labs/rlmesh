#ifndef ESP32_PORT_RANDOM_H_
#define ESP32_PORT_RANDOM_H_

#include <stddef.h>
#include <stdint.h>
#include <sys/random.h>

static inline void esp_fill_random(void* buf, size_t len) {
  while (len > 0) {
    ssize_t n = getrandom(buf, len, 0);
    if (n <= 0) continue;
    buf = (uint8_t*)buf + n;
    len -= (size_t)n;
  }
}
static inline uint32_t esp_random(void) {
  uint32_t r;
  esp_fill_random(&r, sizeof r);
  return r;
}

#endif
