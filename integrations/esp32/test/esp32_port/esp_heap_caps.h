#ifndef ESP32_PORT_HEAP_CAPS_H_
#define ESP32_PORT_HEAP_CAPS_H_

#include <stdlib.h>

#define MALLOC_CAP_8BIT 4
static inline void* heap_caps_malloc(size_t size, unsigned caps) {
  (void)caps;
  return malloc(size);
}
static inline void* heap_caps_realloc(void* ptr, size_t size, unsigned caps) {
  (void)caps;
  return realloc(ptr, size);
}
static inline void heap_caps_free(void* ptr) { free(ptr); }

#endif
