#ifndef ESP32_PORT_H_
#define ESP32_PORT_H_

#include <stdbool.h>
#include <stdint.h>

typedef struct esp32_port_tasks_t {
  uint32_t created;
  uint32_t freed;
  uint32_t awaiting_idle;
  unsigned long stack_bytes;
} esp32_port_tasks_t;

void esp32_port_tasks(esp32_port_tasks_t* out);

#endif
