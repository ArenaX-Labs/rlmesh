/* What the FreeRTOS stand-in (esp32_port.c) counted: tasks created, those
 * freed, those deleted but left for the idle task to free, and the stack
 * bytes held by every task not yet freed. */
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
