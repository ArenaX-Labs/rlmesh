/* The few FreeRTOS types and constants zenoh-pico's ESP32 port uses, for its
 * test on Linux (esp32_port.c implements the calls). */
#ifndef ESP32_PORT_FREERTOS_H_
#define ESP32_PORT_FREERTOS_H_

#include <stddef.h>
#include <stdint.h>

typedef int BaseType_t;
typedef unsigned int UBaseType_t;
typedef uint32_t TickType_t;
typedef uint8_t StackType_t;
typedef struct StaticTask_t StaticTask_t;

#define pdPASS 1
#define pdTRUE 1
#define pdFALSE 0
#define portMAX_DELAY ((TickType_t)0xffffffffu)
#define configMAX_PRIORITIES 25
#define configSUPPORT_STATIC_ALLOCATION 0
#define tskNO_AFFINITY 0x7fffffff
#define portNUM_PROCESSORS 2

#endif
