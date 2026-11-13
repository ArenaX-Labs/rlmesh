#ifndef ESP32_PORT_EVENT_GROUPS_H_
#define ESP32_PORT_EVENT_GROUPS_H_

#include "freertos/FreeRTOS.h"

typedef uint32_t EventBits_t;
typedef struct esp32_port_event_group* EventGroupHandle_t;

EventGroupHandle_t xEventGroupCreate(void);
EventBits_t xEventGroupSetBits(EventGroupHandle_t group, EventBits_t bits);
EventBits_t xEventGroupWaitBits(EventGroupHandle_t group, EventBits_t bits, BaseType_t clear,
                                BaseType_t all, TickType_t ticks);
void vEventGroupDelete(EventGroupHandle_t group);

#endif
