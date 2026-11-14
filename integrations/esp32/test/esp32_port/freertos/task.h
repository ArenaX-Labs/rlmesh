#ifndef ESP32_PORT_TASK_H_
#define ESP32_PORT_TASK_H_

#include "freertos/FreeRTOS.h"

typedef struct esp32_port_task* TaskHandle_t;
typedef void (*TaskFunction_t)(void*);
typedef enum { eRunning, eReady, eBlocked, eSuspended, eDeleted, eInvalid } eTaskState;

BaseType_t xTaskCreate(TaskFunction_t fn, const char* name, uint32_t stack_depth, void* arg,
                       UBaseType_t priority, TaskHandle_t* created);
BaseType_t xTaskCreatePinnedToCore(TaskFunction_t fn, const char* name, uint32_t stack_depth,
                                   void* arg, UBaseType_t priority, TaskHandle_t* created,
                                   BaseType_t core);
void vTaskDelete(TaskHandle_t task);
void vTaskSuspend(TaskHandle_t task);
TaskHandle_t xTaskGetCurrentTaskHandle(void);
void vTaskDelay(TickType_t ticks);
eTaskState eTaskGetState(TaskHandle_t task);

#endif
