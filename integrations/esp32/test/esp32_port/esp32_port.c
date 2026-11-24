#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "esp32_port.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "freertos/event_groups.h"
#include "freertos/task.h"

struct esp32_port_task {
  pthread_t thread;
  TaskFunction_t fn;
  void* arg;
  uint32_t stack;
  pthread_mutex_t lock;
  pthread_cond_t wake;
  bool suspended;
  bool deleted;
};

struct esp32_port_event_group {
  pthread_mutex_t lock;
  pthread_cond_t wake;
  EventBits_t bits;
};

static atomic_uint created;
static atomic_uint freed;
static atomic_uint awaiting_idle;
static atomic_ulong stacks;
static _Thread_local struct esp32_port_task* current;
static struct esp32_port_task outside;

static void pause_ms(unsigned ms) {
  struct timespec pause = {ms / 1000, (long)(ms % 1000) * 1000000L};
  nanosleep(&pause, NULL);
}

static void* run(void* arg) {
  struct esp32_port_task* t = (struct esp32_port_task*)arg;
  current = t;
  t->fn(t->arg);
  fprintf(stderr, "esp32_port: a task function returned\n");
  abort();
}

static void release(struct esp32_port_task* t) {
  atomic_fetch_add(&freed, 1);
  atomic_fetch_sub(&stacks, t->stack);
}

BaseType_t xTaskCreatePinnedToCore(TaskFunction_t fn, const char* name, uint32_t stack_depth,
                                   void* arg, UBaseType_t priority, TaskHandle_t* out,
                                   BaseType_t core) {
  (void)name;
  (void)priority;
  struct esp32_port_task* t = calloc(1, sizeof *t);
  if (t == NULL) return 0;
  t->fn = fn;
  t->arg = arg;
  t->stack = stack_depth;
  pthread_mutex_init(&t->lock, NULL);
  pthread_cond_init(&t->wake, NULL);
  if (out != NULL) *out = t;
  atomic_fetch_add(&created, 1);
  atomic_fetch_add(&stacks, stack_depth);
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  if (core != tskNO_AFFINITY) {
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(core, &cpus);
    pthread_attr_setaffinity_np(&attr, sizeof cpus, &cpus);
  }
  int created = pthread_create(&t->thread, &attr, run, t);
  pthread_attr_destroy(&attr);
  if (created != 0) {
    release(t);
    free(t);
    return 0;
  }
  pthread_detach(t->thread);
  return pdPASS;
}

BaseType_t xTaskCreate(TaskFunction_t fn, const char* name, uint32_t stack_depth, void* arg,
                       UBaseType_t priority, TaskHandle_t* out) {
  return xTaskCreatePinnedToCore(fn, name, stack_depth, arg, priority, out, tskNO_AFFINITY);
}

TaskHandle_t xTaskGetCurrentTaskHandle(void) { return current != NULL ? current : &outside; }

eTaskState eTaskGetState(TaskHandle_t t) {
  if (t == NULL || t == &outside) return eRunning;
  pthread_mutex_lock(&t->lock);
  eTaskState state = t->deleted ? eDeleted : t->suspended ? eSuspended : eRunning;
  pthread_mutex_unlock(&t->lock);
  return state;
}

void vTaskSuspend(TaskHandle_t task) {
  struct esp32_port_task* t = task != NULL ? task : current;
  if (t == NULL || t != current) {
    fprintf(stderr, "esp32_port: only a task suspending itself is supported\n");
    abort();
  }
  pthread_mutex_lock(&t->lock);
  t->suspended = true;
  pthread_cond_broadcast(&t->wake);
  while (!t->deleted) pthread_cond_wait(&t->wake, &t->lock);
  pthread_mutex_unlock(&t->lock);
  free(t);
  pthread_exit(NULL);
}

void vTaskDelete(TaskHandle_t task) {
  struct esp32_port_task* t = task != NULL ? task : current;
  if (t == NULL || t == &outside) return;
  if (t == current) {
    atomic_fetch_add(&awaiting_idle, 1);
    pthread_exit(NULL);
  }
  pthread_mutex_lock(&t->lock);
  bool suspended = t->suspended;
  t->deleted = true;
  pthread_cond_broadcast(&t->wake);
  pthread_mutex_unlock(&t->lock);
  if (suspended) {
    release(t);
  } else {
    atomic_fetch_add(&awaiting_idle, 1);
  }
}

void vTaskDelay(TickType_t ticks) { pause_ms(ticks); }

EventGroupHandle_t xEventGroupCreate(void) {
  struct esp32_port_event_group* g = calloc(1, sizeof *g);
  if (g == NULL) return NULL;
  pthread_mutex_init(&g->lock, NULL);
  pthread_cond_init(&g->wake, NULL);
  return g;
}

EventBits_t xEventGroupSetBits(EventGroupHandle_t g, EventBits_t bits) {
  pthread_mutex_lock(&g->lock);
  g->bits |= bits;
  EventBits_t now = g->bits;
  pthread_cond_broadcast(&g->wake);
  pthread_mutex_unlock(&g->lock);
  if (current != NULL) pause_ms(2);
  return now;
}

EventBits_t xEventGroupWaitBits(EventGroupHandle_t g, EventBits_t bits, BaseType_t clear,
                                BaseType_t all, TickType_t ticks) {
  (void)ticks;
  pthread_mutex_lock(&g->lock);
  while (all ? (g->bits & bits) != bits : (g->bits & bits) == 0) {
    pthread_cond_wait(&g->wake, &g->lock);
  }
  EventBits_t now = g->bits;
  if (clear) g->bits &= ~bits;
  pthread_mutex_unlock(&g->lock);
  return now;
}

void vEventGroupDelete(EventGroupHandle_t g) {
  if (g == NULL) return;
  pthread_mutex_destroy(&g->lock);
  pthread_cond_destroy(&g->wake);
  free(g);
}

void esp32_port_tasks(esp32_port_tasks_t* out) {
  out->created = atomic_load(&created);
  out->freed = atomic_load(&freed);
  out->awaiting_idle = atomic_load(&awaiting_idle);
  out->stack_bytes = atomic_load(&stacks);
}
