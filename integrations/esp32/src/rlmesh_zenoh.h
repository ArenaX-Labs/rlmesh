/* The board's transport: zenoh-pico carrying out the device robot's actions
 * (rlmesh_device.h). It holds no protocol: it subscribes, declares, queries and
 * publishes what the robot asks, and hands back what arrives. Plain C, so the
 * same file runs on a board and in the host test against a real session host.
 *
 * The host runs one session per traffic class, on its address's port plus the
 * class; this opens a client session to each class the robot uses, when it
 * first uses it.
 *
 * zenoh-pico's callbacks run on its read tasks; they only queue. The robot is
 * driven from rlmesh_zenoh_poll, on the caller's thread. */

#ifndef RLMESH_ZENOH_H_
#define RLMESH_ZENOH_H_

#include <stdbool.h>
#include <stdint.h>
#include <zenoh-pico.h>

#include "rlmesh_device.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Subscriptions: the host's state and ticks, the heartbeat echo, key
 * deliveries, and a commandable part's commands each. */
#define RLMESH_ZENOH_SUBS 16
#define RLMESH_ZENOH_TOKENS 2

struct rlmesh_zenoh_event;
struct rlmesh_zenoh_t;

/* A subscription's callback context. */
typedef struct rlmesh_zenoh_sub_t {
  struct rlmesh_zenoh_t* z;
  uint8_t sub;
} rlmesh_zenoh_sub_t;

typedef struct rlmesh_zenoh_t {
  z_owned_session_t sessions[RLMESH_DEVICE_CLASSES];
  bool open[RLMESH_DEVICE_CLASSES];
  /* The host the open sessions reach: a reset to another closes them. */
  char host[64];
  uint16_t port;
  z_owned_subscriber_t subs[RLMESH_ZENOH_SUBS];
  rlmesh_zenoh_sub_t sub_ctx[RLMESH_ZENOH_SUBS];
  uint8_t sub_count;
  z_owned_liveliness_token_t tokens[RLMESH_ZENOH_TOKENS];
  uint8_t token_count;
  z_owned_mutex_t lock;
  struct rlmesh_zenoh_event* head;
  struct rlmesh_zenoh_event* tail;
  rlmesh_device_robot_t* robot;
  rlmesh_device_now_us_fn now_us;
  /* Actions the transport failed to carry out, and events dropped for want
   * of memory. */
  uint32_t failed;
} rlmesh_zenoh_t;

/* Prepares the transport for `robot`, which stays the caller's: sessions open
 * to the robot's host (rlmesh_device_robot_host) as its actions need them. 0
 * on success. */
int rlmesh_zenoh_open(rlmesh_zenoh_t* z, rlmesh_device_robot_t* robot,
                      rlmesh_device_now_us_fn now_us);

/* Feeds the robot what arrived, then carries out its actions. Call it every
 * loop. 0, or the robot's error code. */
int rlmesh_zenoh_poll(rlmesh_zenoh_t* z);

/* Closes every session; the robot is left to the caller. */
void rlmesh_zenoh_close(rlmesh_zenoh_t* z);

#ifdef __cplusplus
}
#endif

#endif /* RLMESH_ZENOH_H_ */
