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
 * driven from rlmesh_zenoh_poll, on the caller's thread. Its signed views are
 * checked on a task of their own first: one signature check takes longer than
 * a control period, and the robot only has to take the result. What it
 * publishes goes out on another: sending waits for the socket and the radio.
 * Both tasks, like zenoh-pico's, stay off the core of a board's control loop. */

#ifndef RLMESH_ZENOH_H_
#define RLMESH_ZENOH_H_

#include <stdbool.h>
#include <stdint.h>
#include <zenoh-pico.h>

#include "rlmesh_device.h"
#include "rlmesh_txq.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Subscriptions: the host's state and ticks, the heartbeat echo, key
 * deliveries, and a commandable part's commands each. */
#define RLMESH_ZENOH_SUBS 16
#define RLMESH_ZENOH_TOKENS 2
/* Bytes of samples and verified views queued for the robot at most: one past
 * it is dropped (and counted in `failed`). Replies and query ends always
 * queue: the robot's own queries bound them. */
#define RLMESH_ZENOH_QUEUE_BYTES (32 * 1024)
/* How long a read blocks zenoh-pico's one task per session, in milliseconds:
 * its keep-alives, lease checks and query timeouts wait behind it, and closing
 * the session waits for it. Long enough for a handshake over Wi-Fi in power
 * save. */
#define RLMESH_ZENOH_RECV_TIMEOUT_MS "1000"

/* The task that checks signatures: its stack (one check needs about 9 KB,
 * test/check_stack.py holds it to this) and FreeRTOS priority, the lowest
 * above the idle task's: a check takes longer than a control period and views
 * come several times a second, so it runs beside the radio, away from the
 * loop. After each check it blocks for a tick, so the idle task always gets
 * time (the task watchdog watches it) whatever arrives. */
#define RLMESH_ZENOH_VERIFY_STACK (12 * 1024)
#define RLMESH_ZENOH_VERIFY_PRIORITY 1

/* The task that publishes: its stack, sized from what the send path was
 * measured to use on a board, and FreeRTOS priority, that of zenoh-pico's read
 * tasks. */
#define RLMESH_ZENOH_TX_STACK (8 * 1024)
#define RLMESH_ZENOH_TX_PRIORITY 12
/* Bytes of publications waiting to go out at most: past it the oldest on the
 * same key is dropped (counted in rlmesh_zenoh_failures), as with a link that
 * can't keep up. */
#define RLMESH_ZENOH_TX_BYTES (8 * 1024)

struct rlmesh_zenoh_event;
struct rlmesh_zenoh_view;
struct rlmesh_zenoh_t;

/* Signed views since the transport opened: arrived, replaced by a newer one
 * before the signature task reached them, refused by it, and taken by the
 * robot, with the longest and total time from arrival to taken. A view the
 * task falls behind on is replaced, never queued: the robot keeps acting on
 * the last view it took until that view's lease expires by its own clock. */
typedef struct rlmesh_zenoh_view_stats_t {
  uint32_t offered;
  uint32_t replaced;
  uint32_t refused;
  uint32_t taken;
  uint64_t wait_max_us;
  uint64_t wait_total_us;
  /* Publications dropped from a full send queue, and those the send task
   * failed to send (or an action the transport failed to carry out). */
  uint32_t tx_dropped;
  uint32_t failed;
} rlmesh_zenoh_view_stats_t;

/* A subscription's callback context. */
typedef struct rlmesh_zenoh_sub_t {
  struct rlmesh_zenoh_t* z;
  uint8_t sub;
} rlmesh_zenoh_sub_t;

typedef struct rlmesh_zenoh_t {
  z_owned_session_t sessions[RLMESH_DEVICE_CLASSES];
  bool open[RLMESH_DEVICE_CLASSES];
  /* Opening this class failed: not tried again before the robot's next reset,
   * so its retries pace the attempts (each can block for seconds). */
  bool unreachable[RLMESH_DEVICE_CLASSES];
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
  size_t queued;
  rlmesh_device_robot_t* robot;
  rlmesh_device_now_us_fn now_us;
  /* Actions the transport failed to carry out, and events dropped for want
   * of memory or queue room. Under `lock`. */
  uint32_t failed;
  /* The signature task, the view waiting for it (only the newest: a newer
   * view replaces it), the verifier of a new link and whether to stop. All
   * but the task under `lock`. */
  z_owned_task_t verify_task;
  z_owned_condvar_t verify_wake;
  struct rlmesh_zenoh_view* view;
  rlmesh_device_checker_t* next_checker;
  bool stopping;
  rlmesh_zenoh_view_stats_t views;
  /* Called (under `lock`) when an event is queued for the robot. */
  void (*wake)(void*);
  void* wake_arg;
  /* The send task and the publications waiting for it, under `lock`. */
  z_owned_task_t tx_task;
  z_owned_condvar_t tx_wake;
  rlmesh_txq_t txq;
  /* Held around every zenoh-pico call this transport makes on a session,
   * from the loop or the send task: declaring, querying, publishing, opening
   * and closing. `open` changes only under it, and only on the loop. */
  z_owned_mutex_t net;
} rlmesh_zenoh_t;

/* Prepares the transport for `robot`, which stays the caller's: sessions open
 * to the robot's host (rlmesh_device_robot_host) as its actions need them. 0
 * on success. */
int rlmesh_zenoh_open(rlmesh_zenoh_t* z, rlmesh_device_robot_t* robot,
                      rlmesh_device_now_us_fn now_us);

/* Feeds the robot what arrived, then carries out its actions. Call it every
 * loop. 0, or the robot's error code. */
int rlmesh_zenoh_poll(rlmesh_zenoh_t* z);

/* Calls wake(arg), from whichever task queued it, whenever something arrives
 * for the robot (and at once if something already waits). A board's loop
 * sleeps until that or until the robot is next due
 * (rlmesh_device_robot_next_due), so it never spins. wake must not block. */
void rlmesh_zenoh_on_input(rlmesh_zenoh_t* z, void (*wake)(void*), void* arg);

/* What happened to the robot's signed views so far. */
void rlmesh_zenoh_view_stats(rlmesh_zenoh_t* z, rlmesh_zenoh_view_stats_t* out);

/* The transport's tasks' stacks, for the firmware's stack check. */
size_t rlmesh_zenoh_verify_stack(void);
size_t rlmesh_zenoh_tx_stack(void);

/* Closes every session; the robot is left to the caller. */
void rlmesh_zenoh_close(rlmesh_zenoh_t* z);

#ifdef __cplusplus
}
#endif

#endif /* RLMESH_ZENOH_H_ */
