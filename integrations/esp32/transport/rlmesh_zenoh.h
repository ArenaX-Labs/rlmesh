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
 * driven from rlmesh_zenoh_poll, on the caller's thread, which only ever takes
 * this transport's own lock, briefly: it never waits on a socket. Its signed
 * views are checked on a task of their own first: one signature check takes
 * longer than a control period, and the robot only has to take the result.
 * Every call into zenoh-pico (opening, declaring, querying, publishing,
 * closing) is made on another, the network task, in the order the robot asked:
 * any of them can wait for the socket and the radio, and with them the locks
 * zenoh-pico holds while it writes. Both tasks, like zenoh-pico's, run on the
 * core the build sets (RLMESH_TASK_CORE, patch_zenoh_pico.py): by default on
 * either, as the scheduler places them; pinned to core 0 they share it with
 * the radio's driver and lwIP, which a send runs inline. */

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
 * it is dropped (and counted in `failed`). A part's commands, the sealed view,
 * the host's tick and the heartbeat echo keep only their newest sample queued
 * (the rest counted in `superseded`), so this holds what a stalled link
 * releases at once: key deliveries, e-stop signals, a view or two. Replies
 * and query ends always queue: the robot's own queries bound them. */
#define RLMESH_ZENOH_QUEUE_BYTES (6 * 1024)
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

/* The network task: its stack (test/check_stack.py holds its deepest call,
 * opening a session, to it) and FreeRTOS priority, that of zenoh-pico's read
 * tasks. */
#define RLMESH_ZENOH_TX_STACK (8 * 1024)
#define RLMESH_ZENOH_TX_PRIORITY 12
/* Bytes of publications waiting to go out at most, with up to half of it of
 * those the network task is sending (so a stalled send never crowds out the
 * newest state): past it the oldest sample, then the oldest heartbeat, is
 * dropped (counted in rlmesh_zenoh_failures), as with a link that can't keep
 * up. */
#define RLMESH_ZENOH_TX_BYTES (4 * 1024)

/* The datagram path (RLMESH_DEVICE_DATAGRAM_OPEN): one UDP socket for the
 * robot, opened, read and closed by a task of its own, so neither a name
 * lookup nor a read ever waits on the loop. The task and its buffer are made
 * when the robot first opens the socket: a board whose host offers no
 * datagram path never has them. It reads with a timeout, so it
 * notices a close within RLMESH_ZENOH_DGRAM_POLL_MS. Datagrams go out on the
 * network task, without waiting: one the socket can't take now is dropped (a
 * newer one replaces it anyway). What arrives queues for the loop like a
 * sample, within RLMESH_ZENOH_QUEUE_BYTES. */
#define RLMESH_ZENOH_RX_STACK (5 * 1024)
#define RLMESH_ZENOH_RX_PRIORITY 12
#define RLMESH_ZENOH_DGRAM_POLL_MS 100
/* The network task's class for datagrams: past the session classes. */
#define RLMESH_ZENOH_DATAGRAM_CLASS RLMESH_DEVICE_CLASSES
/* Sends refused in a row (not ones that would only block) before the robot
 * is told the path failed. */
#define RLMESH_ZENOH_DGRAM_FAILURES 3

/* A measurement build may define RLMESH_ZENOH_TRACE_CHUNK(msg, len, arrived_us)
 * to see each sealed action chunk as it arrives, on either path, before the
 * robot takes it. It runs on the task that received it, and must not block. */

struct rlmesh_zenoh_event;
struct rlmesh_zenoh_view;
struct rlmesh_zenoh_op;
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
  /* Signed views dropped unchecked: the robot takes sealed ones. */
  uint32_t skipped;
  uint64_t wait_max_us;
  uint64_t wait_total_us;
  /* Publications dropped from a full send queue (of them, samples a newer
   * one on the same key replaced before it went: tx_replaced), and those the
   * network task failed to send (or an action the transport failed to carry
   * out). */
  uint32_t tx_dropped;
  uint32_t tx_replaced;
  uint32_t failed;
  /* Sealed views handed to the robot: the longest and total time from
   * arrival until the robot took it, and of that the robot's own (opening,
   * decoding and applying it). */
  uint32_t sealed;
  uint64_t sealed_wait_max_us;
  uint64_t sealed_wait_total_us;
  uint64_t sealed_open_max_us;
  uint64_t sealed_open_total_us;
  /* The datagram socket: datagrams it sent, those it dropped because the
   * socket was not open or could not take them now, those it received, and
   * those it dropped received (too long, or no queue room). */
  uint32_t dgram_sent;
  uint32_t dgram_unsent;
  uint32_t dgram_received;
  uint32_t dgram_dropped;
  /* Samples replaced in the robot's queue by a newer one on the same
   * subscription before the robot took them (RLMESH_ZENOH_QUEUE_BYTES). */
  uint32_t superseded;
} rlmesh_zenoh_view_stats_t;

/* A subscription's callback context. */
typedef struct rlmesh_zenoh_sub_t {
  struct rlmesh_zenoh_t* z;
  uint8_t sub;
  /* Its subscription was dropped: the slot is free for the next. */
  bool dropped;
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
  /* The robot takes sealed views: signed ones need no check. */
  bool skip_signed;
  /* Called (under `lock`) when an event is queued for the robot. */
  void (*wake)(void*);
  void* wake_arg;
  /* The network task, and what waits for it under `lock`: the robot's other
   * actions in order (`ops`), the publications, and the host they go to. */
  z_owned_task_t tx_task;
  z_owned_condvar_t tx_wake;
  struct rlmesh_zenoh_op* ops;
  struct rlmesh_zenoh_op* ops_tail;
  rlmesh_txq_t txq;
  char target_host[64];
  uint16_t target_port;
  /* The datagram socket's task, what the loop asked of it (open to
   * `dgram_host`:`dgram_port` as request `dgram_want`, 0 closed) and
   * whether a send or an open failed since the loop last looked, all under
   * `lock`. The socket itself (`dgram_fd`, -1 none) is opened and closed only
   * by its task, under `dgram_lock`, which the network task holds to use it. */
  z_owned_task_t rx_task;
  uint32_t dgram_want;
  char dgram_host[64];
  uint16_t dgram_port;
  bool dgram_failed;
  uint8_t dgram_refused;
  int dgram_fd;
  z_owned_mutex_t dgram_lock;
  /* The datagram task runs, reading into `dgram_buf` (its own). */
  bool rx_started;
  /* The datagram task returned (the path closed): joined at the next open. */
  bool rx_ended;
  uint8_t* dgram_buf;
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

/* Samples waiting to be sent that a newer one on the same key replaced, and
 * publications dropped from a full send queue, since the transport opened:
 * on a link slower than the robot publishes, a stream's rate drops here. */
void rlmesh_zenoh_send_counts(rlmesh_zenoh_t* z, uint32_t* replaced, uint32_t* dropped);

/* Tasks zenoh-pico started and has not ended (each holds its stack until it
 * has): the transport's own and one per open session. */
uint32_t rlmesh_zenoh_tasks(void);

/* The transport's tasks' stacks, for the firmware's stack check. */
size_t rlmesh_zenoh_verify_stack(void);
size_t rlmesh_zenoh_tx_stack(void);
size_t rlmesh_zenoh_rx_stack(void);

/* Closes every session; the robot is left to the caller. */
void rlmesh_zenoh_close(rlmesh_zenoh_t* z);

#ifdef __cplusplus
}
#endif

#endif /* RLMESH_ZENOH_H_ */
