#include "rlmesh_zenoh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

size_t rlmesh_zenoh_size(void) { return sizeof(rlmesh_zenoh_t); }

uint32_t rlmesh_zenoh_failures(rlmesh_zenoh_t* z) {
  z_mutex_lock(z_loan_mut(z->lock));
  uint32_t failed = z->failed + z->txq.dropped;
  z_mutex_unlock(z_loan_mut(z->lock));
  return failed;
}

enum { EV_SAMPLE = 1, EV_REPLY = 2, EV_DONE = 3, EV_CHECKED = 4 };

typedef struct rlmesh_zenoh_event {
  struct rlmesh_zenoh_event* next;
  uint8_t kind;
  uint8_t sub;
  uint32_t query_id;
  /* An EV_CHECKED's verified message, and when its view arrived. */
  rlmesh_device_checked_t* checked;
  uint64_t arrived_us;
  /* data holds the key (a sample's), then the payload. */
  size_t key_len;
  size_t len;
  uint8_t data[];
} event_t;

/* A signed view waiting for the signature task: the key, then the payload. */
typedef struct rlmesh_zenoh_view {
  uint64_t arrived_us;
  size_t key_len;
  size_t len;
  uint8_t data[];
} view_t;

/* A query's closure context: freed by its dropper. */
typedef struct {
  rlmesh_zenoh_t* z;
  uint32_t id;
} query_t;

static void count_failed(rlmesh_zenoh_t* z) {
  z_mutex_lock(z_loan_mut(z->lock));
  z->failed++;
  z_mutex_unlock(z_loan_mut(z->lock));
}

/* Queues a copy of `key` (a sample's, or NULL) and `payload`, from
 * zenoh-pico's read task. Samples arrive unverified, so they queue only up to
 * RLMESH_ZENOH_QUEUE_BYTES. */
static void push(rlmesh_zenoh_t* z, uint8_t kind, uint8_t sub, uint32_t query_id,
                 const z_loaned_keyexpr_t* key, const z_loaned_bytes_t* payload) {
  z_owned_slice_t slice;
  const uint8_t* data = NULL;
  size_t len = 0;
  bool have = payload != NULL && z_bytes_to_slice(payload, &slice) == Z_OK;
  if (have) {
    data = z_slice_data(z_loan(slice));
    len = z_slice_len(z_loan(slice));
  }
  z_view_string_t key_text;
  const char* key_data = NULL;
  size_t key_len = 0;
  if (key != NULL) {
    z_keyexpr_as_view_string(key, &key_text);
    key_data = z_string_data(z_loan(key_text));
    key_len = z_string_len(z_loan(key_text));
  }
  size_t size = sizeof(event_t) + key_len + len;
  bool room = true;
  if (kind == EV_SAMPLE) {
    z_mutex_lock(z_loan_mut(z->lock));
    room = z->queued + size <= RLMESH_ZENOH_QUEUE_BYTES;
    z_mutex_unlock(z_loan_mut(z->lock));
  }
  event_t* e = room ? (event_t*)malloc(size) : NULL;
  if (e != NULL) {
    e->next = NULL;
    e->kind = kind;
    e->sub = sub;
    e->query_id = query_id;
    e->checked = NULL;
    e->arrived_us = z->now_us();
    e->key_len = key_len;
    e->len = len;
    if (key_len > 0) memcpy(e->data, key_data, key_len);
    if (len > 0) memcpy(e->data + key_len, data, len);
  }
  if (have) z_drop(z_move(slice));
  z_mutex_lock(z_loan_mut(z->lock));
  if (e == NULL) {
    z->failed++;
  } else {
    if (kind == EV_SAMPLE) z->queued += size;
    if (z->tail == NULL) {
      z->head = z->tail = e;
      if (z->wake != NULL) z->wake(z->wake_arg);
    } else {
      z->tail->next = e;
      z->tail = e;
    }
  }
  z_mutex_unlock(z_loan_mut(z->lock));
}

/* Queues a verified message for the robot, from the signature task. */
static void push_checked(rlmesh_zenoh_t* z, uint8_t sub, rlmesh_device_checked_t* checked,
                         uint64_t arrived_us, size_t len) {
  /* Counted against the samples' bound as its bytes, which it holds. */
  size_t size = sizeof(event_t) + len;
  z_mutex_lock(z_loan_mut(z->lock));
  bool room = z->queued + size <= RLMESH_ZENOH_QUEUE_BYTES;
  z_mutex_unlock(z_loan_mut(z->lock));
  event_t* e = room ? (event_t*)malloc(sizeof(event_t)) : NULL;
  if (e == NULL) {
    rlmesh_device_checked_free(checked);
    count_failed(z);
    return;
  }
  memset(e, 0, sizeof *e);
  e->kind = EV_CHECKED;
  e->sub = sub;
  e->checked = checked;
  e->arrived_us = arrived_us;
  z_mutex_lock(z_loan_mut(z->lock));
  z->queued += size;
  if (z->tail == NULL) {
    z->head = z->tail = e;
    if (z->wake != NULL) z->wake(z->wake_arg);
  } else {
    z->tail->next = e;
    z->tail = e;
  }
  z_mutex_unlock(z_loan_mut(z->lock));
}

/* Hands the signature task a copy of a signed view, in place of any it hasn't
 * started on: only the newest counts. */
static void offer_view(rlmesh_zenoh_t* z, const z_loaned_sample_t* sample) {
  z_owned_slice_t slice;
  if (z_bytes_to_slice(z_sample_payload(sample), &slice) != Z_OK) return;
  z_view_string_t key;
  z_keyexpr_as_view_string(z_sample_keyexpr(sample), &key);
  size_t key_len = z_string_len(z_loan(key));
  size_t len = z_slice_len(z_loan(slice));
  view_t* v = (view_t*)malloc(sizeof(view_t) + key_len + len);
  if (v == NULL) {
    z_drop(z_move(slice));
    count_failed(z);
    return;
  }
  v->arrived_us = z->now_us();
  v->key_len = key_len;
  v->len = len;
  memcpy(v->data, z_string_data(z_loan(key)), key_len);
  memcpy(v->data + key_len, z_slice_data(z_loan(slice)), len);
  z_drop(z_move(slice));
  z_mutex_lock(z_loan_mut(z->lock));
  view_t* replaced = z->view;
  z->view = v;
  z->views.offered++;
  if (replaced != NULL) z->views.replaced++;
  z_condvar_signal(z_loan_mut(z->verify_wake));
  z_mutex_unlock(z_loan_mut(z->lock));
  free(replaced);
}

static void on_sample(z_loaned_sample_t* sample, void* ctx) {
  rlmesh_zenoh_sub_t* s = (rlmesh_zenoh_sub_t*)ctx;
  if (s->sub == RLMESH_DEVICE_SUB_VIEW) {
    /* Once the robot takes sealed views, a signed one is dropped unchecked. */
    z_mutex_lock(z_loan_mut(s->z->lock));
    bool skip = s->z->skip_signed;
    if (skip) s->z->views.skipped++;
    z_mutex_unlock(z_loan_mut(s->z->lock));
    if (!skip) offer_view(s->z, sample);
    return;
  }
  push(s->z, EV_SAMPLE, s->sub, 0, z_sample_keyexpr(sample), z_sample_payload(sample));
}

/* Checks each signed view, so the robot's task never spends a control period
 * on one. Only this task touches the checker in use. */
static void* verify_task(void* arg) {
  rlmesh_zenoh_t* z = (rlmesh_zenoh_t*)arg;
  rlmesh_device_checker_t* checker = NULL;
  z_mutex_lock(z_loan_mut(z->lock));
  for (;;) {
    while (!z->stopping && z->view == NULL) {
      z_condvar_wait(z_loan_mut(z->verify_wake), z_loan_mut(z->lock));
    }
    if (z->stopping) break;
    if (z->next_checker != NULL) {
      rlmesh_device_checker_free(checker);
      checker = z->next_checker;
      z->next_checker = NULL;
    }
    view_t* v = z->view;
    z->view = NULL;
    z_mutex_unlock(z_loan_mut(z->lock));
    /* Before the robot's first link there is nothing to check against, and
     * the robot would drop the view anyway. */
    rlmesh_device_checked_t* checked =
        checker == NULL ? NULL
                        : rlmesh_device_checker_check(checker, (const char*)v->data, v->key_len,
                                                      v->data + v->key_len, v->len);
    uint64_t arrived_us = v->arrived_us;
    size_t len = v->key_len + v->len;
    free(v);
    if (checked != NULL) push_checked(z, RLMESH_DEVICE_SUB_VIEW, checked, arrived_us, len);
    /* The idle task's turn, however fast views come. */
    z_sleep_ms(1);
    z_mutex_lock(z_loan_mut(z->lock));
    if (checked == NULL) z->views.refused++;
  }
  z_mutex_unlock(z_loan_mut(z->lock));
  rlmesh_device_checker_free(checker);
  return NULL;
}

__attribute__((noipa)) size_t rlmesh_zenoh_verify_stack(void) { return RLMESH_ZENOH_VERIFY_STACK; }

void rlmesh_zenoh_on_input(rlmesh_zenoh_t* z, void (*wake)(void*), void* arg) {
  z_mutex_lock(z_loan_mut(z->lock));
  z->wake = wake;
  z->wake_arg = arg;
  bool pending = z->head != NULL;
  z_mutex_unlock(z_loan_mut(z->lock));
  if (pending && wake != NULL) wake(arg);
}

void rlmesh_zenoh_view_stats(rlmesh_zenoh_t* z, rlmesh_zenoh_view_stats_t* out) {
  z_mutex_lock(z_loan_mut(z->lock));
  *out = z->views;
  out->tx_dropped = z->txq.dropped;
  out->failed = z->failed;
  z_mutex_unlock(z_loan_mut(z->lock));
}

static void on_reply(z_loaned_reply_t* reply, void* ctx) {
  query_t* q = (query_t*)ctx;
  if (z_reply_is_ok(reply)) {
    push(q->z, EV_REPLY, 0, q->id, NULL, z_sample_payload(z_reply_ok(reply)));
  }
}

static void on_query_done(void* ctx) {
  query_t* q = (query_t*)ctx;
  push(q->z, EV_DONE, 0, q->id, NULL, NULL);
  free(q);
}

static void undeclare(rlmesh_zenoh_t* z);
static void close_sessions(rlmesh_zenoh_t* z);

/* The session for class `c`, opened to the robot's host on its port plus `c`
 * the first time it's needed; NULL when it can't be. The open sessions all
 * reach one host: opening one to another closes them, and drops what was
 * declared on them (the robot's Reset for the move follows). Under `net`, as
 * is every call below that uses a session. */
static const z_loaned_session_t* session(rlmesh_zenoh_t* z, uint8_t c) {
  if (c >= RLMESH_DEVICE_CLASSES) return NULL;
  if (!z->open[c]) {
    if (z->unreachable[c]) return NULL;
    uint16_t port = 0;
    const char* host = rlmesh_device_robot_host(z->robot, &port);
    char connect[96];
    if (host == NULL || strlen(host) >= sizeof z->host) return NULL;
    int n = snprintf(connect, sizeof connect, "tcp/%s:%u#tout=" RLMESH_ZENOH_RECV_TIMEOUT_MS, host,
                     (unsigned)(port + c));
    if (n <= 0 || (size_t)n >= sizeof connect) return NULL;
    if (port != z->port || strcmp(host, z->host) != 0) {
      undeclare(z);
      close_sessions(z);
    }
    z_owned_config_t config;
    z_config_default(&config);
    zp_config_insert(z_loan_mut(config), Z_CONFIG_MODE_KEY, "client");
    zp_config_insert(z_loan_mut(config), Z_CONFIG_CONNECT_KEY, connect);
    if (z_open(&z->sessions[c], z_move(config), NULL) != Z_OK) {
      z->unreachable[c] = true;
      return NULL;
    }
    z->open[c] = true;
    strcpy(z->host, host);
    z->port = port;
  }
  return z_loan(z->sessions[c]);
}

static int subscribe(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  const z_loaned_session_t* s = session(z, a->class_);
  z_view_keyexpr_t ke;
  if (s == NULL || z->sub_count >= RLMESH_ZENOH_SUBS ||
      z_view_keyexpr_from_str(&ke, a->key) != Z_OK) {
    return -1;
  }
  rlmesh_zenoh_sub_t* ctx = &z->sub_ctx[z->sub_count];
  ctx->z = z;
  ctx->sub = a->sub;
  z_owned_closure_sample_t callback;
  z_closure(&callback, on_sample, NULL, ctx);
  if (z_declare_subscriber(s, &z->subs[z->sub_count], z_loan(ke), z_move(callback), NULL) != Z_OK) {
    return -1;
  }
  z->sub_count++;
  return 0;
}

static int token(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  const z_loaned_session_t* s = session(z, a->class_);
  z_view_keyexpr_t ke;
  if (s == NULL || z->token_count >= RLMESH_ZENOH_TOKENS ||
      z_view_keyexpr_from_str(&ke, a->key) != Z_OK) {
    return -1;
  }
  if (z_liveliness_declare_token(s, &z->tokens[z->token_count], z_loan(ke), NULL) != Z_OK) {
    return -1;
  }
  z->token_count++;
  return 0;
}

static int query(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  const z_loaned_session_t* s = session(z, a->class_);
  z_view_keyexpr_t ke;
  query_t* q = NULL;
  if (s == NULL || z_view_keyexpr_from_str(&ke, a->key) != Z_OK ||
      (q = (query_t*)malloc(sizeof(query_t))) == NULL) {
    /* The robot waits on every query it asked for: it's over. */
    push(z, EV_DONE, 0, a->query_id, NULL, NULL);
    return -1;
  }
  q->z = z;
  q->id = a->query_id;
  z_get_options_t opts;
  z_get_options_default(&opts);
  opts.consolidation = z_query_consolidation_none();
  opts.timeout_ms = a->timeout_ms;
  opts.priority = Z_PRIORITY_REAL_TIME;
  opts.is_express = true;
  z_owned_bytes_t payload;
  if (a->has_payload) {
    if (z_bytes_copy_from_buf(&payload, a->payload, a->payload_len) != Z_OK) {
      free(q);
      push(z, EV_DONE, 0, a->query_id, NULL, NULL);
      return -1;
    }
    opts.payload = z_move(payload);
  }
  z_owned_closure_reply_t callback;
  z_closure(&callback, on_reply, on_query_done, q);
  /* On failure the closure is dropped, which frees q and reports the query
   * done: the robot retries. */
  return z_get(s, z_loan(ke), "", z_move(callback), &opts) == Z_OK ? 0 : -1;
}

/* Queues a publication for the send task: sending it takes the socket and the
 * radio's path, which a control loop can't wait for. Opening its session, on
 * first use, is the one part done here. */
static int publish(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  if (a->class_ >= RLMESH_DEVICE_CLASSES) return -1;
  if (!z->open[a->class_]) {
    z_mutex_lock(z_loan_mut(z->net));
    const z_loaned_session_t* s = session(z, a->class_);
    z_mutex_unlock(z_loan_mut(z->net));
    if (s == NULL) return -1;
  }
  rlmesh_txq_item_t* t = rlmesh_txq_item(a->class_, a->key, a->payload, a->payload_len);
  if (t == NULL) return -1;
  z_mutex_lock(z_loan_mut(z->lock));
  bool queued = rlmesh_txq_push(&z->txq, t);
  z_mutex_unlock(z_loan_mut(z->lock));
  /* rlmesh_zenoh_poll wakes the send task once it has queued them all. */
  return queued ? 0 : -1;
}

/* Sends one publication on its session (open, under `net`): express when it
 * goes alone, batched with the others otherwise. */
static int send_one(rlmesh_zenoh_t* z, const rlmesh_txq_item_t* t, bool express) {
  z_view_keyexpr_t ke;
  z_owned_bytes_t payload;
  if (z_view_keyexpr_from_str(&ke, (const char*)t->data) != Z_OK ||
      z_bytes_copy_from_buf(&payload, t->data + t->key_len, t->len) != Z_OK) {
    return -1;
  }
  z_put_options_t opts;
  z_put_options_default(&opts);
  opts.priority = Z_PRIORITY_REAL_TIME;
  opts.is_express = express;
  /* This task is free to wait for the session's send path, which the read
   * task's keep-alives and the loop's queries share: the default drops a
   * publication (a heartbeat, say) whenever that path is busy. */
  opts.congestion_control = Z_CONGESTION_CONTROL_BLOCK;
  return z_put(z_loan(z->sessions[t->class_]), z_loan(ke), z_move(payload), &opts) == Z_OK ? 0 : -1;
}

/* Sends what waited, in order within each class. Several waiting on one
 * session go out as one batch, one write instead of a segment each: on a
 * slow link that is what lets the queue drain. */
static void send_all(rlmesh_zenoh_t* z, rlmesh_txq_item_t* list) {
  z_mutex_lock(z_loan_mut(z->net));
  for (uint8_t c = 0; c < RLMESH_DEVICE_CLASSES; ++c) {
    size_t n = 0;
    for (rlmesh_txq_item_t* t = list; t != NULL; t = t->next) n += t->class_ == c;
    if (n == 0) continue;
    if (!z->open[c]) {
      for (size_t i = 0; i < n; ++i) count_failed(z);
      continue;
    }
    const z_loaned_session_t* s = z_loan(z->sessions[c]);
    bool batch = n > 1 && zp_batch_start(s) == Z_OK;
    for (rlmesh_txq_item_t* t = list; t != NULL; t = t->next) {
      if (t->class_ == c && send_one(z, t, !batch) != 0) count_failed(z);
    }
    if (batch) zp_batch_stop(s);
  }
  z_mutex_unlock(z_loan_mut(z->net));
}

static void* tx_task(void* arg) {
  rlmesh_zenoh_t* z = (rlmesh_zenoh_t*)arg;
  z_mutex_lock(z_loan_mut(z->lock));
  for (;;) {
    while (!z->stopping && z->txq.head == NULL) {
      z_condvar_wait(z_loan_mut(z->tx_wake), z_loan_mut(z->lock));
    }
    if (z->stopping) break;
    /* Everything waiting, at once. */
    rlmesh_txq_item_t* list = z->txq.head;
    z->txq.head = z->txq.tail = NULL;
    z->txq.bytes = 0;
    z_mutex_unlock(z_loan_mut(z->lock));
    send_all(z, list);
    while (list != NULL) {
      rlmesh_txq_item_t* next = list->next;
      free(list);
      list = next;
    }
    z_mutex_lock(z_loan_mut(z->lock));
  }
  z_mutex_unlock(z_loan_mut(z->lock));
  return NULL;
}

__attribute__((noipa)) size_t rlmesh_zenoh_tx_stack(void) { return RLMESH_ZENOH_TX_STACK; }

/* Drops what the send task hasn't sent. */
static void drop_unsent(rlmesh_zenoh_t* z) {
  z_mutex_lock(z_loan_mut(z->lock));
  rlmesh_txq_clear(&z->txq);
  z_mutex_unlock(z_loan_mut(z->lock));
}

/* Closes every class's session. Pending queries are dropped with them:
 * their droppers queue. A publication in flight finishes first, and what
 * waits for the send task would go to the wrong session. */
static void close_sessions(rlmesh_zenoh_t* z) {
  drop_unsent(z);
  for (uint8_t c = 0; c < RLMESH_DEVICE_CLASSES; ++c) {
    if (z->open[c]) z_drop(z_move(z->sessions[c]));
    z->open[c] = false;
  }
}

/* Drops every subscription and token. */
static void undeclare(rlmesh_zenoh_t* z) {
  for (uint8_t i = 0; i < z->sub_count; ++i) z_drop(z_move(z->subs[i]));
  for (uint8_t i = 0; i < z->token_count; ++i) z_drop(z_move(z->tokens[i]));
  z->sub_count = 0;
  z->token_count = 0;
}

/* Drops every subscription and token: the robot moved to another session.
 * The sessions stay for the same host, and close for another (a run on a
 * satellite, another home address). */
static void reset(rlmesh_zenoh_t* z) {
  undeclare(z);
  memset(z->unreachable, 0, sizeof z->unreachable);
  uint16_t port = 0;
  const char* host = rlmesh_device_robot_host(z->robot, &port);
  if (host == NULL || port != z->port || strcmp(host, z->host) != 0) close_sessions(z);
}

/* Drops everything: replies stopped, and a session whose peer is gone never
 * comes back on its own. The robot's next actions open new ones. */
static void reconnect(rlmesh_zenoh_t* z) {
  undeclare(z);
  close_sessions(z);
  memset(z->unreachable, 0, sizeof z->unreachable);
}

int rlmesh_zenoh_open(rlmesh_zenoh_t* z, rlmesh_device_robot_t* robot,
                      rlmesh_device_now_us_fn now_us) {
  if (z == NULL || robot == NULL || now_us == NULL) return -1;
  memset(z, 0, sizeof *z);
  z->txq.cap = RLMESH_ZENOH_TX_BYTES;
  z->robot = robot;
  z->now_us = now_us;
  if (z_mutex_init(&z->lock) != Z_OK) return -1;
  if (z_mutex_init(&z->net) != Z_OK) {
    z_drop(z_move(z->lock));
    return -1;
  }
  if (z_condvar_init(&z->verify_wake) != Z_OK || z_condvar_init(&z->tx_wake) != Z_OK) {
    z_drop(z_move(z->net));
    z_drop(z_move(z->lock));
    return -1;
  }
#if defined(ZENOH_ARDUINO_ESP32)
  z_task_attr_t verify_attrs = {.name = "rlmesh-verify",
                                .priority = RLMESH_ZENOH_VERIFY_PRIORITY,
                                .stack_depth = rlmesh_zenoh_verify_stack()};
  z_task_attr_t tx_attrs = {.name = "rlmesh-tx",
                            .priority = RLMESH_ZENOH_TX_PRIORITY,
                            .stack_depth = rlmesh_zenoh_tx_stack()};
  z_task_attr_t* verify_attr = &verify_attrs;
  z_task_attr_t* tx_attr = &tx_attrs;
#else
  z_task_attr_t* verify_attr = NULL;
  z_task_attr_t* tx_attr = NULL;
#endif
  if (z_task_init(&z->verify_task, verify_attr, verify_task, z) != Z_OK) {
    z_drop(z_move(z->tx_wake));
    z_drop(z_move(z->verify_wake));
    z_drop(z_move(z->net));
    z_drop(z_move(z->lock));
    return -1;
  }
  if (z_task_init(&z->tx_task, tx_attr, tx_task, z) != Z_OK) {
    z_mutex_lock(z_loan_mut(z->lock));
    z->stopping = true;
    z_condvar_signal(z_loan_mut(z->verify_wake));
    z_mutex_unlock(z_loan_mut(z->lock));
    z_task_join(z_move(z->verify_task));
    z_drop(z_move(z->tx_wake));
    z_drop(z_move(z->verify_wake));
    z_drop(z_move(z->net));
    z_drop(z_move(z->lock));
    return -1;
  }
  return 0;
}

int rlmesh_zenoh_poll(rlmesh_zenoh_t* z) {
  rlmesh_device_checker_t* checker = rlmesh_device_robot_take_checker(z->robot);
  bool skip_signed = rlmesh_device_robot_wants_signed_views(z->robot) == 0;
  z_mutex_lock(z_loan_mut(z->lock));
  z->skip_signed = skip_signed;
  rlmesh_device_checker_t* unused = NULL;
  if (checker != NULL) {
    unused = z->next_checker;
    z->next_checker = checker;
  }
  event_t* e = z->head;
  z->head = z->tail = NULL;
  z->queued = 0;
  z_mutex_unlock(z_loan_mut(z->lock));
  rlmesh_device_checker_free(unused);
  while (e != NULL) {
    uint64_t now = z->now_us();
    switch (e->kind) {
      case EV_SAMPLE:
        rlmesh_device_robot_sample(z->robot, e->sub, (const char*)e->data, e->key_len,
                                   e->data + e->key_len, e->len, now);
        if (e->sub == RLMESH_DEVICE_SUB_VIEW_SEALED) {
          uint64_t done = z->now_us();
          uint64_t waited = done - e->arrived_us;
          uint64_t took = done - now;
          z_mutex_lock(z_loan_mut(z->lock));
          z->views.sealed++;
          z->views.sealed_wait_total_us += waited;
          if (waited > z->views.sealed_wait_max_us) z->views.sealed_wait_max_us = waited;
          z->views.sealed_open_total_us += took;
          if (took > z->views.sealed_open_max_us) z->views.sealed_open_max_us = took;
          z_mutex_unlock(z_loan_mut(z->lock));
        }
        break;
      case EV_CHECKED: {
        rlmesh_device_robot_sample_checked(z->robot, e->sub, e->checked, now);
        uint64_t waited = z->now_us() - e->arrived_us;
        z_mutex_lock(z_loan_mut(z->lock));
        z->views.taken++;
        z->views.wait_total_us += waited;
        if (waited > z->views.wait_max_us) z->views.wait_max_us = waited;
        z_mutex_unlock(z_loan_mut(z->lock));
        break;
      }
      case EV_REPLY:
        rlmesh_device_robot_reply(z->robot, e->query_id, e->data, e->len, now);
        break;
      default:
        rlmesh_device_robot_query_done(z->robot, e->query_id, now);
        break;
    }
    event_t* next = e->next;
    free(e);
    e = next;
  }
  rlmesh_device_action_t a;
  int rc;
  while ((rc = rlmesh_device_robot_next(z->robot, z->now_us(), &a)) == 1) {
    if (a.kind == RLMESH_DEVICE_PUBLISH) {
      if (publish(z, &a) != 0) count_failed(z);
      continue;
    }
    /* The send task uses the sessions too: one zenoh-pico call at a time. */
    z_mutex_lock(z_loan_mut(z->net));
    int done = -1;
    switch (a.kind) {
      case RLMESH_DEVICE_SUBSCRIBE:
        done = subscribe(z, &a);
        break;
      case RLMESH_DEVICE_TOKEN:
        done = token(z, &a);
        break;
      case RLMESH_DEVICE_QUERY:
        done = query(z, &a);
        break;
      case RLMESH_DEVICE_RESET:
        reset(z);
        done = 0;
        break;
      case RLMESH_DEVICE_RECONNECT:
        reconnect(z);
        done = 0;
        break;
      default:
        break;
    }
    z_mutex_unlock(z_loan_mut(z->net));
    if (done != 0) count_failed(z);
  }
  /* One wake for everything this poll queued: a tick's samples leave in one
   * write. */
  z_mutex_lock(z_loan_mut(z->lock));
  if (z->txq.head != NULL) z_condvar_signal(z_loan_mut(z->tx_wake));
  z_mutex_unlock(z_loan_mut(z->lock));
  return rc < 0 ? rc : 0;
}

void rlmesh_zenoh_close(rlmesh_zenoh_t* z) {
  z_mutex_lock(z_loan_mut(z->net));
  undeclare(z);
  close_sessions(z);
  z_mutex_unlock(z_loan_mut(z->net));
  z_mutex_lock(z_loan_mut(z->lock));
  z->stopping = true;
  z_condvar_signal(z_loan_mut(z->verify_wake));
  z_condvar_signal(z_loan_mut(z->tx_wake));
  z_mutex_unlock(z_loan_mut(z->lock));
  z_task_join(z_move(z->verify_task));
  z_task_join(z_move(z->tx_task));
  drop_unsent(z);
  free(z->view);
  z->view = NULL;
  rlmesh_device_checker_free(z->next_checker);
  z->next_checker = NULL;
  for (event_t* e = z->head; e != NULL;) {
    event_t* next = e->next;
    rlmesh_device_checked_free(e->checked);
    free(e);
    e = next;
  }
  z->head = z->tail = NULL;
  z_drop(z_move(z->tx_wake));
  z_drop(z_move(z->verify_wake));
  z_drop(z_move(z->net));
  z_drop(z_move(z->lock));
}
