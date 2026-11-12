#include "rlmesh_zenoh.h"

#include <errno.h>
#include <math.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

size_t rlmesh_zenoh_size(void) { return sizeof(rlmesh_zenoh_t); }

/* Counted by zenoh-pico's ports, as patched (patch_zenoh_pico.py). */
void zp_rlmesh_tasks(uint32_t* started, uint32_t* ended);

uint32_t rlmesh_zenoh_tasks(void) {
  uint32_t started = 0, ended = 0;
  zp_rlmesh_tasks(&started, &ended);
  return started - ended;
}

void rlmesh_zenoh_send_counts(rlmesh_zenoh_t* z, uint32_t* replaced, uint32_t* dropped) {
  z_mutex_lock(z_loan_mut(z->lock));
  *replaced = z->txq.replaced;
  *dropped = z->txq.dropped - z->txq.replaced;
  z_mutex_unlock(z_loan_mut(z->lock));
}

uint32_t rlmesh_zenoh_failures(rlmesh_zenoh_t* z) {
  z_mutex_lock(z_loan_mut(z->lock));
  uint32_t failed = z->failed + z->txq.dropped;
  z_mutex_unlock(z_loan_mut(z->lock));
  return failed;
}

enum { EV_SAMPLE = 1, EV_REPLY = 2, EV_DONE = 3, EV_CHECKED = 4, EV_DATAGRAM = 5 };

typedef struct rlmesh_zenoh_event {
  struct rlmesh_zenoh_event* next;
  uint8_t kind;
  uint8_t sub;
  uint32_t query_id;
  /* An EV_DATAGRAM that carries a chunk: its route, and its statistics
   * sequence. */
  bool chunk;
  uint16_t route;
  uint32_t seq;
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

/* The usual Wi-Fi beacon interval, 100 TU, in microseconds. */
#define RLMESH_ZENOH_BEACON_US 102400u

/* Adds a time to `p`. */
static void phase_add(rlmesh_zenoh_phase_t* p, uint64_t at_us) {
  float angle = (float)(at_us % RLMESH_ZENOH_BEACON_US) * (6.2831853f / RLMESH_ZENOH_BEACON_US);
  p->c += cosf(angle);
  p->s += sinf(angle);
  p->n++;
}

/* How closely `p`'s times line up (0 to 1), and starts it anew. */
static float phase_take(rlmesh_zenoh_phase_t* p, uint32_t* n) {
  *n = p->n;
  float lock = p->n > 0 ? sqrtf(p->c * p->c + p->s * p->s) / (float)p->n : 0.0f;
  memset(p, 0, sizeof *p);
  return lock;
}

void rlmesh_zenoh_beacon_lock(rlmesh_zenoh_t* z, float* sends, uint32_t* sent, float* receives,
                              uint32_t* received) {
  z_mutex_lock(z_loan_mut(z->lock));
  *sends = phase_take(&z->sends, sent);
  *receives = phase_take(&z->receives, received);
  z_mutex_unlock(z_loan_mut(z->lock));
}

static void count_failed(rlmesh_zenoh_t* z) {
  z_mutex_lock(z_loan_mut(z->lock));
  z->failed++;
  z_mutex_unlock(z_loan_mut(z->lock));
}

/* Whether only the newest sample of subscription `sub` matters: a part's
 * commands (a newer chunk replaces what an older one still had queued), the
 * sealed view, the host's tick and the heartbeat echo (rlmesh-device's
 * SUB_TICK and SUB_ECHO). Key deliveries and e-stop signals all count. */
static bool latest_wins(uint8_t sub) {
  return sub == 1 || sub == 2 || sub >= RLMESH_DEVICE_SUB_COMMANDS;
}

/* Unlinks and returns the queued sample of `sub`, if any (under `lock`). */
static event_t* unlink_sample(rlmesh_zenoh_t* z, uint8_t sub) {
  for (event_t *prev = NULL, *e = z->head; e != NULL; prev = e, e = e->next) {
    if (e->kind != EV_SAMPLE || e->sub != sub) continue;
    if (prev == NULL) {
      z->head = e->next;
    } else {
      prev->next = e->next;
    }
    if (z->tail == e) z->tail = prev;
    z->queued -= sizeof(event_t) + e->key_len + e->len;
    return e;
  }
  return NULL;
}

/* Queues a copy of `key` (a sample's, or NULL) and `payload`, from
 * zenoh-pico's read task. Samples arrive unverified, so they queue only up to
 * RLMESH_ZENOH_QUEUE_BYTES, and a subscription whose newest sample is all that
 * matters keeps only that: a link that stalls, then delivers what it held at
 * once, queues one chunk per part, not a second's worth. */
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
  bool room = kind != EV_SAMPLE || size <= RLMESH_ZENOH_QUEUE_BYTES;
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
#ifdef RLMESH_ZENOH_TRACE_CHUNK
    if (kind == EV_SAMPLE && sub >= RLMESH_DEVICE_SUB_COMMANDS &&
        sub != RLMESH_DEVICE_SUB_VIEW_SEALED) {
      RLMESH_ZENOH_TRACE_CHUNK(e->data + key_len, len, e->arrived_us);
    }
#endif
  }
  if (have) z_drop(z_move(slice));
  z_mutex_lock(z_loan_mut(z->lock));
  event_t* superseded = NULL;
  if (e != NULL && kind == EV_SAMPLE) {
    if (latest_wins(sub)) superseded = unlink_sample(z, sub);
    if (superseded != NULL) z->views.superseded++;
    if (z->queued + size > RLMESH_ZENOH_QUEUE_BYTES) {
      free(e);
      e = NULL;
    }
  }
  if (e == NULL) {
    z->failed++;
  } else {
    if (kind == EV_SAMPLE) {
      z->queued += size;
      phase_add(&z->receives, e->arrived_us);
    }
    if (z->tail == NULL) {
      z->head = z->tail = e;
      if (z->wake != NULL) z->wake(z->wake_arg);
    } else {
      z->tail->next = e;
      z->tail = e;
    }
  }
  z_mutex_unlock(z_loan_mut(z->lock));
  free(superseded);
}

/* Unlinks and returns the queued chunk datagram of `route`, if any (under
 * `lock`). */
static event_t* unlink_chunk(rlmesh_zenoh_t* z, uint16_t route) {
  for (event_t *prev = NULL, *e = z->head; e != NULL; prev = e, e = e->next) {
    if (e->kind != EV_DATAGRAM || !e->chunk || e->route != route) continue;
    if (prev == NULL) {
      z->head = e->next;
    } else {
      prev->next = e->next;
    }
    if (z->tail == e) z->tail = prev;
    z->queued -= sizeof(event_t) + e->len;
    return e;
  }
  return NULL;
}

/* Queues a datagram for the robot, from the datagram task, within the
 * samples' bound. A chunk replaces one of its route still waiting, as a
 * part's commands do on the session: when the loop is held up (a sketch
 * printing its status, say), the robot takes the newest plan, not each one
 * that arrived meanwhile, late. A chunk older than the one waiting (the
 * network reordered them) goes instead. */
static void push_datagram(rlmesh_zenoh_t* z, const uint8_t* data, size_t len) {
  size_t size = sizeof(event_t) + len;
  z_mutex_lock(z_loan_mut(z->lock));
  bool room = z->queued + size <= RLMESH_ZENOH_QUEUE_BYTES;
  z_mutex_unlock(z_loan_mut(z->lock));
  event_t* e = room ? (event_t*)malloc(size) : NULL;
  if (e != NULL) {
    memset(e, 0, sizeof *e);
    e->kind = EV_DATAGRAM;
    e->arrived_us = z->now_us();
    e->len = len;
    e->chunk = rlmesh_device_datagram_lane(data, len, &e->route, &e->seq) == 1;
    memcpy(e->data, data, len);
  }
  z_mutex_lock(z_loan_mut(z->lock));
  event_t* superseded = e != NULL && e->chunk ? unlink_chunk(z, e->route) : NULL;
  if (superseded != NULL && (int32_t)(e->seq - superseded->seq) < 0) {
    /* The one waiting is the newer: it stays. */
    event_t* older = e;
    e = superseded;
    superseded = older;
    z->queued += sizeof(event_t) + e->len;
    if (z->tail == NULL) {
      z->head = z->tail = e;
    } else {
      z->tail->next = e;
      z->tail = e;
    }
    e->next = NULL;
    z->views.dgram_superseded++;
    z_mutex_unlock(z_loan_mut(z->lock));
    free(superseded);
    return;
  }
  if (superseded != NULL) z->views.dgram_superseded++;
  if (e == NULL) {
    z->views.dgram_dropped++;
  } else {
    z->views.dgram_received++;
    z->queued += size;
    phase_add(&z->receives, e->arrived_us);
    if (z->tail == NULL) {
      z->head = z->tail = e;
      if (z->wake != NULL) z->wake(z->wake_arg);
    } else {
      z->tail->next = e;
      z->tail = e;
    }
  }
  z_mutex_unlock(z_loan_mut(z->lock));
  free(superseded);
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
  /* One in flight as its subscription was dropped. */
  if (s->dropped) return;
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
  out->tx_replaced = z->txq.replaced;
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

/* One of the robot's actions for the network task, copied: its fields, the
 * host the robot meant, and its key (NUL-terminated), then its payload. */
typedef struct rlmesh_zenoh_op {
  struct rlmesh_zenoh_op* next;
  uint8_t kind;
  uint8_t class_;
  uint8_t sub;
  uint8_t has_payload;
  uint32_t query_id;
  uint32_t timeout_ms;
  char host[64];
  uint16_t port;
  size_t key_len;
  size_t len;
  uint8_t data[];
} op_t;

/* The session for class `c`, opened to `host` on `port` plus `c` the first
 * time it's needed; NULL when it can't be. The open sessions all reach one
 * host: opening one to another closes them, and drops what was declared on
 * them (the robot's Reset for the move follows). On the network task, as is
 * every call below that uses a session. */
static const z_loaned_session_t* session(rlmesh_zenoh_t* z, uint8_t c, const char* host,
                                         uint16_t port) {
  if (c >= RLMESH_DEVICE_CLASSES) return NULL;
  if (!z->open[c]) {
    if (z->unreachable[c]) return NULL;
    char connect[96];
    if (host[0] == '\0' || strlen(host) >= sizeof z->host) return NULL;
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

static int subscribe(rlmesh_zenoh_t* z, const op_t* a) {
  const z_loaned_session_t* s = session(z, a->class_, a->host, a->port);
  z_view_keyexpr_t ke;
  uint8_t i = 0;
  while (i < z->sub_count && !z->sub_ctx[i].dropped) ++i;
  if (s == NULL || i >= RLMESH_ZENOH_SUBS ||
      z_view_keyexpr_from_str(&ke, (const char*)a->data) != Z_OK) {
    return -1;
  }
  rlmesh_zenoh_sub_t* ctx = &z->sub_ctx[i];
  ctx->z = z;
  ctx->sub = a->sub;
  ctx->dropped = false;
  z_owned_closure_sample_t callback;
  z_closure(&callback, on_sample, NULL, ctx);
  if (z_declare_subscriber(s, &z->subs[i], z_loan(ke), z_move(callback), NULL) != Z_OK) {
    ctx->dropped = true;
    return -1;
  }
  if (i == z->sub_count) z->sub_count++;
  return 0;
}

/* Drops subscription `sub`: its slot is free for the next. */
static int unsubscribe(rlmesh_zenoh_t* z, const op_t* a) {
  for (uint8_t i = 0; i < z->sub_count; ++i) {
    if (z->sub_ctx[i].dropped || z->sub_ctx[i].sub != a->sub) continue;
    z_drop(z_move(z->subs[i]));
    z->sub_ctx[i].dropped = true;
    return 0;
  }
  return 0;
}

static int token(rlmesh_zenoh_t* z, const op_t* a) {
  const z_loaned_session_t* s = session(z, a->class_, a->host, a->port);
  z_view_keyexpr_t ke;
  if (s == NULL || z->token_count >= RLMESH_ZENOH_TOKENS ||
      z_view_keyexpr_from_str(&ke, (const char*)a->data) != Z_OK) {
    return -1;
  }
  if (z_liveliness_declare_token(s, &z->tokens[z->token_count], z_loan(ke), NULL) != Z_OK) {
    return -1;
  }
  z->token_count++;
  return 0;
}

static int query(rlmesh_zenoh_t* z, const op_t* a) {
  const z_loaned_session_t* s = session(z, a->class_, a->host, a->port);
  z_view_keyexpr_t ke;
  query_t* q = NULL;
  if (s == NULL || z_view_keyexpr_from_str(&ke, (const char*)a->data) != Z_OK ||
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
    if (z_bytes_copy_from_buf(&payload, a->data + a->key_len, a->len) != Z_OK) {
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

/* Queues a publication for the network task: sending it takes the socket and
 * the radio's path, which a control loop can't wait for. */
static int publish(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  if (a->class_ >= RLMESH_DEVICE_CLASSES) return -1;
  rlmesh_txq_item_t* t = rlmesh_txq_item(a->class_, a->key, a->payload, a->payload_len);
  if (t == NULL) return -1;
  z_mutex_lock(z_loan_mut(z->lock));
  bool queued = rlmesh_txq_push(&z->txq, t);
  z_mutex_unlock(z_loan_mut(z->lock));
  /* rlmesh_zenoh_poll wakes the send task once it has queued them all. */
  return queued ? 0 : -1;
}

/* Sends one publication on its session (open): express when it goes alone,
 * batched with the others otherwise. */
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
static void send_all(rlmesh_zenoh_t* z, rlmesh_txq_item_t* list, const char* host, uint16_t port,
                     rlmesh_zenoh_phase_t* sent) {
  for (uint8_t c = 0; c < RLMESH_DEVICE_CLASSES; ++c) {
    size_t n = 0;
    for (rlmesh_txq_item_t* t = list; t != NULL; t = t->next) n += t->class_ == c;
    if (n == 0) continue;
    /* Its session opens on its first publication. */
    const z_loaned_session_t* s = session(z, c, host, port);
    if (s == NULL) {
      for (size_t i = 0; i < n; ++i) count_failed(z);
      continue;
    }
    bool batch = n > 1 && zp_batch_start(s) == Z_OK;
    for (rlmesh_txq_item_t* t = list; t != NULL; t = t->next) {
      if (t->class_ != c) continue;
      if (send_one(z, t, !batch) != 0) {
        count_failed(z);
      } else {
        phase_add(sent, z->now_us());
      }
    }
    if (batch) zp_batch_stop(s);
  }
}

/* Sends the datagrams that waited, without waiting on the socket: one it
 * can't take now is dropped. Refusals in a row tell the robot the path
 * failed. */
static void send_datagrams(rlmesh_zenoh_t* z, const rlmesh_txq_item_t* list,
                           rlmesh_zenoh_phase_t* phases) {
  uint32_t sent = 0, unsent = 0;
  bool failed = false;
  z_mutex_lock(z_loan_mut(z->dgram_lock));
  for (const rlmesh_txq_item_t* t = list; t != NULL; t = t->next) {
    if (t->class_ != RLMESH_ZENOH_DATAGRAM_CLASS) continue;
    if (z->dgram_fd < 0) {
      unsent++;
      continue;
    }
    ssize_t n = send(z->dgram_fd, t->data + t->key_len, t->len, MSG_DONTWAIT);
    if (n >= 0 && (size_t)n == t->len) {
      phase_add(phases, z->now_us());
      sent++;
      z->dgram_refused = 0;
      continue;
    }
    unsent++;
    bool busy = n >= 0 || errno == EAGAIN || errno == EWOULDBLOCK || errno == ENOBUFS ||
                errno == ENOMEM || errno == EINTR;
    if (!busy && ++z->dgram_refused >= RLMESH_ZENOH_DGRAM_FAILURES) failed = true;
  }
  z_mutex_unlock(z_loan_mut(z->dgram_lock));
  if (sent == 0 && unsent == 0) return;
  z_mutex_lock(z_loan_mut(z->lock));
  z->views.dgram_sent += sent;
  z->views.dgram_unsent += unsent;
  if (failed) {
    z->dgram_failed = true;
    if (z->wake != NULL) z->wake(z->wake_arg);
  }
  z_mutex_unlock(z_loan_mut(z->lock));
}

static bool reset(rlmesh_zenoh_t* z, const char* host, uint16_t port);
static void reconnect(rlmesh_zenoh_t* z);

/* Carries out the robot's actions in order, freeing them: whether one closed
 * the sessions (publications taken before it are then stale). */
static bool run_ops(rlmesh_zenoh_t* z, op_t* ops) {
  bool closed = false;
  while (ops != NULL) {
    op_t* next = ops->next;
    int done = 0;
    switch (ops->kind) {
      case RLMESH_DEVICE_SUBSCRIBE:
        done = subscribe(z, ops);
        break;
      case RLMESH_DEVICE_UNSUBSCRIBE:
        done = unsubscribe(z, ops);
        break;
      case RLMESH_DEVICE_TOKEN:
        done = token(z, ops);
        break;
      case RLMESH_DEVICE_QUERY:
        done = query(z, ops);
        break;
      case RLMESH_DEVICE_RESET:
        closed |= reset(z, ops->host, ops->port);
        break;
      case RLMESH_DEVICE_RECONNECT:
        reconnect(z);
        closed = true;
        break;
      default:
        break;
    }
    if (done != 0) count_failed(z);
    free(ops);
    ops = next;
  }
  return closed;
}

/* The network task: every zenoh-pico call, in the order the robot asked. It
 * holds `lock` only to take what waits; a call that waits on the socket waits
 * here, never on the loop. */
static void* tx_task(void* arg) {
  rlmesh_zenoh_t* z = (rlmesh_zenoh_t*)arg;
  z_mutex_lock(z_loan_mut(z->lock));
  for (;;) {
    while (!z->stopping && z->ops == NULL && z->txq.head == NULL) {
      z_condvar_wait(z_loan_mut(z->tx_wake), z_loan_mut(z->lock));
    }
    if (z->stopping) break;
    /* Everything waiting, at once. */
    rlmesh_zenoh_phase_t sent = {0.0f, 0.0f, 0};
    op_t* ops = z->ops;
    z->ops = z->ops_tail = NULL;
    rlmesh_txq_item_t* list = rlmesh_txq_take(&z->txq);
    char host[sizeof z->target_host];
    memcpy(host, z->target_host, sizeof host);
    uint16_t port = z->target_port;
    z_mutex_unlock(z_loan_mut(z->lock));
    if (run_ops(z, ops)) {
      /* Sealed for the links just closed: no use on new ones. */
      for (rlmesh_txq_item_t* t = list; t != NULL; t = t->next) {
        if (t->class_ != RLMESH_ZENOH_DATAGRAM_CLASS) count_failed(z);
      }
    } else {
      send_all(z, list, host, port, &sent);
    }
    send_datagrams(z, list, &sent);
    while (list != NULL) {
      rlmesh_txq_item_t* next = list->next;
      free(list);
      list = next;
    }
    z_mutex_lock(z_loan_mut(z->lock));
    rlmesh_txq_sent(&z->txq);
    z->sends.c += sent.c;
    z->sends.s += sent.s;
    z->sends.n += sent.n;
  }
  z_mutex_unlock(z_loan_mut(z->lock));
  return NULL;
}

__attribute__((noipa)) size_t rlmesh_zenoh_tx_stack(void) { return RLMESH_ZENOH_TX_STACK; }

/* The datagram socket to host:port, connected (so only the host's datagrams
 * arrive) and reading with a timeout, or -1. */
static int dgram_open(const char* host, uint16_t port) {
  char service[8];
  snprintf(service, sizeof service, "%u", (unsigned)port);
  struct addrinfo hints;
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_DGRAM;
  struct addrinfo* found = NULL;
  if (getaddrinfo(host, service, &hints, &found) != 0 || found == NULL) return -1;
  int fd = -1;
  for (struct addrinfo* a = found; a != NULL && fd < 0; a = a->ai_next) {
    fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
    if (fd < 0) continue;
    struct timeval wait = {0, RLMESH_ZENOH_DGRAM_POLL_MS * 1000};
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &wait, sizeof wait) != 0 ||
        connect(fd, a->ai_addr, a->ai_addrlen) != 0) {
      close(fd);
      fd = -1;
    }
  }
  freeaddrinfo(found);
  return fd;
}

/* Swaps the datagram socket the send task uses, and closes the old one. */
static void dgram_swap(rlmesh_zenoh_t* z, int fd) {
  z_mutex_lock(z_loan_mut(z->dgram_lock));
  int old = z->dgram_fd;
  z->dgram_fd = fd;
  z->dgram_refused = 0;
  z_mutex_unlock(z_loan_mut(z->dgram_lock));
  if (old >= 0) close(old);
}

/* The datagram socket's task: opens and closes it as the loop asks, and reads
 * it in between. Only this task opens, closes or reads it. It ends with the
 * path: once the socket is closed and no open is asked for, it returns, and
 * its stack goes back to the heap when the next open (or the transport's
 * close) joins it. */
static void* rx_task(void* arg) {
  rlmesh_zenoh_t* z = (rlmesh_zenoh_t*)arg;
  /* The request the socket serves (0: none). */
  uint32_t serving = 0;
  z_mutex_lock(z_loan_mut(z->lock));
  for (;;) {
    if (z->stopping || (z->dgram_want == 0 && serving == 0)) break;
    uint32_t want = z->dgram_want;
    if (want != serving) {
      char host[sizeof z->dgram_host];
      memcpy(host, z->dgram_host, sizeof host);
      uint16_t port = z->dgram_port;
      z_mutex_unlock(z_loan_mut(z->lock));
      dgram_swap(z, -1);
      int fd = want != 0 ? dgram_open(host, port) : -1;
      dgram_swap(z, fd);
      z_mutex_lock(z_loan_mut(z->lock));
      serving = want;
      if (want != 0 && fd < 0) {
        z->dgram_failed = true;
        if (z->wake != NULL) z->wake(z->wake_arg);
      }
      continue;
    }
    int fd = z->dgram_fd;
    z_mutex_unlock(z_loan_mut(z->lock));
    ssize_t n = -1;
    if (fd >= 0) {
      n = recv(fd, z->dgram_buf, RLMESH_DEVICE_DATAGRAM_MAX + 1, 0);
    } else {
      z_sleep_ms(RLMESH_ZENOH_DGRAM_POLL_MS);
    }
    if (n > RLMESH_DEVICE_DATAGRAM_MAX) {
      z_mutex_lock(z_loan_mut(z->lock));
      z->views.dgram_dropped++;
      z_mutex_unlock(z_loan_mut(z->lock));
    } else if (n > 0) {
#ifdef RLMESH_ZENOH_TRACE_CHUNK
      if (n > 8) RLMESH_ZENOH_TRACE_CHUNK(z->dgram_buf + 8, (size_t)n - 8, z->now_us());
#endif
      push_datagram(z, z->dgram_buf, (size_t)n);
    } else if (n < 0 && fd >= 0 && errno == ECONNREFUSED) {
      /* Nothing listens on the host's port. */
      z_mutex_lock(z_loan_mut(z->lock));
      z->dgram_failed = true;
      if (z->wake != NULL) z->wake(z->wake_arg);
      z_mutex_unlock(z_loan_mut(z->lock));
    }
    z_mutex_lock(z_loan_mut(z->lock));
  }
  z->rx_ended = true;
  z_mutex_unlock(z_loan_mut(z->lock));
  dgram_swap(z, -1);
  return NULL;
}

__attribute__((noipa)) size_t rlmesh_zenoh_rx_stack(void) { return RLMESH_ZENOH_RX_STACK; }

/* Joins the datagram task if it ended with the path (from the loop): its
 * stack and buffer go back to the heap. Whether none runs now. */
static bool join_rx(rlmesh_zenoh_t* z) {
  if (!z->rx_started) return true;
  z_mutex_lock(z_loan_mut(z->lock));
  bool ended = z->rx_ended;
  z_mutex_unlock(z_loan_mut(z->lock));
  if (!ended) return false;
  z_task_join(z_move(z->rx_task));
  z->rx_started = false;
  z->rx_ended = false;
  free(z->dgram_buf);
  z->dgram_buf = NULL;
  return true;
}

/* Starts the datagram task and its buffer when the robot opens the socket
 * (from the loop) and none runs: 0, or -1 without memory for them. */
static int start_rx(rlmesh_zenoh_t* z) {
  if (!join_rx(z)) return 0;
#if defined(ZENOH_ARDUINO_ESP32)
  z_task_attr_t attrs = {.name = "rlmesh-udp",
                         .priority = RLMESH_ZENOH_RX_PRIORITY,
                         .stack_depth = rlmesh_zenoh_rx_stack()};
  z_task_attr_t* attr = &attrs;
#else
  z_task_attr_t* attr = NULL;
#endif
  z->dgram_buf = (uint8_t*)malloc(RLMESH_DEVICE_DATAGRAM_MAX + 1);
  if (z->dgram_buf == NULL || z_task_init(&z->rx_task, attr, rx_task, z) != Z_OK) {
    free(z->dgram_buf);
    z->dgram_buf = NULL;
    return -1;
  }
  z->rx_started = true;
  return 0;
}

/* Asks the datagram task to open the socket to the robot's host on `port`
 * (anew), or with `port` 0 to close it; what waits to be sent goes. The open
 * is asked for before a task that ended is replaced: a task still running
 * takes it instead of ending. */
static void dgram_request(rlmesh_zenoh_t* z, uint16_t port) {
  uint16_t ignored = 0;
  const char* host = port != 0 ? rlmesh_device_robot_host(z->robot, &ignored) : NULL;
  z_mutex_lock(z_loan_mut(z->lock));
  rlmesh_txq_drop_class(&z->txq, RLMESH_ZENOH_DATAGRAM_CLASS);
  z->dgram_failed = false;
  if (host != NULL && strlen(host) < sizeof z->dgram_host) {
    strcpy(z->dgram_host, host);
    z->dgram_port = port;
    z->dgram_want = z->dgram_want + 1 == 0 ? 1 : z->dgram_want + 1;
  } else {
    z->dgram_want = 0;
  }
  bool wanted = z->dgram_want != 0;
  z_mutex_unlock(z_loan_mut(z->lock));
  if (wanted && start_rx(z) != 0) {
    /* No task to open it: the robot is told the socket failed. */
    z_mutex_lock(z_loan_mut(z->lock));
    z->dgram_want = 0;
    z->dgram_failed = true;
    z_mutex_unlock(z_loan_mut(z->lock));
  }
}

/* Queues a datagram for the send task, in place of any on its lane. */
static int datagram(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  rlmesh_txq_item_t* t =
      rlmesh_txq_item(RLMESH_ZENOH_DATAGRAM_CLASS, a->key, a->payload, a->payload_len);
  if (t == NULL) return -1;
  z_mutex_lock(z_loan_mut(z->lock));
  bool queued = rlmesh_txq_push(&z->txq, t);
  z_mutex_unlock(z_loan_mut(z->lock));
  return queued ? 0 : -1;
}

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
 * satellite, another home address): whether they closed. */
static bool reset(rlmesh_zenoh_t* z, const char* host, uint16_t port) {
  undeclare(z);
  memset(z->unreachable, 0, sizeof z->unreachable);
  if (port == z->port && strcmp(host, z->host) == 0) return false;
  close_sessions(z);
  return true;
}

/* Drops everything: replies stopped, and a session whose peer is gone never
 * comes back on its own. The robot's next actions open new ones. */
static void reconnect(rlmesh_zenoh_t* z) {
  undeclare(z);
  close_sessions(z);
  memset(z->unreachable, 0, sizeof z->unreachable);
}

/* Stops the tasks started so far (`tasks` of verify and send, in that order,
 * and the datagram task if it started). */
static void stop_tasks(rlmesh_zenoh_t* z, int tasks) {
  z_mutex_lock(z_loan_mut(z->lock));
  z->stopping = true;
  z_condvar_signal(z_loan_mut(z->verify_wake));
  z_condvar_signal(z_loan_mut(z->tx_wake));
  z_mutex_unlock(z_loan_mut(z->lock));
  if (tasks > 0) z_task_join(z_move(z->verify_task));
  if (tasks > 1) z_task_join(z_move(z->tx_task));
  if (z->rx_started) z_task_join(z_move(z->rx_task));
  z->rx_started = false;
  z->rx_ended = false;
  free(z->dgram_buf);
  z->dgram_buf = NULL;
}

static void drop_locks(rlmesh_zenoh_t* z) {
  z_drop(z_move(z->tx_wake));
  z_drop(z_move(z->verify_wake));
  z_drop(z_move(z->dgram_lock));
  z_drop(z_move(z->lock));
}

int rlmesh_zenoh_open(rlmesh_zenoh_t* z, rlmesh_device_robot_t* robot,
                      rlmesh_device_now_us_fn now_us) {
  if (z == NULL || robot == NULL || now_us == NULL) return -1;
  memset(z, 0, sizeof *z);
  z->txq.cap = RLMESH_ZENOH_TX_BYTES;
  z->robot = robot;
  z->now_us = now_us;
  z->dgram_fd = -1;
  if (z_mutex_init(&z->lock) != Z_OK) return -1;
  if (z_mutex_init(&z->dgram_lock) != Z_OK || z_condvar_init(&z->verify_wake) != Z_OK ||
      z_condvar_init(&z->tx_wake) != Z_OK) {
    drop_locks(z);
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
  int tasks = 0;
  if (z_task_init(&z->verify_task, verify_attr, verify_task, z) == Z_OK) tasks++;
  if (tasks == 1 && z_task_init(&z->tx_task, tx_attr, tx_task, z) == Z_OK) tasks++;
  if (tasks < 2) {
    stop_tasks(z, tasks);
    drop_locks(z);
    return -1;
  }
  return 0;
}

/* Queues one of the robot's session actions for the network task, copied
 * with the host it is meant for: 0, or -1 without memory for it. */
static int enqueue(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  size_t key_len = strlen(a->key) + 1;
  size_t len = a->has_payload ? a->payload_len : 0;
  op_t* op = (op_t*)malloc(sizeof(op_t) + key_len + len);
  if (op == NULL) return -1;
  op->next = NULL;
  op->kind = a->kind;
  op->class_ = a->class_;
  op->sub = a->sub;
  op->has_payload = a->has_payload;
  op->query_id = a->query_id;
  op->timeout_ms = a->timeout_ms;
  uint16_t port = 0;
  const char* host = rlmesh_device_robot_host(z->robot, &port);
  op->host[0] = '\0';
  if (host != NULL && strlen(host) < sizeof op->host) strcpy(op->host, host);
  op->port = port;
  op->key_len = key_len;
  op->len = len;
  memcpy(op->data, a->key, key_len);
  if (len > 0) memcpy(op->data + key_len, a->payload, len);
  z_mutex_lock(z_loan_mut(z->lock));
  if (z->ops_tail == NULL) {
    z->ops = z->ops_tail = op;
  } else {
    z->ops_tail->next = op;
    z->ops_tail = op;
  }
  z_mutex_unlock(z_loan_mut(z->lock));
  return 0;
}

void rlmesh_zenoh_poll_gaps(rlmesh_zenoh_t* z, uint32_t* longest_us, uint32_t* over_50ms) {
  *longest_us = z->poll_gap_max_us > UINT32_MAX ? UINT32_MAX : (uint32_t)z->poll_gap_max_us;
  *over_50ms = z->poll_gaps_long;
  z->poll_gap_max_us = 0;
  z->poll_gaps_long = 0;
}

int rlmesh_zenoh_poll(rlmesh_zenoh_t* z) {
  uint64_t polled = z->now_us();
  if (z->polled_us != 0) {
    uint64_t gap = polled - z->polled_us;
    if (gap > z->poll_gap_max_us) z->poll_gap_max_us = gap;
    if (gap > 50000) z->poll_gaps_long++;
  }
  z->polled_us = polled;
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
      case EV_DATAGRAM:
        rlmesh_device_robot_datagram(z->robot, e->data, e->len, now);
        break;
      default:
        rlmesh_device_robot_query_done(z->robot, e->query_id, now);
        break;
    }
    event_t* next = e->next;
    free(e);
    e = next;
  }
  z_mutex_lock(z_loan_mut(z->lock));
  bool dgram_failed = z->dgram_failed;
  z->dgram_failed = false;
  z_mutex_unlock(z_loan_mut(z->lock));
  if (dgram_failed) rlmesh_device_robot_datagram_failed(z->robot, z->now_us());
  /* A datagram task that ended with the path is gone at once, not at the
   * next path's open. */
  join_rx(z);
  rlmesh_device_action_t a;
  int rc;
  while ((rc = rlmesh_device_robot_next(z->robot, z->now_us(), &a)) == 1) {
    if (a.kind == RLMESH_DEVICE_PUBLISH) {
      if (publish(z, &a) != 0) count_failed(z);
      continue;
    }
    if (a.kind == RLMESH_DEVICE_DATAGRAM) {
      if (datagram(z, &a) != 0) count_failed(z);
      continue;
    }
    if (a.kind == RLMESH_DEVICE_DATAGRAM_OPEN || a.kind == RLMESH_DEVICE_DATAGRAM_CLOSE) {
      dgram_request(z, a.kind == RLMESH_DEVICE_DATAGRAM_OPEN ? a.port : 0);
      continue;
    }
    if (a.kind == RLMESH_DEVICE_RESET || a.kind == RLMESH_DEVICE_RECONNECT) dgram_request(z, 0);
    if (enqueue(z, &a) != 0) {
      /* The robot waits on every query it asked for: it's over. */
      if (a.kind == RLMESH_DEVICE_QUERY) push(z, EV_DONE, 0, a.query_id, NULL, NULL);
      count_failed(z);
    }
  }
  /* One wake for everything this poll queued: a tick's samples leave in one
   * write. The host they go to is the robot's now. */
  uint16_t port = 0;
  const char* host = rlmesh_device_robot_host(z->robot, &port);
  z_mutex_lock(z_loan_mut(z->lock));
  if (host != NULL && strlen(host) < sizeof z->target_host) {
    strcpy(z->target_host, host);
    z->target_port = port;
  }
  if (z->txq.head != NULL || z->ops != NULL) z_condvar_signal(z_loan_mut(z->tx_wake));
  z_mutex_unlock(z_loan_mut(z->lock));
  return rc < 0 ? rc : 0;
}

void rlmesh_zenoh_close(rlmesh_zenoh_t* z) {
  /* The network task first: then nothing else calls into zenoh-pico. */
  stop_tasks(z, 2);
  undeclare(z);
  close_sessions(z);
  drop_unsent(z);
  for (op_t* op = z->ops; op != NULL;) {
    op_t* next = op->next;
    free(op);
    op = next;
  }
  z->ops = z->ops_tail = NULL;
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
  drop_locks(z);
}
