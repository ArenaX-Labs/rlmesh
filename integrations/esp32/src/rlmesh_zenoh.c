#include "rlmesh_zenoh.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { EV_SAMPLE = 1, EV_REPLY = 2, EV_DONE = 3 };

typedef struct rlmesh_zenoh_event {
  struct rlmesh_zenoh_event* next;
  uint8_t kind;
  uint8_t sub;
  uint32_t query_id;
  /* data holds the key (a sample's), then the payload. */
  size_t key_len;
  size_t len;
  uint8_t data[];
} event_t;

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
    } else {
      z->tail->next = e;
      z->tail = e;
    }
  }
  z_mutex_unlock(z_loan_mut(z->lock));
}

static void on_sample(z_loaned_sample_t* sample, void* ctx) {
  rlmesh_zenoh_sub_t* s = (rlmesh_zenoh_sub_t*)ctx;
  push(s->z, EV_SAMPLE, s->sub, 0, z_sample_keyexpr(sample), z_sample_payload(sample));
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
 * declared on them (the robot's Reset for the move follows). */
static const z_loaned_session_t* session(rlmesh_zenoh_t* z, uint8_t c) {
  if (c >= RLMESH_DEVICE_CLASSES) return NULL;
  if (!z->open[c]) {
    uint16_t port = 0;
    const char* host = rlmesh_device_robot_host(z->robot, &port);
    char connect[96];
    if (host == NULL || strlen(host) >= sizeof z->host) return NULL;
    int n = snprintf(connect, sizeof connect, "tcp/%s:%u", host, (unsigned)(port + c));
    if (n <= 0 || (size_t)n >= sizeof connect) return NULL;
    if (port != z->port || strcmp(host, z->host) != 0) {
      undeclare(z);
      close_sessions(z);
    }
    z_owned_config_t config;
    z_config_default(&config);
    zp_config_insert(z_loan_mut(config), Z_CONFIG_MODE_KEY, "client");
    zp_config_insert(z_loan_mut(config), Z_CONFIG_CONNECT_KEY, connect);
    if (z_open(&z->sessions[c], z_move(config), NULL) != Z_OK) return NULL;
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

static int publish(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  const z_loaned_session_t* s = session(z, a->class_);
  z_view_keyexpr_t ke;
  if (s == NULL || z_view_keyexpr_from_str(&ke, a->key) != Z_OK) return -1;
  z_owned_bytes_t payload;
  if (z_bytes_copy_from_buf(&payload, a->payload, a->payload_len) != Z_OK) return -1;
  z_put_options_t opts;
  z_put_options_default(&opts);
  opts.priority = Z_PRIORITY_REAL_TIME;
  opts.is_express = true;
  return z_put(s, z_loan(ke), z_move(payload), &opts) == Z_OK ? 0 : -1;
}

/* Closes every class's session. Pending queries are dropped with them:
 * their droppers queue. */
static void close_sessions(rlmesh_zenoh_t* z) {
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
  uint16_t port = 0;
  const char* host = rlmesh_device_robot_host(z->robot, &port);
  if (host == NULL || port != z->port || strcmp(host, z->host) != 0) close_sessions(z);
}

int rlmesh_zenoh_open(rlmesh_zenoh_t* z, rlmesh_device_robot_t* robot,
                      rlmesh_device_now_us_fn now_us) {
  if (z == NULL || robot == NULL || now_us == NULL) return -1;
  memset(z, 0, sizeof *z);
  z->robot = robot;
  z->now_us = now_us;
  return z_mutex_init(&z->lock) == Z_OK ? 0 : -1;
}

int rlmesh_zenoh_poll(rlmesh_zenoh_t* z) {
  z_mutex_lock(z_loan_mut(z->lock));
  event_t* e = z->head;
  z->head = z->tail = NULL;
  z->queued = 0;
  z_mutex_unlock(z_loan_mut(z->lock));
  while (e != NULL) {
    uint64_t now = z->now_us();
    switch (e->kind) {
      case EV_SAMPLE:
        rlmesh_device_robot_sample(z->robot, e->sub, (const char*)e->data, e->key_len,
                                   e->data + e->key_len, e->len, now);
        break;
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
      case RLMESH_DEVICE_PUBLISH:
        done = publish(z, &a);
        break;
      case RLMESH_DEVICE_RESET:
        reset(z);
        done = 0;
        break;
      default:
        break;
    }
    if (done != 0) count_failed(z);
  }
  return rc < 0 ? rc : 0;
}

void rlmesh_zenoh_close(rlmesh_zenoh_t* z) {
  undeclare(z);
  close_sessions(z);
  for (event_t* e = z->head; e != NULL;) {
    event_t* next = e->next;
    free(e);
    e = next;
  }
  z->head = z->tail = NULL;
  z_drop(z_move(z->lock));
}
