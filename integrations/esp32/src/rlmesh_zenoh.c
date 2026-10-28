#include "rlmesh_zenoh.h"

#include <stdlib.h>
#include <string.h>

enum { EV_SAMPLE = 1, EV_REPLY = 2, EV_DONE = 3 };

typedef struct rlmesh_zenoh_event {
  struct rlmesh_zenoh_event* next;
  uint8_t kind;
  uint8_t sub;
  uint32_t query_id;
  size_t len;
  uint8_t data[];
} event_t;

/* A query's closure context: freed by its dropper. */
typedef struct {
  rlmesh_zenoh_t* z;
  uint32_t id;
} query_t;

/* Queues a copy of `data`, from zenoh-pico's read task. */
static void push(rlmesh_zenoh_t* z, uint8_t kind, uint8_t sub, uint32_t query_id,
                 const z_loaned_bytes_t* payload) {
  z_owned_slice_t slice;
  const uint8_t* data = NULL;
  size_t len = 0;
  bool have = payload != NULL && z_bytes_to_slice(payload, &slice) == Z_OK;
  if (have) {
    data = z_slice_data(z_loan(slice));
    len = z_slice_len(z_loan(slice));
  }
  event_t* e = (event_t*)malloc(sizeof(event_t) + len);
  if (e != NULL) {
    e->next = NULL;
    e->kind = kind;
    e->sub = sub;
    e->query_id = query_id;
    e->len = len;
    if (len > 0) memcpy(e->data, data, len);
  }
  if (have) z_drop(z_move(slice));
  z_mutex_lock(z_loan_mut(z->lock));
  if (e == NULL) {
    z->failed++;
  } else if (z->tail == NULL) {
    z->head = z->tail = e;
  } else {
    z->tail->next = e;
    z->tail = e;
  }
  z_mutex_unlock(z_loan_mut(z->lock));
}

static void on_sample(z_loaned_sample_t* sample, void* ctx) {
  rlmesh_zenoh_sub_t* s = (rlmesh_zenoh_sub_t*)ctx;
  push(s->z, EV_SAMPLE, s->sub, 0, z_sample_payload(sample));
}

static void on_reply(z_loaned_reply_t* reply, void* ctx) {
  query_t* q = (query_t*)ctx;
  if (z_reply_is_ok(reply)) {
    push(q->z, EV_REPLY, 0, q->id, z_sample_payload(z_reply_ok(reply)));
  }
}

static void on_query_done(void* ctx) {
  query_t* q = (query_t*)ctx;
  push(q->z, EV_DONE, 0, q->id, NULL);
  free(q);
}

static int subscribe(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  if (z->sub_count >= RLMESH_ZENOH_SUBS) return -1;
  z_view_keyexpr_t ke;
  if (z_view_keyexpr_from_str(&ke, a->key) != Z_OK) return -1;
  rlmesh_zenoh_sub_t* ctx = &z->sub_ctx[z->sub_count];
  ctx->z = z;
  ctx->sub = a->sub;
  z_owned_closure_sample_t callback;
  z_closure(&callback, on_sample, NULL, ctx);
  if (z_declare_subscriber(z_loan(z->session), &z->subs[z->sub_count], z_loan(ke), z_move(callback),
                           NULL) != Z_OK) {
    return -1;
  }
  z->sub_count++;
  return 0;
}

static int token(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  z_view_keyexpr_t ke;
  if (z->has_token || z_view_keyexpr_from_str(&ke, a->key) != Z_OK) return -1;
  if (z_liveliness_declare_token(z_loan(z->session), &z->token, z_loan(ke), NULL) != Z_OK) {
    return -1;
  }
  z->has_token = true;
  return 0;
}

static int query(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  z_view_keyexpr_t ke;
  if (z_view_keyexpr_from_str(&ke, a->key) != Z_OK) return -1;
  query_t* q = (query_t*)malloc(sizeof(query_t));
  if (q == NULL) return -1;
  q->z = z;
  q->id = a->query_id;
  z_get_options_t opts;
  z_get_options_default(&opts);
  opts.consolidation = z_query_consolidation_none();
  opts.timeout_ms = RLMESH_DEVICE_QUERY_MS;
  opts.priority = Z_PRIORITY_REAL_TIME;
  opts.is_express = true;
  z_owned_bytes_t payload;
  if (a->has_payload) {
    if (z_bytes_copy_from_buf(&payload, a->payload, a->payload_len) != Z_OK) {
      free(q);
      return -1;
    }
    opts.payload = z_move(payload);
  }
  z_owned_closure_reply_t callback;
  z_closure(&callback, on_reply, on_query_done, q);
  /* On failure the closure is dropped, which frees q and reports the query
   * done: the robot retries. */
  return z_get(z_loan(z->session), z_loan(ke), "", z_move(callback), &opts) == Z_OK ? 0 : -1;
}

static int publish(rlmesh_zenoh_t* z, const rlmesh_device_action_t* a) {
  z_view_keyexpr_t ke;
  if (z_view_keyexpr_from_str(&ke, a->key) != Z_OK) return -1;
  z_owned_bytes_t payload;
  if (z_bytes_copy_from_buf(&payload, a->payload, a->payload_len) != Z_OK) return -1;
  z_put_options_t opts;
  z_put_options_default(&opts);
  opts.priority = Z_PRIORITY_REAL_TIME;
  opts.is_express = true;
  return z_put(z_loan(z->session), z_loan(ke), z_move(payload), &opts) == Z_OK ? 0 : -1;
}

int rlmesh_zenoh_open(rlmesh_zenoh_t* z, const char* connect, rlmesh_device_robot_t* robot,
                      rlmesh_device_now_us_fn now_us) {
  if (z == NULL || connect == NULL || robot == NULL || now_us == NULL) return -1;
  memset(z, 0, sizeof *z);
  z->robot = robot;
  z->now_us = now_us;
  if (z_mutex_init(&z->lock) != Z_OK) return -1;
  z_owned_config_t config;
  z_config_default(&config);
  zp_config_insert(z_loan_mut(config), Z_CONFIG_MODE_KEY, "client");
  zp_config_insert(z_loan_mut(config), Z_CONFIG_CONNECT_KEY, connect);
  if (z_open(&z->session, z_move(config), NULL) != Z_OK) {
    z_drop(z_move(z->lock));
    return -1;
  }
  return 0;
}

int rlmesh_zenoh_poll(rlmesh_zenoh_t* z) {
  z_mutex_lock(z_loan_mut(z->lock));
  event_t* e = z->head;
  z->head = z->tail = NULL;
  z_mutex_unlock(z_loan_mut(z->lock));
  while (e != NULL) {
    uint64_t now = z->now_us();
    switch (e->kind) {
      case EV_SAMPLE:
        rlmesh_device_robot_sample(z->robot, e->sub, e->data, e->len, now);
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
      default:
        break;
    }
    if (done != 0) z->failed++;
  }
  return rc < 0 ? rc : 0;
}

void rlmesh_zenoh_close(rlmesh_zenoh_t* z) {
  for (uint8_t i = 0; i < z->sub_count; ++i) z_drop(z_move(z->subs[i]));
  if (z->has_token) z_drop(z_move(z->token));
  /* Pending queries are dropped with the session: their droppers queue. */
  z_drop(z_move(z->session));
  for (event_t* e = z->head; e != NULL;) {
    event_t* next = e->next;
    free(e);
    e = next;
  }
  z->head = z->tail = NULL;
  z_drop(z_move(z->lock));
}
