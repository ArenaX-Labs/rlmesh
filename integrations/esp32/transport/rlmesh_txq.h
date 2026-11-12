/* The transport's queue of publications waiting for its send task, in two
 * lanes. A sample (any class but control) is latest-wins per key expression: a
 * newer one replaces the one waiting, so a slow link lowers a stream's rate
 * instead of queueing it. Control publications (heartbeats) queue in order and
 * go first. Bounded in bytes: past the bound the oldest sample goes, and only
 * when no sample is left the oldest control publication. What the send task
 * took out and is still sending counts against the bound too, up to half of
 * it: a send held up by the link never takes the room of the newest samples,
 * and all of it together stays within one and a half bounds, however long the
 * link stalls. Not locked: the transport holds its lock around every call. */

#ifndef RLMESH_TXQ_H_
#define RLMESH_TXQ_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* One publication: the NUL-terminated key, then the payload. */
typedef struct rlmesh_txq_item {
  struct rlmesh_txq_item* next;
  uint8_t class_;
  size_t key_len;
  size_t len;
  uint8_t data[];
} rlmesh_txq_item_t;

typedef struct rlmesh_txq {
  rlmesh_txq_item_t* head;
  rlmesh_txq_item_t* tail;
  size_t bytes;
  /* Bytes the send task took out and has not finished sending. */
  size_t inflight;
  size_t cap;
  /* Publications dropped to keep to `cap`, and samples a newer one on the
   * same key replaced (counted in `dropped` too). */
  uint32_t dropped;
  uint32_t replaced;
} rlmesh_txq_t;

static inline size_t rlmesh_txq_size(const rlmesh_txq_item_t* t) {
  return sizeof *t + t->key_len + t->len;
}

/* A publication of `len` bytes on `key` for class `class_`, or NULL without
 * memory. */
static inline rlmesh_txq_item_t* rlmesh_txq_item(uint8_t class_, const char* key,
                                                 const uint8_t* payload, size_t len) {
  size_t key_len = strlen(key) + 1;
  rlmesh_txq_item_t* t = (rlmesh_txq_item_t*)malloc(sizeof *t + key_len + len);
  if (t == NULL) return NULL;
  t->next = NULL;
  t->class_ = class_;
  t->key_len = key_len;
  t->len = len;
  memcpy(t->data, key, key_len);
  if (len > 0) memcpy(t->data + key_len, payload, len);
  return t;
}

/* Unlinks the item after `prev` (the head when `prev` is NULL) and frees it. */
static inline void rlmesh_txq_drop_after(rlmesh_txq_t* q, rlmesh_txq_item_t* prev) {
  rlmesh_txq_item_t* t = prev == NULL ? q->head : prev->next;
  if (prev == NULL) {
    q->head = t->next;
  } else {
    prev->next = t->next;
  }
  if (q->tail == t) q->tail = prev;
  q->bytes -= rlmesh_txq_size(t);
  q->dropped++;
  free(t);
}

/* What is being sent counts against the bound, at most half of it. */
static inline size_t rlmesh_txq_sending(const rlmesh_txq_t* q) {
  return q->inflight < q->cap / 2 ? q->inflight : q->cap / 2;
}

/* The class whose publications are never replaced or dropped for a sample. */
#define RLMESH_TXQ_CONTROL 0

/* Takes `t`, which the queue then owns: a sample replaces the one waiting on
 * its key. While over the bound, drops the oldest sample, else the oldest
 * control publication (never `t`). False when `t` alone is over the bound
 * (then dropped). */
static inline bool rlmesh_txq_push(rlmesh_txq_t* q, rlmesh_txq_item_t* t) {
  size_t size = rlmesh_txq_size(t);
  if (size + rlmesh_txq_sending(q) > q->cap) {
    free(t);
    q->dropped++;
    return false;
  }
  if (t->class_ != RLMESH_TXQ_CONTROL) {
    for (rlmesh_txq_item_t *p = NULL, *i = q->head; i != NULL; p = i, i = i->next) {
      if (i->class_ == t->class_ && strcmp((const char*)i->data, (const char*)t->data) == 0) {
        rlmesh_txq_drop_after(q, p);
        q->replaced++;
        break;
      }
    }
  }
  if (q->tail == NULL) {
    q->head = q->tail = t;
  } else {
    q->tail->next = t;
    q->tail = t;
  }
  q->bytes += size;
  while (q->bytes + rlmesh_txq_sending(q) > q->cap) {
    rlmesh_txq_item_t* prev = NULL;
    rlmesh_txq_item_t* sample = NULL;
    for (rlmesh_txq_item_t *p = NULL, *i = q->head; i != t; p = i, i = i->next) {
      if (i->class_ != RLMESH_TXQ_CONTROL) {
        prev = p;
        sample = i;
        break;
      }
    }
    /* No sample left: the oldest control publication (never `t`). */
    rlmesh_txq_drop_after(q, sample != NULL ? prev : NULL);
  }
  return true;
}

/* The oldest publication, now the caller's, or NULL. */
static inline rlmesh_txq_item_t* rlmesh_txq_pop(rlmesh_txq_t* q) {
  rlmesh_txq_item_t* t = q->head;
  if (t == NULL) return NULL;
  q->head = t->next;
  if (q->head == NULL) q->tail = NULL;
  q->bytes -= rlmesh_txq_size(t);
  t->next = NULL;
  return t;
}

/* Drops every publication of class `class_` waiting (not counted: they were
 * withdrawn, not crowded out). */
static inline void rlmesh_txq_drop_class(rlmesh_txq_t* q, uint8_t class_) {
  rlmesh_txq_item_t* prev = NULL;
  for (rlmesh_txq_item_t* i = q->head; i != NULL;) {
    rlmesh_txq_item_t* next = i->next;
    if (i->class_ == class_) {
      rlmesh_txq_drop_after(q, prev);
      q->dropped--;
    } else {
      prev = i;
    }
    i = next;
  }
}

/* Takes every publication waiting, now the caller's to send: they count
 * against the bound until rlmesh_txq_sent. */
static inline rlmesh_txq_item_t* rlmesh_txq_take(rlmesh_txq_t* q) {
  rlmesh_txq_item_t* list = q->head;
  q->inflight += q->bytes;
  q->head = q->tail = NULL;
  q->bytes = 0;
  return list;
}

/* What rlmesh_txq_take handed out is sent (or dropped). */
static inline void rlmesh_txq_sent(rlmesh_txq_t* q) { q->inflight = 0; }

/* Frees every publication waiting. */
static inline void rlmesh_txq_clear(rlmesh_txq_t* q) {
  for (rlmesh_txq_item_t* t = rlmesh_txq_pop(q); t != NULL; t = rlmesh_txq_pop(q)) free(t);
}

#endif /* RLMESH_TXQ_H_ */
