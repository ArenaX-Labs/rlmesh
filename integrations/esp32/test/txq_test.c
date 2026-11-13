/* The transport's send queue under a link that never drains it: memory stays
 * within the bound (with what is being sent), each stream keeps only its
 * newest sample, control
 * publications (heartbeats) are kept before any sample, and what was dropped
 * is counted. */

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "rlmesh_txq.h"

static rlmesh_txq_item_t* sample(const char* key, uint32_t seq, size_t len) {
  uint8_t payload[256] = {0};
  memcpy(payload, &seq, sizeof seq);
  /* Heartbeats ride the control class; samples the realtime one. */
  return rlmesh_txq_item(strstr(key, "/hb") != NULL ? 0 : 1, key, payload, len);
}

static uint32_t seq_of(const rlmesh_txq_item_t* t) {
  uint32_t seq;
  memcpy(&seq, t->data + t->key_len, sizeof seq);
  return seq;
}

int main(void) {
  const char* keys[] = {"d/eyes/servos", "d/eyes/servos/_applied", "p/eyes/hb"};
  rlmesh_txq_t q = {.cap = 8 * 1024};
  /* Ten minutes of a 50 Hz state, its applied stream and 20 Hz heartbeats,
   * with nothing sent. */
  uint32_t pushed = 0;
  for (uint32_t i = 0; i < 30000; ++i) {
    for (int k = 0; k < 3; ++k) {
      if (k == 2 && i % 5 >= 2) continue;
      assert(rlmesh_txq_push(&q, sample(keys[k], i, k == 2 ? 120 : 60)));
      pushed++;
      assert(q.bytes <= q.cap);
    }
  }
  size_t bytes = 0;
  uint32_t items = 0;
  uint32_t newest[3] = {0}, oldest[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
  for (rlmesh_txq_item_t* t = q.head; t != NULL; t = t->next) {
    bytes += rlmesh_txq_size(t);
    items++;
    for (int k = 0; k < 3; ++k) {
      if (strcmp((const char*)t->data, keys[k]) == 0) {
        if (seq_of(t) > newest[k]) newest[k] = seq_of(t);
        if (seq_of(t) < oldest[k]) oldest[k] = seq_of(t);
      }
    }
  }
  assert(bytes == q.bytes);
  assert(items + q.dropped == pushed);
  /* Each sample stream holds only its newest sample; heartbeats queue in
   * order, the newest kept. */
  assert(newest[0] == 29999 && oldest[0] == 29999);
  assert(newest[1] == 29999 && oldest[1] == 29999);
  assert(newest[2] >= 29995 && oldest[2] > 29000);
  uint32_t last[3] = {0};
  for (rlmesh_txq_item_t* t = q.head; t != NULL; t = t->next) {
    for (int k = 0; k < 3; ++k) {
      if (strcmp((const char*)t->data, keys[k]) == 0) {
        assert(seq_of(t) > last[k] || last[k] == 0);
        last[k] = seq_of(t);
      }
    }
  }
  /* A full queue of samples gives way to a heartbeat. */
  rlmesh_txq_t lanes = {.cap = 1024};
  char key[32];
  for (int i = 0; i < 40; ++i) {
    snprintf(key, sizeof key, "d/eyes/part%d", i);
    assert(rlmesh_txq_push(&lanes, sample(key, (uint32_t)i, 60)));
  }
  assert(rlmesh_txq_push(&lanes, sample("p/eyes/hb", 1, 120)));
  assert(lanes.bytes <= lanes.cap && lanes.tail->class_ == 0);
  rlmesh_txq_clear(&lanes);
  /* One publication larger than the bound is refused, not queued. */
  rlmesh_txq_t small = {.cap = 64};
  assert(!rlmesh_txq_push(&small, sample("big", 1, 200)));
  assert(small.head == NULL && small.bytes == 0 && small.dropped == 1);
  /* Datagrams (a class of their own) keep the newest per lane, and closing
   * the socket withdraws them all without counting them dropped. */
  rlmesh_txq_t mixed = {.cap = 4096};
  for (uint32_t i = 0; i < 100; ++i) {
    assert(rlmesh_txq_push(&mixed, rlmesh_txq_item(4, "d/eyes/servos", (uint8_t*)&i, 4)));
    assert(rlmesh_txq_push(&mixed, rlmesh_txq_item(4, "probe", (uint8_t*)&i, 4)));
  }
  assert(rlmesh_txq_push(&mixed, sample("d/eyes/servos", 7, 60)));
  assert(rlmesh_txq_push(&mixed, sample("p/eyes/hb", 7, 60)));
  uint32_t datagrams = 0;
  for (rlmesh_txq_item_t* t = mixed.head; t != NULL; t = t->next) datagrams += t->class_ == 4;
  assert(datagrams == 2 && mixed.dropped == 198);
  rlmesh_txq_drop_class(&mixed, 4);
  assert(mixed.dropped == 198);
  for (rlmesh_txq_item_t* t = mixed.head; t != NULL; t = t->next) assert(t->class_ != 4);
  assert(mixed.head != NULL && mixed.head->next != NULL && mixed.head->next->next == NULL);
  assert(mixed.tail == mixed.head->next);
  rlmesh_txq_clear(&mixed);
  /* What the send task took out and is still sending counts against the
   * bound, up to half of it: a link that blocks mid-send never holds twice
   * the bound, and the newest samples still queue behind it. */
  rlmesh_txq_t sending = {.cap = 2048};
  for (int i = 0; i < 12; ++i) {
    snprintf(key, sizeof key, "d/eyes/part%d", i);
    assert(rlmesh_txq_push(&sending, sample(key, (uint32_t)i, 100)));
  }
  size_t taken_bytes = sending.bytes;
  rlmesh_txq_item_t* taken = rlmesh_txq_take(&sending);
  assert(sending.head == NULL && sending.inflight == taken_bytes);
  for (uint32_t i = 0; i < 1000; ++i) {
    snprintf(key, sizeof key, "d/eyes/part%u", i % 40);
    assert(rlmesh_txq_push(&sending, sample(key, i, 100)));
    assert(sending.bytes + rlmesh_txq_sending(&sending) <= sending.cap);
    assert(sending.bytes + sending.inflight <= sending.cap * 3 / 2);
  }
  assert(sending.tail != NULL && seq_of(sending.tail) == 999);
  rlmesh_txq_sent(&sending);
  while (taken != NULL) {
    rlmesh_txq_item_t* next = taken->next;
    free(taken);
    taken = next;
  }
  assert(rlmesh_txq_push(&sending, sample("d/eyes/again", 1, 100)));
  rlmesh_txq_clear(&sending);
  /* Draining empties it. */
  for (rlmesh_txq_item_t* t = rlmesh_txq_pop(&q); t != NULL; t = rlmesh_txq_pop(&q)) free(t);
  assert(q.head == NULL && q.tail == NULL && q.bytes == 0);
  printf("txq: %u pushed, %u kept in %zu of %zu bytes, %u dropped\n", (unsigned)pushed,
         (unsigned)items, bytes, (size_t)8 * 1024, (unsigned)q.dropped);
  return 0;
}
