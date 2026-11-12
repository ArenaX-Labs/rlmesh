/* The board's loop on Linux through many link losses: the device robot, with
 * two commanded parts, over rlmesh_zenoh against a real session host
 * (rlmesh-device's test host in its soak mode), reaching it through a proxy in
 * this process that resets every link at once, as a host cutting a stalled
 * link or a Wi-Fi outage does. After each cycle, once commands flow again, it
 * reads the bytes the process holds on its heap: the robot's, the
 * transport's and zenoh-pico's together, as a board's one heap holds them.
 *
 *   soak_robot <host:port#secret.hostkey> <cycles> <cut|hold|race|load> [max]
 *
 * `cut` resets every link; `hold` carries nothing for longer than the robot
 * waits for its host, then resets them (an outage the board notices first).
 * Either fails when a cycle does not end with commands flowing, or when the
 * heap held after the last two cycles exceeds that after the first two by more
 * than `max` bytes (default 2048). `race` publishes from two threads on
 * one zenoh-pico session while its links reset every 20 ms, for `cycles`
 * seconds: zenoh-pico clearing a failed link under a publisher crashes it.
 * `load` drives the board for `cycles` seconds as a policy does (30 chunks a
 * second to each part, its state published at 50 Hz and read at 200 Hz), every
 * link stalled for 1.5 s every 10 s, and fails if the heap ever held more than
 * `max` bytes. SOAK_CAP=<bytes> makes the heap that small, as a board's is:
 * past it an allocation fails, and one that aborts fails the run. SOAK_SITES
 * prints where the most was held at the peak; SOAK_TRACE the robot's state. */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <execinfo.h>
#include <malloc.h>
#include <netinet/in.h>
#include <poll.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "rlmesh_device.h"
#include "rlmesh_zenoh.h"

/* Two commanded parts, as a two-axis gimbal board has. */
static const char kMap[] =
    "{\"eyes\":{\"kind\":\"joints\",\"names\":[\"pan\",\"tilt\"],\"fields\":[\"pos\"],"
    "\"hz\":50,\"command\":\"position\",\"safe\":\"hold\","
    "\"limits\":{\"pos\":[[-0.6,0.6],[-0.4,0.4]]}},"
    "\"neck\":{\"kind\":\"joints\",\"names\":[\"yaw\",\"roll\"],\"fields\":[\"pos\"],"
    "\"hz\":50,\"command\":\"position\",\"safe\":\"hold\","
    "\"limits\":{\"pos\":[[-0.6,0.6],[-0.4,0.4]]}}}";
#define PARTS 2

static uint64_t now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u;
}

static int fill_random(uint8_t* buf, size_t len) {
  while (len > 0) {
    ssize_t n = getrandom(buf, len, 0);
    if (n <= 0) return -1;
    buf += n;
    len -= (size_t)n;
  }
  return 0;
}

/* The proxy: one listener per class port, each connection paired with one to
 * the host's same class. */
#define CLASSES 4
#define MAX_PAIRS 32
typedef struct {
  int listen[CLASSES];
  uint16_t base;
  uint16_t host_port;
  int pair[MAX_PAIRS][2];
  int pairs;
  atomic_int cut;
  atomic_int hold;
  atomic_long forwarded;
  atomic_int stop;
} proxy_t;

static int listen_on(uint16_t port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (bind(fd, (struct sockaddr*)&a, sizeof a) != 0 || listen(fd, 8) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

static uint16_t port_of(int fd) {
  struct sockaddr_in a;
  socklen_t len = sizeof a;
  getsockname(fd, (struct sockaddr*)&a, &len);
  return ntohs(a.sin_port);
}

/* Four consecutive free ports. */
static int proxy_listen(proxy_t* p) {
  for (int attempt = 0; attempt < 50; ++attempt) {
    int first = listen_on(0);
    if (first < 0) return -1;
    uint16_t base = port_of(first);
    int ok = 1;
    p->listen[0] = first;
    for (int c = 1; c < CLASSES; ++c) {
      p->listen[c] = listen_on((uint16_t)(base + c));
      if (p->listen[c] < 0) {
        for (int k = 0; k < c; ++k) close(p->listen[k]);
        ok = 0;
        break;
      }
    }
    if (ok) {
      p->base = base;
      return 0;
    }
  }
  return -1;
}

/* Resets a connection: the peer's next read or write fails. */
static void reset(int fd) {
  struct linger l = {1, 0};
  setsockopt(fd, SOL_SOCKET, SO_LINGER, &l, sizeof l);
  close(fd);
}

static void drop_pair(proxy_t* p, int i) {
  reset(p->pair[i][0]);
  reset(p->pair[i][1]);
  p->pair[i][0] = p->pair[--p->pairs][0];
  p->pair[i][1] = p->pair[p->pairs][1];
}

static void* proxy_task(void* arg) {
  proxy_t* p = (proxy_t*)arg;
  static uint8_t buf[16384];
  while (!atomic_load(&p->stop)) {
    if (atomic_exchange(&p->cut, 0)) {
      while (p->pairs > 0) drop_pair(p, 0);
    }
    bool hold = atomic_load(&p->hold);
    struct pollfd fds[CLASSES + 2 * MAX_PAIRS];
    int n = 0;
    for (int c = 0; c < CLASSES; ++c) fds[n++] = (struct pollfd){p->listen[c], POLLIN, 0};
    for (int i = 0; i < p->pairs && !hold; ++i) {
      fds[n++] = (struct pollfd){p->pair[i][0], POLLIN, 0};
      fds[n++] = (struct pollfd){p->pair[i][1], POLLIN, 0};
    }
    if (poll(fds, (nfds_t)n, 20) <= 0) continue;
    for (int c = 0; c < CLASSES; ++c) {
      if (!(fds[c].revents & POLLIN)) continue;
      int board = accept(p->listen[c], NULL, NULL);
      if (board < 0) continue;
      int host = socket(AF_INET, SOCK_STREAM, 0);
      struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(p->host_port + c)};
      a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      if (p->pairs == MAX_PAIRS || connect(host, (struct sockaddr*)&a, sizeof a) != 0) {
        reset(board);
        close(host);
        continue;
      }
      p->pair[p->pairs][0] = board;
      p->pair[p->pairs][1] = host;
      p->pairs++;
    }
    /* Pairs from the poll above, newest first so dropping one keeps the
     * indices of those not yet looked at. */
    for (int i = (n - CLASSES) / 2 - 1; i >= 0; --i) {
      for (int side = 0; side < 2; ++side) {
        if (!(fds[CLASSES + 2 * i + side].revents & (POLLIN | POLLHUP | POLLERR))) continue;
        ssize_t got = read(p->pair[i][side], buf, sizeof buf);
        bool ok = got > 0;
        if (ok) atomic_fetch_add(&p->forwarded, (long)got);
        for (ssize_t off = 0; ok && off < got;) {
          ssize_t w = write(p->pair[i][1 - side], buf + off, (size_t)(got - off));
          if (w <= 0) ok = false;
          off += w;
        }
        if (!ok) {
          drop_pair(p, i);
          break;
        }
      }
    }
  }
  while (p->pairs > 0) drop_pair(p, 0);
  for (int c = 0; c < CLASSES; ++c) close(p->listen[c]);
  return NULL;
}

#ifndef __SANITIZE_ADDRESS__
/* Every allocation in the process goes through these, so the bytes held are
 * counted exactly, on every thread (glibc's own statistics see one arena),
 * with the most ever held. SOAK_CAP=<bytes> makes the heap that small: an
 * allocation past it fails, as a board's would. SOAK_SITES=1 also records
 * where each live block was allocated, and prints the sites holding the most
 * at the peak. */
extern void* __libc_malloc(size_t);
extern void* __libc_calloc(size_t, size_t);
extern void* __libc_realloc(void*, size_t);
extern void* __libc_memalign(size_t, size_t);
extern void __libc_free(void*);
static atomic_long live_bytes;
static atomic_long live_blocks;
static atomic_long peak_bytes;
static atomic_long refused_allocs;
static atomic_long total_allocs;
static long heap_cap;

#define SITE_FRAMES 16
#define SITES 8192
#define SLOTS (1u << 20)
typedef struct {
  uint64_t hash;
  void* frames[SITE_FRAMES];
  int depth;
  long live;
} site_t;
static site_t sites[SITES];
static long sites_at_peak[SITES];
static long sites_peak;
static struct {
  void* p;
  int site;
} slots[SLOTS];
#define TOMBSTONE ((void*)1)
static pthread_mutex_t sites_lock = PTHREAD_MUTEX_INITIALIZER;
static __thread int in_hook;
static int tracking;

static uint32_t slot_of(void* p) { return (uint32_t)(((uintptr_t)p >> 4) * 2654435761u) % SLOTS; }

static void track(void* p, long size) {
  if (!tracking || in_hook) return;
  in_hook = 1;
  void* frames[SITE_FRAMES + 2];
  int depth = backtrace(frames, SITE_FRAMES + 2) - 2;
  uint64_t hash = 1469598103934665603u;
  for (int i = 0; i < depth; ++i) hash = (hash ^ (uintptr_t)frames[i + 2]) * 1099511628211u;
  pthread_mutex_lock(&sites_lock);
  int site = (int)(hash % SITES);
  while (sites[site].depth != 0 && sites[site].hash != hash) site = (site + 1) % SITES;
  if (sites[site].depth == 0) {
    sites[site].hash = hash;
    sites[site].depth = depth > 0 ? depth : 1;
    memcpy(sites[site].frames, frames + 2, sizeof(void*) * (size_t)(depth > 0 ? depth : 0));
  }
  sites[site].live += size;
  uint32_t at = slot_of(p);
  while (slots[at].p != NULL && slots[at].p != TOMBSTONE) at = (at + 1) % SLOTS;
  slots[at].p = p;
  slots[at].site = site;
  long live = atomic_load(&live_bytes);
  if (live > sites_peak + 256) {
    sites_peak = live;
    for (int i = 0; i < SITES; ++i) sites_at_peak[i] = sites[i].live;
  }
  pthread_mutex_unlock(&sites_lock);
  in_hook = 0;
}

static void untrack(void* p, long size) {
  if (!tracking) return;
  pthread_mutex_lock(&sites_lock);
  for (uint32_t at = slot_of(p), n = 0; slots[at].p != NULL && n < SLOTS;
       at = (at + 1) % SLOTS, ++n) {
    if (slots[at].p == p) {
      sites[slots[at].site].live -= size;
      slots[at].p = TOMBSTONE;
      break;
    }
  }
  pthread_mutex_unlock(&sites_lock);
}

static void* main_base;
static void print_sites(void) {
  if (!tracking) return;
  Dl_info self;
  if (dladdr((void*)&print_sites, &self)) main_base = self.dli_fbase;
  printf("allocation sites holding the most at the peak (%ld bytes):\n", sites_peak);
  for (int shown = 0; shown < 30; ++shown) {
    int best = -1;
    for (int i = 0; i < SITES; ++i) {
      if (sites_at_peak[i] > 0 && (best < 0 || sites_at_peak[i] > sites_at_peak[best])) best = i;
    }
    if (best < 0) break;
    printf("  %7ld B  site", sites_at_peak[best]);
    /* Addresses in this executable (not position-independent), for
     * addr2line; those in shared libraries are left out. */
    for (int f = 0; f < sites[best].depth; ++f) {
      Dl_info info;
      uintptr_t pc = (uintptr_t)sites[best].frames[f];
      if (dladdr((void*)pc, &info) && info.dli_fbase == main_base) {
        printf(" %#lx", (unsigned long)(pc - 1));
      }
    }
    printf("\n");
    sites_at_peak[best] = 0;
  }
}

static bool over_cap(size_t n) {
  if (heap_cap > 0 && atomic_load(&live_bytes) + (long)n > heap_cap) {
    atomic_fetch_add(&refused_allocs, 1);
    return true;
  }
  return false;
}

static void* counted(void* p) {
  if (p != NULL) {
    long size = (long)malloc_usable_size(p);
    long live = atomic_fetch_add(&live_bytes, size) + size;
    atomic_fetch_add(&live_blocks, 1);
    atomic_fetch_add(&total_allocs, 1);
    for (long peak = atomic_load(&peak_bytes);
         live > peak && !atomic_compare_exchange_weak(&peak_bytes, &peak, live);) {
    }
    track(p, size);
  }
  return p;
}

static void uncounted(void* p) {
  if (p != NULL) {
    long size = (long)malloc_usable_size(p);
    untrack(p, size);
    atomic_fetch_sub(&live_bytes, size);
    atomic_fetch_sub(&live_blocks, 1);
  }
}

void* malloc(size_t n) { return over_cap(n) ? NULL : counted(__libc_malloc(n)); }
void* calloc(size_t n, size_t size) {
  return over_cap(n * size) ? NULL : counted(__libc_calloc(n, size));
}
void free(void* p) {
  uncounted(p);
  __libc_free(p);
}
void* realloc(void* p, size_t n) {
  size_t before = p != NULL ? malloc_usable_size(p) : 0;
  if (n > before && over_cap(n - before)) return NULL;
  /* Uncounted first: the block may move. */
  uncounted(p);
  void* q = __libc_realloc(p, n);
  if (q == NULL && n != 0) {
    counted(p);
    return NULL;
  }
  return counted(q);
}
void* memalign(size_t align, size_t n) {
  return over_cap(n) ? NULL : counted(__libc_memalign(align, n));
}
void* aligned_alloc(size_t align, size_t n) { return memalign(align, n); }
int posix_memalign(void** out, size_t align, size_t n) {
  void* p = memalign(align, n);
  if (p == NULL) return ENOMEM;
  *out = p;
  return 0;
}

static size_t heap_live(void) { return (size_t)atomic_load(&live_bytes); }
static void heap_watch(void) {
  void* warm[2];
  backtrace(warm, 2); /* loads the unwinder before any hook needs it */
  if (getenv("SOAK_CAP") != NULL) heap_cap = atol(getenv("SOAK_CAP"));
  tracking = getenv("SOAK_SITES") != NULL;
}
static long heap_peak(void) { return atomic_load(&peak_bytes); }
#else
/* Under AddressSanitizer, its allocator is the one that counts. */
static atomic_long live_blocks;
static atomic_long refused_allocs;
static atomic_long total_allocs;
static size_t heap_live(void) { return 0; }
static void heap_watch(void) {}
static long heap_peak(void) { return 0; }
static void print_sites(void) {}
#endif

/* `race`: one zenoh-pico session to the host through the proxy, publishing
 * from two threads as fast as they can while every link resets every 20 ms:
 * zenoh-pico's own handling of the failed link races the publishers. */
static atomic_int race_stop;
static void* race_put(void* arg) {
  const z_loaned_session_t* s = (const z_loaned_session_t*)arg;
  z_view_keyexpr_t ke;
  z_view_keyexpr_from_str(&ke, "rlmesh/soak/race");
  uint8_t payload[64] = {0};
  while (!atomic_load(&race_stop)) {
    z_owned_bytes_t bytes;
    z_bytes_copy_from_buf(&bytes, payload, sizeof payload);
    z_put_options_t opts;
    z_put_options_default(&opts);
    opts.congestion_control = Z_CONGESTION_CONTROL_BLOCK;
    z_put(s, z_loan(ke), z_move(bytes), &opts);
  }
  return NULL;
}

static int race(proxy_t* proxy, int seconds) {
  char connect[64];
  snprintf(connect, sizeof connect, "tcp/127.0.0.1:%u#tout=1000", (unsigned)proxy->base);
  z_owned_config_t config;
  z_config_default(&config);
  zp_config_insert(z_loan_mut(config), Z_CONFIG_MODE_KEY, "client");
  zp_config_insert(z_loan_mut(config), Z_CONFIG_CONNECT_KEY, connect);
  z_owned_session_t session;
  if (z_open(&session, z_move(config), NULL) != Z_OK) {
    fprintf(stderr, "race: no session\n");
    return 1;
  }
  pthread_t threads[2];
  for (int i = 0; i < 2; ++i) pthread_create(&threads[i], NULL, race_put, (void*)z_loan(session));
  uint64_t end = now_us() + (uint64_t)seconds * 1000000u;
  while (now_us() < end) {
    atomic_store(&proxy->cut, 1);
    struct timespec pause = {0, 20000000};
    nanosleep(&pause, NULL);
  }
  atomic_store(&race_stop, 1);
  for (int i = 0; i < 2; ++i) pthread_join(threads[i], NULL);
  z_drop(z_move(session));
  printf("race: survived %d s of resets under two publishers\n", seconds);
  return 0;
}

/* Every receive the robot dropped for want of memory. */
static uint64_t dropped_no_memory(const rlmesh_device_robot_t* robot) {
  uint64_t n = 0;
  for (uint8_t what = 0; what < 4; ++what) n += rlmesh_device_robot_dropped(robot, what, 5);
  return n;
}

int main(int argc, char** argv) {
  heap_watch();
  if (argc < 4) {
    fprintf(stderr,
            "usage: soak_robot <host:port#secret.hostkey> <cycles> <cut|hold|race|load> [max]\n");
    return 2;
  }
  int cycles = atoi(argv[2]);
  bool hold = strcmp(argv[3], "hold") == 0;
  long max_growth = argc > 4 ? atol(argv[4]) : 2048;
  /* The host's address, rewritten to the proxy's ports. */
  const char* colon = strrchr(argv[1], ':');
  const char* hash = strchr(argv[1], '#');
  if (colon == NULL || hash == NULL || hash < colon) {
    fprintf(stderr, "not an address with its key: %s\n", argv[1]);
    return 2;
  }
  static proxy_t proxy;
  proxy.host_port = (uint16_t)atoi(colon + 1);
  if (proxy_listen(&proxy) != 0) {
    fprintf(stderr, "no ports for the proxy\n");
    return 1;
  }
  char address[512];
  snprintf(address, sizeof address, "%.*s:%u%s", (int)(colon - argv[1]), argv[1],
           (unsigned)proxy.base, hash);
  pthread_t proxy_thread;
  pthread_create(&proxy_thread, NULL, proxy_task, &proxy);

  if (strcmp(argv[3], "race") == 0) {
    int rc = race(&proxy, cycles);
    atomic_store(&proxy.stop, 1);
    pthread_join(proxy_thread, NULL);
    printf(rc == 0 ? "ok\n" : "FAIL\n");
    return rc;
  }
  rlmesh_device_robot_config_t config = {address, NULL, NULL, 0, "eyes", kMap, 50, fill_random};
  rlmesh_device_robot_t* robot = NULL;
  if (rlmesh_device_robot_new(&config, &robot) != RLMESH_DEVICE_OK) {
    fprintf(stderr, "rlmesh_device_robot_new failed\n");
    return 1;
  }
  int parts[PARTS] = {rlmesh_device_robot_part(robot, "eyes"),
                      rlmesh_device_robot_part(robot, "neck")};
  static rlmesh_zenoh_t z;
  if (parts[0] < 0 || parts[1] < 0 || rlmesh_zenoh_open(&z, robot, now_us) != 0) {
    fprintf(stderr, "no transport\n");
    return 1;
  }

  rlmesh_device_robot_status_t status;
  float position[PARTS][2] = {{0.1f, 0.2f}, {0.1f, 0.2f}};
  uint64_t start = now_us(), next_sample = 0;
  /* Per cycle: the heap held once commands flowed again after it. */
  size_t* held = calloc((size_t)cycles + 1, sizeof *held);
  int cycle = 0, ok = 1;
  /* Commands taken before the links last broke, and when they did: a cycle
   * ends once commands flow again, attached anew or not. */
  uint64_t accepted_mark = 0, broke_at = now_us(), cycle_start = now_us();
  /* Once commands flow again: the least held over a second (what is queued
   * at any one moment varies). */
  uint64_t settle_until = 0;
  size_t least = SIZE_MAX;
  uint64_t hold_until = 0;
  const uint64_t cycle_limit = 30u * 1000000u;
  /* `load`: driven for `cycles` seconds, every link stalled for 1.5 s every
   * 10 s and then released (a weak link); the heap's peak and standing. */
  if (strcmp(argv[3], "load") == 0) {
    long budget = argc > 4 ? atol(argv[4]) : 0;
    uint64_t end = start + (uint64_t)cycles * 1000000u, warm = start + 5000000u;
    uint64_t next_stall = start + 10000000u, stall_end = 0, released = 0;
    size_t standing = SIZE_MAX;
    long driven_peak = 0, allocs_warm = 0;
    bool warmed = false;
    while (now_us() < end) {
      rlmesh_zenoh_poll(&z);
      rlmesh_device_robot_status(robot, &status);
      uint64_t now = now_us();
      static uint64_t next_trace;
      if (getenv("SOAK_TRACE") != NULL && now >= next_trace) {
        next_trace = now + 500000;
        rlmesh_zenoh_view_stats_t v;
        rlmesh_zenoh_view_stats(&z, &v);
        printf(
            "  t %5.1f echoes %llu accepted %llu superseded %u failed %u unsent %u live %zu "
            "relayed %ld\n",
            (double)(now - start) / 1e6, (unsigned long long)status.echoes,
            (unsigned long long)status.chunks_accepted, v.superseded, v.failed, v.tx_dropped,
            heap_live(), atomic_load(&proxy.forwarded));
      }
      static uint8_t last_phase = 255;
      if (status.phase != last_phase) {
        last_phase = status.phase;
        printf("  %6.1f s  phase %u  fault %u\n", (double)(now - start) / 1e6, status.phase,
               status.fault);
      }
      if (!warmed && now >= warm) {
        /* The join's own allocations are not the burst's. */
        warmed = true;
        driven_peak = (long)heap_live();
        allocs_warm = atomic_load(&total_allocs);
      }
      if (stall_end == 0 && now >= next_stall) {
        atomic_store(&proxy.hold, 1);
        stall_end = now + 1500000u;
      } else if (stall_end != 0 && now >= stall_end) {
        atomic_store(&proxy.hold, 0);
        stall_end = 0;
        released = now;
        next_stall = now + 10000000u;
      }
      if (status.phase == 4 && now >= next_sample) {
        for (int i = 0; i < PARTS; ++i) {
          rlmesh_device_tick_t tick;
          if (rlmesh_device_robot_tick(robot, (uint8_t)parts[i], now, &tick) == 1 &&
              tick.has_action && tick.value_len == sizeof position[i]) {
            memcpy(position[i], tick.value, sizeof position[i]);
          }
          rlmesh_device_robot_publish(robot, (uint8_t)parts[i], now, (const uint8_t*)position[i],
                                      sizeof position[i], now);
        }
        next_sample = now + 20000;
      }
      if (warmed) {
        size_t live = heap_live();
        if ((long)live > driven_peak) driven_peak = (long)live;
        /* Standing: the least held while running, away from a stall. */
        if (status.phase == 4 && stall_end == 0 && now > released + 3000000u && live < standing) {
          standing = live;
        }
      }
      struct timespec pause = {0, 1000000};
      nanosleep(&pause, NULL);
    }
    long peak = heap_peak();
    printf(
        "load: %llu chunks accepted, %u attaches, %llu receives dropped for memory, "
        "%ld allocations refused\n",
        (unsigned long long)status.chunks_accepted, rlmesh_device_robot_attaches(robot),
        (unsigned long long)dropped_no_memory(robot), atomic_load(&refused_allocs));
    printf("load: %.0f allocations a second while driven\n",
           (double)(atomic_load(&total_allocs) - allocs_warm) / (double)(cycles - 5));
    printf("load: heap standing %zu, peak while driven %ld (sampled), peak ever %ld, budget %ld\n",
           standing, driven_peak, peak, budget);
    print_sites();
    bool ok_load =
        status.chunks_accepted > (uint64_t)cycles * 20u && (budget == 0 || peak <= budget);
    rlmesh_zenoh_close(&z);
    rlmesh_device_robot_free(robot);
    atomic_store(&proxy.stop, 1);
    pthread_join(proxy_thread, NULL);
    printf(ok_load ? "ok\n" : "FAIL\n");
    return ok_load ? 0 : 1;
  }
  while (cycle <= cycles) {
    rlmesh_zenoh_poll(&z);
    rlmesh_device_robot_status(robot, &status);
    uint64_t now = now_us();
    if (hold_until != 0 && now >= hold_until) {
      hold_until = 0;
      atomic_store(&proxy.cut, 1);
      atomic_store(&proxy.hold, 0);
      broke_at = now;
      accepted_mark = status.chunks_accepted;
    }
    if (status.phase == 4 && now >= next_sample) {
      for (int i = 0; i < PARTS; ++i) {
        rlmesh_device_tick_t tick;
        if (rlmesh_device_robot_tick(robot, (uint8_t)parts[i], now, &tick) == 1 &&
            tick.has_action && tick.value_len == sizeof position[i]) {
          memcpy(position[i], tick.value, sizeof position[i]);
        }
        rlmesh_device_robot_publish(robot, (uint8_t)parts[i], now, (const uint8_t*)position[i],
                                    sizeof position[i], now);
#ifdef RLMESH_DEVICE_LATE_BUCKETS
        /* What a board's status line reads each period. */
        rlmesh_device_chunk_timing_t timing;
        rlmesh_device_robot_chunk_timing(robot, (uint8_t)parts[i], &timing);
#endif
      }
#ifdef RLMESH_DEVICE_LATE_BUCKETS
      rlmesh_device_datagram_stats_t dgram;
      rlmesh_device_robot_datagram_stats(robot, &dgram);
#endif
      next_sample = now + 20000;
    }
    if (status.phase == 4 && settle_until == 0 && hold_until == 0 && now >= broke_at + 1000000 &&
        status.chunks_accepted >= accepted_mark + 25) {
      settle_until = now + 1000000;
      least = SIZE_MAX;
    }
    if (settle_until != 0) {
      size_t live = heap_live();
      if (live < least) least = live;
      if (now >= settle_until) {
        held[cycle] = least;
        rlmesh_zenoh_view_stats_t views;
        rlmesh_zenoh_view_stats(&z, &views);
        printf("cycle %3d  %6.1f s  attaches %3u  heap held %8zu  (%+6ld)  transport failures %u\n",
               cycle, (double)(now - start) / 1e6, rlmesh_device_robot_attaches(robot), least,
               cycle > 0 ? (long)least - (long)held[cycle - 1] : 0L, views.failed);
        fflush(stdout);
        settle_until = 0;
        cycle++;
        cycle_start = now;
        if (cycle > cycles) break;
        if (hold) {
          atomic_store(&proxy.hold, 1);
          hold_until = now + 6000000;
        } else {
          atomic_store(&proxy.cut, 1);
          broke_at = now;
          accepted_mark = status.chunks_accepted;
        }
      }
    }
    if (now - cycle_start > cycle_limit) {
      printf("cycle %d: no re-attach taking commands within %u s (phase %u, fault %u)\n", cycle,
             (unsigned)(cycle_limit / 1000000u), status.phase, status.fault);
      ok = 0;
      break;
    }
    static uint64_t next_trace;
    if (getenv("SOAK_TRACE") != NULL && now >= next_trace) {
      next_trace = now + 1000000;
      printf(
          "  t %6.1f phase %u fault %u gen %u mode %u echoes %llu accepted %llu refused %llu (last "
          "%u) live %zu blocks %ld\n",
          (double)(now - start) / 1e6, status.phase, status.fault, status.generation, status.mode,
          (unsigned long long)status.echoes, (unsigned long long)status.chunks_accepted,
          (unsigned long long)status.chunks_refused, status.last_refusal, heap_live(),
          atomic_load(&live_blocks));
      fflush(stdout);
    }
    struct timespec pause = {0, 1000000};
    nanosleep(&pause, NULL);
  }
  if (ok && cycles >= 6) {
    /* The first cycles after the first attach, against the last. */
    size_t early = held[1] < held[2] ? held[1] : held[2];
    size_t late = held[cycles - 1] < held[cycles] ? held[cycles - 1] : held[cycles];
    long growth = (long)late - (long)early;
    printf("heap held after cycles 1-2: %zu, after the last two: %zu (%+ld, at most %+ld)\n", early,
           late, growth, max_growth);
    if (growth > max_growth) ok = 0;
  }
  rlmesh_zenoh_close(&z);
  rlmesh_device_robot_free(robot);
  atomic_store(&proxy.stop, 1);
  pthread_join(proxy_thread, NULL);
  free(held);
  printf("heap held once closed: %zu\n", heap_live());
  printf(ok ? "ok\n" : "FAIL\n");
  return ok ? 0 : 1;
}
