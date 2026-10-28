/* The board's loop on Linux: the device robot over rlmesh_zenoh, against a
 * real session host (`rlmesh host`), with Linux's clock and randomness where a
 * board has esp_timer and its RNG. Passes once the robot has joined and had 20
 * heartbeats echoed.
 *
 *   host_robot <host:port#secret.hostkey> [seconds]
 *
 * The address is what `rlmesh host` prints, for its default session. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <time.h>

#include "rlmesh_device.h"
#include "rlmesh_zenoh.h"

static const char kMap[] =
    "{\"eyes\":{\"kind\":\"joints\",\"names\":[\"pan\",\"tilt\"],\"fields\":[\"pos\"],"
    "\"hz\":50,\"command\":\"position\",\"safe\":\"hold\","
    "\"limits\":{\"pos\":[[-0.6,0.6],[-0.4,0.4]]}}}";

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

/* RFC 4648 base32, lowercase, unpadded: 52 characters for 32 bytes. */
static int unbase32(const char* text, size_t len, uint8_t out[32]) {
  static const char kAlphabet[] = "abcdefghijklmnopqrstuvwxyz234567";
  uint32_t acc = 0, bits = 0;
  size_t at = 0;
  if (len != 52) return -1;
  for (size_t i = 0; i < len; ++i) {
    const char* p = strchr(kAlphabet, text[i]);
    if (p == NULL || text[i] == 0) return -1;
    acc = (acc << 5) | (uint32_t)(p - kAlphabet);
    bits += 5;
    if (bits >= 8) {
      bits -= 8;
      out[at++] = (uint8_t)(acc >> bits);
    }
  }
  return at == 32 ? 0 : -1;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: host_robot <host:port#secret.hostkey> [seconds]\n");
    return 2;
  }
  int seconds = argc > 2 ? atoi(argv[2]) : 15;
  const char* hash = strchr(argv[1], '#');
  const char* dot = hash ? strchr(hash, '.') : NULL;
  uint8_t secret[32], host_key[32];
  if (dot == NULL || unbase32(hash + 1, (size_t)(dot - hash - 1), secret) != 0 ||
      unbase32(dot + 1, strlen(dot + 1), host_key) != 0) {
    fprintf(stderr, "not an address with its key: %s\n", argv[1]);
    return 2;
  }
  char connect[128];
  snprintf(connect, sizeof connect, "tcp/%.*s", (int)(hash - argv[1]), argv[1]);

  rlmesh_device_robot_config_t config = {
      "local", "default", "eyes", secret, host_key, kMap, 50, fill_random,
  };
  rlmesh_device_robot_t* robot = NULL;
  int rc = rlmesh_device_robot_new(&config, &robot);
  if (rc != RLMESH_DEVICE_OK) {
    fprintf(stderr, "rlmesh_device_robot_new: %d\n", rc);
    return 1;
  }
  rlmesh_zenoh_t z;
  if (rlmesh_zenoh_open(&z, connect, robot, now_us) != 0) {
    fprintf(stderr, "no zenoh session to %s\n", connect);
    return 1;
  }
  printf("connected to %s\n", connect);

  rlmesh_device_robot_status_t status;
  uint64_t start = now_us(), joined = 0;
  uint8_t phase = 255;
  int ok = 0;
  while (now_us() - start < (uint64_t)seconds * 1000000u) {
    rlmesh_zenoh_poll(&z);
    rlmesh_device_robot_status(robot, &status);
    if (status.phase != phase) {
      phase = status.phase;
      printf("%6.3f s  phase %u  fault %u (status %u)  generation %u\n",
             (double)(now_us() - start) / 1e6, phase, status.fault, status.refused_status,
             status.generation);
      if (phase == 4 && joined == 0) joined = now_us();
    }
    if (status.phase == 4 && status.echoes >= 20) {
      ok = 1;
      break;
    }
    struct timespec pause = {0, 1000000};
    nanosleep(&pause, NULL);
  }
  printf("joined in %.3f s, %llu echoes, mode %u, epoch %llu, %u transport failures\n",
         joined ? (double)(joined - start) / 1e6 : -1.0, (unsigned long long)status.echoes,
         status.mode, (unsigned long long)status.epoch, z.failed);
  rlmesh_zenoh_close(&z);
  rlmesh_device_robot_free(robot);
  printf(ok ? "ok\n" : "FAIL\n");
  return ok ? 0 : 1;
}
