/* The board's loop on Linux: the device robot over rlmesh_zenoh, against a
 * real session host (rlmesh-device's test host), with Linux's clock and randomness where a
 * board has esp_timer and its RNG. Passes once the robot has joined, had 20
 * heartbeats echoed and published samples under its delivered stream key.
 *
 *   host_robot <host:port#secret.hostkey> [seconds]
 *
 * The address is what a host prints, with its key, for its default session. */

#include <stdio.h>
#include <stdlib.h>
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

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: host_robot <host:port#secret.hostkey> [seconds]\n");
    return 2;
  }
  int seconds = argc > 2 ? atoi(argv[2]) : 15;
  rlmesh_device_robot_config_t config = {argv[1], NULL, NULL, 0, "eyes", kMap, 50, fill_random};
  rlmesh_device_robot_t* robot = NULL;
  int rc = rlmesh_device_robot_new(&config, &robot);
  if (rc != RLMESH_DEVICE_OK) {
    fprintf(stderr, "rlmesh_device_robot_new: %d (not an address with its key?)\n", rc);
    return 1;
  }
  int eyes = rlmesh_device_robot_part(robot, "eyes");
  rlmesh_zenoh_t z;
  if (eyes < 0 || rlmesh_zenoh_open(&z, robot, now_us) != 0) {
    fprintf(stderr, "no transport\n");
    return 1;
  }
  uint16_t port = 0;
  const char* host = rlmesh_device_robot_host(robot, &port);
  printf("joining %s:%u\n", host, port);

  rlmesh_device_robot_status_t status;
  uint64_t start = now_us(), joined = 0, next_sample = 0;
  uint8_t phase = 255;
  unsigned published = 0;
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
    if (status.phase == 4 && now_us() >= next_sample) {
      /* A sample of the eyes, sealed under the stream key the host delivered:
       * NOT_READY until it arrives. */
      static const float kPos[2] = {0.1f, 0.2f};
      next_sample = now_us() + 20000;
      uint64_t t = now_us();
      if (rlmesh_device_robot_publish(robot, (uint8_t)eyes, t, (const uint8_t*)kPos, sizeof kPos,
                                      t) == RLMESH_DEVICE_OK) {
        published++;
      }
    }
    if (status.phase == 4 && status.echoes >= 20 && published >= 5) {
      ok = 1;
      break;
    }
    struct timespec pause = {0, 1000000};
    nanosleep(&pause, NULL);
  }
  printf("joined in %.3f s, %llu echoes, %u samples, mode %u, epoch %llu, %u transport failures\n",
         joined ? (double)(joined - start) / 1e6 : -1.0, (unsigned long long)status.echoes,
         published, status.mode, (unsigned long long)status.epoch, z.failed);
  /* A subscription or token the transport couldn't declare fails it too. */
  if (z.failed != 0) ok = 0;
  rlmesh_zenoh_close(&z);
  rlmesh_device_robot_free(robot);
  printf(ok ? "ok\n" : "FAIL\n");
  return ok ? 0 : 1;
}
