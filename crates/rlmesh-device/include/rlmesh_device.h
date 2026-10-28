/* The RLMesh device runtime's C ABI: the static library `rlmesh-device`
 * builds (librlmesh_device.a). The platform passes in its clock and
 * randomness; memory comes from the C library's malloc, and a panic calls
 * abort(). */

#ifndef RLMESH_DEVICE_H_
#define RLMESH_DEVICE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RLMESH_DEVICE_OK 0
/* A null argument. */
#define RLMESH_DEVICE_ERR_ARG (-1)
/* The platform's randomness failed. */
#define RLMESH_DEVICE_ERR_RNG (-2)
/* The crypto failed where it can't (a bug, or memory corrupted). */
#define RLMESH_DEVICE_ERR_CRYPTO (-3)

/* The platform's microsecond clock. */
typedef uint64_t (*rlmesh_device_now_us_fn)(void);
/* The platform's cryptographic randomness: fills `len` bytes, 0 on success. */
typedef int (*rlmesh_device_random_fn)(uint8_t* buf, size_t len);

/* The library's version: major, minor and patch, a byte each. */
uint32_t rlmesh_device_version(void);

/* Mean microseconds per operation, measured on this board. */
typedef struct rlmesh_device_timing_t {
  /* The robot's handshake msg 1: an ephemeral key and two key exchanges. */
  uint32_t handshake_start_us;
  /* Reading msg 2 and splitting the keys: two key exchanges. */
  uint32_t handshake_finish_us;
  /* Verifying a signed 1 KiB authority state. */
  uint32_t state_verify_us;
  /* Opening a signed 256-byte chunk, the default. */
  uint32_t chunk_open_sign_us;
  /* Opening an encrypted chunk, for a part declared `seal`. */
  uint32_t chunk_open_seal_us;
  /* Sealing a signed 64-byte sample, the default. */
  uint32_t sample_seal_sign_us;
} rlmesh_device_timing_t;

/* Measures what the session's crypto costs here. Takes about a second, and
 * blocks: run it before connecting, never from a control loop. */
int rlmesh_device_timing(rlmesh_device_now_us_fn now_us, rlmesh_device_random_fn random,
                         rlmesh_device_timing_t* out);

#ifdef __cplusplus
}
#endif

#endif /* RLMESH_DEVICE_H_ */
