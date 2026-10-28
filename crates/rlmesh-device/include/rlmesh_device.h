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

/* A config field is invalid: a tenant or name that isn't one, a zero heartbeat
 * period. */
#define RLMESH_DEVICE_ERR_CONFIG (-4)

/* The board as a robot in a session. It holds no network: the platform's
 * transport (zenoh-pico) carries out the actions rlmesh_device_robot_next hands
 * out, and passes back replies and samples. */
typedef struct rlmesh_device_robot rlmesh_device_robot_t;

typedef struct rlmesh_device_robot_config_t {
  /* The session's tenant ("local" for an embedded host). */
  const char* tenant;
  /* The session's name ("default") or its id, a hyphenated UUID. */
  const char* session;
  /* This robot's name. */
  const char* name;
  /* The address's local secret and the host's key, 32 bytes each. */
  const uint8_t* secret;
  const uint8_t* host_key;
  /* The robot's map, as authored JSON. */
  const char* robot_json;
  /* The heartbeat period; 0 is refused. */
  uint32_t heartbeat_ms;
  /* Cryptographic randomness, for the handshake. */
  rlmesh_device_random_fn random;
} rlmesh_device_robot_config_t;

/* Action kinds. */
#define RLMESH_DEVICE_SUBSCRIBE 1 /* subscribe to key; pass samples as `sub` */
#define RLMESH_DEVICE_TOKEN 2     /* declare a liveliness token on key */
#define RLMESH_DEVICE_QUERY 3     /* query key, with payload if has_payload */
#define RLMESH_DEVICE_PUBLISH 4   /* publish payload on key */

/* How long a query waits for replies. */
#define RLMESH_DEVICE_QUERY_MS 2000

/* An action, borrowed from the robot until the next call on it. */
typedef struct rlmesh_device_action_t {
  uint8_t kind;
  /* A subscription's id, for rlmesh_device_robot_sample. */
  uint8_t sub;
  uint8_t has_payload;
  /* A query's id, for rlmesh_device_robot_reply and _query_done. */
  uint32_t query_id;
  /* NUL-terminated. */
  const char* key;
  const uint8_t* payload;
  size_t payload_len;
} rlmesh_device_action_t;

typedef struct rlmesh_device_robot_status_t {
  /* 0 reading the host, 1 handshake, 2 attach, 3 declaring its parts,
   * 4 running, 5 waiting to retry. */
  uint8_t phase;
  /* Why it last started over: 0 never, 1 no reply, 2 the host's state
   * unsigned, 3 the handshake failed (another host's key), 4 refused
   * (refused_status, the proto's CtlStatus number), 5 the host lost,
   * 6 internal. */
  uint8_t fault;
  uint8_t refused_status;
  /* 0 none yet, 1 disarmed, 2 armed, 3 active, 4 safe, 5 e-stop. */
  uint8_t mode;
  /* This attach's generation. */
  uint32_t generation;
  /* The authority epoch. */
  uint64_t epoch;
  /* Heartbeat echoes received. */
  uint64_t echoes;
} rlmesh_device_robot_status_t;

/* A new robot, joining on the first rlmesh_device_robot_next. The config's
 * strings are copied. */
int rlmesh_device_robot_new(const rlmesh_device_robot_config_t* config,
                            rlmesh_device_robot_t** out);
/* Frees a robot; NULL is a no-op. */
void rlmesh_device_robot_free(rlmesh_device_robot_t* robot);
/* Runs the timers and hands out the next action: 1 with *out written, 0 when
 * there is none for now. Call it until it returns 0, every loop. */
int rlmesh_device_robot_next(rlmesh_device_robot_t* robot, uint64_t now_us,
                             rlmesh_device_action_t* out);
/* One reply to query query_id. */
int rlmesh_device_robot_reply(rlmesh_device_robot_t* robot, uint32_t query_id, const uint8_t* data,
                              size_t len, uint64_t now_us);
/* Query query_id is over: every reply has arrived, or it timed out. */
int rlmesh_device_robot_query_done(rlmesh_device_robot_t* robot, uint32_t query_id,
                                   uint64_t now_us);
/* A sample on subscription sub. */
int rlmesh_device_robot_sample(rlmesh_device_robot_t* robot, uint8_t sub, const uint8_t* data,
                               size_t len, uint64_t now_us);
int rlmesh_device_robot_status(const rlmesh_device_robot_t* robot,
                               rlmesh_device_robot_status_t* out);

#ifdef __cplusplus
}
#endif

#endif /* RLMESH_DEVICE_H_ */
