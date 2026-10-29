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

/* A config field is invalid: an address without its key, a name that isn't
 * one, a map the host would refuse, a zero heartbeat period. */
#define RLMESH_DEVICE_ERR_CONFIG (-4)
/* Not running yet, or the stream's key hasn't arrived: nothing was sent. */
#define RLMESH_DEVICE_ERR_NOT_READY (-5)
/* A value not in its part's dense layout. */
#define RLMESH_DEVICE_ERR_INVALID (-6)

/* The board as a robot in a session. It holds no network: the platform's
 * transport (zenoh-pico) carries out the actions rlmesh_device_robot_next hands
 * out, and passes back replies and samples. Parts are named by their index in
 * the robot's map (rlmesh_device_robot_part). */
typedef struct rlmesh_device_robot rlmesh_device_robot_t;

typedef struct rlmesh_device_robot_config_t {
  /* The address the host printed, with its key:
   * "{host}:{port}[/{session}]#{secret}.{host key}[/{name}]". NULL for a
   * provisioned board, which waits in its home's lobby (record's first
   * address) as its own device, and joins the runs it is assigned. */
  const char* address;
  /* A provisioned board's device key (32 bytes) and stored record (tether spec
   * §14), with address NULL. */
  const uint8_t* device_key;
  const uint8_t* record;
  size_t record_len;
  /* This robot's name; NULL takes the address's. */
  const char* name;
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
/* Drop every subscription and token: the robot moves to another session (the
 * lobby, or a run) on the same host. */
#define RLMESH_DEVICE_RESET 5

/* Traffic classes: the host listens on its address's port plus the class. */
#define RLMESH_DEVICE_CLASSES 4

/* An action, borrowed from the robot until the next call on it. */
typedef struct rlmesh_device_action_t {
  uint8_t kind;
  /* The traffic class: 0 control, 1 realtime, 2 data, 3 bulk. */
  uint8_t class_;
  /* A subscription's id, for rlmesh_device_robot_sample. */
  uint8_t sub;
  uint8_t has_payload;
  /* A query's id, for rlmesh_device_robot_reply and _query_done. */
  uint32_t query_id;
  /* How long a query waits for replies, in milliseconds. */
  uint32_t timeout_ms;
  /* NUL-terminated. */
  const char* key;
  const uint8_t* payload;
  size_t payload_len;
} rlmesh_device_action_t;

typedef struct rlmesh_device_robot_status_t {
  /* 0 reading the host, 1 handshake, 2 attach, 3 declaring its parts,
   * 4 running, 5 waiting to retry, 6 waiting in the lobby for a run,
   * 7 stopped (another process of this device took its name). */
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
  /* Command chunks the edges accepted. */
  uint64_t chunks_accepted;
  /* The e-stop latch: 0 clear, 1 this robot's own stop, 2 a member's signal. */
  uint8_t estop;
  /* Whether it is in a run it joined from the lobby. */
  uint8_t in_run;
} rlmesh_device_robot_status_t;

/* No part, in a tick. */
#define RLMESH_DEVICE_NO_PART 0xFF

/* Edge modes. */
#define RLMESH_DEVICE_IDLE 0
#define RLMESH_DEVICE_DRIVING 1
#define RLMESH_DEVICE_HOLD 2
#define RLMESH_DEVICE_SAFE 3
#define RLMESH_DEVICE_ESTOP 4

/* One control tick of a command group, borrowed from the robot until the next
 * call on it. */
typedef struct rlmesh_device_tick_t {
  /* The edge mode, RLMESH_DEVICE_IDLE ... _ESTOP. */
  uint8_t mode;
  /* The part it concerns (the active or last-driven member of the group). */
  uint8_t part;
  /* Whether value is an action to dispatch to part. */
  uint8_t has_action;
  /* A handover between members starts: switch the control mode from
   * handover_from to handover_to, then call
   * rlmesh_device_robot_handover_done. */
  uint8_t handover;
  uint8_t handover_from;
  uint8_t handover_to;
  /* The action, in the part's dense command layout. */
  const uint8_t* value;
  size_t value_len;
  /* The declared safe behaviour to run now ("hold", "stop", ...), or NULL. */
  const char* safe;
} rlmesh_device_tick_t;

/* A new robot, joining on the first rlmesh_device_robot_next. The config's
 * strings are copied. */
int rlmesh_device_robot_new(const rlmesh_device_robot_config_t* config,
                            rlmesh_device_robot_t** out);
/* Frees a robot; NULL is a no-op. */
void rlmesh_device_robot_free(rlmesh_device_robot_t* robot);
/* The host to connect to (owned by the robot), and in *port its first port:
 * class i listens on port + i. */
const char* rlmesh_device_robot_host(const rlmesh_device_robot_t* robot, uint16_t* port);
/* The index of the part at path in the robot's map, or -1. */
int rlmesh_device_robot_part(const rlmesh_device_robot_t* robot, const char* path);
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
/* A sample on subscription sub, received on key (key_len bytes, not
 * NUL-terminated: an e-stop signal names its sender in it). */
int rlmesh_device_robot_sample(rlmesh_device_robot_t* robot, uint8_t sub, const char* key,
                               size_t key_len, const uint8_t* data, size_t len, uint64_t now_us);
/* Latches this robot's e-stop for its own stop (an e-stop pin; reason may be
 * NULL): every part stops at once, with or without a host, and the host is
 * told. Only rlmesh_device_robot_clear_estop clears it. */
int rlmesh_device_robot_estop(rlmesh_device_robot_t* robot, const char* reason, uint64_t now_us);
/* Clears the e-stop latch, whoever set it: the only way to. Call it only from a
 * local action on the board, with the e-stop released. The host ends in ESTOP,
 * unlatched; leaving that still needs disarm and arm. */
int rlmesh_device_robot_clear_estop(rlmesh_device_robot_t* robot);
/* One control tick of part's command group: 1 with *out written when there is
 * something to do (an action, a safe behaviour, a handover), 0 when there is
 * nothing, RLMESH_DEVICE_ERR_NOT_READY before the robot runs or for a part
 * that takes no commands. Call it at the part's control rate; it publishes the
 * group's applied samples. */
int rlmesh_device_robot_tick(rlmesh_device_robot_t* robot, uint8_t part, uint64_t now_us,
                             rlmesh_device_tick_t* out);
/* The robot finished the handover of part's group: it may dispatch again. */
int rlmesh_device_robot_handover_done(rlmesh_device_robot_t* robot, uint8_t part);
/* Publishes a sample of part: value in its dense layout, captured at
 * capture_us (this board's clock). RLMESH_DEVICE_ERR_NOT_READY before the
 * robot runs or its key arrived, _ERR_INVALID for a value not in the part's
 * layout, _ERR_ARG for no such part. */
int rlmesh_device_robot_publish(rlmesh_device_robot_t* robot, uint8_t part, uint64_t capture_us,
                                const uint8_t* value, size_t len, uint64_t now_us);
int rlmesh_device_robot_status(const rlmesh_device_robot_t* robot,
                               rlmesh_device_robot_status_t* out);

/* USB provisioning (tether spec §14): the board keeps a device key it makes on
 * first boot (32 random bytes from a cryptographic RNG) and the record the
 * daemon sends; the responder answers the daemon over the serial port. */
typedef struct rlmesh_device_provision rlmesh_device_provision_t;

#define RLMESH_DEVICE_PROVISION_NONE 0
/* Write *data to the serial port. */
#define RLMESH_DEVICE_PROVISION_REPLY 1
/* Store *data as the record (one atomic write), then call
 * rlmesh_device_provision_stored. */
#define RLMESH_DEVICE_PROVISION_STORE 2

/* The public key of a device key: what the daemon enrolls. */
int rlmesh_device_key_public(const uint8_t* device_key, uint8_t* public_key);
/* Whether record (len bytes) is a valid provisioning record: 1 or 0. */
int rlmesh_device_record_valid(const uint8_t* record, size_t len);
/* A responder for robot name with device_key (32 bytes), holding record (NULL
 * or 0 when not provisioned). arch as in the tether's HELLO: 3 RISC-V, 4
 * Xtensa. */
int rlmesh_device_provision_new(const char* name, const uint8_t* device_key, uint8_t arch,
                                const uint8_t* record, size_t record_len,
                                rlmesh_device_provision_t** out);
void rlmesh_device_provision_free(rlmesh_device_provision_t* p);
/* One byte from the serial port: RLMESH_DEVICE_PROVISION_NONE, or _REPLY /
 * _STORE with *data and *len set (valid until the next call on p). */
int rlmesh_device_provision_push(rlmesh_device_provision_t* p, uint8_t byte, const uint8_t** data,
                                 size_t* len);
/* The record from _STORE was stored (ok non-zero) or not: *data and *len are
 * the reply to write. */
int rlmesh_device_provision_stored(rlmesh_device_provision_t* p, int ok, const uint8_t** data,
                                   size_t* len);

#ifdef __cplusplus
}
#endif

#endif /* RLMESH_DEVICE_H_ */
