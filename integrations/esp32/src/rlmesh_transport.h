// Private, opaque bridge to the precompiled transport. Arduino's sketch
// preprocessor must never see zenoh-pico's C headers (it uses -CC).
#ifndef RLMESH_TRANSPORT_H_
#define RLMESH_TRANSPORT_H_

#include "rlmesh_device.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct rlmesh_zenoh_t rlmesh_zenoh_t;
size_t rlmesh_zenoh_size(void);
uint32_t rlmesh_zenoh_failures(rlmesh_zenoh_t* z);
int rlmesh_zenoh_open(rlmesh_zenoh_t* z, rlmesh_device_robot_t* robot,
                      rlmesh_device_now_us_fn now_us);
int rlmesh_zenoh_poll(rlmesh_zenoh_t* z);
void rlmesh_zenoh_on_input(rlmesh_zenoh_t* z, void (*wake)(void*), void* arg);
void rlmesh_zenoh_close(rlmesh_zenoh_t* z);
#ifdef __cplusplus
}
#endif

#endif
