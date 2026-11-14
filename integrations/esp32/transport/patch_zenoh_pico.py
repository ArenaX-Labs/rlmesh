#!/usr/bin/env python3
"""Patch zenoh-pico's sources under ROOT (the directory holding its `src/`)
for the boards' transport, and for the Linux build that tests it.

When zenoh-pico notices a link failing (a reset, a read error, a lease run
out), its executor task closes the transport and clears it: it destroys the
transport's send mutex and frees its buffers and link. Upstream does that
under the session's transport mutex, but nothing that sends takes it, so a
`z_put`, `z_get` or declaration on another task at that moment writes into
freed memory, and on ESP-IDF uses a freed semaphore: heap corruption. Here
every send and batch start take that (recursive) mutex too, and the failure
path holds it from the close on, as does the reconnect's when replaying its
declarations fails. A send after the clear finds no transport and fails, as
it does after a transport that failed to allocate its buffers (which left
itself marked unicast).

A TCP link's connect waits at most 3 s (the handshake's own bound), and a
send at most 1.5 s (SO_SNDTIMEO): a send that times out, or goes out in part,
leaves the stream mid-message, so the link is shut down at once, its read
fails, and zenoh-pico's failure path closes it as for any lost link. 1.5 s is
past the host's presence grace (1 s: by then it has counted the board gone
anyway) and short of its stall guard (2.5 s: the board gives up on a link
before the host resets it). Both ports get it: lwIP's (the boards') and
POSIX's (the Linux build).

The ESP32 port's TCP link gets the receive timeout its other ports have (one
task per session reads, so a silent peer would otherwise block its lease
checks, query timeouts and closing for ever); every task it creates (its
sessions' and the transport's) runs on the core the build sets with
`-DRLMESH_TASK_CORE` (-1, the default, no affinity, as FreeRTOS's own
`xTaskCreate`; 0 or 1 pins them); and a joined task is deleted only once it
has suspended itself, with its event group (one leaked per closed session),
whichever cores the two run on. A task tells its joiner it is done just
before it suspends: deleted in between, while it still runs (on another core),
FreeRTOS leaves its stack (5 KB for a session's) for that core's idle task
to free, which under load may not run for a long time; deleted suspended, it
is freed at once.

Every port counts the tasks it started and those it ended
(`zp_rlmesh_tasks`), so a board can show that none outlives its session.

A subscription or queryable lookup that fails to allocate its result
reports success with no result, which the caller then dereferences: it now
reports the failure, and the sample is dropped.

Exits non-zero when the sources are not the zenoh-pico this patches.
"""

import pathlib
import sys

root = pathlib.Path(sys.argv[1])
patched: dict[pathlib.Path, str] = {}


def patch(relative: str, edits: list[tuple[str, str]], port: bool = False) -> None:
    """Checks and makes `edits` in memory: nothing is written unless all apply.
    A `port`'s file (one platform's) is left alone where the tree has none:
    the boards' library holds only the ESP32 port's."""
    path = root / relative
    if port and not path.exists():
        return
    source = path.read_text()
    if "_unlocked(" in source or "_z_rlmesh_" in source:
        sys.exit(f"{path}: already patched")
    for old, new in edits:
        if source.count(old) != 1:
            sys.exit(
                f"{path}: not the zenoh-pico this patches: update patch_zenoh_pico.py"
            )
        source = source.replace(old, new)
    patched[path] = source


def locked(name: str, params: str, args: str) -> tuple[str, str]:
    """`name` renamed to `name_unlocked`, behind a wrapper taking the lock."""
    head = f"z_result_t {name}({params}) {{\n"
    return (
        head,
        f"static z_result_t {name}_unlocked({params});\n"
        f"z_result_t {name}({params}) {{\n"
        "    _z_session_transport_mutex_lock(zn);\n"
        f"    z_result_t ret = {name}_unlocked({args});\n"
        "    _z_session_transport_mutex_unlock(zn);\n"
        "    return ret;\n"
        "}\n\n"
        f"static z_result_t {name}_unlocked({params}) {{\n",
    )


patch(
    "src/transport/common/tx.c",
    [
        (
            '#include "zenoh-pico/transport/common/tx.h"\n',
            '#include "zenoh-pico/transport/common/tx.h"\n\n#include "zenoh-pico/session/utils.h"\n',
        ),
        locked(
            "_z_send_n_msg",
            "_z_session_t *zn, const _z_network_message_t *z_msg, z_reliability_t reliability,\n"
            "                         z_congestion_control_t cong_ctrl, void *peer",
            "zn, z_msg, reliability, cong_ctrl, peer",
        ),
        locked(
            "_z_send_n_batch",
            "_z_session_t *zn, z_congestion_control_t cong_ctrl",
            "zn, cong_ctrl",
        ),
    ],
)
patch(
    "src/transport/unicast/lease.c",
    [
        (
            "    _z_unicast_transport_close(ztu, _Z_CLOSE_EXPIRED);\n"
            "    _z_session_transport_mutex_lock(zs);\n",
            "    _z_session_transport_mutex_lock(zs);\n"
            "    _z_unicast_transport_close(ztu, _Z_CLOSE_EXPIRED);\n",
        ),
    ],
)
patch(
    "src/api/api.c",
    [
        (
            "    _z_session_t *session = _Z_RC_IN_VAL(zs);\n"
            "    return _z_transport_start_batching(&session->_tp);\n",
            "    _z_session_t *session = _Z_RC_IN_VAL(zs);\n"
            "    _z_session_transport_mutex_lock(session);\n"
            "    z_result_t ret = _z_transport_start_batching(&session->_tp);\n"
            "    _z_session_transport_mutex_unlock(session);\n"
            "    return ret;\n",
        ),
    ],
)
patch(
    "src/net/session.c",
    [
        (
            "                _z_transport_clear(&s->_tp);\n",
            "                _z_session_transport_mutex_lock(s);\n"
            "                _z_transport_clear(&s->_tp);\n"
            "                _z_session_transport_mutex_unlock(s);\n",
        ),
    ],
)
patch(
    "src/transport/unicast/transport.c",
    [
        (
            "        _z_wbuf_clear(&ztu->_common._wbuf);\n"
            "        _z_zbuf_clear(&ztu->_common._zbuf);\n"
            "    }\n"
            "    return ret;\n",
            "        _z_wbuf_clear(&ztu->_common._wbuf);\n"
            "        _z_zbuf_clear(&ztu->_common._zbuf);\n"
            "        zt->_type = _Z_TRANSPORT_NONE;\n"
            "    }\n"
            "    return ret;\n",
        ),
    ],
)
CORE = """
/* rlmesh: the core every task runs on (patch_zenoh_pico.py): -1, the default,
 * no affinity; 0 or 1 pins them there. */
#ifndef RLMESH_TASK_CORE
#define RLMESH_TASK_CORE -1
#endif
#if RLMESH_TASK_CORE < -1 || RLMESH_TASK_CORE >= portNUM_PROCESSORS
#error "RLMESH_TASK_CORE is -1 (no affinity) or a core of this chip"
#endif
#define Z_RLMESH_TASK_CORE (RLMESH_TASK_CORE < 0 ? tskNO_AFFINITY : RLMESH_TASK_CORE)
"""
COUNTERS = """
/* rlmesh: tasks started and ended, for zp_rlmesh_tasks (patch_zenoh_pico.py). */
static uint32_t _z_rlmesh_started;
static uint32_t _z_rlmesh_ended;
static void _z_rlmesh_count(uint32_t *n) { __atomic_fetch_add(n, 1, __ATOMIC_RELAXED); }
void zp_rlmesh_tasks(uint32_t *started, uint32_t *ended) {
    *started = __atomic_load_n(&_z_rlmesh_started, __ATOMIC_RELAXED);
    *ended = __atomic_load_n(&_z_rlmesh_ended, __ATOMIC_RELAXED);
}
"""
patch(
    "src/system/arduino/esp32/system.c",
    [
        (
            "/*------------------ Task ------------------*/\n",
            COUNTERS + CORE + "/*------------------ Task ------------------*/\n",
        ),
        (
            "xTaskCreateStatic((void *)z_task_wrapper, attr->name, attr->stack_depth, z_arg, attr->priority,\n"
            "                                         attr->stack_buffer, attr->task_buffer);\n",
            "xTaskCreateStaticPinnedToCore((void *)z_task_wrapper, attr->name, attr->stack_depth, z_arg,\n"
            "                                         attr->priority, attr->stack_buffer, attr->task_buffer,\n"
            "                                         Z_RLMESH_TASK_CORE);\n",
        ),
        (
            "xTaskCreate((void *)z_task_wrapper, attr->name, attr->stack_depth, z_arg, attr->priority, "
            "&task->handle) !=\n            pdPASS) {\n            z_free(z_arg);\n",
            "xTaskCreatePinnedToCore((void *)z_task_wrapper, attr->name, attr->stack_depth, z_arg, "
            "attr->priority, &task->handle, Z_RLMESH_TASK_CORE) !=\n"
            "            pdPASS) {\n            vEventGroupDelete(task->join_event);\n            z_free(z_arg);\n",
        ),
        (
            "#endif /* SUPPORT_STATIC_ALLOCATION */\n\n    return _Z_RES_OK;\n}\n",
            "#endif /* SUPPORT_STATIC_ALLOCATION */\n\n    _z_rlmesh_count(&_z_rlmesh_started);\n"
            "    return _Z_RES_OK;\n}\n",
        ),
        (
            "    return _z_task_cancel(task);\n}\n\nz_result_t _z_task_detach",
            "    while (eTaskGetState(task->handle) != eSuspended) {\n"
            "        vTaskDelay(1);\n"
            "    }\n"
            "    z_result_t ret = _z_task_cancel(task);\n"
            "    vEventGroupDelete(task->join_event);\n"
            "    task->join_event = NULL;\n"
            "    return ret;\n}\n\nz_result_t _z_task_detach",
        ),
        (
            "    vTaskDelete(task->handle);\n    task->handle = NULL;\n",
            "    vTaskDelete(task->handle);\n    task->handle = NULL;\n    _z_rlmesh_count(&_z_rlmesh_ended);\n",
        ),
    ],
    port=True,
)
patch(
    "src/system/unix/system.c",
    [
        (
            "/*------------------ Task ------------------*/\n",
            COUNTERS + "/*------------------ Task ------------------*/\n",
        ),
        (
            "    _Z_CHECK_SYS_ERR(pthread_create(task, attr, fun, arg));\n",
            "    int created = pthread_create(task, attr, fun, arg);\n"
            "    if (created == 0) {\n"
            "        _z_rlmesh_count(&_z_rlmesh_started);\n"
            "    }\n"
            "    _Z_CHECK_SYS_ERR(created);\n",
        ),
        (
            "z_result_t _z_task_join(_z_task_t *task) { _Z_CHECK_SYS_ERR(pthread_join(*task, NULL)); }\n",
            "z_result_t _z_task_join(_z_task_t *task) {\n"
            "    int joined = pthread_join(*task, NULL);\n"
            "    if (joined == 0) {\n"
            "        _z_rlmesh_count(&_z_rlmesh_ended);\n"
            "    }\n"
            "    _Z_CHECK_SYS_ERR(joined);\n"
            "}\n",
        ),
        (
            "z_result_t _z_task_detach(_z_task_t *task) { _Z_CHECK_SYS_ERR(pthread_detach(*task)); }\n",
            "z_result_t _z_task_detach(_z_task_t *task) {\n"
            "    int detached = pthread_detach(*task);\n"
            "    if (detached == 0) {\n"
            "        _z_rlmesh_count(&_z_rlmesh_ended);\n"
            "    }\n"
            "    _Z_CHECK_SYS_ERR(detached);\n"
            "}\n",
        ),
    ],
    port=True,
)
for relative, drop in (
    ("src/session/subscription.c", "_z_subscription_rc_svec_rc_drop(sub_infos);"),
    ("src/session/queryable.c", "_z_session_queryable_rc_svec_rc_drop(qle_infos);"),
):
    patch(
        relative,
        [
            (
                f"        {drop}\n    }}\n    return _Z_RES_OK;\n",
                f"        {drop}\n    }}\n    return ret;\n",
            )
        ],
    )
BOUNDED = """
/* rlmesh: a link's connect and sends, bounded in time (patch_zenoh_pico.py). */
#include <errno.h>
#include <fcntl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>

#define Z_RLMESH_CONNECT_MS 3000
#define Z_RLMESH_SEND_MS 1500

/* connect() within Z_RLMESH_CONNECT_MS, then sends bounded by SO_SNDTIMEO. */
static int _z_rlmesh_connect(int fd, const struct sockaddr *addr, socklen_t len) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        return -1;
    }
    int ret = connect(fd, addr, len);
    if (ret < 0 && errno == EINPROGRESS) {
        fd_set writable;
        FD_ZERO(&writable);
        FD_SET(fd, &writable);
        struct timeval wait = {Z_RLMESH_CONNECT_MS / 1000, (Z_RLMESH_CONNECT_MS % 1000) * 1000};
        int err = -1;
        socklen_t err_len = sizeof(err);
        ret = (select(fd + 1, NULL, &writable, NULL, &wait) == 1 &&
               getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &err_len) == 0 && err == 0)
                  ? 0
                  : -1;
    }
    if (fcntl(fd, F_SETFL, flags) < 0) {
        ret = -1;
    }
    struct timeval send_wait = {Z_RLMESH_SEND_MS / 1000, (Z_RLMESH_SEND_MS % 1000) * 1000};
    if (ret == 0 && setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &send_wait, sizeof(send_wait)) < 0) {
        ret = -1;
    }
    return ret;
}

/* What send() returned for `len` bytes: a link that took less (timed out)
 * is mid-message, so it is shut down, and its read fails. */
static size_t _z_rlmesh_sent(int fd, ssize_t wb, size_t len) {
    if (wb < 0 || (size_t)wb < len) {
        shutdown(fd, SHUT_RDWR);
        return wb < 0 ? SIZE_MAX : (size_t)wb;
    }
    return (size_t)wb;
}
"""
CONNECT = "connect(sock->_fd, it->ai_addr, it->ai_addrlen) < 0"
BOUNDED_CONNECT = "_z_rlmesh_connect(sock->_fd, it->ai_addr, it->ai_addrlen) < 0"
LOGGING = '#include "zenoh-pico/utils/logging.h"\n'
patch(
    "src/link/transport/tcp/tcp_esp32.c",
    [
        (LOGGING, LOGGING + BOUNDED),
        (
            "    _ZP_UNUSED(tout);\n    z_result_t ret = _Z_RES_OK;\n",
            "    z_result_t ret = _Z_RES_OK;\n",
        ),
        (
            "    if (sock->_fd != -1) {\n        int optflag = 1;\n",
            "    if (sock->_fd != -1) {\n"
            "        z_time_t tv;\n"
            "        tv.tv_sec = tout / (uint32_t)1000;\n"
            "        tv.tv_usec = (tout % (uint32_t)1000) * (uint32_t)1000;\n"
            "        if (setsockopt(sock->_fd, SOL_SOCKET, SO_RCVTIMEO, (char *)&tv, sizeof(tv)) < 0) {\n"
            "            _Z_ERROR_LOG(_Z_ERR_GENERIC);\n"
            "            ret = _Z_ERR_GENERIC;\n"
            "        }\n\n"
            "        int optflag = 1;\n",
        ),
        (CONNECT, BOUNDED_CONNECT),
        (
            "    ssize_t wb = send(sock._fd, ptr, len, 0);\n"
            "    if (wb < (ssize_t)0) {\n"
            "        return SIZE_MAX;\n"
            "    }\n\n"
            "    return (size_t)wb;\n",
            "    return _z_rlmesh_sent(sock._fd, send(sock._fd, ptr, len, 0), len);\n",
        ),
    ],
    port=True,
)
patch(
    "src/link/transport/tcp/tcp_posix.c",
    [
        (LOGGING, LOGGING + BOUNDED),
        (CONNECT, BOUNDED_CONNECT),
        (
            "    return (size_t)send(sock._fd, ptr, len, MSG_NOSIGNAL);\n",
            "    return _z_rlmesh_sent(sock._fd, send(sock._fd, ptr, len, MSG_NOSIGNAL), len);\n",
        ),
        (
            "    return (size_t)send(sock._fd, ptr, len, 0);\n",
            "    return _z_rlmesh_sent(sock._fd, send(sock._fd, ptr, len, 0), len);\n",
        ),
    ],
    port=True,
)
for path, source in patched.items():
    path.write_text(source)
