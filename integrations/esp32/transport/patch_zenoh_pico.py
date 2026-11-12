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
path holds it from the close on. A send after the clear finds no transport
and fails.

Exits non-zero when the sources are not the zenoh-pico this patches.
"""

import pathlib
import sys

root = pathlib.Path(sys.argv[1])
patched: dict[pathlib.Path, str] = {}


def patch(relative: str, edits: list[tuple[str, str]]) -> None:
    """Checks and makes `edits` in memory: nothing is written unless all apply."""
    path = root / relative
    source = path.read_text()
    if "_unlocked(" in source:
        sys.exit(f"{path}: already patched")
    for old, new in edits:
        if source.count(old) != 1:
            sys.exit(f"{path}: not the zenoh-pico this patches: update patch_zenoh_pico.py")
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
for path, source in patched.items():
    path.write_text(source)
