"""Run a command under a machine-wide lock so heavy jobs queue instead of overlapping.

usage: machine_lock.py <name> <command> [args...]

Several worktrees (and the agents working in them) share one host. Two full
cargo builds of this workspace at once can exhaust its memory, so heavy commands
take the same named lock and run one at a time. The command is exec'd with the
lock held: its exit status and signals are its own, and the lock is released
when it exits. Locks live under RLMESH_LOCK_DIR (default
~/.local/state/rlmesh/locks), which every checkout on the host shares.
"""

import fcntl
import os
import re
import sys
from pathlib import Path


def lock_dir() -> Path:
    configured = os.environ.get("RLMESH_LOCK_DIR")
    return Path(configured) if configured else Path.home() / ".local/state/rlmesh/locks"


def main() -> None:
    if len(sys.argv) < 3 or not re.fullmatch(r"[a-z0-9-]+", sys.argv[1]):
        sys.exit("usage: machine_lock.py <name> <command> [args...]")
    name, command = sys.argv[1], sys.argv[2:]

    locks = lock_dir()
    locks.mkdir(parents=True, exist_ok=True)
    holder = locks / f"{name}.holder"
    fd = os.open(locks / f"{name}.lock", os.O_RDWR | os.O_CREAT, 0o600)

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        try:
            current = holder.read_text().strip() or "another job"
        except OSError:
            current = "another job"
        print(f"waiting for the {name} lock, held by: {current}", file=sys.stderr, flush=True)
        fcntl.flock(fd, fcntl.LOCK_EX)

    holder.write_text(f"{' '.join(command)} (in {os.getcwd()}, pid {os.getpid()})\n")
    # flock belongs to the open file description; keeping the fd across exec
    # hands the lock to the command for its whole lifetime.
    os.set_inheritable(fd, True)
    try:
        os.execvp(command[0], command)
    except OSError as error:
        sys.exit(f"machine_lock.py: cannot run {command[0]}: {error}")


if __name__ == "__main__":
    main()
