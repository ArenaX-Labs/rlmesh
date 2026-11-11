#!/usr/bin/env python3
"""Hold a linked RLMesh firmware's stack use to its budgets.

Reads a firmware ELF (Xtensa or RISC-V) with the toolchain's objdump: the
frame each function allocates at its entry, and the call graph, with Xtensa's
long calls resolved through their literals. Nothing runs on a board.

Paths that never return (panics, aborts) don't count. Calls through a function
pointer aren't followed, except snow's to its DH, cipher and hash: the margin
below covers the rest, which are shallow (the board's clock and randomness).
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The loop task's stack the library declares (src/platform.cpp).
LOOP_STACK = 12 * 1024
# What FreeRTOS keeps of a task's stack: an interrupted task's frame, the
# coprocessor area, the canary and the task's entry.
RTOS_RESERVE = 768
# Left free under the deepest call.
MARGIN = 1536
# The deepest call from the loop task.
LOOP_BUDGET = LOOP_STACK - RTOS_RESERVE - MARGIN
# `Robot::begin` down: joining fits Arduino's default stack on its own.
JOIN_BUDGET = 4096
# A function of the C ABI's own frame: a sketch calls these from loop().
FRAME_BUDGET = 1024

# The transport's own tasks: each entry point and the function that returns
# the stack it is created with.
TASKS = (
    ("verify_task", "rlmesh_zenoh_verify_stack"),
    ("tx_task", "rlmesh_zenoh_tx_stack"),
)

ROOT = "loopTask(void*)"
JOIN = "rlmesh::Robot::start(rlmesh_device_robot_config_t const&)"
ABI_PREFIX = "rlmesh_device_"
STACK_SIZE = "getArduinoLoopTaskStackSize()"

HEADER = re.compile(r"^([0-9a-f]+) <(.+)>:$")
INSN = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
ADDRESS = re.compile(r"\b([0-9a-f]{8})\b")
# Paths that never return: their stack use ends the program either way.
NORETURN = re.compile(
    r"^(core::panicking::|core::option::(unwrap_failed|expect_failed)|core::result::unwrap_failed"
    r"|core::slice::.*_fail|core::str::slice_error_fail|alloc::alloc::handle_alloc_error"
    r"|alloc::raw_vec::(capacity_overflow|handle_error)|__rustc::rust_begin_unwind|__assert_func"
    r"|abort$|panic_abort|esp_system_abort|__stack_chk_fail)"
)
# What snow's trait objects dispatch to.
DISPATCHED = re.compile(r"crypto::noise::.* as snow::types::|snow::resolvers::.* as snow::types::")
RETURNS = ("ret", "ret.n", "retw", "retw.n")


@dataclass
class Function:
    """One function: its frame and whom it calls."""

    name: str
    address: int
    frame: int = 0
    calls: set[int] = field(default_factory=set)
    jumps: set[int] = field(default_factory=set)
    indirect: int = 0
    returned: int | None = None


class Image:
    """A linked ELF's loadable bytes, for reading Xtensa's call literals."""

    def __init__(self, path: Path) -> None:
        """Read `path`'s program headers."""
        self.data = path.read_bytes()
        (phoff,) = struct.unpack_from("<I", self.data, 28)
        phentsize, phnum = struct.unpack_from("<HH", self.data, 42)
        self.segments = []
        for i in range(phnum):
            kind, offset, vaddr, _, filesz = struct.unpack_from(
                "<IIIII", self.data, phoff + i * phentsize
            )
            if kind == 1:
                self.segments.append((vaddr, filesz, offset))

    def word(self, address: int) -> int | None:
        """Return the 32-bit word at `address`, if the image holds it."""
        for vaddr, size, offset in self.segments:
            if vaddr <= address and address + 4 <= vaddr + size:
                return struct.unpack_from("<I", self.data, offset + address - vaddr)[0]
        return None


def parse(objdump: str, elf: Path) -> dict[int, Function]:
    """Return every function in `elf`, by address."""
    text = subprocess.run(
        [objdump, "-d", "-C", "--no-show-raw-insn", str(elf)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    image = Image(elf)
    functions: dict[int, Function] = {}
    current: Function | None = None
    literals: dict[str, int] = {}
    constants: dict[str, int] = {}
    ended = False
    for line in text.splitlines():
        header = HEADER.match(line)
        if header:
            address = int(header[1], 16)
            current = functions.setdefault(address, Function(header[2], address))
            literals, constants = {}, {}
            ended = False
            continue
        insn = INSN.match(line)
        if insn is None or current is None:
            continue
        address, mnemonic, operands = int(insn[1], 16), insn[2], insn[3]
        args = [a.strip() for a in operands.split("#")[0].split(",")]
        grows = mnemonic in ("addi", "c.addi16sp") and args[:1] == ["sp"] and int(args[-1], 0) < 0
        # Internal functions can have no symbol: the code runs on under the
        # previous one, so a prologue starts a new one.
        if (mnemonic == "entry" or (grows and ended)) and address != current.address:
            if current.frame or ended:
                name = f"{current.name}+{address - current.address:#x}"
                current = functions.setdefault(address, Function(name, address))
                literals, constants = {}, {}
        ended = mnemonic in (*RETURNS, "c.jr", "jr", "j", "c.j")
        if mnemonic in RETURNS:
            current.returned = constants.get("a2", literals.get("a2", constants.get("a0")))
        if mnemonic == "entry":
            current.frame += int(args[1], 0)
        elif grows:
            current.frame -= int(args[-1], 0)
        elif mnemonic in ("call0", "call4", "call8", "call12", "jal"):
            target = ADDRESS.search(operands)
            if target:
                current.calls.add(int(target[1], 16))
        elif mnemonic in ("callx0", "callx4", "callx8", "callx12"):
            target = literals.get(args[0])
            if target is None:
                current.indirect += 1
            else:
                current.calls.add(target)
        elif mnemonic == "jalr":
            target = ADDRESS.search(operands.partition("#")[2])
            if target:
                current.calls.add(int(target[1], 16))
            else:
                current.indirect += 1
        elif mnemonic in ("j", "c.j", "jr"):
            # A RISC-V tail call, when it lands on another function.
            target = ADDRESS.search(operands)
            if target:
                current.jumps.add(int(target[1], 16))
        elif mnemonic == "sub" and args[:2] == ["sp", "sp"]:
            # RISC-V's large frames: the size is built in a register.
            current.frame += constants.get(args[2], 0)
        elif mnemonic in ("lui", "c.lui"):
            constants[args[0]] = int(args[1], 0) << 12
        elif mnemonic in ("li", "c.li", "movi", "movi.n"):
            constants[args[0]] = int(args[1], 0)
        elif mnemonic == "slli" and args[1] in constants:
            constants[args[0]] = constants[args[1]] << int(args[2], 0)
        elif mnemonic in ("addi", "c.addi") and args[0] in constants:
            constants[args[0]] += int(args[-1], 0)
        elif mnemonic == "l32r":
            target = ADDRESS.search(operands)
            word = image.word(int(target[1], 16)) if target else None
            constants.pop(args[0], None)
            if word is not None:
                literals[args[0]] = word
        elif args and args[0] in literals:
            del literals[args[0]]
    dispatched = {f.address for f in functions.values() if DISPATCHED.search(f.name)}
    for function in functions.values():
        function.calls |= {j for j in function.jumps if j in functions and j != function.address}
        if function.indirect and function.name.startswith(("<snow::", "snow::")):
            function.calls |= dispatched
    return functions


def deepest(functions: dict[int, Function], root: int) -> tuple[int, list[Function]]:
    """Return the heaviest call chain from `root`, each function counted once."""
    best: dict[int, tuple[int, list[int]]] = {}
    active: set[int] = set()

    def visit(address: int) -> tuple[int, list[int]]:
        if address in best:
            return best[address]
        function = functions.get(address)
        if function is None or address in active or NORETURN.match(function.name):
            return (0, [])
        active.add(address)
        total: int = 0
        chain: list[int] = []
        for callee in sorted(function.calls):
            depth, path = visit(callee)
            if depth > total:
                total, chain = depth, path
        active.discard(address)
        best[address] = (total + function.frame, [address, *chain])
        return best[address]

    sys.setrecursionlimit(100000)
    total, chain = visit(root)
    return total, [functions[a] for a in chain]


def show(functions: dict[int, Function], root: Function) -> None:
    """Print the deepest chain from `root`."""
    total, chain = deepest(functions, root.address)
    print(f"{root.name}: {total} bytes", file=sys.stderr)
    for f in chain:
        print(f"  {f.frame:6d} {f.name}", file=sys.stderr)


def check(functions: dict[int, Function], elf: Path) -> list[str]:
    """Return what is over budget in `elf`."""
    by_name = {f.name: f for f in functions.values()}
    problems = []
    stack = by_name.get(STACK_SIZE)
    if stack is None or stack.returned != LOOP_STACK:
        got = None if stack is None else stack.returned
        problems.append(f"the loop task's stack is {got} bytes, not the declared {LOOP_STACK}")
    root = by_name.get(ROOT)
    if root is None:
        return [*problems, f"no {ROOT} in {elf}"]
    total, _ = deepest(functions, root.address)
    print(
        f"{elf.name}: deepest call {total} of {LOOP_STACK} bytes, "
        f"{LOOP_STACK - RTOS_RESERVE - total} free after the RTOS's {RTOS_RESERVE}"
    )
    if total > LOOP_BUDGET:
        show(functions, root)
        problems.append(f"the deepest call takes {total} bytes, over its {LOOP_BUDGET}")
    join = by_name.get(JOIN)
    if join is not None:
        total, _ = deepest(functions, join.address)
        print(f"{elf.name}: Robot::start down {total} bytes")
        if total > JOIN_BUDGET:
            show(functions, join)
            problems.append(f"joining takes {total} bytes, over its {JOIN_BUDGET}")
    for entry, size_fn in TASKS:
        task = by_name.get(entry)
        size = by_name.get(size_fn)
        if task is None and size is None:
            # A sketch without the transport (the Timing example).
            continue
        if task is None or size is None or size.returned is None:
            problems.append(f"no {entry} or {size_fn} in {elf}")
            continue
        budget = size.returned - RTOS_RESERVE - MARGIN
        total, _ = deepest(functions, task.address)
        print(f"{elf.name}: {entry} deepest call {total} of {size.returned} bytes")
        if total > budget:
            show(functions, task)
            problems.append(f"{entry} takes {total} bytes, over its {budget}")
    for function in functions.values():
        if function.name.startswith(ABI_PREFIX) and function.frame > FRAME_BUDGET:
            problems.append(
                f"{function.name}'s frame is {function.frame} bytes, over its {FRAME_BUDGET}"
            )
    return problems


def main() -> int:
    """Check one firmware, or print the chains from the functions named."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objdump", required=True)
    parser.add_argument("--elf", required=True, type=Path)
    parser.add_argument("--root", action="append", default=[], help="print this chain instead")
    args = parser.parse_args()
    functions = parse(args.objdump, args.elf)
    if args.root:
        for f in functions.values():
            if any(f.name == n or f.name.startswith(n + "(") for n in args.root):
                show(functions, f)
        return 0
    problems = check(functions, args.elf)
    for problem in problems:
        print(f"{args.elf}: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
