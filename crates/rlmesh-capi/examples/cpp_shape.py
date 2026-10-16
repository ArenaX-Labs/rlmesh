"""Compare only preprocessed tokens originating in rlmesh.hpp, not its includes."""

import difflib
import pathlib
import re
import sys

header = pathlib.Path(sys.argv[1]).resolve()
source = header.read_text().splitlines()
start = source.index("#if defined(__cpp_lib_ranges)") + 1
end = source.index("#endif", start) + 1
allowed = set(range(start, end + 1))


def tokens(path):
    """Extract header tokens using compiler line markers and the sole exception."""
    own = False
    line_number = 0
    text = []
    for line in pathlib.Path(path).read_text().splitlines():
        marker = re.match(r'#\s+(\d+)\s+"([^"]+)"', line)
        if marker:
            line_number = int(marker[1])
            own = pathlib.Path(marker[2]).resolve() == header
        else:
            if own and line_number not in allowed and not line.startswith("#"):
                text.append(line)
            line_number += 1
    # Preserve literals and multi-character punctuators while ignoring whitespace.
    result = re.findall(
        r"""(?:u8|[uUL])?(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"""
        r"|[a-zA-Z_][a-zA-Z_0-9]*|[0-9]+(?:\.[0-9]+)?"
        r"|::|->|&&|\|\||==|!=|<=|>=|<<|>>|\+\+|--|\.\.\.|[^\s]",
        "\n".join(text),
    )
    assert len(result) > 10000, "line markers did not select rlmesh.hpp"
    return result


before, after = map(tokens, sys.argv[2:])
if before != after:
    delta = "\n".join(
        difflib.unified_diff(before, after, fromfile="C++17", tofile="C++20")
    )
    raise AssertionError("rlmesh.hpp must have one token-identical shape:\n" + delta)
print(
    f"one-shape OK: {len(before)} identical header tokens (excluding std::ranges specializations)"
)
