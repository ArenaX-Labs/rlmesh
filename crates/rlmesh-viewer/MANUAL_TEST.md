# Manual test — terminal viewer (native graphics)

Ground truth: `src/terminal.rs`. The protocol bytes can't be unit-tested meaningfully
(they need a real terminal), so verify them by hand here. Kitty/iTerm2 byte formats
and `q=2` semantics were spec-verified against the kitty graphics protocol, kitty's
own C source, and the iTerm2 inline-images docs.

## Trigger it

Run an eval with the viewer attached and a **real TTY on stdout** (no pipe, no redirect):

```python
rlmesh.run(model, env, view="terminal")   # or view=True
```

`Terminal::new` returns `None` when stdout is not a TTY, so the viewer silently
no-ops under a pipe/redirect — run it directly in a terminal window. It takes over
the alt-screen; your scrollback returns intact on quit.

## Per-terminal expectations

| Terminal                       | Detected as | Expected rendering                               |
| ------------------------------ | ----------- | ------------------------------------------------ |
| Ghostty, Kitty                 | `Kitty`     | Crisp Kitty-graphics PNG, aspect-preserved       |
| iTerm2, WezTerm, mintty ≥3.4   | `Iterm`     | Crisp iTerm2 inline image, aspect-preserved      |
| plain xterm, Terminal.app, SSH | `None`      | Truecolor ANSI **half-blocks** (▀, 2 px/cell)    |
| inside tmux / screen           | `None`      | Half-blocks (graphics auto-disabled — see below) |

Detection (`detect_graphics`) is **env-only** — no terminal query is written, so
nothing competes with the key thread for stdin. Kitty path: `TERM` contains
`kitty`/`ghostty`, `TERM_PROGRAM=ghostty`, or `KITTY_WINDOW_ID` set. iTerm path:
`TERM_PROGRAM` is `iTerm.app`/`WezTerm`/`mintty`. tmux/screen (via `$TMUX`/`$STY` or
`TERM=screen*`/`tmux*`) force half-blocks. Everything else → half-blocks.

## Confirm which path you got / debug

```bash
echo $TERM $TERM_PROGRAM $KITTY_WINDOW_ID $TMUX
```

- **Half-blocks where you expected an image** = a _detection_ miss (terminal not in
  the allowlist, or you're inside tmux/screen). Harmless — half-blocks render fine
  anywhere truecolor works.
- **Garbage / torn region instead of an image** = a _framing/transport_ problem: a
  graphics escape reached something that didn't honor it. Most often a multiplexer
  without passthrough — but tmux/screen are auto-detected, so this should only happen
  on an exotic setup. Capture `$TERM`/`$TERM_PROGRAM` and report it.

## Window size & resize

The frame is re-fitted to the live terminal size **every frame** (`terminal::size()`
is queried in `draw()`), so the image tracks the window. Resize during an **active
run** reflows on the next frame (≤ 1/fps). Note: redraw is driven by fed frames, not
the resize event — a **paused** eval won't reflow until the next step.

## Interaction (must all work)

| Key                           | Action           |
| ----------------------------- | ---------------- |
| Tab / Space / →               | next source      |
| ←                             | previous source  |
| 1–9                           | jump to source N |
| q / Q / Esc / Ctrl-C / Ctrl-D | quit             |

These are reliable because (a) detection never writes a query to stdin, and (b) Kitty
transmits with `q=2`, which suppresses **both** OK and error acks — so the terminal
sends nothing back and the key thread stays the sole stdin reader. Verify: spam Tab to
cycle sources, jump with digit keys, and confirm `q` exits cleanly and restores
scrollback. The footer shows the source selector plus the `step / R±x.xx` HUD.

## Known limitations / watch-for

- **Per-frame flicker on Kitty:** each frame issues `a=d,d=A` (delete-all + free)
  before redrawing, so fast sources can show mild flicker on some Kitty builds. If it's
  distracting, switch to placement-replace instead of delete-all.
- **Old mintty (<3.4):** grouped with the iTerm path but predates inline-image support —
  it would print the escape as visible garbage. Newer mintty is fine.
- **Conservative allowlist:** Konsole, Warp, and Sixel-only terminals (foot, contour,
  xterm) aren't detected and fall back to half-blocks. Safe, just not pixel-crisp there.
- **Long-side cap:** graphics frames are capped to 1024 px on the long side before PNG
  encode (the terminal scales into the cells), so very large source frames look slightly
  softened rather than native-res.
