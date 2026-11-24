# /// script
# requires-python = ">=3.10"
# dependencies = ["mujoco>=3.17,<3.18", "numpy", "pillow", "imageio[ffmpeg]"]
# ///
"""Render a recorded agent run as a video: the agent's activity on top, LIBERO below.

Inputs are the two logs a recorded run writes:
    libero_env.py --trace run/trace.jsonl   (scene XML + joint positions per step)
    agent.py      --log   run/agent.jsonl   (narration, tool calls, results)

They are merged by wall-clock time. Agent events hold the frame long enough to
read; env steps play as motion. The first policy is shown in full, later ones
as diffs against the previous version. Physics ran in LIBERO's MuJoCo 2.3.7;
frames are rendered here in MuJoCo 3.17 from the saved XML and joint positions,
which also works on macOS without OpenGL.
"""

import argparse
import difflib
import json
import textwrap
from collections.abc import Iterator
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, PANEL_H, SCENE_H = 960, 324, 540  # heights divisible by 16 for H.264
BG, FG, ACCENT, DIM = (18, 18, 24), (235, 235, 235), (120, 200, 255), (150, 150, 160)
ADDED, REMOVED, CONTEXT = (110, 220, 120), (240, 110, 110), (170, 170, 180)
CODE_TOP, CODE_LINE = 76, 18
HOLD = {"tool": 0.6, "result": 3.0, "note": 3.0, "agent": 4.0, "start": 2.0}
CODE_PAUSE = 2.0
CODE_SCROLL = 6.0


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


# Pillow's built-in font lacks the arrows and dashes the agent writes.
FONTS = (
    "Menlo.ttc",
    "DejaVuSansMono.ttf",
    "LiberationMono-Regular.ttf",
    "SourceCodePro-Regular.otf",
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in FONTS:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def policy_view(prev: str | None, code: str, version: int):
    new = code.splitlines()
    if prev is None:
        lines = [(f"{i:3d}  {line}", FG) for i, line in enumerate(new, 1)]
        return f"write_policy v1: {len(new)} lines", lines
    lines, added, removed = [], 0, 0
    for line in difflib.unified_diff(prev.splitlines(), new, lineterm="", n=1):
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            start = line.split("+")[1].split(",")[0].split(" ")[0]
            lines.append((f"··· line {start}", DIM))
        elif line.startswith("+"):
            added += 1
            lines.append((line, ADDED))
        elif line.startswith("-"):
            removed += 1
            lines.append((line, REMOVED))
        else:
            lines.append((line, CONTEXT))
    header = f"write_policy v{version}: +{added} -{removed} lines ({len(new)} total)"
    return header, lines or [("(no changes)", DIM)]


class Panel:
    def __init__(self, instruction: str):
        self.big, self.small, self.mono = font(22), font(17), font(14)
        self.cols = int((WIDTH - 40) / self.small.getlength("M"))
        self.code_cols = int((WIDTH - 40) / self.mono.getlength("M"))
        self.rows = (PANEL_H - CODE_TOP - 36) // CODE_LINE
        self.instruction = instruction
        self.action = "starting"
        self.narration = ""
        self.result = ""
        self.code: list[tuple[str, tuple]] | None = None
        self.scroll = 0
        self._prev_code: str | None = None
        self._version = 0

    def update(self, event: dict) -> None:
        kind, text = event["kind"], event["text"]
        if kind == "tool" and text.startswith("write_policy:"):
            code = text.split("\n", 1)[1] if "\n" in text else ""
            self._version += 1
            self.action, self.code = policy_view(self._prev_code, code, self._version)
            self._prev_code, self.scroll = code, 0
        elif kind == "tool":
            self.action, self.code = text, None
        elif kind == "result":
            self.result = text
        elif kind == "start":
            self.narration, self.result = text, ""
            self._prev_code, self._version = None, 0
        else:
            self.narration = text

    def draw(self, scene: np.ndarray) -> np.ndarray:
        img = Image.new("RGB", (WIDTH, PANEL_H + SCENE_H), BG)
        d = ImageDraw.Draw(img)
        d.text((20, 12), f"Task: {self.instruction}", font=self.small, fill=DIM)
        d.text((20, 40), self.action[: self.cols], font=self.big, fill=ACCENT)
        if self.code is not None:
            y = CODE_TOP
            for text, color in self.code[self.scroll : self.scroll + self.rows]:
                if len(text) > self.code_cols:
                    text = text[: self.code_cols - 1] + "…"
                d.text((20, y), text, font=self.mono, fill=color)
                y += CODE_LINE
            if len(self.code) > self.rows:
                last = min(self.scroll + self.rows, len(self.code))
                where = f"lines {self.scroll + 1}-{last} of {len(self.code)}"
                d.text((WIDTH - 200, 46), where, font=self.mono, fill=DIM)
        else:
            y = 74
            for line in textwrap.wrap(self.narration, self.cols)[:9]:
                d.text((20, y), line, font=self.small, fill=FG)
                y += 22
        if self.result:
            d.text(
                (20, PANEL_H - 30), self.result[: self.cols], font=self.small, fill=DIM
            )
        img.paste(Image.fromarray(scene), (0, PANEL_H))
        return np.asarray(img)

    def code_frames(self, scene: np.ndarray, fps: int) -> Iterator[np.ndarray]:
        assert self.code is not None, "code_frames follows a write_policy event"
        extra = max(0, len(self.code) - self.rows)
        speed = max(CODE_SCROLL, extra / 10)
        pause, scroll = int(CODE_PAUSE * fps), int(extra / speed * fps)
        for i in range(2 * pause + scroll if extra else pause + fps):
            self.scroll = min(extra, max(0, i - pause) * extra // max(1, scroll))
            yield self.draw(scene)


class Scene:
    def __init__(self, xml_path: str, camera: str):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.vis.global_.offwidth = WIDTH
        self.model.vis.global_.offheight = SCENE_H
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, SCENE_H, WIDTH)
        self.camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if self.camera < 0:
            raise SystemExit(f"camera {camera!r} not in the scene")
        # robosuite keeps collision shapes in geom group 0; show only the visual meshes.
        self.options = mujoco.MjvOption()
        self.options.geomgroup[0] = 0

    def render(self, qpos: list[float]) -> np.ndarray:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(
            self.data, camera=self.camera, scene_option=self.options
        )
        return self.renderer.render()


def frames(
    trace: list[dict],
    events: list[dict],
    scene: Scene,
    panel: Panel,
    fps: int,
    eval_stride: int,
    max_eval_episodes: int,
) -> Iterator[np.ndarray]:
    timeline = sorted(
        [(s["t"], "state", s) for s in trace] + [(e["t"], "event", e) for e in events],
        key=lambda item: item[0],
    )
    image = scene.render(trace[0]["qpos"])
    in_eval, eval_episode, eval_step = False, 0, 0
    for _, kind, item in timeline:
        if kind == "event":
            panel.update(item)
            if item["kind"] == "tool" and item["text"].startswith("evaluate("):
                in_eval, eval_episode, eval_step = True, -1, 0
            elif item["kind"] == "result":
                in_eval = False
            if panel.code is not None and item["kind"] == "tool":
                yield from panel.code_frames(image, fps)
            else:
                frame = panel.draw(image)
                for _ in range(int(HOLD[item["kind"]] * fps)):
                    yield frame
            continue
        if in_eval:
            if item["kind"] == "reset":
                eval_episode += 1
                eval_step = 0
            eval_step += 1
            if eval_episode >= max_eval_episodes or eval_step % eval_stride:
                continue
            panel.result = f"evaluation episode {eval_episode + 1}" + (
                "  -  success" if item.get("success") else ""
            )
        image = scene.render(item["qpos"])
        yield panel.draw(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a recorded agent run.")
    parser.add_argument("trace", help="trace.jsonl from libero_env.py --trace")
    parser.add_argument("log", help="agent.jsonl from agent.py --log")
    parser.add_argument("--out", default="agent_run.mp4")
    parser.add_argument("--xml", help="scene XML (default: the trace's .xml sibling)")
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--eval-stride", type=int, default=2, help="speed-up in evals")
    parser.add_argument("--max-eval-episodes", type=int, default=2)
    args = parser.parse_args()

    scene = Scene(args.xml or str(Path(args.trace).with_suffix(".xml")), args.camera)
    trace = [s for s in load(args.trace) if len(s["qpos"]) == scene.model.nq]
    if not trace:
        raise SystemExit("no trace rows match this scene XML")
    start, end = trace[0]["t"] - 60, trace[-1]["t"] + 60
    events = [e for e in load(args.log) if start <= e["t"] <= end]
    panel = Panel(next((s["instruction"] for s in trace if "instruction" in s), ""))

    count = 0
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8) as w:
        for frame in frames(
            trace,
            events,
            scene,
            panel,
            args.fps,
            args.eval_stride,
            args.max_eval_episodes,
        ):
            w.append_data(frame)
            count += 1
    print(f"wrote {args.out}: {count} frames, {count / args.fps:.0f} s")


if __name__ == "__main__":
    main()
