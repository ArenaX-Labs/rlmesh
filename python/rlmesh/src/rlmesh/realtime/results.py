"""What the realtime calls return, typed (Experimental).

Every result is a plain ``dict`` at run time; these ``TypedDict`` classes
say which keys it has and what each holds, so an editor completes them and
a type checker catches a misspelt key. Durations are in milliseconds unless
a key says otherwise.
"""

from __future__ import annotations

from typing import Literal, TypedDict

__all__ = [
    "ChunkTiming",
    "ClockRefusal",
    "DatagramEnd",
    "DatagramRelay",
    "EdgeClockRefusal",
    "EdgeStats",
    "Event",
    "HandBack",
    "Health",
    "Held",
    "HostLink",
    "InferencePendingEvent",
    "InputLostEvent",
    "Link",
    "LinkRun",
    "NoticeEvent",
    "ParkEvent",
    "PathTime",
    "PolicyStats",
    "Published",
    "RateDriftEvent",
    "ReboundEvent",
]


class ChunkTiming(TypedDict):
    """How one commanded part's chunks arrived at its edge."""

    #: Chunks received, whatever became of them.
    received: int
    #: Chunks the edge accepted.
    accepted: int
    #: Gaps in their seqs under one lease.
    lost: int
    #: Accepted chunks by lateness on arrival against their first slot, in
    #: buckets of ``[0, 1)``, ``[1, 2)``, ``[2, 4)`` ... ``[64, inf)`` ms.
    late_ms: list[int]
    late_max_ms: float
    #: The longest time between two accepted chunks.
    gap_max_ms: float


class DatagramEnd(TypedDict):
    """One end of a board's datagram path."""

    active: bool
    sent: int
    received: int
    lost: int
    reordered: int
    #: Datagrams that did not open (wrong key, replayed, damaged).
    unopened: int
    #: Moves back to TCP.
    fallbacks: int


class PathTime(TypedDict):
    """A board's seconds on each path since it attached, and the host's moves between them."""

    udp_s: float
    tcp_s: float
    fallbacks: int
    returns: int


class DatagramRelay(TypedDict):
    """How long the host took to relay a board's chunks, subscriber to socket."""

    #: Counts in buckets of ``[0, 1)``, ``[1, 2)``, ... ``[64, inf)`` ms.
    ms: list[int]
    max_ms: float


class HostLink(TypedDict):
    """One participant's link as the hosting process measured it (:meth:`Host.health`)."""

    participant: str
    heartbeats: int
    rtt_ms: float | None
    last_heartbeat_ms: float | None
    #: The path a robot's realtime parts take: ``"tcp"`` or ``"udp"``.
    data_path: str
    chunks_accepted: int
    #: Per commanded part, by ``robot:path``.
    chunk_timing: dict[str, ChunkTiming]
    datagram_board: DatagramEnd | None
    datagram_host: DatagramEnd | None
    path_time: PathTime | None
    datagram_relay: DatagramRelay | None


class ClockRefusal(TypedDict):
    """A part refusing chunks while a clock's error is above its bound."""

    #: ``robot:path``.
    part: str
    #: Whose clock.
    participant: str
    error_ms: float
    bound_ms: float


class Published(TypedDict):
    """Samples of one part a participant publishes."""

    published: int
    coalesced: int


class LinkRun(TypedDict):
    """A bound policy's counters, from its latest report to the host."""

    starts: int
    completions: int
    #: Skipped starts, by reason.
    skipped: dict[str, int]
    chunks_sent: int
    #: The inference in flight's age (``None``: none).
    pending_ms: float | None
    latency_p50_ms: float | None
    latency_p99_ms: float | None
    drifting: bool


class Link(TypedDict):
    """One participant's link, as the host measures it and publishes it."""

    heartbeats: int
    #: The latest heartbeat round trip (``None`` before one).
    rtt_ms: float | None
    jitter_ms: float
    heartbeat_gap_max_ms: float
    last_heartbeat_age_ms: float
    #: Its clock mapping, host minus participant, and the mapping's error
    #: bound (``None`` until mapped).
    offset_ms: float | None
    offset_error_ms: float | None
    #: What it reports of itself: ``ok``, ``degraded``, ``unhealthy`` or
    #: ``unknown``.
    health: str
    #: A robot's newest chunk seq accepted and applied (0 for none).
    last_chunk_accepted: int
    last_chunk_applied: int
    queue_depth: int
    #: Chunks refused since it attached, and by reason.
    rejected: int
    rejected_by: dict[str, int]
    clock_refused: list[ClockRefusal]
    #: Per part it publishes, ``robot:path``.
    published: dict[str, Published]
    #: A bound policy's counters, else ``None``.
    run: LinkRun | None
    #: The newest seq it received per part it reads.
    last_seq_seen: dict[str, int]


class Health(TypedDict):
    """The session's live link health, as the host last published it."""

    epoch: int
    #: The host's clock when it published.
    host_ns: int
    #: Each attached participant's link, by its name.
    links: dict[str, Link]


class Held(TypedDict):
    """A robot held until re-armed by name."""

    robot: str
    reason: str
    detail: str
    #: Since when, session time.
    at_ns: int
    message: str


class EdgeClockRefusal(TypedDict):
    """An edge refusing a participant's chunks for its clock's error."""

    part: str
    participant: str
    error_ms: float
    bound_ms: float
    message: str


class EdgeStats(TypedDict):
    """The counters of the edge serving one commanded part (:meth:`Robot.edge_stats`)."""

    accepted: int
    rejected: int
    #: Refusals by reason (only the reasons seen).
    rejected_by: dict[str, int]
    clock_refused: EdgeClockRefusal | None
    clamped: int
    handovers: int
    safe_reports: int
    #: Per mode (``driving``, ``hold``, ``safe``, ...), ``(ticks, entries)``.
    modes: dict[str, tuple[int, int]]
    received: int
    lost: int
    late_ms: list[int]
    late_max_ms: float
    gap_max_ms: float


class PolicyStats(TypedDict):
    """A policy's run counters (:meth:`Policy.stats`, and what :meth:`Policy.run` returns)."""

    #: The input whose publication the starts are phased just after, and how far.
    aligned_to: str | None
    phase_offset_ms: float
    starts: int
    completions: int
    #: Skipped starts, by reason (every reason, zero or not).
    skipped: dict[str, int]
    chunks_sent: int
    #: Starts a shadow run predicted without a lease.
    shadow_predictions: int
    send_errors: int
    latency_p50_ms: float | None
    latency_p99_ms: float | None
    latency_max_ms: float | None
    drifting: bool
    #: Per leased part refusing this policy's chunks for its clock, the message.
    clock_refused: list[str]
    #: What the policy waits for before it can command, else ``None``.
    waiting: str | None
    #: Parts a higher class took, ``robot:path``.
    parked: list[str]


class HandBack(TypedDict):
    """One policy an operator's ``release(to="model")`` handed parts back to."""

    #: Its name.
    to: str
    #: ``robot:path``.
    parts: list[str]
    #: Whether it held them again in time.
    taken: bool


class RateDriftEvent(TypedDict):
    """The policy's start rate drifted from its declared rate, or came back."""

    kind: Literal["rate_drift"]
    message: str
    measured_hz: float
    expected_hz: float
    cause: str
    drifting: bool


class InferencePendingEvent(TypedDict):
    """``predict`` has been running for a while."""

    kind: Literal["inference_pending"]
    message: str
    pending_s: float


class NoticeEvent(TypedDict):
    """A change the message says all of."""

    kind: Literal[
        "robot_held", "clock_refused", "waiting_to_arm", "shadowing", "commanding"
    ]
    message: str


class InputLostEvent(TypedDict):
    """An input's publisher went away."""

    kind: Literal["input_lost"]
    message: str
    part: str


class ReboundEvent(TypedDict):
    """The policy bound again to parts that came back."""

    kind: Literal["rebound"]
    message: str
    parts: list[str]


class ParkEvent(TypedDict):
    """A part a higher class took (``parked``), or handed back (``handed_back``)."""

    kind: Literal["parked", "handed_back"]
    message: str
    part: str
    #: Who took it, or handed it back.
    by: str


#: One of :meth:`Policy.events`, told apart by ``kind``.
Event = (
    RateDriftEvent
    | InferencePendingEvent
    | NoticeEvent
    | InputLostEvent
    | ReboundEvent
    | ParkEvent
)
