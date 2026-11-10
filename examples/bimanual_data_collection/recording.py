"""Streaming, multi-rate trajectory storage. No robot or camera drivers are imported here."""

import json
import os
import pickle
import signal
import time
from pathlib import Path
from typing import Any, BinaryIO, Iterator


def request_shutdown(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def worker_signals() -> None:
    # The parent owns Ctrl+C and sets the shared stop event. SIGTERM still unwinds arm cleanup.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, request_shutdown)


class PeriodicClock:
    """Absolute monotonic deadlines; an overrun skips slots instead of sending a burst."""

    def __init__(self, hz: float, start_ns: int) -> None:
        if hz <= 0:
            raise ValueError("frequency must be positive")
        self.period_ns = 1e9 / hz
        self.start_ns = start_ns
        self.index = 0

    @property
    def deadline_ns(self) -> int:
        return self.start_ns + round(self.index * self.period_ns)

    def consume(self, now_ns: int) -> tuple[int, int, int]:
        if now_ns < self.deadline_ns:
            raise ValueError("deadline has not arrived")
        index = max(self.index, int((now_ns - self.start_ns) / self.period_ns))
        missed = index - self.index
        deadline = self.start_ns + round(index * self.period_ns)
        self.index = index + 1
        return index, deadline, missed


class ButtonEdges:
    """Independent rising edges; holding either button cannot repeatedly toggle it."""

    def __init__(self, debounce_s: float = 0.15) -> None:
        self.previous: dict[int, bool] = {}
        self.last_edge: dict[int, int] = {}
        self.debounce_ns = round(debounce_s * 1e9)

    def rising(self, index: int, pressed: bool, now_ns: int) -> bool:
        previous = self.previous.get(index, False)
        self.previous[index] = pressed
        if pressed and not previous and now_ns - self.last_edge.get(index, -(10**18)) >= self.debounce_ns:
            self.last_edge[index] = now_ns
            return True
        return False


def read_robot_stream(path: Path) -> Iterator[dict[str, Any]]:
    """Read the sequence of pickle records (not a single giant in-memory list)."""
    with path.open("rb") as stream:
        while True:
            try:
                yield pickle.load(stream)
            except EOFError:
                return


class EpisodeWriter:
    """One directory per trajectory, bounded memory, original samples and clocks preserved."""

    def __init__(self, root: Path, start_ns: int, metadata: dict[str, Any]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        number = max((int(p.name.split(".")[0]) for p in root.iterdir() if p.name.split(".")[0].isdigit()), default=0)
        while True:
            number += 1
            self.path = root / f"{number}.partial"
            try:
                self.path.mkdir()
                break
            except FileExistsError:
                continue
        self.start_ns = start_ns
        self.end_ns: int | None = None
        self.streams: dict[str, BinaryIO] = {
            side: (self.path / f"robot_{side}.pkl").open("wb") for side in ("left", "right")
        }
        self.camera_index = (self.path / "cameras.jsonl").open("w")
        self.counts = dict.fromkeys(("left", "right", "left_wrist", "right_wrist", "external"), 0)
        self.metadata = {
            **metadata,
            "schema": "forcegello.multirate.v1",
            "start_monotonic_ns": start_ns,
            "created_timestamp_ns": time.time_ns(),
            "status": "recording",
            "image_color": "RGB after JPEG decoding and BGR-to-RGB conversion",
            "clocks": "int64 host monotonic ns for alignment; Unix ns for wall time; no hardware camera sync",
            "action_semantics": "command.pos/vel are the targets passed to follower.command_joint_state; "
            "joint angles rad, velocities rad/s, gripper position normalized [0,1]; "
            "motor-side clipping/force limiting can further modify execution",
            "effort_semantics": "joint_eff_external = measured actuator effort - modeled gravity; "
            "includes tracking, friction and inertial effects; not a six-axis force sensor",
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        temporary = self.path / "metadata.json.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(self.path / "metadata.json")

    def accepts(self, timestamp_ns: int) -> bool:
        return self.start_ns <= timestamp_ns and (self.end_ns is None or timestamp_ns < self.end_ns)

    def append(self, packet: dict[str, Any]) -> None:
        if not self.accepts(packet["monotonic_ns"]):
            return
        if packet["kind"] == "robot":
            side = packet["side"]
            pickle.dump(packet, self.streams[side], protocol=pickle.HIGHEST_PROTOCOL)
            self.counts[side] += 1
        elif packet["kind"] == "camera":
            name = packet["camera"]
            directory = self.path / name
            directory.mkdir(exist_ok=True)
            filename = f"{name}/{self.counts[name]:08d}.jpg"
            (self.path / filename).write_bytes(packet["jpeg"])
            entry = {key: value for key, value in packet.items() if key != "jpeg"}
            entry["path"] = filename
            self.camera_index.write(json.dumps(entry) + "\n")
            self.counts[name] += 1

    def flush(self) -> None:
        for stream in self.streams.values():
            stream.flush()
        self.camera_index.flush()

    def finish(self, end_ns: int, status: str = "complete", reason: str | None = None) -> Path:
        self.end_ns = end_ns
        self.flush()
        for stream in (*self.streams.values(), self.camera_index):
            os.fsync(stream.fileno())
            stream.close()
        self.metadata.update(end_monotonic_ns=end_ns, status=status, counts=self.counts, reason=reason)
        self._write_metadata()
        final = self.path.with_suffix("")
        self.path.rename(final)
        self.path = final
        return final
