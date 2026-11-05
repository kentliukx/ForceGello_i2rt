import signal
from typing import Any

from examples.bimanual_lead_follower import bimanual_lead_follower


class _FakeProcess:
    def __init__(self, pid: int, events: list[tuple[str, int]]) -> None:
        self.pid = pid
        self.events = events
        self.exited = False

    def poll(self) -> int | None:
        return 0 if self.exited else None

    def send_signal(self, sig: int) -> None:
        assert sig == signal.SIGINT
        self.events.append(("signal", self.pid))

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(("wait", self.pid))
        if self.pid == 1:
            signal.raise_signal(signal.SIGINT)
            signal.raise_signal(signal.SIGTERM)
        self.exited = True
        return 0


def test_shutdown_not_interrupted_by_second_sigint(monkeypatch: Any) -> None:
    events: list[tuple[str, int]] = []
    processes = [_FakeProcess(pid, events) for pid in range(1, 5)]
    monkeypatch.setattr(bimanual_lead_follower.os, "killpg", lambda pid, sig: None)
    original_handler = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    bimanual_lead_follower._stop_processes(processes)

    assert events[:4] == [("signal", pid) for pid in range(1, 5)]
    assert all(process.exited for process in processes)
    assert signal.getsignal(signal.SIGINT) == original_handler
    assert signal.getsignal(signal.SIGTERM) == original_sigterm
