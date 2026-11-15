from types import SimpleNamespace
from typing import Any

import pytest

from examples.rdp_policy import buttons


class ScriptStop:
    def __init__(self, count: int) -> None:
        self.remaining = count

    def wait(self, duration: float) -> bool:
        if self.remaining == 0:
            return True
        self.remaining -= 1
        return False

    def set(self) -> None:
        self.remaining = 0


def test_handle_edges_ignore_startup_hold_debounce_and_discard_pre_ready_presses(monkeypatch: Any) -> None:
    now = [1_000_000_000]
    states = iter([True, True, False, True, True, False, True])
    closed, opened, requests = [], [], []

    def read(encoder_id: int) -> Any:
        requests.append(encoder_id)
        now[0] += 100_000_000
        return SimpleNamespace(io_inputs=[next(states), False])

    def open_handle(channel: str) -> tuple[Any, Any]:
        opened.append(channel)
        return SimpleNamespace(close=lambda: closed.append(True)), SimpleNamespace(read_encoder=read)

    monkeypatch.setattr(buttons, "_open_handle", open_handle)
    monkeypatch.setattr(buttons.time, "monotonic_ns", lambda: now[0])
    monkeypatch.setattr(
        buttons.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None, join=lambda **kw: None),
    )
    handle = buttons.HandleButtons("can_leader_r")
    handle.stop = ScriptStop(6)
    try:
        handle._poll()
        assert len(handle.events) == 2  # initial/held pressed state did not create an edge
        assert handle.consume_toggles(1_450_000_000) == 1  # discard press before control became ready
        assert handle.consume_toggles(0) == 0  # each edge is consumed once
        now[0] += 300_000_000
        with pytest.raises(RuntimeError, match="stale"):
            handle.consume_toggles(0)
    finally:
        handle.close()
    assert opened == ["can_leader_r"] and closed == [True]
    assert requests == [0x50E] * 7


def test_handle_read_error_is_reported_to_control_loop(monkeypatch: Any) -> None:
    calls = []

    def read(encoder_id: int) -> Any:
        if calls:
            raise OSError("CAN disconnected")
        calls.append(encoder_id)
        return SimpleNamespace(io_inputs=[False, False])

    monkeypatch.setattr(
        buttons,
        "_open_handle",
        lambda channel: (
            SimpleNamespace(close=lambda: None),
            SimpleNamespace(read_encoder=read),
        ),
    )
    monkeypatch.setattr(
        buttons.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None, join=lambda **kw: None),
    )
    handle = buttons.HandleButtons("can_leader_r")
    handle.stop = ScriptStop(1)
    try:
        handle._poll()
        with pytest.raises(RuntimeError, match="polling failed"):
            handle.consume_toggles(0)
    finally:
        handle.close()
