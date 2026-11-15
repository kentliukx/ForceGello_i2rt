"""Poll the teaching handle only; never construct or enable a leader motor chain."""

import threading
import time
from collections import deque
from typing import Any

from examples.bimanual_data_collection.recording import ButtonEdges


def _open_handle(channel: str) -> tuple[Any, Any]:
    from i2rt.motor_drivers.can_interface import CanInterface
    from i2rt.motor_drivers.dm_driver import PassiveEncoderReader
    from i2rt.motor_drivers.utils import ReceiveMode

    interface = CanInterface(channel=channel, receive_mode=ReceiveMode.plus_one)
    try:
        return interface, PassiveEncoderReader(interface)
    except BaseException:
        interface.close()
        raise


class HandleButtons:
    def __init__(self, channel: str, index: int = 0, timeout_s: float = 0.2) -> None:
        self.interface, self.reader = _open_handle(channel)
        self.index, self.timeout_s = index, timeout_s
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.edges = ButtonEdges()
        self.events: deque[int] = deque()
        self.error: BaseException | None = None
        try:
            initial = self.reader.read_encoder(0x50E)
            # Holding the button across startup must not start a policy without a new press.
            self.edges.previous[index] = bool(initial.io_inputs[index])
            self.received_ns = time.monotonic_ns()
        except BaseException:
            self.interface.close()
            raise
        self.thread = threading.Thread(target=self._poll, name="rdp-right-yellow-button", daemon=True)
        self.thread.start()

    def _poll(self) -> None:
        try:
            while not self.stop.wait(0.01):
                info = self.reader.read_encoder(0x50E)
                now_ns = time.monotonic_ns()
                with self.lock:
                    self.received_ns = now_ns
                    if self.edges.rising(self.index, bool(info.io_inputs[self.index]), now_ns):
                        self.events.append(now_ns)
        except BaseException as error:
            with self.lock:
                self.error = error

    def consume_toggles(self, since_ns: int) -> int:
        with self.lock:
            if self.error is not None:
                raise RuntimeError("right teaching-handle polling failed") from self.error
            if time.monotonic_ns() - self.received_ns > self.timeout_s * 1e9:
                raise RuntimeError("right teaching-handle input is stale")
            events = list(self.events)
            self.events.clear()
        return sum(timestamp_ns >= since_ns for timestamp_ns in events)

    def close(self) -> None:
        self.stop.set()
        try:
            self.thread.join(timeout=1.0)
        finally:
            self.interface.close()
