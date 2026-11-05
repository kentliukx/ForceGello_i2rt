import threading
import time
import unittest
from types import SimpleNamespace

from i2rt.motor_drivers.dm_driver import DMChainCanInterface


class CanCloseTests(unittest.TestCase):
    def test_close_waits_for_control_thread_before_closing_socket(self) -> None:
        driver = DMChainCanInterface.__new__(DMChainCanInterface)
        driver.running = True
        driver.channel = "can_test"
        entered = threading.Event()
        exited = threading.Event()

        def control_loop() -> None:
            entered.set()
            while driver.running:
                time.sleep(0.001)
            exited.set()

        def close_socket() -> None:
            self.assertTrue(exited.is_set())

        driver.motor_interface = SimpleNamespace(close=close_socket)
        driver._control_thread = threading.Thread(target=control_loop)
        driver._control_thread.start()
        self.assertTrue(entered.wait(timeout=1.0))

        driver.close()

        self.assertFalse(driver._control_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
