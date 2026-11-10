"""Each camera captures in its own process, outside the arm feedback/control loops."""

import glob
import queue
import time
from pathlib import Path
from typing import Any

from examples.bimanual_data_collection.recording import PeriodicClock, worker_signals


def list_cameras() -> None:
    for path in sorted(glob.glob("/dev/v4l/by-id/*") or glob.glob("/dev/video*")):
        print(f"USB: {path} -> {Path(path).resolve()}")
    try:
        import pyrealsense2 as rs

        for device in rs.context().query_devices():
            print(
                f"RealSense: realsense:{device.get_info(rs.camera_info.serial_number)} "
                f"({device.get_info(rs.camera_info.name)})"
            )
    except ImportError:
        print("RealSense enumeration requires pyrealsense2 (optional).")


def camera_worker(name: str, device: str, args: Any, packets: Any, stop: Any, origin_ns: int) -> None:
    import cv2

    worker_signals()
    cv2.setNumThreads(1)
    capture = None
    pipeline = None
    try:
        if device.startswith("realsense:"):
            import pyrealsense2 as rs

            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(device.split(":", 1)[1])
            config.enable_stream(
                rs.stream.color, args.camera_width, args.camera_height, rs.format.bgr8, args.camera_capture_hz
            )
            pipeline.start(config)
        else:
            source = int(device) if device.isdigit() else device
            capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not capture.isOpened():
                raise RuntimeError(f"cannot open {name} camera {device}")
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
            capture.set(cv2.CAP_PROP_FPS, args.camera_capture_hz)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        clock = PeriodicClock(args.camera_hz, origin_ns)
        ready = False
        while not stop.is_set():
            # Drain the acquisition stream continuously. Only persist frames at camera_hz.
            # Sleeping between read() calls would accumulate stale buffered USB frames.
            hardware_timestamp_ms = None
            if pipeline is not None:
                color = pipeline.wait_for_frames(timeout_ms=2000).get_color_frame()
                if not color:
                    raise RuntimeError(f"no color frame from {name}")
                import numpy as np

                frame = np.asanyarray(color.get_data())
                hardware_timestamp_ms = color.get_timestamp()
            else:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(f"camera read failed: {name}")
            monotonic_ns = time.monotonic_ns()
            timestamp_ns = time.time_ns()
            if monotonic_ns < clock.deadline_ns:
                continue
            index, scheduled_ns, missed = clock.consume(monotonic_ns)
            success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not success:
                raise RuntimeError(f"JPEG encoding failed: {name}")
            packet = {
                "kind": "camera",
                "camera": name,
                "device": device,
                "monotonic_ns": monotonic_ns,
                "timestamp_ns": timestamp_ns,
                "scheduled_monotonic_ns": scheduled_ns,
                "tick_index": index,
                "missed_periods": missed,
                "hardware_timestamp_ms": hardware_timestamp_ms,
                "timestamp_source": "host frame receive (after read/wait_for_frames)",
                "shape": list(frame.shape),
                "jpeg": encoded.tobytes(),
            }
            if not ready:
                packets.put_nowait({"kind": "ready", "name": name})
                ready = True
            packets.put_nowait(packet)
    except BaseException as error:
        if not stop.is_set():
            try:
                packets.put_nowait({"kind": "error", "name": name, "error": repr(error)})
            except queue.Full:
                pass
        raise
    finally:
        if capture is not None:
            capture.release()
        if pipeline is not None:
            pipeline.stop()
