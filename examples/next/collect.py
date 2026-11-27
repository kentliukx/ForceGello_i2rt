"""Collect contact-free NEXT data using the exact LEGO teleoperation controller."""

import sys
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.bimanual_data_collection.collect_data import Args, main

if __name__ == "__main__":
    args = tyro.cli(Args, default=Args(output=Path("data/next/free_motion"), no_cameras=True))
    args.next_calibration = True
    args.feedback_mode = "none"
    args.bilateral_kp = 0.0
    main(args)
