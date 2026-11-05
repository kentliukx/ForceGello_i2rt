"""Render recorded right-arm J3 teleoperation data and open the figure."""

import csv
import math
import webbrowser
from pathlib import Path
from xml.sax.saxutils import escape

import tyro


_SERIES = (
    ("leader_angle_rad", "Leader J3 angle", "#2563eb"),
    ("follower_angle_rad", "Follower J3 angle", "#ea580c"),
    ("follower_motor_torque_nm", "Follower J3 measured motor torque", "#16a34a"),
    ("leader_feedback_nm", "Leader J3 feedback torque command", "#dc2626"),
)
_J3_LOG_DIR = Path(__file__).resolve().parent / "tests" / "log"


def latest_j3_csv() -> Path:
    """Find the newest right-arm J3 recording for direct IDE execution."""
    recordings = _J3_LOG_DIR.glob("i2rt_j3_can_leader_r_*.csv")
    latest = max(recordings, key=lambda path: path.stat().st_mtime_ns, default=None)
    if latest is None:
        raise FileNotFoundError(f"No right-arm J3 CSV found in {_J3_LOG_DIR}; run with --record-j3 first")
    return latest


def _limits(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return -1.0, 1.0
    lower, upper = min(finite), max(finite)
    padding = max((upper - lower) * 0.08, 0.05)
    return lower - padding, upper + padding


def plot_j3_csv(csv_path: Path, output_path: Path | None = None) -> Path:
    """Plot angles (rad) and torque commands/estimates (Nm) on one time-aligned figure."""
    with csv_path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No J3 samples in {csv_path}")

    samples: list[dict[str, float]] = []
    for row in rows:
        try:
            sample = {key: float(row[key]) for key in ("monotonic_ns", *(item[0] for item in _SERIES))}
        except (KeyError, TypeError, ValueError):
            continue  # Ignore a final partial line after an interrupted hardware run.
        if math.isfinite(sample["monotonic_ns"]):
            samples.append(sample)
    if not samples:
        raise ValueError(f"No valid J3 samples in {csv_path}")

    t0 = samples[0]["monotonic_ns"]
    times = [(sample["monotonic_ns"] - t0) / 1e9 for sample in samples]
    duration = max(times[-1], 0.001)
    width, height = 1400, 880
    left, right = 115, width - 95
    top_angle, bottom_angle = 115, 405
    top_torque, bottom_torque = 515, 805
    angle_limits = _limits([sample[key] for sample in samples for key, _, _ in _SERIES[:2]])
    torque_limits = _limits([sample[key] for sample in samples for key, _, _ in _SERIES[2:]])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#111827}'
        ' .grid{stroke:#e5e7eb;stroke-width:1}</style>',
        '<text x="115" y="38" font-size="25" font-weight="bold">Right arm J3 teleoperation</text>',
        '<text x="115" y="64" font-size="14" fill="#4b5563">'
        'Follower motor torque is reported by CAN and includes gravity and PD response.</text>',
    ]

    for panel_top, panel_bottom, limits, title in (
        (top_angle, bottom_angle, angle_limits, "Joint angle (rad)"),
        (top_torque, bottom_torque, torque_limits, "Joint torque (Nm)"),
    ):
        parts.append(f'<text x="{left}" y="{panel_top - 24}" font-size="18" font-weight="bold">{title}</text>')
        for tick in range(6):
            y = panel_bottom - tick * (panel_bottom - panel_top) / 5
            value = limits[0] + tick * (limits[1] - limits[0]) / 5
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
            parts.append(f'<text x="{left - 12}" y="{y + 5:.1f}" font-size="13" text-anchor="end">{value:.2f}</text>')
        for tick in range(7):
            x = left + tick * (right - left) / 6
            parts.append(f'<line class="grid" x1="{x:.1f}" y1="{panel_top}" x2="{x:.1f}" y2="{panel_bottom}"/>')
        parts.append(
            f'<rect x="{left}" y="{panel_top}" width="{right - left}" height="{panel_bottom - panel_top}" '
            'fill="none" stroke="#9ca3af" stroke-width="1"/>'
        )

    for key, label, color in _SERIES:
        panel_top, panel_bottom, limits = (
            (top_angle, bottom_angle, angle_limits)
            if key.endswith("_rad")
            else (top_torque, bottom_torque, torque_limits)
        )
        segment: list[str] = []
        for sample, elapsed in zip(samples, times):
            value = sample[key]
            if not math.isfinite(value):
                if len(segment) > 1:
                    parts.append(
                        f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>'
                    )
                segment = []
                continue
            x = left + elapsed / duration * (right - left)
            y = panel_bottom - (value - limits[0]) / (limits[1] - limits[0]) * (panel_bottom - panel_top)
            segment.append(f"{x:.2f},{y:.2f}")
        if len(segment) > 1:
            parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>')

    for index, (_, label, color) in enumerate(_SERIES):
        x = left + (index % 2) * 430
        y = 432 + (index // 2) * 24
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x + 36}" y="{y + 5}" font-size="14">{escape(label)}</text>')

    for tick in range(7):
        x = left + tick * (right - left) / 6
        parts.append(f'<text x="{x:.1f}" y="830" font-size="13" text-anchor="middle">{tick * duration / 6:.1f}</text>')
    parts.append(
        f'<text x="{(left + right) / 2:.1f}" y="861" font-size="16" text-anchor="middle">'
        "Time since first sample (s)</text>"
    )
    parts.append("</svg>")

    output_path = output_path or csv_path.with_suffix(".svg")
    output_path.write_text("\n".join(parts))
    return output_path


def main(csv_path: Path | None = None, output_path: Path | None = None, show: bool = True) -> None:
    """Plot one CSV, or the newest right-arm recording when launched without arguments."""
    source = csv_path if csv_path is not None else latest_j3_csv()
    figure = plot_j3_csv(source, output_path)
    print(f"J3 figure: {figure}")
    if show and not webbrowser.open(figure.resolve().as_uri()):
        print(f"Could not open a browser automatically. Open {figure} manually.")


if __name__ == "__main__":
    tyro.cli(main)
