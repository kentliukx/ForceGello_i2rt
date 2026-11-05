import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

from examples.bimanual_lead_follower.plot_j3 import plot_j3_csv


def test_plot_contains_both_angles_and_both_torques() -> None:
    with TemporaryDirectory() as directory:
        csv_path = Path(directory) / "j3.csv"
        with csv_path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(
                ("monotonic_ns", "leader_angle_rad", "follower_angle_rad", "follower_pd_est_nm", "leader_feedback_nm")
            )
            writer.writerow((1_000_000_000, 0.1, 0.2, 0.3, -0.4))
            writer.writerow((1_010_000_000, 0.2, 0.3, 0.4, -0.5))

        svg_path = plot_j3_csv(csv_path)
        root = ElementTree.parse(svg_path).getroot()
        polylines = root.findall("{http://www.w3.org/2000/svg}polyline")
        assert len(polylines) == 4
        assert "Joint angle (rad)" in svg_path.read_text()
        assert "Joint torque (Nm)" in svg_path.read_text()
