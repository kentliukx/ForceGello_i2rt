from unittest.mock import patch

from examples.bimanual_lead_follower import bimanual_lead_follower


def test_launcher_passes_feedback_mode_and_gains_to_leader() -> None:
    with patch.object(bimanual_lead_follower.subprocess, "Popen") as popen:
        bimanual_lead_follower.launch_gello_process(
            can_channel="can_leader_l",
            gripper="yam_teaching_handle",
            mode="leader",
            server_port=1235,
            bilateral_kp=2.0,
            bilateral_alpha=0.5,
            bilateral_beta=3.0,
            feedback_mode="torque",
            trace_latency=True,
            record_j3=True,
        )

    cmd = popen.call_args.args[0]
    assert cmd[cmd.index("--feedback-mode") + 1] == "torque"
    assert cmd[cmd.index("--bilateral-kp") + 1] == "2.0"
    assert cmd[cmd.index("--bilateral-alpha") + 1] == "0.5"
    assert cmd[cmd.index("--bilateral-beta") + 1] == "3.0"
    assert "--trace-latency" in cmd
    assert "--record-j3" in cmd
