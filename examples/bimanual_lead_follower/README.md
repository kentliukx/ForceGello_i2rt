# Bimanual Lead-Follower System Setup

For 60 Hz robot data, three 20 Hz cameras, and independent white-button trajectory recording,
see [双臂数据采集](../bimanual_data_collection/README.md). Yellow-button teleoperation and force feedback are retained.

This tutorial demonstrates how to set up a bimanual lead-follower system. This tutorial assumes you've followed the main README and finished setting up the environment.

## Required Hardware

- **Two YAM Leader arms**
- **Two YAM Follower arms**

## Hardware Setup

1. Install two YAM leader and two YAM follower arms safely on the tabletop
2. Connect the arms to power
3. Set up the CANable devices
4. **Important:** Do NOT plug any CANable devices into your computer yet

## Setup Different CAN IDs for Leader and Follower Arms

**⚠️ Critical Step:** Plug in **one CAN device at a time** to avoid conflicts.

1. Follow the instructions in [set-persistent-can-ids.md](../../docs/guides/set-persistent-can-ids.md) to configure each arm:
   - **Left leader arm:** `can_leader_l`
   - **Right leader arm:** `can_leader_r`
   - **Left follower arm:** `can_follower_l`
   - **Right follower arm:** `can_follower_r`

> **⚠️ This step is critical!** Our code uses these CAN IDs to match arms together.

## Verification

After setting up all CAN IDs:

1. Plug in all 4 CAN devices
2. Run `ip a` to verify the CAN devices are mapped correctly
3. You should see output similar to:

```bash
i2rt@ioheart-dev03:~/i2rt$
.........
.........
9: can_follower_r: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP group default qlen 10
    link/can
10: can_follower_l: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP group default qlen 10
    link/can
11: can_leader_r: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP group default qlen 10
    link/can
12: can_leader_l: <NOARP,UP,LOWER_UP,ECHO> mtu 16 qdisc pfifo_fast state UP group default qlen 10
    link/can
```

## Launch the System

Once all CAN devices are properly configured and connected, launch the system:
Assume you've installed the python virtual environment and run `source .venv/bin/activate`

```bash
python examples/bimanual_lead_follower/bimanual_lead_follower.py --feedback-mode angle --bilateral-kp 0.05
```

Run this from the repository root. The flags are passed to both leader processes. `feedback_mode` defaults to `angle`, `bilateral_kp` defaults to `0.0`, `bilateral_alpha` defaults to `0.0` (no speed attenuation), and `bilateral_beta` defaults to `1.0`. Larger `beta` reduces the angle/torque feedback more strongly. All gains must be finite and non-negative. Start with a small `bilateral_kp`, check that each joint resists motion in the expected direction, and increase gradually if needed. `bilateral_kp=0` disables the leader's extra feedback torque.

## Operation

- **Press the top button on the teaching handle to enable the system**
- This code essentially launches two `minimum_gello.py` instances for bimanual control
- Press the button once to synchronize each leader with its follower; press it again to disable synchronization and feedback. Repeat for the other handle.
- On synchronization, the follower takes about 3 seconds to move from its current pose to the leader pose. During continuous following, the leader sends its measured joint angles (rad) and joint velocities (rad/s) from the same CAN-backed state update as the follower's position and velocity targets. The follower's motor PD controller uses both targets. The teaching-handle gripper has a zero target velocity. The initial 3-second alignment still sends position targets with zero target velocity. Joint targets are sent without angle smoothing; stale queued targets are discarded. Command arrivals wake the leader IO loop and the robot gravity-compensation loop; fresh CAN states also wake the robot loop. The motor position controller can still produce a tracking offset under load.
- `--feedback-mode angle` uses `follower_angle + 0.020 s * current_leader_velocity` as the leader's virtual target angle (radians). It forms a spring torque from `leader_base_kp * (leader_angle - virtual_target_angle)`, then reflects the opposite torque on the leader. The same compensated angle error drives the angle-correction factor below. The spring uses the same base joint gains as the earlier angle-tracking controller. The 20 ms prediction reduces the feedback caused by follower tracking delay during motion; it does not change the follower's position target.
- `--feedback-mode torque` uses each follower motor's measured output-shaft torque minus its modeled gravity-compensation torque (both in Nm), retaining the follower's P and D response and other motor effects.
- In both modes each joint's feedback is multiplied by `bilateral_kp / (1 + bilateral_alpha * abs(follower_joint_velocity)) * (2/pi) * atan((100 / bilateral_beta) * abs(angle_error))` when `bilateral_beta > 0`. In angle mode, `angle_error = leader_angle - (follower_angle + 0.020 s * current_leader_velocity)`; in torque mode it remains `leader_angle - follower_angle`. Velocity is in rad/s and angle is in radians; `bilateral_alpha` has units of s/rad, and `bilateral_beta` is dimensionless (100 is the angle sensitivity in 1/rad). `bilateral_alpha=0` disables speed attenuation; `bilateral_beta=0` disables angle attenuation entirely. For positive beta, the angle correction is zero when the relevant angle error is zero and approaches 1 as it grows. There is no low-pass filter or separate software cap on the feedback term. The motor driver still clamps the final MIT torque command to each motor's encoded range.
- In both modes the leader motor uses `kp=0` and the configured `grav_comp_kd` from the arm YAML. Its additional command is the angle/torque feedback term scaled by `bilateral_kp`; gravity compensation remains active. The follower still uses its configured position and velocity PD gains. Feedback clears if the follower sample is more than 0.2 seconds old, and a leader command expires after 0.1 seconds without renewal. The follower returns to gravity-compensation idle after 0.5 seconds without commands, including after synchronization is disabled.
- Motor torque also includes tracking, friction, and acceleration effects. This is an estimated load, not a force/torque sensor measurement.

## Measure CAN receive-to-send latency

Run the launcher with `--trace-latency` and move one or more joints after synchronization:

```bash
python examples/bimanual_lead_follower/bimanual_lead_follower.py --trace-latency
```

Each follower writes `/tmp/i2rt_can_latency_<follower_can_channel>_<pid>.csv`. Its log reports the sample count and per-joint P50, P95, and maximum latency every 5 seconds. One in every 10 leader commands is tagged; a tagged command dropped by the latest-only queue will not appear in the CSV. For each CSV row, `(follower_can_tx_ns - leader_can_rx_ns) / 1e6` is the CAN receive-to-send latency in milliseconds. The intermediate timestamp columns show where the time was spent: leader read and queue, leader RPC send, follower RPC receive and worker, robot update, driver acceptance, and CAN send.

Each leader also writes `/tmp/i2rt_feedback_latency_<leader_can_channel>_<pid>.csv` for the reverse force-feedback path. For each row, `(leader_can_tx_ns - follower_can_rx_ns) / 1e6` measures follower CAN reception to the leader's feedback CAN send. The intermediate columns mark follower feedback publication, follower RPC read, leader RPC reception, leader feedback command, robot update, and driver acceptance. The leader logs per-joint P50/P95 every 5 seconds. Trace records are sampled every 10 leader control updates and may be dropped if a newer command replaces a tagged one before CAN transmission.

The timestamps use the same host's monotonic clock; this measurement is intended for the bimanual launcher running all arms on one computer. CAN receive is stamped when `bus.recv` returns, and CAN send when `bus.send` returns after handing the frame to the driver. It does not include motor firmware or mechanical response, or a hardware timestamp for the frame reaching the wire.

## Plot right-arm J3 angles and torques

Run with `--record-j3`, synchronize the right pair, and move J3. The right leader saves CSV files in `examples/bimanual_lead_follower/tests/log/`; stopping the launcher also renders a matching `.svg` there:

```bash
python examples/bimanual_lead_follower/bimanual_lead_follower.py --record-j3 --bilateral-kp 0.05
```

To view the latest recording, run `examples/bimanual_lead_follower/plot_j3.py` directly from the IDE (the Run triangle) or from the repository root with `python examples/bimanual_lead_follower/plot_j3.py`. It writes the SVG beside the CSV and opens it in your default browser. To view a specific run, pass `--csv-path PATH`.

The figure shares one time axis and shows leader/follower J3 angles in radians and two torque curves in Nm. The follower curve is its CAN-reported, output-shaft motor torque in Nm, with the motor direction applied. It includes gravity compensation, PD output, and other motor effects; torque mode still subtracts the modeled follower gravity torque before computing leader feedback. The leader curve is the extra angle/torque feedback command sent in addition to gravity compensation; it does not include motor saturation or the modeled gravity torque. All four curves are sampled by the leader control loop; follower values are the latest RPC-cached sample, so the points are not simultaneous CAN samples. Samples with stale follower feedback are shown as gaps. Historical latency CSV files do not contain these signals, so this plot requires a J3 recording.

## Troubleshooting

- If you don't see all four CAN interfaces, double-check that:
  - All CAN devices are properly connected
  - Each device has a unique CAN ID as configured
  - The devices are powered on and in UP state
- Use `ip link show` to check the status of individual CAN interfaces
- Press `Ctrl+C` once to stop the launcher. It signals all four arm processes before waiting and terminates any workers left in their process groups.
