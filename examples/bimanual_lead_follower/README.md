# Bimanual Lead-Follower System Setup

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
python examples/bimanual_lead_follower/bimanual_lead_follower.py --feedback-mode angle --bilateral-kp 0.05 --bilateral-alpha 1.0 --bilateral-beta 1.0
```

Run this from the repository root. The flags are passed to both leader processes. `feedback_mode` defaults to `angle`, `bilateral_kp` defaults to `0.0` (feedback off), and `bilateral_alpha` and `bilateral_beta` default to `1.0`. All gains must be finite and non-negative. Start with a small `bilateral_kp`, check that each joint resists motion in the expected direction, and increase gradually if needed. The leader torque cap still applies even when `bilateral_kp` exceeds 1.

## Operation

- **Press the top button on the teaching handle to enable the system**
- This code essentially launches two `minimum_gello.py` instances for bimanual control
- Press the button once to synchronize each leader with its follower; press it again to disable synchronization and feedback. Repeat for the other handle.
- On synchronization, the follower takes about 3 seconds to move from its current pose to the leader pose. During continuous following, joint targets are sent directly without angle smoothing; the leader sends one target per new arm state, and stale queued targets are discarded. The follower worker wakes when a target arrives. The motor position controller can still produce a tracking offset under load.
- `--feedback-mode angle` makes the leader follow the follower's measured joint angles. It forms a spring torque from `leader_base_kp * (leader_angle - follower_angle)`, then reflects the opposite torque on the leader. The spring uses the same base joint gains as the earlier angle-tracking controller, with software torque caps. Follower tracking delay also creates angle error, so this mode can resist fast motion even without contact.
- `--feedback-mode torque` uses each follower motor's measured output-shaft torque minus its modeled gravity-compensation torque (both in Nm), retaining the follower's P and D response and other motor effects.
- In both modes each joint's feedback is multiplied by `bilateral_kp / (1 + bilateral_alpha * abs(follower_joint_velocity)) * (2/pi) * atan(bilateral_beta * abs(leader_angle - follower_angle))`. Velocity is in rad/s and angle is in radians, so `bilateral_alpha` has units of s/rad and `bilateral_beta` has units of 1/rad. `bilateral_alpha=0` disables speed attenuation; `bilateral_beta=0` disables feedback. The angle correction is zero when the joint angles match, including in torque mode, and approaches 1 as their difference grows. There is no low-pass filter on the feedback signal. Feedback is capped at 10 Nm for joints 1–3 and 5 Nm for joints 4–6, then added to the leader's gravity compensation. Feedback clears if the follower sample is more than 0.2 seconds old. A leader torque command also expires after 0.1 seconds without a new command.
- If follower position commands stop for 0.5 seconds, the follower returns to gravity-compensation idle. This also happens after synchronization is disabled.
- Motor torque also includes tracking, friction, and acceleration effects. This is an estimated load, not a force/torque sensor measurement.

## Measure CAN receive-to-send latency

Run the launcher with `--trace-latency` and move one or more joints after synchronization:

```bash
python examples/bimanual_lead_follower/bimanual_lead_follower.py --trace-latency
```

Each follower writes `/tmp/i2rt_can_latency_<follower_can_channel>_<pid>.csv`. Its log reports the sample count and per-joint P50, P95, and maximum latency every 5 seconds. One in every 10 leader commands is tagged; a tagged command dropped by the latest-only queue will not appear in the CSV. For each CSV row, `(follower_can_tx_ns - leader_can_rx_ns) / 1e6` is the CAN receive-to-send latency in milliseconds. The intermediate timestamp columns show where the time was spent: leader read and queue, leader RPC send, follower RPC receive and worker, robot update, driver acceptance, and CAN send.

The timestamps use the same host's monotonic clock; this measurement is intended for the bimanual launcher running all arms on one computer. CAN receive is stamped when `bus.recv` returns, and CAN send when `bus.send` returns after handing the frame to the driver. It does not include the follower motor's mechanical response or a hardware timestamp for the frame reaching the wire.

## Plot right-arm J3 angles and torques

Run with `--record-j3`, synchronize the right pair, and move J3. When you stop the launcher, the right leader writes `/tmp/i2rt_j3_can_leader_r_<pid>.csv` and renders the matching `.svg` figure:

```bash
python examples/bimanual_lead_follower/bimanual_lead_follower.py --record-j3 --bilateral-kp 0.05
```

The figure shares one time axis and shows leader/follower J3 angles in radians and two torque curves in Nm. The follower curve is an **estimate** of its motor's onboard PD contribution, `kp * (commanded_angle - feedback_angle) + kd * (commanded_velocity - feedback_velocity)`, using the values from each CAN transaction. The motor does not report this contribution separately; firmware saturation and timing can make actual torque differ. The leader curve is the extra feedback torque commanded by the bilateral controller, before the motor's internal limits. All four curves are sampled by the leader control loop; follower values are the latest RPC-cached sample, so the points are not simultaneous CAN samples. Samples with stale follower feedback are shown as gaps. Historical latency CSV files do not contain these signals, so this plot requires a new run.

## Troubleshooting

- If you don't see all four CAN interfaces, double-check that:
  - All CAN devices are properly connected
  - Each device has a unique CAN ID as configured
  - The devices are powered on and in UP state
- Use `ip link show` to check the status of individual CAN interfaces
- Press `Ctrl+C` once to stop the launcher. It signals all four arm processes before waiting and terminates any workers left in their process groups.
