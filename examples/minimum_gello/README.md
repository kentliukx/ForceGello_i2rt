# Minimum Gello (Teleoperation)

Minimal leader-follower teleoperation example. Supports different arm and gripper assemblies, simulation mode, and local/remote visualization.

## Quick Start

```bash
# Follower mode (default) — serves robot state over portal
python examples/minimum_gello/minimum_gello.py --can-channel can0

# Follower mode in simulation (no hardware required)
python examples/minimum_gello/minimum_gello.py --sim

# Leader mode — reads teaching handle and drives a remote follower
python examples/minimum_gello/minimum_gello.py --mode leader --can-channel can1

# Visualizer — MuJoCo viewer mirrors live robot state
python examples/minimum_gello/minimum_gello.py --mode visualizer_local

# Different arm and gripper combinations
python examples/minimum_gello/minimum_gello.py --arm big_yam --gripper linear_4310 --sim
python examples/minimum_gello/minimum_gello.py --arm yam_pro --gripper flexible_4310 --sim
```

## Leader-Follower Setup

Run the follower on one terminal (or machine):

```bash
python examples/minimum_gello/minimum_gello.py \
    --gripper linear_4310 --mode follower --can-channel can0
```

Run the leader on another:

```bash
python examples/minimum_gello/minimum_gello.py \
    --gripper yam_teaching_handle --mode leader --can-channel can1 \
    --feedback-mode angle --bilateral-kp 0.2 --bilateral-alpha 1.0 --bilateral-beta 1.0
```

Press **button 0** on the teaching handle to synchronize. Press again to desync.

> **Note:** Leader mode requires real hardware (`--sim` is not supported).

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--arm` | `yam` | Arm type (`yam`, `yam_pro`, `yam_ultra`, `yam_ultra_2`, `big_yam`, `no_arm`) |
| `--gripper` | `yam_teaching_handle` | Gripper type (`crank_4310`, `linear_3507`, `linear_4310`, `flexible_4310`, `yam_teaching_handle`, `no_gripper`) |
| `--mode` | `follower` | Operation mode (`follower`, `leader`, `visualizer_local`, `visualizer_remote`) |
| `--sim` | `False` | Use SimRobot instead of real hardware |
| `--can-channel` | `can0` | CAN interface name |
| `--server-host` | `localhost` | Portal server host (for leader/remote visualizer) |
| `--server-port` | `11333` | Portal server port |
| `--feedback-mode` | `angle` | `angle` tracks follower joint angles; `torque` reflects follower motor torque minus modeled gravity (leader mode) |
| `--bilateral-kp` | `0.0` | Any finite non-negative feedback gain (leader mode); leader torque caps still apply |
| `--bilateral-alpha` | `1.0` | Non-negative speed attenuation coefficient in s/rad; per-joint gain is `bilateral_kp / (1 + bilateral_alpha * abs(follower_joint_velocity))` |
| `--bilateral-beta` | `1.0` | Non-negative angle correction coefficient in 1/rad; multiplies feedback by `(2/pi) * atan(bilateral_beta * abs(leader_angle - follower_angle))`; zero disables feedback |
| `--trace-latency` | `False` | Tag one in ten leader commands and write per-joint CAN receive-to-send timestamps to `/tmp/i2rt_can_latency_<channel>_<pid>.csv` on the follower; use on one host |
| `--record-j3` | `False` | In leader mode, save J3 angles, follower PD torque estimate, and leader extra feedback torque to CSV and render a standalone SVG after shutdown |
| `--ee-mass` | XML default | Override link_6 mass (kg) for gravity compensation with different handles |

## Overriding Handle Weight

Older 3D-printed teaching handles may have a different mass than the default (0.258 kg). Use `--ee-mass` to compensate:

```bash
python examples/minimum_gello/minimum_gello.py --ee-mass 0.350 --can-channel can0
```
