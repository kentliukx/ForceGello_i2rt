# NEXT for YAM (without ROS)

Run from the i2rt repository root in the existing `ForceGello` environment. Requires PyTorch;
there is no ROS dependency. Method reference: [FACTR 2 NEXT](https://github.com/philiphan0109/factr2_next).

## Signal and controller

This adaptation subtracts the already applied gravity term once, as requested:

```text
x(t) = [q(t), qdot(t), q_command_held(t) - q(t)]       # 6 joints, rad / rad/s
y(t) = measured_motor_torque(t) - applied_gravity(t)  # 6 joints, Nm
free_residual(t) = LSTM(x(t-49:t))
external_torque(t) = y(t) - free_residual(t)
leader_feedback(t) = -bilateral_kp * external_torque(t)
```

The checkpoint learns the non-gravity free-motion residual, unlike upstream NEXT which learns
the total free-space torque. Checkpoint metadata enforces this convention so gravity cannot
accidentally be subtracted twice. No Jacobian conversion, speed/angle attenuation, contact gate,
or extra low-pass filter is used on NEXT feedback. The existing leader gravity compensation and
its configured damping remain active.

Follower commands, leader feedback, recording and NEXT histories all update at **60 Hz**.
The follower receives the leader position (with the existing startup alignment/joint limits),
with **all velocity targets zero**. The low-level CAN loop is unchanged.
Inputs are read before submitting the next position command, paired with the previously held
target. This is an application-level convention used identically during calibration and inference;
it does not claim per-motor CAN transmit/receive synchronization.

The model is a stateless two-layer LSTM (hidden 128), followed by a 256-wide MLP head. Each
50-sample history predicts the residual torque at its last input timestep, not future torque.
The first 50 samples after engagement or a timing gap warm the history; force feedback is zero
until it is ready. Warmup samples are flagged and excluded from active-arm Zarr training segments.
Inactive arms have zero, invalid NEXT effort; `joint_effort_external_valid` records availability.

## 1. Record contact-free calibration motion

```bash
conda activate ForceGello
python examples/next/collect.py
```

This entry point always disables haptic feedback, even if feedback flags are supplied.
Each yellow button starts/stops that pair's normal alignment/following. Right white starts/stops
recording; Ctrl+C closes all arms. Cameras are disabled by default. Record approximately 10 minutes
of **contact-free** motion spanning the workspace and speeds used by the LEGO task, then stop the
recording with white before exiting. Contact-free is a requirement on the operator; the recorder
cannot certify absence of contact. Output: `data/next/free_motion/<episode>/robot_left.pkl` and
`robot_right.pkl`, including the exact NEXT samples and applied gravity values.

## 2. Train (does not open hardware)

```bash
python examples/next/train.py --device cuda:0
```

Trains separate left/right models, defaults: history 50, 20 epochs, batch 256. `--device cpu` also
works; `--side right` trains only the right arm. Each continuous run holds out its last 20% for
validation, with no overlapping history frames across the split. Histories never cross episodes,
disengagements or timing gaps. Normalization uses training frames only.

Latest completed epoch checkpoints: `data/next/left.pt`, `data/next/right.pt`. These files are
overwritten after every epoch and are loaded by default during collection and deployment. Training and validation
normalized MSE and per-joint validation RMSE in Nm are written to `left.metrics.json` / `right.metrics.json`.
These models must be trained with the same arm, gains, gravity settings and tool used at runtime.
Changing these conditions may require new calibration data. Checkpoints validate side and rate.

## 3. LEGO recording with NEXT haptics

```bash
python examples/bimanual_data_collection/collect_data.py \
  --output data/demonstrations_lego_next --bilateral-kp 1
```

NEXT is now the default. The normal collector still records all three cameras. Both checkpoint
files are required before hardware startup. Override with `--next-left-checkpoint` and
`--next-right-checkpoint`. Raw measured torque, applied gravity, gravity residual and NEXT estimates
remain distinguishable in the recording; `follower.joint_eff_external[:6]` contains NEXT output.
The Zarr converter passes that signal into `*_robot_joint_effort_external` and writes zero target
velocities into new recordings' `*_robot_command` without changing measured leader velocities.

For teleoperation without cameras:

```bash
python examples/bimanual_lead_follower/bimanual_lead_follower.py --bilateral-kp 1
```

## 4. RDP deployment

```bash
python examples/rdp_policy/run.py \
  --checkpoint ../RDP/data/outputs/YOUR_RUN/rdp/checkpoints/latest.ckpt \
  --next-checkpoint data/next/right.pt
```

Position still comes from RDP, with the existing 20 Hz targets / 60 Hz interpolation. Arm and gripper
velocity targets are zero. NEXT runs at 60 Hz, and its latest output is sampled by the 20 Hz slow/fast
observation pipeline. The default NEXT path is `data/next/{side}.pt`. NEXT is warmed before policy
observations are accumulated. Old policies trained on gravity-only residual torque have a different
force-input distribution: collect/convert NEXT demonstrations and retrain AT and RDP for this setup.

No calibration data, checkpoints, or robot motion are generated merely by adding these files.
