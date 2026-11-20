# YAM 上直接运行 RDP（无 ROS）

此入口用于 ForceGello 的单臂关节命令模型，默认右臂。
读取 RDP checkpoint 与其 AT checkpoint，直接调用 i2rt 的 `command_joint_state()`。
不启动 ROS、Flexiv 服务或原来的 Minimum Gello 通信进程。
通过 `can_leader_r` 轮询右手柄黄色按钮，不构造主臂 motor chain、不使能主臂电机。
现有双臂遥操作和数据采集入口继续使用原来的逻辑。

## 控制时序

```text
60 Hz 命令循环（与采集时的从臂频率一致）：
  读取从臂 CAN 最新状态、消费右黄色按钮按下沿
  停用状态：清空 PD/速度/额外力矩，提交重力补偿，保持待机
  启用状态：继续以下策略控制步骤
  在上一条已发送位置与下一目标之间插值；关节速度取 20 Hz 目标位置差分，夹爪速度为 0
  每 3 个命令周期执行一次 20 Hz 观测与快解码：
  计算当前 Jacobian 和局部 TCP wrench（与训练转换器共用实现）
  读取观测时刻之前的最新相机帧、维护观测历史
  按 5 Hz 提交慢策略观测快照（视觉每 4 步一次，关节观测每步一次）
  用最新可用 latent + 截止当前时刻的关节位置、速度历史，运行 CPU GRU
  解码下一帧关节目标，在随后 3 个 60 Hz 命令周期按 1/3、2/3、1 插值到达

后台慢推理线程：按观测预测 latent，默认 GPU、8 次 DDIM 迭代
相机进程：复用采集器的实现，独立取流、5 Hz 提供最新帧（底层采集仍默认 30 Hz，持续排空 USB 缓冲）
CAN 线程：保留原有持续收发逻辑
```

快解码器使用独立的 CPU AT，避免等待 GPU 扩散计算。
慢策略结果包含观测对应的控制步号；推理花掉的控制周期会计入动作位置，不会从过时 Chunk 的第一步重新执行。
GRU 只使用已经观测到的关节位置和速度历史，不填充未来反馈，也不一次解码后开环播放整个 Chunk。
启用期间，新 latent 到来后使用新计划，计划过期或时间轴有缺口时先保持上次位置、速度置零；持续无可用计划则退出。
按黄色按钮停用则直接进入纯重力补偿，不保持旧位置；仍沿用硬件配置的重力补偿阻尼。
目标为 20 Hz，Linux 调度不提供硬实时保证；超时时钟槽会跳过，日志记录 `missed`。

当前支持 `space=joint`、`key=right_robot_command`（或左臂对应字段）、输出 7 维的 RDP checkpoint。
动作布局 `[q1..q6,gripper]`；夹爪位置保持归一化 [0,1]，速度强制为 0。
六个关节的速度由本次目标与上次已发送的位置之差除以 50 ms 得到；60 Hz 插值期间保持该速度。
该布局对应训练标签 `action[t] = 主臂[t+1]`。对二阶段训练时设置了
`slow_obs_latency_steps: 1` 的新 checkpoint，t=0 ms 的慢视觉向量最早在
t=50 ms 的快解码周期与新本体状态结合，预测 t=100 ms 的关节命令；
60 Hz 命令周期在 50～100 ms 间插值到该目标。
TCP 动作模型、旧 14 维单臂模型和双臂模型会在打开硬件前报错。

## 安装与启动

在原 i2rt 的 uv 环境中补充模型运行依赖（PyTorch 的 CPU/CUDA 版本应与本机一致）：

```bash
cd /home/kentliukx/ForceGello/i2rt
uv sync --extra collection
uv pip install -r examples/rdp_policy/requirements.txt
```

先验证 checkpoint，不打开任何 CAN 或相机：

```bash
uv run --no-sync python examples/rdp_policy/run.py \
  --checkpoint ../RDP/data/outputs/YOUR_RUN/rdp/checkpoints/latest.ckpt \
  --device cpu --validate-only
```

运行右臂策略（替换模型路径；相机默认使用与数据采集相同的固定设备）：

```bash
uv run --no-sync python examples/rdp_policy/run.py \
  --checkpoint ../RDP/data/outputs/YOUR_RUN/rdp/checkpoints/latest.ckpt \
  --can-channel can_follower_r \
  --device cuda:0
```

排查策略动作时加 `--debug-actions`，程序每秒打印一次快解码状态、目标关节角、
差分速度和目标相对实测关节角的最大误差。打印的关节角单位为 rad，速度单位为 rad/s。

相机编号统一写在 `examples/bimanual_data_collection/camera_devices.py`：顶部相机
`realsense:036422060422`，右腕相机 `realsense:135122075425`，左腕相机
`realsense:135122079769`。部署时只打开 checkpoint 的 `obs` 实际使用的相机；
本右臂模型默认需要右从臂、右手柄、顶部相机和右腕相机。
换相机时修改该文件，或通过 `--external-camera`、`--right-wrist-camera` 覆盖。
不要同时运行占用相同 CAN 接口的采集器或遥操作程序。

## 夹爪初始化与黄色按钮

- 程序启动时，沿用 `get_yam_robot()` 的夹爪初始化。默认 `linear_4310` 的配置为自动标定，
  会用小力矩分别向两个方向运动，检测开合限位。每次启动程序执行一次，不在每次启停策略时重做。
- 标定完成后，程序以 20 Hz 运行纯重力补偿，等待新的黄色按钮按下沿，不自动执行策略。
- 按右黄色按钮一次：重新收集观测历史、生成新 latent，历史和计划就绪后执行策略。
  等待首条计划时暂时保持按下时的当前位置。
- 再按一次：在收到事件的控制周期内清空策略 PD/速度/额外力矩，先提交重力补偿，再处理相机队列。
  夹爪也不再保持原位置目标。
- 再次按下可重新启动；清空旧历史和计划，即使旧推理随后完成，也不会恢复旧命令。
- 长按只切换一次，按下沿去抖。初始化期间的按键、启动时已经按住的按钮不会自动启动策略；
  等看到 `gravity idle; waiting for yellow` 后松开再按。

默认右手柄通道 `--button-can-channel can_leader_r`、黄色输入 `--yellow-button-index 0`。
按键独立线程轮询，避免 CAN 等待阻塞 20 Hz 控制；不会运行主臂重力补偿或力反馈控制。
按 Ctrl+C 退出，先清空控制命令，再关闭机器人及所有输入线程。

AT 路径优先使用训练配置中的路径；若移动了整个训练目录，会尝试相邻的 `at/checkpoints/latest.ckpt`。
也可用 `--at-checkpoint PATH` 指定。应保留训练时使用的 AT 权重。

模型使用当前从臂状态计算输入，默认包括关节位置7、速度7、估算 wrench6 与两帧 RGB 历史。
`--arm`、`--gripper`、`--wrench-damping`、`--wrench-sign`、`--gripper-max-width-m` 必须与训练数据转换时一致。
默认分别为 yam、linear_4310、0.01、+1、0.095 m。力估计仍包含摩擦、跟踪与模型误差。
只有按黄色按钮启用后，等待观测和第一条有效计划时才保持按下时的位置；待机时始终是重力补偿。
网络输出先反归一化，再转换为关节位置目标和差分速度目标发送。

`--inference-hz` 默认 5，必须整除 20 Hz；`--inference-steps` 默认 8。
`--command-hz` 默认 60，必须是 `--control-hz` 的整数倍；`--max-steps` 限制 60 Hz 命令周期数。
推理中相机断流、CAN 反馈过期、手柄输入失联、推理异常或超时会退出并清理资源。
正常推理使用训练时相同的 **20/5 Hz** 本体感知与视觉历史。
慢策略取最近两帧 RGB（间隔 0.2 秒）和最近两帧关节状态（间隔 1/20 秒），分别选择历史，不共用降采样步长。
快 GRU 每周期读取关节位置7、速度7，共14维反馈；慢策略仍读取估算 wrench。
新模型 Chunk 为32步，观测前缀8步，可执行部分25步；旧模型需重新训练 AT 和 RDP。
相机最大年龄默认0.5秒以容纳5 Hz更新间隔；模型视觉采样率与启动参数不一致时，在打开硬件前报错。
原始采集仍可保存20 Hz相机；训练加载器会抽成5 Hz并模拟图像保持，不需要重采20 Hz Zarr。

## 验证

```bash
uv run --no-sync pytest examples/rdp_policy/tests -q
```

测试使用假机器人检查 20 Hz 观测、60 Hz 插值命令、7 维位置和差分速度映射、延迟计划的时间对齐和错误处理。
安装 RDP 依赖且存在相邻 RDP checkout 时，还会生成测试 checkpoint，验证实际模型加载、latent 推理与快解码。
不连接真实机器人。
