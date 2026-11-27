# YAM 上直接运行 RDP（无 ROS）

此入口用于 ForceGello 的单臂关节命令模型，默认右臂。
读取 RDP checkpoint 与其 AT checkpoint，直接调用 i2rt 的 `command_joint_state()`。
不启动 ROS、Flexiv 服务或原来的 Minimum Gello 通信进程。
通过 `can_leader_r` 轮询右手柄黄色按钮，不构造主臂 motor chain、不使能主臂电机。
NEXT 配置和校准流程见 [NEXT](../next/README.md)，默认加载 `data/next/{side}.pt`。

## 控制时序

```text
60 Hz 命令循环（与采集时的从臂频率一致）：
  读取从臂 CAN 最新状态、消费右黄色按钮按下沿
  停用状态：清空 PD/速度/额外力矩，提交重力补偿，保持待机
  启用状态：继续以下策略控制步骤
  用上一条位置目标和当前状态更新 NEXT（先减一次重力，再减 NEXT 预测残差）
  在上一条已发送命令与下一目标之间插值位置，速度目标为 0，提交命令并更新重力补偿
  每 3 个命令周期执行一次 20 Hz 观测与快解码：
  读取从臂六个关节的 NEXT 外力矩估计（单位 Nm）
  读取观测时刻之前的最新相机帧、维护观测历史
  按 5 Hz 提交慢策略观测快照（视觉每 4 步一次，关节观测每步一次）
  用最新可用 latent + 截止当前时刻的关节残余力矩历史，运行 CPU GRU
  解码下一帧关节目标，在随后 3 个 60 Hz 命令周期按 1/3、2/3、1 插值到达

后台慢推理线程：按观测预测 latent，默认 GPU、8 次 DDIM 迭代
相机进程：复用采集器的实现，独立取流、5 Hz 提供最新帧（底层采集仍默认 30 Hz，持续排空 USB 缓冲）
CAN 线程：保留原有持续收发逻辑
```

快解码器使用独立的 CPU AT，避免等待 GPU 扩散计算。
慢策略结果包含观测对应的控制步号；推理花掉的控制周期会计入动作位置，不会从过时 Chunk 的第一步重新执行。
GRU 只使用已经观测到的关节残余力矩历史，不填充未来反馈，也不一次解码后开环播放整个 Chunk。
启用期间，新 latent 到来后使用新计划，计划过期或时间轴有缺口时先保持上次位置、速度置零；持续无可用计划则退出。
按黄色按钮停用则直接进入纯重力补偿，不保持旧位置；仍沿用硬件配置的重力补偿阻尼。
目标为 20 Hz，Linux 调度不提供硬实时保证；超时时钟槽会跳过，日志记录 `missed`。

当前支持 `space=joint`、`key=right_robot_command`（或左臂对应字段）、输出 7 维的含力观测 RDP checkpoint。
网络动作布局 `[Δq1..Δq6, gripper_abs]`。同一个 latent 的整段手臂动作都相对于产生这个 latent 时的慢策略从臂关节观测；夹爪是绝对归一化开度。
部署时保存固定的关节锚点；每次快解码只对前六维加上该锚点，夹爪直接使用第七维。
全部七个关节的速度目标均为 0，位置继续按 60 Hz 插值。夹爪目标开度限制在归一化 [0,1]。
原来使用重力残差训练的 AT/RDP 需要用 NEXT 数据重新训练，避免力观测分布不一致。
源 Zarr 的 `right_robot_command` 仍为 14 维；训练加载器只取前七维，在每个训练窗口内仅对前六维减去固定的慢观测关节角，无需重新转换已有数据。旧的相对夹爪、逐样本相对动作、绝对动作或 TCP wrench checkpoint 需要重新训练 AT 与 RDP；部署入口会拒绝旧的相对夹爪 checkpoint。
训练标签为 `action_window[k,:6] = command_zarr[k+50ms][:6] - follower_joint_pos_zarr[slow_anchor,:6]`，`action_window[k,6] = command_zarr[k+50ms][6]`；同一窗口所有 k 共用 `slow_anchor`。两阶段均设置
`slow_obs_latency_steps: 1`，t=0 ms 的慢视觉向量最早在
t=50 ms 的快解码周期与新本体状态结合，预测 t=100 ms 的关节命令；
60 Hz 命令周期在 50～100 ms 间插值到该目标。
TCP 动作模型和双臂 28 维模型会在打开硬件前报错。

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

模型使用当前从臂状态计算输入，默认包括关节位置7、速度7、六个关节的残余力矩与两帧 RGB 历史。
关节残余力矩与数据采集一样，等于实测力矩减最近一次提交的重力补偿力矩；其中仍包含摩擦、惯性和跟踪误差。
模型不使用 TCP wrench，也不在部署时计算力的 Jacobian 逆变换。
只有按黄色按钮启用后，等待观测和第一条有效计划时才保持按下时的位置；待机时始终是重力补偿。
网络输出先反归一化为相对关节位移，再加到所属 latent 保存的慢观测关节锚点上；速度目标统一为 0。

`--inference-hz` 默认 5，必须整除 20 Hz；`--inference-steps` 默认 8。
`--command-hz` 默认 60，必须是 `--control-hz` 的整数倍；`--max-steps` 限制 60 Hz 命令周期数。
推理中相机断流、CAN 反馈过期、手柄输入失联、推理异常或超时会退出并清理资源。
正常推理使用训练时相同的 **20/5 Hz** 本体感知与视觉历史。
慢策略取最近两帧 RGB（间隔 0.2 秒）和最近两帧关节状态（间隔 1/20 秒），分别选择历史，不共用降采样步长。
快 GRU 每周期只读取关节残余力矩6维反馈；关节位置和速度仍用于慢策略观测，关节位置还用于固定动作锚点。
新模型 Chunk 为32步，观测前缀8步，可执行部分25步；旧模型需重新训练 AT 和 RDP。
相机最大年龄默认0.5秒以容纳5 Hz更新间隔；模型视觉采样率与启动参数不一致时，在打开硬件前报错。
原始采集仍可保存20 Hz相机；训练加载器会抽成5 Hz并模拟图像保持，不需要重采20 Hz Zarr。

## 验证

```bash
uv run --no-sync pytest examples/rdp_policy/tests -q
```

测试使用假机器人检查 20 Hz 观测、60 Hz 插值命令、零速度目标、NEXT 输入时序、延迟计划的时间对齐和错误处理。
安装 RDP 依赖且存在相邻 RDP checkout 时，还会生成测试 checkpoint，验证实际模型加载、latent 推理与快解码。
不连接真实机器人。
