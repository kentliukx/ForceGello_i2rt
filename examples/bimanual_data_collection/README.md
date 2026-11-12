# 双臂遥操作与多频率数据采集

双臂共用一个控制进程、一个 **300 Hz** 上层控制循环。每轮读取四条 CAN 的最新状态；
第 1、6、11……轮先计算两个从臂的重力补偿并提交命令（**60 Hz**），再计算两个主臂的重力补偿、
力反馈并提交命令（**300 Hz**）。其余轮只更新主臂。机器人数据在从臂更新轮记录，仍为 60 Hz。

采集器通过 `get_yam_robot(background_control=False)` 关闭各机械臂的独立 `robot_server` 线程，
显式调用 `refresh_state()` 读取 CAN buffer、`update()` 计算重力并更新命令 buffer。底层 CAN 线程保持原样，
持续收发最新状态/命令。原来的 `minimum_gello.py` 和 `bimanual_lead_follower.py` 保持不变，默认仍开启后台控制。
力反馈复用 `minimum_gello` 的 `angle`/`torque` 公式、20 ms 角度预测和 alpha/beta 调节。
力矩残差扣除各臂最近一次提交的重力项，并保存该项为 `joint_eff_gravity`，从臂不会在 60 Hz 更新之间额外计算重力。

三路相机各自独立取流，以 **20 Hz** 保存最新帧；文件写入在主进程进行。
控制线程使用绝对单调时钟，超时跳过时钟槽、不突发补发；从臂始终每五次实际迭代更新一次。
这里保证上层计算和命令 buffer 提交的先后顺序；各 CAN 线程独立发送，实际总线发送顺序和 Linux 唤醒时间不构成硬实时保证。

## 启动

在 i2rt 根目录激活 ForceGello conda 环境。该环境需要 opencv-python 和 pyrealsense2：

```bash
conda activate ForceGello
python -m pip install "opencv-python>=4.10" pyrealsense2
python examples/bimanual_data_collection/collect_data.py --list-cameras
```

--list-cameras 只枚举设备，不打开机械臂。三台相机按序列号固定在
[collect_data.py](collect_data.py) 的 CAMERA_DEVICES：左腕 D435 135122075425、
右腕 D435 135122079769、外部 D415 036422060422。这些是 pyrealsense2 报出的序列号，
与 `/dev/v4l/by-id` 名称里的 USB 序列号不同。更换相机或调整安装位置时修改此映射；
录制的 metadata.json 也保存实际使用的映射。

```bash
python examples/bimanual_data_collection/collect_data.py
```

默认预设：`--control-hz 60 --camera-hz 20 --feedback-mode angle --bilateral-kp 1 --bilateral-alpha 0 --bilateral-beta 5`。
输出目录默认为 `data/demonstrations`；命令行参数仍可覆盖预设。`--bilateral-kp 0` 可关闭额外反馈。
不要同时运行旧的 `bimanual_lead_follower` 或其他占用同一 CAN 接口的程序。
默认四个 CAN 接口是 `can_leader_l/r`、`can_follower_l/r`；可以通过 `--left-leader-can` 等参数覆盖。

## 按钮

- **左黄色按钮**：只切换左臂遥操作；首次同步用约 3 秒对齐从臂。
- **右黄色按钮**：只切换右臂遥操作。
- **右白色按钮**：第一次按开始记录双臂和三路相机；再按一次结束当前轨迹。继续按可以记录下一条。
- 白色按钮不改变两组遥操作状态，结束录制后仍可继续遥操作。

默认黄色为 `io_inputs[0]`、白色为 `io_inputs[1]`。用 `--print-buttons` 查看实际输入；若接线不同，使用
`--yellow-button-index 1 --white-button-index 0` 对调。按下沿去抖，长按不会重复开始/结束。
只有两组机械臂和三路相机都 READY 后才接受录制开始。程序不强制要求两组处于同步状态，记录中的
`teleoperation_mode` 会如实保存 `idle`、`aligning` 或 `following`。

## 保存内容

ImplicitRDP 原采集器保存 `{'sensor_message_list': SensorMessageList(sensorMessages=[...])}`，每段一个 pickle。
单帧包括时间、双臂 TCP 位姿/RPY、速度、六维力矩、夹爪宽度/力、相机图像，以及可选深度和触觉。
原采集器没有保存真正下发的 action，后处理才用下一帧状态构造标签。

本采集器保留异步原始流，避免把 20 Hz 图像重复保存三次，也不丢弃 60 Hz 机器人/力矩数据：

```text
data/demonstrations/
  1/                           # 一条轨迹；录制中目录名为 1.partial
    metadata.json              # 参数、字段语义、开始/结束时间、状态和计数
    robot_left.pkl             # 左主从臂 60 Hz pickle 流
    robot_right.pkl            # 右主从臂 60 Hz pickle 流
    cameras.jsonl              # 每帧相机名、JPEG 路径、实际/计划时间、尺寸
    left_wrist/00000000.jpg     # 各 20 Hz
    right_wrist/00000000.jpg
    external/00000000.jpg
  2/
    ...
```

每个机器人样本是普通字典，包括：

- `leader` / `follower`：实测关节位置、速度、电机力矩、扣除重力后的力矩残差、CAN 接收时间。
- `leader_joint_pos_with_gripper`：主臂六个关节加手柄夹爪输入。
- `command`：当次传给从臂位置/速度控制接口的目标；非遥操作时为 `None`。关节限位后保存，和实测状态分开。
- `command_monotonic_ns`：从臂高层命令调用时间；`follower_update_monotonic_ns` 是重力计算及命令 buffer 更新完成时间。夹爪限力可能进一步改变目标。
- `leader_feedback_torque` / `leader_feedback_monotonic_ns`：主臂额外反馈力矩及其更新时间。
- `cartesian`：同一从臂状态导出的 TCP/RPY、TCP 速度、局部 TCP wrench 估计。计算在保存进程中进行，不占用控制线程。
- `buttons`、`teleoperation_mode`、实际/计划采样时间。`tick_index` 是 300 Hz 调度槽编号，`missed_periods` 是上次机器人采样以来跳过的主循环时钟槽数。

关节单位 rad/rad·s⁻¹，力矩 Nm；夹爪位置是归一化开度 [0,1]。时间使用 int64 纳秒，保留 Unix 与单调时钟。
两组臂共用上层循环，CAN 反馈不是硬件同步，各自样本和 CAN 时间分别保留。相机时间为主机取到图像时刻，不能等同于曝光时刻；
RealSense 额外保留设备毫秒时间，但跨设备对齐仍使用同一主机单调时钟。

`robot_*.pkl` 是逐条 pickle 写入的流，不是一次 `pickle.load` 得到整个列表。读取方式：

```python
from pathlib import Path
from examples.bimanual_data_collection.recording import read_robot_stream

for sample in read_robot_stream(Path("data/demonstrations/1/robot_left.pkl")):
    print(sample["command"], sample["follower"]["joint_eff_external"])
```

磁盘逐条写入，不在内存累计整段图像。停止录制后有短暂排空时间，按采样时间归入对应轨迹；队列溢出、
相机断开或设备长时间无样本会终止采集并保存当前轨迹为 `interrupted`，不静默丢数据。
正常结束保存为 `complete`。断电可能留下 `.partial`，已写入的记录仍可恢复，但不能认为轨迹完整。

## 回放采集数据

在 ForceGello 环境、i2rt 根目录运行：

```bash
python examples/bimanual_data_collection/play.py
# 播放其他轨迹：
python examples/bimanual_data_collection/play.py --episode data/demonstrations/2
```

默认循环播放第 1 段。MuJoCo 窗口展示左右**从臂实测姿态**和夹爪，另外打开
`left_wrist`、`right_wrist`、`external` 三个相机窗口；各流按记录的单调时间戳同步。
左臂的青色点、右臂的橙色点分别表示末端空间 Kp 控制器的**虚拟目标点**，不是力作用点。
若末端受到估计外力 `F_ext`，点的位置满足 `F_cmd = Kp × (目标点 − TCP) = −F_ext`：
从真实末端指向点的方向，就是控制器为抵消外力而输出的力方向。
默认 `--cartesian-kp 100` N/m（1 N 对应 1 cm 位移），可调大刚度以缩短点的距离；显示距离最多 20 cm。
估计使用记录的电机力矩减去重力补偿后的残差，经阻尼雅可比反推；其中仍有跟踪、摩擦和惯性影响，
点的位置只是可视化，不代表真实接触位置或力传感器读数；它不会向机械臂发送控制命令。
回放根据力学平衡关系估计 `F_ext`，并消去采集时的 `wrench_sign`；快速运动时残差中的惯性和摩擦会让估计偏离真实接触力。
相机窗口获得焦点时，空格暂停/继续，`R` 从头播放，`Q` 或 `Esc` 退出；关闭 MuJoCo 窗口也会退出。
可用 `--speed 0.5` 慢放或 `--no-loop` 播放一次。`--validate-only` 只检查数据和模型，不打开窗口。
播放只读取磁盘数据，不连接 CAN。仿真中右臂基座位于左臂基座右侧 0.60 m（模型 -Y 方向）；
两臂朝向暂按平行处理，其他基座外参尚未标定。

## 导出 ImplicitRDP 格式

离线导出每条轨迹，无需连接机械臂或启动 ROS：

```bash
uv run python examples/bimanual_data_collection/export_implicitrdp.py \
  --episode data/demonstrations/1 --output data/implicitrdp/1.pkl
```

输出保留原项目的 `sensor_message_list.sensorMessages` 属性接口，使用标准库 `SimpleNamespace` 存储，
不要求反序列化时安装 i2rt 或 ROS。每个 SensorMessage 包含双臂 TCP/RPY、TCP 速度、估计 wrench、夹爪状态、
`externalCameraRGB`、`leftWristCameraRGB` 和 `rightWristCameraRGB`。图像明确转为 RGB。
附加 `alignment` 记录各流来源时间和真实控制目标，`export_metadata` 说明坐标系、缺失传感器和估计方式。

导出以左腕 20 Hz 图像为锚，其他相机和双臂状态均选择**不晚于锚时刻**的最新样本，超过允许年龄则跳过。
不按黄色按钮的遥操作状态筛选；单臂操作、双臂操作都保留双臂状态和三路图像，数据字段一致。
网络使用哪些机械臂或相机输入由训练配置决定。原始 60 Hz 数据不受影响。
`--max-robot-age-s`、`--max-camera-age-s` 控制对齐容差；输出报告丢弃的帧数。

TCP 由当前所选 YAM + 夹爪模型的 `grasp_site` 正运动学计算，双臂各在自己的基座坐标系，需要统一世界系时应另外标定。
速度用几何雅可比计算。六维 wrench 是关节力矩残差经阻尼雅可比投影的**估计量**，不等价于 Flexiv 六维力传感器。
它仍含跟踪、摩擦、加速等影响；默认正号表示电机残差对应方向，`--wrench-sign -1` 可翻转，接触力符号需实际标定。
采集时已保存 `cartesian` 派生量和代码版本；导出优先复用这些量。若改变阻尼参数，会用当前模型重新计算，
因此应保留采集时使用的代码/模型版本，不能修改模型后假定 FK 不变。
默认夹爪宽度按归一化开度乘名义 0.095 m，实际行程可用 `--gripper-max-width-m` 设置。
没有标定的夹爪力用 NaN 标记；没有触觉/点云，触觉兼容字段是零占位、点云为空，不能作为真实训练输入。

原 `post_process_data.py` 仍需按任务修改路径、传感器配置和双臂 action 定义；它不会自动使用本次保存的真实命令。
特别是原脚本默认会用下一帧实测状态生成 action，训练前应明确要用这种标签还是采集到的 `command`。

## 验证

```bash
uv run pytest examples/bimanual_data_collection/tests -q
```

测试包含 300/60 Hz 更新次数和从臂先于主臂的顺序、独立按钮、录制边界、迟到数据归属、异常结束、60/20 Hz 原始数据与导出、因果时间对齐、
RGB 顺序、雅可比力矩投影和原力反馈公式的一致性；都不连接硬件。
