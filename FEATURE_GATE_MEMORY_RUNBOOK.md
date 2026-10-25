# Feature Gate + Memory 联合训练（B / G / GM）

## 当前状态（2026-10-24）

新增模型、序列输入、全量训练/评估、epoch 边界恢复和三组配置已经实现。
没有修改旧 Fusion A、数据增强、EventStore、前三阶段路由源文件，没有启动正式训练。

已执行：

- 12 项新增 CPU 合成测试 + 40 项旧路由回归，共 52 项通过，无 skip。
- 真实完整 epoch40 权重 CPU 严格恢复，SHA256 与历史来源配置一致。
- 真实 G/GM 公共新增参数逐项相等；原权重严格恢复后数值逐项相等。
- 完整 Train/Test 清单及 EventStore 身份检查；Train 10,296 视频 / 108,066 窗口，Test 2,555 视频 / 27,384 窗口。
- 少量真实 Train 窗口 CPU 读取、空间增强、三投影和 2048 点构建。输入 `[2,3,2048]`，各图像 `[2,3,224,224]`；首窗口 gap=0，下一连续片段首窗口仍为真实 gap=0.5 秒。
- 合成训练走正式 `run_pass`，验证两段以上 TBPTT、singleton 尾批、完整覆盖及 epoch 恢复后参数与不中断运行逐项相同。

**真实 CUDA 前向/反向、资源峰值及线上零残差数值回归未执行。** 检查时 GPU0/1 均为 100% 利用率，分别占用约 29,129 / 46,409 MiB（各 49,140 MiB）。没有占用他人的卡。以下默认资源预算尚待空闲 GPU 验收，不宣称已经可运行；如需下调，应统一修改 B/G/GM 的 video batch / TBPTT 长度。

## 文件与模型

- `gcstr_feature_memory_fusion.py`：保留原模型为 `base`，池化前交互、独立 Frame 路径、标量 Gate、共享 Memory、安全 mask、参数组。
- `gcstr_video_sequence_dataset.py`：视频排程、按片段读取 mmap、几何参数与 jitter/采点 RNG 分离、真实 gap、padding。
- `train_gcstr_feature_memory.py`：真实输入、TBPTT、确定性评估采样、完整日志、原子保存、恢复和独立 smoke。
- `evaluate_gcstr_feature_memory.py`：完整视频评估及与 B/历史参考的纠错比较。
- `test_gcstr_feature_memory.py`：CPU 合成测试，不冒充真实 EventMamba 验收。
- `configs/feature_memory/{B,G,GM}.json`：共同来源与预算。

| 组 | 保留原交互 | 独立 Frame + Gate | Memory | 可训练参数 |
|---|---|---|---|---:|
| B | 是 | 否（不注册无用模块） | 否 | 2,522,830 |
| G | 是 | 是 | 否 | 3,766,225 |
| GM | 是 | 是 | 16×128 单共享状态 | 4,118,225 |

共同参数：Point 928,233；原融合 1,594,597。新增参数：G 1,243,395，GM 1,595,395。冻结图像参数均为 33,684,975。

三路 ResNet 和原图像 FC 冻结且 eval；图像 adapters/位置映射可训练。Point 包括辅助分类头可训练，但每次 `train()` 后都只冻结其 BN 运行统计，BN affine 继续训练，其余 Point 模块不进入 eval。不使用 SyncBN/DDP。

每窗只编码一次图像和 Point。旧交互 `H0` 原样保留；新增 Frame 使用原静态 Q，不使用当前 Point prototypes；Frame 注意力和 MLP 复制旧模块的数值，参数独立。Q 只注册于原模块。

每 stage 一个 Gate，输入为 LN 后 P/F/历史池化及真实 `log1p(dt)`。Gate 末层为零，初值 0.5；补充 MLP 最后一层、Memory 输出投影为零，无双零残差标量。G/GM 用相同独立初始化随机流；Memory 用另一个流，不扰动数据 RNG。

Memory 在当前预测后写入，只读取过去；48 个候选融合 token 经 attention 写入 16 槽位。无历史行不调用读 attention，且 LN 后历史观测显式乘历史有效标记；padding 不进入网络或写入。状态由调用者显式传递，无全局视频字典。

## 权重、数据和实验解释

源：`/remote-home/datasets/event_project_cache/ucf101_filename_v2_fusion_A_seed0_v1_checkpoints/epoch_040.pt`

SHA256：`616c07083ba01df1f9cefc6b27c192775247e18066cf58f709a2753ca590eb62`

历史来源：`research_exports/ucf101_routing_epoch040_2026-10-23/static/config.json` 中的 checkpoint identity。先核对 SHA 和旧 checkpoint 的所有受保护源文件哈希，再仅处理 `module.`/`fusion.` 前缀、严格加载全部原参数；不依赖另外的预训练分类权重重新初始化原模型。

沿用 TXT split、原审计与 EventStore，窗口 1s / 步长 0.5s / 原事件数 >10,000。Point 使用 2048 点、采样后时间归一化、t/x/y 顺序、x/320、y/240；图像来自同一份增强后的全部窗口事件。Test 不增强，可复用旧投影缓存。没有新增大数据缓存。

`selection_protocol=original_test_as_val`、`independent_test=false`，按原 Test 主输出视频 NLL 选模。此结果不是独立 Test 估计。旧缓存 Top1 不是新在线 FPS 路径逐位复现目标；每组开始记录同流程初始化评估。历史 Frame / Fusion+Frame 只作为旧缓存参考，正式因果比较是同流程 B/G/GM。

B→G 同时添加独立 Frame 与 Gate，不能将差异全部归因于 Gate。g 是新增补充路径比例，不是整个模型的模态贡献。GM→G 是历史特征增量测试，不预设 Memory 必须有效。

## 序列、增强与目标

- 每 epoch 打乱视频，每个视频内部按 start_us 排序。两视频组成一组，长视频结束前不向该 lane 填入新视频。
- 每片段最多 4 窗口，仅当前片段事件/投影进入 GPU；DataLoader 预取在 CPU。尾部 padding 不编码、不监督、不投票，不复制 singleton。
- 同一视频跨片段保留状态数值，片段末 backward/step 后 detach；新视频组重置。这是 TBPTT 近似：下一片段使用由上一版参数产生的状态。仅支持 epoch 边界恢复，不支持中途片段恢复。
- 每视频/epoch 的旋转（±15°）和四边裁剪（各 0–10%）显式抽取一次；事件 jitter（±0.5 像素）与点采样用独立 RNG。不同重叠窗口的事件 jitter 不保证逐事件一致。
- 过滤为空时沿用旧增强的保留后裁剪 fallback，不静默丢弃监督窗口；日志记录 fallback 次数。
- 损失：主 CE + 0.5 × Point CE；每段有效窗口求和，再乘 `本 epoch 片段数 / 全部 Train 窗口数`，不会把长短尾段等权平均。
- 三组相同 seed 的窗口排程和优化步一致。第 1 轮 Train 19,472 步；后续打乱后的长短视频组合可能改变步数，每轮重新预先确定系数、完整记录。固定 Test 共 4,427 片段（不是优化步）。
- 视频预测是有效窗口 softmax 概率均值，不是 logits 均值。

## 默认共同预算

seed=0；50 epochs；每 GPU 两视频；TBPTT=4；FP32；AdamW；weight decay=1e-4；梯度裁剪=1；Point 辅助权重=0.5。

Point LR=1e-5；原融合 LR=5e-5；新增模块 LR=1e-4。无 scheduler、无 early stopping，无新增辅助损失。

日志包括主/Point 窗口 CE、Top1/5、视频 Top1/5/NLL/Brier/ECE；ECE 为 15 个等宽置信度箱，Brier 为所有类别平方误差之和。记录各 stage Gate、残差/H0 范数比、无历史 read、Memory read/state/write gate 的 count/mean/std/min/max，参数组 LR/梯度范数、步数/覆盖/吞吐/峰值显存。训练在线指标与固定权重评估分开。

## 验收命令（先确认卡空闲）

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -m unittest test_gcstr_feature_memory test_fusion_routing test_fusion_routing_stage2 test_fusion_routing_v3 -v
nvidia-smi
```

空闲卡上，真实 smoke 使用 3 个完整 Train 视频（45 窗口、当前第1轮9片段），不使用 Test。每次命令只跑一个小 epoch：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -u train_gcstr_feature_memory.py --config configs/feature_memory/GM.json --output experiments/feature_memory_GM_smoke_v1 --smoke
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -u train_gcstr_feature_memory.py --config configs/feature_memory/GM.json --output experiments/feature_memory_GM_smoke_v1 --smoke --resume
```

首次 smoke 除真实训练外还检查初始连续两窗的旧模型重复前向与新模型零残差前向：logits `rtol=1e-6, atol=1e-5`，概率 `rtol=1e-5, atol=1e-6`，Top1 不变；参数/buffer 回归严格相等。容差沿用旧真实推理回归，不根据观测误差临时放宽。必须实际出现跨片段和 singleton，有限损失/梯度，保存检查以及第二条命令恢复优化器/RNG 后才能算真实验收通过。

同样把 config 和输出改成 B、G 检查；B/G/GM 不同时占用一张繁忙卡。

## 正式训练 / 恢复 / 评估（尚未执行）

GPU smoke 通过且用户批准正式训练后，两张空闲卡可以分别跑 B 与 GM，完成后安排 G，不拆分数据：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -u train_gcstr_feature_memory.py --config configs/feature_memory/B.json --output experiments/feature_memory_B_seed0_v1
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -u train_gcstr_feature_memory.py --config configs/feature_memory/GM.json --output experiments/feature_memory_GM_seed0_v1
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -u train_gcstr_feature_memory.py --config configs/feature_memory/G.json --output experiments/feature_memory_G_seed0_v1
```

这些是前台命令，可在已有 tmux 会话中执行，或由操作者明确重定向日志后后台启动。这里没有自动队列或隐含的长时任务。输出目录必须为新目录，进程锁阻止重复启动。每组只保留 `last.pt` 和 `best_video_nll.pt`，不每轮永久保留整套权重；可通过 `--output` 指向有写权限且确认容量的独立目录。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 .venv-eventmamba/bin/python -u train_gcstr_feature_memory.py --config configs/feature_memory/GM.json --output experiments/feature_memory_GM_seed0_v1 --resume
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 .venv-eventmamba/bin/python evaluate_gcstr_feature_memory.py eval --config configs/feature_memory/GM.json --checkpoint experiments/feature_memory_GM_seed0_v1/best_video_nll.pt --output experiments/feature_memory_GM_seed0_v1/evaluation_best.json
```

可用 `--stop-after-epoch N` 在 epoch 边界保存并退出。恢复核对完整配置、新源代码、原源代码、checkpoint/data身份；含模型、优化器、Python/NumPy/Torch/CUDA RNG、data epoch、累计步数。CUDA 执行本身的 FP32 非确定性仍应按上述固定容差解释，不承诺硬件逐位复现。

监控：

```bash
tail -f experiments/feature_memory_GM_seed0_v1/metrics.jsonl
watch -n 2 nvidia-smi
```

纠错比较（完整覆盖和标签必须匹配）：

```bash
.venv-eventmamba/bin/python evaluate_gcstr_feature_memory.py compare --reference experiments/feature_memory_B_seed0_v1/predictions_best.json --candidate experiments/feature_memory_GM_seed0_v1/predictions_best.json --output experiments/feature_memory_GM_seed0_v1/vs_B.json
.venv-eventmamba/bin/python evaluate_gcstr_feature_memory.py compare --reference reports/routing_stage2_full_v1/frame__normal__prefix_1/video_index.json --candidate experiments/feature_memory_GM_seed0_v1/predictions_best.json --output experiments/feature_memory_GM_seed0_v1/vs_historical_Frame.json
.venv-eventmamba/bin/python evaluate_gcstr_feature_memory.py compare --reference reports/routing_stage2_full_v1/fusion_frame__normal__prefix_1/video_index.json --candidate experiments/feature_memory_GM_seed0_v1/predictions_best.json --output experiments/feature_memory_GM_seed0_v1/vs_historical_FusionFrame.json
```

历史对照不是重新训练，在线采样/FPS差异必须和模型改动分开解释。尚无本轮方法效果结论。

## 等待空闲 GPU 自动 smoke（脚本已测试，尚未启动）

`wait_gpu_feature_smoke.py` 使用标准库及 nvidia-smi，等待期间不导入 Torch、不分配显存。
默认每 10 秒检查 GPU0/1：连续 120 秒无计算进程、显存 ≤512 MiB、利用率 ≤5% 后，
获取同用户启动器的建议锁，再次检查并通过 GPU UUID 绑定设备。查询失败会清零计时，
最长等 24 小时；只运行 GM smoke 首次和 resume 两条命令，完成或失败后退出，不启动正式训练。
它不能阻止其他用户同时启动任务，不是 Slurm 或独占预约；使用前应符合实验室 GPU 分配规则。

启动与监控（在项目根目录执行）：

```bash
nohup python -u wait_gpu_feature_smoke.py > feature_gpu_wait.log 2>&1 &
echo $!
tail -f feature_gpu_wait.log
```

记录 `echo $!` 返回的等待器 PID，取消时 `kill <该PID>`；若已进入 smoke，也会终止本启动器的子进程，不操作他人任务。
关闭窗口不影响 nohup 任务。默认结果目录为 `experiments/feature_memory_GM_auto_smoke_v1/smoke/`，
fresh/resume 详细日志位于其父目录的 `launcher_fresh.log` / `launcher_resume.log`。
已有实验结果拒绝覆盖；再次执行请用新的 `--output experiments/feature_memory_GM_auto_smoke_v2`。

只读检查：`python wait_gpu_feature_smoke.py --check-only`。
可用 `--gpus 1` 仅等 GPU1，`--timeout-hours 12` 修改最长等待时间。
`python -m unittest test_wait_gpu_feature_smoke -v` 的 6 项测试通过；测试不会启动真实 GPU 任务。
