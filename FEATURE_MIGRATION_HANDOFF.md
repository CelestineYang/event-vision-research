# 给新服务器 Codex：Feature Gate–Memory 实验迁移交接

## 状态与授权

这是独立副本，原实例代码、数据、checkpoint 均未移动或删除。仅 B/G/GM 特征联合训练相关代码及依赖被复制，不含整个项目、Git、虚拟环境、旧路由 logits 或无关实验权重。
旧服务器已通过 52 项 CPU 测试、真实权重 CPU 严格加载和少量真实数据 CPU 读取；**尚未完成真实 GPU smoke，也未启动正式训练**。目标有四张 RTX3090，先检查实际占用及单卡显存，不能假定显存足够，也不能把四张卡当作一张大显存卡。

包根目录以下称 BUNDLE。先确认 `COMPLETE.json` 的 status 为 verified，缺少该文件表示复制或校验尚未完成。

## 包内容

- `code/`：本轮入口、模型/数据层、必要依赖、源任务文档、运行说明、B/G/GM 原配置及路径迁移工具。
- `weights/epoch_040.pt`：完整 Fusion A 权重。SHA256 `616c07083ba01df1f9cefc6b27c192775247e18066cf58f709a2753ca590eb62`。
- `data/event_store/`：完整 Train/Test mmap 事件流与索引（约145 GiB）。
- `data/experiment/`：原审计、窗口清单、视频划分、类别映射，以及实际 Test 三投影缓存（约47 GiB）。不是指向旧实例的软链接。
- `environment/versions.json`：当前已验证环境版本。
- `environment/mamba_build/`：原构建脚本/CUDA 源码 + 当前实际使用的 BiMamba-v2 Python 实现，含 causal-conv1d 1.4 接口适配。不要换成标准 pip mamba-ssm 后假定兼容。
- `MANIFEST.json` / `SHA256SUMS`：每个复制文件的来源、大小、SHA256。

清单中的旧绝对 source_path 仅用于审计身份，**不要批量替换清单或修改审计哈希**。当前训练从 EventStore 读取，不依赖那些旧原始 NPZ 路径。完整 Fusion A 已包括三路图像权重和 Point 权重，不必另复制旧图像 checkpoint 或 best_checkpoint.pth。

## 在新实例准备工作目录

数据集挂载可能只读，推荐只把小体积 code 复制到可写工作目录，大数据保持挂载读取。以下路径需按新机器实际修改：

```bash
export BUNDLE='/实际挂载路径/ucf101_feature_memory_migration_20261024_v1'
export WORK='/可写工作目录/feature_memory'
mkdir -p "$WORK"
cp -a "$BUNDLE/code/." "$WORK/"
cd "$WORK"
```

WORK 应为新的空目录，勿覆盖已有代码。首次建议校验完整副本（约192 GiB读取，耗时取决于存储）：

```bash
cd "$BUNDLE"
sha256sum -c SHA256SUMS
cd "$WORK"
```

生成新路径配置，不改原代码和审计文件：

```bash
python prepare_feature_migration_config.py --bundle "$BUNDLE" --output configs/feature_memory_target
```

生成后的 experiment/store/checkpoint 均指向新挂载绝对路径；其他训练协议不变。输出配置目录必须不存在。

## 环境（目标服务器必须重新验证）

当前 Python3.10，Torch2.0.1+cu118，torchvision0.15.2+cu118，NumPy1.26.3，SciPy1.13.1，causal-conv1d1.4.0，定制 Mamba1.1.1，einops0.7.0，transformers4.35.2，triton2.0.0。

不要复制旧 .venv（继承旧绝对路径/系统包）。在新实例建立 Python3.10 虚拟环境，检查驱动、CUDA toolkit/nvcc 和编译器。参考安装顺序如下；下载需要目标机器网络，本包不是完全离线 wheel 仓库：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install 'pip<25' wheel setuptools packaging ninja
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.3 scipy==1.13.1 einops==0.7.0 transformers==4.35.2
MAX_JOBS=4 python -m pip install causal-conv1d==1.4.0 --no-build-isolation --no-deps
cp -a "$BUNDLE/environment/mamba_build" "$WORK/mamba_build"
MAMBA_FORCE_BUILD=TRUE MAX_JOBS=4 python -m pip install ./mamba_build --no-build-isolation --no-deps
```

构建在 WORK 的副本中完成，不能往只读数据挂载写 build 文件。若 CUDA/toolchain 不兼容，先修环境，不静默替换模型或修改受保护源码。当前 FPS 为 `eventmamba_v2_layers.py` 的 Torch CUDA 张量实现，不依赖 pointnet2_ops；Mamba/causal-conv1d 扩展仍必须真实执行 CUDA。

安装后确认 `Mamba.__init__` 包含 `bimamba_type`，确认已安装 mamba_ssm 的 .py 源码与包内逐文件一致，再执行 CPU 测试、完整权重加载和 GPU smoke。目标环境尚未经原实例验证，不承诺以上安装在所有驱动/toolchain 上直接成功。

## 正式训练前的验收

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 python -m unittest test_gcstr_feature_memory -v
nvidia-smi
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -u train_gcstr_feature_memory.py --config configs/feature_memory_target/GM.json --output runs/GM_smoke_v1 --smoke
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -u train_gcstr_feature_memory.py --config configs/feature_memory_target/GM.json --output runs/GM_smoke_v1 --smoke --resume
```

先确认 GPU0 空闲再执行。每次 smoke 只使用3个完整 Train 视频，不使用 Test，不是正式效果实验；检查连续片段、singleton、finite梯度、显存、零残差回归及断点恢复。B/G 也分别用新的输出目录验收。

若 24GB OOM，统一降低三组 video_batch_size_per_gpu 或 tbptt_steps（例如先 batch=1）；保留相同采样/窗口覆盖、FP32及损失，不单独给GM减预算，不擅自开 AMP。记下最终三组共同预算。

## 正式运行安排

验收通过并获得启动许可后，B/G/GM 可以各用一张空闲3090，第四张保留。不需要新增DDP，三组各自使用完整数据。下面命令在不同 tmux 窗口执行，非自动提交队列：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 python -u train_gcstr_feature_memory.py --config configs/feature_memory_target/B.json --output runs/B_seed0_v1
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 python -u train_gcstr_feature_memory.py --config configs/feature_memory_target/G.json --output runs/G_seed0_v1
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 python -u train_gcstr_feature_memory.py --config configs/feature_memory_target/GM.json --output runs/GM_seed0_v1
```

恢复在原命令后加 `--resume`；只支持 epoch 边界。输出目录必须可写且容量足够，不写到共享只读数据集。保持50 epochs、AdamW、Point LR1e-5/原融合5e-5/新模块1e-4、TBPTT4、video batch2、Point辅助0.5、无调度/早停；资源调整须三组一致。

本次仍以原Test作为Val按视频NLL选模，`independent_test=false`。不要改成独立测试协议的叙述。详细指标、结构、状态流转与局限见 code/FEATURE_GATE_MEMORY_RUNBOOK.md。

如旧服务器已手动启动 wait_gpu_feature_smoke.py，请先核实并由用户决定是否取消，避免重复实验；本次迁移操作不会自动停止任何旧进程。
