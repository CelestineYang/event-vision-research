# THU-EACT-50-CHL 独立预处理包

下载本文件夹即可，不依赖原项目目录、GPU、PyTorch 或 EventMamba 模型。
本包针对 **CHL NPY**，不是标准 THU 的 CSV，也不读取已经采样的 HDF5。

## 核实后的协议

官方 EventMamba `dataprocess/generate_thu.py` 处理 CHL：
1 秒窗、0.5 秒步长；x/346、y/260；窗口时间归一化后在 (t,x,y) 空间使用
Open3D radius=0.01、nb_points=5 去离群点；过滤后至少 100 个事件才保留；
有序随机采样 2048 点（不足则有放回），采样后再次 min/max 归一化时间。
模型输入顺序为 t,x,y，输出 HDF5 为 [windows,2048,3]，另有 label/mark。

官方窗口函数有特殊边界：end 必须严格小于最后时间；找到最后一个 t<=end
事件的下标后，将该下标作为 Python 切片的排他结束位置。本包保留这个行为，
不替换为普通半开窗。首尾不足整窗的片段不补窗；同一视频所有窗口继承其 split。
不沿用 UCF 的 >10000 原始事件阈值或最多 39 个窗口限制。

原始 NPY 顺序 x,y,t,p，分辨率 346x260，时间按官方生成代码解释为微秒，p=0/1。
缓存保存原 NPY 全部事件和极性，无下采样、无空间增强、无自动时间修复。
图像使用有效窗口索引内的全部原始事件；点云使用同窗经过离群点过滤后的事件。
帧与点云窗口一一对应，但帧的全事件表示是本项目融合适配，不是 EventMamba 的原始输入。

**必须使用发布的 train.txt/test.txt 标签。文件名 A编号不等于标签！**
程序拒绝跨 split 重复样本、缺失文件、非法值、乱序时间、错误极性和越界坐标。
丢弃的窗口仍记录到 JSONL，point_row=null。原始视频的 mark 在每个 split 内独立编号。
官方测试的 Majority Voting 可使用 mark；样本级预测须进一步关联 samples.json，不能跨 split 混合 mark。

## 与公开实现的明确差异及未确认事项

- 官方随机采样未固定种子；这里用 seed+sample_id+window_id 派生独立 RNG，便于重现。
  不承诺与作者预生成 HDF5 逐位相同。
- 时间跨度为零的窗口被记录为无效，避免官方代码除零；全零 x/y 不进行无用的临时除法。
- Open3D 固定 0.18.0；作者实际版本未确认，邻居计数边缘情况可能有差别。
- 完整事件只保存一次，窗口共享索引；HDF5 增量写入，不把整个数据集装入内存。
- 当前无断点恢复。输出目录必须全新，失败数据保留，status.json 明确标记失败。
- README 的配置行写 THU-CHA、下载行写 THU-CHL，列出 V1、[64,128,256]、groups=[1024,512,256]。
  **尚未检查实际权重；本包仅对齐公开输入处理，不能保证某个 checkpoint 可直接加载。**
  后续仍须核对权重架构、类别映射、划分、采样协议与官方训练日志。

## 新服务器运行

Python 3.10 或 3.11，新建虚拟环境后：

```bash
pip install -r requirements.txt
python -m unittest test_preprocessing -v
python prepare_chl.py --root '/挂载目录/THU-EACT-50-CHL/THU-EACT-50-CHL' \
  --output '/可写大容量目录/chl_smoke' --smoke-per-split 2
```

默认读取 root/train.txt 与 root/test.txt，可用 --train-list / --test-list 覆盖。
先确认 smoke 的 status 为 complete、scope 为 smoke，然后用新输出路径全量执行：

```bash
python -u prepare_chl.py --root '/挂载目录/THU-EACT-50-CHL/THU-EACT-50-CHL' \
  --output '/可写大容量目录/chl_prepared' > chl_prepare.log 2>&1
```

需要预生成三张图时，可另运行（占用空间较多，也可在训练 Dataset 内按索引在线调用 render）：

```bash
python render_frames.py --store '/可写大容量目录/chl_prepared' \
  --output '/可写大容量目录/chl_frames'
```

投影为 cstr3/xt_p/ty_p、[3,224,224]，尚未应用 ImageNet 标准化。
图像缓存用于无增强评估；若训练需要事件空间增强，应读取 events 后增强再投影，
且点云和图像使用同一增强。不要把固定图像缓存当作事件级增强的等价替代。

输出：train.h5/test.h5、samples.json、config.json、status.json、两个窗口 JSONL、
train/test 下完整原始 NPY 与每视频 SHA256/元数据。只有 status complete 且 HDF5
complete=True 才能被下游当作完成产物；smoke 数据不得当全量实验使用。

## 来源

- https://github.com/rhwxmx/EventMamba/blob/main/dataprocess/generate_thu.py
- https://github.com/rhwxmx/EventMamba/blob/main/dataprocess/uti.py
- https://github.com/rhwxmx/EventMamba
- https://github.com/lujiaxuan0520/THU-EACT-50/blob/main/dataset.py
- https://github.com/lujiaxuan0520/THU-EACT-50

查阅日期 2026-10-22。窗口和点采样为参考公开算法的独立实现；
gcstr_representations.py 来自当前项目，render_frames.py 复用现有 UCF 三投影计算方式。
当前实例没有安装 Open3D，因此不能把本地基础测试当作真实离群点过滤全流程已验收；
请在目标服务器安装 requirements 后运行上述真实 smoke。

本地已执行 4 项基础测试并通过；另外只读检查本地发布列表：Train 1786 个视频、
50 个标签；Test 544 个视频、46 个标签，两侧样本不重复且文件均存在。
该类别覆盖差异保留到 split_audit.json，不自动重划分或补标签；迁移后应与自己
下载的官方列表逐项/哈希核对。本次未验证本地列表与网盘原包逐字节相同。
Train/Test 各一条真实样本均成功读取并生成三张 [3,224,224] 投影。

## 三路 gCSTR 独立预训练

新增 `train_gcstr.py`，与本文件夹的 gcstr_models/gcstr_augmentation/render_frames 配合。
在目标服务器安装匹配的 PyTorch/torchvision（可参考原环境 torch 2.0.1+cu118、
torchvision 0.15.2+cu118），再安装 requirements-train.txt。训练本身不调用 Open3D。
只支持单 GPU 或 CPU；不要用 torchrun 启动多份同一输出目录。

三支 cstr3/xt_p/ty_p 各有 ImageNet-pretrained ResNet-18 和 50 类分类头。
每批窗口共享全事件读取、空间增强和投影生成；三支依次前向/反向更新，各有 Adam，
不做分支融合。训练使用全部有效窗口，Test 无增强、窗口概率算术平均得到视频预测。
训练记录的视频指标来自在线更新中的模型；Test 指标来自当轮固定权重。

默认 50 epochs、batch=64（窗口数）、LR=1e-4、Adam、weight_decay=0、
无 scheduler、无 early stopping、FP32；训练时骨干可训练，未来融合才冻结。
沿用用户此前实验协议，需显式 `--test-as-val`：Test 视频级 NLL 选模，
`independent_test=false`，不应把它描述为独立测试成绩。
不另划 Val；只训练清单中 valid=true 的窗口（与 Point 保持窗口对齐）。

```bash
# 基础与合成训练测试：不使用真实 Test，也不下载 ImageNet 权重
OMP_NUM_THREADS=2 python -m unittest test_preprocessing test_training -v

# 用预处理得到的小规模 store 做流程检查；随机初始化，不能当正式结果
CUDA_VISIBLE_DEVICES=0 python train_gcstr.py --store /data/chl_smoke \
  --output /work/chl_gcstr_smoke --smoke --batch-size 4 --workers 0 --test-as-val

# 正式：必须是 scope=full 的预处理产物；输出为全新目录
CUDA_VISIBLE_DEVICES=0 python -u train_gcstr.py --store /data/chl_prepared \
  --output /work/chl_gcstr_run1 --epochs 50 --batch-size 64 --workers 4 \
  --test-as-val > /work/chl_gcstr_run1.log 2>&1

# 续训：相同配置、相同输出目录，增加 --resume
CUDA_VISIBLE_DEVICES=0 python -u train_gcstr.py --store /data/chl_prepared \
  --output /work/chl_gcstr_run1 --epochs 50 --batch-size 64 --workers 4 \
  --test-as-val --resume >> /work/chl_gcstr_run1.log 2>&1
```

首次正式运行 torchvision 可能需要下载 ImageNet 权重；离线实例先准备 torch hub 缓存。
每支保存 `best_loss.pt`、`last.pt`、`metrics.jsonl`、`predictions_best.json`。
共同 `resume.pt` 保存三个模型/优化器/随机状态，以完整 epoch 为续训边界；
源码、数据清单和配置改变时拒绝续训。中途失败可能留下当轮未提交的分支日志，
恢复时以 resume.pt.epoch 为已完成边界，不把额外日志行当有效训练历史。
status.json 记录完成 epoch；控制台含每支窗口/视频 Top1、Top5、NLL，
日志另含当轮耗时、峰值 CUDA 显存。

后续冻结加载（无需下载 ImageNet 权重）：

```python
from train_gcstr import load_frozen_branch
model = load_frozen_branch('/work/chl_gcstr_run1/cstr3/best_loss.pt', device='cuda:0')
# 模型已 eval 且 requires_grad=False；融合父模型调用 train() 时须继续保持该分支 eval。
```

当前训练不读取预生成 frame cache，而是在线事件空间增强后渲染；
Test 也在线渲染，保证独立包无需配置旧 UCF 缓存路径。
