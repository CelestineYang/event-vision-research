# UCF101 第 40 轮冻结专家路由实验：迁移交接

## 当前状态与授权边界（2026-10-21）

已经创建独立目录：
`/remote-home/datasets/Joind-dataset for Action Recognition/ucf101_routing_epoch040_migration_20261021`

目前目录为空：尚未复制任何权重、数据或代码，尚未启动全量导出或 Static。
这份文档是准备与交接，不是宣称迁移完成。不要删除或移动原始文件。
平台可能挂载只读数据集快照，创建新实例前应确认复制完成后再发布/刷新数据集。
代码放新实例的可写工作目录，数据集只读挂载；缓存与输出需要另选可写且容量足够的位置。
不能假定 datasets 路径在新实例可写，也不能假定数据集新增内容自动同步。

## 下载代码清单

保持相对路径下载以下文件（最新工作区，而不是仅下载 Git HEAD）：

### 路由入口、配置与测试

```text
export_fusion_evidence.py
fusion_evidence_dataset.py
routing_data.py
fusion_router.py
train_fusion_routing.py
prepare_fusion_routing_roles.py
test_fusion_routing.py
configs/fusion_routing.json
configs/fusion_routing_roles.original_test_as_val.json
configs/fusion_routing_roles.template.json
FUSION_ROUTING.md
CODEX_ROUTING_REVISION_TASK.md
ROUTING_MIGRATION_HANDOFF.md
train_ucf.txt
test_ucf(1).txt
```

### 必须原样保留的 checkpoint 校验源码（即使不运行对应训练入口）

```text
train_ucf101_pretrained_fusion.py
ucf101_pretrained_fusion.py
ucf101_fusion_window_data.py
eventmamba_v2_pretrained.py
eventmamba_v2_model.py
eventmamba_v2_layers.py
ucf101_event_store.py
gcstr_eventmamba_fusion.py
gcstr_secnet_fusion.py
train_ucf101_window_gcstr.py
ucf101_window_data.py
gcstr_augmentation.py
gcstr_representations.py
gcstr_models.py
```

### 间接导入、测试与许可文件

```text
eventmamba_encoder.py
secnet_encoder.py
analyze_ucf101_complementarity.py
test_ucf101_window_gcstr.py
test_ucf101_event_store.py
preprocess_ucf101_event_store.py
train_ucf101_window_joint.py
eventmamba_v2_upstream/LICENSE
```

以上清单针对导出与路由训练；不需要运行或迁移以前所有实验。
若希望保留所有项目测试，直接下载当前项目的全部小型源码/配置/文档也可以，
但不要打包 .git、虚拟环境、历史 checkpoints、datasets、cache、旧实验产物。

## 待复制的数据（下一步；当前未复制）

以本次独立迁移目录为目标根，建议组织如下：

| 来源 | 目标相对路径 | 必需性 |
|---|---|---|
| `/remote-home/datasets/event_project_cache/ucf101_filename_v2_fusion_A_seed0_v1_checkpoints/epoch_040.pt` | `checkpoints/epoch_040.pt` | 必需，已读取确认为 epoch 40；约 158 MiB |
| `/remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_events_v1/` | `events/` | 必需，完整目录含 summary.json、videos.json、逐视频 metadata.json/events.npy；历史约 145 GiB，复制前重新统计 |
| 项目 `experiments/ucf101dvs_gcstr_sliding_txtsplit_v2/window_manifest.csv` | `experiment/window_manifest.csv` | 必需 |
| 同目录 `video_splits.csv` | `experiment/video_splits.csv` | 必需 |
| 同目录 `class_to_idx.json` | `experiment/class_to_idx.json` | 必需 |
| 同目录 `dataset_audit.json` | `experiment/dataset_audit.json` | 必需 |
| `/remote-home/datasets/event_project_cache/ucf101_txtsplit_v2_test_images_v1/` | `test_images/` | 可选加速，历史约 47 GiB；先检查布局再关联 experiment/cache |

事件缓存已含 t/x/y/p，当前导出使用 EventStore，不必再复制原始压缩包。
Test 图像不存在时，FusionWindowDataset 会用窗口全部事件重新生成三投影。
完整 Fusion 权重恢复不需要重新加载原始 ImageNet、图像最佳权重或官方 Point initializer。
不要仅复制指向旧实例的软链接；核验真实文件、相对目录结构、文件数、容量及 SHA256。
复制大目录前检查剩余容量和共享目录管理规则；不要覆盖已有同名目录。

## 新实例 Codex 必须处理的路径与环境事项

1. 保持原 checkpoint 字节不变，计算 SHA256 留档；不要重写 checkpoint 内 config。
2. 当前 export_fusion_evidence.py 从 checkpoint config 读取旧绝对路径 experiment/store。
   它目前没有迁移路径覆盖参数。在不改 14 个受保护源码的前提下，为 exporter 增加
   显式 `--experiment` / `--store` 路径覆盖，传递给审计及 Dataset；记录新旧路径。
   不得删除源码/清单哈希校验，不得更改原 CSV 字节来替换路径。
   原 manifest 的 source_path 可以保留为历史来源；EventStore 实际读取迁移后的事件缓存。
3. 如使用 prepare_fusion_routing_roles.py --check，先检查它的默认旧路径参数并适配调用。
   已有完整角色映射可直接携带，不要重新划分数据或重新生成不同哈希的角色文件。
4. 原环境版本：Python 3.10.14；torch 2.0.1+cu118、torchvision 0.15.2+cu118、
   numpy 1.26.3、scipy 1.13.1、mamba-ssm 1.1.1、causal-conv1d 1.4.0。
   **mamba 包版本号不足以复现环境**：必须支持 Mamba.__init__(bimamba_type=...)。
   在原实例保存安装来源/commit 或对应源码再重建；普通上游同名包不保证兼容。
   安装元数据记录来源为 `file:///tmp/eventmamba-vim-dd0358ad-build`，不是可移植下载地址；
   该路径是否仍有源码及其实际 commit 尚需核验，不能仅凭目录名确认版本。
   CUDA 扩展根据新实例 Python/PyTorch/CUDA 重装编译，不复制旧 venv/二进制扩展。
5. 原 14 个源码的哈希应逐项匹配 checkpoint config.code。新代码下载时避免自动换行转换。
6. 不要直接运行历史训练脚本；首次运行先检查 GPU、只读数据权限及可写输出容量。

## 已确认实验计划（优先于旧文档默认值）

- 专家：仅 epoch_040.pt，不做 epoch 6 对照；五头顺序 fusion/point/cstr3/xt_p/ty_p。
- 全部专家冻结；2048 点、官方归一化、窗口全事件图像；导出不增强。
- 原 Train 全部有效视频用于拟合，原 Test 验证并按视频 NLL 选模。
- selection_protocol=original_test_as_val，independent_test=false。
- Train 10296 视频 / 108066 窗口；Test 2555 视频 / 27384 窗口。
- Static：80 epochs、global batch 32 视频、Adam LR 1e-3、weight_decay=0、clip=1、seed=0，不早停。
- 当前 configs/fusion_routing.json 仍是 20 epochs / batch 8；新运行前写独立运行配置，不能照旧默认启动。
- 当前 exporter 和训练入口均是单进程，并未实现 DDP；双卡测试尚未执行成功，不能宣称已支持。
- Static 只有两个参数，双卡不保证更快。双卡导出如实现，需要显式分片、覆盖/重复校验和共同身份；
  分片导出涉及 FPS 随机流，必须记录。不能直接 torchrun 原入口（exporter 会拒绝 WORLD_SIZE>1）。
- 只先启动 Static；后三种 Current/GRU/Conditional 需加载同一 Static 最优参数和标准化器，之后才并行。
- 旧文档中 best_loss.pt / epoch6 / 20epochs / batch8 的示例不是本次最终运行配置。

## 新实例验收顺序

1. 核验下载文件、14 项源码哈希、epoch40 权重、审计/角色/事件缓存完整性。
2. 重建环境，CPU 测试 `python -m unittest test_fusion_routing test_ucf101_window_gcstr test_ucf101_event_store -v`。
3. 路径覆盖完成后，单卡 batch1 执行 exporter 的 --integration-regression（不写证据缓存）。
4. Train/Test 各导出两个完整视频，验证五头缓存、17 维观测、真实 gap、冻结状态、身份匹配。
5. 做 Static 小规模训练检查；如实现双卡，另验单/双卡损失、全局 batch 和样本覆盖。
6. 全量导出到新实例可写的大容量目录；目前原实例尚无可直接搬走的全量证据缓存。
7. 确认数量完整再运行 Static 80 轮，保存配置/日志/best_val_nll.pt/last.pt。
   当前训练入口没有 resume 命令；不要把存在 optimizer/RNG checkpoint 误认为已支持 CLI 续训。

以上未执行步骤不得报告为已通过；新实例先检查实际资源与路径，再启动用户授权的工作。
