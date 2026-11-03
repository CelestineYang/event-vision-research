# Event vision research

本项目研究 gCSTR 三路事件投影与 EventMamba 点特征的融合，以及 Gate 和跨窗口
Memory 的消融。涵盖 THU-CHL、UCF101-DVS、DVSAction 和 DailyDVS-200。

**最新完整结果：[2026-11-03 主实验与消融汇总](research_exports/main_ablation_2026-11-03/README.md)**。
该快照包含已完成实验的完整曲线、选模指标、逐视频预测、数据与权重来源校验、
未完成消融清单，以及单独列示的历史概率路由实验。

| 已完成方法 | 视频 Top-1 | 说明 |
|---|---:|---|
| THU-CHL G | 80.6985% | 独立预训练分支，融合重新初始化，epoch 40 |
| THU-CHL GM | 81.2500% | 相同核心超参数，epoch 10；与 G 的单卡/DDP 执行有差异 |
| DVSAction GM | 89.4737% | 独立预训练分支，融合重新初始化，epoch 47 |
| UCF101-DVS Fusion A | 96.1252% | 历史 150 轮训练的 Top-1 峰值，epoch 40 |
| UCF101-DVS 历史 GM | 96.7123% | 从 Fusion A epoch 40 续训；峰值 epoch 47 仅保留日志 |

这些实验使用原始 Test 选模，不能称为未参与模型选择的独立测试结果。
UCF 历史结果不属于独立初始化的同协议消融；可用的历史 GM NLL 最佳权重对应
96.2427%。UCF 独立初始化 GM 和 DailyDVS-200 当前窗口协议下的融合主实验尚无正式结果。

## 代码与复现入口

| 内容 | 入口及说明 |
|---|---|
| 项目衔接与实验边界 | [PROJECT_CONTINUITY.md](PROJECT_CONTINUITY.md) |
| B/G/GM 特征消融 | `gcstr_feature_memory_ablation.py`、`train_thu_chl_ablation.py` |
| M：基础融合加 Memory | [THU_CHL_MEMORY_ONLY.md](THU_CHL_MEMORY_ONLY.md)、`train_thu_chl_memory_only.py` |
| UCF 独立初始化 GM | [UCF101_FRESH_GM_RUNBOOK.md](UCF101_FRESH_GM_RUNBOOK.md) |
| UCF 历史 Fusion A | [UCF101_FILENAME_V2_FUSION_RUNBOOK.md](UCF101_FILENAME_V2_FUSION_RUNBOOK.md) |
| DVSAction 预处理、三路投影 | [dvsaction_preprocessing/README.md](dvsaction_preprocessing/README.md) |
| DVSAction GM | [DVSACTION_GM_RUNBOOK.md](DVSACTION_GM_RUNBOOK.md) |
| DailyDVS-200 窗口图像适配 | [DAILYDVS200_PREPARED_ADAPTER.md](DAILYDVS200_PREPARED_ADAPTER.md) |
| DailyDVS-200 点分支训练 | [EVENTMAMBA_PAPER_REPRODUCTION.md](EVENTMAMBA_PAPER_REPRODUCTION.md) |
| 本轮结果导出 | `export_main_ablation_results.py` |
| 路径模板 | `configs/examples/`，按各数据集分别使用 |

B 是基础 prototype 融合；G 增加完整 PF 补充残差及其 Gate；M 删除 PF 补充残差、
仅在 B 上增加 Memory；GM 同时包含二者。M 仍有记忆写入门，B→G 也不只改变一个 Gate。
历史概率路由的 G1/G2 等命名与这里的特征级 G/GM 不同。

模板中的 `LOCAL_PATH`、`DATA_ROOT`、`ASSET_ROOT` 等是需要替换的字面占位值，
不是自动展开的环境变量。训练入口验证实际数据和独立预训练权重的身份。
THU 使用 EventMamba V1，UCF 和 DVSAction 使用 V2；不得互换检查点或数据协议。

## 数据和验证

仓库保存数据准备代码、划分清单、轻量指标和来源校验，不托管原始事件、完整图像/点云
缓存、模型权重或虚拟环境。DVSAction 的两个划分文件在
`dvsaction_preprocessing/splits/`；UCF 的清单是根目录 `train_ucf.txt` 与 `test_ucf(1).txt`。
完整数据与权重仍需在本地配置。见 [Git 管理约定](GIT_WORKFLOW.md)。

在已安装项目依赖的 Python 环境中运行本轮 CPU 检查，不加载真实权重、不使用 GPU：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python -m unittest test_gcstr_memory_only test_eventmamba_paper test_ucf101_fresh_gm test_main_ablation_results -v
```

真实 CUDA smoke、显存探测和正式训练均需显式执行；导入代码或阅读文档不会自动启动。
本次发布在隔离目录整理，未热更新正在运行的训练依赖。
