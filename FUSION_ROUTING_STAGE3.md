# 第三阶段：外层 Gate 与视频目标

## 已交付与停止边界

基于 `e19ff10` 新增独立 V3，不修改 `fusion_router.py`、`routing_data.py`、
`train_fusion_routing.py` 及历史行为，不重导出缓存，不重训专家，不增加 Memory。
模型 architecture=`current_alpha_outer_eta_v3`，model_schema=3；数据缓存仍为schema2。
代码验收后，经用户确认已完成 G1–G4 的真实训练：各80轮、25760步，四个独立CPU后台进程均退出。
启动记录为 `logs/routing_stage3_launch.json`，日志为 `logs/routing_stage3_G*_seed0.log`。
四个最优checkpoint复评均通过；完整配对、Gate诊断及曲线见 `reports/routing_stage3/full_analysis_v1/report.md`。
完整分析后，经用户批准完成G1/G2各seed1/2，共4次补充训练，每次80轮；只改变seed，源码与其他配置同seed0。
启动记录为 `logs/routing_stage3_multiseed_launch.json`。8个选中模型均已复评，指标一致。
三种子配对及最终结果快照见 `research_exports/ucf101_routing_stage3_2026-10-24/README.md`。
没有运行 B1/B2、Memory扩展或任何GPU任务。

已执行：

- 40项CPU单元/回归测试全部通过，包括4个主配置各2轮微型合成缓存smoke。
- sources.json真实产物SHA、缓存全文件完整性、角色/类别/观测/专家身份只读检查。
- 真实4视频小批：三种eta初始化与原Static完整log-probability最大差均为0；
  显式复制旧Current epoch58参数后的最大差也为0。此项不是重训，也不是全Test回归。
- 一次性Train专家与固定组合基线统计（CPU读取已有logits），与已有同缓存选模集结果并列。

报告：

- `reports/routing_stage3/cpu_tests.log`
- `reports/routing_stage3/audit_v1/audit.json`
- `reports/routing_stage3/train_baselines_v1/analysis.md`、`summary.json`

审计时4视频仅Gate前向约1.7–2.3ms（CPU、2线程），不是整轮耗时估计。
训练启动前不需要重新执行22项第二阶段suite。

## 文件与主配置

| 文件 | 用途 |
|---|---|
| fusion_router_v3.py | 独立alpha/eta Gate、三种可微loss及epoch系数 |
| train_fusion_routing_v3.py | 共同初始化、严格loader、epoch0、fit、指标和梯度日志 |
| evaluate_fusion_routing_v3.py | audit、一次性Train基线、eval、统一配对表 |
| test_fusion_routing_v3.py | 数学/梯度/身份/保存恢复/4配置smoke |
| configs/routing_stage3/*.json | G1–G4与条件性B1/B2 |

| ID | eta_mode | loss_mode | 可训练参数 |
|---|---|---|---:|
| 回归接口（不重训） | fixed | window_nll | 1217 |
| G1 | global | window_nll | 1218 |
| G2 | current | window_nll | 2435 |
| G3 | global | video_nll | 1218 |
| G4 | current | video_nll | 2435 |
| B1（条件性） | global | video_balanced_window_nll | 1218 |
| B2（条件性） | current | video_balanced_window_nll | 2435 |

统一80epochs、batch32完整视频、Adam lr1e-3、wd0、clip1、hidden64、seed0、FP32；
无scheduler、early stopping、AMP、DDP、额外损失或均衡约束。
真实Train 10296视频、108066窗口；每轮322步，总25760步，尾批不丢弃。
alpha为17→64→1/Tanh；current eta是独立同结构MLP。
公共alpha初始化使用隔离的seed随机流，eta使用seed+100003，增加eta网络不会改变alpha初始化。

初始化仅用原unrestricted Static epoch10的原始alpha_logit/eta_logit。
b_alpha冻结；fixed eta用同dtype sigmoid，global/current的b_eta是可训练参数。
修正输出层零初始化；保留有限logit饱和与显式endpoint的区别，不截断后重新求logit。
两路MLP只读取原17维Train标准化观测，不读取label/video_id/真实类别概率。
标准化器严格复制原Static状态；不重新fit。

## 训练目标与保存

- window_nll：所有有效窗口等权；batch目标为窗口NLL之和 × 322 / 108066。
- video_nll：在梯度图内mask后logsumexp，再减log有效窗口数，视频NLL等权；batch目标为视频NLL之和 × 322 / 10296。
- video_balanced_window_nll：各视频内部先平均窗口NLL，再视频等权；使用同视频数系数。

这些系数使固定模型下所有batch目标的平均等于完整epoch目标，避免尾批或变长视频隐式重加权。
实际在线loss记录按窗口数或视频数加权汇总，和每轮结束后的固定参数Train评估分开。
video_nll不是平均logits，也不是平均窗口NLL；padding不产生loss/梯度。

epoch0单独记录Train/selection指标及checkpoint，不参与epoch1–80选模。
每epoch保存：完整Train/selection指标，alpha/eta及w_A/w_F/w_P分位数/饱和比例，
delta_eta标准差，裁剪前alpha/eta参数梯度范数，在线训练目标，优化步数和时间。
eta接近0仅记录，不干预。

选模始终是原Test视频NLL。`best_val_nll.pt`与`last.pt`含真实logit/buffer/MLP、normalizer、
optimizer状态、来源身份和源码hash；新loader严格拒绝身份或代码不一致，旧loader拒绝V3权重。
仅提供fit/eval，不提供隐式resume。
`best/`、`last/`保存视频完整log-probabilities、索引、NLL及指标。
大型窗口轨迹仅由eval显式`--traces`导出，不进入普通训练日志。

## 只读复核命令

以下在项目根目录执行；示例输出已存在时必须换新目录，禁止覆盖。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-eventmamba/bin/python -m unittest test_fusion_routing test_fusion_routing_stage2 test_fusion_routing_v3 -v

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-eventmamba/bin/python evaluate_fusion_routing_v3.py audit \
  --config configs/routing_stage3/G1.json --output reports/routing_stage3/audit_repeat

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv-eventmamba/bin/python evaluate_fusion_routing_v3.py baseline-train \
  --config configs/routing_stage3/G1.json --suite reports/routing_stage2_full_v1 \
  --output reports/routing_stage3/train_baselines_repeat
```

基线统计已经完成，不需要在每次训练前重复运行。部分实测结果：

| 方法 | Train视频Top1 | Train视频NLL | 选模集视频Top1 | 选模集视频NLL |
|---|---:|---:|---:|---:|
| Fusion A | 100% | 0.000813 | 96.04697% | 0.359257 |
| Frame | 100% | 0.021406 | 97.10372% | 0.280180 |
| 固定Fusion+Frame | 100% | 0.010839 | 97.02544% | 0.217572 |

Fusion在样本内Train概率质量明显占优。新增eta自由度不会自动消除这种分布差异；
若后续模型回到Fusion，不应人工提高eta或根据Test结果改变数据来源。

## G1–G4真实训练命令（已完成，勿重复执行同一输出目录）

以下四次为受控2×2设计。每条命令独立，可选择执行；脚本不自动启动其他配置。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-eventmamba/bin/python train_fusion_routing_v3.py fit --config configs/routing_stage3/G1.json --output checkpoints/routing_stage3_G1_seed0 --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-eventmamba/bin/python train_fusion_routing_v3.py fit --config configs/routing_stage3/G2.json --output checkpoints/routing_stage3_G2_seed0 --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-eventmamba/bin/python train_fusion_routing_v3.py fit --config configs/routing_stage3/G3.json --output checkpoints/routing_stage3_G3_seed0 --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-eventmamba/bin/python train_fusion_routing_v3.py fit --config configs/routing_stage3/G4.json --output checkpoints/routing_stage3_G4_seed0 --device cpu
```

允许显式`--device cuda:0`进行单GPU训练，但本批未运行。source路径从配置指定的第一阶段sources索引解析，
快照目录不是权重或logits目录；迁移时需重新核验来源，不能跳过旧loader。

四个模型分别复评（训练完成后，命令不会训练）：

```bash
ROUTING_CACHE=/remote-home/datasets/event_project_cache/routing_epoch040_static_20261022_v3/test
for routing_id in G1 G2 G3 G4; do
  OMP_NUM_THREADS=2 .venv-eventmamba/bin/python evaluate_fusion_routing_v3.py eval \
    --checkpoint "checkpoints/routing_stage3_${routing_id}_seed0/best_val_nll.pt" \
    --cache "$ROUTING_CACHE" --output "reports/routing_stage3/${routing_id}_eval" --device cpu
done
```

需要逐窗口权重时单独加`--traces`，使用新目录。选中epoch不变。

## 条件性归因对照（尚未执行）

仅当video目标显示收益后再决定，区分视频重加权与监督聚合次序：

```bash
OMP_NUM_THREADS=2 .venv-eventmamba/bin/python train_fusion_routing_v3.py fit --config configs/routing_stage3/B1.json --output checkpoints/routing_stage3_B1_seed0 --device cpu
OMP_NUM_THREADS=2 .venv-eventmamba/bin/python train_fusion_routing_v3.py fit --config configs/routing_stage3/B2.json --output checkpoints/routing_stage3_B2_seed0 --device cpu
```

## 统一结果与配对（真实训练后执行）

```bash
.venv-eventmamba/bin/python evaluate_fusion_routing_v3.py compare \
  --results checkpoints/routing_stage3_G1_seed0/best checkpoints/routing_stage3_G2_seed0/best \
            checkpoints/routing_stage3_G3_seed0/best checkpoints/routing_stage3_G4_seed0/best \
  --suite reports/routing_stage2_full_v1 \
  --static-frame reports/routing_stage2_static_frame_comparison \
  --output reports/routing_stage3/main_comparison
```

主表包含Fusion、Frame、固定Fusion+Frame、旧Current、Static Frame强对照；
逐模型按视频/类别/窗口集合精确配对，报告纠错/新增错误及四种对错组的NLL贡献。
路径w_A=1−eta、w_F=eta×alpha、w_P=eta×(1−alpha)只表示显式混合，非因果模态贡献。
只改善NLL但Top1低于Frame必须称为取舍，不称全面超越。
原Test参与方法开发及选模，independent_test=false；没有独立测试结论。
若无历史双Gate获得稳定收益，才考虑对称预算下的Memory扩展；本批不自动展开。
