# THU-CHL M：基础融合＋历史残差

M 在完整 GM 公式中删除整个第二项：

`H_t^l = H0_t^l + phi_mem^l(LN(R_t^l)) * history_valid`

基础 H0 本身仍是点云与图像融合。M 不创建额外 Frame reader、PF Gate、Point/Frame LayerNorm 或补充 MLP；保留每 stage 的 history LayerNorm、Memory reader/output、writer 和更新门 U。没有历史时在投影之后再次 mask，防止学习到的 bias 污染首窗口。当前预测完成之后，把 M 自己的最终 H 写入 Memory，不使用删除的补充路径产生候选。

这是整个 PF 补充模块的结构消融，不是只去掉一个 Gate，也不等于 GM-FeatureOnly。它与 B/G/GM 形成模块级对照；若讨论动态门控本身，仍使用 G-FixedGate。

## 实现与运行

- `gcstr_memory_only_fusion.py`：独立 M 模型，复用原 Memory 读写及基础融合。
- `thu_chl_memory_only_runtime.py`：严格加载独立预训练分支、M 配置/源码身份、N/A 门控统计。
- `train_thu_chl_memory_only.py`：独立训练入口；沿用原数据、TBPTT、优化器、Top-1 选模和 epoch 边界恢复。
- `configs/examples/thu_chl_memory_only/M.json`：公开路径模板；替换路径后另存为本地配置。

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-eventmamba/bin/python train_thu_chl_memory_only.py \
  --config configs/thu_chl_memory_only/M.json \
  --output experiments/thu_chl_m_smoke --smoke
```

上面仅为手动验收命令，不会自动执行。正式训练用独立输出目录且去掉 `--smoke`。`--resume` 严格验证配置、源码、数据和初始化来源。全局 video batch48、TBPTT4、50轮、seed0、基础融合LR5e-5、Point LR1e-5、Memory LR1e-4；视频Top-1选模，同分按NLL；另存NLL最优。单卡直接处理全局batch，不含梯度累积。

M 与 GM 的基础模型、Memory 参数使用相同初始化；history LayerNorm 数值也一致，但为明确结构改用了独立 `history_norm.*` 键。M 有独立 architecture 标识，不可把 GM checkpoint 当作 M 训练起点。需要从独立预训练分支重新训练。

## 验收范围

`test_gcstr_memory_only.py` 覆盖：无 PF 参数、共有初始化和调用方 RNG 不变；与零第二项 GM 参考路径逐窗一致；Memory 更新使用修改后的 H；首窗和 padding 掩码；写入不影响当前预测；多步训练后 Memory/归一化/基础模型梯度；冻结图像及 Point BN；正式 run_pass 的 TBPTT4、singleton、完整覆盖和两轮断点恢复等价；配置协议。

已完成合成 CPU 测试以及真实 THU 预训练分支的严格 CPU 加载/身份审计。尚未执行真实 CUDA smoke、显存容量验证或正式训练；这些是启动前的剩余验收工作。原 B/G/GM 及当前 G 训练涉及的源码文件保持不变。
