# Top-K Checkpoints 保存功能说明

## 功能概述

训练期间自动保存 **Top-3** 验证 mIoU 最高的 checkpoint，解决"val 最优 ≠ 推理最优"的问题。

## 修改内容

### 1. 新增全局变量
```python
top_k_checkpoints = []  # 存储 (epoch, mIoU) 元组列表
```

### 2. 保存逻辑
每次验证后：
- 自动更新 Top-3 列表（按 mIoU 降序，mIoU 相同时按 epoch 降序）
- 如果当前 epoch 进入 Top-3，额外保存为 `model_best_rank{1/2/3}.pth.tar`
- 始终保持 `model_last.pth.tar`（最新）和 `model_best.pth.tar`（历史最优）

### 3. 保存的文件
训练结束后，`{save_path}/model/` 目录下会有：
- `model_last.pth.tar` - 最后一个 epoch
- `model_best.pth.tar` - 历史 val mIoU 最高的 epoch
- `model_best_rank1.pth.tar` - Top-1 的 checkpoint（与 model_best 相同）
- `model_best_rank2.pth.tar` - Top-2 的 checkpoint
- `model_best_rank3.pth.tar` - Top-3 的 checkpoint

### 4. 训练日志
每次评估后会打印：
```
Top-3 checkpoints: [(100, '0.4587'), (98, '0.4562'), (95, '0.4521')]
Saved top-1 checkpoint: epoch 100, mIoU 0.4587
```

训练结束时会汇总：
```
==>Training done!
Best Iou: 0.459
Top-3 checkpoints saved:
  Rank 1: Epoch 100 - mIoU 0.4587
  Rank 2: Epoch 98 - mIoU 0.4562
  Rank 3: Epoch 95 - mIoU 0.4521
```

## 使用建议

### 训练期间
- 正常训练即可，无需额外配置
- 观察日志中的 Top-3 列表，判断模型是否已收敛

### 训练结束后（推荐流程）
1. **快速对比**：如果 Top-3 的 mIoU 非常接近（差距 < 0.002），可以直接用 `model_best.pth.tar`

2. **严格选择**（推荐）：
   ```bash
   # 用最终推理配置（ensemble、test_repeats=5、prompt_eng=True）
   # 分别评估 Top-3 checkpoints
   
   python run/evaluate.py --config config/scannet/mink_pe.yaml \
       TEST.model_path out/scannet_mink_pe3/model/model_best_rank1.pth.tar
   
   python run/evaluate.py --config config/scannet/mink_pe.yaml \
       TEST.model_path out/scannet_mink_pe3/model/model_best_rank2.pth.tar
   
   python run/evaluate.py --config config/scannet/mink_pe.yaml \
       TEST.model_path out/scannet_mink_pe3/model/model_best_rank3.pth.tar
   ```

3. **选择真正最优**：对比推理结果，选择 mIoU 最高的作为最终发布权重

## 技术细节

### 排序规则
```python
top_k_checkpoints.sort(key=lambda x: (-x[1], -x[0]))
```
- 第一优先级：mIoU 降序（越高越好）
- 第二优先级：epoch 降序（mIoU 相同时，偏好更晚的 epoch，因为可能更稳定）

### 断点续训
- Top-K 列表会保存在 checkpoint 中
- 从 checkpoint 恢复训练时，Top-K 列表也会恢复
- 续训时会继续更新 Top-K 列表

### 内存和存储
- 每个 checkpoint 约 100-500 MB（取决于模型大小）
- Top-3 保存会额外占用 ~300-1500 MB 磁盘空间
- 内存中只维护 (epoch, mIoU) 的小列表，几乎无额外开销

## 示例场景

### 场景1：训练不稳定
```
Epoch 95: mIoU 0.4521  → Rank 1
Epoch 96: mIoU 0.4489  → 跌出 Top-3
Epoch 97: mIoU 0.4495  → 跌出 Top-3
Epoch 98: mIoU 0.4562  → Rank 2
Epoch 99: mIoU 0.4501  → Rank 3
Epoch 100: mIoU 0.4587 → Rank 1（新的最优）
```
结果：保存了 epoch 95/98/100，可以事后对比选择

### 场景2：过拟合
```
Epoch 80: mIoU 0.4821 → Rank 1
Epoch 85: mIoU 0.4815 → Rank 2
Epoch 90: mIoU 0.4798 → Rank 3
Epoch 95: mIoU 0.4765 → 跌出 Top-3
Epoch 100: mIoU 0.4721 → 跌出 Top-3（验证性能下降）
```
结果：保存了 epoch 80/85/90，避免选到过拟合的权重

### 场景3：val 与推理不完全一致
```
Val mIoU (训练验证):
  Epoch 98: 0.4562 → Rank 2
  Epoch 100: 0.4587 → Rank 1

Inference mIoU (最终推理):
  model_best_rank2.pth.tar (epoch 98): 0.5142
  model_best_rank1.pth.tar (epoch 100): 0.5099  ← 排序颠倒！
```
结果：虽然 val 上 epoch 100 更优，但推理时 epoch 98 更好。通过 Top-K 保存，可以对比后选择 epoch 98 作为最终权重。

## 向后兼容

- 仍然保存 `model_best.pth.tar`，与原有逻辑兼容
- 不影响现有的推理脚本和评估流程
- 可以选择忽略 Top-K 功能，只用 `model_best.pth.tar`

## 注意事项

1. **存储空间**：确保有足够磁盘空间（预留 1-2 GB）
2. **K 值调整**：如需修改 Top-K 数量，在 `distill.py` 第 308 行修改 `K = 3`
3. **评估一致性**：训练期间的 validation 现已与推理对齐（不归一化、支持 prompt_eng），但推理常用 ensemble 和 test_repeats>1，仍可能产生小偏差


