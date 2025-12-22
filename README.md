# NEW-Res-spectrum-predict — Residual Network for Spectrum Reconstruction from Color-Card RGB

本项目实现一种基于残差卷积网络（Residual CNN）的光谱重建方法：  
输入为“多色块色卡（4×6 patches）提取到的 RGB 编码数据（.npy）”，输出为样本的可见光谱向量（默认 76 维：380–760 nm，5 nm 间隔）。

本仓库包含：
- **单头模型（Single-Head）**：直接回归 76 维光谱
- **双头模型（Dual-Head）**：低频基底（DCT 系数）+ 逐波长残差，融合得到最终光谱
- **训练策略**：逐波长加权 MSE（峰段增强）+ DC 基线约束 + TV 平滑约束 +（可选）动态重加权
- **推理脚本**：加载权重，对单样本预测并可视化（含 MSE/RMSE/MAE/PSNR/R²）

> 注意：本仓库假设你已得到色卡 patch 的 RGB 数据文件（`.npy`）；本仓库不包含从原始照片中检测色卡、切分 patch 的过程。

---

## 1. 输入输出与任务定义（Scope）

### 1.1 输入（RGB 编码）
每个样本对应一个 `rgb_XXXX.npy`，典型形状支持两类：

- **(4, 6, K, 3)**：每个 patch 随机采样 K 个像素点 RGB（例如 K=100）
  - 训练/推理会对 K 维做均值，得到 `(4,6,3)` 的 patch 均值 RGB
- **(4, 6, 3)**：每个 patch 的均值/统计 RGB（已聚合）

RGB 若为 0–255，会自动归一化到 0–1。

模型最终输入张量为：
- **(B, 3, 4, 6)**（RGB 作为通道，4×6 网格作为空间结构）

### 1.2 输出（光谱向量）
每个样本对应一个 `spectral_XXXX.npy`（ground truth）：
- 形状建议为 **(76,)**

默认波长轴：
- **380–760 nm**，步长 **5 nm**（共 76 点）

---

## 2. 仓库结构（Repository Structure）

（以本仓库代码为准，常见文件如下）

```text
.
├── train_single_head.py            # 单头训练（含动态重加权、早停、LR调度、权重/MAE可视化导出）
├── model_single_head.py            # 单头残差网络：RGB网格 -> 76维光谱
├── train_dual_head.py              # 双头训练（低频基底+残差；脚本内可切换损失）
├── model_dual_head.py              # 双头网络：DCT基底系数 + per-λ 残差（含 token attention）
├── SpectralReconstructionLoss.py   # 训练损失：WeightedMSE + DC + TV + 可选 base_coef L2
├── WeightedMSEloss.py              # 峰段权重构造 build_peak_weight + WeightedMSELoss
├── predict.py                      # 单头推理：加载权重、预测、绘图、保存Excel（可选）
└── dataset.py                      # 数据集读取（读取 rgb_*.npy 与 spectral_*.npy 的配对）
