# m2metric

**M2（误差谱二阶矩）——面向下游产品（导数型诊断）的同化验证指标。**

一个轻量、自包含（仅依赖 `numpy`）的 Python 工具包，用于暴露 RMSE 在
导数型诊断上的盲区。

---

## 为什么需要 M2？

数据同化系统 50 年来的默认验证指标是 RMSE。但 RMSE 度量的是误差谱的
**零阶矩** `M0 = ∫E(k)dk`（场误差总能量），而下游的导数型诊断——
层结稳定性 N²、风切变 S²、理查森数 Ri、涡度、CAPE——的误差由误差谱的
**二阶矩** `M2 = ∫k²E(k)dk` 主导（导数算子在波数空间是高通，权重 k²）。

于是出现一个系统性的"指标失明"：

| 同化算子 | 动哪个尺度 | RMSE (M0) | M2 | 下游诊断 |
|---|---|---|---|---|
| EnOI（光滑增量） | 仅大尺度 | **显著下降** | 几乎不动 | **无改善** |
| 高斯平滑（尺度分离） | 仅小尺度 | 几乎不动 | **显著下降** | **有改善** |

- 当同化只修大尺度时，**RMSE 会说谎**：它报告了改进，但下游诊断毫无改善；
- 当尺度分离消除小尺度噪声时，**RMSE 又会掩盖**这次真正有价值的改进。

M2 正是为暴露这种盲区而设计的。

## 安装

```bash
pip install -e .        # 开发模式，从本目录
# 或
pip install m2metric   # 发布后
```

## 快速开始

```python
import numpy as np
from m2metric import compute_m2, discriminate

# 1. 单个误差场
error = np.random.randn(128, 128)   # 分析场 − 真值
r = compute_m2(error, dx=0.25, dy=0.25)
print(r["M0"], r["M2"], r["M2_over_M0"])

# 2. 对比"同化 vs 尺度分离"的判别力（核心场景）
bg_errors   = [...]   # 背景误差场列表（多算例）
an_errors   = [...]   # 同化后误差场
scale_errors = [...]  # 平滑后误差场
d = discriminate(bg_errors, an_errors, scale_errors, dx=0.25, dy=0.25)
print("M0 判别力:", d["disc_M0"], "  M2 判别力:", d["disc_M2"])
print("Δ判别力显著:", d["significant"], "  95%CI:", d["ci_diff"])
```

## API

### `compute_m2(error_field, dx=1.0, dy=1.0, mask=None, auto_truncate=True)`
对单个误差场计算 M0、M2。返回 dict，含 `M0`、`M2`、`M2_over_M0`、`k_cut`。

### `discriminate(bg_errors, an_errors, scale_errors, dx=1.0, dy=1.0, masks=None)`
对比同化与尺度分离的判别力，返回 M0/M2 判别力 + 配对 bootstrap 95% CI
+ 显著性。

### `spectrum2d(field, dx=1.0, dy=1.0, mask=None)`
二维水平功率谱（径向平均，含去趋势、汉宁窗、NaN/掩码处理）。

### `auto_k_cut(k, power, target_bias=0.05)`
自动选择截断波数（M2 的偏倚—方差权衡正则化）。

### `moments(k, power, k_cut=None)` / `M0(...)` / `M2(...)`
谱矩计算。

### `bootstrap_ci(vals, n=10000, seed=11)`
中位数的 bootstrap 置信区间。

## 口径（与论文一致）

- 波数用**归一化频率**（`np.fft.fftfreq(n)`，不加 `*n`）；
- `M0 = ∫E(k)dk`，`M2 = ∫k²E(k)dk`，`dk = np.gradient(k)`；
- 判别力 = (尺度分离的相对降低) − (同化的相对降低)，**逐样本先算差值再取中位**；
- 二维谱用汉宁窗，NaN/掩码用有限值均值填充。

## 决策速查

| 下游产品 | 指标 |
|---|---|
| 标量场（T、q、位势高度） | RMSE（∝ M0） |
| 垂直导数诊断（N²、S²、Ri、∂T/∂z） | **M2** |
| 水平导数诊断（涡度、散度、锋生） | M2（且对水平平滑敏感） |
| 非线性/积分型（CAPE、CIN） | M2（边界反转，见论文 step28） |

## 参考

- `step26_m2_metric.py` — M2 的统计性质（偏倚/方差/截断权衡）
- `step31_m2_ci.py` — M2 判别力的 bootstrap CI
- `step21_spectral.py` — 真实 GFS 误差谱分解

## 许可

MIT
