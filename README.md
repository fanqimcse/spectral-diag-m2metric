# 风场—温度—湿度三维精细场实时重构 + 飞行路径气象风险评估

> **代码仓库**：https://github.com/fanqimcse/spectral-diag-m2metric
> **版本**：v1.0.0 ｜ **许可**：MIT

> **谱—诊断对应原理：场空间精度不是导数型诊断精度的代理量**
> （The spectral–diagnostic correspondence principle: field-space accuracy is
> not a proxy for derivative-diagnostic accuracy）

本仓库是一个真实预报误差 OSSE + 独立观测（IGRA2 探空 / COSMIC-2 GNSS 掩星）
验证框架，定量证伪了资料同化领域 50 年来的一个隐含假设，并给出其成立的充要条件。

---

## 一句话结论

> 资料同化系统的评价范式默认"把场同化得更准（更低 RMSE），下游物理诊断就会更准"。
> 本项目证明：**任何以空间导数为输入的下游诊断，其误差由背景误差的谱（spectrum）
> 决定，而非由误差的幅度决定。** 导数算子在高波数是高通滤波器，因此诊断误差
> ∝ 误差谱二阶矩 `M2 = ∫k²E(k)dk`，而场误差 ∝ 零阶矩 `M0 = ∫E(k)dk`。

这解释了否则无法解释的反常：EnOI 同化把场 RMSE 降 6–8%，但对湍流 Ri 诊断改善
仅 1.8%；而一次不引入任何新观测的水平平滑却改善 6.8%。

---

## 核心成果（可复现证据链）

| 层面 | 脚本 | 结论 |
|---|---|---|
| 解析定理 | `step22`, `step26` | M2/M0 = k₀²/2；M2 判别力 +38.3% [33.0, 40.1] |
| OSSE 谱分解 | `step21` | 同化治大尺度、平滑治小尺度（4/4） |
| 跨诊断泛化 | `step23`, `step28` | 线性导数型成立，非线性/积分型反转 |
| 偏差/噪声分离 | `step24` | 同化治偏差、平滑治噪声、可叠加 |
| 方向约束 | `step25` | 垂直平滑一律有害 |
| 尺度分离算子 | `step27` | 无振铃渐近核 > 锐利谱滤波 |
| 独立观测 | `step29` | 导数放大 4.3 倍（探空） |
| 格点化独立观测 | `step33` | 导数放大 4.2×（COSMIC-2 掩星） |
| M2 判别力 CI | `step31` | 判别力 +38.3%，跨 2 年 2 季稳健 |
| **算子充分条件** | `step36` | 无振铃(S1)+方向保持(S2) 是尺度分离安全的充分条件 |

核心定理（精确形式）：分歧成立 **当且仅当** 同时满足
(C1) 背景小尺度误差相对显著 + (C2) 同化增量谱带宽受限 + (C3) 小尺度是
"可消除噪声"而非"真实信号"。

---

## 工具包：`m2metric`

轻量、自包含（仅依赖 `numpy`）的 Python 包，暴露 RMSE 在导数型诊断上的盲区：

```python
from m2metric import compute_m2, discriminate

error = np.random.randn(128, 128)
r = compute_m2(error, dx=0.25, dy=0.25)
# -> {"M0": ..., "M2": ..., "M2_over_M0": ..., "k_cut": ...}

d = discriminate(bg_errors, an_errors, scale_errors, dx=0.25, dy=0.25)
# -> M0/M2 判别力 + 配对 bootstrap 95% CI + 显著性
```

详见 [`m2metric/README.md`](m2metric/README.md)。

---

## 仓库结构

```
avwx/            # 核心库：thermo / risk / enoi / gfs / igra / verify ...
m2metric/        # M2 指标工具包（独立可复用）
step00-36_*.py   # 端到端可复现实验脚本
figures/         # 28 张结果图
*.md             # 成果文档（REPORT / SPECTRAL_RESULT / P123 / P3 / P4 / P5）
```

> 说明：`data/`（13 GB 观测与再分析数据）、`work/`（中间产物）、`logs/`
> 未纳入版本控制，按脚本头部注释可重新下载。

---

## 复现

环境：Python 3.10+，依赖 `numpy scipy netCDF4 matplotlib`。

```bash
pip install numpy scipy netCDF4 matplotlib
python step22_spectral_theory.py   # 解析定理（秒级，无需数据）
python test_core.py                # m2metric 单元测试
```

需要真实数据（GFS / IGRA2 / COSMIC-2）的脚本按头部注释下载数据后运行。

---

## 许可

[MIT](LICENSE)

## 引用

见 [`CITATION.cff`](CITATION.cff)。
