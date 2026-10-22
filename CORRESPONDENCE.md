# 致 GTG / GTG-N 维护方的技术核对函（草稿）

**主题：GTG 诊断中 Richardson 数所用的 Brunt–Väisälä 频率口径求证，以及晴空分支的一个可选改进**

**发往建议**：NCAR RAL（R. Sharman）· NOAA AWC / GSL · 帝国理工 rojak 维护者
**状态**：本函为**求证与建议**，不预设对方实现有误。事实上我们自己的一项主张已被本函的证据推翻（见 §3），一并说明。

---

## 1. 我们要问的问题

GTG（Sharman et al. 2006; Sharman & Pearson 2017, *J. Appl. Meteor. Climatol.* 56, 317–337）在若干诊断上做了如下修正操作：

> "…divide the computed raw diagnostic value by a local modified Richardson number
> Ri\* = max(Ri, 10⁻³)."

这个 `Ri` 的分子是 Brunt–Väisälä 频率的平方 N²。而 N² 至少有三种互不等价的"湿"口径：

| 口径 | 表达式 | 物理含义 |
|---|---|---|
| **干式** | N²_d = (g/θ)·∂θ/∂z | 忽略水汽 |
| **未饱和虚温式** | N²_v = (g/θ_v)·∂θ_v/∂z | 计入水汽对密度的稀释（密度效应），**不含**潜热 |
| **饱和式（Durran–Klemp 1982）** | N²_DK | 计入凝结潜热与饱和水汽的斜压贡献，仅在**饱和**时成立 |

**请问贵方实现中的 Ri 用的是哪一种？** 这直接决定了晴空（未饱和）条件下该修正项是否被正确适用。

## 2. 我们找到的第三方证据（指向"干式"）

帝国理工的开源实现 `rojak`（Apache-2.0，`rojak.turbulence.diagnostic`）明确以 Sharman2006 为基准，其源码给出：

```python
class BruntVaisalaFrequency:
    """
    .. math::  N^{2} = \frac{g}{\theta} \frac{\partial \theta}{\partial z}
    """
    def _compute(self):
        dθ_dz = altitude_derivative_on_pressure_level(self._potential_temperature,
                                                      self._geopotential)
        return (g / self.potential_temperature) * dθ_dz

class GradientRichardson:
    """
    In the GTG, it is defined as [Sharman2006]:  Ri_g = N² / S_v²
    """
```

即：输入是**普通位温 θ**，而非虚位温 θ_v 或相当位温 θ_e；公式中不含任何水汽项。
该模块共 27 个诊断类，**没有**名称含 moist / virtual / saturation / equivalent-potential-temperature 的条目。

**若贵方实现与之一致，则 GTG 的 Ri\* 是干式，不含水汽修正。**

## 3. 必须一并说明：我们撤回了一项自己的主张

我们在本项目的 v2.0 报告中曾写道：

> "文献证据确认 GTG 的 `1/Rimoist` 是 Durran–Klemp **饱和式**；晴空使用饱和式是物理误用，这是本项目 C 级增量最有力的一条。"

**上述"确认为饱和式"的表述缺乏实证支持，现予撤回。** 依据即 §2：现有可核验的实现与公式引用都指向干式。我们未能找到 GTG 源码或文档中使用饱和式的证据。若贵方确知存在使用饱和式（或 θ_e 基）的分支，请赐教，我们会在论文中准确引用。

需要强调：**撤回的是"GTG 正在误用"这一指控，不是物理结论本身。** 后者由两套独立观测支持（§4），它回答的是"**不应**在晴空这样做"，而非"贵方正在这样做"。

## 4. 我们得到的物理结论（供参考）

用两套**完全独立**的湍流观测——3706 份 PIREP（主观）与 10278 份 AMDAR 客观 EDR（MADIS 公开存档，机型无关）——对三种 Ri 做判别检验，结论如下。

### 4.1 虚温修正：在巡航层无技巧增益（两套观测一致）

| 观测 | 子集 | ΔAUC(虚温 − 干) |
|---|---|---|
| PIREP | 全样本 | −0.0024 [−0.0035, −0.0013] |
| PIREP | 晴空 + 高层 | −0.0001 ≈ 0 |
| EDR | 晴空 + 高层 | −0.0002 [−0.0003, +0.0000] |

原因可从解析恒等式看出（我们已数值核验到 2.4×10⁻¹⁴）：

```
N²_v − N²_d = 0.608·g·(dq/dz) / (1 + 0.608·q)
```

巡航层 q 极小 ⇒ 该差值只有干项的约 0.3%，不足以改变判别。

### 4.2 饱和式用于晴空：确有技巧损失，但幅度远小于我们先前的报告

| 观测 | 子集 | ΔAUC(饱和式 − 干式) |
|---|---|---|
| PIREP | **全样本** | −0.1241 [−0.1449, −0.1052] |
| PIREP | **晴空 + 高层** | −0.0241 |
| EDR | 晴空 + 高层（medEDR） | −0.0123 / −0.0243 / −0.0195（阈值 0.02/0.05/0.10） |
| EDR | 晴空 + 高层（maxEDR） | −0.0037 / −0.0200 / −0.0245 / −0.0183 |

**关键**：在真正对应 CAT 业务场景的"晴空 + 高层"子集上，两套独立观测给出高度一致的 **ΔAUC ≈ −0.02**。
而 PIREP 全样本的 −0.124 是**被低层与云区样本放大的**，EDR 未予证实。
我们据此把该结论的幅度从 −0.13 修正为 **≈ −0.02**。

## 5. 建议（若贵方确认当前为干式）

1. **保持晴空分支的干式（或虚温式）口径** —— 我们的观测证据不支持在晴空引入饱和式。
2. 若某些诊断确需在云区/对流区使用湿 N²，建议按 **RH 或云水含量做条件分支**，而不是全域使用同一湿式。我们在 RH ≥ 80% 的样本上并未观测到饱和式的优势（ΔAUC 在多数阈值为正但幅度接近 CI 边界）。
3. 建议把 **AMDAR/EDR 客观观测**纳入 GTG 权重标定，与 PIREP 并列。我们的 logistic 标定显示两者在某些子集上结论并不一致（§4.2），依赖单一主观数据源存在风险。

## 6. 最小可复现算例

随函提供 `avwx/thermo.py` 中三个函数与自检：

```python
from avwx import thermo as TH
TH.brunt_vaisala_sq(...)             # 干式
TH.brunt_vaisala_virtual_sq(...)     # 未饱和虚温式
TH.brunt_vaisala_durran_klemp_sq(...)  # Durran–Klemp 饱和式
```

恒等式自检（2.4×10⁻¹⁴ 精度，不含 S²）：

```
N²_v − N²_d = 0.608·g·(dq/dz)/(1 + 0.608·q)
```

以及一条判别性检查：**湿中性极限**。Durran–Klemp 饱和式在"饱和且湿中性"条件下应给出 N²_DK ≈ 0（实测残差 0.136）；干式与虚温式在同一廓线上则给出显著正值。这条检查可以在不看任何湍流观测的情况下，快速判定一个实现用的是不是饱和式 —— 建议贵方用它自查。

---

**联系人**：（本项目研究者）
**附**：`REPORT.md`、`step11b_edr_verification.py`、`avwx/thermo.py`、`avwx/verify.py`（全部可匿名获取、可复现）
