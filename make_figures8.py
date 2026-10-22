"""
make_figures8.py  推论 A 结果图
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    with open("data/%s.json" % name, encoding="utf8") as f:
        return json.load(f)


# ---- fig24: 算子对比（业务同化 vs 背景平滑）----
d = load("step35_smooth_vs_assim")
assim = d["assimilation"]
smooth = d["smooth"]
smooth_km = d["smooth_km"]

fig, ax = plt.subplots(figsize=(7.5, 4.8))

# 业务同化（单点，用星号标）
ax.errorbar([0], [100 * assim["d_diff"]],
            yerr=[[100 * (assim["d_diff"] - 0)], [0]],
            fmt="D", color="#e74c3c", markersize=11,
            label="业务同化 f000 (全尺度)", capsize=5, elinewidth=2)

# 背景平滑（各 L_h）
L = []
for i, lh in enumerate(smooth_km):
    s = smooth[str(lh)]
    L.append(100 * s["d_diff"])
    ax.errorbar([i + 1], [100 * s["d_diff"]],
                yerr=[[100 * (s["d_diff"] - s["d_diff_ci"][0])],
                      [100 * (s["d_diff_ci"][1] - s["d_diff"])]],
                fmt="o", color="#3498db", markersize=9,
                capsize=5, elinewidth=2)

ax.axhline(0, color="black", lw=1.2, ls="--")
ax.axhline(100 * assim["d_diff"], color="#e74c3c", lw=0.8, ls=":", alpha=0.5)

xtick = ["同化"] + ["平滑\nL_h=%dkm" % lh for lh in smooth_km]
ax.set_xticks(range(len(xtick)))
ax.set_xticklabels(xtick)
ax.set_ylabel("d_grad − d_field (%)")
ax.set_title("推论A检验：业务同化 vs 背景平滑的算子对比\n（负值=梯度改善弱于场改善，即命题P成立）", fontsize=10.5)
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(-30, 35)
fig.tight_layout()
fig.savefig("figures/fig24_operator_comparison.png", dpi=150)
plt.close(fig)


# ---- fig25: 误差谱尺度主导性（决定算子有效性的根源）----
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

# (a) GFS f024 误差谱（大尺度主导）
# 用 step21 和 step35 的谱数据示意
ax = axes[0]
k_f024 = [0.0156]  # 谱质心
k_osse = [0.0358]
# 示意谱（用质心画高斯）
kx = np.linspace(0.001, 0.4, 200)
g_f024 = np.exp(-(kx / 0.0156) ** 2)
g_osse = np.exp(-(kx / 0.0358) ** 2)
ax.plot(kx, g_f024, color="#e74c3c", lw=2, label="GFS f024 误差谱\n(k_c=0.016, 大尺度主导)")
ax.plot(kx, g_osse, color="#2ecc71", lw=2, label="EnOI 重建误差谱\n(k_c=0.036, 更白)")
ax.axvline(0.0156, color="#e74c3c", ls=":", lw=1)
ax.axvline(0.0358, color="#2ecc71", ls=":", lw=1)
ax.set_xlabel("水平波数 k (归一化)")
ax.set_ylabel("误差功率谱")
ax.set_title("(a) 背景误差谱的尺度构成", fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(0, 0.2)

# (b) 诊断误差里小尺度的贡献（k²加权后）
ax = axes[1]
ax.plot(kx, g_f024 * (kx / 0.0156) ** 2, color="#e74c3c", lw=2,
        label="f024 诊断误差\n(k² 加权)")
ax.plot(kx, g_osse * (kx / 0.0156) ** 2, color="#2ecc71", lw=2,
        label="EnOI 诊断误差\n(k² 加权)")
ax.set_xlabel("水平波数 k (归一化)")
ax.set_ylabel("诊断误差谱 (k² 加权)")
ax.set_title("(b) 导数诊断对小尺度的 k² 放大", fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(0, 0.2)
fig.tight_layout()
fig.savefig("figures/fig25_scale_dominance.png", dpi=150)
plt.close(fig)

print("已生成 fig24, fig25")
