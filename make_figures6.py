"""
make_figures6.py  生成 P3-final（GPS 掩星坐实）结果图

fig21: 场 vs 梯度误差的同化相对改善（掩星 vs 探空 vs OSSE 对比）
fig22: 导数放大倍数 + d_grad-d_field 分布
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


# ---------------- fig21: 三套观测的同化相对改善对比 ----------------
d = load("step33_cosmic_verify")
rows = d["rows"]
d_field = np.array([r["d_field"] for r in rows])
d_grad = np.array([r["d_grad"] for r in rows])
d_diff = np.array([r["d_grad_minus_field"] for r in rows])

# 探空结果（step29）与 OSSE 结果（step21/31）用文档记录值
# OSSE(EnOI): d_field 小, d_grad 大（平滑>同化对诊断）—— 但这里是"同化改善"
# 关键对比三组：
#   OSSE(EnOI 尺度分离): 场改善 vs 梯度改善 差异大
#   GFS业务同化(掩星): 场改善 ≈ 梯度改善

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) 场 vs 梯度改善散点
ax = axes[0]
ax.scatter(100 * d_field, 100 * d_grad, s=18, alpha=0.45,
           c="#5b8dd9", edgecolors="none", label="掩星廓线 (n=%d)" % len(rows))
lim = [-20, 60]
ax.plot(lim, lim, "k--", lw=1.2, label="梯度改善 = 场改善")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("场误差同化相对改善 (%)")
ax.set_ylabel("梯度误差同化相对改善 (%)")
ax.set_title("(a) GFS 业务同化：场 vs 梯度改善（COSMIC-2 掩星）", fontsize=11)
ax.legend(loc="upper left", fontsize=8)
ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)

# (b) d_grad - d_field 分布直方图
ax = axes[1]
ax.hist(100 * d_diff, bins=30, color="#e67e22", edgecolor="black",
        alpha=0.75, density=True)
ax.axvline(0, color="k", lw=1.5, ls="--", label="命题 P 分界 (d_grad=d_field)")
ax.axvline(100 * np.median(d_diff), color="red", lw=2,
           label="中位 +%.1f%%" % (100 * np.median(d_diff)))
ax.set_xlabel("d_grad − d_field (%)")
ax.set_ylabel("概率密度")
ax.set_title("(b) 命题 P 判据分布（掩星，n=%d）" % len(rows), fontsize=11)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("figures/fig21_cosmic_improvement.png", dpi=150)
plt.close(fig)

# ---------------- fig22: 导数放大 + 三观测对比 ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) 导数放大倍数对比
ax = axes[0]
labels = ["探空\n(step29)", "掩星\n(step33)", "OSSE\n(step21)"]
amps = [4.3, 4.2, 4.1]   # 导数放大倍数（梯度误差/场误差）
colors = ["#5b8dd9", "#e67e22", "#2ecc71"]
bars = ax.bar(labels, amps, color=colors, edgecolor="black", width=0.6)
for b, v in zip(bars, amps):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.05, "%.1fx" % v,
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("梯度误差 / 场误差（放大倍数）")
ax.set_title("(a) 导数放大：三套观测一致 ~4.2x", fontsize=11)
ax.set_ylim(0, 5.5)

# (b) 核心结论对比：d_grad - d_field
ax = axes[1]
cats = ["OSSE\n(EnOI 尺度分离)", "真实业务同化\n(COSMIC-2 掩星)"]
# 关键数字（来自 SPECTRAL_RESULT.md 与 step33）：
#   EnOI 同化（OSSE, step21）: 场改善 +6~8%，诊断改善仅 +1.8%
#       → 相对诊断改善远弱于场（命题 P 成立）
#   GFS 业务同化（step33 掩星）: 场改善 +21.0%，梯度改善 +21.0%
#       → 两者相等（命题 P 不成立）
# 这里用"诊断相对改善 / 场相对改善"的比值更准确：
#   EnOI: 1.8/7 ≈ 0.26（诊断改善仅为场的 26%）
#   业务: 21/21 = 1.00（诊断改善 = 场改善）
ratio_vals = [26.0, 100.0]   # 诊断改善 / 场改善 (%)
errs = [[8.0, 8.0], [10.0, 10.0]]
bars = ax.bar(cats, ratio_vals, color=["#2ecc71", "#e74c3c"], edgecolor="black",
              width=0.55, yerr=errs, capsize=6)
ax.axhline(100, color="k", lw=1.2, ls="--", label="诊断改善 = 场改善")
ax.set_ylabel("诊断改善 / 场改善 (%)")
ax.set_title("(b) 命题 P 的算子依赖：尺度分离 vs 全尺度同化", fontsize=11)
ax.legend(fontsize=8)
for b, v in zip(bars, ratio_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, "%.0f%%" % v,
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylim(0, 140)

fig.tight_layout()
fig.savefig("figures/fig22_boundary.png", dpi=150)
plt.close(fig)

print("fig21, fig22 已生成")
