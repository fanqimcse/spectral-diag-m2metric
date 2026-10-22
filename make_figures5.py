# -*- coding: utf-8 -*-
"""make_figures5 —— P1/P2/P3 结果图。

fig17: M0 vs M2 判别力 + bootstrap 95%CI（step31）
fig18: 跨季节/跨年稳健性（step31 分组）
fig19: 谱质心转移 + 水平平滑谱作用（step30b）
fig20: 代表性误差分离（step32）
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load(name):
    with open("data/%s.json" % name, encoding="utf8") as f:
        return json.load(f)


# ---- fig17: 判别力 + CI ----
d = load("step31_m2_ci")
s = d["summary"]
fig, ax = plt.subplots(figsize=(7, 4.5))
labels = ["M0 (场误差)", "M2 (诊断误差)"]
disc = [100 * s["disc_M0"], 100 * s["disc_M2"]]
lo = [100 * s["disc_M0_ci"][0], 100 * s["disc_M2_ci"][0]]
hi = [100 * s["disc_M0_ci"][1], 100 * s["disc_M2_ci"][1]]
err_lo = [disc[i] - lo[i] for i in range(2)]
err_hi = [hi[i] - disc[i] for i in range(2)]
colors = ["#888888", "#c0392b"]
bars = ax.bar(labels, disc, yerr=[err_lo, err_hi], capsize=6,
              color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
for b, v in zip(bars, disc):
    ax.text(b.get_x() + b.get_width() / 2, v + 2, "%+.1f%%" % v,
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("判别力 = 平滑 − 同化 (%)")
ax.set_title("M0 vs M2 判别力（8 配对，跨 2 年×2 季；误差棒 = bootstrap 95% CI）",
             fontsize=11)
ax.set_ylim(-5, 50)
ax.text(0.02, 0.96, "Δ判别力(M2−M0) = +%.1f%%  [%.1f, %.1f]\nCI 不含 0 ⇒ 显著"
        % (100 * s["disc_diff"], 100 * s["disc_diff_ci"][0],
           100 * s["disc_diff_ci"][1]),
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="#fff3cd", ec="#f0ad4e"))
fig.tight_layout()
fig.savefig("figures/fig17_m2_discrimination.png", dpi=150)
plt.close(fig)

# ---- fig18: 跨季节稳健性 ----
rows = d["rows"]
groups = {}
for r in rows:
    g = (r["target"][:4], "冬" if r["target"][4:6] == "01" else "夏")
    groups.setdefault(g, []).append(r)
fig, ax = plt.subplots(figsize=(7, 4.5))
gkeys = sorted(groups)
x = np.arange(len(gkeys))
m0 = [100 * np.median([r["dM0_sbg"] - r["dM0_an"] for r in groups[g]])
      for g in gkeys]
m2 = [100 * np.median([r["dM2_sbg"] - r["dM2_an"] for r in groups[g]])
      for g in gkeys]
w = 0.35
ax.bar(x - w / 2, m0, w, label="M0 判别力", color="#888888", edgecolor="black")
ax.bar(x + w / 2, m2, w, label="M2 判别力", color="#c0392b", edgecolor="black")
ax.set_xticks(x)
ax.set_xticklabels(["%s\n%s" % (g[0], g[1]) for g in gkeys])
ax.set_ylabel("判别力 (%)")
ax.set_title("判别力的跨季节/跨年稳健性（每组 8 个 算例×变量）", fontsize=11)
ax.legend()
ax.axhline(0, color="black", lw=0.8)
fig.tight_layout()
fig.savefig("figures/fig18_season_robustness.png", dpi=150)
plt.close(fig)

# ---- fig19: 谱质心转移散点 ----
d = load("step30b_diag_spectrum")
rows = d["rows"]
kc_shift = [r["kc_shift"] for r in rows]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
# 左：谱质心转移
axes[0].hist(kc_shift, bins=15, color="#5b8dd9", edgecolor="black", alpha=0.8)
axes[0].axvline(np.median(kc_shift), color="red", ls="--",
                label="中位 %+.3f" % np.median(kc_shift))
axes[0].axvline(0, color="black", lw=0.8)
axes[0].set_xlabel("k质心转移 = k_c(诊断) − k_c(场)")
axes[0].set_ylabel("算例数")
axes[0].set_title("导数算子是否把谱权推向高波数？\n(16/24 为正，高通机制成立)",
                  fontsize=10)
axes[0].legend(fontsize=9)
# 右：水平平滑对场 vs 诊断
imp_f = [r["imp_field"] for r in rows]
imp_d = [r["imp_diag"] for r in rows]
axes[1].scatter(imp_f, imp_d, alpha=0.6, color="#e67e22", edgecolor="black")
axes[1].plot([0, 0.7], [0, 0.7], "k--", lw=0.8, label="1:1")
axes[1].axvline(np.median(imp_f), color="blue", ls=":", lw=1)
axes[1].axhline(np.median(imp_d), color="red", ls=":", lw=1)
axes[1].set_xlabel("水平平滑对场误差的改善")
axes[1].set_ylabel("水平平滑对诊断误差的改善")
axes[1].set_title("水平平滑：场改善(中位 %+.0f%%) vs 诊断改善(中位 %+.0f%%)"
                  % (100 * np.median(imp_f), 100 * np.median(imp_d)), fontsize=10)
axes[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig19_hv_coupling.png", dpi=150)
plt.close(fig)

# ---- fig20: 代表性误差 ----
d = load("step32_representativeness")
sgl = d["summary"]["single"]
cmp = d["summary"]["composite"]
fig, ax = plt.subplots(figsize=(6.5, 4.5))
labels = ["单点探空\n(ε_rep 大, n=195)", "空间合成\n(ε_rep 小, n=15)"]
# 关键指标：逐样本配对差值 d_grad - d_field 的中位（正确口径）
dmf = [100 * sgl["d_grad_minus_field"], 100 * cmp["d_grad_minus_field"]]
colors = ["#5b8dd9", "#e67e22"]
bars = ax.bar(labels, dmf, color=colors, edgecolor="black", alpha=0.85)
for b, v in zip(bars, dmf):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.5 if v >= 0 else -2),
            "%+.1f%%" % v, ha="center", va="bottom" if v >= 0 else "top",
            fontsize=11, fontweight="bold")
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("d_grad − d_field 中位 (%)")
ax.set_title("同化对梯度改善 vs 场改善的差异（配对差值中位）\n"
             "命题 P 预测 < 0；空间合成后反而 > 0 ⇒ 代表性误差非主因",
             fontsize=9)
fig.tight_layout()
fig.savefig("figures/fig20_representativeness.png", dpi=150)
plt.close(fig)

print("fig17-20 已生成")
