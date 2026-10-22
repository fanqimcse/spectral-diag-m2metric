# -*- coding: utf-8 -*-
"""make_figures7.py —— P4 的图：RMSE vs M2 对照表 + 决策速查表。"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# ---------------- fig23: RMSE vs M2 判别力对照 ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# (a) 判别力对比
ax = axes[0]
categories = ["合成演示\n(n=12)", "真实 EnOI\n(GFS 8配对)", "真实业务\n(掩星180样本)"]
# [M0判别力, M2判别力]
m0 = [2.0, 7.5, 0.0]      # 业务场景两者一致，判别力差 0
m2 = [30.6, 38.3, 0.0]
x = np.arange(3)
w = 0.35
b1 = ax.bar(x - w/2, m0, w, label="RMSE (∝ M0) 判别力", color="#5b8dd9",
            edgecolor="black")
b2 = ax.bar(x + w/2, m2, w, label="M2 判别力", color="#e67e22", edgecolor="black")
for b, v in zip(b1, m0):
    ax.text(b.get_x()+b.get_width()/2, v+0.8, "%+.0f%%" % v, ha="center", fontsize=9)
for b, v in zip(b2, m2):
    ax.text(b.get_x()+b.get_width()/2, v+0.8, "%+.0f%%" % v, ha="center", fontsize=9,
            fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel("判别力 = 尺度分离改善 − 同化改善 (%)")
ax.set_title("(a) RMSE 系统性低估尺度分离对下游诊断的价值", fontsize=11)
ax.axhline(0, color="k", lw=0.8)
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(-5, 48)

# (b) "RMSE 会说谎"示意
ax = axes[1]
ax.axis("off")
lines = [
    ("真实 EnOI 场景（GFS 8 配对）", "title"),
    ("", "gap"),
    ("RMSE 判读：平滑比同化好 +7.5%", "rmse"),
    ("   → '聊胜于无，可砍'", "sub"),
    ("", "gap"),
    ("M2 判读：平滑比同化好 +38.3%", "m2"),
    ("   → '立竿见影，值得上线'", "sub"),
    ("", "gap"),
    ("RMSE 掩盖了 30.8 个百分点的", "concl"),
    ("真实下游产品收益", "concl"),
]
y = 0.95
for text, kind in lines:
    if kind == "title":
        ax.text(0.05, y, text, fontsize=12, fontweight="bold", va="top")
        y -= 0.09
    elif kind == "gap":
        y -= 0.04
    elif kind == "rmse":
        ax.text(0.05, y, text, fontsize=12, color="#5b8dd9", va="top",
                fontweight="bold")
        y -= 0.07
    elif kind == "m2":
        ax.text(0.05, y, text, fontsize=12, color="#e67e22", va="top",
                fontweight="bold")
        y -= 0.07
    elif kind == "sub":
        ax.text(0.12, y, text, fontsize=10, color="gray", va="top")
        y -= 0.06
    elif kind == "concl":
        ax.text(0.05, y, text, fontsize=12, color="#c0392b", va="top",
                fontweight="bold")
        y -= 0.07
ax.set_title("(b) RMSE 会说谎、M2 会说真话", fontsize=11)

fig.tight_layout()
fig.savefig("figures/fig23_rmse_lies.png", dpi=150)
plt.close(fig)

print("fig23 已生成")
