# -*- coding: utf-8 -*-
"""make_figures9 —— P5 充分条件定理的结果图。

三张图：
  fig26_s1_analytic.png       S1 无振铃充分性的解析核验（高斯 vs 锐利，6 尺度）
  fig27_synth_operators.png   [B] 受控合成：三类算子对 Ri_dry 误差的影响
  fig28_real_operators.png    [C] 真实 GFS：三类算子跨 4 算例对比
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.dpi": 110,
})

D = json.load(open("data/step36_sufficient_condition.json", encoding="utf8"))


def fig26():
    rows = D["analytic"]
    sigmas = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    gauss = [next(r["R"] for r in rows if r["op"] == "a_hgauss" and abs(r["param"] - s) < 1e-9)
             for s in sigmas]
    sharp = [next(r["R"] for r in rows if r["op"] == "b_hsharp"
                  and abs(r["param"] - np.sqrt(np.log(2)) / (2 * np.pi * s)) < 1e-9)
             for s in sigmas]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(sigmas, [100 * g for g in gauss], "o-", color="#c0392b", lw=2.5,
            ms=7, label="无振铃核（高斯平滑）")
    ax.plot(sigmas, [100 * s for s in sharp], "s--", color="#2c3e50", lw=2.5,
            ms=7, label="有振铃核（锐利谱截断）")
    for s, g, sh in zip(sigmas, gauss, sharp):
        ax.annotate("%+.0f%%" % (100 * g), (s, 100 * g),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="#c0392b")
        ax.annotate("%+.0f%%" % (100 * sh), (s, 100 * sh),
                    textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=8, color="#2c3e50")
    ax.fill_between(sigmas, [100 * g for g in gauss], [100 * s for s in sharp],
                    color="#e67e22", alpha=0.12,
                    label="无振铃优势（S1 充分性）")
    ax.set_xlabel("平滑尺度 σ（格点）", fontsize=11)
    ax.set_ylabel("净诊断误差下降 R（%）", fontsize=11)
    ax.set_title("S1 无振铃充分性：同截止尺度下无振铃核恒优于有振铃核",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.5, 8.5)
    fig.tight_layout()
    fig.savefig("figures/fig26_s1_analytic.png", bbox_inches="tight")
    plt.close(fig)
    print("fig26 done")


def fig27():
    synth = D["synth"]
    results = synth["results"]
    labels = ["无振铃\n+方向保持\n(水平高斯)",
              "有振铃\n+方向保持\n(水平锐利)",
              "无振铃\n+方向破坏\n(垂直高斯)"]
    ops = ["a_hgauss", "b_hsharp", "c_vgauss"]
    vals = [results[o]["d"] for o in ops]
    colors = ["#27ae60", "#c0392b", "#c0392b"]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.55)
    ax.axhline(0, color="black", lw=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + (2 if v >= 0 else -18),
                "%+.1f%%" % v, ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ri_dry 误差变化（%）", fontsize=11)
    ax.set_title("受控合成：充分条件两条件的必要性（破坏任一 → 有害）",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-3200, 100)
    fig.tight_layout()
    fig.savefig("figures/fig27_synth_operators.png", bbox_inches="tight")
    plt.close(fig)
    print("fig27 done")


def fig28():
    rows = D["real"]["rows"]
    targets = [r["target"] for r in rows]
    a = [r["a_hgauss"] for r in rows]
    b = [r["b_hsharp"] for r in rows]
    c = [r["c_vgauss"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5.2))
    x = np.arange(len(targets))
    w = 0.25
    ax.bar(x - w, a, w, color="#27ae60", label="无振铃+方向保持（水平高斯）", alpha=0.9)
    ax.bar(x, b, w, color="#c0392b", label="有振铃+方向保持（水平锐利）", alpha=0.9)
    ax.bar(x + w, c, w, color="#e67e22", label="无振铃+方向破坏（垂直高斯）", alpha=0.9)
    ax.axhline(0, color="black", lw=1.2)
    for i, (va, vb, vc) in enumerate(zip(a, b, c)):
        ax.text(i - w, va + 0.4, "%+.0f%%" % va, ha="center", fontsize=8, color="#1e8449")
        ax.text(i, vb - 350, "%+.0f%%" % vb, ha="center", fontsize=8, color="#922b21")
        ax.text(i + w, vc - 12, "%+.0f%%" % vc, ha="center", fontsize=8, color="#9a5b0f")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=9)
    ax.set_ylabel("Ri_dry 误差变化（%）", fontsize=11)
    ax.set_title("真实 GFS（4 算例）：充分条件在真实大气成立（4/4 一致）",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-3400, 80)
    fig.tight_layout()
    fig.savefig("figures/fig28_real_operators.png", bbox_inches="tight")
    plt.close(fig)
    print("fig28 done")


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    fig26()
    fig27()
    fig28()
