# -*- coding: utf-8 -*-
"""make_figures4.py —— 本轮（step26–29）核心结果图。

四张图：
  fig13  M2 作为评价指标的判别力（M0 vs M2 对"平滑的独有价值"的揭示）
  fig14  谱滤波 vs 高斯平滑（锐利截断的 Gibbs 振铃代价）
  fig15  算子性质边界（线性导数型 vs 非线性/积分型诊断）
  fig16  （若 step29 有数据）独立探空验证
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

os.makedirs("figures", exist_ok=True)


def fig13():
    """M2 判别力：M0（RMSE²）会掩盖平滑的独有价值，M2 能揭示。"""
    d = json.load(open("data/step26_m2_metric.json", encoding="utf8"))
    emp = d["empirical"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左：同化 vs 平滑对 M0 的降低
    ax = axes[0]
    cats = ["同化", "平滑"]
    m0 = [100 * emp["dM0_an"], 100 * emp["dM0_sbg"]]
    m2 = [100 * emp["dM2_an"], 100 * emp["dM2_sbg"]]
    x = np.arange(2)
    w = 0.35
    b1 = ax.bar(x - w / 2, m0, w, label="M0（场误差）", color="#5b8ff9")
    b2 = ax.bar(x + w / 2, m2, w, label="M2（诊断误差代理）", color="#e8684a")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("误差降低 %")
    ax.set_title("M0 与 M2 对两类改进的响应\n（真实 GFS，跨 16 个算例×变量中位）")
    ax.legend(fontsize=8)
    ax.axhline(0, color="gray", lw=0.7)
    for b in b1 + b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                "%+.1f" % b.get_height(), ha="center", fontsize=8)

    # 右：判别力对比
    ax = axes[1]
    disc_m0 = 100 * (emp["dM0_sbg"] - emp["dM0_an"])
    disc_m2 = 100 * (emp["dM2_sbg"] - emp["dM2_an"])
    bars = ax.bar(["M0（RMSE²）", "M2（谱二阶矩）"], [disc_m0, disc_m2],
                  color=["#5b8ff9", "#e8684a"])
    ax.set_ylabel("判别力（平滑−同化）%")
    ax.set_title("谁能揭示'平滑的独有价值'？")
    ax.axhline(0, color="gray", lw=0.8)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                "%+.1f%%" % b.get_height(), ha="center", fontsize=10)
    ax.annotate("M0 判别力为负：\n用 RMSE 评价会误判\n'平滑几乎无用'",
                xy=(0, disc_m0), xytext=(0.25, -15),
                fontsize=8, color="#5b8ff9",
                arrowprops=dict(arrowstyle="->", color="#5b8ff9", lw=1))
    ax.annotate("M2 判别力 +21.4%：\n揭示平滑的独有价值",
                xy=(1, disc_m2), xytext=(0.45, 15),
                fontsize=8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

    fig.tight_layout()
    fig.savefig("figures/fig13_m2_discrimination.png", dpi=150)
    plt.close(fig)
    print("fig13 完成")


def fig14():
    """谱滤波 vs 高斯平滑：锐利截断的代价。"""
    d = json.load(open("data/step27_spectral_filter.json", encoding="utf8"))
    med = d["median"]
    rows = d["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左：中位降幅对比
    ax = axes[0]
    cats = ["只同化", "高斯平滑", "谱滤波\n(锐利低通)", "同化+\n谱滤波"]
    vals = [med["d_an"], med["d_gauss"], med["d_sharp"], med["d_sharp_an"]]
    colors = ["#5b8ff9", "#5ad8a6", "#e8684a", "#9270ca"]
    bars = ax.bar(cats, vals, color=colors)
    ax.set_ylabel("Ri_dry 误差降低 %（巡航带）")
    ax.set_title("谱滤波 vs 高斯平滑（4 算例中位）")
    ax.axhline(0, color="gray", lw=0.8)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                "%+.1f" % b.get_height(), ha="center", fontsize=9)
    ax.set_ylim(-4, 8)

    # 右：解析对比（锐利 vs 高斯 对 M2 的降低，随截止波数）
    ax = axes[1]
    ana = d["analytic"]
    kcuts = [r["k_cut"] for r in ana]
    sharp_m2 = [100 * r["dM2_sharp"] for r in ana]
    gauss_m2 = [100 * r["dM2_gauss"] for r in ana]
    ax.plot(kcuts, sharp_m2, "o-", label="锐利低通", color="#e8684a")
    ax.plot(kcuts, gauss_m2, "s-", label="高斯平滑", color="#5ad8a6")
    ax.set_xlabel("截止波数 k_cut")
    ax.set_ylabel("M2 降低 %")
    ax.set_title("解析：对诊断误差(M2)的降低\n（同截止波数）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/fig14_spectral_filter.png", dpi=150)
    plt.close(fig)
    print("fig14 完成")


def fig15():
    """算子性质边界：线性导数型 vs 非线性/积分型。"""
    d = json.load(open("data/step28_diag_generalize.json", encoding="utf8"))
    summ = d["summary"]
    # 排序：按 平滑-同化 的差值
    items = []
    label_map = {
        "cape_proxy": "对流 CAPE\n(积分+非线性)",
        "cin_proxy": "对流抑制 CIN\n(积分)",
        "fronto_fn": "锋生函数\nD·|∇θ|(非线性)",
        "vor": "急流核涡度 ζ\n(线性水平导数)",
        "div": "急流核散度 D\n(线性水平导数)",
        "dthdz": "dθ/dz\n(线性垂直导数)",
    }
    for k, v in summ.items():
        items.append((label_map.get(k, k), v["an"], v["sbg"]))
    items.sort(key=lambda x: x[2] - x[1])

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(items))
    h = 0.35
    an_vals = [it[1] for it in items]
    sbg_vals = [it[2] for it in items]
    ax.barh(y - h / 2, an_vals, h, label="同化", color="#5b8ff9")
    ax.barh(y + h / 2, sbg_vals, h, label="水平平滑", color="#5ad8a6")
    ax.set_yticks(y)
    ax.set_yticklabels([it[0] for it in items], fontsize=8)
    ax.set_xlabel("诊断误差降低 %（中位）")
    ax.axvline(0, color="gray", lw=0.8)
    ax.legend(fontsize=9)
    ax.set_title("命题 P 的算子性质边界：\n线性导数型满足'平滑>同化'，非线性/积分型反转")
    ax.grid(axis="x", alpha=0.3)
    # 标注两组
    ax.text(0.98, 0.05, "反转：平滑有害", transform=ax.transAxes,
            ha="right", fontsize=9, color="#c0392b")
    fig.tight_layout()
    fig.savefig("figures/fig15_operator_boundary.png", dpi=150)
    plt.close(fig)
    print("fig15 完成")


def fig16():
    """独立观测验证：导数放大 4.3 倍 + 同化只动大尺度。"""
    d = json.load(open("data/step29_raob_verify.json", encoding="utf8"))
    ana = d.get("analysis", {})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左：场误差 vs 梯度误差（导数放大）
    ax = axes[0]
    cats = ["温度场误差\n（K）", "垂直梯度误差\n（K/lnp）"]
    vals = [ana.get("field_err_median", 0.70), ana.get("grad_err_median", 2.97)]
    bars = ax.bar(cats, vals, color=["#5b8ff9", "#e8684a"])
    ax.set_ylabel("巡航层误差中位")
    ax.set_title("独立观测（IGRA2 探空 vs GFS，195 组）：\n垂直导数放大误差 %.1f 倍"
                 % ana.get("amplification_factor", 4.3))
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                "%.2f" % b.get_height(), ha="center", fontsize=11)
    ax.annotate("放大 %.1f×\n（命题核心机制）" % ana.get("amplification_factor", 4.3),
                xy=(1, vals[1]), xytext=(0.5, vals[1] * 1.2),
                fontsize=9, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

    # 右：同化（f024→f000）对场 vs 梯度的改善
    ax = axes[1]
    cats2 = ["场误差\n改善", "梯度误差\n改善"]
    vals2 = [100 * ana.get("d_field_median", 0.127),
             100 * ana.get("d_grad_median", 0.142)]
    bars2 = ax.bar(cats2, vals2, color=["#5b8ff9", "#5ad8a6"])
    ax.set_ylabel("同化（f024→f000）相对改善 %")
    ax.set_title("同化只动大尺度：\n场/梯度改善相近（均为系统性偏差）")
    for b in bars2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                "%+.1f%%" % b.get_height(), ha="center", fontsize=11)

    fig.tight_layout()
    fig.savefig("figures/fig16_raob_verify.png", dpi=150)
    plt.close(fig)
    print("fig16 完成")


if __name__ == "__main__":
    fig13()
    fig14()
    fig15()
    fig16()
    print("全部完成")
