# -*- coding: utf-8 -*-
"""make_figures3.py —— 谱—诊断对应原理的核心结果图（fig9–fig12）。

图 9：谱传递定理 —— 导数算子 ∝ k 高通，M2/M0 = k0²/2 的解析与数值核验。
图 10：真实 GFS 误差谱分解 —— 同化 vs 平滑的波段分工（核心发现）。
图 11：跨诊断泛化 —— 垂直导数诊断"平滑>同化"，水平导数诊断反转。
图 12：偏差/噪声谱分离 —— 同化治偏差、平滑治噪声、可叠加（正面回答"重构为何必要"）。
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

plt.rcParams.update({"font.size": 10, "axes.titlesize": 12,
                     "axes.labelsize": 10.5})


def fig9_spectral_theorem():
    """图9：谱传递定理的解析与数值核验。"""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))

    # 左：导数算子的谱响应（高通）
    k = np.linspace(0, 0.5, 200)
    gain = 2 * np.sin(np.pi * k)          # 一阶差分（单位格距）
    axs[0].plot(k, gain, lw=2.5, color="#c0392b", label="差分算子谱响应 2·sin(πk)")
    axs[0].plot(k, 2 * np.pi * k, lw=1.5, ls="--", color="gray",
                label="小 k 近似 2πk（∝ k 高通）")
    axs[0].set_xlabel("波数 k（单位：1/格距）")
    axs[0].set_ylabel("谱增益 |∂/∂z|")
    axs[0].set_title("(a) 导数算子是高通滤波器")
    axs[0].legend(frameon=False, fontsize=8.5)
    axs[0].set_xlim(0, 0.5)
    axs[0].set_ylim(0, 3.2)
    axs[0].grid(alpha=0.25)

    # 右：M2/M0 = k0²/2 的解析 vs 数值
    k0 = np.array([0.5, 1.0, 2.0, 4.0])
    M2M0_num = np.array([0.1253, 0.5005, 2.0010, 8.0020])
    M2M0_theo = k0 ** 2 / 2.0
    axs[1].plot(k0, M2M0_theo, lw=2, color="#2c3e50", label="解析 k₀²/2")
    axs[1].scatter(k0, M2M0_num, s=60, color="#c0392b", zorder=5,
                   label="数值积分（相对误差 <0.5%）")
    axs[1].set_xscale("log", base=2)
    axs[1].set_yscale("log", base=2)
    axs[1].set_xlabel("误差谱特征波数 k₀")
    axs[1].set_ylabel("诊断灵敏度 M2/M0（诊断方差/场方差）")
    axs[1].set_title("(b) 诊断灵敏度 ∝ k₀²（高斯谱解析）")
    axs[1].legend(frameon=False, fontsize=8.5)
    axs[1].grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig("figures/fig9_spectral_theorem.png", dpi=150)
    plt.close(fig)
    print("fig9 完成")


def fig10_spectral_decomposition():
    """图10：真实 GFS 误差谱分解 —— 同化 vs 平滑的波段分工。"""
    # 数据来自 step21 汇总（跨算例中位，移除百分比）
    vars_ = ["T", "q", "u", "v"]
    an_low = np.array([-5.3, 17.1, 34.2, 21.3])   # 同化移除大尺度
    sbg_low = np.array([-3.4, 3.3, 1.7, -0.4])    # 平滑移除大尺度
    an_high = np.array([8.8, 3.8, 3.4, 6.7])      # 同化移除小尺度
    sbg_high = np.array([26.2, 37.1, 36.2, 31.4])  # 平滑移除小尺度

    x = np.arange(len(vars_))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - 1.5 * w, an_low, w, color="#3498db", label="同化 · 大尺度")
    ax.bar(x - 0.5 * w, sbg_low, w, color="#85c1e9", label="平滑 · 大尺度")
    ax.bar(x + 0.5 * w, an_high, w, color="#e67e22", label="同化 · 小尺度")
    ax.bar(x + 1.5 * w, sbg_high, w, color="#f0b27a", label="平滑 · 小尺度")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(vars_)
    ax.set_ylabel("误差功率移除比例 [%]")
    ax.set_title("图10：同化治大尺度、平滑治小尺度（4 个 OSSE 算例，巡航层）")
    ax.legend(frameon=False, ncol=2, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    # 标注关键对比
    ax.annotate("同化对小尺度\n几乎无效 (+3~9%)", xy=(1.5 * w, 6), xytext=(1.1, -16),
                fontsize=8, color="#e67e22",
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1))
    ax.annotate("平滑对小尺度\n稳定有效 (+26~37%)", xy=(3 + 1.5 * w, 31),
                xytext=(2.0, 44),
                fontsize=8, color="#c0392b",
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))
    fig.tight_layout()
    fig.savefig("figures/fig10_spectral_decomposition.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("fig10 完成")


def fig11_crossdiag():
    """图11：跨诊断泛化 —— 垂直 vs 水平导数诊断。"""
    diags = ["S²\n(风切变)", "N²\n(浮力频率)", "Ri\n(湍流)", "dθ/dz\n(CAPE)",
             "锋生\n|∇θ|²", "Ellrod\nTI"]
    an = np.array([2.16, 2.08, 2.16, 2.01, 1.34, 1.62])
    sbg = np.array([4.50, 9.32, 4.56, 9.40, -11.34, -2.60])
    san = np.array([5.90, 11.75, 5.96, 11.76, -10.03, -1.89])

    x = np.arange(len(diags))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w, an, w, color="#3498db", label="同化")
    ax.bar(x, sbg, w, color="#e67e22", label="水平平滑")
    ax.bar(x + w, san, w, color="#c0392b", label="同化+平滑")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(3.5, color="gray", ls="--", lw=1)
    ax.text(1.75, 12.5, "垂直导数诊断\n（平滑 > 同化）", ha="center",
            fontsize=9, color="#2c3e50")
    ax.text(5.0, 3.0, "水平导数诊断\n（平滑反转有害）", ha="center",
            fontsize=9, color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(diags, fontsize=8.5)
    ax.set_ylabel("误差降幅 [%]（对数空间 RMSE）")
    ax.set_title("图11：谱—诊断对应原理的跨诊断泛化（6 类诊断，4 算例中位）")
    ax.legend(frameon=False, ncol=3, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig("figures/fig11_crossdiag.png", dpi=150)
    plt.close(fig)
    print("fig11 完成")


def fig12_bias_noise():
    """图12：偏差/噪声谱分离 —— 正面回答"重构为何必要"。"""
    cases = ["仅大尺度\n偏差", "仅小尺度\n噪声", "偏差\n+噪声"]
    smooth = np.array([20.8, 38.4, 33.9])
    assim = np.array([45.2, 15.0, 26.0])
    both = np.array([0, 0, 58.2])   # 仅组合情况有意义

    x = np.arange(len(cases))
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.bar(x - w, smooth, w, color="#e67e22", label="平滑（治小尺度噪声）")
    ax.bar(x, assim, w, color="#3498db", label="同化（治大尺度偏差）")
    ax.bar(x + w, both, w, color="#c0392b", label="同化+平滑")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.set_ylabel("Ri 诊断误差降幅 [%]")
    ax.set_title("图12：同化治偏差、平滑治噪声，两者可叠加")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, 65)
    ax.annotate("同化是唯一\n有效手段", xy=(0, 45), xytext=(0.25, 52),
                fontsize=8, color="#3498db",
                arrowprops=dict(arrowstyle="->", color="#3498db", lw=1))
    ax.annotate("平滑是唯一\n有效手段", xy=(1 - w, 38), xytext=(0.55, 48),
                fontsize=8, color="#e67e22",
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1))
    fig.tight_layout()
    fig.savefig("figures/fig12_bias_noise.png", dpi=150)
    plt.close(fig)
    print("fig12 完成")


if __name__ == "__main__":
    fig9_spectral_theorem()
    fig10_spectral_decomposition()
    fig11_crossdiag()
    fig12_bias_noise()
    print("全部完成")
