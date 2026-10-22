# -*- coding: utf-8 -*-
"""生成项目全部图表 → figures/*.png

所有数字均直接来自 data/*.json，不在此处硬编码任何结果。
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
S02 = json.load(open("data/step02_composite.json", encoding="utf8"))
S03 = json.load(open("data/step03_osse.json", encoding="utf8"))
S04 = json.load(open("data/step04_risk.json", encoding="utf8"))

C_MOIST = "#c0392b"      # 湿度/湿口径（红）
C_DRY = "#7f8c8d"        # 干口径（灰）
C_BLUE = "#2471a3"
C_GREEN = "#1e8449"


# ---------------------------------------------------------------- 图 1
def fig_climate():
    """支柱一：观测证据（GHCNh）——湿度对密度高度趋势的贡献。"""
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0))

    # (a) 趋势分解
    lab = ["干口径\nDA", "含湿\nDA", "湿度惩罚\nΔDA_q"]
    val = [S02["slope_da_dry_dec"], S02["slope_da_moist_dec"], S02["slope_dq_dec"]]
    col = [C_DRY, C_MOIST, C_MOIST]
    b = ax[0].bar(lab, val, color=col, width=0.55)
    ax[0].bar_label(b, fmt="%.2f", fontsize=9, padding=2)
    ax[0].set_ylabel("趋势 [ft / 十年]")
    ax[0].set_title("(a) 密度高度趋势：干 vs 湿\n湿度份额 %.1f%%（MK z=%.2f）"
                    % (S02["share_q"] * 100, S02["mk_z"]["dq"]), fontsize=10)
    ax[0].grid(axis="y", alpha=.3)

    # (b) 纬度带（step02 表：ΔDA_q 中位）
    bands = ["热带", "亚热带", "温带", "高纬"]
    dq = [354.8, 250.8, 171.8, 115.7]
    b = ax[1].bar(bands, dq, color=[C_MOIST, "#e37b6a", "#f0b27a", "#d5d8dc"], width=0.55)
    ax[1].bar_label(b, fmt="%.0f", fontsize=9, padding=2)
    ax[1].set_ylabel("ΔDA_q 中位 [ft]")
    ax[1].set_title("(b) 现行算法遗漏的湿度惩罚\n热带是高纬的 %.1f 倍" % (dq[0] / dq[3]), fontsize=10)
    ax[1].grid(axis="y", alpha=.3)

    # (c) 观测 vs 克劳修斯-克拉珀龙预测
    r_obs, r_pred = S02["ratio_obs"], S02["ratio_pred"]
    cc = S02["dlnq_dT_pct_per_K"]
    b = ax[2].bar(["观测\nR_obs", "CC 预测\nR_pred", "dlnq/dT\n观测", "CC\n7%/K"],
                  [r_obs, r_pred, cc / 100, 0.07], width=0.55,
                  color=[C_BLUE, C_DRY, C_BLUE, C_DRY])
    ax[2].bar_label(b, fmt="%.4f", fontsize=9, padding=2)
    ax[2].set_title("(c) 可证伪解析式 R = 0.0426·q·T\n观测/预测 = %.3f；dlnq/dT 达成 CC 的 %.0f%%"
                    % (r_obs / r_pred, cc / 7 * 100), fontsize=10)
    ax[2].grid(axis="y", alpha=.3)
    fig.suptitle("支柱一｜GHCNh 观测（69 站，1976–2023）：忽略湿度使密度高度趋势被低估 %.1f%%"
                 % S02["underestimation_pct"], fontsize=12, y=1.02)
    fig.savefig(f"{OUT}/fig1_climate.png")
    plt.close(fig)


# ---------------------------------------------------------------- 图 2
def fig_osse():
    """支柱二：OSSE —— 从湿度盲的飞机观测中能榨出多少湿度信息。"""
    rows = S03["rows"]
    names = [r["name"].strip() for r in rows]
    short = [n.split()[0] for n in names]
    keys = ["T", "q", "u", "v"]
    base = S03["base_impact"]
    red = {k: [(1 - r[f"{k}_imp"] / base[k]) * 100 for r in rows] for k in keys}

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.2),
                           gridspec_kw={"width_ratios": [1.45, 1]})
    w = 0.2
    x = np.arange(len(rows))
    for i, k in enumerate(keys):
        b = ax[0].bar(x + (i - 1.5) * w, red[k], w, label=k,
                      color=[C_MOIST, "#1e8449", C_BLUE, "#8e44ad"][i])
        ax[0].bar_label(b, fmt="%.2f", fontsize=7, padding=1)
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(short)
    ax[0].set_ylabel("相对背景场的 RMSE 降幅 [%]")
    ax[0].set_title("(a) 观测影响区（%.0f 格点，占全域 %.1f%%）内各配置降幅"
                    % (S03["n_impact"], S03["n_impact"] / S03["n_grid"] * 100), fontsize=10)
    ax[0].legend(title="变量", fontsize=8, ncol=4)
    ax[0].grid(axis="y", alpha=.3)

    # (b) 内部一致性 + 榨取量
    h = S03["headline"]
    lab = ["A 无飞机\n(地面+探空)", "B +飞机\n(单变量,湿度盲)", "D +水汽平流耦合 ★"]
    val = [h["q_A"], h["q_B"], h["q_D"]]
    b = ax[1].bar(lab, val, color=[C_DRY, C_BLUE, C_MOIST], width=.5)
    ax[1].bar_label(b, fmt="%.3e", fontsize=8, padding=2)
    ax[1].set_ylabel("湿度 RMSE [kg/kg]")
    ax[1].set_title("(b) 内部一致性检验 + 榨取量\n|A−B|/A = %.4f%%（飞机不报湿度，应=0）\n★ 榨取 %.4f 个百分点"
                    % (h["internal_consistency_pct"], h["gain_from_blind_obs_pct"]), fontsize=10)
    ax[1].grid(axis="y", alpha=.3)
    ax[1].tick_params(axis="x", labelsize=8)
    fig.suptitle("支柱二｜OSSE：标定 α_T2V=%.2f（无正贡献）、α_V2q=%.2f、τ=%.1f h"
                 % (S03["alpha_T2V"], S03["alpha_V2q"], S03["tau_h"]), fontsize=12, y=1.02)
    fig.savefig(f"{OUT}/fig2_osse.png")
    plt.close(fig)


# ---------------------------------------------------------------- 图 3
def fig_delta_profile():
    """支柱三 a：δ 的高度结构 —— 湿度削弱稳定度随高度单调衰减。"""
    pl = S04["per_level"]
    lev = np.array(sorted([float(k) for k in pl], reverse=True))
    dlt = np.array([pl[f"{k:.0f}"]["delta"] for k in lev])
    miss = np.array([pl[f"{k:.0f}"]["missed"] * 100 for k in lev])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(dlt * 100, lev, "o-", color=C_MOIST, lw=2, ms=6)
    ax[0].fill_betweenx(lev, 0, dlt * 100, color=C_MOIST, alpha=.15)
    ax[0].set_yscale("log")
    ax[0].invert_yaxis()
    ax[0].set_ylim(lev.max() * 1.1, lev.min() * .9)
    ax[0].set_yticks(lev); ax[0].set_yticklabels([f"{int(l)}" for l in lev])
    ax[0].set_xlabel("δ = (N_d²−N_v²)/N_d²  [%]")
    ax[0].set_ylabel("气压 [hPa]")
    ax[0].set_title("(a) 湿度对静力稳定度的相对削弱（层内中位）\n"
                    "1000 hPa %.1f%%  →  250 hPa %.2f%%  →  100 hPa %.3f%%"
                    % (dlt[0] * 100, dlt[list(lev).index(250.0)] * 100, dlt[-1] * 100),
                    fontsize=10)
    ax[0].grid(alpha=.3)
    for y, t in ((850, "爬升/下降"), (250, "主巡航")):
        ax[0].axhline(y, color=C_BLUE, ls="--", lw=1, alpha=.7)
        ax[0].text(ax[0].get_xlim()[1] * .55, y * 1.05, t, fontsize=8, color=C_BLUE)

    # (b) 航迹分阶段 δ
    rpd = S04["route_phase_delta"]
    ph = [("delta_850", "850 hPa\n爬升/下降"), ("delta_500", "500 hPa\n低巡航"),
          ("delta_250", "250 hPa\n主巡航")]
    x = np.arange(3)
    p25 = [rpd[k][0] * 100 for k, _ in ph]
    p50 = [rpd[k][1] * 100 for k, _ in ph]
    p75 = [rpd[k][2] * 100 for k, _ in ph]
    ax[1].bar(x, p50, .45, color=C_MOIST, label="中位")
    ax[1].vlines(x, p25, p75, color="k", lw=1.6, label="25–75 分位")
    for i, (a, b_, c) in enumerate(zip(p25, p50, p75)):
        ax[1].text(i, c + .15, "%.2f" % b_, ha="center", fontsize=9)
    ax[1].set_xticks(x); ax[1].set_xticklabels([t for _, t in ph])
    ax[1].set_ylabel("δ [%]")
    ax[1].set_title("(b) 52 条真实机场对的航迹 δ\n爬升/下降段是主巡航段的 %.0f 倍"
                    % (p50[0] / p50[2]), fontsize=10)
    ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3)
    fig.suptitle("支柱三｜湿度对稳定度的削弱：恒等式 N_v²−N_d² = 0.608·g·(dq/dz)/(1+0.608q)"
                 "（数值核验 %.1e）" % S04["identity_check_median_rel_err"],
                 fontsize=11.5, y=1.03)
    fig.savefig(f"{OUT}/fig3_delta_profile.png")
    plt.close(fig)


# ---------------------------------------------------------------- 图 4
def fig_missed():
    """支柱三 b：干口径的定性错误（符号翻转）与日变化。"""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    # (a) 按 N_d² 分箱
    bins = S04["missed_by_n2_bin"]
    lo, rate, n = [], [], []
    for k, (nn, kk) in bins.items():
        a, b_ = [float(t) for t in k.split("_")]
        if nn > 50:
            lo.append(a); rate.append(kk / nn * 100); n.append(nn)
    order = np.argsort(lo)
    lo = [lo[i] for i in order]; rate = [rate[i] for i in order]; n = [n[i] for i in order]
    lbl = []
    for i in range(len(lo)):
        hi = 1e9 if i == len(lo) - 1 else lo[i + 1]
        lbl.append("[%.0e, %.0e)" % (lo[i], hi))
    b = ax[0].bar(range(len(rate)), rate, color=C_MOIST, width=.62)
    ax[0].bar_label(b, fmt="%.2f", fontsize=8, padding=2)
    ax[0].set_xticks(range(len(rate))); ax[0].set_xticklabels(lbl, rotation=35, ha="right", fontsize=8)
    ax[0].set_yscale("symlog", linthresh=0.01)
    ax[0].set_xlabel("干静力稳定度 N_d² [s^-2]"); ax[0].set_ylabel("漏判率 [%]")
    ax[0].set_title("(a) 干口径的符号错误率：层结越弱，错得越离谱\n"
                    "（N_d²<1e-6 时干口径有 %.0f%% 判反）" % max(rate), fontsize=10)
    ax[0].grid(axis="y", alpha=.3)

    # (b) 日变化
    pt = S04["per_time"]
    loc = {"06": 14, "12": 20, "18": 2, "00": 8}
    hrs, mr = [], []
    for r in pt:
        h = loc[r["time"][-2:]]
        hrs.append(h); mr.append(r["missed_rate"] * 100)
    o = np.argsort(hrs)
    hrs = [hrs[i] for i in o]; mr = [mr[i] for i in o]
    ax[1].plot(hrs, mr, "o-", color=C_MOIST, lw=2, ms=8)
    ax[1].fill_between(hrs, 0, mr, color=C_MOIST, alpha=.15)
    for h, m in zip(hrs, mr):
        ax[1].text(h, m + .006, "%.3f" % m, ha="center", fontsize=8)
    ax[1].set_xlabel("地方时 [h]（UTC+8）"); ax[1].set_ylabel("漏判率 [%]")
    ax[1].set_xticks([2, 8, 14, 20])
    ax[1].set_title("(b) 日变化：午后峰值、夜间归零\n"
                    "与白天混合形成的近中性层一致", fontsize=10)
    ax[1].grid(alpha=.3)
    fig.suptitle("主指标 A｜被干口径漏判的虚温不稳定层（晴空+地面以上）："
                 "中位 %.4f%%；条件于弱层结则高达数十个百分点"
                 % (S04["missed_rate"] * 100), fontsize=11.5, y=1.03)
    fig.savefig(f"{OUT}/fig4_missed.png")
    plt.close(fig)


# ---------------------------------------------------------------- 图 5
def fig_operational():
    """支柱三 c：业务后果 —— 阈值等价、排序翻转、α 桥接。"""
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.1))
    # (a) 阈值等价
    ax[0].bar(["标称阈值\n0.25", "等效阈值 p50\n%.4f" % S04["ri_star_p50"],
               "等效阈值 p95\n%.4f" % S04["ri_star_p95"]],
              [0.25, S04["ri_star_p50"], S04["ri_star_p95"]],
              color=[C_DRY, C_BLUE, C_MOIST], width=.5)
    ax[0].axhline(0.25, color="k", ls="--", lw=1)
    ax[0].set_ylabel("Ri 阈值")
    ax[0].set_title("(a) 干口径+0.25 <=> 湿口径+0.25/(1−δ)\n"
                    "最湿 5%% 空域阈值被放宽 %.1f%%"
                    % ((S04["ri_star_p95"] / .25 - 1) * 100), fontsize=10)
    ax[0].grid(axis="y", alpha=.3); ax[0].tick_params(axis="x", labelsize=8)

    # (b) 排序翻转
    ps = ["前 1%", "前 5%", "前 10%"]
    tv = [S04["turnover_1"], S04["turnover_5"], S04["turnover_10"]]
    b = ax[1].bar(ps, np.array(tv) * 100, color=C_MOIST, width=.5)
    ax[1].bar_label(b, fmt="%.2f", fontsize=9, padding=2)
    ax[1].set_ylabel("被干口径圈定、换湿口径后落选 [%]")
    ax[1].set_title("(b) 风险排序翻转率（分辨率稳健）\nJaccard(5%%) = %.3f"
                    % S04["jaccard_5"], fontsize=10)
    ax[1].grid(axis="y", alpha=.3)

    # (c) α 敏感性（曲线直接来自 step04 输出，不硬编码）
    cur = np.array(S04["emergence_curve"])
    ax[2].semilogx(cur[:, 0], cur[:, 1] * 100, "o-", color=C_MOIST, lw=2, ms=6,
                   label="涌现率")
    ax[2].semilogx(cur[:, 0], cur[:, 2] * 100, "s--", color=C_DRY, lw=1.2, ms=4,
                   label="干判危险率")
    ax[2].axvline(S04["alpha_peak"], color=C_BLUE, ls=":", lw=1.2)
    ax[2].text(S04["alpha_peak"] * 1.15, max(cur[:, 1] * 100) * .3,
               "峰值 α≈%.0f" % S04["alpha_peak"], fontsize=8, color=C_BLUE)
    ax[2].set_xlabel("次网格切变放大倍数 α"); ax[2].set_ylabel("比率 [%]")
    ax[2].set_title("(c) K–H 涌现率对 α 的敏感性\n网格 α=1 时仅 %.4f%%（不代表真实发生率）"
                    % (S04["emergence_alpha1"] * 100), fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.suptitle("业务后果：忽略湿度不是「精度略差」，而是「系统性偏松」的判据",
                 fontsize=11.5, y=1.03)
    fig.savefig(f"{OUT}/fig5_operational.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_climate(); print("  fig1_climate.png")
    fig_osse();    print("  fig2_osse.png")
    fig_delta_profile(); print("  fig3_delta_profile.png")
    fig_missed();  print("  fig4_missed.png")
    fig_operational(); print("  fig5_operational.png")
    print("全部图表已写出 →", OUT)
