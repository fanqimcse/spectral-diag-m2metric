# -*- coding: utf-8 -*-
"""第二轮追加图表（fig6–fig8）→ figures/*.png

    fig6  垂直分辨率气候学（3 区域 × 3 季节 × 2 时次，原生 21 层 vs 业务 11 层）
    fig7  PIREP 观测检验：干 / 虚温 / Durran-Klemp 饱和式三种 Ri 的 AUC
    fig8  OSSE 归因：湿度盲观测能榨出多少湿度信息（含"直报湿度"硬件上限）

所有数字直接来自 data/*.json，不在此处硬编码。
"""
import io
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
C_MOIST = "#c0392b"
C_DRY = "#7f8c8d"
C_BLUE = "#2471a3"
C_GREEN = "#1e8449"
C_ORANGE = "#d68910"


def jload(p):
    return json.load(io.open(p, encoding="utf8"))


# ------------------------------------------------------------------ fig6
def fig6_resolution():
    d = jload("data/step06_resolution_climatology.json")
    S, P = d["summary"], d["profile"]
    regs = ["EA  东亚", "NA  北美", "TWP 热带西太平洋"]
    seas = ["冬", "夏", "秋"]
    ls = ["L21", "L11", "L07"]
    lab_ls = {"L21": "原生 21 层", "L11": "业务 11 层", "L07": "粗 7 层"}

    # (a) δ 的垂直结构（跨区域/季节/时次的中位）
    import collections
    by = collections.defaultdict(list)
    for x in P:
        by[x["lev"]].append(x["delta"])
    lev = np.array(sorted(by, reverse=True))
    med = np.array([np.nanmedian(by[k]) for k in lev])
    lo = np.array([np.nanpercentile(by[k], 25) for k in lev])
    hi = np.array([np.nanpercentile(by[k], 75) for k in lev])

    fig, AX = plt.subplots(1, 3, figsize=(15.5, 4.4))
    ax = AX[0]
    ax.semilogx(np.maximum(med * 100, 1e-3), lev, "-o", color=C_MOIST, ms=4, lw=1.8)
    ax.fill_betweenx(lev, np.maximum(lo * 100, 1e-3), hi * 100,
                     color=C_MOIST, alpha=.18, lw=0)
    ax.axvline(1, color=C_DRY, ls="--", lw=1)
    ax.set_ylim(1000, 90)
    ax.set_yscale("log")
    ax.set_yticks([1000, 850, 700, 500, 300, 200, 100])
    ax.set_yticklabels(["1000", "850", "700", "500", "300", "200", "100"])
    ax.set_xlabel(r"$\delta$ = (N_d$^2$ − N_v$^2$)/N_d$^2$  [%]")
    ax.set_ylabel("气压 [hPa]")
    ax.grid(alpha=.3, which="both")
    ax.set_title("(a) δ 的气候学垂直结构\n3 区域×3 季节×2 时次（掩蔽地面以下）", fontsize=10)
    for lvl, v in ((1000, med[list(lev).index(1000.0)]), (250, med[list(lev).index(250.0)])):
        ax.annotate("%.2f%%" % (v * 100), (max(v * 100, 1e-3), lvl),
                    textcoords="offset points", xytext=(8, -2), fontsize=8,
                    color=C_MOIST)

    # (b) δ_p50：原生 vs 业务，按区域×季节
    ax = AX[1]
    x = np.arange(len(regs) * len(seas))
    w = 0.27
    for i, l in enumerate(ls[:2]):
        v = np.array([S["%s|%s|%s" % (r, s, l)]["delta"] * 100
                      for r in regs for s in seas])
        b = ax.bar(x + (i - 0.5) * w, v, w, label=lab_ls[l],
                   color=(C_MOIST if l == "L21" else C_BLUE))
        ax.bar_label(b, fmt="%.1f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["%s·%s" % (r.split()[0], s) for r in regs for s in seas],
                       fontsize=8)
    ax.set_ylabel("δ 中位 [%]")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    ax.set_title("(b) 湿度对静力稳定度的贡献\n业务 11 层系统性低于原生 21 层", fontsize=10)

    # (c) 漏判率：三档分辨率
    ax = AX[2]
    col = [C_MOIST, C_BLUE, C_GREEN]
    for i, l in enumerate(ls):
        v = np.array([max(S["%s|%s|%s" % (r, s, l)]["missed"], 1e-6) * 100
                      for r in regs for s in seas])
        b = ax.bar(x + (i - 1) * w, v, w, label=lab_ls[l], color=col[i])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["%s·%s" % (r.split()[0], s) for r in regs for s in seas],
                       fontsize=8)
    ax.set_ylabel("干口径漏判率 [%]（对数）")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=.3)
    ax.set_title("(c) 符号级漏判率（对数轴）\n层越粗 → 弱层结被平滑掉 → 业务层组看不见这些事件",
                 fontsize=10)
    fig.suptitle("图 6  垂直分辨率气候学：湿稳定度偏差不是个例，而是气候学的",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(OUT, "fig6_resolution_climatology.png"))
    plt.close(fig)
    print("  figures/fig6_resolution_climatology.png")


# ------------------------------------------------------------------ fig7
def fig7_pirep():
    d = jload("data/pirep/step08_pirep_verification.json")
    keys = ["全部（含低层）", "晴空 RH<80%", "晴空 + 高层 p<=400hPa"]
    short = ["全部\nn=%d" % d["subsets"][keys[0]]["n"],
             "晴空\nn=%d" % d["subsets"][keys[1]]["n"],
             "晴空+高层\nn=%d" % d["subsets"][keys[2]]["n"]]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.3))

    A = ax[0]
    x = np.arange(len(keys)); w = 0.26
    for i, (k, nm, c) in enumerate([("auc_dry", "干 N_d$^2$（业务）", C_DRY),
                                    ("auc_virt", "虚温 N_v$^2$", C_BLUE),
                                    ("auc_dk", "Durran-Klemp 饱和式\n(GTG 的 Rimoist)",
                                     C_MOIST)]):
        v = [d["subsets"][kk][k] for kk in keys]
        b = A.bar(x + (i - 1) * w, v, w, label=nm, color=c)
        A.bar_label(b, fmt="%.3f", fontsize=7.5, padding=1)
    A.axhline(0.5, color="k", ls=":", lw=1)
    A.set_xticks(x); A.set_xticklabels(short, fontsize=9)
    A.set_ylim(0.45, 0.78)
    A.set_ylabel("AUC（PIREP 颠簸报告为阳性）")
    A.legend(fontsize=8, loc="lower right")
    A.grid(axis="y", alpha=.3)
    A.set_title("(a) 三种口径理查森数的判别技巧\n—— 3706 份 PIREP 独立观测", fontsize=10)

    B = ax[1]
    labs, dv, ci_dv, dk, ci_dk = [], [], [], [], []
    for kk, nm in zip(keys, short):
        v = d["subsets"][kk]
        labs.append(nm.replace("\n", " "))
        dv.append(v["d_auc_virt"])
        ci_dv.append([[abs(v["d_auc_virt"] - v["d_auc_virt_ci"][0])],
                      [abs(v["d_auc_virt_ci"][1] - v["d_auc_virt"])]])
        dk.append(v["d_auc_dk"])
        ci_dk.append([[abs(v["d_auc_dk"] - v["d_auc_dk_ci"][0])],
                      [abs(v["d_auc_dk_ci"][1] - v["d_auc_dk"])]])
    yv, yd = np.arange(len(labs)), np.arange(len(labs))
    B.errorbar(dv, yv + 0.12, xerr=np.array(ci_dv).squeeze().T, fmt="o",
               color=C_BLUE, capsize=4, label="虚温 − 干式  ΔAUC")
    B.errorbar(dk, yd - 0.12, xerr=np.array(ci_dk).squeeze().T, fmt="s",
               color=C_MOIST, capsize=4, label="DK 饱和式 − 干式  ΔAUC")
    B.axvline(0, color="k", lw=1)
    B.set_yticks(yv); B.set_yticklabels(labs, fontsize=9)
    B.set_xlabel("ΔAUC（95% 自助置信区间，1000 次重抽样）")
    B.legend(fontsize=8, loc="lower left")
    B.grid(axis="x", alpha=.3)
    B.set_title("(b) 成对差值：假设 H1（虚温更优）被否证，\n假设 H2（晴空误用饱和式有害）被强支持",
                fontsize=10)
    fig.suptitle("图 7  这不是模式自洽性检验，而是 3706 份真实颠簸报告的判别检验",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(OUT, "fig7_pirep_verification.png"))
    plt.close(fig)
    print("  figures/fig7_pirep_verification.png")


# ------------------------------------------------------------------ fig8
def fig8_osse():
    p = "data/step10_realerror_osse.json"
    if not os.path.exists(p):
        print("  (跳过 fig8：尚无 %s)" % p)
        return
    d = jload(p)
    leads = sorted(int(x) for x in d["results"])
    NAMES = ["A  地面+探空       无飞机", "B  +飞机 单变量(湿度盲)",
             "C  B+T2V 静力-地转", "D  B+V2q 水汽平流", "E  B+全耦合",
             "F  +飞机直报湿度(上限)"]

    def med(rows, k):
        v = [r[k] for r in rows if k in r]
        return float(np.median(v)) if v else float("nan")

    def med_gain(rows, nm, kv):
        """逐时次归一化 → 跨时次中位（与 step10.med_pct 同一口径）。

        旧写法是"中位 RMSE 相减"，跨时次背景误差差 3 倍时会失真。
        """
        v = [100 * (r["bg|" + kv] - r[nm + "|" + kv]) / r["bg|" + kv]
             for r in rows if (nm + "|" + kv) in r and r["bg|" + kv] > 0]
        return float(np.median(v)) if v else float("nan")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    KEYS = [("T", "T [K]"), ("q", "q [kg/kg]"), ("u", "u [m/s]"), ("v", "v [m/s]")]

    A = ax[0]
    rows = d["results"][str(leads[-1])]["rows"]
    x = np.arange(len(KEYS)); w = 0.8 / len(NAMES)
    for i, nm in enumerate(NAMES):
        vals = [med_gain(rows, nm, kv) for kv, _ in KEYS]
        A.bar(x + (i - len(NAMES) / 2 + .5) * w, vals, w,
              label=nm.strip()[:16], alpha=.9)
    A.set_xticks(x); A.set_xticklabels([kv for kv, _ in KEYS])
    A.axhline(0, color="k", lw=1)
    A.set_ylabel("相对背景的 RMSE 降幅 [%]")
    A.legend(fontsize=7, ncol=2)
    A.grid(axis="y", alpha=.3)
    A.set_title("(a) f%03d 各配置 RMSE 降幅（影响区 ∩ 地面以上）" % leads[-1],
                fontsize=10)

    B = ax[1]
    NUL = jload("data/step10c_null.json") if os.path.exists(
        "data/step10c_null.json") else None
    if NUL:
        variants = [("real", "real", C_MOIST, "real（真实分析增量）"),
                    ("smoothq", "smoothq", C_DRY, "纯低通（平滑冒领）"),
                    ("randsmooth", "randsmooth", C_ORANGE, "随机平滑位移（安慰剂）")]
        xl = np.arange(len(leads))
        w = 0.8 / len(variants)
        for i, (key, _, c, lab) in enumerate(variants):
            v = [NUL[str(ld)]["median"][key] for ld in leads]
            b = B.bar(xl + (i - 1) * w, v, w, label=lab, color=c)
            B.bar_label(b, fmt="%.2f", fontsize=7.5, padding=1)
        B.set_xticks(xl); B.set_xticklabels(["f%03d" % l for l in leads])
        B.axhline(0, color="k", lw=1)
        B.set_ylabel("湿度 RMSE 改善 [百分点]")
        B.legend(fontsize=7.5)
        B.grid(axis="y", alpha=.3)
        B.set_title("(b) V2q 增益分解：真实 vs 安慰剂\n（净流动依赖 = real − max(安慰剂)）",
                    fontsize=10)
    else:
        show = [NAMES[0], NAMES[1], NAMES[3], NAMES[5]]
        lab_s = ["A 无飞机", "B +飞机\n(湿度盲)", "D +V2q\n水汽平流",
                 "F 飞机直报\n(硬件上限)"]
        col = [C_DRY, C_BLUE, C_MOIST, C_ORANGE]
        xl = np.arange(len(leads))
        w = 0.8 / len(show)
        for ld in leads:
            rows = d["results"][str(ld)]["rows"]
            bgq = med(rows, "bg|q")
            xi = xl[leads.index(ld)]
            for i, nm in enumerate(show):
                v = 100 * (bgq - med(rows, nm + "|q")) / bgq
                b = B.bar(xi + (i - len(show) / 2 + .5) * w, v, w,
                          label=lab_s[i] if ld == leads[0] else None, color=col[i])
                B.bar_label(b, fmt="%.2f", fontsize=7, padding=1)
        B.set_xticks(xl); B.set_xticklabels(["f%03d" % l for l in leads])
        B.axhline(0, color="k", lw=1)
        B.set_ylabel("湿度 RMSE 改善 [百分点]")
        B.legend(fontsize=7.5)
        B.grid(axis="y", alpha=.3)
        B.set_title("(b) 湿度信息的归因：软件耦合 vs 加装硬件", fontsize=10)

    C = ax[2]
    lev = np.array([1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
                    550, 500, 450, 400, 350, 300, 250, 200, 150, 100], float)
    for ld in leads:
        sb = np.array(d["results"][str(ld)]["sig_b"]["q"])
        C.semilogx(np.maximum(sb, 1e-9) * 1e3, lev, "-o", ms=3.5, label="f%03d" % ld)
    C.set_ylim(1000, 90); C.set_yscale("log")
    C.set_yticks([1000, 850, 700, 500, 300, 200, 100])
    C.set_yticklabels(["1000", "850", "700", "500", "300", "200", "100"])
    C.set_xlabel("背景湿度误差 σ_b(q) [g/kg]")
    C.legend(fontsize=8)
    C.grid(alpha=.3, which="both")
    C.set_title("(c) 真实预报误差的逐层幅度\n（决定 OI 增益 K = σ_b²/(σ_b²+σ_o²)）",
                fontsize=10)

    fig.suptitle("图 8  真实预报误差 OSSE：背景误差是动力学协调的，且必须过安慰剂对照",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(OUT, "fig8_osse_realerror.png"))
    plt.close(fig)
    print("  figures/fig8_osse_realerror.png")


if __name__ == "__main__":
    print("生成追加图表 …")
    fig6_resolution()
    fig7_pirep()
    fig8_osse()
    print("完成。")
