# -*- coding: utf-8 -*-
"""step18c —— 定案："业务层组到底把 δ 低估还是高估？"

矛盾
----
- step06（REPORT §0 的头号数字）：业务 11 层把湿度对稳定度的贡献
  **低估 1.2–3.9 倍**、把符号级湿不稳定事件漏掉 4–30 倍。
- step18（沿航迹 EDR 样本，分层）：δ(L11)/δ(L21) = 1.06/1.36/1.46/0.89，
  即中层**高估**。
- step18b：证明这不是垂直插值造成的（共享层口径给出同样比值）。

根因假设（本步检验）
--------------------
step06 的统计量是"**跨所有层合并后取中位**"。但 L21 与 L11 是
**不同的气压层集合**：L21 的中位落在 ~400–500 hPa 附近，L11 的中位落在
别的高度。δ 随高度变化一个数量级以上（上对流层 0.03% → 850 hPa 12%）。
所以"跨层合并中位"这个统计量把 **垂直分辨率效应** 与
**高度采样差异** 混在了一起 —— 它是一个**混淆的**量，不能用于
回答"更粗的分辨率会怎样"。

受控比较必须**固定气压层**：在两个层组共有的气压上，各自用自己的
垂直差分算诊断，然后逐层对比。

本步做法
--------
6 个算例（3 个时次 × 2 个区域），在 L11 的每一层上分别用 L21/L11/L07
三个层组算诊断（各自用自己层组的有限差分，但**求值在同一气压**），
给出：
  1. 逐层 δ 与比值（受控口径）；
  2. 跨层合并中位（复现 step06 口径），用于展示两者差异；
  3. 同一口径下的"符号级湿不稳定漏判率" missed = P(Nd²>0 且 Nv²<0)。
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from avwx import aws_gfs as AG
from avwx import risk as RK
import step18_ops_levelset_skill as S

OUT = "data/step18c_delta_convention.json"
RH_CLEAR = 80.0
N2_MIN = 1e-4

CASES = [("20260115", "18", "EA", (100.0, 145.0, 15.0, 45.0)),
         ("20260115", "18", "NA", (235.0, 295.0, 25.0, 55.0)),
         ("20260715", "18", "EA", (100.0, 145.0, 15.0, 45.0)),
         ("20260715", "18", "NA", (235.0, 295.0, 25.0, 55.0)),
         ("20261019", "18", "EA", (100.0, 145.0, 15.0, 45.0)),
         ("20261019", "18", "NA", (235.0, 295.0, 25.0, 55.0))]

LEVELS = ("L21", "L11", "L07")


def diag_on(d, want):
    idx = S.level_index(d["level"], want)
    lev = np.asarray(d["level"], float)[idx]
    f = {v: np.asarray(d[v], float)[idx] for v in ("T", "q", "u", "v", "z")}
    return RK.stability_fields(f, d["lat"], d["lon"], lev), lev


def main():
    res = {"per_level": {}, "pooled": {}, "cases": []}
    acc = {t: {} for t in LEVELS}          # acc[tag][p] = [delta_med, ...]
    accm = {t: {} for t in LEVELS}
    pool = {t: [] for t in LEVELS}         # 跨层合并（step06 口径）
    poolm = {t: [] for t in LEVELS}

    for date, cyc, rname, bb in CASES:
        g = sorted(glob.glob("data/aws/gfs_%s_%s_f000_L21_1000_100.grib2"
                             % (date, cyc)))
        if not g:
            print("%s %sz %s 缺 GFS，跳过" % (date, cyc, rname))
            continue
        try:
            d = AG.load(g[0], bbox=bb)
        except Exception as e:
            print("%s %sz %s 载入失败: %s" % (date, cyc, rname, str(e)[:60]))
            continue
        dg, levs, idx = {}, {}, {}
        above = (np.asarray(d["level"], float)[:, None, None] * 100.0
                 <= np.asarray(d["p_sfc"], float)[None, ...])
        for t in LEVELS:
            dg[t], levs[t] = diag_on(d, S.LEVEL_SETS[t])
            idx[t] = S.level_index(d["level"], S.LEVEL_SETS[t])
        # 共有气压：L11 的每一层，且三个层组都能在 ±1 hPa 内取到
        for t in LEVELS:
            k_lv = levs[t]
            for kk in range(len(k_lv)):
                mm = (RK.valid_mask(dg[t], RH_CLEAR, 1e-5) & above[idx[t]])[kk]
                if mm.sum() > 200:
                    pool[t].append(float(np.median(dg[t]["delta"][kk][mm])))
        for pl in S.LEVEL_SETS["L11"]:
            ks = {}
            ok = True
            for t in LEVELS:
                k = int(np.argmin(np.abs(levs[t] - pl)))
                if abs(levs[t][k] - pl) > 1.0:
                    ok = False
                    break
                ks[t] = k
            if not ok:
                continue
            for t in LEVELS:
                k = ks[t]
                m = (RK.valid_mask(dg[t], RH_CLEAR, 1e-5)
                     & above[idx[t]][None][0] if False else
                     RK.valid_mask(dg[t], RH_CLEAR, 1e-5)[k]
                     & above[idx[t]][k])
                if m.sum() < 200:
                    continue
                acc[t].setdefault(pl, []).append(
                    float(np.median(dg[t]["delta"][k][m])))
                miss = RK.missed_virtual_instability(dg[t], RH_CLEAR)[k] \
                    & above[idx[t]][k]
                clr = ((dg[t]["rh"][k] < RH_CLEAR) & (dg[t]["Nd2"][k] > 0)
                       & above[idx[t]][k])
                if clr.sum() > 200:
                    accm[t].setdefault(pl, []).append(
                        float(miss.sum()) / float(clr.sum()))
            # 跨层合并（step06 口径）：每层算一次中位，再把各层混在一起取中位
            for t in LEVELS:
                pass
        res["cases"].append("%s_%sz_%s" % (date, cyc, rname))
        print("  完成 %s %sz %s" % (date, cyc, rname), flush=True)
        del d, dg
    # ---------------- 输出 ----------------
    print("\n" + "=" * 100)
    print("受控口径（固定气压层，各自用自己层组差分）  晴空 RH<80，Nd²>1e-4")
    print("=" * 100)
    print("%-8s %6s | %10s %10s %10s | %8s %8s | %10s %10s %10s"
          % ("p[hPa]", "n层", "δ L21", "δ L11", "δ L07", "11/21", "07/21",
             "漏判21", "漏判11", "漏判07"))
    for pl in sorted(acc["L21"], reverse=True):
        def md(tag, src):
            v = src[tag].get(pl)
            return float(np.median(v)) if v else float("nan")
        d21, d11, d07 = md("L21", acc), md("L11", acc), md("L07", acc)
        m21, m11, m07 = md("L21", accm), md("L11", accm), md("L07", accm)
        res["per_level"][str(int(pl))] = dict(
            d21=d21, d11=d11, d07=d07, m21=m21, m11=m11, m07=m07,
            r11=(d11 / d21 if d21 and np.isfinite(d21) and d21 > 0 else None),
            r07=(d07 / d21 if d21 and np.isfinite(d21) and d21 > 0 else None))
        print("%-8d %6d | %9.4f%% %9.4f%% %9.4f%% | %8.2f %8.2f | "
              "%9.5f %9.5f %9.5f"
              % (int(pl), len(acc["L21"][pl]), 100 * d21, 100 * d11,
                 100 * d07, (d11 / d21 if d21 > 0 else float("nan")),
                 (d07 / d21 if d21 > 0 else float("nan")),
                 m21, m11, m07))

    print("\n" + "-" * 100)
    print("跨层合并口径（复现 step06）：把各层的中位混在一起再取中位")
    print("-" * 100)
    for t in LEVELS:
        v = float(np.median(pool[t])) if pool[t] else float("nan")
        res["pooled"][t] = v
        print("  %-4s  合并中位 δ = %.5f%%   （参与层次数 %d）"
              % (t, 100 * v, len(pool[t])))
    if pool["L21"] and pool["L11"]:
        r = res["pooled"]["L11"] / res["pooled"]["L21"]
        res["pooled"]["ratio_11_21"] = r
        print("  ⇒ 合并口径 L11/L21 = %.3f  （step06 报的是 <1）" % r)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n已写出 " + OUT)


if __name__ == "__main__":
    main()
