# -*- coding: utf-8 -*-
"""
— 垂直分辨率敏感性 + 多区域多季节气候学 + 干/虚温/饱和三口径逐项对照 —

使用 AWS 抽取的 GFS **原生 21 层**（1000→100 hPa）分析场。同一份全球文件
可切出三个区域，因此"多区域"不需要额外下载；每个区域单独 load 以避免把
全球场整份驻留（21 层 × 全球 float64 约 0.9 GB，风险过高）。

本脚本回答四个问题：

【问题 1】垂直分辨率：δ 与漏判率在 7 / 11 / 21 层是否单调收敛？
          若 11 层严重低估，则先前基于 NOMADS 11 层的结果必须重估。

【问题 2】气候学：δ 与漏判率在东亚 / 北美 / 热带西太平洋 ×
          冬(1月) / 夏(7月) / 秋(10月) 各是多少？是否只是个例？

【问题 3】地形遮蔽：NOMADS 等压面不做地形遮蔽，未遮蔽会引入多大偏差？

【问题 4】口径对照：业务事实标准 Durran–Klemp **饱和式** 与物理正确的
          **虚温式** 在晴空区差多少？特别是把空气人为置干之后，虚温修正
          应恒等于 0，而饱和式仍会给出非零"湿度修正"——这个虚假值有多大？
          （判定晴空区误用饱和式的关键证据）
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath("."))
from avwx import aws_gfs as AG
from avwx import risk as RK

RH_CLEAR = 80.0
N2_FLOOR = 1e-5

REGIONS = {
    "EA  东亚": (100, 130, 15, 40),
    "NA  北美": (235, 290, 25, 50),
    "TWP 热带西太平洋": (120, 165, -10, 20),
}
SEASONS = ("冬", "夏", "秋")

SEASON_TIMES = [
    ("20260115", "06", "冬"), ("20260115", "18", "冬"),
    ("20260715", "06", "夏"), ("20260715", "18", "夏"),
    ("20261019", "06", "秋"), ("20261019", "18", "秋"),
]

# 三档垂直分辨率（均取自同一份 21 层数据，按层抽稀后重新诊断）
LEVEL_SETS = {
    "L21 原生": [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
                 550, 500, 450, 400, 350, 300, 250, 200, 150, 100],
    "L11 业务": [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100],
    "L07 粗": [1000, 850, 700, 500, 300, 200, 100],
}


def level_index(lev, want):
    """返回 want 各层在 lev 中的索引，保证 lev[idx] **降序**（气压随索引递减）。

    踩坑记录（务必保留此断言）
    ------------------------
    曾经写成 ``sorted(set(idx))[::-1]``。lev 本身是降序的 1000→100，
    索引升序 0,1,2,… 已经对应降序气压；再 [::-1] 会把整条廓线**上下颠倒**，
    于是 lev[idx] 变成升序，stability_fields 收到的 z 序列随之反向，
    所有垂直导数、δ、漏判率全部反号而不报错 ——
    实测把 δ 从 +26% 变成 −0.6%，隐蔽性极强。
    """
    idx = []
    for L in want:
        j = int(np.argmin(np.abs(lev - L)))
        if abs(lev[j] - L) > 1.0:
            raise RuntimeError("层 %.0f 不在数据中" % L)
        idx.append(j)
    idx = np.array(sorted(set(idx)), dtype=int)
    if not np.all(np.diff(lev[idx]) < 0):
        raise RuntimeError("level_index 未返回降序气压：%s" % lev[idx])
    return idx


def main():
    t0 = time.time()
    recs = []      # 一行 = 一个 (区域, 时次, 层组)
    prof = []      # 一行 = 一个 (区域, 时次, 层)
    conv = []      # 一行 = 一个 (区域, 时次) 的三口径对照

    print("=" * 80)
    print("GFS 原生 21 层 · 三区域 · 三季节 · 三垂直分辨率 · 三 N² 口径")
    print("=" * 80)

    for dte, cyc, seas in SEASON_TIMES:
        pth = "data/aws/gfs_%s_%s_f000_L21_1000_100.grib2" % (dte, cyc)
        if not os.path.exists(pth):
            print("  缺失 %s，跳过" % pth)
            continue
        for rname, bbox in REGIONS.items():
            d = AG.load(pth, bbox=bbox)
            lat, lon, lev = d["lat"], d["lon"], d["level"]
            # ---- 地形遮蔽：气压层必须在地面之上 ----
            above = (lev[:, None, None] * 100.0) <= d["p_sfc"][None, ...]

            for lname, want in LEVEL_SETS.items():
                idx = level_index(lev, want)
                ab = above[idx]
                f = dict(T=d["T"][idx], q=d["q"][idx], u=d["u"][idx],
                         v=d["v"][idx], z=None if d["z"] is None else d["z"][idx])
                diag = RK.stability_fields(f, lat, lon, lev[idx])
                base = ((diag["Nd2"] > 0) & (diag["rh"] < RH_CLEAR) & ab)
                ok = RK.valid_mask(diag, RH_CLEAR, N2_FLOOR) & ab
                miss = RK.missed_virtual_instability(diag, RH_CLEAR) & ab

                def q(a, mask, pf=50.0):
                    v = a[mask]
                    v = v[np.isfinite(v)]
                    return float(np.percentile(v, pf)) if v.size else np.nan

                recs.append(dict(
                    region=rname, season=seas, time="%s%sz" % (dte, cyc),
                    lset=lname, n_base=int(base.sum()), n_ok=int(ok.sum()),
                    n_missed=int(miss.sum()),
                    missed_rate=float(miss.sum() / max(int(base.sum()), 1)),
                    delta_p50=q(diag["delta"], ok),
                    delta_p95=q(diag["delta"], ok, 95),
                    dsat_p50=q(diag["delta_sat"], ok),
                    dsat_p95=q(diag["delta_sat"], ok, 95),
                    masked_frac=float(1.0 - ab.mean()),
                ))

                if lname.startswith("L21"):
                    for k in range(len(idx)):
                        ba = ((diag["Nd2"][k] > 0) & (diag["rh"][k] < RH_CLEAR)
                              & ab[k])
                        ob = ((diag["Nd2"][k] > N2_FLOOR)
                              & (diag["rh"][k] < RH_CLEAR) & ab[k])
                        prof.append(dict(
                            region=rname, season=seas,
                            time="%s%sz" % (dte, cyc),
                            lev=float(lev[idx[k]]),
                            delta=float(np.nanmedian(diag["delta"][k][ob]))
                            if ob.any() else np.nan,
                            dsat=float(np.nanmedian(diag["delta_sat"][k][ob]))
                            if ob.any() else np.nan,
                            missed_rate=float((ba & (diag["Nv2"][k] < 0)).sum()
                                              / max(int(ba.sum()), 1)),
                            n=int(ba.sum())))
                del diag, f
                gc.collect()

            # ---- 口径对照：空气人为置干 ----
            f0 = dict(T=d["T"], q=np.zeros_like(d["q"]), u=d["u"], v=d["v"],
                      z=None if d["z"] is None else d["z"])
            d0 = RK.stability_fields(f0, lat, lon, lev)
            ok0 = ((d0["Nd2"] > N2_FLOOR) & (d0["rh"] < RH_CLEAR) & above)
            with np.errstate(invalid="ignore", divide="ignore"):
                dsat0 = np.where(d0["Nd2"] > 1e-9,
                                 (d0["Nd2"] - d0["Ndk2"]) / d0["Nd2"], np.nan)
            v0 = d0["delta"][ok0]; v0 = v0[np.isfinite(v0)]
            s0 = dsat0[ok0]; s0 = s0[np.isfinite(s0)]
            # 真实晴空（RH<80%）下两口径之差
            okR = RK.valid_mask(d, RH_CLEAR, N2_FLOOR) if False else None
            conv.append(dict(
                region=rname, season=seas, time="%s%sz" % (dte, cyc),
                dry_delta_maxabs=float(np.max(np.abs(v0))) if v0.size else np.nan,
                dry_dsat_p50=float(np.median(s0)) if s0.size else np.nan,
                dry_dsat_p95=float(np.percentile(s0, 95)) if s0.size else np.nan,
                dry_dsat_maxabs=float(np.max(np.abs(s0))) if s0.size else np.nan,
            ))
            del d0, f0, okR
            gc.collect()
            del d
            gc.collect()
        print("  %s %sz 三区域完成 (%.0f s)" % (dte, cyc, time.time() - t0),
              flush=True)

    # ------------------------------------------------------------ 汇总
    def agg(**kw):
        return [r for r in recs if all(r[k] == v for k, v in kw.items())]

    def med(v):
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if v else np.nan

    print("\n" + "=" * 80)
    print("【表 A】垂直分辨率敏感性（晴空 + 地形以上 + N_d²>1e-5 的格点）")
    print("=" * 80)
    print("%-18s %-3s %-8s %9s %9s %10s %10s %8s" %
          ("区域", "季", "层组", "δ中位", "δ p95", "δ_sat中位", "漏判率", "遮蔽"))
    summary = {}
    for rn in REGIONS:
        for se in SEASONS:
            for ln in LEVEL_SETS:
                rr = agg(region=rn, season=se, lset=ln)
                if not rr:
                    continue
                s = dict(delta=med([x["delta_p50"] for x in rr]),
                         dp95=med([x["delta_p95"] for x in rr]),
                         dsat=med([x["dsat_p50"] for x in rr]),
                         missed=med([x["missed_rate"] for x in rr]),
                         masked=med([x["masked_frac"] for x in rr]),
                         n=med([x["n_ok"] for x in rr]))
                summary["%s|%s|%s" % (rn, se, ln.split()[0])] = s
                print("%-18s %-3s %-8s %8.3f%% %8.3f%% %9.3f%% %9.4f%% %7.1f%%" %
                      (rn, se, ln.split()[0], s["delta"] * 100, s["dp95"] * 100,
                       s["dsat"] * 100, s["missed"] * 100, s["masked"] * 100))
            print("-" * 80)

    print("\n【表 B】21 层相对 11 层的变化（11 层低估量）")
    print("%-18s %-3s %10s %10s %8s %10s %10s %8s" %
          ("区域", "季", "δ21", "δ11", "δ比", "漏判21", "漏判11", "漏判比"))
    for rn in REGIONS:
        for se in SEASONS:
            a = summary.get("%s|%s|L21" % (rn, se))
            b = summary.get("%s|%s|L11" % (rn, se))
            if not a or not b:
                continue
            r1 = a["delta"] / b["delta"] if b["delta"] > 0 else np.nan
            r2 = a["missed"] / b["missed"] if b["missed"] > 0 else np.nan
            print("%-18s %-3s %9.3f%% %9.3f%% %7.2f %9.4f%% %9.4f%% %7.2f" %
                  (rn, se, a["delta"] * 100, b["delta"] * 100, r1,
                   a["missed"] * 100, b["missed"] * 100, r2))

    LV = LEVEL_SETS["L21 原生"]
    for col, nm, pc in (("delta", "δ 中位 [%]", 100),
                        ("missed_rate", "漏判率 [%]", 100),
                        ("dsat", "δ_sat 中位 [%]", 100)):
        print("\n【表 C】逐层廓线：%s（21 层，晴空，地形以上）" % nm)
        print("%-16s %-3s " % ("区域", "季") +
              " ".join("%7.0f" % L for L in LV[::3]))
        for rn in REGIONS:
            for se in SEASONS:
                vals = []
                for L in LV[::3]:
                    v = [x[col] for x in prof
                         if x["region"] == rn and x["season"] == se
                         and abs(x["lev"] - L) < 1 and np.isfinite(x[col])]
                    vals.append(np.median(v) if v else np.nan)
                print("%-16s %-3s " % (rn, se) + " ".join(
                    "%6.2f" % (v * pc) if np.isfinite(v) else "     -"
                    for v in vals))

    print("\n【表 D】三口径对照：把空气人为置干（q≡0）后各口径的「湿度修正」")
    print("  物理要求：q≡0 时虚温式修正必须恒为 0；饱和式若非零 ⇒ 晴空区误用")
    print("%-18s %-3s %14s %14s %14s" %
          ("区域", "季", "虚温δ 最大|.|", "饱和δ中位", "饱和δ p95"))
    for rn in REGIONS:
        for se in SEASONS:
            cc = [x for x in conv if x["region"] == rn and x["season"] == se]
            if not cc:
                continue
            print("%-18s %-3s %13.2e %13.3f%% %13.3f%%" %
                  (rn, se,
                   med([x["dry_delta_maxabs"] for x in cc]),
                   med([x["dry_dsat_p50"] for x in cc]) * 100,
                   med([x["dry_dsat_p95"] for x in cc]) * 100))

    with open("data/step06_resolution_climatology.json", "w", encoding="utf8") as f:
        json.dump(dict(summary=summary, profile=prof, convention=conv,
                       records=[{k: v for k, v in r.items()} for r in recs],
                       meta=dict(rh_clear=RH_CLEAR, n2_floor=N2_FLOOR,
                                 regions={k: list(v) for k, v in REGIONS.items()},
                                 level_sets={k: v for k, v in LEVEL_SETS.items()},
                                 times=SEASON_TIMES)),
                  f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step06_resolution_climatology.json   用时 %.0f s"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
