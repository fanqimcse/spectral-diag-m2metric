# -*- coding: utf-8 -*-
"""step38 —— 核验技术报告 §15.4.2 引用的两个谱量。

背景
----
技术报告（REPORT.md §15.4 推论 A）与 P5_COROLLARY_A_RESULT.md 引用了：

    | GFS f024 背景误差 | k_c = 0.0156 | 波长 1783 km | 大尺度占比 99.8% |
    | EnOI 重建误差     | k_c = 0.0358 | 波长  777 km | 大尺度占比 98.9% |

但审计发现两个问题：
  1. `step35_smooth_vs_assim.py` 里**没有**谱分解代码，其 JSON 也**没有**
     对应字段——该表来自内联计算，无脚本/产物支撑；
  2. `step30b_diag_spectrum.json` 中 kc_field 的**跨 24 行中位数 = 0.0335**，
     而 0.0156 只是单个配对（2022011500，变量 T）的值。

本脚本用 step30b 同一套框架（同样的 8 配对、250 hPa、同样的谱分解与质心定义）
**实测重算**，回答：
  (Q1) GFS f024 背景误差的谱质心 k_c 到底是多少（给中位数 + 四分位）？
  (Q2) "大尺度占比"在若干明确切分下分别是多少？99.8% 在什么口径下成立？

口径说明（重要）
----------------
"大尺度占比"依赖切分波数 k_cut 的选取，本身不是唯一量。本脚本同时报告：
  · 以谱质心 k_c 为切分（大尺度 = k < k_c）；
  · 以固定物理波长为切分（L > 1000 / 500 / 300 / 200 km）；
  · 以报告中隐含的 1783 km 为切分。
由此可判断"99.8%"是否稳健，还是切分方式造成的假象。
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import step10_realerror_osse as S10
from step30b_diag_spectrum import PAIRS, BG_LEAD, spectrum2d, centroid

OUT = "data/step38_verify_spectral_claim.json"
CRUISE = (200.0, 400.0)
# 物理波长切分（km）——格距按 0.25° 估算，与报告 1783 km 的换算口径一致
WAVELENGTH_CUTS_KM = (1783.0, 1000.0, 500.0, 300.0, 200.0)


def large_scale_fraction(k, power, k_cut):
    """大尺度功率占比 = ∫_{k<k_cut} P dk / ∫ P dk。"""
    k = np.asarray(k)
    p = np.asarray(power)
    dk = np.gradient(k)
    tot = np.sum(p * dk)
    if tot <= 0:
        return float("nan")
    sel = k < k_cut
    return float(np.sum(p[sel] * dk[sel]) / tot)


def wavelength_to_k(lambda_km, dlat_m):
    """物理波长（km）→ 波数（cycles per grid point）。"""
    lambda_m = lambda_km * 1000.0
    n_grid = lambda_m / dlat_m          # 波长跨越的格点数
    return float(1.0 / n_grid)


def main():
    t0 = time.time()
    print("=" * 96)
    print("step38  核验：GFS f024 背景误差的谱质心与大尺度占比")
    print("=" * 96)

    rows = []
    for (dte, cyc) in PAIRS:
        bspec = S10.bg_cycle(dte, cyc, BG_LEAD)
        nat_path = S10.native_path(dte, cyc, 0)
        bg_path = S10.native_path(bspec[0], bspec[1], BG_LEAD)
        if not (os.path.exists(nat_path) and os.path.exists(bg_path)):
            print("  跳过（数据缺失）%s %sz" % (dte, cyc))
            continue
        nat = S10.prepare(S10.AG.load(nat_path, bbox=S10.BBOX), "nat")
        bg = S10.prepare(S10.AG.load(bg_path, bbox=S10.BBOX), "bg")
        lat, lon, lev = nat["lat"], nat["lon"], nat["lev"]
        above = nat["above"] & bg["above"]
        i250 = int(np.argmin(np.abs(lev - 250.0)))
        mask2d = above[i250]
        dlat = float(np.mean(np.diff(lat))) * 111195.0

        for v in ("T", "u", "v"):
            eb2d = (bg["f"][v] - nat["f"][v])[i250]
            m = np.isfinite(eb2d) & mask2d
            k, p = spectrum2d(eb2d, m)
            kc = centroid(k, p)
            rec = dict(target="%s%sz" % (dte, cyc), var=v,
                       kc_field=kc,
                       lambda_km=float(dlat / 1000.0 / kc) if kc > 0 else np.nan)
            # 以谱质心为切分
            rec["frac_at_kc"] = large_scale_fraction(k, p, kc)
            # 以固定物理波长为切分
            for wl in WAVELENGTH_CUTS_KM:
                kc_wl = wavelength_to_k(wl, dlat)
                rec["frac_%.0fkm" % wl] = large_scale_fraction(k, p, kc_wl)
            rows.append(rec)

        del nat, bg
        print("  完成 %s%sz" % (dte, cyc), flush=True)

    def med(key):
        v = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    def q(key, pct):
        v = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.percentile(v, pct)) if v else float("nan")

    print("\n" + "=" * 96)
    print("【Q1】GFS f024 背景误差谱质心 k_c（跨 %d 个配对×变量）" % len(rows))
    print("=" * 96)
    print("  中位数 k_c        = %.4f" % med("kc_field"))
    print("  四分位 [25%%, 75%%] = [%.4f, %.4f]" % (q("kc_field", 25), q("kc_field", 75)))
    print("  范围              = [%.4f, %.4f]"
          % (min(r["kc_field"] for r in rows), max(r["kc_field"] for r in rows)))
    print("  对应波长中位数     = %.0f km" % med("lambda_km"))
    print("\n  ⚠ 报告中引用的 k_c = 0.0156 是**单个配对**值；跨样本中位数为 %.4f。"
          % med("kc_field"))

    print("\n" + "=" * 96)
    print("【Q2】大尺度功率占比（依赖切分口径，非唯一量）")
    print("=" * 96)
    print("  %-22s %14s %14s" % ("切分口径", "中位占比", "四分位[25,75]"))
    print("  %-22s %14s %14s" % ("以谱质心 k_c 为切分",
                                 "%.2f%%" % (100 * med("frac_at_kc")),
                                 "[%.1f, %.1f]" % (100 * q("frac_at_kc", 25),
                                                   100 * q("frac_at_kc", 75))))
    for wl in WAVELENGTH_CUTS_KM:
        key = "frac_%.0fkm" % wl
        print("  %-22s %14s %14s" % ("L > %.0f km" % wl,
                                     "%.2f%%" % (100 * med(key)),
                                     "[%.1f, %.1f]" % (100 * q(key, 25),
                                                       100 * q(key, 75))))

    # 判定：报告中 99.8% 的口径是否可复现
    frac_1783 = med("frac_1783km")
    print("\n  判定：报告称'大尺度占 99.8%%'。在 L>1783 km 切分下实测中位 = %.2f%%"
          % (100 * frac_1783))
    print("        （该数字仅在**极低的切分波数**下才会接近 99.8%；")
    print("         切分波数一旦移到谱质心附近，占比显著下降。）")

    out = dict(
        n=int(len(rows)),
        kc=dict(median=med("kc_field"),
                q25=q("kc_field", 25), q75=q("kc_field", 75),
                min=float(min(r["kc_field"] for r in rows)),
                max=float(max(r["kc_field"] for r in rows)),
                lambda_km_median=med("lambda_km")),
        frac=dict(**{("at_kc"): med("frac_at_kc")},
                  **{("%.0fkm" % wl): med("frac_%.0fkm" % wl)
                     for wl in WAVELENGTH_CUTS_KM}),
        frac_q25={("%.0fkm" % wl): q("frac_%.0fkm" % wl, 25)
                  for wl in WAVELENGTH_CUTS_KM},
        frac_q75={("%.0fkm" % wl): q("frac_%.0fkm" % wl, 75)
                  for wl in WAVELENGTH_CUTS_KM},
        rows=rows,
        note=("报告 §15.4.2 的 k_c=0.0156/99.8% 无脚本支撑；"
              "实测跨样本中位数 k_c 见 kc.median"),
    )
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
