# -*- coding: utf-8 -*-
"""step38b —— 稳健性附录：爆炸性百分比（锐利谱滤波）的非对数口径复核。

背景：step36/37b 中"锐利谱滤波 → 诊断误差 −3067.5% / −13300.8%"来自**对数空间 RMSE**，
Ri_dry 在近零奇异点会被推至任意大。本脚本用非对数口径重算同一组算例，
证明"锐利核有害"是稳健的物理结论，而非 log 度量的伪影。

口径（三种，全部复用 step36 的 diag 与算子）：
  1. log RMSE（原口径，基线对照）
  2. 绝对 Ri 误差中位数（对奇异不敏感）
  3. clip 相对误差：Ri 先 clip 到 [0.01, 100]，再算相对误差 RMSE
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import step12_enoi as S12
import step36_operator_sufficient_condition as S36

OUT = "data/step38b_robustness.json"
KEYS = tuple(S12.KEYS) + ("z",)
CRUISE = (200.0, 400.0)
CLIP = (0.01, 100.0)


def _log_rmse(dg_a, dg_nat, mask):
    a = np.log10(np.clip(np.asarray(dg_a, float), 1e-14, None))
    t = np.log10(np.clip(np.asarray(dg_nat, float), 1e-14, None))
    e = a - t
    m = mask & np.isfinite(e)
    return float(np.sqrt(np.nanmean(e[m] ** 2))) if m.sum() > 3 else np.nan


def _abs_median(dg_a, dg_nat, mask):
    a = np.asarray(dg_a, float)
    t = np.asarray(dg_nat, float)
    e = np.abs(a - t)
    m = mask & np.isfinite(e)
    return float(np.nanmedian(e[m])) if m.sum() > 3 else np.nan


def _clip_rel_rmse(dg_a, dg_nat, mask):
    a = np.clip(np.asarray(dg_a, float), CLIP[0], CLIP[1])
    t = np.clip(np.asarray(dg_nat, float), CLIP[0], CLIP[1])
    e = a - t
    m = mask & np.isfinite(e)
    s = float(np.nanstd(t[m]))
    if s <= 0:
        return np.nan
    return float(np.sqrt(np.nanmean(e[m] ** 2))) / s


def main():
    t0 = time.time()
    print("=" * 98)
    print("step38b  稳健性附录：锐利谱滤波的非对数口径复核")
    print("=" * 98)

    rows = []
    for tgt in S12.TARGETS:
        X = S12.load_case(tgt)
        lat, lon, lev = X["lat"], X["lon"], X["lev"]
        above = X["above"]
        cruise = ((lev[:, None, None] >= CRUISE[0])
                  & (lev[:, None, None] <= CRUISE[1])) & above
        truth = X["nat"]
        bg = X["bg"]
        dlat = float(np.mean(np.diff(lat))) * 111195.0
        g_h = max(float(X["dg"].get("L_h1", 45e3)) / dlat, 0.5)
        k_cut = np.sqrt(np.log(2.0)) / (2 * np.pi * g_h)

        dg_nat = S36.diag(truth, lat, lon, lev)["Ri_dry"]
        dg_bg = S36.diag(bg, lat, lon, lev)["Ri_dry"]

        # 高斯（无振铃）与锐利（有振铃）两种算子
        res = {}
        for op, kern in (("hgauss", S36.gaussian_kernel_1d(g_h)),
                         ("hsharp", S36.sinc_kernel_1d(k_cut))):
            fo = {v: S36.apply_horizontal(bg[v], kern) for v in KEYS}
            fo["z"] = truth["z"]
            dg_op = S36.diag(fo, lat, lon, lev)["Ri_dry"]
            res[op] = dict(
                log_rmse=_log_rmse(dg_op, dg_nat, cruise),
                abs_median=_abs_median(dg_op, dg_nat, cruise),
                clip_rel=_clip_rel_rmse(dg_op, dg_nat, cruise),
            )
            del fo, dg_op
            gc.collect()

        row = dict(target="%s%sz" % (tgt[0], tgt[1]), **res)
        rows.append(row)
        print("  %s%sz: 锐利 logRMSE=%.3f absMed=%.3f clipRel=%.3f"
              % (tgt[0], tgt[1], res["hsharp"]["log_rmse"],
                 res["hsharp"]["abs_median"], res["hsharp"]["clip_rel"]))

    # 跨算例：锐利 vs 高斯的比值（>1 即锐利更差）
    def ratio(key):
        vals = []
        for r in rows:
            a = r["hsharp"][key]
            b = r["hgauss"][key]
            if np.isfinite(a) and np.isfinite(b) and b > 0:
                vals.append(a / b)
        return float(np.median(vals)) if vals else np.nan

    print("\n  【跨算例中位：锐利/高斯 误差比值（>1 = 锐利有害）】")
    r_log = ratio("log_rmse")
    r_abs = ratio("abs_median")
    r_clip = ratio("clip_rel")
    print("  log 空间 RMSE      比值 = %.1f×" % r_log)
    print("  绝对 Ri 误差中位数  比值 = %.1f×" % r_abs)
    print("  clip 相对误差      比值 = %.1f×" % r_clip)
    print("  ⇒ 三种口径下锐利均显著劣于高斯：%s"
          % ("是" if (r_log > 1 and r_abs > 1 and r_clip > 1) else "否"))

    out = dict(rows=rows, ratio=dict(log=r_log, abs=r_abs, clip=r_clip),
               robust=bool(r_log > 1 and r_abs > 1 and r_clip > 1))
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n完成，用时 %.0f s → %s" % (time.time() - t0, OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
