# -*- coding: utf-8 -*-
"""step20b —— 为什么"场变好了、风险产品却变差了"：机理定位。

step20 的反常结果
-----------------
同一个 EnOI 分析场：
  * 原始场 RMSE 相对背景降 6–8.5%（step12，可复算）
  * 但**标定后的航迹风险概率**反而更差：全网格 RMSE(P−P*) **+13%**、
    飞机影响区 **+18%**（负增益）
  * 而沿单条航线（250 hPa）却**改善 4.9%**

一个场不可能同时又好又坏。要查的是：**增量到底改坏了 P 依赖的哪个量**。

候选机理（按可检验性排序）
--------------------------
M1 **原始场在这里其实没改善** —— 先核对，若成立则后面全部作废。
M2 **导数被放大**：P 依赖 Ri = N²/S²，两者都是**垂直导数**。
   增量能把 T/q/u/v 本身的 RMSE 降下来，却可能不降（甚至抬高）
   dθ/dz 与 dV/dz 的误差 —— 尤其当增量带有小垂直尺度噪声时。
M3 **局地化接缝**：EnOI 按 tile 独立求解 + 余弦过渡，tile 边界上的
   不连续会在**水平**梯度上留下痕迹；若 S² 含水平切变分量就会被放大。
M4 **水汽被搞坏**：q 的 σ_b 跨层差 6263×（step16c 实测），
   单标量缩放可能把上对流层的 q 增量放大到失真 ⇒ RH 与 N²_v 全错，
   而巡航层产品恰恰最依赖这两个量。

本脚本逐个量：原始场 RMSE、**逐层** RMSE 廓线、导数场误差、
以及 q 增量的相对幅度。不预设结论。
"""
import gc
import json
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
import step10_realerror_osse as S10
import step12_enoi as S12

from avwx import calib as CB
from avwx import enoi as EN
from avwx import risk as RK

OUT = "data/step20b_mechanism.json"
TARGET = ("20261001", "00")
KEYS = ("T", "q", "u", "v")

CAL = CB.load()


def diag_of(fld, lat, lon, lev):
    return RK.stability_fields(dict(T=fld["T"], q=fld["q"], u=fld["u"],
                                    v=fld["v"], z=fld["z"]),
                               lat, lon, lev)


def rmse(a, b, m=None):
    d = np.asarray(a, float) - np.asarray(b, float)
    if m is not None:
        d = d[m]
    return float(np.sqrt(np.nanmean(d ** 2)))


def main():
    t0 = time.time()
    print("=" * 96)
    print("step20b  机理定位：目标 %s%sz" % (TARGET[0], TARGET[1]))
    print("=" * 96)
    X = S12.load_case(TARGET)
    lat, lon, lev = X["lat"], X["lon"], X["lev"]
    others = [t for t in S12.TARGETS if t != TARGET]

    pairs = S12.member_pairs(others)
    sb = S12.sigma_from_pairs(pairs)
    print("集合成员 %d 个；σ_b(标量) = %s"
          % (len(pairs), "  ".join("%s=%.3g" % (v, sb[v]) for v in KEYS)),
          flush=True)
    members, _ = EN.build_members(pairs, sb, clip={"q": 0.02})
    eobs = S12.to_enoi_obs(X["obs"], use=("sfc", "raob", "ac"),
                           lat=lat, lon=lon, lev=lev)
    E = EN.EnOI(members, lat, lon, lev, sigma_b=sb,
                tiles=(6, 5), tile_pad_km=350.0, max_obs=1000)
    an = E.update(X["bg"], eobs, S12.OERR)
    an["z"] = X["bg"]["z"]
    del E, members
    gc.collect()

    above = X["above"]
    am = S10.impact_mask(X["obs"]["ac"][1], lat, lon, lev) & above

    # ---------------- M1：原始场到底改善没有 ----------------
    print("\n【M1】原始场 RMSE 相对背景的降幅 [%%]（飞机影响区，n=%d）"
          % int(am.sum()))
    print("%-6s %12s %12s %10s" % ("变量", "RMSE(bg)", "RMSE(an)", "降幅%"))
    m1 = {}
    for v in KEYS:
        rb = rmse(X["bg"][v], X["nat"][v], am)
        ra = rmse(an[v], X["nat"][v], am)
        m1[v] = 100 * (rb - ra) / max(rb, 1e-30)
        print("%-6s %12.5g %12.5g %10.2f" % (v, rb, ra, m1[v]))
    verdict_m1 = all(m1[v] > 0 for v in KEYS)
    print("⇒ M1 %s" % ("不成立（原始场确实全面改善，反常结果需要别的解释）"
                       if verdict_m1 else "成立：原始场就没改善，step20 的比较作废"))

    # ---------------- M2/M3：导数场 ----------------
    dgb = diag_of(X["bg"], lat, lon, lev)
    dga = diag_of(an, lat, lon, lev)
    dgn = diag_of(X["nat"], lat, lon, lev)

    def relerr(k, log=True, lo=1e-12):
        a, b, t = dgb[k], dga[k], dgn[k]
        if log:
            a, b, t = np.log10(np.clip(a, lo, None)), \
                np.log10(np.clip(b, lo, None)), np.log10(np.clip(t, lo, None))
        m = above & np.isfinite(a) & np.isfinite(b) & np.isfinite(t)
        rb = rmse(a, t, m)
        ra = rmse(b, t, m)
        return rb, ra, 100 * (rb - ra) / max(rb, 1e-30)

    print("\n【M2】导数/诊断量误差（相对真值）的降幅 [%%]，全网格")
    print("%-12s %12s %12s %10s" % ("量", "RMSE(bg)", "RMSE(an)", "降幅%"))
    m2 = {}
    for k, lg in (("S2", True), ("Nd2", True), ("Nv2", True), ("dTdz", False),
                  ("dqdz", False), ("rh", False), ("TI", True)):
        if k not in dgb:
            continue
        rb, ra, d = relerr(k, lg)
        m2[k] = d
        print("%-12s %12.5g %12.5g %10.2f" % (k, rb, ra, d))
    print("⇒ M2 %s" % ("成立：导数误差没被改善甚至变差"
                       if any(m2.get(k, 0) < 0 for k in ("S2", "Nd2", "dTdz"))
                       else "不成立：导数误差同样被改善"))

    # ---------------- M4：逐层 q / RH ----------------
    print("\n【M4】逐层 RMSE 降廓线 [%%]（正值=改善）")
    print("%-8s %9s %9s %9s %9s | %9s %9s"
          % ("p[hPa]", "T", "q", "u", "v", "RH", "Nv2"))
    prof = {}
    for k, pl in enumerate(lev):
        row = {}
        for v in KEYS:
            m = am[k] if am.ndim == 3 else am
            rb = rmse(X["bg"][v][k], X["nat"][v][k], m)
            ra = rmse(an[v], X["nat"][v], None)          # 占位，下面替换
            ra = rmse(an[v][k], X["nat"][v][k], m)
            row[v] = 100 * (rb - ra) / max(rb, 1e-30)
        mr = above[k] & np.isfinite(dgb["rh"][k]) & np.isfinite(dgn["rh"][k])
        rb = rmse(dgb["rh"][k], dgn["rh"][k], mr)
        ra = rmse(dga["rh"][k], dgn["rh"][k], mr)
        row["RH"] = 100 * (rb - ra) / max(rb, 1e-30)
        mn = above[k] & np.isfinite(dgb["Nv2"][k]) & np.isfinite(dgn["Nv2"][k])
        rb = rmse(dgb["Nv2"][k], dgn["Nv2"][k], mn)
        ra = rmse(dga["Nv2"][k], dgn["Nv2"][k], mn)
        row["Nv2"] = 100 * (rb - ra) / max(rb, 1e-30)
        prof[str(int(pl))] = row
        print("%-8d %9.2f %9.2f %9.2f %9.2f | %9.2f %9.2f"
              % (pl, row["T"], row["q"], row["u"], row["v"],
                 row["RH"], row["Nv2"]))

    # q 增量的**相对**幅度：|Δq| / q_bg，按层看是否上对流层失控
    print("\n【M4′】q 增量相对幅度 |Δq|/q_bg 的中位（按层，%）")
    print("%-8s %12s %12s %12s" % ("p[hPa]", "q_bg 中位", "|Δq| 中位", "相对%"))
    rel = {}
    for k, pl in enumerate(lev):
        qb = X["bg"]["q"][k][above[k]]
        dq = (an["q"][k] - X["bg"]["q"][k])[above[k]]
        if qb.size < 100:
            continue
        qm = float(np.median(qb))
        dm = float(np.median(np.abs(dq)))
        rel[str(int(pl))] = 100 * dm / max(qm, 1e-12)
        print("%-8d %12.3g %12.3g %12.1f" % (pl, qm, dm, rel[str(int(pl))]))
    neg = float(np.mean(an["q"][above] < 0))
    print("分析场 q<0 的格点占比 %.4f%%（背景 %.4f%%）"
          % (100 * neg, 100 * float(np.mean(X["bg"]["q"][above] < 0))))

    out = dict(target="%s%sz" % (TARGET[0], TARGET[1]), M1_field=m1,
               M1_all_improve=bool(verdict_m1), M2_deriv=m2,
               M4_profile=prof, M4_q_rel=rel, q_neg_frac=neg)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
