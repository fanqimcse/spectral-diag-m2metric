# -*- coding: utf-8 -*-
"""step23 —— 跨诊断泛化：谱—诊断对应原理是否普适于所有导数型诊断？

step21/22 证明了：下游诊断（Ri = N²/S²）的误差由背景误差谱的**小尺度分量**
主导（垂直导数 ∝ k 高通），所以"平滑（消小尺度）> 同化（消大尺度）"。

但 Ri 只是**一个**导数型诊断。命题 P 若要成立（可外推、可发表），必须证明
这**不是湍流诊断的特例，而是所有"以空间导数为输入"的物理诊断的共性**。

本脚本在同一批 OSSE 算例上，对以下**四类**导数型诊断逐一检验同一命题：
  (1) 湍流          Ri = N²/S²                （垂直导数 dT/dz, dV/dz）
  (2) 对流          CAPE / 条件不稳定度       （垂直导数 dθ/dz）
  (3) 风切变        S² = |∂V/∂z|²              （垂直导数 dV/dz）
  (4) 锋生          F = |∇θ|² 或 D·|∇θ|       （水平导数 ∂θ/∂x, ∂θ/∂y）

判据：对每个诊断 D，比较四种场（bg / an / smooth_bg / smooth_an）的
误差 RMSE(D−D*)。若命题 P 普适，则对**所有**诊断都应满足：
    smooth_bg 的改善 ≥ an 的改善（平滑 > 同化），
    且 smooth_an（同化+平滑）≥ 两者。

这与 step20 只对 Ri 一个诊断成立形成对照，是本命题从"特例"到"普适"的
决定性一步。
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import step10_realerror_osse as S10
import step12_enoi as S12
from scipy.ndimage import gaussian_filter
from avwx import risk as RK
from avwx import enoi as EN

OUT = "data/step23_crossdiag.json"
KEYS = S12.KEYS
CRUISE = (200.0, 400.0)


def compute_diagnostics(fld, lat, lon, lev):
    """对一个场计算全部导数型诊断量，返回 dict。

    除 risk.stability_fields 已有的 S²/N²/Ri 外，补充：
      - CAPE 代理（条件不稳定度）：低层 θ 递减率
      - 风切变 S²（即 stability_fields 的 S2，直接取）
      - 锋生 F：|∇_h θ|²（水平位温梯度平方，水平导数）
      - 水平风切变 DEF²（形变，水平导数）
    """
    dg = RK.stability_fields(dict(T=fld["T"], q=fld["q"], u=fld["u"],
                                  v=fld["v"], z=fld["z"]), lat, lon, lev)
    # 位温 θ = T·(p0/p)^κ，用于 CAPE 与锋生
    T = fld["T"]
    p = np.asarray(lev, float)[:, None, None] * 100.0
    p0 = 100000.0
    kappa = 0.286
    theta = T * (p0 / p) ** kappa

    # 垂直位温梯度 dθ/dz（用于 CAPE / 条件不稳定度）
    Z = dg["Z"]
    zspan = np.maximum(Z[2:] - Z[:-2], 1.0)
    zgap = np.maximum(np.diff(Z, axis=0), 1.0)
    dthdz = np.empty_like(theta)
    dthdz[1:-1] = (theta[2:] - theta[:-2]) / zspan
    dthdz[0] = (theta[1] - theta[0]) / zgap[0]
    dthdz[-1] = (theta[-1] - theta[-2]) / zgap[-1]

    # 水平位温梯度（用于锋生）
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dy = np.gradient(LAT, axis=0) * 111195.0
    dx = np.gradient(LON, axis=1) * 111195.0 * np.cos(np.deg2rad(LAT))
    dthdx = np.gradient(theta, axis=2) / np.where(dx == 0, 1e9, dx)
    dthdy = np.gradient(theta, axis=1) / np.where(dy == 0, 1e9, dy)
    fronto = dthdx ** 2 + dthdy ** 2          # |∇_h θ|²

    out = dict(
        S2=dg["S2"],                          # 垂直风切变（垂直导数）
        Nd2=dg["Nd2"],                        # 浮力频率（垂直导数）
        Ri_dry=dg["Ri_dry"],                  # 湍流（垂直导数之比）
        dthdz=dthdz,                          # 垂直位温梯度（CAPE 输入）
        fronto=fronto,                        # 锋生（水平导数）
        TI=dg["TI"],                          # Ellrod TI（含水平形变）
    )
    return out


def main():
    t0 = time.time()
    print("=" * 98)
    print("step23  跨诊断泛化：谱—诊断对应原理是否普适？")
    print("=" * 98)

    cases = {}
    for t in S12.TARGETS:
        try:
            cases[t] = S12.load_case(t)
        except Exception as e:
            print("载入 %s 失败: %s" % (t, str(e)[:90]))

    rows = []
    for target in list(cases):
        X = cases[target]
        lat, lon, lev = X["lat"], X["lon"], X["lev"]
        others = [t for t in cases if t != target]
        print("\n目标 %s%sz" % (target[0], target[1]), flush=True)

        def pair_for(ts, lead):
            nat = S10.prepare(S10.AG.load(S10.native_path(ts[0], ts[1], 0),
                                          bbox=S10.BBOX))
            b = S10.bg_cycle(ts[0], ts[1], lead)
            p = S10.native_path(b[0], b[1], lead)
            if not os.path.exists(p):
                return None
            bg = S10.prepare(S10.AG.load(p, bbox=S10.BBOX))
            out = ({v: nat["f"][v] for v in ("T", "q", "u", "v")},
                   {v: bg["f"][v] for v in ("T", "q", "u", "v")})
            del nat, bg
            gc.collect()
            return out

        base = S12.member_pairs(others)
        same_day = [q for q in (pair_for(target, ld) for ld in (6, 12))
                    if q is not None]
        pairs = base + same_day
        if len(pairs) < 3:
            continue
        sb = S12.sigma_from_pairs(pairs)
        members, _ = EN.build_members(pairs, sb, clip={"q": 0.02})
        eobs = S12.to_enoi_obs(X["obs"], use=("sfc", "raob", "ac"),
                               lat=lat, lon=lon, lev=lev)
        E = EN.EnOI(members, lat, lon, lev, sigma_b=sb,
                    tiles=(6, 5), tile_pad_km=350.0, max_obs=1000)
        an = E.update(X["bg"], eobs, S12.OERR)
        an["z"] = X["bg"]["z"]
        del E, members, pairs
        gc.collect()

        dlat = float(np.mean(np.diff(lat))) * 111195.0
        g = max(float(X["dg"].get("L_h1", 124e3)) / dlat, 0.5)
        smbg = {v: gaussian_filter(X["bg"][v], sigma=(0, g, g),
                                   mode="nearest") for v in KEYS}
        smbg["z"] = X["bg"]["z"]
        sman = {v: gaussian_filter(an[v], sigma=(0, g, g), mode="nearest")
                for v in KEYS}
        sman["z"] = X["bg"]["z"]

        above = X["above"]
        cruise = ((lev[:, None, None] >= CRUISE[0])
                  & (lev[:, None, None] <= CRUISE[1])) & above

        # 计算真值与各场的诊断量
        dg_nat = compute_diagnostics(X["nat"], lat, lon, lev)
        dg_bg = compute_diagnostics(X["bg"], lat, lon, lev)
        dg_an = compute_diagnostics(an, lat, lon, lev)
        dg_sbg = compute_diagnostics(smbg, lat, lon, lev)
        dg_san = compute_diagnostics(sman, lat, lon, lev)

        # 评估域：巡航带（与 step20 一致）
        row = dict(target="%s%sz" % (target[0], target[1]))
        print("  %-10s %10s %10s %10s %10s | %10s %10s"
              % ("诊断量", "RMSE(bg)", "an降幅%", "平滑bg%", "平滑an%",
                 "an−平滑bg", "平滑an−max"))
        for key in ("S2", "Nd2", "Ri_dry", "dthdz", "fronto", "TI"):
            # 对数空间（这些量跨度大，且 Ri/切变本身是正的偏态分布）
            def lg(a):
                return np.log10(np.clip(np.asarray(a, float), 1e-14, None))
            eb = lg(dg_bg[key]) - lg(dg_nat[key])
            ea = lg(dg_an[key]) - lg(dg_nat[key])
            esbg = lg(dg_sbg[key]) - lg(dg_nat[key])
            esan = lg(dg_san[key]) - lg(dg_nat[key])
            m = cruise & np.isfinite(eb) & np.isfinite(ea) & np.isfinite(esbg) \
                & np.isfinite(esan)
            if m.sum() < 200:
                continue
            rb = float(np.sqrt(np.nanmean(eb[m] ** 2)))
            ra = float(np.sqrt(np.nanmean(ea[m] ** 2)))
            rsbg = float(np.sqrt(np.nanmean(esbg[m] ** 2)))
            rsan = float(np.sqrt(np.nanmean(esan[m] ** 2)))
            d_an = 100 * (rb - ra) / rb
            d_sbg = 100 * (rb - rsbg) / rb
            d_san = 100 * (rb - rsan) / rb
            row["%s|an" % key] = d_an
            row["%s|sbg" % key] = d_sbg
            row["%s|san" % key] = d_san
            print("  %-10s %10.4f %10.2f %10.2f %10.2f | %10.2f %10.2f"
                  % (key, rb, d_an, d_sbg, d_san, d_an - d_sbg,
                     d_san - max(d_an, d_sbg)))
        rows.append(row)

        del X, an, smbg, sman
        del dg_nat, dg_bg, dg_an, dg_sbg, dg_san
        gc.collect()

    # 汇总
    print("\n" + "=" * 98)
    print("【汇总】跨 %d 算例取中位：各诊断的 an / 平滑bg / 平滑an 降幅%%" % len(rows))
    print("=" * 98)
    def med(k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    print("%-10s %10s %10s %10s | %12s" %
          ("诊断量", "an", "平滑bg", "平滑an", "平滑bg>an?"))
    summary = {}
    for key in ("S2", "Nd2", "Ri_dry", "dthdz", "fronto", "TI"):
        a = med("%s|an" % key)
        s = med("%s|sbg" % key)
        san = med("%s|san" % key)
        if not np.isfinite(a):
            continue
        summary[key] = dict(an=a, sbg=s, san=san, sbg_gt_an=bool(s > a),
                            san_gt_max=bool(san > max(a, s)))
        print("%-10s %10.2f %10.2f %10.2f | %12s"
              % (key, a, s, san, "是" if s > a else "否"))

    n_sbg_gt = sum(1 for v in summary.values() if v["sbg_gt_an"])
    n_total = len(summary)
    print("\n⇒ 在 %d 个导数型诊断中，有 %d 个满足'平滑 > 同化'（%.0f%%）"
          % (n_total, n_sbg_gt, 100 * n_sbg_gt / max(n_total, 1)))
    print("⇒ 命题 P 的普适性：%s" %
          ("成立（平滑对所有导数型诊断都优于同化）" if n_sbg_gt == n_total
           else "部分成立（见上表，需讨论例外）"))

    out = dict(summary=summary, rows=rows, n_sbg_gt=n_sbg_gt, n_total=n_total)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
