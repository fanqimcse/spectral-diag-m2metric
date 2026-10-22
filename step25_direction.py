# -*- coding: utf-8 -*-
"""step25 —— 尺度分离的"方向匹配"理论（命题 P 的精细化）。

step23 发现：水平平滑对**垂直导数**诊断（S²/N²/Ri/dθ/dz）有益，
但对**水平导数**诊断（fronto/TI）有害。本脚本检验一个更精细的命题：

    命题 P*（方向匹配原理）：
    尺度分离的收益方向必须**匹配**下游诊断的导数方向。
      · 垂直导数诊断 D(d/dz)：应做**水平**平滑（消水平小尺度噪声，
        因为垂直差分把水平小尺度噪声放大了）。
      · 水平导数诊断 D(d/dx, d/dy)：应做**垂直**平滑（消垂直小尺度噪声），
        而水平平滑会直接抹掉它依赖的水平梯度信号，故有害。

机制：离散导数算子只在**它作用的方向**上是高通。垂直导数 ∂/∂z 放大
垂直波数 k_z，对水平波数 k_h 无高通响应；但**水平平滑**只滤 k_h，
因此它对"由垂直差分放大的水平小尺度噪声"是有效的——这正是 step21
观测到"同化（光滑增量）改善有限、平滑改善显著"的根源。

反之，水平导数 ∂/∂x 放大水平波数 k_h，而**水平平滑**恰恰滤掉 k_h 的
真实信号（而不只是噪声），所以有害；此时正确的尺度分离是**垂直平滑**
（滤 k_z，不动 k_h）。

本脚本在同一批算例上，对水平导数诊断（fronto/TI）改用**垂直平滑**，
检验是否恢复收益、甚至超过水平平滑时的表现。
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

OUT = "data/step25_direction.json"
KEYS = S12.KEYS
CRUISE = (200.0, 400.0)


def compute_diagnostics(fld, lat, lon, lev):
    dg = RK.stability_fields(dict(T=fld["T"], q=fld["q"], u=fld["u"],
                                  v=fld["v"], z=fld["z"]), lat, lon, lev)
    T = fld["T"]
    p = np.asarray(lev, float)[:, None, None] * 100.0
    p0 = 100000.0
    kappa = 0.286
    theta = T * (p0 / p) ** kappa
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dy = np.gradient(LAT, axis=0) * 111195.0
    dx = np.gradient(LON, axis=1) * 111195.0 * np.cos(np.deg2rad(LAT))
    dthdx = np.gradient(theta, axis=2) / np.where(dx == 0, 1e9, dx)
    dthdy = np.gradient(theta, axis=1) / np.where(dy == 0, 1e9, dy)
    fronto = dthdx ** 2 + dthdy ** 2
    return dict(S2=dg["S2"], Nd2=dg["Nd2"], Ri_dry=dg["Ri_dry"],
                fronto=fronto, TI=dg["TI"])


def smooth_h(fld, sigma_h):
    out = {v: gaussian_filter(fld[v], sigma=(0, sigma_h, sigma_h),
                              mode="nearest") for v in KEYS}
    out["z"] = fld["z"]
    return out


def smooth_v(fld, sigma_v):
    out = {v: gaussian_filter(fld[v], sigma=(sigma_v, 0, 0),
                              mode="nearest") for v in KEYS}
    out["z"] = fld["z"]
    return out


def main():
    t0 = time.time()
    print("=" * 98)
    print("step25  尺度分离的'方向匹配'：水平导数诊断需要垂直平滑")
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
        g_h = max(float(X["dg"].get("L_h1", 45e3)) / dlat, 0.5)
        g_v = 2.0                          # 垂直平滑：跨 2 层

        sm_h = smooth_h(X["bg"], g_h)      # 水平平滑（step20/23 用的）
        sm_v = smooth_v(X["bg"], g_v)      # 垂直平滑（本脚本新增）
        sm_hv = smooth_v(sm_h, g_v)        # 先水平后垂直

        above = X["above"]
        cruise = ((lev[:, None, None] >= CRUISE[0])
                  & (lev[:, None, None] <= CRUISE[1])) & above

        dg_nat = compute_diagnostics(X["nat"], lat, lon, lev)
        dg_bg = compute_diagnostics(X["bg"], lat, lon, lev)
        dg_h = compute_diagnostics(sm_h, lat, lon, lev)
        dg_v = compute_diagnostics(sm_v, lat, lon, lev)
        dg_hv = compute_diagnostics(sm_hv, lat, lon, lev)

        row = dict(target="%s%sz" % (target[0], target[1]))
        print("  %-10s %10s %10s %10s %10s" %
              ("诊断量", "水平平滑%", "垂直平滑%", "水平+垂直%", "最佳"))
        for key in ("S2", "Nd2", "Ri_dry", "fronto", "TI"):
            def lg(a):
                return np.log10(np.clip(np.asarray(a, float), 1e-14, None))
            eb = lg(dg_bg[key]) - lg(dg_nat[key])
            eh = lg(dg_h[key]) - lg(dg_nat[key])
            ev = lg(dg_v[key]) - lg(dg_nat[key])
            ehv = lg(dg_hv[key]) - lg(dg_nat[key])
            m = cruise & np.isfinite(eb) & np.isfinite(eh) & np.isfinite(ev) \
                & np.isfinite(ehv)
            if m.sum() < 200:
                continue
            rb = float(np.sqrt(np.nanmean(eb[m] ** 2)))
            rh = float(np.sqrt(np.nanmean(eh[m] ** 2)))
            rv = float(np.sqrt(np.nanmean(ev[m] ** 2)))
            rhv = float(np.sqrt(np.nanmean(ehv[m] ** 2)))
            d_h = 100 * (rb - rh) / rb
            d_v = 100 * (rb - rv) / rb
            d_hv = 100 * (rb - rhv) / rb
            row["%s|h" % key] = d_h
            row["%s|v" % key] = d_v
            row["%s|hv" % key] = d_hv
            best = max(d_h, d_v, d_hv)
            print("  %-10s %10.2f %10.2f %10.2f %10s"
                  % (key, d_h, d_v, d_hv,
                     "水平" if best == d_h else ("垂直" if best == d_v else "水平+垂直")))
        rows.append(row)

        del X, an, sm_h, sm_v, sm_hv
        del dg_nat, dg_bg, dg_h, dg_v, dg_hv
        gc.collect()

    print("\n" + "=" * 98)
    print("【汇总】跨 %d 算例取中位" % len(rows))
    print("=" * 98)
    def med(k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    print("%-10s %10s %10s %10s | %14s" %
          ("诊断量", "水平平滑", "垂直平滑", "水平+垂直", "方向匹配结论"))
    summary = {}
    for key in ("S2", "Nd2", "Ri_dry", "fronto", "TI"):
        h = med("%s|h" % key)
        v = med("%s|v" % key)
        hv = med("%s|hv" % key)
        if not np.isfinite(h):
            continue
        summary[key] = dict(h=h, v=v, hv=hv)
        # 垂直导数诊断应偏好水平平滑；水平导数诊断应偏好垂直平滑
        print("%-10s %10.2f %10.2f %10.2f | %s"
              % (key, h, v, hv,
                 "水平平滑胜" if h >= v else "垂直平滑胜"))

    out = dict(summary=summary, rows=rows)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
