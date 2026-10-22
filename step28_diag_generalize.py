# -*- coding: utf-8 -*-
"""step28 —— 泛化到对流降水 / 锋面降水 / 急流核：命题 P 的普适边界。

背景
----
step23 已证明谱—诊断对应原理在"垂直导数型"诊断（S²/N²/Ri/dθdz）上成立
（平滑>同化），在"水平导数型"诊断（锋生 |∇θ|²、Ellrod TI）上反转。
但那批诊断大多是**简化代理量**，且没有覆盖三类对航空/中尺度气象最
重要的真实诊断：

  (1) 对流降水 —— 需要接近 CAPE 的量（不只是递减率代理）
  (2) 锋面降水 —— 需要含形变的锋生函数 F = D|∇θ|（不只是 |∇θ|²）
  (3) 急流核   —— 需要涡度/散度（不只是垂直风切变 S²）

本脚本补上这三类**物理上更真实**的诊断，检验命题 P 的普适边界：

  命题：对任何"以空间导数为输入"的诊断，其误差由背景误差谱的小尺度
  分量主导（垂直导数 ∝ k 高通）。但"导数方向"决定尺度分离的方向：
  · 垂直导数诊断 → 水平平滑有益（消水平小尺度噪声）
  · 水平导数诊断 → 水平平滑有害（抹掉水平梯度信号）

预期（可证伪）：
  · 对流 CAPE/CIN（垂直导数 dθ/dz）→ 平滑>同化（与 step23 的 dthdz 一致）
  · 锋生函数 F=D|∇θ|（水平导数+形变）→ 同化>平滑（反转，与 fronto 一致）
  · 急流核涡度 ζ（水平导数 ∂v/∂x-∂u/∂y）→ 同化>平滑（反转）
  · 急流核散度 D（水平导数）→ 同化>平滑（反转，且对平滑最敏感）

判据与 step23 一致：比较 bg/an/smooth_bg/smooth_an 四场的诊断 RMSE 降幅。
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

OUT = "data/step28_diag_generalize.json"
KEYS = S12.KEYS
CRUISE = (200.0, 400.0)
# 对流诊断在低层（850-500），急流核在高层（200-300），需分别评估域
TROPO_LOW = (700.0, 1000.0)
JET = (200.0, 300.0)


def horizontal_derivatives(field, lat, lon, lev):
    """计算水平导数 ∂/∂x、∂/∂y（对三维场的每一层）。"""
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dy = np.gradient(LAT, axis=0) * 111195.0
    dx = np.gradient(LON, axis=1) * 111195.0 * np.cos(np.deg2rad(LAT))
    dfdx = np.gradient(field, axis=2) / np.where(dx == 0, 1e9, dx)
    dfdy = np.gradient(field, axis=1) / np.where(dy == 0, 1e9, dy)
    return dfdx, dfdy


def compute_real_diagnostics(fld, lat, lon, lev):
    """计算三类真实物理诊断量。

    返回 dict：
      cape_proxy  —— 对流有效位能的解析近似（真正积分，而非递减率代理）
      cin_proxy   —— 对流抑制（负浮力积分）
      fronto_fn   —— 锋生函数 F = |∇θ| · (D cos2β + ...) 的简化：D·|∇θ|
                     （形变 × 水平位温梯度，即 Petterssen 锋生的主项）
      vor         —— 相对涡度 ζ = ∂v/∂x − ∂u/∂y
      div         —— 水平散度 D = ∂u/∂x + ∂v/∂y
    """
    T = fld["T"]; q = fld["q"]; u = fld["u"]; v = fld["v"]
    p = np.asarray(lev, float)[:, None, None] * 100.0
    p0 = 100000.0
    kappa = 0.286
    theta = T * (p0 / p) ** kappa
    theta_e = theta * np.exp(2.5e6 * q / (1004.0 * T))   # 相当位温（近似）

    # 垂直导数（复用 risk 的 height_coordinate）
    Z = RK.height_coordinate(lev, TH_virtual(T, q), fld.get("z"))
    zspan = np.maximum(Z[2:] - Z[:-2], 1.0)
    zgap = np.maximum(np.diff(Z, axis=0), 1.0)
    dthdz = np.empty_like(theta)
    dthdz[1:-1] = (theta[2:] - theta[:-2]) / zspan
    dthdz[0] = (theta[1] - theta[0]) / zgap[0]
    dthdz[-1] = (theta[-1] - theta[-2]) / zgap[-1]

    # ---- 对流 CAPE 解析近似（对每列，从抬升凝结层向上积分正浮力）----
    # 简化：用"整层条件不稳定度"的积分作为 CAPE 代理
    # CAPE ∝ ∫ (θ_parcel − θ_env)/θ_env · g dz，取 θ_e 与 θ 之差的正部积分
    nz, ny, nx = theta.shape
    # 抬升气块的 θ 用最低层 θ_e 近似（干对流下守恒）
    theta_pcl = theta_e[0][None, :, :]  # (1, ny, nx) 广播
    buoy = (theta_pcl - theta_e) / np.maximum(theta_e, 1.0)  # 正浮力
    # 只积分 buoy>0 的部分（CAPE），buoy<0 的部分（CIN）
    g = 9.81
    dz_all = np.concatenate([zgap[:1], zgap], axis=0)  # 每层厚度近似
    dz_all = np.maximum(dz_all, 1.0)
    cape = np.sum(np.maximum(buoy, 0) * g * dz_all, axis=0)   # (ny,nx)
    cin = np.sum(np.minimum(buoy, 0) * g * dz_all, axis=0)    # (ny,nx, 负值)
    cin = -cin  # CIN 取正（抑制量）

    # ---- 锋生函数 F = D·|∇θ|（Petterssen 锋生主项）----
    dthdx, dthdy = horizontal_derivatives(theta, lat, lon, lev)
    dudx, dudy = horizontal_derivatives(u, lat, lon, lev)
    dvdx, dvdy = horizontal_derivatives(v, lat, lon, lev)
    mag_grad_th = np.sqrt(dthdx ** 2 + dthdy ** 2)
    # 形变 D = sqrt((∂u/∂x-∂v/∂y)² + (∂v/∂x+∂u/∂y)²)
    stretch = dudx - dvdy
    shear = dvdx + dudy
    deform = np.sqrt(stretch ** 2 + shear ** 2)
    fronto_fn = deform * mag_grad_th   # F = D·|∇θ|（锋生函数）

    # ---- 涡度 / 散度（急流核）----
    vor = dvdx - dudy                     # 相对涡度 ζ
    div = dudx + dvdy                     # 水平散度 D

    return dict(cape_proxy=cape, cin_proxy=cin, fronto_fn=fronto_fn,
                vor=vor, div=div, dthdz=dthdz)


def TH_virtual(T, q):
    return T * (1.0 + 0.608 * q)


def main():
    t0 = time.time()
    print("=" * 98)
    print("step28  泛化到对流降水/锋面降水/急流核：命题 P 的普适边界")
    print("=" * 98)

    cases = {}
    for t in S12.TARGETS:
        try:
            cases[t] = S12.load_case(t)
            print("载入 %s%sz" % (t[0], t[1]), flush=True)
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
        # 各诊断的评估域
        dom = {
            "cape_proxy": (TROPO_LOW, "对流低层"),
            "cin_proxy": (TROPO_LOW, "对流低层"),
            "fronto_fn": ((500.0, 1000.0), "锋面低层"),
            "vor": (JET, "急流核"),
            "div": (JET, "急流核"),
            "dthdz": (TROPO_LOW, "对流低层"),
        }

        dg_nat = compute_real_diagnostics(X["nat"], lat, lon, lev)
        dg_bg = compute_real_diagnostics(X["bg"], lat, lon, lev)
        dg_an = compute_real_diagnostics(an, lat, lon, lev)
        dg_sbg = compute_real_diagnostics(smbg, lat, lon, lev)
        dg_san = compute_real_diagnostics(sman, lat, lon, lev)

        row = dict(target="%s%sz" % (target[0], target[1]))
        print("  %-12s %10s %10s %10s %10s | %8s" %
              ("诊断", "RMSE(bg)", "an%", "平滑bg%", "平滑an%", "平滑>同化?"))
        for key, ((p_lo, p_hi), label) in dom.items():
            # 评估域掩码
            m = (lev[:, None, None] >= p_lo) & (lev[:, None, None] <= p_hi) \
                & above
            if key in ("cape_proxy", "cin_proxy"):  # 二维量
                m = m[0] if m.ndim == 3 else m
                # 对二维量，掩码取任意层在地面以上
                m2d = above[0]  # 最低层掩码
            # 用对数空间比较（诊断量偏态）
            def lg(a):
                return np.log10(np.clip(np.asarray(a, float), 1e-14, None))
            a_nat = lg(dg_nat[key])
            a_bg = lg(dg_bg[key])
            a_an = lg(dg_an[key])
            a_sbg = lg(dg_sbg[key])
            a_san = lg(dg_san[key])
            if key in ("cape_proxy", "cin_proxy"):
                mm = m2d & np.isfinite(a_bg) & np.isfinite(a_an)
            else:
                mm = m & np.isfinite(a_bg) & np.isfinite(a_an)
            if mm.sum() < 50:
                continue
            rb = float(np.sqrt(np.nanmean((a_bg - a_nat)[mm] ** 2)))
            ra = float(np.sqrt(np.nanmean((a_an - a_nat)[mm] ** 2)))
            rsbg = float(np.sqrt(np.nanmean((a_sbg - a_nat)[mm] ** 2)))
            rsan = float(np.sqrt(np.nanmean((a_san - a_nat)[mm] ** 2)))
            d_an = 100 * (rb - ra) / rb
            d_sbg = 100 * (rb - rsbg) / rb
            d_san = 100 * (rb - rsan) / rb
            row["%s|an" % key] = d_an
            row["%s|sbg" % key] = d_sbg
            row["%s|san" % key] = d_san
            print("  %-12s %10.4f %10.2f %10.2f %10.2f | %8s"
                  % (key, rb, d_an, d_sbg, d_san, "是" if d_sbg > d_an else "否"))
        rows.append(row)

        del X, an, smbg, sman
        del dg_nat, dg_bg, dg_an, dg_sbg, dg_san
        gc.collect()

    # 汇总
    def med(k):
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    print("\n" + "=" * 98)
    print("【汇总】跨 %d 算例中位：各诊断 an/平滑bg/平滑an 降幅%%" % len(rows))
    print("=" * 98)
    print("%-12s %8s %8s %8s | %8s" %
          ("诊断", "an", "平滑bg", "平滑an", "平滑>同化?"))
    summary = {}
    for key in ("cape_proxy", "cin_proxy", "fronto_fn", "vor", "div", "dthdz"):
        a = med("%s|an" % key)
        s = med("%s|sbg" % key)
        san = med("%s|san" % key)
        if not np.isfinite(a):
            continue
        summary[key] = dict(an=a, sbg=s, san=san, sbg_gt_an=bool(s > a))
        print("%-12s %8.2f %8.2f %8.2f | %8s"
              % (key, a, s, san, "是" if s > a else "否"))

    out = dict(summary=summary, rows=rows)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
