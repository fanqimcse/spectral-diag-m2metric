# -*- coding: utf-8 -*-
"""step30b —— P1（决定性版）：诊断误差的水平谱是否真的比场误差更"蓝"。

step30 用"水平/垂直谱斜率"直接对比，但垂直方向只有 21 层（巡航带仅 5-6 层），
垂直谱斜率被边界效应污染，不是正确的检验量。

本脚本改用**命题 P 的真正闭环**来检验，分三步：

  [A] 谱转移验证（核心）：对真实 GFS 误差场，比较
        · 场误差 ε 的水平功率谱 E_ε(k_h)
        · 垂直导数诊断误差（如 N² 误差、dT/dlnp 误差）的水平功率谱 E_diag(k_h)
      若导数算子真是高通，则 E_diag(k_h) 应比 E_ε(k_h) 更"蓝"（谱权向高波数
      移动）。用"谱质心波数" k_centroid = ∫k·E dk / ∫E dk 量化：
      若 k_centroid(diag) > k_centroid(field)，则导数放大高波数在真实大气成立。

  [B] 水平平滑的谱作用：水平平滑后，场误差谱与诊断误差谱各下降多少？
      检验"水平平滑移除的高波数场误差"是否正好命中"诊断误差里被放大的部分"，
      即水平平滑对诊断误差的改善 > 对场误差的改善。

  [C] 显式假设 H_vh 的陈述与判定：
      H_vh：真实大气的背景误差，其水平小尺度分量通过垂直导数算子
      耦合进下游诊断，因此"水平尺度分离"能降低垂直导数诊断误差。
      若 [A] 的 k_centroid 转移显著、且 [B] 的"诊断改善 > 场改善"成立，
      则 H_vh 成立，命题 P 的水平谱 M2 代理是合法的。

自包含：复用 8 个配对的 GFS 数据（与 step30 相同）。
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import step10_realerror_osse as S10
from scipy.ndimage import gaussian_filter

OUT = "data/step30b_diag_spectrum.json"
KEYS = ("T", "q", "u", "v")
CRUISE = (200.0, 400.0)

PAIRS = [
    ("20220115", "00"), ("20220115", "12"),
    ("20220715", "00"), ("20220715", "12"),
    ("20230115", "00"), ("20230115", "12"),
    ("20230715", "00"), ("20230715", "12"),
]
BG_LEAD = 24


def spectrum2d(field_2d, mask=None):
    """二维水平功率谱（径向平均）。"""
    f = np.asarray(field_2d, float).copy()
    if mask is not None:
        f = np.where(mask, f, np.nan)
    m = np.isfinite(f)
    if not m.all():
        f = np.where(m, f, np.nanmean(f[m]))
    f = f - np.nanmean(f)
    ny, nx = f.shape
    wy = np.hanning(ny)[:, None]
    wx = np.hanning(nx)[None, :]
    f = f * wy * wx
    F = np.fft.fft2(f)
    P = np.abs(F) ** 2
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    kbins = np.arange(0, 0.5, 1.0 / max(ny, nx))
    power = np.zeros(len(kbins) - 1)
    for i in range(len(kbins) - 1):
        sel = (kr >= kbins[i]) & (kr < kbins[i + 1])
        if sel.sum() == 0:
            continue
        power[i] = np.mean(P[sel])
    k_center = 0.5 * (kbins[:-1] + kbins[1:])
    return k_center, power


def centroid(k, power):
    """谱质心波数 k_centroid = ∫k·E dk / ∫E dk。"""
    k = np.asarray(k)
    p = np.asarray(power)
    dk = np.gradient(k)
    return float(np.sum(k * p * dk) / max(np.sum(p * dk), 1e-30))


def vert_deriv(f3d, lev):
    """垂直导数（对数气压坐标）：d/dlnp。返回 (nz, ny, nx)。"""
    lp = np.log(np.asarray(lev, float))
    out = np.empty_like(f3d)
    out[:] = np.nan
    # 中心差分
    for k in range(1, len(lev) - 1):
        out[k] = (f3d[k + 1] - f3d[k - 1]) / (lp[k + 1] - lp[k - 1])
    return out


def main():
    t0 = time.time()
    print("=" * 98)
    print("step30b  P1（决定性版）：诊断误差的水平谱是否比场误差更'蓝'")
    print("=" * 98)

    rows = []
    for (dte, cyc) in PAIRS:
        bspec = S10.bg_cycle(dte, cyc, BG_LEAD)
        nat_path = S10.native_path(dte, cyc, 0)
        bg_path = S10.native_path(bspec[0], bspec[1], BG_LEAD)
        if not (os.path.exists(nat_path) and os.path.exists(bg_path)):
            continue
        nat = S10.prepare(S10.AG.load(nat_path, bbox=S10.BBOX), "nat")
        bg = S10.prepare(S10.AG.load(bg_path, bbox=S10.BBOX), "bg")
        lat, lon, lev = nat["lat"], nat["lon"], nat["lev"]
        above = nat["above"] & bg["above"]

        # 巡航带内取 250 hPa 层（与 step21 一致）
        i250 = int(np.argmin(np.abs(lev - 250.0)))
        mask2d = above[i250]

        dlat = float(np.mean(np.diff(lat))) * 111195.0
        g = max(124e3 / dlat, 0.5)   # L_h≈124 km 的水平平滑（与 step21 同）

        for v in ("T", "u", "v"):
            # 场误差
            eb = bg["f"][v] - nat["f"][v]
            eb2d = eb[i250]
            # 水平平滑后的场误差
            ebs = gaussian_filter(eb2d, sigma=g, mode="nearest")

            # 垂直导数误差（诊断误差）d/dlnp
            deb = vert_deriv(eb, lev)
            deb2d = deb[i250]
            # 水平平滑后场的垂直导数误差
            bg_smooth = gaussian_filter(bg["f"][v], sigma=(0, g, g),
                                        mode="nearest")
            eb_smooth_full = bg_smooth - nat["f"][v]
            deb_smooth = vert_deriv(eb_smooth_full, lev)
            deb_smooth2d = deb_smooth[i250]

            m = np.isfinite(eb2d) & np.isfinite(deb2d)
            # 场误差谱 vs 诊断误差谱
            kf, pf = spectrum2d(eb2d, m)
            kd, pd = spectrum2d(deb2d, m)
            # 平滑后
            _, pfs = spectrum2d(ebs, m)
            _, pds = spectrum2d(deb_smooth2d, m)

            kc_f = centroid(kf, pf)
            kc_d = centroid(kd, pd)

            # 总功率（RMS²，用谱积分近似）
            def rms2(p):
                return float(np.sum(np.asarray(p) * np.gradient(kf)))
            rms2_f = rms2(pf)
            rms2_d = rms2(pd)
            rms2_fs = rms2(pfs)
            rms2_ds = rms2(pds)
            # 水平平滑的相对改善
            imp_field = (rms2_f - rms2_fs) / rms2_f if rms2_f > 0 else 0
            imp_diag = (rms2_d - rms2_ds) / rms2_d if rms2_d > 0 else 0

            rows.append(dict(
                target="%s%s" % (dte, cyc), var=v,
                kc_field=kc_f, kc_diag=kc_d,
                kc_shift=kc_d - kc_f,
                imp_field=imp_field, imp_diag=imp_diag,
                imp_diag_minus_field=imp_diag - imp_field))
            print("  %s%s %s  k质心 场=%.3f 诊断=%.3f 转移%+.3f | "
                  "平滑改善 场%+.1f%% 诊断%+.1f%%"
                  % (dte, cyc, v, kc_f, kc_d, kc_d - kc_f,
                     100 * imp_field, 100 * imp_diag), flush=True)

        del nat, bg
        gc.collect()

    # ---- 汇总 ----
    print("\n" + "=" * 98)
    print("【汇总】谱质心转移 + 水平平滑的谱作用（%d 个 算例×变量）" % len(rows))
    print("=" * 98)
    kc_shift = np.array([r["kc_shift"] for r in rows])
    imp_field = np.array([r["imp_field"] for r in rows])
    imp_diag = np.array([r["imp_diag"] for r in rows])
    n_shift_pos = int((kc_shift > 0).sum())
    print("  谱质心转移 k_c(diag) − k_c(field)：中位 %+.3f，%d/%d 为正"
          % (np.median(kc_shift), n_shift_pos, len(rows)))
    print("  ⇒ 导数算子是否把谱权推向高波数（更'蓝'）？%s"
          % ("是（命题 P 的高通机制在真实大气成立）" if np.median(kc_shift) > 0
             else "否"))
    print()
    print("  水平平滑的相对改善：场误差 中位 %+.1f%%   诊断误差 中位 %+.1f%%"
          % (100 * np.median(imp_field), 100 * np.median(imp_diag)))
    n_diag_gt = int((imp_diag > imp_field).sum())
    print("  ⇒ 水平平滑对诊断的改善 > 对场的改善？%d/%d 算例" % (n_diag_gt, len(rows)))
    print("     （若成立，则'水平小尺度耦合进垂直诊断'在真实大气成立，")
    print("       命题 P 的水平谱 M2 代理是合法的。）")

    out = dict(meta=dict(pairs=PAIRS, bg_lead=BG_LEAD, n=len(rows)),
               rows=rows,
               summary=dict(
                   kc_shift_median=float(np.median(kc_shift)),
                   n_shift_pos=n_shift_pos,
                   imp_field_median=float(np.median(imp_field)),
                   imp_diag_median=float(np.median(imp_diag)),
                   n_diag_gt_field=n_diag_gt))
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
