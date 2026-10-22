"""
step35  推论 A：把业务同化增量截断为尺度分离型，重现"场改善 > 诊断改善"
=========================================================================

科学问题（命题 P 的算子依赖机制）：
  step33 发现——GFS 业务同化（f024→f000）对"场误差"和"垂直梯度误差"的
  相对改善几乎相等（都约 +21%），d_grad - d_field ≈ 0（不显著）。这证伪了
  命题 P 的朴素版本，但精确化为：命题 P 仅在"同化增量谱带宽受限（只动大
  尺度）"时成立。

推论 A（本步验证）：
  把 GFS 业务同化增量  Δ(x) = T_f000(x) - T_f024(x)  做二维水平高斯平滑，
  只保留大尺度分量，构造"截断分析场"：

      T_trunc = T_f024 + G_sigma * Δ

  其中 G_sigma 是水平高斯平滑核（sigma 越大，保留的尺度越大、去的小尺度
  越多）。当 sigma → 0 时 T_trunc ≈ T_f000（全尺度业务同化，d_grad≈d_field）；
  当 sigma → ∞ 时 T_trunc ≈ T_f024 + 大尺度增量（尺度分离型同化）。

  命题 P 的算子依赖机制预测：
    随着 sigma 增大（同化增量越来越"只动大尺度"），
    "场改善 d_field" 与 "梯度改善 d_grad" 的差距 d_grad - d_field 应
    单调变负，重现 OSSE（step21）里 EnOI 的"场改善 6-8% >> 诊断改善 1.8%"
    的分歧。

  若这一单调趋势在真实 GFS 数据 + 独立掩星观测上成立，则直接坐实：
  "命题 P 的分歧来源于同化增量的谱带宽，而非观测或代表性误差"。

数据：
  - COSMIC-2 掩星（独立观测，同 step33）
  - GFS f000/f024（2022-01-15/16）
"""

import glob
import json
import os
import sys
import time

import numpy as np

import step10_realerror_osse as S10
from scipy.ndimage import gaussian_filter

# ---------------- 复用 step33 的配置与函数 ----------------
RO_DAYS = ["014", "015", "016"]
RO_ROOT = "data/cosmic2"
BBOX_LAT = (15.0, 40.0)
BBOX_LON = (100.0, 130.0)
CRUISE = (200.0, 400.0)
ANALYSIS_TIMES = [("20220115", "00"), ("20220115", "06"),
                  ("20220115", "12"), ("20220115", "18"),
                  ("20220116", "00"), ("20220116", "06"),
                  ("20220116", "12"), ("20220116", "18")]

# 平滑尺度扫描（单位：格点，格距 0.25° ≈ 27.8 km）
# sigma=0 → 全尺度业务同化（基准）；sigma 越大 → 越尺度分离
SIGMAS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]


def read_ro_profile(fp):
    import netCDF4 as nc
    hour = None
    base = os.path.basename(fp)
    parts = base.split('.')
    if len(parts) >= 5:
        try:
            hh = int(parts[3]); mm = int(parts[4])
            hour = hh + mm / 60.0
        except (ValueError, IndexError):
            hour = None
    try:
        f = nc.Dataset(fp)
    except Exception:
        return None
    try:
        if 'Pres' not in f.variables or 'Temp' not in f.variables:
            f.close()
            return None
        p = np.asarray(f.variables['Pres'][:], float)
        T = np.asarray(f.variables['Temp'][:], float)
        lat = np.asarray(f.variables['lat'][:], float)
        lon = np.asarray(f.variables['lon'][:], float)
        f.close()
    except Exception:
        try:
            f.close()
        except Exception:
            pass
        return None
    return dict(p=p, T=T, lat=lat, lon=lon, hour=hour)


def collect_ro_profiles():
    profs = []
    for day in RO_DAYS:
        d = os.path.join(RO_ROOT, "wetpf2_%s" % day)
        files = glob.glob(os.path.join(d, "wetPf2_*_nc"))
        n_day = 0
        for fp in files:
            pr = read_ro_profile(fp)
            if pr is None:
                continue
            p = pr['p']; lat = pr['lat']; lon = pr['lon']
            sel = (p >= CRUISE[0]) & (p <= CRUISE[1]) & np.isfinite(lat) & np.isfinite(lon)
            if sel.sum() < 10:
                continue
            mlat = float(np.nanmean(lat[sel]))
            mlon = float(np.nanmean(lon[sel]))
            if not (BBOX_LAT[0] <= mlat <= BBOX_LAT[1] and
                    BBOX_LON[0] <= mlon <= BBOX_LON[1]):
                continue
            valid = np.isfinite(pr['T']) & np.isfinite(pr['p']) & (pr['p'] > 0)
            if valid.sum() < 20:
                continue
            pr['mlat'] = mlat
            pr['mlon'] = mlon
            pr['day'] = day
            profs.append(pr)
            n_day += 1
        print("  day %s: %d 有效（累计 %d）" % (day, n_day, len(profs)), flush=True)
    return profs


def load_gfs_cache():
    cache = {}
    need = set()
    for (date, cyc) in ANALYSIS_TIMES:
        need.add((date, cyc, 0))
        bs = S10.bg_cycle(date, cyc, 24)
        need.add((bs[0], bs[1], 24))
    for (date, cyc, fhr) in sorted(need):
        key = (date, cyc, fhr)
        p = S10.native_path(date, cyc, fhr)
        if not os.path.exists(p):
            cache[key] = None
            continue
        try:
            cache[key] = S10.prepare(S10.AG.load(p, bbox=S10.BBOX))
        except Exception:
            cache[key] = None
    return cache


def ddx_logp(x, logp):
    return np.gradient(x, logp)


def gaussian_smooth_2d(field, sigma):
    """对三维场 (nz, ny, nx) 沿水平二维做高斯平滑（sigma 单位格点）。"""
    if sigma <= 0:
        return field.copy()
    return gaussian_filter(field, sigma=(0, sigma, sigma), mode="nearest")


def main():
    t0 = time.time()
    print("=" * 98)
    print("step35  推论 A：截断业务同化增量为尺度分离型，重现命题 P 分歧")
    print("=" * 98)

    # ---------- 1. 收集掩星 ----------
    print("\n[1] 收集东亚掩星廓线")
    print("-" * 98)
    profs = collect_ro_profiles()
    print("  东亚有效廓线: %d" % len(profs))

    # ---------- 2. 加载 GFS ----------
    print("\n[2] 加载 GFS f000/f024")
    print("-" * 98)
    gfs_cache = load_gfs_cache()

    # ---------- 3. 逐廓线：先算基准（全尺度）误差与梯度 ----------
    print("\n[3] 逐廓线误差分析（含同化增量谱截断扫描）")
    print("-" * 98)

    # 先收集所有可用的配对廓线（一次匹配，多次截断评估）
    matched = []   # 每个元素含 GFS 单格点廓线 + 掩星廓线 + 时间/位置信息
    for pr in profs:
        hour = pr.get('hour'); day = pr.get('day')
        if hour is None or day is None:
            continue
        day_int = int(day)
        date = "202201%02d" % day_int
        best = None; best_dt = 99.0
        for cyc in ("00", "06", "12", "18"):
            c = float(cyc)
            dt = abs(hour - c)
            dt = min(dt, abs(dt - 24))
            if dt < best_dt:
                best_dt = dt; best = cyc
        if best_dt > 3.0:
            continue
        cyc = best
        nat = gfs_cache.get((date, cyc, 0))
        if nat is None:
            continue
        bspec = S10.bg_cycle(date, cyc, 24)
        bg = gfs_cache.get((bspec[0], bspec[1], 24))
        if bg is None:
            continue
        lat = nat['lat']; lon = nat['lon']; lev = np.asarray(nat['lev'], float)
        mlat = pr['mlat']; mlon = pr['mlon']
        ilat = int(np.argmin(np.abs(lat - mlat)))
        ilon = int(np.argmin(np.abs(lon - mlon)))
        if abs(lat[ilat] - mlat) > 2.0 or abs(lon[ilon] - mlon) > 2.0:
            continue
        # 掩星 -> GFS 层
        p_ro = pr['p']; T_ro = pr['T']
        valid = np.isfinite(T_ro) & np.isfinite(p_ro) & (p_ro > 0)
        if valid.sum() < 20:
            continue
        p_ro = p_ro[valid]; T_ro = T_ro[valid]
        logp_g = np.log(lev)
        logp_r = np.log(p_ro)
        order = np.argsort(p_ro)
        T_ro_glev = np.interp(logp_g, logp_r[order], T_ro[order])

        matched.append(dict(
            ilat=ilat, ilon=ilon,
            T_ro_glev=T_ro_glev,
            nat_T=nat['f']['T'], bg_T=bg['f']['T'],
            lev=lev, logp_g=logp_g,
        ))

    print("  有效匹配样本: %d" % len(matched))
    if len(matched) == 0:
        print("  !! 无有效样本"); return

    cruise_mask = (np.asarray(matched[0]['lev']) >= CRUISE[0]) & \
                  (np.asarray(matched[0]['lev']) <= CRUISE[1])

    def rmse(e, mask):
        m = mask & np.isfinite(e)
        return float(np.sqrt(np.nanmean(e[m] ** 2))) if m.sum() > 3 else np.nan

    # ---------- 4. 对每个 sigma 计算 d_field / d_grad ----------
    print("\n[4] 同化增量谱截断扫描")
    print("-" * 98)
    print("  sigma(格点) | d_field | d_grad | d_grad-d_field | 支持命题P占比")

    results = {}
    for sigma in SIGMAS:
        d_field_l = []; d_grad_l = []
        for m in matched:
            T_f024 = m['bg_T'][:, m['ilat'], m['ilon']] - 273.15
            T_f000 = m['nat_T'][:, m['ilat'], m['ilon']] - 273.15
            T_ro = m['T_ro_glev']
            logp = m['logp_g']

            if sigma <= 0:
                T_an = T_f000
            else:
                # 截断分析场 = 背景 + 平滑后的增量（全场上做，再取该格点）
                delta = m['nat_T'] - m['bg_T']          # (nz, ny, nx) K
                delta_s = gaussian_smooth_2d(delta, sigma)
                T_an = m['bg_T'][:, m['ilat'], m['ilon']] + delta_s[:, m['ilat'], m['ilon']]
                T_an = T_an - 273.15

            e_field_bg = T_f024 - T_ro
            e_field_an = T_an - T_ro
            grad_bg = ddx_logp(T_f024, logp)
            grad_an = ddx_logp(T_an, logp)
            grad_ro = ddx_logp(T_ro, logp)
            e_grad_bg = grad_bg - grad_ro
            e_grad_an = grad_an - grad_ro

            r_bg_f = rmse(e_field_bg, cruise_mask)
            r_an_f = rmse(e_field_an, cruise_mask)
            r_bg_g = rmse(e_grad_bg, cruise_mask)
            r_an_g = rmse(e_grad_an, cruise_mask)
            if not (np.isfinite(r_bg_f) and np.isfinite(r_an_f) and
                    np.isfinite(r_bg_g) and np.isfinite(r_an_g)):
                continue
            if r_bg_f == 0 or r_bg_g == 0:
                continue
            d_field_l.append((r_bg_f - r_an_f) / r_bg_f)
            d_grad_l.append((r_bg_g - r_an_g) / r_bg_g)

        d_field_l = np.array(d_field_l)
        d_grad_l = np.array(d_grad_l)
        diff = d_grad_l - d_field_l
        med_f = np.median(d_field_l)
        med_g = np.median(d_grad_l)
        med_d = np.median(diff)
        p_support = float((diff < 0).mean())
        results[sigma] = dict(
            d_field=float(med_f), d_grad=float(med_g),
            d_diff=float(med_d), p_support=p_support,
            n=int(len(d_field_l)),
        )
        print("  %8.1f   | %+6.1f%% | %+6.1f%% | %+8.1f%% | %.1f%%" %
              (sigma, 100 * med_f, 100 * med_g, 100 * med_d, 100 * p_support))

    # ---------- 5. bootstrap CI 对关键 sigma ----------
    print("\n[5] 关键 sigma 的 bootstrap 95% CI（d_grad - d_field）")
    print("-" * 98)

    def boot_ci(x, n=2000, seed=42):
        rng = np.random.default_rng(seed)
        meds = np.array([np.median(x[rng.integers(0, len(x), len(x))])
                         for _ in range(n)])
        return np.median(x), np.percentile(meds, [2.5, 97.5])

    # 重新收集每个 sigma 的 diff 数组（用于 CI）
    ci_results = {}
    for sigma in SIGMAS:
        d_field_l = []; d_grad_l = []
        for m in matched:
            T_f024 = m['bg_T'][:, m['ilat'], m['ilon']] - 273.15
            T_f000 = m['nat_T'][:, m['ilat'], m['ilon']] - 273.15
            T_ro = m['T_ro_glev']; logp = m['logp_g']
            if sigma <= 0:
                T_an = T_f000
            else:
                delta = m['nat_T'] - m['bg_T']
                delta_s = gaussian_smooth_2d(delta, sigma)
                T_an = m['bg_T'][:, m['ilat'], m['ilon']] + delta_s[:, m['ilat'], m['ilon']]
                T_an = T_an - 273.15
            e_field_bg = T_f024 - T_ro; e_field_an = T_an - T_ro
            grad_bg = ddx_logp(T_f024, logp); grad_an = ddx_logp(T_an, logp)
            grad_ro = ddx_logp(T_ro, logp)
            r_bg_f = rmse(e_field_bg, cruise_mask); r_an_f = rmse(e_field_an, cruise_mask)
            r_bg_g = rmse(grad_bg - grad_ro, cruise_mask); r_an_g = rmse(grad_an - grad_ro, cruise_mask)
            if not (np.isfinite(r_bg_f) and np.isfinite(r_an_f) and
                    np.isfinite(r_bg_g) and np.isfinite(r_an_g)):
                continue
            if r_bg_f == 0 or r_bg_g == 0:
                continue
            d_field_l.append((r_bg_f - r_an_f) / r_bg_f)
            d_grad_l.append((r_bg_g - r_an_g) / r_bg_g)
        d_field_l = np.array(d_field_l); d_grad_l = np.array(d_grad_l)
        diff = d_grad_l - d_field_l
        med_d, ci_d = boot_ci(diff)
        med_f, ci_f = boot_ci(d_field_l)
        med_g, ci_g = boot_ci(d_grad_l)
        ci_results[sigma] = dict(
            d_field=float(med_f), d_field_ci=list(ci_f),
            d_grad=float(med_g), d_grad_ci=list(ci_g),
            d_diff=float(med_d), d_diff_ci=list(ci_d),
            p_support=float((diff < 0).mean()),
            n=int(len(diff)),
        )
        print("  sigma=%4.1f: d_field=%+5.1f%%[%+.1f,%+.1f]  d_grad=%+5.1f%%[%+.1f,%+.1f]  "
              "d_diff=%+5.1f%%[%+.1f,%+.1f]  显著=%s" %
              (sigma, 100*med_f, 100*ci_f[0], 100*ci_f[1],
               100*med_g, 100*ci_g[0], 100*ci_g[1],
               100*med_d, 100*ci_d[0], 100*ci_d[1],
               "否" if (ci_d[0] < 0 < ci_d[1]) else "是"))

    # ---------- 6. 保存 ----------
    out = dict(
        n_samples=len(matched),
        sigmas=SIGMAS,
        scan=results,
        ci=ci_results,
    )
    os.makedirs("data", exist_ok=True)
    with open("data/step35_truncate_assim.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n  结果已保存: data/step35_truncate_assim.json")
    print("  总耗时: %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
