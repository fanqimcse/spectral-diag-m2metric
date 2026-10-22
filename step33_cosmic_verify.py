"""
step33  P3-final：用 GPS 掩星（COSMIC-2）格点化独立观测坐实命题 P
=================================================================

科学问题：
  命题 P 声称——同化（EnOI，f024→f000）对"垂直导数型诊断"（如 dT/dlnp、
  N²、Ri）的相对改善，弱于对"场量"（T、q）的相对改善。原因：诊断误差由
  误差谱的高波数（小尺度）分量主导，而同化只移除大尺度偏差。

step29（探空）受"代表性误差"限制：探空是点观测，GFS 是格点均值，二者的
小尺度差异不可分。step32（探空空间合成）得到负结果，说明代表性误差不是
主因，而是探空的"垂直小尺度扰动"既非背景偏差、也无法被同化修正。

本步（step33）用 COSMIC-2 掩星做关键升级：
  1. 掩星是**完全独立于 GFS** 的观测（GPS 信号反演，非任何 NWP 同化）；
  2. 掩星垂直分辨率 ~100 m（787 层），远高于 GFS 21 层，能分辨 GFS 无法
     表示的小尺度梯度；
  3. 掩星覆盖全球，东亚区域每天 ~90 条廓线，可做格点化/区域合成。

核心检验（与 step29 同构，但观测更独立、分辨率更高）：
  A. 导数放大：垂直梯度误差相对场误差被放大多少倍（GFS f024 对掩星）；
  B. 谱分工：同化（f024→f000）对场改善 vs 对梯度改善的相对大小；
  C. 命题 P 判据：d_grad - d_field 的符号与显著性。

方法要点（避免 step29/32 的坑）：
  - 掩星廓线插值到 GFS 21 层（log-p 线性），确保场与梯度同口径；
  - 巡航带 200-400 hPa（飞行路径风险评估的核心层）；
  - 逐廓线配对差值（d_grad - d_field 先逐样本再聚合），不用独立中位相减；
  - 用配对 bootstrap 给 95% CI。

数据：
  - COSMIC-2 wetPf2 掩星：data/cosmic2/wetpf2_{014,015,016}/（2022-01-14~16）
  - GFS f000（分析）/ f024（背景）：data/aws/（2022-01-15/16）
"""

import glob
import json
import os
import sys
import time

import numpy as np

import step10_realerror_osse as S10

# ---------------- 掩星数据配置 ----------------
RO_DAYS = ["014", "015", "016"]          # 2022-01-14/15/16 儒略日
RO_ROOT = "data/cosmic2"
BBOX_LAT = (15.0, 40.0)                   # 与 S10.BBOX 一致 (lat 15-40)
BBOX_LON = (100.0, 130.0)                 # 与 S10.BBOX 一致 (lon 100-130)
CRUISE = (200.0, 400.0)                   # 巡航层 hPa

# GFS 目标时次（分析场 f000 与背景 f024 配对）。
# 用 6h 时次（00/06/12/18Z）以充分利用掩星全天覆盖。
# 掩星日期 = 2022-01-<day>，day 为儒略日 014/015/016（即 01-14/15/16）。
# 注意：01-14 没有 f000 分析场（只有 f024 背景），故 01-14 的掩星廓线
# 无法做"同化改善"检验，会在匹配时被自然跳过。
ANALYSIS_TIMES = [("20220115", "00"), ("20220115", "06"),
                  ("20220115", "12"), ("20220115", "18"),
                  ("20220116", "00"), ("20220116", "06"),
                  ("20220116", "12"), ("20220116", "18")]


# ---------------- 掩星读取 ----------------
def read_ro_profile(fp):
    """读取一个 wetPf2 廓线，返回 dict(p, T, q, lat, lon, alt, hour)。
    失败/坏文件返回 None。

    文件名形如 wetPf2_C2E1.2022.015.04.21.R18_0001.0001_nc，
    其中 .04.21 是 UTC 时.分。hour = 4 + 21/60 ≈ 4.35。
    """
    import netCDF4 as nc
    # 从文件名解析 UTC 小时
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
        p = np.asarray(f.variables['Pres'][:], float)     # hPa
        T = np.asarray(f.variables['Temp'][:], float)     # °C
        sph = np.asarray(f.variables['sph'][:], float)    # g/kg
        lat = np.asarray(f.variables['lat'][:], float)
        lon = np.asarray(f.variables['lon'][:], float)
        alt = np.asarray(f.variables['MSL_alt'][:], float) if 'MSL_alt' in f.variables else None
        f.close()
    except Exception:
        try:
            f.close()
        except Exception:
            pass
        return None
    return dict(p=p, T=T, q=sph, lat=lat, lon=lon, alt=alt, hour=hour)


def collect_ro_profiles():
    """收集所有东亚区域的有效掩星廓线。"""
    profs = []
    for day in RO_DAYS:
        d = os.path.join(RO_ROOT, "wetpf2_%s" % day)
        files = glob.glob(os.path.join(d, "wetPf2_*_nc"))
        n_day = 0
        for fp in files:
            pr = read_ro_profile(fp)
            if pr is None:
                continue
            # 廓线中点位置（掩星切点漂移，用巡航层中点的平均位置）
            p = pr['p']
            lat = pr['lat']
            lon = pr['lon']
            sel = (p >= CRUISE[0]) & (p <= CRUISE[1]) & np.isfinite(lat) & np.isfinite(lon)
            if sel.sum() < 10:
                continue
            mlat = float(np.nanmean(lat[sel]))
            mlon = float(np.nanmean(lon[sel]))
            if not (BBOX_LAT[0] <= mlat <= BBOX_LAT[1] and
                    BBOX_LON[0] <= mlon <= BBOX_LON[1]):
                continue
            # 校验有效温度层数
            valid = np.isfinite(pr['T']) & np.isfinite(pr['p']) & (pr['p'] > 0)
            if valid.sum() < 20:
                continue
            pr['mlat'] = mlat
            pr['mlon'] = mlon
            pr['day'] = day            # 儒略日，用于匹配 GFS 日期
            profs.append(pr)
            n_day += 1
        print("  day %s: %d 有效（累计 %d）" % (day, n_day, len(profs)),
              flush=True)
    return profs


# ---------------- GFS 匹配 ----------------
def load_gfs_cache():
    """预加载 GFS f000/f024 场。返回 dict[(date, cyc, fhr)] -> prepare dict。

    需要加载两类：
    1. 分析场 f000：ANALYSIS_TIMES 各时次；
    2. 背景场 f024：由 bg_cycle 反推的起报时次（可能在前一天）。
    """
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
        except Exception as e:
            cache[key] = None
    return cache


def interp_gfs_to_ro(gfs_field, lev, pr):
    """把 GFS 场插值到掩星的气压层。gfs_field: (nz,) 单格点廓线；lev: (nz,) hPa。"""
    # 掩星气压升序（掩星通常从顶到底，p 递减 → 转升序）
    p_ro = pr['p']
    order = np.argsort(p_ro)
    p_ro_s = p_ro[order]
    # 用 log-p 线性插值
    logp_g = np.log(np.asarray(lev, float))
    logp_r = np.log(p_ro_s)
    # GFS 层升序
    g_sorted = gfs_field[np.argsort(lev)]
    return np.interp(logp_r, logp_g, g_sorted)


def ddx_logp(x, logp):
    """垂直导数（对 log-p 差分），用 np.gradient。"""
    return np.gradient(x, logp)


def main():
    t0 = time.time()
    print("=" * 98)
    print("step33  GPS 掩星（COSMIC-2）独立观测坐实命题 P")
    print("=" * 98)

    # ---------- 1. 收集掩星廓线 ----------
    print("\n[1] 收集东亚掩星廓线")
    print("-" * 98)
    profs = collect_ro_profiles()
    print("  东亚区域有效廓线总数: %d" % len(profs))

    # ---------- 2. 加载 GFS ----------
    print("\n[2] 加载 GFS f000/f024 配对场")
    print("-" * 98)
    gfs_cache = load_gfs_cache()

    # ---------- 3. 逐廓线误差分析 ----------
    print("\n[3] 逐廓线：场误差 vs 梯度误差，同化相对改善")
    print("-" * 98)

    rows = []
    # 按时间匹配：每条廓线匹配到最接近的 GFS 分析时次（00/06/12/18Z），
    # ±3h 时间窗保证严格对齐。
    for pr in profs:
        hour = pr.get('hour')
        day = pr.get('day')
        if hour is None or day is None:
            continue
        # 儒略日 -> 公历日期（2022年1月：儒略日 = 月内天数）
        # day 是 "014"/"015"/"016" -> 01-14/01-15/01-16
        day_int = int(day)
        date = "202201%02d" % day_int
        # 找最近的 6h 分析时次
        best = None
        best_dt = 99.0
        for cyc in ("00", "06", "12", "18"):
            c = float(cyc)
            dt = abs(hour - c)
            dt = min(dt, abs(dt - 24))   # 跨日环绕
            if dt < best_dt:
                best_dt = dt
                best = cyc
        if best_dt > 3.0:
            continue                       # 时间窗太远，跳过

        cyc = best
        nat = gfs_cache.get((date, cyc, 0))
        if nat is None:
            continue
        bspec = S10.bg_cycle(date, cyc, 24)
        bg = gfs_cache.get((bspec[0], bspec[1], 24))
        if bg is None:
            continue

        lat = nat['lat']
        lon = nat['lon']
        lev = np.asarray(nat['lev'], float)

        mlat = pr['mlat']
        mlon = pr['mlon']
        ilat = int(np.argmin(np.abs(lat - mlat)))
        ilon = int(np.argmin(np.abs(lon - mlon)))
        dlat = abs(lat[ilat] - mlat)
        dlon = abs(lon[ilon] - mlon)
        if dlat > 2.0 or dlon > 2.0:
            continue

        # 掩星温度 → 插值到 GFS 层（log-p）
        p_ro = pr['p']
        T_ro = pr['T']
        valid = np.isfinite(T_ro) & np.isfinite(p_ro) & (p_ro > 0)
        if valid.sum() < 20:
            continue
        p_ro = p_ro[valid]
        T_ro = T_ro[valid]

        # 掩星 → GFS 层（log-p 线性插值）
        logp_g = np.log(lev)
        logp_r = np.log(p_ro)
        order = np.argsort(p_ro)
        T_ro_glev = np.interp(logp_g, logp_r[order], T_ro[order])

        # GFS 温度（K → °C）
        T_f000 = nat['f']['T'][:, ilat, ilon] - 273.15
        T_f024 = bg['f']['T'][:, ilat, ilon] - 273.15

        # 巡航层掩码
        cruise = (lev >= CRUISE[0]) & (lev <= CRUISE[1])

        # 误差（对掩星）
        e_T_f024 = T_f024 - T_ro_glev
        e_T_f000 = T_f000 - T_ro_glev

        # 垂直梯度（dT/dlnp）
        grad_ro = ddx_logp(T_ro_glev, logp_g)
        grad_f024 = ddx_logp(T_f024, logp_g)
        grad_f000 = ddx_logp(T_f000, logp_g)
        e_grad_f024 = grad_f024 - grad_ro
        e_grad_f000 = grad_f000 - grad_ro

        def rmse(e, mask):
            m = mask & np.isfinite(e)
            return float(np.sqrt(np.nanmean(e[m] ** 2))) if m.sum() > 3 else np.nan

        rmse_T_f024 = rmse(e_T_f024, cruise)
        rmse_T_f000 = rmse(e_T_f000, cruise)
        rmse_grad_f024 = rmse(e_grad_f024, cruise)
        rmse_grad_f000 = rmse(e_grad_f000, cruise)

        if not (np.isfinite(rmse_T_f024) and np.isfinite(rmse_T_f000) and
                np.isfinite(rmse_grad_f024) and np.isfinite(rmse_grad_f000)):
            continue
        if rmse_T_f024 == 0 or rmse_grad_f024 == 0:
            continue

        # 相对改善（同化 f024→f000，正值=改善）
        d_field = (rmse_T_f024 - rmse_T_f000) / rmse_T_f024
        d_grad = (rmse_grad_f024 - rmse_grad_f000) / rmse_grad_f024

        rows.append(dict(
            date=date, cyc=cyc, hour=hour,
            mlat=float(mlat), mlon=float(mlon),
            rmse_T_f024=rmse_T_f024, rmse_T_f000=rmse_T_f000,
            rmse_grad_f024=rmse_grad_f024, rmse_grad_f000=rmse_grad_f000,
            d_field=float(d_field), d_grad=float(d_grad),
            d_grad_minus_field=float(d_grad - d_field),
        ))

    print("  有效匹配样本: %d" % len(rows))
    if len(rows) == 0:
        print("  !! 无有效样本，终止")
        return

    # ---------- 4. 汇总统计 ----------
    print("\n[4] 汇总统计（配对 bootstrap 95% CI）")
    print("-" * 98)

    d_field = np.array([r['d_field'] for r in rows])
    d_grad = np.array([r['d_grad'] for r in rows])
    d_diff = np.array([r['d_grad_minus_field'] for r in rows])

    def boot_ci(x, n=2000, seed=42):
        rng = np.random.default_rng(seed)
        meds = []
        for _ in range(n):
            idx = rng.integers(0, len(x), len(x))
            meds.append(np.median(x[idx]))
        meds = np.array(meds)
        return np.median(x), np.percentile(meds, [2.5, 97.5])

    med_field, ci_field = boot_ci(d_field)
    med_grad, ci_grad = boot_ci(d_grad)
    med_diff, ci_diff = boot_ci(d_diff)

    # 导数放大倍数：梯度误差 / 场误差（f024 背景）
    amp = np.array([r['rmse_grad_f024'] / r['rmse_T_f024'] for r in rows])
    med_amp, ci_amp = boot_ci(amp)

    print("  === 同化相对改善（正值=改善）===")
    print("  场误差 d_field      : %+6.1f%%  CI[%+.1f, %+.1f]" %
          (100 * med_field, 100 * ci_field[0], 100 * ci_field[1]))
    print("  梯度误差 d_grad     : %+6.1f%%  CI[%+.1f, %+.1f]" %
          (100 * med_grad, 100 * ci_grad[0], 100 * ci_grad[1]))
    print("  d_grad - d_field    : %+6.1f%%  CI[%+.1f, %+.1f]" %
          (100 * med_diff, 100 * ci_diff[0], 100 * ci_diff[1]))
    print("  === 导数放大倍数（背景误差梯度/场）===")
    print("  放大倍数            : %5.1fx   CI[%.1f, %.1f]" %
          (med_amp, ci_amp[0], ci_amp[1]))

    # 命题 P 判据：d_grad - d_field < 0 ?
    p_support = (d_diff < 0).mean()
    print("\n  === 命题 P 判据 ===")
    print("  命题 P 预测: 梯度改善 < 场改善 (d_grad - d_field < 0)")
    print("  实测 d_grad-d_field 中位 = %+.1f%%" % (100 * med_diff))
    print("  CI 是否含 0: %s" % ("是（不显著）" if (ci_diff[0] < 0 < ci_diff[1]) else "否（显著）"))
    print("  支持命题 P 的样本占比: %.1f%%" % (100 * p_support))

    # ---------- 5. 保存 ----------
    out = dict(
        n_samples=len(rows),
        d_field=dict(median=float(med_field), ci=list(ci_field)),
        d_grad=dict(median=float(med_grad), ci=list(ci_grad)),
        d_diff=dict(median=float(med_diff), ci=list(ci_diff)),
        amp=dict(median=float(med_amp), ci=list(ci_amp)),
        p_support=float(p_support),
        rows=rows,
    )
    os.makedirs("data", exist_ok=True)
    with open("data/step33_cosmic_verify.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n  结果已保存: data/step33_cosmic_verify.json")
    print("  总耗时: %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
