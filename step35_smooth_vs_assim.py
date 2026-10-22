"""
step35  推论 A（修正版）：在掩星观测框架下复现"平滑 vs 同化"的谱分工
=========================================================================

背景：
  step35 初版（对业务同化增量 Δ=f000-f024 做高斯平滑截断）得到诚实负结果：
  仅把增量截断为尺度分离型，不能重现"场改善 > 诊断改善"的分歧。

  根本原因（仔细核对 step21 后确认）：
  step21 的"同化 vs 平滑"谱分工，对比的是**背景场本身的误差谱**（对真值），
  其中"平滑"是 gaussian_filter(bg, sigma=L_h/dlat) 直接平滑背景场，
  而非"对同化增量做平滑"。这是两个本质不同的算子。

推论 A 的正确表述与检验：
  命题 P 声称——同化（光滑增量）只修大尺度，对导数型诊断改善弱；
  而"尺度分离"（平滑背景消小尺度噪声）命中诊断最敏感部分。
  因此，在**同一个独立掩星观测框架**下，应比较两种"改进"操作：

    (1) 业务同化  f024 → f000        （全尺度同化，GFS 混合 4DEnVar）
    (2) 背景平滑  f024 → smooth(f024) （尺度分离，只消小尺度噪声）

  命题 P 预测：
    - 业务同化：场改善 ≈ 梯度改善（step33 已证，都约 +21%）
    - 背景平滑：场改善 > 梯度改善，即 d_grad - d_field 显著为负

  若 (2) 成立而 (1) 不成立，则**直接坐实命题 P 的算子依赖机制**：
  分歧不是"同化 vs 观测"的问题，而是"全尺度算子 vs 尺度分离算子"的问题。

  同时做平滑尺度扫描（sigma = L_h/dlat，L_h ∈ {60, 124, 200, 300} km），
  检验 d_grad - d_field 是否随平滑尺度增大而单调变负（越只消大尺度、
  分歧越小；越消到小尺度、分歧越大）。

数据（同 step33）：
  - COSMIC-2 掩星 2022-01-14~16
  - GFS f000/f024 2022-01-15/16
"""

import glob
import json
import os
import sys
import time

import numpy as np

import step10_realerror_osse as S10
from scipy.ndimage import gaussian_filter

RO_DAYS = ["014", "015", "016"]
RO_ROOT = "data/cosmic2"
BBOX_LAT = (15.0, 40.0)
BBOX_LON = (100.0, 130.0)
CRUISE = (200.0, 400.0)
ANALYSIS_TIMES = [("20220115", "00"), ("20220115", "06"),
                  ("20220115", "12"), ("20220115", "18"),
                  ("20220116", "00"), ("20220116", "06"),
                  ("20220116", "12"), ("20220116", "18")]

# 平滑尺度（单位 km，对应水平高斯 sigma = L_h / 27.8 km 格距）
# L_h = 124 km 是 step21 里 GFS 背景误差诊断的水平相关长度 L_h1
SMOOTH_KM = [60.0, 124.0, 200.0, 300.0]
DLAT_KM = 27.8   # 0.25° ≈ 27.8 km


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


def main():
    t0 = time.time()
    print("=" * 98)
    print("step35  推论 A（修正版）：平滑背景 vs 业务同化的谱分工检验")
    print("=" * 98)

    print("\n[1] 收集东亚掩星廓线")
    print("-" * 98)
    profs = collect_ro_profiles()
    print("  东亚有效廓线: %d" % len(profs))

    print("\n[2] 加载 GFS f000/f024")
    print("-" * 98)
    gfs_cache = load_gfs_cache()

    # ---------- 收集匹配样本（含 GFS 全场，供平滑） ----------
    print("\n[3] 匹配掩星廓线到 GFS 格点")
    print("-" * 98)
    # 为避免对每条廓线重复加载/平滑全场，先按 (date,cyc) 缓存平滑后的背景场
    matched = []
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
            date=date, cyc=cyc, ilat=ilat, ilon=ilon,
            T_ro_glev=T_ro_glev, logp_g=logp_g, lev=lev,
            nat_T=nat['f']['T'], bg_T=bg['f']['T'],
        ))

    print("  有效匹配样本: %d" % len(matched))
    if len(matched) == 0:
        print("  !! 无有效样本"); return

    cruise_mask = (np.asarray(matched[0]['lev']) >= CRUISE[0]) & \
                  (np.asarray(matched[0]['lev']) <= CRUISE[1])

    def rmse(e, mask):
        m = mask & np.isfinite(e)
        return float(np.sqrt(np.nanmean(e[m] ** 2))) if m.sum() > 3 else np.nan

    def boot_ci(x, n=2000, seed=42):
        rng = np.random.default_rng(seed)
        meds = np.array([np.median(x[rng.integers(0, len(x), len(x))])
                         for _ in range(n)])
        return np.median(x), np.percentile(meds, [2.5, 97.5])

    # 缓存每个 (date,cyc) 的平滑背景场（不同 L_h）
    print("\n[4] 平滑背景场缓存（L_h ∈ %s km）" % SMOOTH_KM)
    print("-" * 98)
    smbg_cache = {}   # key: (date, cyc, L_h) -> dict{var: 全场}
    field_keys = ("T", "q", "u", "v")
    for L_h in SMOOTH_KM:
        sigma = L_h / DLAT_KM
        for m in matched:
            key = (m['date'], m['cyc'], L_h)
            if key in smbg_cache:
                continue
            # 只平滑 T（本检验用温度），其它变量也一并缓存以保持通用
            sm = {}
            for v in field_keys:
                if v in m and m[v] is not None:
                    pass
            # 直接从 gfs_cache 取背景场平滑
            bg_full = gfs_cache.get((m['date'], m['cyc'], 24))
            if bg_full is None:
                smbg_cache[key] = None
                continue
            sm = {v: gaussian_filter(bg_full['f'][v],
                                     sigma=(0, sigma, sigma), mode="nearest")
                  for v in field_keys}
            smbg_cache[key] = sm
        print("  L_h=%6.1f km (sigma=%.2f 格点) 已缓存 %d 个背景场"
              % (L_h, sigma, sum(1 for k in smbg_cache if k[2] == L_h)))

    # ---------- 对比：业务同化 vs 背景平滑 ----------
    print("\n[5] 对比：业务同化(f000) vs 背景平滑(sm_bg) 的场/梯度改善")
    print("-" * 98)
    print("  算子                 | d_field | d_grad | d_grad-d_field | 支持P占比")

    # (1) 业务同化
    d_field_a = []; d_grad_a = []
    for m in matched:
        T_bg = m['bg_T'][:, m['ilat'], m['ilon']] - 273.15
        T_an = m['nat_T'][:, m['ilat'], m['ilon']] - 273.15
        T_ro = m['T_ro_glev']; logp = m['logp_g']
        r_bg_f = rmse(T_bg - T_ro, cruise_mask)
        r_an_f = rmse(T_an - T_ro, cruise_mask)
        r_bg_g = rmse(ddx_logp(T_bg, logp) - ddx_logp(T_ro, logp), cruise_mask)
        r_an_g = rmse(ddx_logp(T_an, logp) - ddx_logp(T_ro, logp), cruise_mask)
        if not (np.isfinite(r_bg_f) and np.isfinite(r_an_f) and
                np.isfinite(r_bg_g) and np.isfinite(r_an_g)):
            continue
        if r_bg_f == 0 or r_bg_g == 0:
            continue
        d_field_a.append((r_bg_f - r_an_f) / r_bg_f)
        d_grad_a.append((r_bg_g - r_an_g) / r_bg_g)
    d_field_a = np.array(d_field_a); d_grad_a = np.array(d_grad_a)
    diff_a = d_grad_a - d_field_a
    mf, cf = boot_ci(d_field_a); mg, cg = boot_ci(d_grad_a); md, cd = boot_ci(diff_a)
    print("  业务同化 f000        | %+6.1f%% | %+6.1f%% | %+8.1f%%[%+.1f,%+.1f] | %.1f%%" %
          (100*mf, 100*mg, 100*md, 100*cd[0], 100*cd[1], 100*(diff_a < 0).mean()))

    # (2) 背景平滑（各 L_h）
    smooth_results = {}
    for L_h in SMOOTH_KM:
        d_field_s = []; d_grad_s = []
        for m in matched:
            key = (m['date'], m['cyc'], L_h)
            sm = smbg_cache.get(key)
            if sm is None:
                continue
            T_bg = m['bg_T'][:, m['ilat'], m['ilon']] - 273.15
            T_sm = sm['T'][:, m['ilat'], m['ilon']] - 273.15
            T_ro = m['T_ro_glev']; logp = m['logp_g']
            r_bg_f = rmse(T_bg - T_ro, cruise_mask)
            r_sm_f = rmse(T_sm - T_ro, cruise_mask)
            r_bg_g = rmse(ddx_logp(T_bg, logp) - ddx_logp(T_ro, logp), cruise_mask)
            r_sm_g = rmse(ddx_logp(T_sm, logp) - ddx_logp(T_ro, logp), cruise_mask)
            if not (np.isfinite(r_bg_f) and np.isfinite(r_sm_f) and
                    np.isfinite(r_bg_g) and np.isfinite(r_sm_g)):
                continue
            if r_bg_f == 0 or r_bg_g == 0:
                continue
            d_field_s.append((r_bg_f - r_sm_f) / r_bg_f)
            d_grad_s.append((r_bg_g - r_sm_g) / r_bg_g)
        d_field_s = np.array(d_field_s); d_grad_s = np.array(d_grad_s)
        diff_s = d_grad_s - d_field_s
        mf, cf = boot_ci(d_field_s); mg, cg = boot_ci(d_grad_s); md, cd = boot_ci(diff_s)
        smooth_results[L_h] = dict(
            d_field=float(mf), d_field_ci=list(cf),
            d_grad=float(mg), d_grad_ci=list(cg),
            d_diff=float(md), d_diff_ci=list(cd),
            p_support=float((diff_s < 0).mean()),
            n=int(len(d_field_s)),
        )
        print("  背景平滑 L_h=%5.1fkm | %+6.1f%% | %+6.1f%% | %+8.1f%%[%+.1f,%+.1f] | %.1f%%" %
              (L_h, 100*mf, 100*mg, 100*md, 100*cd[0], 100*cd[1], 100*(diff_s < 0).mean()))

    # ---------- 保存 ----------
    out = dict(
        n_samples=len(matched),
        assimilation=dict(
            d_field=float(np.median(d_field_a)),
            d_grad=float(np.median(d_grad_a)),
            d_diff=float(np.median(diff_a)),
            p_support=float((diff_a < 0).mean()),
            n=int(len(d_field_a)),
        ),
        smooth=smooth_results,
        smooth_km=SMOOTH_KM,
    )
    os.makedirs("data", exist_ok=True)
    with open("data/step35_smooth_vs_assim.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n  结果已保存: data/step35_smooth_vs_assim.json")
    print("  总耗时: %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
