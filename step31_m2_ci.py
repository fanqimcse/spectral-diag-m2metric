# -*- coding: utf-8 -*-
"""step31 —— P2：M2 判别力的置信区间 + 跨时次/跨季节稳健性。

step26 的头号数字"M2 判别力 = +21.4%"是**点估计**，没有置信区间；
而 M2 本身已被证明有截断偏倚—方差权衡（CV 高达 4.7%）。独立审稿人指出：
没有 CI 的 +21.4%，在 Nature 级标准下不够。

本脚本补两件事：

  [A] 把算例从 step21 的 4 个（全是 10 月）扩到 **8 个配对**，跨
      2 年（2022/2023）× 2 季（冬 1 月 / 夏 7 月）× 2 时次（00/12），
      检验 M0 vs M2 判别力在跨季节、跨年时的稳健性。

  [B] 对"判别力 Δ = M2 判别力 − M0 判别力"做**配对 bootstrap**，
      给出 95% 置信区间，回答"+21.4% 是否显著异于 0"。

方法：复用 step21 的谱分解（EnOI 同化 + 高斯平滑 + 二维水平谱 + 波段归因），
但按 step26 的 M0/M2 定义从误差谱直接积分，得到每个 (算例×变量) 的
dM0_an/dM0_sbg/dM2_an/dM2_sbg，再跨算例取中位并 bootstrap。

自包含：复用本仓库已有 8 个配对的 GFS 数据（data/aws），不新下载。
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
from avwx import enoi as EN

OUT = "data/step31_m2_ci.json"
KEYS = S12.KEYS
CRUISE = (200.0, 400.0)
RNG = np.random.default_rng(20261022)

# 8 个配对（跨 2 年 × 2 季 × 2 时次）
PAIRS = [
    ("20220115", "00"), ("20220115", "12"),
    ("20220715", "00"), ("20220715", "12"),
    ("20230115", "00"), ("20230115", "12"),
    ("20230715", "00"), ("20230715", "12"),
]
BG_LEAD = 24


def spectrum2d(field_2d, mask=None):
    """二维水平功率谱（径向平均），复用 step21 的实现。"""
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


def moments(k, power):
    """M0 = ∫E dk, M2 = ∫k²E dk（与 step26 同口径）。"""
    k = np.asarray(k)
    p = np.asarray(power)
    dk = np.gradient(k)
    M0 = float(np.sum(p * dk))
    M2 = float(np.sum(k ** 2 * p * dk))
    return M0, M2


def boot_ci_paired(vals, n=10000, seed=11):
    """配对 bootstrap：对差值序列的中位数给 95% CI。"""
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if vals.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n, vals.size))
    bs = np.median(vals[idx], axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    t0 = time.time()
    print("=" * 98)
    print("step31  P2：M2 判别力的置信区间 + 跨季节稳健性（8 配对）")
    print("=" * 98)

    # 预加载所有 case（真值+背景+观测），按 target 分组
    cases = {}
    for tspec in PAIRS:
        try:
            cases[tspec] = S12.load_case(tspec, lead=BG_LEAD)
            print("载入 %s%s" % (tspec[0], tspec[1]), flush=True)
        except Exception as e:
            print("载入 %s%s 失败: %s" % (tspec[0], tspec[1], str(e)[:90]))

    rows = []   # 每个 (target, var) 的 dM0/dM2 相对降低
    for target in list(cases):
        X = cases[target]
        lat, lon, lev = X["lat"], X["lon"], X["lev"]
        others = [t for t in cases if t != target]
        print("\n目标 %s%s  集合来源 %d 个其它时次" %
              (target[0], target[1], len(others)), flush=True)

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
            print("  成员不足，跳过")
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

        above = X["above"]
        cruise = ((lev[:, None, None] >= CRUISE[0])
                  & (lev[:, None, None] <= CRUISE[1])) & above
        i250 = int(np.argmin(np.abs(lev - 250.0)))
        mask2d = above[i250] & cruise[i250]

        for v in KEYS:
            eb = X["bg"][v][i250] - X["nat"][v][i250]
            ea = an[v][i250] - X["nat"][v][i250]
            esbg = smbg[v][i250] - X["nat"][v][i250]
            m = np.isfinite(eb) & np.isfinite(ea)
            kb, pb = spectrum2d(eb, m)
            _, pa = spectrum2d(ea, m)
            _, psbg = spectrum2d(esbg, m)
            M0_bg, M2_bg = moments(kb, pb)
            M0_an, M2_an = moments(kb, pa)
            M0_sbg, M2_sbg = moments(kb, psbg)
            rows.append(dict(
                target="%s%s" % (target[0], target[1]), var=v,
                dM0_an=(M0_bg - M0_an) / M0_bg if M0_bg > 0 else 0,
                dM0_sbg=(M0_bg - M0_sbg) / M0_bg if M0_bg > 0 else 0,
                dM2_an=(M2_bg - M2_an) / M2_bg if M2_bg > 0 else 0,
                dM2_sbg=(M2_bg - M2_sbg) / M2_bg if M2_bg > 0 else 0))
        del X, an, smbg
        gc.collect()

    # ---- 汇总 + 判别力 + bootstrap CI ----
    print("\n" + "=" * 98)
    print("【汇总】M0 与 M2 判别力（%d 个 算例×变量）" % len(rows))
    print("=" * 98)

    def med(key):
        v = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    dM0_an = med("dM0_an")
    dM0_sbg = med("dM0_sbg")
    dM2_an = med("dM2_an")
    dM2_sbg = med("dM2_sbg")
    print("  M0（场误差/RMSE² 代理）：同化 %+6.1f%%   平滑 %+6.1f%%"
          % (100 * dM0_an, 100 * dM0_sbg))
    print("  M2（诊断误差代理）：    同化 %+6.1f%%   平滑 %+6.1f%%"
          % (100 * dM2_an, 100 * dM2_sbg))

    # 判别力 = 平滑 − 同化（每个 row 内部先算差值，再中位 + bootstrap）
    disc_M0 = [r["dM0_sbg"] - r["dM0_an"] for r in rows]
    disc_M2 = [r["dM2_sbg"] - r["dM2_an"] for r in rows]
    m0_disc = float(np.median(disc_M0))
    m2_disc = float(np.median(disc_M2))
    ci_m0 = boot_ci_paired(disc_M0)
    ci_m2 = boot_ci_paired(disc_M2)
    print("\n  M0 判别力（平滑−同化）= %+6.1f%%   95%%CI [%+6.1f%%, %+6.1f%%]"
          % (100 * m0_disc, 100 * ci_m0[0], 100 * ci_m0[1]))
    print("  M2 判别力（平滑−同化）= %+6.1f%%   95%%CI [%+6.1f%%, %+6.1f%%]"
          % (100 * m2_disc, 100 * ci_m2[0], 100 * ci_m2[1]))

    # 判别力之差的显著性：Δ = M2判别力 − M0判别力
    disc_diff = [r2 - r1 for r1, r2 in zip(disc_M0, disc_M2)]
    dd_med = float(np.median(disc_diff))
    ci_dd = boot_ci_paired(disc_diff)
    print("  Δ判别力（M2−M0）      = %+6.1f%%   95%%CI [%+6.1f%%, %+6.1f%%]"
          % (100 * dd_med, 100 * ci_dd[0], 100 * ci_dd[1]))
    sig = "显著（CI 不含 0）" if ci_dd[0] > 0 or ci_dd[1] < 0 else "不显著（CI 含 0）"
    print("  ⇒ M2 相对 M0 的判别力优势 %s" % sig)

    # ---- 跨季节/跨年稳健性：按季节分组 ----
    print("\n" + "=" * 98)
    print("【稳健性】按季节/年份分组（判别力 = 平滑−同化 的中位）")
    print("=" * 98)
    def season_of(target):
        mm = target[4:6]
        return "冬(1月)" if mm == "01" else "夏(7月)"
    def year_of(target):
        return target[:4]
    groups = {}
    for r in rows:
        g = (year_of(r["target"]), season_of(r["target"]))
        groups.setdefault(g, []).append(r)
    print("  %-16s %6s %10s %10s" % ("分组", "n", "M0判别力", "M2判别力"))
    for g in sorted(groups):
        rs = groups[g]
        d0 = float(np.median([r["dM0_sbg"] - r["dM0_an"] for r in rs]))
        d2 = float(np.median([r["dM2_sbg"] - r["dM2_an"] for r in rs]))
        print("  %s/%s  %6d %+9.1f%% %+9.1f%%"
              % (g[0], g[1], len(rs), 100 * d0, 100 * d2))

    out = dict(meta=dict(pairs=PAIRS, bg_lead=BG_LEAD, n_rows=len(rows),
                         note="M0/M2 判别力 + 配对 bootstrap 95%CI"),
               rows=rows,
               summary=dict(
                   dM0_an=dM0_an, dM0_sbg=dM0_sbg,
                   dM2_an=dM2_an, dM2_sbg=dM2_sbg,
                   disc_M0=m0_disc, disc_M0_ci=ci_m0,
                   disc_M2=m2_disc, disc_M2_ci=ci_m2,
                   disc_diff=dd_med, disc_diff_ci=ci_dd,
                   significant=bool(ci_dd[0] > 0 or ci_dd[1] < 0)))
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
