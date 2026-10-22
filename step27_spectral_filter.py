# -*- coding: utf-8 -*-
"""step27 —— 谱滤波算子：把"高斯平滑"升级为波数空间显式谱滤波。

背景与科学问题
--------------
step20/21 证明"尺度分离（平滑）> 同化"对导数型诊断，但其尺度分离用的是
**高斯平滑** gaussian_filter(sigma=L_h)。高斯平滑有两个固有缺陷：

  1. **渐近衰减**：高斯核在波数空间也是高斯，它对**所有**波数都有非零
     衰减，没有锐利的截止。因此它在消除小尺度噪声的同时，也**泄漏**地
     衰减了一部分大尺度信号（大尺度被轻微破坏）。
  2. **单一参数**：sigma 只能控制"整体平滑程度"，无法精确指定"保留
     哪个波段、砍掉哪个波段"。

本脚本把尺度分离升级为**波数空间的显式谱滤波**：

  · 低通（low-pass）：保留 k < k_cut 的分量，砍掉 k > k_cut 的噪声。
  · 带通（band-pass）：只保留 [k_lo, k_hi] 的中间波段（用于提取诊断
    最敏感的波段）。

并检验一个可证伪的命题：

  **命题（谱滤波优于高斯平滑）**：对导数型下游诊断，波数空间的锐利低通
  滤波比同截止尺度的高斯平滑能**更干净**地分离大/小尺度，因此对诊断
  误差的降低**不小于**高斯平滑，且对"场本身"的大尺度信号破坏更小。

设计（自包含 + 真实 GFS）
--------------------------
  [A] 解析对比：在理想高斯谱误差场上，比较"锐利低通"与"高斯平滑"两种
      尺度分离对 (M0, M2) 的影响，证明锐利低通在**同样的截止波数**下
      对 M2 的降低更大、对 M0 的大尺度破坏更小。

  [B] 真实 GFS：对 4 个算例，用谱滤波（低通）替代高斯平滑，重新评估
      下游诊断 Ri_dry 的误差降低，与 step21 的高斯平滑结果对比。

  [C] 管线定义：给出业务可用的"同化 + 谱滤波"管线的伪代码与参数约定
      （截止波数的物理含义、二维谱滤波的实现、区域边缘处理）。
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
from avwx import risk as RK
from avwx import enoi as EN

OUT = "data/step27_spectral_filter.json"
KEYS = S12.KEYS
CRUISE = (200.0, 400.0)


# ======================================================================
# 谱滤波核心算子
# ======================================================================
def spectral_lowpass(field_2d, k_cut, mask=None):
    """二维波数空间锐利低通滤波。

    field_2d: (ny, nx)。k_cut: 归一化频率截止（0~0.5，每格距周数）。
    保留 |k| < k_cut 的分量，其余置零。
    mask: 可选的地形/海洋掩码，用于把 NaN 区域填充后再变换。
    """
    f = np.asarray(field_2d, float).copy()
    if mask is not None:
        m = np.isfinite(f) & mask
        if not m.all():
            f = np.where(m, f, np.nanmean(f[np.isfinite(f)]))
    else:
        m = np.isfinite(f)
        if not m.all():
            f = np.where(m, f, np.nanmean(f[m]))
    ny, nx = f.shape
    # 去均值（避免直流分量泄漏）
    f = f - np.nanmean(f)
    F = np.fft.fft2(f)
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX ** 2 + KY ** 2)
    # 锐利低通：|k| < k_cut 保留
    H = (kr < k_cut).astype(float)
    out = np.fft.ifft2(F * H).real
    # 恢复均值
    out = out + np.nanmean(np.asarray(field_2d, float))
    return out


def gaussian_lowpass(field_2d, sigma_px):
    """高斯平滑（对照），sigma_px 为像素单位。"""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(field_2d, sigma=sigma_px, mode="nearest")


def gaussian_sigma_to_k_cut(sigma_px):
    """高斯平滑 sigma（像素）对应的等效截止波数（归一化频率）。

    高斯核在波数空间的功率谱为 exp(-(2πσk)²)，其半功率点
    exp(-(2πσk_half)²) = 1/2 → k_half = sqrt(ln2)/(2πσ)。
    用这个半功率波数作为"等效截止"，使高斯平滑与锐利低通可比。
    """
    return np.sqrt(np.log(2.0)) / (2 * np.pi * sigma_px)


# ======================================================================
# [A] 解析对比：锐利低通 vs 高斯平滑（同截止波数）
# ======================================================================
def analytic_sharp_vs_gaussian():
    """在理想高斯谱误差场上，比较锐利低通与高斯平滑。

    设背景误差谱 E(k)=exp(-(k/k0)²)（特征波数 k0）。
    锐利低通：E_sharp(k) = E(k)·1[k < k_cut]
    高斯平滑：E_gauss(k) = E(k)·exp(-(2πσk)²)，σ 取等效截止 k_cut。

    比较两者对 M0（场）与 M2（诊断）的降低，以及"大尺度信号保留率"
    （k < k_cut/2 部分的保留程度，衡量对有效信号的无谓破坏）。
    """
    k = np.linspace(0.001, 2.0, 100000)
    dk = np.gradient(k)
    k0 = 0.1
    Eb = np.exp(-(k / k0) ** 2)

    def moments(E):
        return (np.sum(E * dk), np.sum(k ** 2 * E * dk))

    M0_b, M2_b = moments(Eb)
    rows = []
    for k_cut in (0.05, 0.1, 0.15, 0.2, 0.3):
        # 锐利低通
        E_sharp = Eb * (k < k_cut).astype(float)
        M0_s, M2_s = moments(E_sharp)
        # 高斯平滑（等效截止）
        sigma = np.sqrt(np.log(2.0)) / (2 * np.pi * k_cut)
        E_gauss = Eb * np.exp(-(2 * np.pi * sigma * k) ** 2)
        M0_g, M2_g = moments(E_gauss)
        # 大尺度信号保留率（k < k_cut/2 的功率占比）
        keep_sharp = np.sum((Eb * (k < k_cut / 2).astype(float)) * dk) / M0_b
        keep_gauss = np.sum((Eb * np.exp(-(2 * np.pi * sigma * k) ** 2)
                             * (k < k_cut / 2).astype(float)) * dk) / M0_b
        rows.append(dict(
            k_cut=float(k_cut), sigma=float(sigma),
            dM0_sharp=(M0_b - M0_s) / M0_b, dM2_sharp=(M2_b - M2_s) / M2_b,
            dM0_gauss=(M0_b - M0_g) / M0_b, dM2_gauss=(M2_b - M2_g) / M2_b,
            keep_sharp=float(keep_sharp), keep_gauss=float(keep_gauss)))
    return rows


# ======================================================================
# [B] 真实 GFS：谱滤波 vs 高斯平滑
# ======================================================================
def main():
    t0 = time.time()
    print("=" * 98)
    print("step27  谱滤波算子：锐利低通 vs 高斯平滑")
    print("=" * 98)

    # ---------- [A] 解析对比 ----------
    print("\n[A] 解析对比（高斯谱误差，同截止波数）")
    print("-" * 98)
    rows_a = analytic_sharp_vs_gaussian()
    print("  %-8s | %-24s | %-24s | %-16s" %
          ("k_cut", "锐利低通 M0/M2降%", "高斯平滑 M0/M2降%", "大尺度保留 锐利/高斯"))
    for r in rows_a:
        print("  %-8.2f | %6.1f%% / %6.1f%% | %6.1f%% / %6.1f%% | %5.1f%% / %5.1f%%"
              % (r["k_cut"], 100 * r["dM0_sharp"], 100 * r["dM2_sharp"],
                 100 * r["dM0_gauss"], 100 * r["dM2_gauss"],
                 100 * r["keep_sharp"], 100 * r["keep_gauss"]))
    # 判断：锐利低通是否对 M2 降低更多、对大尺度保留更多
    sharp_better = all(r["dM2_sharp"] >= r["dM2_gauss"] for r in rows_a)
    sharp_keep = all(r["keep_sharp"] >= r["keep_gauss"] for r in rows_a)
    print("  ⇒ 锐利低通对 M2 的降低 ≥ 高斯平滑：%s" % ("是" if sharp_better else "否"))
    print("  ⇒ 锐利低通对大尺度信号保留 ≥ 高斯平滑：%s" % ("是" if sharp_keep else "否"))

    # ---------- [B] 真实 GFS ----------
    print("\n[B] 真实 GFS：谱滤波（低通）替代高斯平滑（4 个算例，巡航带）")
    print("-" * 98)

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

        # 高斯平滑的 sigma（沿用 step20/21 的 L_h1 换算）
        dlat = float(np.mean(np.diff(lat))) * 111195.0
        g_px = max(float(X["dg"].get("L_h1", 124e3)) / dlat, 0.5)
        # 谱滤波的截止波数：用与高斯平滑 sigma **严格对应**的等效截止。
        # 高斯核功率谱 exp(-(2πσk)²)，半功率点 k_half = sqrt(ln2)/(2πσ)。
        # 用这个半功率波数作为锐利低通的截止，两者才可比。
        k_cut = gaussian_sigma_to_k_cut(g_px)

        # 对 4 个变量的背景场做谱滤波（低通）与分析场做谱滤波
        above = X["above"]
        sf_bg = {}
        sf_an = {}
        for v in KEYS:
            sf_bg[v] = np.empty_like(X["bg"][v])
            sf_an[v] = np.empty_like(an[v])
            for i in range(len(lev)):
                sf_bg[v][i] = spectral_lowpass(X["bg"][v][i], k_cut,
                                               mask=above[i])
                sf_an[v][i] = spectral_lowpass(an[v][i], k_cut,
                                               mask=above[i])
        sf_bg["z"] = X["bg"]["z"]
        sf_an["z"] = X["bg"]["z"]

        # 高斯平滑场（对照，复用 step21 口径）
        from scipy.ndimage import gaussian_filter
        gb = {v: gaussian_filter(X["bg"][v], sigma=(0, g_px, g_px),
                                 mode="nearest") for v in KEYS}
        gb["z"] = X["bg"]["z"]

        # 评估下游诊断 Ri_dry 的误差（对数空间）
        def diag(fld):
            return RK.stability_fields(dict(T=fld["T"], q=fld["q"],
                                            u=fld["u"], v=fld["v"],
                                            z=fld["z"]), lat, lon, lev)
        dg_nat = diag(X["nat"])
        cruise = ((lev[:, None, None] >= CRUISE[0])
                  & (lev[:, None, None] <= CRUISE[1])) & above

        def rmse_ri(fld):
            d = diag(fld)
            e = np.log10(np.clip(d["Ri_dry"], 1e-14, None)) - \
                np.log10(np.clip(dg_nat["Ri_dry"], 1e-14, None))
            m = cruise & np.isfinite(e)
            return float(np.sqrt(np.nanmean(e[m] ** 2)))

        rb = rmse_ri(X["bg"])
        r_an = rmse_ri(an)
        r_gauss = rmse_ri(gb)
        r_sharp = rmse_ri(sf_bg)
        r_sharp_an = rmse_ri(sf_an)

        row = dict(target="%s%sz" % (target[0], target[1]),
                   d_an=100 * (rb - r_an) / rb,
                   d_gauss=100 * (rb - r_gauss) / rb,
                   d_sharp=100 * (rb - r_sharp) / rb,
                   d_sharp_an=100 * (rb - r_sharp_an) / rb,
                   k_cut=k_cut, g_px=g_px)
        rows.append(row)
        print("  Ri_dry 降幅%%: 同化 %+5.1f | 高斯平滑 %+5.1f | "
              "谱滤波 %+5.1f | 同化+谱滤波 %+5.1f  (k_cut=%.4f)"
              % (row["d_an"], row["d_gauss"], row["d_sharp"],
                 row["d_sharp_an"], k_cut))

        del X, an, gb, sf_bg, sf_an, dg_nat
        gc.collect()

    # 汇总
    def med(k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    print("\n" + "=" * 98)
    print("【汇总】跨 %d 算例中位：Ri_dry 误差降幅%%" % len(rows))
    print("=" * 98)
    print("  只同化       : %+5.1f%%" % med("d_an"))
    print("  高斯平滑     : %+5.1f%%" % med("d_gauss"))
    print("  谱滤波(低通) : %+5.1f%%" % med("d_sharp"))
    print("  同化+谱滤波  : %+5.1f%%" % med("d_sharp_an"))
    print("  ⇒ 谱滤波 %s 高斯平滑（对诊断误差的降低）"
          % ("优于" if med("d_sharp") > med("d_gauss") else "不优于"))

    out = dict(analytic=rows_a, rows=rows,
               median=dict(d_an=med("d_an"), d_gauss=med("d_gauss"),
                           d_sharp=med("d_sharp"), d_sharp_an=med("d_sharp_an")))
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
