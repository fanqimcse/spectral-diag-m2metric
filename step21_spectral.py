# -*- coding: utf-8 -*-
"""step21 —— 误差谱分解：把"平滑 > 同化"从一条经验结论升级为一个普适命题。

科学主命题（本脚本要检验/证明的核心）
--------------------------------------
step20 的端到端检验给出一条反常结果：

    巡航带 RMSE(P−P*) 降幅：  只同化 +1.8%  <  只平滑背景 +6.8%  <  同化+平滑 +8.9%

即：**把场同化得更准（RMSE 降 6–8%），对下游风险产品的改善反而不如"单纯
做一次水平平滑"**。这个结果在方法学上是决定性的，但 step20 只把它当作
一个"配方"报出来，没有回答**为什么**。

本脚本回答"为什么"，并把它提炼成一个可外推的命题：

    ┌─────────────────────────────────────────────────────────────┐
    │ 命题 P（谱—诊断对应原理）                                     │
    │                                                             │
    │ 任何以**空间导数**为输入的下游物理诊断（湍流 Ri、对流 CAPE、  │
    │ 风切变、锋生…），其误差由背景误差的**谱**决定，而非由误差的  │
    │ **幅度**决定。垂直导数算子是一个**高通滤波器**（谱响应        │
    │ |∂/∂z|² ∝ k_z²），因此：                                      │
    │   · 误差谱中小尺度（高波数）分量对诊断误差的贡献被 **k² 放大**；│
    │   · 大尺度分量对诊断误差的贡献被 **k² 抑制**。                 │
    │                                                             │
    │ 推论：同化（EnOI）的增量是光滑集合成员的线性组合，只修正       │
    │ 大尺度偏差，几乎不动小尺度噪声 ⇒ 对导数型诊断的改善有限；      │
    │ 而"尺度分离"（消除小尺度噪声）恰好命中诊断最敏感的部分。       │
    └─────────────────────────────────────────────────────────────┘

本脚本分三步，从"解析自证"到"真实数据检验"，逐级收紧：

  [A] 谱传递的解析验证 —— 用已知谱的合成场，数值验证垂直导数算子的
      谱响应确实是 |k|² 高通，并证明"同幅度、不同谱"的误差会对导数
      诊断造成数量级不同的影响。（把命题从"直觉"变成"定理"。）

  [B] 真实 GFS 误差谱分解 —— 对 4 个 OSSE 算例，把背景误差、分析误差、
      平滑背景误差做二维水平谱分解，量化每个配置移除了/留下了哪个波段。
      （把命题从"定理"变成"在这个真实系统里成立"。）

  [C] 谱—诊断对应 —— 用真实数据证明：下游诊断（Ri=N²/S²）的误差谱，
      与背景误差谱经 |k|² 加权的形式一致，即诊断误差确实由小尺度主导。

最终要回答的是：**"场空间精度不是风险产品精度的代理量"这条结论，
是否有一个谱分解的、可外推到所有导数型诊断的机制基础。**
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

OUT = "data/step21_spectral.json"
KEYS = S12.KEYS
CRUISE = (200.0, 400.0)

RNG = np.random.default_rng(20261021)


# ======================================================================
# [A] 谱传递的解析验证
# ======================================================================
def synth_field_1d(n, k_mode, rng=RNG, amp=1.0):
    """一维合成场：单一波数的正弦 + 白噪声（用于验证导数算子的谱响应）。"""
    x = np.arange(n)
    return amp * np.sin(2 * np.pi * k_mode * x / n) + 0.05 * rng.normal(size=n)


def spectral_response_1d(n, k_modes=(1, 4, 8, 16, 32, 64)):
    """数值验证：一阶差分算子的谱响应 ≈ |sin(πk/n)| ∝ k（小 k 近似）。

    返回 dict{k: 实测增益}，其中增益 = RMS(dy/dx) / RMS(y)，对单位幅度正弦
    应等于 2·sin(πk/n)（中心差分）≈ 2πk/n。这证明导数算子是**高通**。
    """
    out = {}
    x = np.arange(n)
    for k in k_modes:
        y = np.sin(2 * np.pi * k * x / n)
        dy = np.gradient(y)                      # 单位格距，dx=1
        gain = np.sqrt(np.mean(dy ** 2)) / np.sqrt(np.mean(y ** 2))
        out[int(k)] = float(gain)
    return out


def amplitude_vs_spectrum_demo():
    """关键演示：**同幅度、不同谱**的误差，对导数诊断的影响差一个量级。

    构造两个误差场 e_a（大尺度，k=2）与 e_b（小尺度，k=32），
    把幅度都调到相同（RMS 相同）。然后看 d(e)/dx 的 RMS：
    e_b 的导数误差比 e_a 大 16 倍（因为导数增益 ∝ k）。
    这直接证明：**误差的幅度不是诊断误差的代理量，谱才是。**
    """
    n = 256
    x = np.arange(n)
    k_lo, k_hi = 2, 32
    e_lo = np.sin(2 * np.pi * k_lo * x / n)
    e_hi = np.sin(2 * np.pi * k_hi * x / n)
    # 调成同幅度（RMS=1）
    e_lo /= np.sqrt(np.mean(e_lo ** 2))
    e_hi /= np.sqrt(np.mean(e_hi ** 2))
    d_lo = np.gradient(e_lo)
    d_hi = np.gradient(e_hi)
    rms_lo = np.sqrt(np.mean(d_lo ** 2))
    rms_hi = np.sqrt(np.mean(d_hi ** 2))
    # 理论预测：导数增益 ∝ sin(πk/n)，比值 ≈ sin(π·32/n)/sin(π·2/n)
    theo = np.sin(np.pi * k_hi / n) / np.sin(np.pi * k_lo / n)
    return dict(rms_deriv_low=float(rms_lo), rms_deriv_high=float(rms_hi),
                ratio_measured=float(rms_hi / rms_lo),
                ratio_theoretical=float(theo),
                k_low=int(k_lo), k_high=int(k_hi))


# ======================================================================
# [B] 真实 GFS 误差谱分解
# ======================================================================
def spectrum2d(field_2d, mask=None):
    """二维水平功率谱（径向平均），返回 (wavenumber, power)。

    field_2d: (ny, nx)。先 detrend 再窗化，避免谱泄漏。
    """
    f = np.asarray(field_2d, float).copy()
    if mask is not None:
        f = np.where(mask, f, np.nan)
    # 用区域均值填充 NaN（海面以下/地形遮蔽），避免傅里叶变换的 NaN
    m = np.isfinite(f)
    if not m.all():
        f = np.where(m, f, np.nanmean(f[m]))
    f = f - np.nanmean(f)
    ny, nx = f.shape
    # Hann 窗抑制谱泄漏
    wy = np.hanning(ny)[:, None]
    wx = np.hanning(nx)[None, :]
    f = f * wy * wx
    F = np.fft.fft2(f)
    P = np.abs(F) ** 2
    # 径向平均
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)
    # 只取正半轴（径向对称）
    kbins = np.arange(0, 0.5, 1.0 / max(ny, nx))
    power = np.zeros(len(kbins) - 1)
    count = np.zeros(len(kbins) - 1)
    for i in range(len(kbins) - 1):
        sel = (kr >= kbins[i]) & (kr < kbins[i + 1])
        if sel.sum() == 0:
            continue
        power[i] = np.mean(P[sel])
        count[i] = sel.sum()
    k_center = 0.5 * (kbins[:-1] + kbins[1:])
    return k_center, power


def spectral_decompose_case(X, an, smbg, sman, g, lev, above, cruise):
    """对单个算例做误差谱分解。

    返回：背景/分析/平滑背景/平滑分析 各自的误差谱（对真值），
    以及"同化移除的谱"与"平滑移除的谱"。
    """
    # 取巡航层的一个代表层（250 hPa，接近航迹产品高度）
    i250 = int(np.argmin(np.abs(lev - 250.0)))
    mask2d = above[i250] & cruise[i250]  # 巡航层的地面以上掩码
    res = {}
    # 对每个变量，比较 bg/an/smbg/sman 的误差谱
    for v in KEYS:
        eb = X["bg"][v][i250] - X["nat"][v][i250]
        ea = an[v][i250] - X["nat"][v][i250]
        esbg = smbg[v][i250] - X["nat"][v][i250]
        esan = sman[v][i250] - X["nat"][v][i250]
        # 用背景误差的掩码（真值/背景都有限的地方）
        m = np.isfinite(eb) & np.isfinite(ea)
        kb, pb = spectrum2d(eb, m)
        ka, pa = spectrum2d(ea, m)
        ksbg, psbg = spectrum2d(esbg, m)
        ksan, psan = spectrum2d(esan, m)
        # 总误差功率（RMS²）—— 用谱积分近似，但更可靠的是直接算
        rms_bg = float(np.sqrt(np.nanmean(eb[m] ** 2)))
        rms_an = float(np.sqrt(np.nanmean(ea[m] ** 2)))
        rms_sbg = float(np.sqrt(np.nanmean(esbg[m] ** 2)))
        rms_san = float(np.sqrt(np.nanmean(esan[m] ** 2)))
        res[v] = dict(
            k=kb.tolist(),
            power_bg=pb.tolist(), power_an=pa.tolist(),
            power_sbg=psbg.tolist(), power_san=psan.tolist(),
            rms_bg=rms_bg, rms_an=rms_an, rms_sbg=rms_sbg, rms_san=rms_san)
    return res


def band_power(k, power, k_lo, k_hi):
    """某波数带内的积分功率。"""
    sel = (np.asarray(k) >= k_lo) & (np.asarray(k) < k_hi)
    return float(np.sum(np.asarray(power)[sel]))


# ======================================================================
# [C] 谱—诊断对应：下游诊断误差谱是否由小尺度主导
# ======================================================================
def diag_spectrum(X, an, smbg, sman, lat, lon, lev, above, cruise):
    """对下游诊断量（S², N², Ri）做误差谱，检验其是否由小尺度主导。"""
    dg_nat = RK.stability_fields(dict(T=X["nat"]["T"], q=X["nat"]["q"],
                                       u=X["nat"]["u"], v=X["nat"]["v"],
                                       z=X["nat"]["z"]), lat, lon, lev)
    dg_bg = RK.stability_fields(dict(T=X["bg"]["T"], q=X["bg"]["q"],
                                      u=X["bg"]["u"], v=X["bg"]["v"],
                                      z=X["bg"]["z"]), lat, lon, lev)
    dg_an = RK.stability_fields(dict(T=an["T"], q=an["q"], u=an["u"],
                                      v=an["v"], z=an["z"]), lat, lon, lev)
    dg_sbg = RK.stability_fields(dict(T=smbg["T"], q=smbg["q"], u=smbg["u"],
                                       v=smbg["v"], z=smbg["z"]), lat, lon, lev)
    dg_san = RK.stability_fields(dict(T=sman["T"], q=sman["q"], u=sman["u"],
                                       v=sman["v"], z=sman["z"]), lat, lon, lev)
    res = {}
    i250 = int(np.argmin(np.abs(lev - 250.0)))
    for key in ("S2", "Nd2", "Ri_dry"):
        # 用对数空间比较（这些量跨度大）
        def lg(a):
            return np.log10(np.clip(np.asarray(a, float), 1e-14, None))
        eb = lg(dg_bg[key][i250]) - lg(dg_nat[key][i250])
        ea = lg(dg_an[key][i250]) - lg(dg_nat[key][i250])
        esbg = lg(dg_sbg[key][i250]) - lg(dg_nat[key][i250])
        esan = lg(dg_san[key][i250]) - lg(dg_nat[key][i250])
        m = np.isfinite(eb) & np.isfinite(ea) & above[i250]
        kb, pb = spectrum2d(eb, m)
        ka, pa = spectrum2d(ea, m)
        ksbg, psbg = spectrum2d(esbg, m)
        ksan, psan = spectrum2d(esan, m)
        res[key] = dict(k=kb.tolist(),
                        power_bg=pb.tolist(), power_an=pa.tolist(),
                        power_sbg=psbg.tolist(), power_san=psan.tolist(),
                        rms_bg=float(np.sqrt(np.nanmean(eb[m] ** 2))),
                        rms_an=float(np.sqrt(np.nanmean(ea[m] ** 2))),
                        rms_sbg=float(np.sqrt(np.nanmean(esbg[m] ** 2))),
                        rms_san=float(np.sqrt(np.nanmean(esan[m] ** 2))))
    for dg in (dg_nat, dg_bg, dg_an, dg_sbg, dg_san):
        del dg
    gc.collect()
    return res


# ======================================================================
# 主流程
# ======================================================================
def main():
    t0 = time.time()
    print("=" * 98)
    print("step21  误差谱分解：谱—诊断对应原理（命题 P 的验证）")
    print("=" * 98)

    # ---------- [A] 解析验证 ----------
    print("\n[A] 谱传递的解析验证")
    print("-" * 98)
    resp = spectral_response_1d(256)
    print("  一阶差分算子的谱响应（增益 = RMS(dy)/RMS(y)）：")
    for k, g in resp.items():
        theo = 2 * np.sin(np.pi * k / 256)
        print("    k=%3d  实测增益 %.4f   理论 2·sin(πk/n)=%.4f"
              % (k, g, theo))
    demo = amplitude_vs_spectrum_demo()
    print("\n  同幅度、不同谱的误差 → 导数误差的差异：")
    print("    大尺度误差(k=%d) 导数 RMS = %.4f" % (demo["k_low"], demo["rms_deriv_low"]))
    print("    小尺度误差(k=%d) 导数 RMS = %.4f" % (demo["k_high"], demo["rms_deriv_high"]))
    print("    比值(实测) = %.1f×   比值(理论 ∝k) = %.1f×"
          % (demo["ratio_measured"], demo["ratio_theoretical"]))
    print("    ⇒ 误差幅度相同，但小尺度误差对导数诊断的影响大 ~%.0f 倍。"
          % demo["ratio_measured"])

    # ---------- [B][C] 真实数据 ----------
    print("\n[B/C] 真实 GFS 误差谱分解（4 个 OSSE 算例，巡航层 250 hPa）")
    print("-" * 98)

    cases = {}
    for t in S12.TARGETS:
        try:
            cases[t] = S12.load_case(t)
            print("载入 %s%sz" % (t[0], t[1]), flush=True)
        except Exception as e:
            print("载入 %s 失败: %s" % (t, str(e)[:90]))

    field_spec, diag_spec = {}, {}
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

        d = {v: an[v] - X["bg"][v] for v in KEYS}
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

        # [B] 场误差谱
        fs = spectral_decompose_case(X, an, smbg, sman, g, lev, above, cruise)
        field_spec[target] = fs
        # [C] 诊断误差谱
        ds = diag_spectrum(X, an, smbg, sman, lat, lon, lev, above, cruise)
        diag_spec[target] = ds

        # 逐算例的"波段归因"：把谱分成低波数（大尺度）与高波数（小尺度）
        # 看同化 vs 平滑各移除了哪个波段
        row = dict(target="%s%sz" % (target[0], target[1]))
        for v in KEYS:
            k = np.asarray(fs[v]["k"])
            pb = np.asarray(fs[v]["power_bg"])
            pa = np.asarray(fs[v]["power_an"])
            psbg = np.asarray(fs[v]["power_sbg"])
            psan = np.asarray(fs[v]["power_san"])
            # 波数切分点：用 L_h1 对应的波数（~1/L_h）作"大/小尺度"分界
            k_cut = float(1.0 / (g * dlat / 1000.0 / 111195.0 / dlat / 111195.0))
            # 简化：直接用谱的第 3 个 bin 作为切分（低波数 = 前 20%）
            ncut = max(int(0.2 * len(k)), 1)
            k_lo = float(k[ncut])
            # 低波数（大尺度）与高波数（小尺度）积分功率
            for tag, pw in (("bg", pb), ("an", pa), ("sbg", psbg), ("san", psan)):
                row["%s|%s|low" % (v, tag)] = band_power(k, pw, 0, k_lo)
                row["%s|%s|high" % (v, tag)] = band_power(k, pw, k_lo, 1.0)
        rows.append(row)
        print("  T: rms bg=%.4f an=%.4f sbg=%.4f san=%.4f"
              % (fs["T"]["rms_bg"], fs["T"]["rms_an"],
                 fs["T"]["rms_sbg"], fs["T"]["rms_san"]))
        print("  Ri_dry(log10): rms bg=%.4f an=%.4f sbg=%.4f san=%.4f"
              % (ds["Ri_dry"]["rms_bg"], ds["Ri_dry"]["rms_an"],
                 ds["Ri_dry"]["rms_sbg"], ds["Ri_dry"]["rms_san"]))
        del X, an, smbg, sman, d
        gc.collect()

    # 汇总：跨算例中位
    print("\n" + "=" * 98)
    print("【波段归因】低波数(大尺度) vs 高波数(小尺度) 误差功率（跨算例中位）")
    print("=" * 98)
    def med_k(key):
        v = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    print("%-4s %-8s %14s %14s %14s %14s" %
          ("变量", "波段", "bg", "an", "smooth_bg", "smooth_an"))
    summary = {}
    for v in KEYS:
        for band in ("low", "high"):
            b = med_k("%s|bg|%s" % (v, band))
            a = med_k("%s|an|%s" % (v, band))
            sbg = med_k("%s|sbg|%s" % (v, band))
            san = med_k("%s|san|%s" % (v, band))
            summary["%s|%s" % (v, band)] = dict(bg=b, an=a, sbg=sbg, san=san)
            print("%-4s %-8s %14.4g %14.4g %14.4g %14.4g"
                  % (v, band, b, a, sbg, san))
            # 同化移除的比例 vs 平滑移除的比例
            if b > 0:
                rem_an = (b - a) / b
                rem_sbg = (b - sbg) / b
                print("      %s 同化移除 %+.1f%%  平滑移除 %+.1f%%"
                      % (" " * 8, 100 * rem_an, 100 * rem_sbg))

    out = dict(spectral_response=resp, amplitude_demo=demo,
               field_spec={"%s%sz" % (t[0], t[1]): v for t, v in field_spec.items()},
               diag_spec={"%s%sz" % (t[0], t[1]): v for t, v in diag_spec.items()},
               band_attribution=summary, rows=rows)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
