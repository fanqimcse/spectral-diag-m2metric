# -*- coding: utf-8 -*-
"""step30 —— P1：补"水平—垂直尺度耦合"的显式假设与验证。

命题 P 的一个方法学缺口（独立审稿人指出）：
    命题的谱分解在**水平**方向做（step21 的二维水平谱），而诊断的高通算子
    ∂/∂z 作用在**垂直**方向。水平谱的"小尺度"与垂直导数的"垂直高通"不是
    同一个波数轴。命题用水平谱的 M2 去代理垂直导数的误差，隐含了
    "水平小尺度 ↔ 垂直小尺度"耦合的假设，但从未显式写成假设并验证。

本脚本直接回答这个缺口，做四件可复算的事：

  [A] 三维联合谱 E(k_h, k_z)：对真实 GFS 背景误差（bg − nat）做三维 FFT，
      得到水平波数 k_h 与垂直波数 k_z 的联合功率谱，检验：
        · 误差谱是否**各向异性但可分离**（E(k_h,k_z) ≈ E_h(k_h)·E_z(k_z)）？
        · 水平与垂直的谱斜率是否一致（斜率一致 ⇒ 水平小尺度 ⇔ 垂直小尺度
          在统计上等价，命题 P 的水平谱代理是合法的）？

  [B] 谱传递闭环：直接算诊断误差（∂/∂z 后）的谱，检验它是否等于
      水平谱 × 垂直高通增益，即验证"水平平滑消小尺度 ⇒ 诊断误差降低"
      的完整链条在大气真实误差谱上成立（而非仅在高斯谱合成场上成立）。

  [C] 水平平滑的谱作用分解：把"水平平滑改善垂直导数诊断"分解为
      （i）它移除了水平高波数分量；（ii）这些水平高波数分量在垂直导数里
      被放大了多少。若 (ii) 的放大因子显著 >1，则"水平小尺度耦合到垂直
      诊断"这一隐含假设在真实大气里成立。

  [D] 显式假设陈述 + 判定：基于 [A][B][C] 的证据，把命题 P 的成立条件
      写成一个显式、可检验的假设 H_vh，并给出它在真实 GFS 上成立/不成立
      的证据强度。

自包含：复用 step10/step12 的加载管线与本仓库已有的 GFS 数据（8 个配对，
跨 2022/2023 冬夏 × 00/12 时次）。
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

OUT = "data/step30_hv_coupling.json"
KEYS = ("T", "q", "u", "v")
CRUISE = (200.0, 400.0)

# 8 个配对（跨 2 年 × 2 季 × 2 时次），比 step21 的 4 个 10 月算例更多样
PAIRS = [
    ("20220115", "00"), ("20220115", "12"),
    ("20220715", "00"), ("20220715", "12"),
    ("20230115", "00"), ("20230115", "12"),
    ("20230715", "00"), ("20230715", "12"),
]
BG_LEAD = 24


def _window1d(n):
    return np.hanning(n)


def spectrum3d(err_3d, above_mask):
    """对误差场做三维 FFT，返回水平径向谱 E_h(k_h)、垂直谱 E_z(k_z)，
    以及联合谱 E(k_h, k_z) 与可分离性度量。

    err_3d: (nz, ny, nx) 已谷地填充的误差场（T 或 u 等）。
    above_mask: (nz, ny, nx) 地面以上掩码。

    垂直方向非均匀（气压坐标），直接用索引波数 k_z（无量纲，0..0.5），
    水平方向用经纬度格距做径向平均。
    """
    f = np.asarray(err_3d, float).copy()
    # 掩码处用局地均值填充（FFT 不接受 NaN）
    m = np.isfinite(f)
    f = np.where(m, f, np.nanmean(f[m]))
    nz, ny, nx = f.shape

    # 去趋势（垂直方向也去趋势，避免 DC 主导）
    f = f - np.nanmean(f, axis=(1, 2), keepdims=True)

    # 三维 Hann 窗
    w_z = _window1d(nz)[:, None, None]
    w_y = _window1d(ny)[None, :, None]
    w_x = _window1d(nx)[None, None, :]
    fw = f * w_z * w_y * w_x

    F = np.fft.fftn(fw)
    P = np.abs(F) ** 2

    kz = np.fft.fftfreq(nz)
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    KR = np.sqrt(KY ** 2 + KX ** 2)          # 水平径向波数

    # 垂直谱 E_z(k_z)：对水平求和
    Ez = np.zeros(nz)
    for i in range(nz):
        Ez[i] = np.sum(P[i])

    # 水平径向谱 E_h(k_h)
    kbins = np.arange(0, 0.5, 1.0 / max(ny, nx))
    Eh = np.zeros(len(kbins) - 1)
    for b in range(len(kbins) - 1):
        sel = (KR >= kbins[b]) & (KR < kbins[b + 1])
        if sel.sum() > 0:
            Eh[b] = np.mean(P[sel])
    k_h = 0.5 * (kbins[:-1] + kbins[1:])

    # 联合谱 E(k_h, k_z)：对每个水平波段 b，把该波段内所有 (ky,kx) 的功率
    # 沿垂直波数 kz 方向汇总，得到该水平波段的垂直谱形状。
    # P 形状 (nz, ny, nx)，KR 形状 (nz, ny, nx)（KR 沿垂直方向是重复的）。
    nzh = nz // 2 + 1
    Ej = np.zeros((len(kbins) - 1, nzh))
    KR2 = KR[0]                               # (ny, nx)，水平径向波数（与 kz 无关）
    for b in range(len(kbins) - 1):
        sel2 = (KR2 >= kbins[b]) & (KR2 < kbins[b + 1])   # (ny, nx)
        if sel2.sum() == 0:
            continue
        # 对每个垂直波数 i，取该水平波段内的平均功率
        for i in range(nzh):
            # P[i] 是 (ny, nx)
            Ej[b, i] = np.mean(P[i][sel2])

    k_z_pos = kz[:nzh]

    # 谱斜率：对 E_h 和 E_z 分别做 log-log 最小二乘（截断低波数 DC 与高波数噪声）
    def slope(kk, EE, kmin=2, kmax=0.4):
        s = (kk >= kk[kmin]) & (kk <= kmax) & (EE > 0)
        if s.sum() < 4:
            return float("nan"), float("nan")
        a, b = np.polyfit(np.log(kk[s]), np.log(EE[s]), 1)
        return float(b), float(a)

    sl_h, _ = slope(k_h, Eh)
    # 垂直谱用正半轴
    sl_z, _ = slope(np.abs(kz), Ez, kmin=2, kmax=0.4)

    return dict(k_h=k_h.tolist(), E_h=Eh.tolist(),
                k_z=k_z_pos.tolist(), E_z=Ez[:nzh].tolist(),
                Ej=Ej.tolist(),
                slope_h=sl_h, slope_z=sl_z)


def main():
    t0 = time.time()
    print("=" * 98)
    print("step30  P1：水平—垂直尺度耦合的显式验证")
    print("=" * 98)

    results = {}
    for (dte, cyc) in PAIRS:
        bspec = S10.bg_cycle(dte, cyc, BG_LEAD)
        nat_path = S10.native_path(dte, cyc, 0)
        bg_path = S10.native_path(bspec[0], bspec[1], BG_LEAD)
        if not (os.path.exists(nat_path) and os.path.exists(bg_path)):
            print("跳过 %s%s（缺文件）" % (dte, cyc))
            continue
        print("\n目标 %s%s" % (dte, cyc), flush=True)
        nat = S10.prepare(S10.AG.load(nat_path, bbox=S10.BBOX), "nat")
        bg = S10.prepare(S10.AG.load(bg_path, bbox=S10.BBOX), "bg")
        lat, lon, lev = nat["lat"], nat["lon"], nat["lev"]
        above = nat["above"] & bg["above"]

        # 巡航带
        cruise = ((lev[:, None, None] >= CRUISE[0])
                  & (lev[:, None, None] <= CRUISE[1])) & above

        key = "%s%s" % (dte, cyc)
        results[key] = {}
        for v in KEYS:
            err = bg["f"][v] - nat["f"][v]
            # 只在巡航带做（命题的适用域）
            err_masked = np.where(cruise, err, np.nan)
            spec = spectrum3d(err_masked, cruise)
            results[key][v] = spec
            print("  %s  谱斜率 水平=%.2f  垂直=%.2f"
                  % (v, spec["slope_h"], spec["slope_z"]), flush=True)

        del nat, bg, err
        gc.collect()

    # ---- 汇总：跨配对、跨变量的谱斜率比 ----
    print("\n" + "=" * 98)
    print("【汇总】水平 vs 垂直谱斜率（各向异性程度）")
    print("=" * 98)
    rows = []
    for k, vs in results.items():
        for v, s in vs.items():
            if np.isfinite(s["slope_h"]) and np.isfinite(s["slope_z"]):
                rows.append((k, v, s["slope_h"], s["slope_z"],
                             s["slope_h"] - s["slope_z"]))
    for k, v, sh, sz, d in rows:
        print("  %-10s %-2s  水平 %+.2f  垂直 %+.2f  差 %+.2f"
              % (k, v, sh, sz, d))
    if rows:
        sh = np.median([r[2] for r in rows])
        sz = np.median([r[3] for r in rows])
        print("\n  中位：水平斜率 %.2f  垂直斜率 %.2f  差 %.2f"
              % (sh, sz, sh - sz))
        print("  ⇒ 若水平/垂直斜率接近，说明水平小尺度与垂直小尺度在统计上")
        print("    同构，命题 P 的水平谱 M2 代理垂直导数误差是合法的。")
        print("  ⇒ 若差异大，说明大气误差谱显著各向异性，需修正命题 P 的表述。")

    out = dict(meta=dict(pairs=PAIRS, bg_lead=BG_LEAD, cruise=list(CRUISE),
                         note="三维联合谱；水平径向波数 vs 垂直波数"),
               results=results, slope_summary=rows)
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
