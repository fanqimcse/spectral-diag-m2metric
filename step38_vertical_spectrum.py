# -*- coding: utf-8 -*-
"""step38 —— 垂直谱 M2 直接验证命题 P′（"垂直导数诊断误差由垂直谱高波数主导"）。

========================================================================
背景与要回答的问题
========================================================================

命题 P（原始）：诊断误差 ∝ 误差谱二阶矩 M2 = ∫k²E(k)dk，用**水平**二维谱的 M2
代理垂直导数 ∂/∂z 的误差。step30b 发现这是一个**不完整代理**——垂直平滑能消除的
水平小尺度分量只占诊断误差的一部分，并据此把命题修正为：

  命题 P′：导数型诊断的误差由误差谱在**导数所取方向**上的高波数分量主导。
  垂直导数诊断（Ri_dry = N²/S²）的误差由**垂直**谱的高波数主导；
  水平平滑只能消除其中与水平小尺度耦合的部分，无法消除纯垂直方向的小尺度噪声。

但 P′ 从未被**直接验证**——step30b 是通过"水平平滑消不掉诊断误差"间接推断的，
从未实际计算过**垂直波数谱**并检验其与诊断误差的绑定关系。

本脚本补上这块：对真实 GFS 背景误差场，同时计算
  (i)  水平谱二阶矩 M2_h（逐层，口径与 step31 一致）
  (ii) 垂直谱二阶矩 M2_v（逐格点列，沿等压面方向做 DCT）
并检验哪一个对 Ri_dry 诊断误差的空间分布有更强的解释力。

========================================================================
检验设计（可证伪的预言）
========================================================================

设 Ri_dry 诊断误差（log 空间，逐格点）为 Y。
  H0（水平谱假设）：Y 主要由水平小尺度误差驱动 → Y ~ M2_h 解释力更强；
  H1（垂直谱假设，即 P′）：Y 主要由垂直小尺度误差驱动 → Y ~ M2_v 解释力更强。

用一元回归的决定系数 R² 比较两个预测器的解释力。命题 P′ 成立 ⟺ R²(M2_v) > R²(M2_h)。

========================================================================
口径（务必与项目其它步骤一致）
========================================================================

  - 水平谱 M2_h：复用 m2metric.spectrum2d（汉宁窗、径向平均、归一化波数），
    对每个垂直层单独算，得 (nlev,) 的 M2_h，再广播到格点。
  - 垂直谱 M2_v：对每个格点列 (lat, lon)，沿 lev 轴做 DCT-II（正交、无泄漏、
    适合非周期垂直廓线），取 k_z 为归一化频率，M2_v = ∫k_z² E(k_z) dk_z。
    用物理层厚（dlnp 或 dz）修正非均匀层距的权重。
  - 诊断误差：Ri_dry 的 log 误差 |Δ log10(Ri)|，巡航带 200–400 hPa。
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
import step12_enoi as S12
import step36_operator_sufficient_condition as S36
from m2metric import spectrum2d as m2_spectrum2d
from scipy.fft import dct

OUT = "data/step38_vertical_spectrum.json"
KEYS = tuple(S12.KEYS) + ("z",)
CRUISE = (200.0, 400.0)


# ======================================================================
# 垂直谱：沿等压面方向的 DCT 谱矩
# ======================================================================
def vertical_m2(field, lev):
    """对 (nlev, nlat, nlon) 场，沿 lev 轴逐格点列做 DCT，返回逐列 M2_v。

    返回 (nlat, nlon) 的 M2_v 分布。波数用归一化频率（每层距周数）。
    用物理层厚（dlnp）作权重，修正等压面非均匀间距（高空的层更厚）。
    """
    f = np.asarray(field, float)
    nlev, nlat, nlon = f.shape
    # 逐列去均值（DCT 的 DC 分量本就单独，去均值可让误差场谱聚焦于扰动）
    f = f - np.nanmean(f, axis=0, keepdims=True)
    f = np.where(np.isfinite(f), f, 0.0)

    # DCT-II（ortho 归一化），沿 axis=0
    F = dct(f, type=2, axis=0, norm="ortho")
    P = np.abs(F) ** 2                      # (nlev, nlat, nlon)

    # 归一化波数 k_z：DCT-II 的第 j 个分量对应 k = j / (2*(nlev-1))（每层距周数）
    kz = np.arange(nlev, dtype=float) / (2.0 * (nlev - 1))

    # 层厚权重：用 dlnp 近似（等压面层越往上越稀疏）
    p = np.asarray(lev, float)
    lnp = np.log(p)
    dlnp = np.abs(np.gradient(lnp))
    w = dlnp / dlnp.sum()                   # 归一化

    # M2_v = Σ_j k_z(j)² · P(j) · w(j)，对每个格点列
    M2_v = np.sum((kz[:, None, None] ** 2) * P * w[:, None, None], axis=0)
    return M2_v


def vertical_m0(field, lev):
    """同 vertical_m2，但零阶矩（场误差能量的垂直谱代理）。"""
    f = np.asarray(field, float)
    nlev = f.shape[0]
    f = f - np.nanmean(f, axis=0, keepdims=True)
    f = np.where(np.isfinite(f), f, 0.0)
    F = dct(f, type=2, axis=0, norm="ortho")
    P = np.abs(F) ** 2
    p = np.asarray(lev, float)
    w = np.abs(np.gradient(np.log(p)))
    w = w / w.sum()
    return np.sum(P * w[:, None, None], axis=0)


# ======================================================================
# 水平谱：逐层的 2D 谱矩（口径与 m2metric 一致）
# ======================================================================
def horizontal_m2_per_level(field):
    """对 (nlev, nlat, nlon) 场，逐层做 2D 径向谱，返回逐层 M2_h（长度 nlev）。"""
    nlev = field.shape[0]
    m2 = np.full(nlev, np.nan)
    for k in range(nlev):
        f = np.asarray(field[k], float)
        if not np.isfinite(f).any():
            continue
        kk, pp = m2_spectrum2d(f)
        sel = np.isfinite(kk) & np.isfinite(pp) & (kk > 0)
        if sel.sum() < 3:
            continue
        dk = np.gradient(kk[sel])
        m2[k] = float(np.sum((kk[sel] ** 2) * pp[sel] * dk))
    return m2


# ======================================================================
# 主流程
# ======================================================================
def main():
    t0 = time.time()
    print("=" * 98)
    print("step38  垂直谱 M2 直接验证命题 P′")
    print("=" * 98)

    # ---- 收集 4 个 GFS 时次的逐格点样本 ----
    # 对每个时次：诊断误差 Y（巡航带逐格点）、M2_h（逐层广播）、M2_v（逐列广播）
    all_Y = []
    all_M2h = []
    all_M2v = []
    all_M0v = []
    per_case = []

    for tgt in S12.TARGETS:
        X = S12.load_case(tgt)
        lat, lon, lev = X["lat"], X["lon"], X["lev"]
        above = X["above"]
        cruise = ((lev[:, None, None] >= CRUISE[0])
                  & (lev[:, None, None] <= CRUISE[1])) & above
        truth = X["nat"]
        bg = X["bg"]

        # ---- 诊断误差 Y = |Δ log10(Ri_dry)|，巡航带 ----
        dg_t = S36.diag(truth, lat, lon, lev)
        dg_b = S36.diag(bg, lat, lon, lev)
        a = np.log10(np.clip(np.asarray(dg_b["Ri_dry"], float), 1e-14, None))
        t = np.log10(np.clip(np.asarray(dg_t["Ri_dry"], float), 1e-14, None))
        Y = np.abs(a - t)                   # (nlev, nlat, nlon)

        # ---- 误差场的水平/垂直谱矩（对 4 个变量取中位，与 rel_field_error 口径一致）----
        m2h_per_var = []
        m2v_per_var = []
        m0v_per_var = []
        for v in ("T", "q", "u", "v"):
            e = bg[v] - truth[v]            # 背景误差场
            m2h_per_var.append(horizontal_m2_per_level(e))
            m2v_per_var.append(vertical_m2(e, lev))
            m0v_per_var.append(vertical_m0(e, lev))
        # 逐层 M2_h：跨变量取中位
        M2h = np.nanmedian(np.stack(m2h_per_var), axis=0)   # (nlev,)
        # 逐列 M2_v / M0_v：跨变量取中位
        M2v = np.nanmedian(np.stack(m2v_per_var), axis=0)   # (nlat, nlon)
        M0v = np.nanmedian(np.stack(m0v_per_var), axis=0)   # (nlat, nlon)

        # ---- 广播到格点（巡航带内）----
        sel = cruise & np.isfinite(Y)
        # M2_h 逐层广播到格点
        M2h_3d = np.broadcast_to(M2h[:, None, None], Y.shape)
        # M2_v 逐列广播到所有层
        M2v_3d = np.broadcast_to(M2v[None, :, :], Y.shape)
        M0v_3d = np.broadcast_to(M0v[None, :, :], Y.shape)

        all_Y.append(Y[sel])
        all_M2h.append(M2h_3d[sel])
        all_M2v.append(M2v_3d[sel])
        all_M0v.append(M0v_3d[sel])

        # 逐时次的相关系数
        def corr(a, b):
            a = np.asarray(a, float); b = np.asarray(b, float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10:
                return float("nan")
            a, b = a[m], b[m]
            if a.std() < 1e-12 or b.std() < 1e-12:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        rh = corr(all_Y[-1], all_M2h[-1])
        rv = corr(all_Y[-1], all_M2v[-1])
        per_case.append(dict(target="%s%sz" % (tgt[0], tgt[1]),
                             r_Y_M2h=rh, r_Y_M2v=rv,
                             r2_h=rh ** 2, r2_v=rv ** 2))
        print("  %s%sz  r(Y,M2_h)=%+.3f  r(Y,M2_v)=%+.3f  → %s"
              % (tgt[0], tgt[1], rh, rv,
                 "垂直谱更强" if rv > rh else "水平谱更强"))
        gc.collect()

    # ---- 汇总（合并所有时次的逐格点样本）----
    Y = np.concatenate(all_Y)
    M2h = np.concatenate(all_M2h)
    M2v = np.concatenate(all_M2v)
    M0v = np.concatenate(all_M0v)

    def corr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        return float(np.corrcoef(a, b)[0, 1])

    r_h = corr(Y, M2h)
    r_v = corr(Y, M2v)
    r0_v = corr(Y, M0v)

    # 用对数谱矩（谱矩跨量级，log 更稳健）
    r_h_log = corr(Y, np.log10(np.clip(M2h, 1e-30, None)))
    r_v_log = corr(Y, np.log10(np.clip(M2v, 1e-30, None)))

    print("\n" + "=" * 98)
    print("汇总（%d 个逐格点样本，4 时次合并）" % len(Y))
    print("-" * 98)
    print("  诊断误差 Y = |Δ log10(Ri_dry)| 巡航带")
    print("  Pearson r(Y, M2_h 水平谱二阶矩) = %+.4f  (R²=%.4f)"
          % (r_h, r_h ** 2))
    print("  Pearson r(Y, M2_v 垂直谱二阶矩) = %+.4f  (R²=%.4f)"
          % (r_v, r_v ** 2))
    print("  Pearson r(Y, M0_v 垂直谱零阶矩) = %+.4f  (R²=%.4f)"
          % (r0_v, r0_v ** 2))
    print("  Pearson r(Y, log M2_h)         = %+.4f" % r_h_log)
    print("  Pearson r(Y, log M2_v)         = %+.4f" % r_v_log)
    print("  ⇒ 命题 P′（垂直谱主导）成立：%s"
          % ("是" if r_v > r_h else "否"))

    out = dict(
        n_samples=int(len(Y)),
        n_times=len(per_case),
        per_case=per_case,
        pooled=dict(r_Y_M2h=r_h, r_Y_M2v=r_v, r_Y_M0v=r0_v,
                    r2_h=r_h ** 2, r2_v=r_v ** 2,
                    r_log_h=r_h_log, r_log_v=r_v_log),
        p_prime_holds=bool(r_v > r_h),
    )
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n完成，用时 %.0f s → %s" % (time.time() - t0, OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
