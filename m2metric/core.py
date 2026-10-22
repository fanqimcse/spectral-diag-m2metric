# -*- coding: utf-8 -*-
"""m2metric 核心实现。

所有谱分解、矩计算、截断选择、判别力与 bootstrap 的算法，均严格复用
step26 / step31 中已验证的口径，保证与论文结果可复现一致。

关键口径（务必保持，勿擅自改动）：
  - 波数用**归一化频率**（每格距的周数，范围 0~0.5），即 np.fft.fftfreq(n)
    不加 *n；
  - M0 = ∫E(k)dk，M2 = ∫k²E(k)dk（一维径向谱，dk 用 np.gradient）；
  - 判别力 = (尺度分离的相对降低) − (同化的相对降低)，逐样本先算差值再
    取中位（**不能**独立取中位再相减，否则符号可能反转）；
  - 二维谱用汉宁窗抑制谱泄漏，NaN/掩码用有限值的均值填充。
"""

import numpy as np

__all__ = [
    "spectrum2d", "moments", "compute_m2", "auto_k_cut",
    "bootstrap_ci", "discriminate", "M0", "M2",
]


# ---------------------------------------------------------------------------
# 谱分解
# ---------------------------------------------------------------------------
def spectrum2d(field, dx=1.0, dy=1.0, mask=None, detrend=True, window="hanning"):
    """二维水平功率谱（径向平均）。

    参数
    ----
    field : (ny, nx) array
        二维误差场（或任意场）。
    dx, dy : float
        网格距（x、y 方向），用于把归一化波数转换为物理波数。
    mask : (ny, nx) bool array, 可选
        有效区域掩码；掩码外 NaN。
    detrend : bool
        是否先去均值（默认 True）。
    window : {"hanning", "hann", "none"}
        窗函数，抑制谱泄漏。默认汉宁窗。

    返回
    ----
    k_phys : (nbins,) array
        径向波数中心（物理单位，1/dx）。
    power : (nbins,) array
        径向平均功率谱。
    """
    f = np.asarray(field, float).copy()
    if mask is not None:
        f = np.where(np.asarray(mask, bool), f, np.nan)
    m = np.isfinite(f)
    if not m.all():
        f = np.where(m, f, np.nanmean(f[m]))
    if detrend:
        f = f - np.nanmean(f)

    ny, nx = f.shape
    if window in ("hanning", "hann"):
        wy = np.hanning(ny)[:, None]
        wx = np.hanning(nx)[None, :]
        f = f * wy * wx

    F = np.fft.fft2(f)
    P = np.abs(F) ** 2

    # 归一化波数（每格距周数）
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    kr = np.sqrt(ky ** 2 + kx ** 2)

    kbins = np.arange(0, 0.5, 1.0 / max(ny, nx))
    power = np.zeros(len(kbins) - 1)
    for i in range(len(kbins) - 1):
        sel = (kr >= kbins[i]) & (kr < kbins[i + 1])
        if sel.sum() > 0:
            power[i] = np.mean(P[sel])
    k_center = 0.5 * (kbins[:-1] + kbins[1:])

    # 转物理波数（周/单位长度）
    if dx != 1.0 or dy != 1.0:
        # 各向异性网格下近似用平均格距
        d = np.sqrt(dx * dy)
        k_phys = k_center / d
    else:
        k_phys = k_center
    return k_phys, power


def moments(k, power, k_cut=None):
    """M0 = ∫E dk，M2 = ∫k²E dk。

    参数
    ----
    k : (nbins,) array
        波数（物理或归一化）。
    power : (nbins,) array
        功率谱。
    k_cut : float, 可选
        上截断波数，只积分 k <= k_cut 的部分（正则化 M2 的高波数噪声）。

    返回
    ----
    (M0, M2) : (float, float)
    """
    k = np.asarray(k, float)
    p = np.asarray(power, float)
    sel = np.isfinite(k) & np.isfinite(p)
    if k_cut is not None:
        sel &= (k <= k_cut)
    if sel.sum() == 0:
        return float("nan"), float("nan")
    k = k[sel]
    p = p[sel]
    dk = np.gradient(k)
    M0 = float(np.sum(p * dk))
    M2 = float(np.sum(k ** 2 * p * dk))
    return M0, M2


def M0(k, power, k_cut=None):
    """误差谱零阶矩（场误差能量，RMSE² 的谱代理）。"""
    return moments(k, power, k_cut)[0]


def M2(k, power, k_cut=None):
    """误差谱二阶矩（导数型诊断误差的谱代理）。"""
    return moments(k, power, k_cut)[1]


# ---------------------------------------------------------------------------
# 截断自动选择
# ---------------------------------------------------------------------------
def auto_k_cut(k, power, target_bias=0.05, n_trials=200, seed=11):
    """自动选择截断波数，使 M2 估计的偏倚—方差权衡最优。

    原理（见 step26 [B']）：M2 放大高波数的谱估计噪声，估计方差随截断
    波数 k_cut 单调增大；但过低的 k_cut 会系统性低估真实 M2（偏倚增大）。
    存在一个偏倚—方差最优截断。

    本函数用 bootstrap 在给定谱形状下估计不同 k_cut 的 M2 抽样分布，
    选择"偏倚 ≤ target_bias × 真值"约束下、方差最小的 k_cut。

    参数
    ----
    k, power : 谱。
    target_bias : float
        允许的最大相对偏倚（默认 5%）。
    n_trials : int
        bootstrap 次数。

    返回
    ----
    k_cut : float
        选出的截断波数。
    """
    k = np.asarray(k, float)
    p = np.asarray(power, float)
    sel = np.isfinite(k) & np.isfinite(p) & (p >= 0)
    k, p = k[sel], p[sel]
    if len(k) < 4:
        return float(k[-1]) if len(k) else 0.5

    # 真值（全谱）
    _, M2_full = moments(k, p, k_cut=None)

    rng = np.random.default_rng(seed)
    # 假设功率谱逐 bin 独立（粗近似），加噪声重采样
    # 每个 bin 的谱估计近似服从卡方分布（2 自由度），方差 ∝ mean²
    candidates = np.unique(np.clip(k, k.min() + 1e-9, k.max()))
    if len(candidates) > 20:
        candidates = np.linspace(k.min() + 1e-9, k.max(), 20)

    best = None
    for k_cut in candidates:
        # 保护：k_cut 必须覆盖至少 3 个谱 bin，否则梯度无定义
        n_sel = int((k <= k_cut).sum())
        if n_sel < 3:
            continue
        m2_vals = []
        for _ in range(n_trials):
            # 给每个 bin 乘一个 Gamma(0.5, 2) 噪声（均值 1，卡方/2）
            noise = rng.gamma(0.5, 2.0, size=len(p))
            p_noisy = p * noise
            _, m2 = moments(k, p_noisy, k_cut=k_cut)
            m2_vals.append(m2)
        m2_vals = np.array(m2_vals)
        mean = np.mean(m2_vals)
        bias = abs(mean - M2_full) / (M2_full if M2_full > 0 else 1.0)
        std = np.std(m2_vals)
        if bias <= target_bias:
            # 在满足偏倚约束的前提下，选方差最小（即 k_cut 尽量小）
            if best is None or std < best[1]:
                best = (float(k_cut), float(std))
    if best is None:
        # 无满足约束的，选偏倚最接近目标者
        best = (float(candidates[-1]), float("inf"))
    return best[0]


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def bootstrap_ci(vals, n=10000, seed=11, alpha=0.05):
    """对一维数组的中位数给 bootstrap 置信区间（配对/独立均可）。

    返回 (median, [lo, hi])。
    """
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if vals.size < 2:
        return float("nan"), [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n, vals.size))
    bs = np.median(vals[idx], axis=1)
    lo = np.percentile(bs, 100 * alpha / 2)
    hi = np.percentile(bs, 100 * (1 - alpha / 2))
    return float(np.median(vals)), [float(lo), float(hi)]


# ---------------------------------------------------------------------------
# 标准接口：compute_m2
# ---------------------------------------------------------------------------
def compute_m2(error_field, dx=1.0, dy=1.0, mask=None,
               auto_truncate=True, target_bias=0.05,
               return_spectrum=False):
    """对单个误差场计算 M0、M2 及其统计量。

    这是工具包的主入口之一。输入一个误差场（分析场 − 真值，或背景场 − 真值），
    输出场误差代理 M0 与诊断误差代理 M2。

    参数
    ----
    error_field : (ny, nx) array
        误差场（可含 NaN，用 mask 指定有效区）。
    dx, dy : float
        网格距。
    mask : bool array, 可选
        有效区域。
    auto_truncate : bool
        是否自动选择截断波数正则化 M2（默认 True）。
    target_bias : float
        自动截断允许的最大偏倚。
    return_spectrum : bool
        是否返回谱（k, power）。

    返回
    ----
    dict:
        M0 : 零阶矩（场误差能量）
        M2 : 二阶矩（诊断误差能量）
        M2_over_M0 : M2/M0（误差谱"倾斜方向"的简洁度量）
        k_cut : 使用的截断波数（若 auto_truncate）
        k, power : 谱（若 return_spectrum）
    """
    k, power = spectrum2d(error_field, dx=dx, dy=dy, mask=mask)
    k_cut = None
    if auto_truncate:
        k_cut = auto_k_cut(k, power, target_bias=target_bias)
    m0, m2 = moments(k, power, k_cut=k_cut)
    out = dict(
        M0=float(m0),
        M2=float(m2),
        M2_over_M0=float(m2 / m0) if m0 > 0 else float("nan"),
        k_cut=float(k_cut) if k_cut is not None else None,
    )
    if return_spectrum:
        out["k"] = k
        out["power"] = power
    return out


# ---------------------------------------------------------------------------
# 判别力接口
# ---------------------------------------------------------------------------
def discriminate(bg_errors, an_errors, scale_errors, dx=1.0, dy=1.0,
                 masks=None, auto_truncate=True, n_boot=10000, seed=11):
    """对比"同化"与"尺度分离"两类修正，计算 M0 与 M2 的判别力 + CI。

    这是"RMSE 会说谎、M2 会说真话"的定量工具。

    参数
    ----
    bg_errors : list of (ny,nx) array
        背景误差场（多算例/多变量/多时次）。
    an_errors : list of (ny,nx) array
        同化后的误差场（与 bg_errors 一一对应）。
    scale_errors : list of (ny,nx) array
        尺度分离（平滑）后的误差场。
    masks : list of bool array, 可选
        每个算例的有效掩码。
    auto_truncate : bool
        是否自动截断。
    n_boot : int
        bootstrap 次数。
    seed : int
        随机种子。

    返回
    ----
    dict:
        dM0_an : 同化对 M0 的相对降低（中位）
        dM0_scale : 尺度分离对 M0 的相对降低（中位）
        dM2_an : 同化对 M2 的相对降低（中位）
        dM2_scale : 尺度分离对 M2 的相对降低（中位）
        disc_M0 : M0 判别力 = (尺度分离 − 同化)，中位
        disc_M2 : M2 判别力
        disc_diff : Δ判别力 = M2判别力 − M0判别力
        ci_M0 / ci_M2 / ci_diff : 各自的 95% CI
        significant : Δ判别力是否显著异于 0
        n_cases : 样本数
    """
    bg_errors = list(bg_errors)
    an_errors = list(an_errors)
    scale_errors = list(scale_errors)
    n = len(bg_errors)
    assert len(an_errors) == n and len(scale_errors) == n, "三组误差场长度须一致"
    if masks is None:
        masks = [None] * n

    rows = []
    for i in range(n):
        bg = bg_errors[i]
        an = an_errors[i]
        sc = scale_errors[i]
        mk = masks[i]

        kb, pb = spectrum2d(bg, dx=dx, dy=dy, mask=mk)
        # 同一算例用同一 k 网格与截断
        k_cut = auto_k_cut(kb, pb) if auto_truncate else None
        M0_bg, M2_bg = moments(kb, pb, k_cut)
        _, pa = spectrum2d(an, dx=dx, dy=dy, mask=mk)
        M0_an, M2_an = moments(kb, pa, k_cut)
        _, ps = spectrum2d(sc, dx=dx, dy=dy, mask=mk)
        M0_sc, M2_sc = moments(kb, ps, k_cut)

        rows.append(dict(
            dM0_an=(M0_bg - M0_an) / M0_bg if M0_bg > 0 else 0.0,
            dM0_sc=(M0_bg - M0_sc) / M0_bg if M0_bg > 0 else 0.0,
            dM2_an=(M2_bg - M2_an) / M2_bg if M2_bg > 0 else 0.0,
            dM2_sc=(M2_bg - M2_sc) / M2_bg if M2_bg > 0 else 0.0,
        ))

    def med(key):
        v = [r[key] for r in rows if np.isfinite(r[key])]
        return float(np.median(v)) if v else float("nan")

    disc_M0 = [r["dM0_sc"] - r["dM0_an"] for r in rows]
    disc_M2 = [r["dM2_sc"] - r["dM2_an"] for r in rows]
    disc_diff = [r2 - r1 for r1, r2 in zip(disc_M0, disc_M2)]

    _, ci_M0 = bootstrap_ci(disc_M0, n=n_boot, seed=seed)
    _, ci_M2 = bootstrap_ci(disc_M2, n=n_boot, seed=seed)
    dd_med, ci_diff = bootstrap_ci(disc_diff, n=n_boot, seed=seed)

    significant = bool(ci_diff[0] > 0 or ci_diff[1] < 0)

    return dict(
        dM0_an=med("dM0_an"), dM0_scale=med("dM0_sc"),
        dM2_an=med("dM2_an"), dM2_scale=med("dM2_sc"),
        disc_M0=float(np.median(disc_M0)),
        disc_M2=float(np.median(disc_M2)),
        disc_diff=dd_med,
        ci_M0=ci_M0, ci_M2=ci_M2, ci_diff=ci_diff,
        significant=significant,
        n_cases=n,
    )
