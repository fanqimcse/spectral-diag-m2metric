# -*- coding: utf-8 -*-
"""avwx/verify —— 预报技巧检验的**唯一实现**。

为什么要抽出来
--------------
本项目先用 PIREP（step08）、再用 EDR（step11b）做同一件事：比较两个预报量的
判别能力。若两套脚本各写一份 AUC / bootstrap / 插值，哪怕只差一行，两套结论
就无法放在一起比较 —— 这是比任何统计方法更致命的"实现口径漂移"。

因此这里集中提供：双线性取样、按气压垂插、AUC、配对 bootstrap ΔAUC、以及
连续标签的 Spearman 相关。step08 与 step11b 都从这里取。

约定
----
* AUC 采用等级（rank）定义，等价于 Mann–Whitney U：AUC = U / (n_pos·n_neg)，
  并列取平均秩，因此**天然处理了 EDR 里大量的并列值（0）**。
* ΔAUC 一律用**配对** bootstrap：同一批样本重抽，保留两个预测量之间的相关性，
  否则 CI 会被样本间协方差撑大好几倍。
"""
import numpy as np


# ---------------------------------------------------------------- 空间取样
def bilinear_cols(field, lat1d, lon1d, qlat, qlon):
    """把 (nz,ny,nx) 场双线性插到散点，返回 (npts, nz)。

    要求 lat1d 升序、lon1d 升序（aws_gfs.load 已保证）。
    """
    yi = np.clip(np.searchsorted(lat1d, qlat) - 1, 0, len(lat1d) - 2)
    xi = np.clip(np.searchsorted(lon1d, qlon) - 1, 0, len(lon1d) - 2)
    ty = (qlat - lat1d[yi]) / np.maximum(lat1d[yi + 1] - lat1d[yi], 1e-9)
    tx = (qlon - lon1d[xi]) / np.maximum(lon1d[xi + 1] - lon1d[xi], 1e-9)
    ty = np.clip(ty, 0, 1)
    tx = np.clip(tx, 0, 1)
    nz = field.shape[0]
    out = np.empty((len(qlat), nz))
    for k in range(nz):
        f0 = field[k]
        out[:, k] = ((1 - ty) * (1 - tx) * f0[yi, xi] +
                     (1 - ty) * tx * f0[yi, xi + 1] +
                     ty * (1 - tx) * f0[yi + 1, xi] +
                     ty * tx * f0[yi + 1, xi + 1])
    return out


def vert_interp(col, lev_desc, ptar):
    """按气压垂插。col: (npts,nz)，列顺序与 lev_desc（降序）一致。"""
    xp = np.asarray(lev_desc, float)[::-1]          # 升序
    ptar = np.asarray(ptar, float)
    out = np.empty(len(ptar))
    for i in range(len(ptar)):
        out[i] = np.interp(ptar[i], xp, col[i][::-1])
    return out


# ---------------------------------------------------------------- 判别指标
def _ranks(s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    ss = np.asarray(s)[order]
    i = 0
    while i < len(ss):                              # 并列取平均秩
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def auc_score(score, label):
    """AUC = P(score_pos > score_neg) + 0.5 P(=)，即 Mann–Whitney U 归一化。"""
    m = np.isfinite(score) & np.isfinite(label)
    s = np.asarray(score)[m]
    y = np.asarray(label)[m].astype(int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _ranks(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def bootstrap_dauc(sa, sb, y, n=2000, seed=20261021):
    """配对 bootstrap：sa 的 AUC 减 sb 的 AUC 的抽样分布。"""
    rng = np.random.default_rng(seed)
    m = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(y)
    sa, sb, yy = np.asarray(sa)[m], np.asarray(sb)[m], np.asarray(y)[m].astype(int)
    N = len(yy)
    if N < 10:
        return np.full(n, np.nan)
    out = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, N, N)
        yy2 = yy[idx]
        if yy2.sum() == 0 or (1 - yy2).sum() == 0:
            out[b] = np.nan
            continue
        out[b] = auc_score(sa[idx], yy2) - auc_score(sb[idx], yy2)
    return out


def bootstrap_stat(sa, sb, y, n=2000, seed=4242):
    """配对 bootstrap 任意 (sa,sb,y) 之差的抽样分布（对 Spearman 用）。"""
    rng = np.random.default_rng(seed)
    m = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(y)
    sa, sb, yy = np.asarray(sa)[m], np.asarray(sb)[m], np.asarray(y)[m]
    N = len(yy)
    out = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, N, N)
        x, z, w = sa[idx], sb[idx], yy[idx]
        out[b] = _spearman(x, w) - _spearman(z, w)
    return out


def _spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x, y = x[m], y[m]
    rx, ry = _ranks(x), _ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if den < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / den)


def spearman(x, y):
    """Spearman 秩相关。用于**连续**标签（EDR）——不需要人为阈值。"""
    return _spearman(x, y)


def ci_of(bs):
    bs = np.asarray([v for v in bs if np.isfinite(v)], float)
    if bs.size < 20:
        return [float("nan"), float("nan")]
    return [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


# ---------------------------------------------------------------- 自检
def selfcheck():
    out = []

    # 1. 完美判别 -> AUC = 1；完全相反 -> 0
    y = np.array([0, 0, 0, 1, 1, 1])
    out.append(("auc_perfect", abs(auc_score(np.array([1., 2, 3, 4, 5, 6]), y) - 1) < 1e-9,
                auc_score(np.array([1., 2, 3, 4, 5, 6]), y)))
    out.append(("auc_reversed", abs(auc_score(np.array([6., 5, 4, 3, 2, 1]), y)) < 1e-9,
                auc_score(np.array([6., 5, 4, 3, 2, 1]), y)))

    # 2. 随机 -> 0.5（大样本）
    rng = np.random.default_rng(7)
    yy = (rng.random(4000) < 0.3).astype(int)
    ss = rng.random(4000)
    a = auc_score(ss, yy)
    out.append(("auc_random_half", abs(a - 0.5) < 0.03, a))

    # 3. 并列：所有 score 相同 -> AUC = 0.5
    out.append(("auc_all_ties", abs(auc_score(np.ones(20), np.r_[np.ones(10), np.zeros(10)]) - 0.5) < 1e-9,
                auc_score(np.ones(20), np.r_[np.ones(10), np.zeros(10)])))

    # 4. 双线性：目标点落在格点上必须精确还原该点值
    lat = np.array([10., 11, 12]); lon = np.array([100., 101, 102])
    f = (np.arange(9) * 1.0).reshape(3, 3)
    got = bilinear_cols(f[None], lat, lon, np.array([11.0]), np.array([101.0]))
    out.append(("bilinear_on_grid", abs(got[0, 0] - f[1, 1]) < 1e-12, float(got[0, 0])))
    got2 = bilinear_cols(f[None], lat, lon, np.array([11.5]), np.array([101.5]))
    exp2 = 0.25 * (f[1, 1] + f[1, 2] + f[2, 1] + f[2, 2])
    out.append(("bilinear_center", abs(got2[0, 0] - exp2) < 1e-12, float(got2[0, 0])))

    # 5. 垂插：1000->100 降序，target 500 应取中间层值
    lev = np.array([1000., 500, 100])
    col = np.array([[1., 2, 3]])
    v = vert_interp(col, lev, np.array([500.0]))
    out.append(("vert_interp_mid", abs(v[0] - 2.0) < 1e-12, float(v[0])))
    v2 = vert_interp(col, lev, np.array([750.0]))
    out.append(("vert_interp_between", abs(v2[0] - 1.5) < 1e-12, float(v2[0])))

    # 6. Spearman：单调 1；反单调 -1
    x = np.arange(50.0)
    out.append(("spearman_mono", abs(spearman(x, x * 3 + 1) - 1) < 1e-9,
                spearman(x, x * 3 + 1)))
    out.append(("spearman_anti", abs(spearman(x, -x) + 1) < 1e-9, spearman(x, -x)))

    # 7. 配对 bootstrap CI 必须包住点估计
    rng = np.random.default_rng(3)
    yy = (rng.random(600) < 0.25).astype(int)
    s1 = rng.normal(yy * 1.4, 1)
    s0 = rng.normal(0, 1, 600)
    bs = bootstrap_dauc(s1, s0, yy, n=600)
    lo, hi = ci_of(bs)
    pt = auc_score(s1, yy) - auc_score(s0, yy)
    out.append(("boot_ci_covers", bool(lo <= pt <= hi), float(pt)))
    return out


if __name__ == "__main__":
    for name, ok, val in selfcheck():
        print("%-22s %s   %.6g" % (name, "OK " if ok else "FAIL", val))
