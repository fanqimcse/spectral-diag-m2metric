# -*- coding: utf-8 -*-
"""avwx/enoi —— 用集合协方差替代双高斯 B（Ensemble Optimal Interpolation）。

为什么必须走出这一步
--------------------
step03/09/10 的 B 矩阵是**双高斯、静态、变量间块对角**：

    corr(r) = (1-w) exp(-r^2/2L1^2) + w exp(-r^2/2L2^2)
    B_{V,q} = 0                       ← 变量之间**没有**任何协方差

这有三个后果，而且一个比一个致命：

  (i)  真实背景误差不是高斯形。GFS 的湿度误差高偏度、有锋面/Altiplano 各向异性，
       高斯会把协方差抹到错误的方向上去；
  (ii) 变量之间没有耦合 ⟹ 无论观测多密，**湿度都不可能从风/温观测里得到任何
       增量**。这正是 step03 里耦合系数恒为零的数学根源；
  (iii) B 不随流型变 ⟹ 在急流轴附近与静稳区用同一个相关长度。

标准答案之一是 EnOI（Oke et al. 2008）：不需要实时集合预报，只要一个
**stationary anomaly ensemble** —— 从历史预报误差里抽 N 个样本，
再乘上局地背景误差标准差 σ_b(x)。本项目恰好有真实的 GFS 预报误差样本，
可以把 NMC 法的"用一个差估计方差"升级为"用一组样本估计完整协方差"。

理论形态（ensemble subspace transform）
--------------------------------------
设 A = [a^(1) … a^(N)]，a^(n) = √c · ε^(n)，ε 为标准化误差样本，
则 B ≈ A A^T。把增量限制在集合张成的 N 维子空间：δx = A w，先验 w ~ N(0, I)。

    J(w) = ½ w^T w + ½ (Y w − d)^T R^{-1} (Y w − d),   Y = H A  ∈ R^{nobs×N}
    ⟹  (I + Y^T R^{-1} Y) w = Y^T R^{-1} d            ← 只需解 **N×N** 系统

于是代价与网格点数无关。这一步把 n_grid ~ 10^6 的求逆压缩到 10^1 量级，
正是它让"在业务网格上跑集合方法"变成可以在单机上完成的事。

局地化
------
N 只有 O(10)，远端会出现伪相关。这里采用 **domain localization**：
把水平域切成 tile，每个 tile 只用自身 + 缓冲内的观测做上述 N 维更新，
再以余弦锥化权重合并，避免 tile 边界出现不连续。
"""
import os
import sys

import numpy as np

if __package__ in (None, ""):                 # 允许 `python avwx/enoi.py` 直接跑
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avwx import verify as VF

KEYS = ("T", "q", "u", "v")


# ------------------------------------------------------------------ 工具
def interp_cols(field, lat1d, lon1d, lev_desc, qlat, qlon, qp):
    """把 (nz,ny,nx) 的三维场插值到一批散点 (lat,lon,p)。

    qp 的单位必须与 lev_desc 一致（默认都用 hPa）。
    """
    col = VF.bilinear_cols(field, lat1d, lon1d, qlat, qlon)
    return VF.vert_interp(col, lev_desc, qp)


def taper_cos(dist, radius):
    """Gaspari–Cohn 的廉价替代：余弦锥化，radius 外严格为 0。"""
    d = np.asarray(dist, float)
    w = np.zeros_like(d)
    m = d < radius
    w[m] = 0.5 * (1.0 + np.cos(np.pi * d[m] / radius))
    return w


# ------------------------------------------------------------------ 集合构造
def build_members(pairs, sigma_b, keys=KEYS, clip=None):
    """由若干 (analysis, forecast) 样本对构造 EnOI 集合。

    pairs    : [(ana, fst)]，两者都是 {var: (nz,ny,nx)}
    sigma_b  : {var: float 或 (nz,) }  背景误差标准差（绝对量纲）
    clip     : 可选的卫星式截断，比如 q 用 0.06

    标准化策略（很关键，见 README 讨论）：
    逐变量用**同一个**标量把整个三维场缩放到 σ_b 的量级 —— 不要逐层标准化，
    否则会抹掉误差的垂直相关结构，而那正是我们想要估计的东西之一。
    """
    N = len(pairs)
    if N < 3:
        raise ValueError("集合太小：N=%d" % N)
    members = {}
    info = {}
    for v in keys:
        arr = np.stack([np.asarray(f[v], float) - np.asarray(a[v], float)
                        for a, f in pairs])
        if clip is not None and v in clip:
            arr = np.clip(arr, -clip[v], clip[v])
        sd = float(np.nanstd(arr))                    # 全三维一个标量
        sb = sigma_b.get(v, 1.0)
        sbf = np.mean(sb) if np.ndim(sb) else float(sb)
        members[v] = arr * (sbf / max(sd, 1e-12))
        info[v] = dict(raw_std=sd, sigma_b=sbf, scale=sbf / max(sd, 1e-12))
    return members, info


def members_from_grid(members_dense, factor=4):
    """把集合粗化到 1/factor 的分析网格（可选，用于降低内存）。"""
    out = {}
    for v, a in members_dense.items():
        nz, ny, nx = a.shape[1:]
        ky = slice(0, ny - (ny % factor), factor)
        kx = slice(0, nx - (nx % factor), factor)
        out[v] = np.ascontiguousarray(a[:, :, ky, kx])
    return out


# ------------------------------------------------------------------ EnOI
class EnOI:
    """集合同化增量算子。

    用法：
        E = EnOI(members, lat, lon, lev, sigma_b)
        an = E.update(bg, obs, oerr)
    其中 obs 是 [(var, value_array, pos_array)] 的列表，
    pos 的列为 (p_hPa, lat, lon)。
    """

    def __init__(self, members, lat, lon, lev, sigma_b=None,
                 tiles=(5, 4), tile_pad_km=400.0, inflation=1.0,
                 max_obs=1200, seed=20261021):
        self.A = {k: np.asarray(v, float) for k, v in members.items()}
        self.N = next(iter(self.A.values())).shape[0]
        self.lat = np.asarray(lat, float)
        self.lon = np.asarray(lon, float)
        self.lev = np.asarray(lev, float)
        self.sigma_b = sigma_b or {}
        self.tiles = tiles
        self.tile_pad_km = tile_pad_km
        self.inflation = inflation
        self.max_obs = max_obs
        self.rng = np.random.default_rng(seed)
        nz, ny, nx = next(iter(self.A.values())).shape[1:]
        self.shape = (nz, ny, nx)
        dlat = float(np.mean(np.diff(self.lat)))
        self.dlat_km = dlat * 111.195
        self.sqrtN1 = np.sqrt(max(self.N - 1, 1))

    # ---------------------------------------------------------- 观测提取
    def _normalise_obs(self, obs):
        """obs: dict{var: (vals, pos)}  ->  (vars[], vals[], p[], lat[], lon[])"""
        vs, vv, vp, vla, vlo = [], [], [], [], []
        for var, item in obs.items():
            if item is None:
                continue
            vals, pos = item
            pos = np.asarray(pos, float)
            vals = np.asarray(vals, float).ravel()
            m = np.isfinite(vals) & np.all(np.isfinite(pos), axis=1)
            vs.append(np.full(int(m.sum()), var))
            vv.append(vals[m])
            vp.append(pos[m, 0])
            vla.append(pos[m, 1])
            vlo.append(pos[m, 2])
        if not vs:
            return None
        return (np.concatenate(vs), np.concatenate(vv), np.concatenate(vp),
                np.concatenate(vla), np.concatenate(vlo))

    def _H(self, field_dict, ov, op, olat, olon, oidx):
        """把集合成员/背景插值到观测位置 -> (nobs,)"""
        out = np.empty(oidx.size)
        for var in KEYS:
            sel = np.where(ov[oidx] == var)[0]
            if sel.size == 0:
                continue
            idx = oidx[sel]
            out[sel] = interp_cols(np.asarray(field_dict[var], float),
                                   self.lat, self.lon, self.lev,
                                   olat[idx], olon[idx], op[idx])
        return out

    # ---------------------------------------------------------- 主更新
    def update(self, bg, obs, oerr):
        """返回 {var: (nz,ny,nx)} 的分析场。

        obs  : dict{var: (vals, pos)}，pos 列 = (p_hPa, lat, lon)
        oerr : dict{var: float}
        """
        no = self._normalise_obs(obs)
        if no is None:
            return {v: bg[v].copy() for v in KEYS}
        ov, oval, op, olat, olon = no
        if olon.max() > 180:
            olon = np.where(olon > 180, olon - 360, olon)
        an = {v: np.asarray(bg[v], float).copy() for v in KEYS}
        nz, ny, nx = self.shape

        # ---- 观测算子作用于背景 ----
        allidx = np.arange(oval.size)
        hbg = self._H(bg, ov, op, olat, olon, allidx)
        d = oval - hbg
        good = np.isfinite(d)
        Rinv = np.array([1.0 / max(oerr.get(v, 1.0) ** 2, 1e-12)
                         for v in ov])

        # ---- 观测算子作用于每个成员：Y (nobs × N) ----
        Y = np.empty((oval.size, self.N))
        for n in range(self.N):
            mem = {v: self.A[v][n] for v in KEYS}
            Y[:, n] = self._H(mem, ov, op, olat, olon, allidx)

        # ---- tile 局地化 ----
        nty, ntx = self.tiles
        ys = np.linspace(0, ny, nty + 1).astype(int)
        xs = np.linspace(0, nx, ntx + 1).astype(int)
        acc = {v: np.zeros((nz, ny, nx)) for v in KEYS}
        wsum = np.zeros((ny, nx))
        pad_g = self.tile_pad_km / max(self.dlat_km, 1e-6)
        lonv = np.asarray(self.lon, float)
        llon = np.where(lonv > 180, lonv - 360, lonv)
        coslat_scale = 111.195 * np.cos(np.deg2rad(45.0))

        for it in range(nty):
            for jt in range(ntx):
                y0, y1 = ys[it], ys[it + 1]
                x0, x1 = xs[jt], xs[jt + 1]
                cy = 0.5 * (y0 + y1)
                cx = 0.5 * (x0 + x1)
                # 该 tile 到各观测的距离（km）
                dys = (olat - self.lat[min(int(cy), ny - 1)]) * 111.195
                dxs = (olon - llon[min(int(cx), nx - 1)]) * coslat_scale
                dist = np.sqrt(dys ** 2 + dxs ** 2)
                m = dist < self.tile_pad_km
                idx = np.where(m & good)[0]
                if idx.size < 1:
                    continue
                if idx.size > self.max_obs:
                    idx = self.rng.choice(idx, self.max_obs, replace=False)
                w = taper_cos(dist[idx], self.tile_pad_km)
                Rw = Rinv[idx] * w
                Yl = Y[idx]
                dl = d[idx]
                # (I + Y^T Rw Y) wgt = Y^T Rw d
                #   —— 不做任何额外的"归一化"缩放：代价函数里第一项 ½w^T w
                #      已经代表 N(0,I) 先验，任何手工缩放都会破坏它。
                #      （踩坑：一度乘了个 sqrt(mean(w))，纯属凭空发明，直接把
                #       增量压到噪声量级；已删除。）
                M = np.einsum("ij,i,ik->jk", Yl, Rw, Yl)
                rhs = np.einsum("ij,i,i->j", Yl, Rw, dl)
                A = (np.eye(self.N) + M + np.eye(self.N) * 1e-8 *
                     max(np.trace(M) / self.N, 1.0))
                try:
                    wgt = np.linalg.solve(A, rhs)
                except np.linalg.LinAlgError:
                    continue
                wgt = wgt * self.inflation
                # 增量 = Σ_n wgt_n · a^(n) / sqrt(N-1)
                dy_ = self.lat[y0:y1] if y1 > y0 else self.lat[y0:y0 + 1]
                dx_ = llon[x0:x1] if x1 > x0 else llon[x0:x0 + 1]
                YY, XX = np.meshgrid(
                    (llon[x0:x1] - llon[min(int(cx), nx - 1)]),
                    (self.lat[y0:y1] - self.lat[min(int(cy), ny - 1)]),
                    indexing="xy")
                wn = taper_cos(np.sqrt((YY * coslat_scale) ** 2 +
                                       (XX * 111.195) ** 2),
                               self.tile_pad_km)
                for v in KEYS:
                    inc = np.einsum("n,nabc->abc", wgt, self.A[v]) / self.sqrtN1
                    slice_v = (slice(None), slice(y0, max(y1, y0 + 1)),
                               slice(x0, max(x1, x0 + 1)))
                    target = acc[v][slice_v]
                    patch = inc[slice_v]
                    if patch.shape != target.shape:
                        continue
                    target += patch * wn[None]
                sub = np.zeros((ny, nx))
                sub[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)] += wn
                wsum += sub

        for v in KEYS:
            base = np.broadcast_to(an[v][:, :, :], (nz, ny, nx))
            inc = np.zeros_like(base)
            w3 = np.maximum(wsum, 1e-12)[None]
            inc = np.where(wsum[None] > 1e-12, acc[v] / w3, 0.0)
            an[v] = np.asarray(base + inc, float)
        return an


def selfcheck():
    out = []

    # 1. 单观测、单变量：无耦合成员时应退化为卡尔曼
    nz, ny, nx = 3, 12, 12
    lat = np.linspace(10, 20, ny)
    lon = np.linspace(100, 110, nx)
    lev = np.array([500., 700, 850])
    rng = np.random.default_rng(0)
    N = 8
    members = {}
    for v in KEYS:
        members[v] = rng.normal(0, 1, (N, nz, ny, nx)) * 0.3
    bg = {v: np.zeros((nz, ny, nx)) for v in KEYS}
    E = EnOI(members, lat, lon, lev, tiles=(2, 2), tile_pad_km=1e6)
    obs = {"T": (np.array([1.0]), np.array([[500.0, 15.0, 105.0]]))}
    an = E.update(bg, obs, {"T": 0.5})
    got = interp_cols(an["T"], lat, lon, lev,
                      np.array([15.0]), np.array([105.0]), np.array([500.0]))[0]
    out.append(("single_obs_moves_toward", got > 0.05 and got < 1.0,
                float(got)))
    # 负 innovations 必须向反方向走
    obs2 = {"T": (np.array([-1.0]), np.array([[500.0, 15.0, 105.0]]))}
    an2 = E.update(bg, obs2, {"T": 0.5})
    got2 = interp_cols(an2["T"], lat, lon, lev,
                       np.array([15.0]), np.array([105.0]), np.array([500.0]))[0]
    out.append(("sign_symmetry", got2 < -0.05 and abs(got2 + got) < 1e-6,
                float(got2)))
    # T 观测不能污染与之零协方差的成员 → 这里成员是随机的，无跨变量结构，
    # 检查更新有限且非 NaN
    out.append(("finite_update", bool(np.all(np.isfinite(an["T"]))),
                float(np.nanmax(np.abs(an["T"])))))

    # 2. build_members：缩放后 std 必须等于 sigma_b
    pairs = []
    for i in range(6):
        pairs.append(({v: np.zeros((2, 5, 5)) for v in KEYS},
                      {v: rng.normal(0, 2.0, (2, 5, 5)) for v in KEYS}))
    mem, info = build_members(pairs, {"T": 1.0, "q": 1e-4, "u": 1.5, "v": 1.5})
    sd_T = float(np.nanstd(mem["T"]))
    out.append(("member_scaled_to_sigma_b", abs(sd_T - 1.0) < 1e-9, sd_T))
    sd_u = float(np.nanstd(mem["u"]))
    out.append(("member_scaled_u", abs(sd_u - 1.5) < 1e-6, sd_u))

    # 3. taper：半径外为 0，中心为 1，单调递减
    w = taper_cos(np.array([0.0, 50.0, 100.0, 200.0]), 200.0)
    out.append(("taper_center", abs(w[0] - 1.0) < 1e-12, float(w[0])))
    out.append(("taper_edge_zero", abs(w[3]) < 1e-12, float(w[3])))
    out.append(("taper_monotone", bool(np.all(np.diff(w[:3]) < 0)), float(w[1])))

    # 4. 无观测时必须原样返回背景
    an3 = E.update(bg, {}, {"T": 0.5})
    out.append(("no_obs_identity", bool(np.allclose(an3["T"], 0.0)),
                float(np.nanmax(np.abs(an3["T"])))))
    return out


if __name__ == "__main__":
    for name, ok, val in selfcheck():
        print("%-26s %s   %.6g" % (name, "OK " if ok else "FAIL", val))
