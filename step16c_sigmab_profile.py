# -*- coding: utf-8 -*-
"""step16c —— 排除"σ_b 垂直廓线口径"这个混淆项。

step16b 留下的悬案
------------------
    S    静态双高斯 OI（满秩）      T +25.5% / +13.8% / +11.6% / +10.2%
    静态样本 EnOI  M=40（β_e=0）    T  +2.7% /  +1.3% /  +1.2% /  +0.8%
把样本数从 11 提到 40（秩补够 4 倍）**几乎没用**。这说明秩不是主因，
一定还有第二个混淆项。

真正的第二个混淆项：σ_b 的垂直廓线口径。
----------------------------------------
- OI 路线用的是 `X["sigma"]`，其中 σ_b 是 **逐层诊断的真实预报误差**
  （step10: `sig_b[v] = [sqrt(mean((b-nat)^2)) for k in levels]`，形状 (nz,)）。
- EnOI 路线用的是 `sigma_from_pairs()`，它把整个三维场压成 **每变量一个标量**
  （`build_members` 里 `sbf = np.mean(sb)`）。
- 于是合成出来的静态样本在**每一层振幅相同**，而真实背景误差随高度变化
  可达一个量级。静态支的增量被错误地摊到"误差其实很小"的层上。

为什么这个修正对**集合支**不适用、对**静态支**必须做
----------------------------------------------------
`build_members` 的注释写着"不要逐层标准化，否则会抹掉误差的垂直相关结构"。
这条理由对集合成立：集合成员的逐层方差廓线**本身就是**要估计的信息。
但对合成的静态样本不成立 —— 它的逐层均匀方差是**生成器的产物**，
不是 B 的性质。静态 B 的定义就是

    B = D^{1/2} C D^{1/2}      D = diag(σ_b²)，C = 相关矩阵

而且关键性质：**相关矩阵在正对角缩放下不变**
（corr(X_i c_i, X_j c_j) = C_ij/sqrt(C_ii C_jj)）。
所以"先施加垂直相关（Cholesky）、再逐层缩放到诊断的 σ_b 廓线"
**严格等于** B = D^{1/2} C D^{1/2}，不破坏任何东西。

对照设计（每个目标 5 个配置）
----------------------------
    S        静态双高斯 OI（满秩）          基线
    β_e=0 sc 纯静态样本 EnOI，标量 σ_b      复现 step16b 的 +1.4%
    β_e=0 pl 纯静态样本 EnOI，逐层 σ_b      ← 检验项
    β_e=.5   混合（逐层静态支 M=40 + 集合 N=11）
    β_e=1.0  纯集合 EnOI（N=11）
"""
import gc
import json
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
import step10_realerror_osse as S10
import step12_enoi as S12
import step12c_flowdep as S12C
import step16_hybrid as S16
from avwx import enoi as EN

KEYS = S10.KEYS
OERR = S10.OERR
M_STATIC = 40


def static_members_Lh(shape, sigma_b, dg, lat, lon, vert_corr=None, n=40,
                      seed=S16.SEED, sigma_b_lev=None, per_level=False):
    """从静态双高斯 B 抽样。

    sigma_b      : {var: float}           EnOI 用的标量口径（对照组）
    sigma_b_lev  : {var: (nz,)}           OI 真正用的逐层诊断廓线（检验组）
    per_level    : True → 逐层缩放（= B 的正确实现）

    ⚠ 两个已修的坑都写在注释里：
      1) 水平相关长度必须用 OI **拟合出的** L_h（~124 km），不能照抄
         recon3d 生成合成误差时的多尺度谱（~490 km）—— 那是真实误差的谱，
         不是 B 的相关长度。
      2) 方差必须逐层给（若给了廓线），否则合成的均匀垂直方差与真实
         σ_b 廓线差一个量级，静态支的增量会被摊到错误的层上。
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    dlat = float(np.mean(np.diff(lat))) * 111195.0
    L1 = float(dg.get("L_h1", 124e3)) / dlat
    L2 = float(dg.get("L_h2", L1 * 3)) / dlat
    w2 = float(dg.get("w2", 0.0))
    Lz = None
    if vert_corr is not None:
        Cv = np.asarray(vert_corr, float)
        if Cv.ndim == 2 and Cv.shape[0] == nz:
            Cv = 0.5 * (Cv + Cv.T)
            ev = np.linalg.eigvalsh(Cv)
            Cv = Cv + np.eye(nz) * max(1e-6, -1.05 * float(np.min(ev)) + 1e-6)
            Lz = np.linalg.cholesky(Cv)
    out = {}
    for v in S16.KEYS:
        comp = np.zeros((n, nz, ny, nx))
        for k in range(n):
            c = rng.standard_normal((nz, ny, nx))
            a = gaussian_filter(c, sigma=(0, L1, L1), mode="nearest")
            a = a / (float(np.nanstd(a)) + 1e-12)
            if w2 > 1e-6:
                b = gaussian_filter(c, sigma=(0, L2, L2), mode="nearest")
                b = b / (float(np.nanstd(b)) + 1e-12)
                a = np.sqrt(1 - w2) * a + np.sqrt(w2) * b
            if Lz is not None:
                # 垂直相关：Cholesky 因子左乘（相关结构在此建立）
                a = (Lz @ a.reshape(nz, -1)).reshape(nz, ny, nx)
            comp[k] = a
        if per_level and sigma_b_lev is not None and v in sigma_b_lev:
            # 逐层缩放到诊断廓线：只改方差，不改相关（正对角缩放保相关）
            sdl = np.nanstd(comp.reshape(n, nz, -1), axis=(0, 2))
            prof = np.asarray(sigma_b_lev[v], float)
            fac = prof / np.maximum(sdl, 1e-12)
            comp = comp * fac[None, :, None, None]
        else:
            sd = float(np.nanstd(comp))
            comp = comp * (float(sigma_b.get(v, 1.0)) / max(sd, 1e-12))
        out[v] = comp
    return out


def profile_report(X):
    """报告 σ_b 逐层廓线的层间变化幅度 —— 混淆项的量化强度。"""
    lines = []
    for v in KEYS:
        p = np.asarray(X["sig_b"][v], float)
        p = p[np.isfinite(p) & (p > 0)]
        if p.size < 2:
            continue
        lines.append("%s: 逐层 σ_b 最大/最小 = %.1f×  (max %.4g @%s, min %.4g)"
                     % (v, p.max() / p.min(), p.max(),
                        int(X["lev"][int(np.argmax(X["sig_b"][v]))]), p.min()))
    return lines


def run_case(target, X, am, others):
    lat, lon, lev = X["lat"], X["lon"], X["lev"]
    shape = X["bg"]["T"].shape
    same_day = []
    for lead in S12C.OTHER_LEADS:
        pr = S12C.pair_for(target, lead)
        if pr is not None:
            same_day.append(pr)
    pz = S12.member_pairs(others) + same_day
    if len(pz) < 3:
        return None
    sb = S12.sigma_from_pairs(pz)
    mem_ens, _ = EN.build_members(pz, sb, clip={"q": 0.02})
    del pz
    gc.collect()

    mem_sc = static_members_Lh(shape, sb, X["dg"], lat, lon,
                               vert_corr=X["dg"].get("vert_corr"),
                               n=M_STATIC, per_level=False)
    mem_pl = static_members_Lh(shape, sb, X["dg"], lat, lon,
                               vert_corr=X["dg"].get("vert_corr"),
                               n=M_STATIC, per_level=True,
                               sigma_b_lev=X["sig_b"])

    eobs = S12.to_enoi_obs(X["obs"], use=("sfc", "raob", "ac"),
                           lat=lat, lon=lon, lev=lev)
    rec = S10.make_rec(lat, lon, lev, X["dg"], X["sigma"])
    rec.alpha_T2V, rec.alpha_V2q, rec.tau_h = 0.0, S12.ALPHA_V2Q, S12.TAU_V2Q
    uniB = rec.univariate(X["bg"], X["obs"], use=("sfc", "raob", "ac"))
    coupled = rec.apply_coupling(uniB, X["bg"], coupling=("V2q",))["an"]
    del uniB
    gc.collect()

    out = {}
    vals = {v: S16.pct(X["bg"][v], coupled[v], X["nat"][v], am) for v in KEYS}
    out["S_静态OI满秩"] = vals
    print("  %-24s " % "S 静态 OI（满秩）" +
          "  ".join("%s %+6.2f%%" % (v, vals[v]) for v in KEYS), flush=True)
    del coupled
    gc.collect()

    cfgs = [("β_e=0.00 静态 标量σ_b", mem_sc),
            ("β_e=0.00 静态 逐层σ_b", mem_pl),
            ("β_e=0.50 混合 逐层σ_b", S16.hybrid_members(mem_ens, mem_pl, 0.5)),
            ("β_e=1.00 纯集合", mem_ens)]
    for tag, mm in cfgs:
        E = EN.EnOI(mm, lat, lon, lev, sigma_b=sb, tiles=(6, 5),
                    tile_pad_km=350.0, max_obs=1000)
        an = E.update(X["bg"], eobs, OERR)
        vals = {v: S16.pct(X["bg"][v], an[v], X["nat"][v], am) for v in KEYS}
        out[tag] = vals
        print("  %-24s " % ("%s (N=%d)" % (tag, mm["T"].shape[0])) +
              "  ".join("%s %+6.2f%%" % (v, vals[v]) for v in KEYS), flush=True)
        del E, an
        gc.collect()
    del mem_sc, mem_pl, mem_ens
    gc.collect()
    return out


def main():
    t0 = time.time()
    print("=" * 96)
    print("step16c  排除 σ_b 垂直廓线口径混淆：静态支改用逐层 σ_b")
    print("=" * 96)
    res = {}
    prof_lines = []
    for target in S12.TARGETS:
        try:
            X = S12.load_case(target)
        except Exception as e:
            print("%s 跳过: %s" % (target, str(e)[:80]))
            continue
        am = S10.impact_mask(X["obs"]["ac"][1], X["lat"], X["lon"],
                             X["lev"]) & X["above"]
        others = [t for t in S12.TARGETS if t != target]
        print("\n目标 %s%sz" % target, flush=True)
        if not prof_lines:
            prof_lines = profile_report(X)
            print("  [σ_b 廓线] " + " | ".join(prof_lines), flush=True)
        res["%s%sz" % target] = run_case(target, X, am, others)
        del X
        gc.collect()

    print("\n" + "=" * 96)
    print("四目标中位")
    print("=" * 96)
    keys = ["S_静态OI满秩", "β_e=0.00 静态 标量σ_b", "β_e=0.00 静态 逐层σ_b",
            "β_e=0.50 混合 逐层σ_b", "β_e=1.00 纯集合"]
    summary = {}
    for tag in keys:
        row = {}
        for v in KEYS:
            row[v] = float(np.median([c[tag][v] for c in res.values()
                                      if c and tag in c]))
        summary[tag] = row
        print("  %-24s " % tag + "  ".join("%s %+6.2f%%" % (v, row[v])
                                           for v in KEYS))
    with open("data/step16c_sigmab_profile.json", "w", encoding="utf8") as f:
        json.dump(dict(detail=res, summary=summary, sigma_b_profile=prof_lines),
                  f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step16c_sigmab_profile.json   用时 %.0f s"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
