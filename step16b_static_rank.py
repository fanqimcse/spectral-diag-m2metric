# -*- coding: utf-8 -*-
"""step16b —— 静态支的秩：把"静态样本个数"从 11 提到 40，再谈混合权重。

step16 暴露的问题
-----------------
    S    静态双高斯 OI（满秩）        T +25.5%   q +11.0%
    Sens 纯静态**样本** EnOI (N=11)   T  +2.5%   q  -0.1%
同样都是"静态 B"，差了 10 个百分点以上。原因不是静态 B 不好，而是
**11 个样本的样本协方差是秩 11 的**，根本表达不了满秩的静态 B ——
这正是业务从不使用"纯集合 B"的原因，也是 step16 里混合曲线只能
在两个都很弱的表示之间插值的根本原因。

如果不先把静态支的秩补够，混合实验的结论就会是"静态支没用"，
而那是一个**由秩亏造成的假象**，不是物理结论。

本脚本做两件事
--------------
  1. **保真度检验**：把静态样本数提到 M=40，看"纯静态 EnOI"能否逼近
     真正的静态 OI。逼近 ⇒ 生成器合格，之后的混合权重才可读。
  2. **混合权重曲线**：在 M=40 的静态支上叠加 N=11 的流依赖集合，
     扫 β_e ∈ {0, 0.1, 0.2, 0.3, 0.5, 1.0}，看是否存在内部最优，
     以及流依赖支在"满秩静态 B 之上"还能再加多少。
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
BETAS = (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)


def static_members_Lh(shape, sigma_b, dg, lat, lon, vert_corr=None, n=40,
                      seed=S16.SEED):
    """用 **OI 自己拟合出的** 水平相关长度生成静态样本。

    ⚠ 踩坑：第一版直接照抄 recon3d 生成合成背景误差时的多尺度谱
      （盒式滤波 n=3/9/25/61 格，等效 σ 最大 ~490 km）。但那是
      **真实误差的谱**，不是 B 矩阵的相关长度 —— OI 里用的 L_h 是
      拟合出来的，量级 124 km，相差 4 倍。用 490 km 的样本去代表
      L_h=124 km 的 B，等于把增量抹平到几乎无梯度，静态支自然
      做不出东西（实测 M=40 仍只有 +2.7%，而满秩 OI 有 +25.5%）。

    正确做法：水平用两个高斯（L_h1, L_h2，权重 1−w2, w2），
    垂直用诊断出的 vert_corr 的 Cholesky 因子。
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
                a = (Lz @ a.reshape(nz, -1)).reshape(nz, ny, nx)
            comp[k] = a
        sd = float(np.nanstd(comp))
        out[v] = comp * (float(sigma_b.get(v, 1.0)) / max(sd, 1e-12))
    return out


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
    mem_stat = static_members_Lh(shape, sb, X["dg"], lat, lon,
                                 vert_corr=X["dg"].get("vert_corr"), n=M_STATIC)
    eobs = S12.to_enoi_obs(X["obs"], use=("sfc", "raob", "ac"),
                           lat=lat, lon=lon, lev=lev)
    rec = S10.make_rec(lat, lon, lev, X["dg"], X["sigma"])
    rec.alpha_T2V, rec.alpha_V2q, rec.tau_h = 0.0, S12.ALPHA_V2Q, S12.TAU_V2Q
    uniB = rec.univariate(X["bg"], X["obs"], use=("sfc", "raob", "ac"))
    coupled = rec.apply_coupling(uniB, X["bg"], coupling=("V2q",))["an"]
    del uniB
    gc.collect()

    out = {}
    print("  %-24s " % "S 静态 OI（满秩）" +
          "  ".join("%s %+6.2f%%" % (v, S16.pct(X["bg"][v], coupled[v],
                                                 X["nat"][v], am))
                    for v in KEYS), flush=True)
    del coupled
    gc.collect()

    for be in BETAS:
        mm = (mem_ens if be >= 1.0
              else (mem_stat if be <= 0.0
                    else S16.hybrid_members(mem_ens, mem_stat, be)))
        E = EN.EnOI(mm, lat, lon, lev, sigma_b=sb, tiles=(6, 5),
                    tile_pad_km=350.0, max_obs=1000)
        an = E.update(X["bg"], eobs, OERR)
        vals = {v: S16.pct(X["bg"][v], an[v], X["nat"][v], am) for v in KEYS}
        out["beta_e=%.2f" % be] = vals
        print("  %-24s " % ("β_e=%.2f (N=%d)" % (be, mm["T"].shape[0])) +
              "  ".join("%s %+6.2f%%" % (v, vals[v]) for v in KEYS), flush=True)
        del E, an, mm
        gc.collect()
    return out


def main():
    t0 = time.time()
    print("=" * 96)
    print("step16b  静态支的秩：M=40 静态样本 + N=11 流依赖集合")
    print("=" * 96)
    res = {}
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
        res["%s%sz" % target] = run_case(target, X, am, others)
        del X
        gc.collect()
    print("\n" + "=" * 96)
    print("四目标中位")
    print("=" * 96)
    summary = {}
    tags = list(next(iter([v for v in res.values() if v])).keys())
    for tag in tags:
        row = {}
        for v in KEYS:
            row[v] = float(np.median([c[tag][v] for c in res.values()
                                      if c and tag in c]))
        summary[tag] = row
        print("  %-18s " % tag + "  ".join("%s %+6.2f%%" % (v, row[v])
                                           for v in KEYS))
    with open("data/step16b_static_rank.json", "w", encoding="utf8") as f:
        json.dump(dict(detail=res, summary=summary), f,
                  ensure_ascii=False, indent=1)
    print("\n已写出 data/step16b_static_rank.json   用时 %.0f s"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
