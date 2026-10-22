# -*- coding: utf-8 -*-
"""step16 —— 混合背景误差协方差 B = β_e·B_ens + β_s·B_static。

为什么必须做这一步
------------------
step12/step12c 只比较了两个端点：纯静态双高斯 OI 与纯集合 EnOI，结论是
"跨日期的气候学集合没用（q −0.76%），加入同日异时效成员后转正（q +6.2%）"。

但业务上**没有任何一个中心**会用"纯集合 B"替代静态 B —— 因为集合只有 10–50 个
成员，样本协方差必然秩亏、必然有虚假远距离相关。业务的通行做法是 **hybrid**：
        B = β_e·B_ens + β_s·B_static,   β_e + β_s = 1
（Hamill & Snyder 2000；WRFDA 的 hybrid 3DEnVar、JEDI 的 hybrid B 皆如此。）

所以"以集合协方差替代双高斯 B"这句话在本项目里必须被精确地重述为：
        **在混合框架下，流依赖那一支能贡献多少？最优 β_e 是多少？**
只报纯集合的结果，会被审稿人正确地指出"这不是业务可比的配置"。

实现要点
--------
EnOI 的增量限制在集合子空间 δx = A w，B ≈ A Aᵀ/(N−1)。
把静态样本**作为额外的成员**拼进 A，就得到混合：

    A' = [ √(β_e·(N−1)/N_e) · A_ens  ,  √(β_s·(N−1)/N_s) · A_stat ]

则 A' A'ᵀ/(N−1) = β_e·B_ens + β_s·B_static，正是混合 B。
（权重里的 (N−1)/N_e 因子不能漏：两个子块成员数不同，直接拼接会让
  实际权重偏离 β，变成"谁成员多谁权重大"。）

静态样本的生成
--------------
静态 B 是"双高斯"相关模型，等价于两个尺度的各向同性高斯随机场加权叠加。
这里用 recon3d 生成合成背景误差时**同一套**多尺度盒式滤波近似
（n = 3/9/25/61 格，权重 0.32/0.43/0.60/0.52），再整体缩放到 σ_b。
为验证这套近似是否忠实，本脚本并列了一个 **纯静态样本** 的 EnOI 配置：
若它与真正的静态 OI 结果接近，就说明静态样本生成器是合格的，
混合结果的解读才站得住。
"""
import gc
import json
import sys
import time

import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, ".")
from avwx import enoi as EN
import step10_realerror_osse as S10
import step12_enoi as S12
import step12c_flowdep as S12C
from avwx import recon3d as R3

KEYS = S10.KEYS
OERR = S10.OERR
NS = 11                       # 静态样本个数（与集合成员数对齐，避免权重偏置）
SCALES = ((3, 0.32), (9, 0.43), (25, 0.60), (61, 0.52))
SEED = 20261021


def static_members(shape, sigma_b, vert_corr=None, n=NS, seed=SEED,
                   scales=SCALES):
    """从静态（双高斯 + 诊断垂直相关）B 中抽取 n 个样本，形状 (n, nz, ny, nx)。

    ⚠ 踩坑：第一版只做水平多尺度平滑，**垂直 lag-1 相关只有 0.013** ——
      等于给了一个垂直白噪声的 B。那会把"纯静态 EnOI"这一配置人为做坏，
      进而把混合实验的结论带偏（β_s 那一支被低估）。
      修正：垂直相关直接用 OI 自己诊断出的矩阵 Cv 的 Cholesky 因子施加，
      而不是凭空发明一个平滑次数。
    """
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    Lz = None
    if vert_corr is not None:
        Cv = np.asarray(vert_corr, float)
        if Cv.ndim == 2 and Cv.shape[0] == nz:
            Cv = 0.5 * (Cv + Cv.T)
            ev = np.linalg.eigvalsh(Cv)
            Cv = Cv + np.eye(nz) * max(1e-6, -1.05 * float(np.min(ev)) + 1e-6)
            Lz = np.linalg.cholesky(Cv)
    out = {}
    for v in KEYS:
        comp = np.zeros((n, nz, ny, nx))
        for k in range(n):
            c = np.zeros((nz, ny, nx))
            for nn, wt in scales:
                s = uniform_filter(rng.standard_normal((nz, ny, nx)),
                                   size=(1, nn, nn), mode="nearest")
                sd = float(np.nanstd(s))
                c = c + wt * s / (sd + 1e-12)
            if Lz is not None:
                c = (Lz @ c.reshape(nz, -1)).reshape(nz, ny, nx)
            comp[k] = c
        sd = float(np.nanstd(comp))                  # 全三维一个标量
        out[v] = comp * (float(sigma_b.get(v, 1.0)) / max(sd, 1e-12))
    return out


def hybrid_members(mem_ens, mem_stat, beta_e):
    """按 B = β_e B_ens + β_s B_static 拼接，修正子块成员数不等带来的权重偏置。"""
    Ne = next(iter(mem_ens.values())).shape[0]
    Ns = next(iter(mem_stat.values())).shape[0]
    N = Ne + Ns
    beta_s = 1.0 - beta_e
    ce = np.sqrt(beta_e * (N - 1) / Ne)
    cs = np.sqrt(beta_s * (N - 1) / Ns)
    out = {}
    for v in KEYS:
        out[v] = np.concatenate([ce * np.asarray(mem_ens[v], float),
                                 cs * np.asarray(mem_stat[v], float)], axis=0)
    return out


def pct(bg, an, nat, m):
    def rmse(a, b):
        dv = (np.asarray(a, float) - np.asarray(b, float))[m]
        return float(np.sqrt(np.nanmean(dv ** 2)))
    b = rmse(bg, nat)
    return 100 * (b - rmse(an, nat)) / b


def run_case(target, X, am, others):
    lat, lon, lev = X["lat"], X["lon"], X["lev"]
    shape = X["bg"]["T"].shape

    # ---- 集合：其它日期 ×(6,12,24) + 目标日异时效(6,12)  = step12c 的最优配置 B
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
    mem_stat = static_members(shape, sb, vert_corr=X["dg"].get("vert_corr"))

    eobs = S12.to_enoi_obs(X["obs"], use=("sfc", "raob", "ac"),
                           lat=lat, lon=lon, lev=lev)

    # ---- 静态 OI 基线（带 V2q 耦合，即 step10 的最佳配置 D）
    rec = S10.make_rec(lat, lon, lev, X["dg"], X["sigma"])
    rec.alpha_T2V, rec.alpha_V2q, rec.tau_h = 0.0, S12.ALPHA_V2Q, S12.TAU_V2Q
    uniB = rec.univariate(X["bg"], X["obs"], use=("sfc", "raob", "ac"))
    coupled = rec.apply_coupling(uniB, X["bg"], coupling=("V2q",))["an"]

    out = {}
    cfgs = [("S  静态双高斯 OI", None),
            ("Sens 纯静态样本 EnOI", 0.0),
            ("E  纯集合 EnOI", 1.0),
            ("H25 混合 β_e=0.25", 0.25),
            ("H50 混合 β_e=0.50", 0.50),
            ("H75 混合 β_e=0.75", 0.75)]
    for tag, be in cfgs:
        if be is None:
            an = coupled
        else:
            mm = mem_ens if be >= 1.0 else (
                mem_stat if be <= 0.0 else hybrid_members(mem_ens, mem_stat, be))
            E = EN.EnOI(mm, lat, lon, lev, sigma_b=sb, tiles=(6, 5),
                        tile_pad_km=350.0, max_obs=1000)
            an = E.update(X["bg"], eobs, OERR)
            del E
            gc.collect()
        vals = {v: pct(X["bg"][v], an[v], X["nat"][v], am) for v in KEYS}
        out[tag] = vals
        print("  %-22s " % tag + "  ".join("%s %+6.2f%%" % (v, vals[v])
                                           for v in KEYS), flush=True)
        del an
        gc.collect()
    return out


def main():
    t0 = time.time()
    print("=" * 96)
    print("step16  混合 B：β_e·B_ens(流依赖) + β_s·B_static(双高斯)")
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
        print("\n目标 %s%sz  集合成员 = 其它日期×3 + 同日异时效×2" % target,
              flush=True)
        res["%s%sz" % target] = run_case(target, X, am, others)
        del X
        gc.collect()

    print("\n" + "=" * 96)
    print("四个目标的中位（逐变量）")
    print("=" * 96)
    agg = {}
    for c in res.values():
        for tag, vals in (c or {}).items():
            agg.setdefault(tag, {}).setdefault("_rows", []).append(vals)
    summary = {}
    for tag, d in agg.items():
        line = {}
        for v in KEYS:
            line[v] = float(np.median([r[v] for r in d["_rows"]]))
        summary[tag] = line
        print("  %-22s " % tag + "  ".join("%s %+6.2f%%" % (v, line[v])
                                           for v in KEYS))
    with open("data/step16_hybrid.json", "w", encoding="utf8") as f:
        json.dump(dict(detail=res, summary=summary), f,
                  ensure_ascii=False, indent=1)
    print("\n已写出 data/step16_hybrid.json   用时 %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
