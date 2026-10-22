# -*- coding: utf-8 -*-
"""step12b —— EnOI 为什么失败：实现 bug 还是集合不适用？

step12 里 EnOI 得到 q −0.81%（比背景还差），而同样的观测交给双高斯 OI 却
得到 +7%。在报告"集合协方差不如静态 B"之前，必须先把两种可能分开：

    H1  实现有 bug（代价函数、局地化、缩放写错了）
    H2  集合**不适用**：成员是"其它日期"的误差样本，与目标日真实误差
        在集合张成的 N 维空间中不相关，于是 δx = A w 根本表达不出真实误差

鉴别实验（twin experiment，最干净的判据）：
    人工构造一个"真值"，使它**精确等于** background + 某个集合成员的线性组合：
        x_true = x_bg + Σ_n c_n · a^(n)
    此时真实误差**必然**落在集合子空间里。如果 EnOI 能把 RMSE 压下去，
    说明实现正确（H1 排除）；再看它在真实数据上失败 → 支持 H2。

第二组对照：**泄漏版** —— 把目标日自己的预报误差也放进集合。若泄漏版明显
优于留一版，同样指向 H2（成员的"相关性"才是决定性因素）。
"""
import gc
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from avwx import enoi as EN
from avwx import recon3d as R3
import step10_realerror_osse as S10
import step12_enoi as S12

KEYS = S10.KEYS
OERR = S10.OERR


def rmse(a, b, mask=None):
    dv = np.asarray(a, float) - np.asarray(b, float)
    if mask is not None:
        dv = dv[mask]
    return float(np.sqrt(np.nanmean(dv ** 2)))


def pct(bg, an, nat, m):
    b = rmse(bg, nat, m)
    return 100 * (b - rmse(an, nat, m)) / b


def main():
    t0 = time.time()
    target = ("20261008", "12")
    X = S12.load_case(target)
    lat, lon, lev = X["lat"], X["lon"], X["lev"]
    am = S10.impact_mask(X["obs"]["ac"][1], lat, lon, lev) & X["above"]

    others = [t for t in S12.TARGETS if t != target]
    pairs = S12.member_pairs(others)
    sb = S12.sigma_from_pairs(pairs)
    members, _ = EN.build_members(pairs, sb, clip={"q": 0.02})
    N = members["T"].shape[0]
    print("留一集合 N=%d" % N, flush=True)

    eobs = S12.to_enoi_obs(X["obs"], use=("sfc", "raob", "ac"),
                           lat=lat, lon=lon, lev=lev)

    # ---------------- 实验 1：twin experiment ----------------
    print("\n" + "=" * 92)
    print("实验 1  twin experiment：真值 = 背景 + 集合成员的线性组合")
    print("=" * 92)
    rng = np.random.default_rng(7)
    c = rng.normal(0, 1, N)
    c = c / np.linalg.norm(c) * 1.2
    twin_nat = {}
    for v in KEYS:
        # 成员量级已缩放到 σ_b；组合后作为"真误差"
        err = np.einsum("n,nabc->abc", c, members[v]) / np.sqrt(max(N - 1, 1))
        twin_nat[v] = X["bg"][v] + err
    sim = R3.ObsSim(dict(lat=lat, lon=lon, level=lev,
                         **{v: twin_nat[v] for v in KEYS}), seed=3)
    obs_twin = dict(sfc=sim.surface(n=150), raob=sim.raob(n=12), ac=sim.aircraft())
    eo_twin = S12.to_enoi_obs(obs_twin, use=("sfc", "raob", "ac"),
                              lat=lat, lon=lon, lev=lev)
    E = EN.EnOI(members, lat, lon, lev, sigma_b=sb, tiles=(6, 5),
                tile_pad_km=350.0, max_obs=1000)
    an_twin = E.update(X["bg"], eo_twin, OERR)
    print("  %-6s %10s %10s %10s %10s" % ("", *KEYS))
    print("  %-6s " % "EnOI" +
          "  ".join("%9.2f%%" % pct(X["bg"][v], an_twin[v], twin_nat[v], am)
                    for v in KEYS))
    ok_twin = all(pct(X["bg"][v], an_twin[v], twin_nat[v], am) > 1.0
                  for v in KEYS)
    print("  → 实现正确性判据（所有变量都应 > 1%%）：%s"
          % ("通过" if ok_twin else "不通过"))
    del E, an_twin
    gc.collect()

    # ---------------- 实验 2：泄漏版 vs 留一版（真实数据） ----------------
    print("\n" + "=" * 92)
    print("实验 2  成员是否必须共享目标日的误差结构")
    print("=" * 92)
    out = {}
    for tag, pz in (("留一（其它日期）", pairs),
                    ("泄漏（含目标日本身）",
                     pairs + S12.member_pairs([target]))):
        sbx = S12.sigma_from_pairs(pz)
        mem, _ = EN.build_members(pz, sbx, clip={"q": 0.02})
        E = EN.EnOI(mem, lat, lon, lev, sigma_b=sbx, tiles=(6, 5),
                    tile_pad_km=350.0, max_obs=1000)
        an = E.update(X["bg"], eobs, OERR)
        vals = {v: pct(X["bg"][v], an[v], X["nat"][v], am) for v in KEYS}
        out[tag] = dict(N=mem["T"].shape[0], **vals)
        print("  %-20s N=%2d  " % (tag, mem["T"].shape[0]) +
              "  ".join("%s %+6.2f%%" % (v, vals[v]) for v in KEYS))
        del E, an, mem
        gc.collect()

    # ---------------- 实验 3：inflation 敏感性 ----------------
    print("\n" + "=" * 92)
    print("实验 3  inflation 敏感性（留一集合）")
    print("=" * 92)
    for inf in (1.0, 1.5, 2.5, 4.0):
        mem, _ = EN.build_members(pairs, sb, clip={"q": 0.02})
        E = EN.EnOI(mem, lat, lon, lev, sigma_b=sb, tiles=(6, 5),
                    tile_pad_km=350.0, max_obs=1000, inflation=inf)
        an = E.update(X["bg"], eobs, OERR)
        vals = {v: pct(X["bg"][v], an[v], X["nat"][v], am) for v in KEYS}
        print("  inflation=%4.1f  " % inf +
              "  ".join("%s %+6.2f%%" % (v, vals[v]) for v in KEYS))
        out["infl_%.1f" % inf] = vals
        del E, an, mem
        gc.collect()

    with open("data/step12b_enoi_diag.json", "w", encoding="utf8") as f:
        json.dump(dict(target=list(target), twin_ok=bool(ok_twin),
                       experiments=out), f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step12b_enoi_diag.json   用时 %.0f s"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
