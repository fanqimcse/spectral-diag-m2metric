# -*- coding: utf-8 -*-
"""
OSSE：加入**地形遮蔽**并重做标定与内部一致性检验。

为什么要做这一步
----------------
GFS/任何模式在气压面上的分析**不做地形遮蔽**：青藏高原东缘实测
p_sfc 低至 587 hPa，而 1000/925/850 hPa 面在那里位于**地面以下**，
其数值是外推出来的。把它们当"自然真值"会带来三重污染：

  1. 地表边界处的 T、q 廓线被强行外推，垂直梯度失真；
  2. 观测（地面站、爬升段飞机）会被放在地面以下 —— 物理上不可能；
  3. RMSE 统计包含了永不可观测的体积，稀释真实技巧。

本脚本的处理：
  * **谷地填充**：每列位于地面以下的气压层，用该列最低的地面以上层的值
    纵向填充 —— 使廓线在垂直方向变为中性延伸，消除虚假梯度。
    （这正是业务上把模式场插到气压面时的常规做法。）
  * 观测只放在地面以上的层；评估掩码只统计地面以上格点。
  * 其余重构算子、耦合算子、内部一致性口径与 step03 完全一致，
    以便逐项对照 step03（未遮蔽）与本脚本（遮蔽）的差异。

所有权限由用户全权授权，参数按推荐方案选取。
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(".")))
sys.path.insert(0, ".")
from avwx import aws_gfs as AG
from avwx import recon3d as R3

BBOX = (100, 130, 15, 40)
OERR = dict(T=0.5, q=5e-4, u=1.5, v=1.5)
IMPACT_R = 150e3
IMPACT_LNP = 0.5
COARSE, NOISE = 4, 1.2
NATIVE_ROOT = "data/aws/"

# 与时 Zwischen variation：优先使用与原 OSSE 一致的 10 月东亚时次
CAL_TIMES = [("20261001", "00"), ("20261004", "18"), ("20261008", "00")]
VAL_TIMES = [("20261008", "12"), ("20261012", "18"), ("20261016", "00")]


# --------------------------------------------------------------- 工具
def native_path(dte, cyc):
    return os.path.join(NATIVE_ROOT,
                        "gfs_%s_%s_f000_L21_1000_100.grib2" % (dte, cyc))


def load_case(dte, cyc, coarse=COARSE, noise=NOISE, seed=20261020):
    """读取原生 21 层、做谷地填充、构造合成背景场与观测。"""
    p = native_path(dte, cyc)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    d = AG.load(p, bbox=BBOX)
    lat, lon, lev = d["lat"], d["lon"], d["level"]
    psfc = np.asarray(d["p_sfc"], float)                       # Pa
    play = np.asarray(lev, float)[:, None, None] * 100.0
    above = play <= psfc[None, ...]                            # (nz,ny,nx)
    nbelow = int((~above).sum())

    nat = {v: np.asarray(d[v], float).copy() for v in ("T", "q", "u", "v", "z")}

    # ---- 谷地填充（向量化）----
    kk = np.arange(play.shape[0])[:, None, None]
    k0 = np.argmax(above, axis=0)                 # 每列最低的地面以上层
    fill = kk < k0[None]                          # 更低（更高气压）且在地下的层
    for v in ("T", "q", "u", "v"):
        a = nat[v]
        vals = np.take_along_axis(a, k0[None], axis=0)
        a[fill] = np.broadcast_to(vals, a.shape)[fill]
    nat["lat"], nat["lon"], nat["level"] = lat, lon, lev

    bg = R3.make_background(nat, coarse=coarse, noise=noise, seed=seed)
    sim = R3.ObsSim(nat, seed=7)
    obs = dict(sfc=sim.surface(n=150), raob=sim.raob(n=12), ac=sim.aircraft())
    dg = R3.diagnose_background_error(nat, bg, lat, lon, lev)
    sig_b = {v: np.array([float(np.sqrt(np.nanmean((bg[v][k] - nat[v][k]) ** 2)))
                          for k in range(len(lev))]) for v in ("T", "q", "u", "v")}
    sigma = {v: (sig_b[v], np.full(len(lev), OERR[v])) for v in sig_b}
    del d
    gc.collect()
    return nat, bg, obs, lat, lon, lev, dg, sigma, above, nbelow


def impact_mask(ac_pos, lat, lon, lev, r=IMPACT_R, dlnp=IMPACT_LNP):
    from scipy.spatial import cKDTree

    def _xy(la, lo):
        return np.column_stack([np.asarray(la, float) * 111195.0,
                                np.asarray(lo, float) * 111195.0 *
                                np.cos(np.deg2rad(np.asarray(la, float)))])

    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    gxy = _xy(LAT.ravel(), LON.ravel())
    m = np.zeros((len(lev), LAT.size), bool)
    lp = np.log(np.asarray(lev, float))
    lpo = np.log(ac_pos[:, 0])
    for k in range(len(lev)):
        sel = np.abs(lpo - lp[k]) < dlnp
        if sel.sum() == 0:
            continue
        tree = cKDTree(_xy(ac_pos[sel, 1], ac_pos[sel, 2]))
        cnt = tree.query_ball_point(gxy, r, return_length=True)
        m[k] = np.asarray(cnt, int) > 0
    return m.reshape(len(lev), len(lat), len(lon))


def rmse(a, b, mask=None):
    dv = (np.asarray(a, float) - np.asarray(b, float))
    if mask is not None:
        dv = dv[mask]
    return float(np.sqrt(np.nanmean(dv ** 2)))


CONFIGS = [
    ("A  sfc+raob            无飞机", ("sfc", "raob"), ()),
    ("B  +ac 单变量(湿度盲)", ("sfc", "raob", "ac"), ()),
    ("C  B+T2V 静力-地转", ("sfc", "raob", "ac"), ("T2V",)),
    ("D  B+V2q 水汽平流 ★", ("sfc", "raob", "ac"), ("V2q",)),
    ("E  B+全耦合", ("sfc", "raob", "ac"), ("T2V", "V2q")),
]


def make_rec(lat, lon, lev, dg, sigma, **kw):
    return R3.Reconstructor(lat, lon, lev, L_h=dg["L_h1"], L_z=dg["L_z"],
                            L_h2=dg["L_h2"], w2=dg["w2"], sigma=sigma,
                            max_obs=900, vert_corr=dg.get("vert_corr"), **kw)


def main():
    t0 = time.time()
    print("=" * 86)
    print("OSSE（原生 21 层 + 地形遮蔽 · 谷地填充）：重新标定与检验")
    print("=" * 86)

    print("\n[1] 载入标定时次 …", flush=True)
    cal = []
    for dte, cyc in CAL_TIMES:
        try:
            cal.append(load_case(dte, cyc))
            print("    %s %sz  OK" % (dte, cyc), flush=True)
        except Exception as e:
            print("    %s %sz  跳过 %r" % (dte, cyc, e)[:70], flush=True)
    if not cal:
        raise SystemExit("无可用标定时次")
    lat, lon, lev = cal[0][3], cal[0][4], cal[0][5]
    ngrid = len(lat) * len(lon) * len(lev)
    print("    网格 %dx%d  层数 %d  全域 %d 格点"
          % (len(lat), len(lon), len(lev), ngrid), flush=True)
    nbg = int(np.median([c[9] for c in cal]))
    print("    每时次地面以下格点 = %d（占 %.1f%%）← 这些在遮蔽口径下被剔除"
          % (nbg, 100.0 * nbg / ngrid), flush=True)

    print("\n[2] 标定耦合系数（单变量 OI 只算一次）…", flush=True)
    nat, bg, obs, _, _, _, dg, sg, above, _ = cal[0]
    rec = make_rec(lat, lon, lev, dg, sg)
    uni = rec.univariate(bg, obs, use=("sfc", "raob", "ac"))
    msk = impact_mask(obs["ac"][1], lat, lon, lev) & above
    bi = {v: rmse(bg[v], nat[v], msk) for v in ("T", "q", "u", "v")}
    q0 = rmse(uni["q"], nat["q"], msk)
    u0 = rmse(uni["u"], nat["u"], msk)
    best, scan = None, []
    for aT in (0.0, 0.3, 0.6, 1.0):
        for aq in (0.0, 0.5, 1.0, 1.5):
            for tau in (1.0, 2.0, 3.0):
                rec.alpha_T2V, rec.alpha_V2q, rec.tau_h = aT, aq, tau
                r = rec.apply_coupling(uni, bg, coupling=("T2V", "V2q"))
                s = ((q0 - rmse(r["an"]["q"], nat["q"], msk)) / bi["q"] +
                     (u0 - rmse(r["an"]["u"], nat["u"], msk)) / bi["u"])
                scan.append(dict(alpha_T2V=aT, alpha_V2q=aq, tau_h=tau, score=s))
                if best is None or s > best[0]:
                    best = (s, aT, aq, tau)
                del r
    _, A_T2V, A_V2Q, TAU = best
    print("    标定结果: alpha_T2V=%.2f  alpha_V2q=%.2f  tau_h=%.1f  (评分 %+.3f%%)"
          % (A_T2V, A_V2Q, TAU, best[0] * 100), flush=True)
    del cal, nat, bg, obs, uni, rec, msk
    gc.collect()

    print("\n[3] 独立检验时次 …", flush=True)
    rows = []
    for dte, cyc in VAL_TIMES:
        try:
            nat, bg, obs, lat, lon, lev, dg, sg, above, nb = load_case(dte, cyc)
        except Exception as e:
            print("    %s %sz 跳过 %r" % (dte, cyc, str(e)[:40]), flush=True)
            continue
        imp = impact_mask(obs["ac"][1], lat, lon, lev)
        am = imp & above
        rec = make_rec(lat, lon, lev, dg, sg)
        rec.alpha_T2V, rec.alpha_V2q, rec.tau_h = A_T2V, A_V2Q, TAU
        base = {v: rmse(bg[v], nat[v], am) for v in ("T", "q", "u", "v")}
        base_all = {v: rmse(bg[v], nat[v], above) for v in ("T", "q", "u", "v")}
        entry = dict(time="%s%sz" % (dte, cyc), n_impact=int(am.sum()),
                     n_above=int(above.sum()), n_below=nb)
        for nm, use, cpl in CONFIGS:
            r = rec.run(bg, obs, use=use, coupling=cpl)
            an = r["an"]
            for v in ("T", "q", "u", "v"):
                entry["%s|%s" % (nm, v)] = rmse(an[v], nat[v], am)
                entry["ALL|%s|%s" % (nm, v)] = rmse(an[v], nat[v], above)
            del r, an
            gc.collect()
        for v in ("T", "q", "u", "v"):
            entry["bg|%s" % v] = base[v]
            entry["bgALL|%s" % v] = base_all[v]
        rows.append(entry)
        print("    %s %sz 完成（影响区 %d 格点 / 地面以上 %d）"
              % (dte, cyc, entry["n_impact"], entry["n_above"]), flush=True)
        del nat, bg, obs, rec, imp, am
        gc.collect()

    def med(k):
        v = [r[k] for r in rows if k in r]
        return float(np.median(v)) if v else float("nan")

    print("\n" + "=" * 86)
    print("【表】相对背景场的 RMSE 降幅 —— 观测影响区 ∩ 地面以上")
    print("=" * 86)
    keys = ("T", "q", "u", "v")
    print("%-30s %11s %11s %11s %11s"
          % ("配置", *keys))
    print("%-30s " % "(背景场绝对值)" +
          "  ".join("%11.4g" % med("bg|%s" % k) for k in keys))
    for nm, _, _ in CONFIGS:
        print("%-30s " % nm +
              "  ".join("%10.2f%%" % (100 * (med("bg|%s" % k) -
                                             med("%s|%s" % (nm, k))) /
                                      med("bg|%s" % k)) for k in keys))

    print("\n【对照】整个地面以上区域（含无观测区）")
    print("%-30s " % "(背景场绝对值)" +
          "  ".join("%11.4g" % med("bgALL|%s" % k) for k in keys))
    for nm, _, _ in CONFIGS:
        print("%-30s " % nm +
              "  ".join("%10.2f%%" % (100 * (med("bgALL|%s" % k) -
                                             med("ALL|%s|%s" % (nm, k))) /
                                      med("bgALL|%s" % k)) for k in keys))

    qa = med("A  sfc+raob            无飞机|q")
    qb = med("B  +ac 单变量(湿度盲)|q")
    qd = med("D  B+V2q 水汽平流 ★|q")
    print("\n" + "=" * 86)
    print("内部一致性（飞机不报湿度 ⟹ A 与 B 的湿度 RMSE 必须完全相同）")
    print("=" * 86)
    print("  A 无飞机          q RMSE = %.9g" % qa)
    print("  B 加飞机·单变量   q RMSE = %.9g" % qb)
    print("  |B−A|/A = %.4f%%   （应 ≈ 0）" % (100 * abs(qb - qa) / qa))
    print("  D 加飞机·水汽平流 q RMSE = %.9g" % qd)
    print("  从湿度盲观测榨取的湿度信息 = %+.4f 个百分点（相对背景）"
          % (100 * (qa - qd) / med("bg|q")))
    print("  相对单变量分析的再改善 = %+.4f%%" % (100 * (qb - qd) / qb))

    out = dict(
        meta=dict(bbox=list(BBOX), cal=CAL_TIMES, val=VAL_TIMES,
                  coarse=COARSE, noise=NOISE, impact_r=IMPACT_R,
                  terrain_mask="valley-fill + above-ground only",
                  n_below_ground_median=nbg, ngrid=ngrid),
        calibration=dict(alpha_T2V=A_T2V, alpha_V2q=A_V2Q, tau_h=TAU,
                         score=best[0], scan=scan),
        rows=rows,
        consistency=dict(q_A=qa, q_B=qb, q_D=qd,
                         rel_BA=abs(qb - qa) / qa,
                         gain_pp=100 * (qa - qd) / med("bg|q"),
                         gain_rel=100 * (qb - qd) / qb),
    )
    with open("data/step09_osse_masked.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step09_osse_masked.json   用时 %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
