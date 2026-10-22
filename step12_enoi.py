# -*- coding: utf-8 -*-
"""step12 —— 把 V2q 接进同化框架：集合协方差 EnOI vs 双高斯静态 B。

用户关心的两件事：
  (1) V2q 作为**弱约束增量算子**，在业务网格上到底兑现多少净贡献；
  (2) 用**集合协方差**替代双高斯 B 矩阵之后，结论是变强还是被取代。

对应的wrf-jedi 语言：δx = B^{1/2} v；本项目原先的 B 是变量间块对角双高斯，
所以 " Windsor **跨变量**增量 Cov(u, q) 恒为零 —— 这就是湿度永远榨不出来的数学根源。
step10 用一个显式的物理算子 M_{V→q} 绕过了它（事后施加平流位移）；
本脚本则直接把 B 换成从**真实预报误差集合**估计的多变量协方差，
看纯统计的、数据驱动的跨变量协方差能否替代（或超越）手写物理算子。

试验设计：留一交叉验证（leave-one-case-out）
-------------------------------------------
对每个目标时次 t：
    真值 = GFS f000 分析(t)，背景 = GFS f024 预报(t)
    集合 = **其它目标时次**的预报误差样本（严格排除 t 自己，避免信息泄漏）
    σ_b  = 由其它时次的 f024 误差诊断

对比五种配置：
    A  OI  +地面+探空                     —— 基线
    B  OI  +飞机（湿度盲，单变量）
    D  OI  +飞机 + V2q 显式物理算子        —— step10 的路线
    G  EnOI +飞机（集合 B，跨变量协方差自动生效）  ★ 本脚本新增
    H  EnOI +飞机 + V2q

关键判据：**G 与 B 的湿度之差**，就是"数据驱动的跨变量协方差"单独贡献的部分；
拿它与 D−B（物理算子部分）相比，就能回答"要不要手写物理"。
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from avwx import enoi as EN
from avwx import recon3d as R3
from scipy.ndimage import gaussian_filter
import step10_realerror_osse as S10

BBOX = S10.BBOX
KEYS = S10.KEYS
OERR = S10.OERR
TARGETS = [("20261001", "00"), ("20261008", "12"),
           ("20261012", "18"), ("20261016", "00")]
LEADS_FOR_ENSEMBLE = (6, 12, 24)
BG_LEAD = 24
ALPHA_V2Q, TAU_V2Q = 0.5, 12.0       # step10 f024 标定值（未被本轮再标定）


def _empty():
    return []


def load_case(tspec, lead=BG_LEAD):
    bg = S10.bg_cycle(tspec[0], tspec[1], lead)
    return S10.load_pair(tspec, bg, lead)


def member_pairs(targets_except):
    """返回 [(analysis, forecast)] 的 pair 列表，用于构造 EnOI 集合。"""
    pairs = []
    for t in targets_except:
        nat = S10.prepare(S10.AG.load(S10.native_path(t[0], t[1], 0),
                                      bbox=BBOX))
        for lead in LEADS_FOR_ENSEMBLE:
            b = S10.bg_cycle(t[0], t[1], lead)
            p = S10.native_path(b[0], b[1], lead)
            if not os.path.exists(p):
                continue
            try:
                bg = S10.prepare(S10.AG.load(p, bbox=BBOX))
            except Exception as e:
                print("    跳过成员 %s f%03d: %s" % (t, lead, str(e)[:60]),
                      flush=True)
                continue
            a = {v: nat["f"][v] for v in ("T", "q", "u", "v")}
            f = {v: bg["f"][v] for v in ("T", "q", "u", "v")}
            pairs.append((a, f))
            del bg
        del nat
        gc.collect()
    return pairs


def sigma_from_pairs(pairs):
    """从集合样本估计逐变量的背景误差标准差（ scalar，用于 EnOI 缩放）。"""
    sb = {}
    for v in KEYS:
        vals = np.concatenate([np.r_[np.asarray(f[v] - a[v], float).ravel()]
                               for a, f in pairs])
        sb[v] = float(np.nanstd(vals))
    return sb


def to_enoi_obs(obs, use=("sfc", "raob", "ac"), superob=True,
                lat=None, lon=None, lev=None):
    """step10 的观测元组 -> EnOI 需要的 dict{var:(vals,pos)}。"""
    out = {}
    for v in KEYS:
        vals, poss = [], []
        for tag in use:
            if tag not in obs or obs[tag] is None:
                continue
            o, p = obs[tag]
            if v not in o:
                continue
            if tag == "ac" and v == "q":
                continue                      # 飞机不报湿度
            vals.append(np.asarray(o[v], float).ravel())
            poss.append(np.asarray(p, float))
        if not vals:
            continue
        vv = np.concatenate(vals)
        pp = np.vstack(poss)
        if superob and vv.size:
            try:
                vv2, pp2, _ = R3.superob(vv, pp, lat, lon, lev)
                vv, pp = np.asarray(vv2, float), np.asarray(pp2, float)
            except Exception:
                pass
        out[v] = (vv, pp)
    return out


def main():
    t0 = time.time()
    print("=" * 96)
    print("step12  EnOI（集合协方差） vs 双高斯静态 B —— 留一交叉验证")
    print("=" * 96)
    cases = {}
    for t in TARGETS:
        try:
            cases[t] = load_case(t)
            print("载入 %s%sz：网格 %dx%d×%d，地面以上 %.1f%%"
                  % (t[0], t[1], len(cases[t]["lat"]), len(cases[t]["lon"]),
                     len(cases[t]["lev"]),
                     100 * cases[t]["above"].sum() / cases[t]["above"].size),
                  flush=True)
        except Exception as e:
            print("载入 %s 失败: %s" % (t, str(e)[:100]), flush=True)

    rows = []
    detail = {}
    for target in list(cases):
        others = [t for t in cases if t != target]
        X = cases[target]
        lat, lon, lev = X["lat"], X["lon"], X["lev"]
        am = S10.impact_mask(X["obs"]["ac"][1], lat, lon, lev) & X["above"]
        print("\n" + "-" * 96)
        print("目标 %s%sz   集合来源：其它 %d 个时次 ×%s"
              % (target[0], target[1], len(others), LEADS_FOR_ENSEMBLE),
              flush=True)

        pairs = member_pairs(others)
        print("  集合成员 %d 个" % len(pairs), flush=True)
        if len(pairs) < 3:
            print("  成员不足，跳过")
            continue
        sb_diag = sigma_from_pairs(pairs)
        print("  σ_b 诊断: " + "  ".join("%s=%.4g" % (v, sb_diag[v])
                                        for v in KEYS), flush=True)
        members, _info = EN.build_members(pairs, sb_diag,
                                          clip={"q": 0.02})

        # ---------- OI 路线 ----------
        rec = S10.make_rec(lat, lon, lev, X["dg"], X["sigma"])
        rec.alpha_T2V, rec.alpha_V2q, rec.tau_h = 0.0, ALPHA_V2Q, TAU_V2Q
        uniA = rec.univariate(X["bg"], X["obs"], use=("sfc", "raob"))
        uniB = rec.univariate(X["bg"], X["obs"], use=("sfc", "raob", "ac"))
        coupled = rec.apply_coupling(uniB, X["bg"], coupling=("V2q",))["an"]

        # ---------- EnOI 路线 ----------
        eobsA = to_enoi_obs(X["obs"], use=("sfc", "raob"),
                            lat=lat, lon=lon, lev=lev)
        eobsB = to_enoi_obs(X["obs"], use=("sfc", "raob", "ac"),
                            lat=lat, lon=lon, lev=lev)
        E = EN.EnOI(members, lat, lon, lev, sigma_b=sb_diag,
                    tiles=(6, 5), tile_pad_km=350.0, max_obs=1000)
        anG = E.update(X["bg"], eobsB, OERR)
        anH = dict(anG)
        dU = anG["u"] - X["bg"]["u"]
        dV = anG["v"] - X["bg"]["v"]
        anH["q"] = np.clip(anH["q"] + ALPHA_V2Q * R3.moisture_displacement(
            dU, dV, X["bg"]["q"], lat, lon, tau_h=TAU_V2Q), 0, 0.06)
        del E, members
        gc.collect()

        cfgs = [("A  OI 地面+探空", uniA), ("B  OI +飞机(湿度盲)", uniB),
                ("D  OI +飞机 +V2q", coupled),
                ("G  EnOI +飞机 ★", anG), ("H  EnOI +飞机 +V2q", anH)]
        entry = dict(target="%s%sz" % target, n_impact=int(am.sum()),
                     n_member=len(pairs))
        for nm, an in cfgs:
            for v in KEYS:
                entry["%s|%s" % (nm, v)] = S10.rmse(an[v], X["nat"][v], am)
        for v in KEYS:
            entry["bg|%s" % v] = S10.rmse(X["bg"][v], X["nat"][v], am)
            entry["sigmab|%s" % v] = float(np.mean(X["sig_b"][v]))
        rows.append(entry)
        print("  %-20s " % "配置" + "  ".join("%9s" % v for v in KEYS))
        for nm, an in cfgs:
            print("  %-20s " % nm +
                  "  ".join("%8.2f%%" % (100 * (S10.rmse(X["bg"][v],
                                                         X["nat"][v], am) -
                                                S10.rmse(an[v], X["nat"][v],
                                                         am)) /
                                         S10.rmse(X["bg"][v], X["nat"][v], am))
                            for v in KEYS))
        del uniA, uniB, coupled, anG, anH, X
        gc.collect()

    # ---------------- 汇总 ----------------
    def med(k):
        v = [r[k] for r in rows if k in r]
        return float(np.median(v)) if v else float("nan")

    print("\n" + "=" * 96)
    print("【汇总】相对背景场的 RMSE 降幅 [%]，跨 %d 个目标时次取中位" % len(rows))
    print("=" * 96)
    names = ["A  OI 地面+探空", "B  OI +飞机(湿度盲)", "D  OI +飞机 +V2q",
             "G  EnOI +飞机 ★", "H  EnOI +飞机 +V2q"]
    print("%-22s %10s %10s %10s %10s" % ("配置", *KEYS))
    print("%-22s " % "(背景场绝对值)" +
          "  ".join("%10.4g" % med("bg|" + v) for v in KEYS))
    for nm in names:
        print("%-22s " % nm +
              "  ".join("%9.2f%%" % (100 * (med("bg|" + v) -
                                            med(nm + "|" + v)) /
                                     med("bg|" + v)) for v in KEYS))

    # 归因：q 的三种路径
    bgq = med("bg|q")
    q = {nm: 100 * (bgq - med(nm + "|q")) / bgq for nm in names}
    print("\n" + "=" * 96)
    print("【核心】湿度信息的三条路径（相对背景 q RMSE 的百分点）")
    print("=" * 96)
    print("  单变量 OI（+飞机）            = %+6.2f pp  ← 湿度盲，应该 ≈ 0" % q["B  OI +飞机(湿度盲)"])
    print("  显式物理算子 V2q              = %+6.2f pp  ← step10 路线" % q["D  OI +飞机 +V2q"])
    print("  集合协方差（数据驱动）        = %+6.2f pp  ← 本脚本新增 ★"
          % q["G  EnOI +飞机 ★"])
    print("  集合协方差 + V2q              = %+6.2f pp" % q["H  EnOI +飞机 +V2q"])
    attrib = dict(single=q["B  OI +飞机(湿度盲)"],
                  explicit_phys=q["D  OI +飞机 +V2q"],
                  ensemble_cov=q["G  EnOI +飞机 ★"],
                  ensemble_plus_phys=q["H  EnOI +飞机 +V2q"],
                  ensemble_over_explicit=(q["G  EnOI +飞机 ★"] /
                                          q["D  OI +飞机 +V2q"]
                                          if abs(q["D  OI +飞机 +V2q"]) > 1e-9
                                          else float("nan")))
    print("  → 集合协方差 / 显式物理算子 = %.2f" % attrib["ensemble_over_explicit"])

    out = dict(meta=dict(targets=[list(t) for t in TARGETS], bg_lead=BG_LEAD,
                         ensemble_leads=list(LEADS_FOR_ENSEMBLE),
                         alpha_v2q=ALPHA_V2Q, tau_v2q=TAU_V2Q,
                         note="leave-one-case-out; ensemble members exclude target"),
               rows=rows, attribution=attrib)
    with open("data/step12_enoi.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step12_enoi.json   用时 %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
