# -*- coding: utf-8 -*-
"""step20 —— 端到端闭合：三维重构 → 飞行路径气象风险。

为什么必须有这一步
------------------
到 step19 为止，本项目是**两段各自成立、但没有接起来**的证据：

  左段（重构）  OSSE/EnOI 证明同化能把背景场 RMSE 降 6–8.5%（§12.4）
  右段（风险）  step13/13b 用客观 EDR 标定出航迹概率产品（§12.6）

但右段算在**分析场**上，左段验证的是**背景场**。
"重构好了，风险产品会不会跟着好？好多少？" —— 这恰恰是
"风场—温度—湿度三维精细场实时重构 + 飞行路径气象风险评估"
这个命题能不能成立的关键。

设计
----
用 OSSE 自带的**真值场**（f000 分析）当参照：

    x_nat  真值（同化系统的最优估计）
    x_bg   背景（f024 预报）      ← 重构**前**
    x_an   背景 + EnOI 增量        ← 重构**后**

每个场独立走一遍**已标定好的**产品链路
（stability_fields → route_probability(subset_policy="switch")），
得到 P_nat / P_bg / P_an，然后比较。

⚠ 两个踩过的坑（第一版就栽在这两个上）
--------------------------------------
**坑 1：不能用 Brier(·, 1[P_nat ≥ P*])。**
硬标签与连续概率之间有固有失配：即使拿真值自己的概率去打，
Brier 也 ≈ p(1−p) 量级（实测 0.184），而各场之间只差第 4 位小数
（0.19218 vs 0.19232）⇒ 判据被失配淹没，根本分辨不出来。
改用**连续指标** RMSE(P_x − P_nat) 与相关系数。

**坑 2：评估域不能取全网格。**
飞机观测集中在航路上 ⇒ 增量也只在**飞机观测影响区**里非零
（实测该区仅占全网格约 1/10）。全网格评估把信号稀释一个量级。
主评估域必须是 `S10.impact_mask(飞机观测位置)` —— 也就是
"同化**够得着**的地方"，这才是命题关心的域。
全网格作为稀释对照一并报告，用来说明稀释了多少。

**安慰剂对照不可省**（step10 的教训：约一半"增益"是平滑冒领的）：
  smooth   bg + 高斯平滑后的增量（保留平滑、去掉结构）
  shuffle  bg + 空间随机置换后的增量（保留增幅、去掉一切结构）
只有 an 显著优于两者，才能说增益来自同化而不是平滑。

判据口径：所有指标**逐个算例算百分比、再跨算例取中位**
（step10 修正过的聚合口径）。
"""
import gc
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
import step10_realerror_osse as S10
import step12_enoi as S12

from avwx import calib as CB
from avwx import enoi as EN
from avwx import risk as RK
from avwx import verify as VF

OUT = "data/step20_recon_to_risk.json"
TAG, EDR, THR = "light", "max", 0.10

# 落在 OSSE 区域 (100–130°E, 15–40°N) 内的航线，用于产品形态演示
ROUTE = ((31.23, 121.47), (22.31, 113.91))     # 上海 → 香港
ROUTE_LEV = 250.0
NPT = 120
CRUISE = (200.0, 400.0)          # 巡航带：气压 ∈ [200, 400] hPa

CAL = CB.load()


def diagP(fld, lat, lon, lev):
    """场 → 标定概率（产品端的完整链路，一步不少）。"""
    dg = RK.stability_fields(dict(T=fld["T"], q=fld["q"], u=fld["u"],
                                  v=fld["v"], z=fld["z"]),
                             lat, lon, lev)
    p = CB.route_probability(dg["Ri_dry"], dg["Ri_moist"], dg["Ri_dk"],
                             dg["rh"], CAL, tag=TAG, edr=EDR,
                             subset_policy="switch")["p_point"]
    return dg, p


def cont_scores(px, pt):
    """连续指标：对**真值概率**而非硬标签（见坑 1）。"""
    a = np.asarray(px, float)
    b = np.asarray(pt, float)
    e = a - b
    return dict(rmse=float(np.sqrt(np.nanmean(e ** 2))),
                bias=float(np.nanmean(e)),
                corr=float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else np.nan)


def decision(px, pt, thr=THR):
    """决策指标：以真值产品的行动判定为参照。"""
    fx = np.asarray(px, float) >= thr
    ft = np.asarray(pt, float) >= thr
    hit = int(np.sum(fx & ft))
    fa = int(np.sum(fx & ~ft))
    miss = int(np.sum(~fx & ft))
    ts = hit / max(hit + fa + miss, 1)
    pod = hit / max(hit + miss, 1)
    far = fa / max(hit + fa, 1)
    return dict(ts=float(ts), pod=float(pod), far=float(far),
                rate=float(np.mean(fx)), truth_rate=float(np.mean(ft)))


def route_profile(fields, lat, lon, lev, plev=ROUTE_LEV, n=NPT):
    (la1, lo1), (la2, lo2) = ROUTE
    qlat = np.linspace(la1, la2, n)
    qlon = np.linspace(lo1, lo2, n)
    qlev = np.full(n, plev)
    out = {}
    for nm, fld in fields.items():
        dg = RK.stability_fields(dict(T=fld["T"], q=fld["q"], u=fld["u"],
                                      v=fld["v"], z=fld["z"]),
                                 lat, lon, lev)

        def at(k):
            return VF.vert_interp(
                VF.bilinear_cols(np.asarray(dg[k], float), lat, lon,
                                 qlat, qlon), lev, qlev)
        out[nm] = CB.route_probability(at("Ri_dry"), at("Ri_moist"),
                                       at("Ri_dk"), at("rh"), CAL,
                                       tag=TAG, edr=EDR,
                                       subset_policy="switch")["p_point"]
        del dg
    return out


def main():
    t0 = time.time()
    print("=" * 98)
    print("step20  端到端：三维重构 → 航迹风险概率（OSSE 真值为参照）")
    print("=" * 98)
    if CAL is None:
        print("标定文件缺失，先跑 step13")
        return
    print("标定 tag=%s edr=%s  行动阈值 P*=%.2f  巡航带 %.0f–%.0f hPa"
          % (TAG, EDR, THR, CRUISE[0], CRUISE[1]), flush=True)

    cases = {}
    for t in S12.TARGETS:
        try:
            cases[t] = S12.load_case(t)
            print("载入 %s%sz" % (t[0], t[1]), flush=True)
        except Exception as e:
            print("载入 %s 失败: %s" % (t, str(e)[:90]), flush=True)

    rows, details = [], {}
    for target in list(cases):
        X = cases[target]
        lat, lon, lev = X["lat"], X["lon"], X["lev"]
        others = [t for t in cases if t != target]
        print("\n" + "-" * 98)
        print("目标 %s%sz" % (target[0], target[1]), flush=True)

        # ⚠ 踩坑（第一版就栽在这里）：一开始只用了 `member_pairs(others)`，
        #   也就是 step12c 里标记为 **A 仅其它日期** 的**最弱**配置 —— 那个配置
        #   早被证明无效（q −0.76%），拿它去做端到端检验，得到的"风险产品变差"
        #   其实是"同化根本没起作用"，不是产品的性质。
        #   必须用 step12c 验证有效的 **B：其它日期 + 同日异时效**。
        def pair_for(ts, lead):
            nat = S10.prepare(S10.AG.load(S10.native_path(ts[0], ts[1], 0),
                                          bbox=S10.BBOX))
            b = S10.bg_cycle(ts[0], ts[1], lead)
            p = S10.native_path(b[0], b[1], lead)
            if not os.path.exists(p):
                return None
            bg = S10.prepare(S10.AG.load(p, bbox=S10.BBOX))
            out = ({v: nat["f"][v] for v in ("T", "q", "u", "v")},
                   {v: bg["f"][v] for v in ("T", "q", "u", "v")})
            del nat, bg
            gc.collect()
            return out

        base = S12.member_pairs(others)
        same_day = [q for q in (pair_for(target, ld) for ld in (6, 12))
                    if q is not None]
        pairs = base + same_day                       # 配置 B ★
        if len(pairs) < 3:
            print("  集合成员不足（%d），跳过" % len(pairs))
            continue
        print("  集合成员 %d 个（其它日期 %d + 同日异时效 %d）"
              % (len(pairs), len(base), len(same_day)), flush=True)
        sb = S12.sigma_from_pairs(pairs)
        members, _ = EN.build_members(pairs, sb, clip={"q": 0.02})
        eobs = S12.to_enoi_obs(X["obs"], use=("sfc", "raob", "ac"),
                               lat=lat, lon=lon, lev=lev)
        E = EN.EnOI(members, lat, lon, lev, sigma_b=sb,
                    tiles=(6, 5), tile_pad_km=350.0, max_obs=1000)
        an = E.update(X["bg"], eobs, S12.OERR)
        del E, members, pairs
        gc.collect()

        # ⚠ z（位势）不是 EnOI 控制变量，分析场里没有它，必须显式透传。
        d = {v: an[v] - X["bg"][v] for v in S12.KEYS}
        dlat = float(np.mean(np.diff(lat))) * 111195.0
        g = max(float(X["dg"].get("L_h1", 124e3)) / dlat, 0.5)
        sm, sh = dict(X["bg"]), dict(X["bg"])
        rng = np.random.default_rng(23)
        for v in S12.KEYS:
            sm[v] = X["bg"][v] + gaussian_filter(d[v], sigma=(0, g, g),
                                                 mode="nearest")
            dd = np.asarray(d[v], float).copy().reshape(d[v].shape[0], -1)
            for k in range(dd.shape[0]):
                rng.shuffle(dd[k])
            sh[v] = X["bg"][v] + dd.reshape(d[v].shape)
        # 真正的"只是平滑"对照：**直接平滑背景场本身**。
        # ⚠ 前一个版本只做了 smooth(增量)，但增量本身已经是光滑的（它是
        #   光滑集合成员的线性组合），再平滑几乎不变 ⇒ 该对照与 an 几乎重合
        #   （实测巡航带 +1.62% vs +1.77%），根本区分不出"平滑"与"结构"。
        #   要回答"增益是不是只来自平滑"，必须平滑**背景场**。
        smbg = {v: gaussian_filter(X["bg"][v], sigma=(0, g, g),
                                  mode="nearest") for v in S12.KEYS}
        smbg["z"] = X["bg"]["z"]
        # 还能不能叠加？把**分析场**再做同样的平滑 —— 若优于两者，
        # 则产品端的正确配方是"同化 + 尺度分离"，而不是二选一。
        sman = {v: gaussian_filter(an[v], sigma=(0, g, g), mode="nearest")
                for v in S12.KEYS}
        sman["z"] = X["bg"]["z"]
        for f in (an, sm, sh):
            f.setdefault("z", X["bg"]["z"])

        # ---- 自检：原始场到底改善没有（step20b 教训）----
        # 若这一步不改善，后面的风险产品比较**没有意义**，直接跳过。
        imp0 = S10.impact_mask(X["obs"]["ac"][1], lat, lon, lev) & X["above"]
        pcts = {}
        for v in S12.KEYS:
            rb = S10.rmse(X["bg"][v], X["nat"][v], imp0)
            ra = S10.rmse(an[v], X["nat"][v], imp0)
            pcts[v] = 100 * (rb - ra) / max(rb, 1e-30)
        print("  [自检] 原始场 RMSE 降幅%%：%s"
              % "  ".join("%s %+.2f" % (v, pcts[v]) for v in S12.KEYS),
              flush=True)
        n_bad = sum(1 for v in S12.KEYS if pcts[v] <= 0)
        if n_bad >= 3:
            print("  ⚠ 原始场未改善（%d/4 个变量变差）⇒ 本次同化无效，"
                  "风险产品比较不具解释力，跳过该算例" % n_bad, flush=True)
            del an, sm, sh, d, X
            gc.collect()
            continue

        fields = dict(truth=X["nat"], bg=X["bg"], an=an, smooth=sm,
                      shuffle=sh, smooth_bg=smbg, smooth_an=sman)
        P = {}
        for nm, fld in fields.items():
            dg, P[nm] = diagP(fld, lat, lon, lev)
            del dg
            gc.collect()

        above = X["above"]
        imp = S10.impact_mask(X["obs"]["ac"][1], lat, lon, lev) & above
        cruise = ((lev[:, None, None] >= CRUISE[0])
                  & (lev[:, None, None] <= CRUISE[1])) & above
        fin = above & np.isfinite(P["truth"])
        for nm in P:
            fin = fin & np.isfinite(P[nm])
        doms = dict(全网格=fin, 飞机影响区=(fin & imp),
                    巡航带=(fin & cruise),
                    影响区且巡航=(fin & imp & cruise))

        rec = dict(target="%s%sz" % (target[0], target[1]),
                   n_grid=int(fin.sum()), n_imp=int((fin & imp).sum()))
        print("  全网格 %d 点，飞机影响区 %d 点（占 %.1f%%）"
              % (rec["n_grid"], rec["n_imp"],
                 100 * rec["n_imp"] / max(rec["n_grid"], 1)), flush=True)

        for dn, dm in doms.items():
            if int(dm.sum()) < 500:
                print("  %-12s 样本不足 (%d)" % (dn, int(dm.sum())))
                continue
            pt = P["truth"][dm]
            sc = {nm: cont_scores(P[nm][dm], pt) for nm in P}
            dec = {nm: decision(P[nm][dm], pt) for nm in P}
            base = sc["bg"]["rmse"]
            print("  [%s] n=%d  真值行动占比 %.2f%%"
                  % (dn, int(dm.sum()), 100 * dec["truth"]["rate"]))
            print("      %-9s %12s %12s %10s | %8s %8s %8s"
                  % ("场", "RMSE(P−P*)", "降幅%", "corr", "TS", "POD", "FAR"))
            for nm in ("bg", "an", "smooth", "smooth_bg", "smooth_an",
                       "shuffle"):
                dr = 100 * (base - sc[nm]["rmse"]) / max(base, 1e-15)
                rec["drmse|%s|%s" % (dn, nm)] = dr
                rec["corr|%s|%s" % (dn, nm)] = sc[nm]["corr"]
                rec["ts|%s|%s" % (dn, nm)] = dec[nm]["ts"]
                rec["far|%s|%s" % (dn, nm)] = dec[nm]["far"]
                print("      %-9s %12.6f %12.3f %10.4f | %8.4f %8.4f %8.4f"
                      % (nm, sc[nm]["rmse"], dr, sc[nm]["corr"],
                         dec[nm]["ts"], dec[nm]["pod"], dec[nm]["far"]))
        rows.append(rec)

        prof = route_profile(dict(truth=X["nat"], bg=X["bg"],
                                  an=dict(an, z=X["bg"]["z"])),
                             lat, lon, lev)
        rbg = float(np.sqrt(np.nanmean((prof["bg"] - prof["truth"]) ** 2)))
        ran = float(np.sqrt(np.nanmean((prof["an"] - prof["truth"]) ** 2)))
        rec.update(route_rmse_bg=rbg, route_rmse_an=ran,
                   route_drmse=100 * (rbg - ran) / max(rbg, 1e-15))
        print("  航线 上海→香港 %.0f hPa：RMSE(P−P*) bg %.5f → an %.5f  "
              "降幅 %.2f%%" % (ROUTE_LEV, rbg, ran, rec["route_drmse"]),
              flush=True)
        del P, prof, an, sm, sh, d, X
        gc.collect()

    # ---------------- 汇总 ----------------
    def med(k):
        v = [r[k] for r in rows if np.isfinite(r.get(k, np.nan))]
        return float(np.median(v)) if v else float("nan")

    # ⚠ 先落盘再打印：上一版在汇总 print 里因格式串少一个占位符崩溃，
    #   导致 4 个算例算了 9 分钟的结果全丢。落盘必须在任何打印之前。
    _out = dict(tag=TAG, edr=EDR, thr=THR, cruise=CRUISE, n_case=len(rows),
                rows=rows,
                median={("%s|%s" % (dn, nm)): med("drmse|%s|%s" % (dn, nm))
                        for dn in ("全网格", "飞机影响区", "巡航带",
                                   "影响区且巡航")
                        for nm in ("an", "smooth", "smooth_bg", "smooth_an",
                                   "shuffle")},
                route_drmse_median=med("route_drmse"))
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(_out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s" % OUT, flush=True)

    print("\n" + "=" * 98)
    print("【汇总】跨 %d 个算例取中位（逐算例先算百分比再取中位）" % len(rows))
    print("=" * 98)
    print("%-14s %10s %10s %10s %10s %10s %10s"
          % ("评估域", "an 降幅%", "平滑增量%", "平滑背景%", "平滑分析%",
             "shuffle%", "an−最强安慰剂"))
    for dn in ("全网格", "飞机影响区", "巡航带", "影响区且巡航"):
        a = med("drmse|%s|an" % dn)
        s1 = med("drmse|%s|smooth" % dn)
        s2 = med("drmse|%s|smooth_bg" % dn)
        s3 = med("drmse|%s|smooth_an" % dn)
        h = med("drmse|%s|shuffle" % dn)
        if not np.isfinite(a):
            continue
        print("%-14s %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f"
              % (dn, a, s1, s2, s3, h, a - max(s1, s2, s3, h)))
        print("%-14s 平滑后的分析场 smooth_an = %.3f%%（若 > an 且 > smooth_bg，"
              "则配方是'同化 + 尺度分离'）" % ("", s3))
    print("\n航线产品（上海→香港 %.0f hPa）：RMSE 降幅中位 %.2f%%"
          % (ROUTE_LEV, med("route_drmse")))

    out = dict(tag=TAG, edr=EDR, thr=THR, cruise=CRUISE, n_case=len(rows),
               rows=rows,
               median={("%s|%s" % (dn, nm)): med("drmse|%s|%s" % (dn, nm))
                       for dn in ("全网格", "飞机影响区", "巡航带",
                                  "影响区且巡航")
                       for nm in ("an", "smooth", "smooth_bg", "smooth_an",
                                  "shuffle")},
               route_drmse_median=med("route_drmse"))
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
