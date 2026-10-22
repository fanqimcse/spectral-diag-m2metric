# -*- coding: utf-8 -*-
"""
step04 —— 飞行路径气象风险评估：**湿度如何改变稳定度判定**
==========================================================

科学问题
--------
晴空湍流（CAT）的业务诊断普遍采用**干**的静力稳定度

    N_d² = (g/T)(dT/dz + g/c_p)        （Prosser et al. 2023 *Nature* 的指标族亦然）

完全不含湿度项。但 CAT 发生在无云区——气块受扰上升不凝结、比湿守恒，
浮力由密度（虚温）决定，正确的未饱和稳定度是 N_v²（虚温形式）。

由 thermo 模块可直接解析得到本项目的一条**可证伪解析恒等式**：

    N_v² − N_d² = 0.608 · g · (dq/dz) / (1 + 0.608 q)          (*)

（本步骤数值核验到 1e-14。）由于对流层中 dq/dz < 0，恒有 N_v² < N_d²。

**(*) 式不含 S²** —— 这是整个课题的关键，它带来两条后果：

  1. 一个**彻底绕开垂直分辨率问题**的指标（主指标 A）；
  2. 一个**受分辨率严重污染**的指标（K–H 涌现率），必须显式处理。

两条路线
--------
【主指标 A｜虚温不稳定漏判率】
    晴空 + 干口径判"静力稳定"（N_d²>0）+ 湿口径判"不稳定"（N_v²<0）。
    这是**符号翻转**而非阈值穿越，且完全不依赖 S²，
    因此不受分析场垂直分辨率缺陷影响。它度量的是干口径的**定性错误**。

【次指标 B–E｜稳定度削弱 δ 及其业务后果】
    δ ≡ (N_d²−N_v²)/N_d²，且恒有 Ri_moist = Ri_dry·(1−δ)（两口径共用 S²）。
    由此得到与分辨率无关的业务结论：
      B. δ 的量值与高度结构；
      C. 阈值等价误差：干口径 + 0.25 ⟺ 湿口径 + 0.25/(1−δ)，即干算法
         实际使用了一个**偏松**的阈值；
      D. 风险排序翻转率：CAT 通告按"最危险前 p%"圈定时，换口径后区域变多少；
      E. K–H 涌现率对次网格切变放大倍数 α 的敏感性（诚实桥接）。

必须诚实说明的三点
------------------
1. **垂直分辨率**：GFS 分析场 11 层，层厚 0.7–2.7 km；而 K–H 判据 Ri<0.25
   是针对湍流尺度（∂z ~ 10–100 m）定义的。故网格 Ri 被系统性高估 1–2 个
   量级，全域几乎找不到 Ri<0.25 的格点。所有依赖 S² 的结论都受此限制。
2. **近中性层的发散**：δ 的分母是 N_d²，N_d²→0 时 δ 发散（实测达 10²）。
   本项目实测"涌现格点"的 N_d² 中位 3.4e-6，比全域中位 1.3e-4 小 38 倍——
   涌现完全由近中性层主导。因此必须施加稳定度门槛 N²>1e-5 s-2
   （见 risk.N2_FLOOR 的理由），否则会把"近中性层的湿不稳定"
   误报为"CAT 涌现"。这是本步骤最重要的一处自我纠错。
3. **模式场非观测**：三维场取自 GFS 分析而非观测，故本步骤的绝对数值是
   "模式的气候学统计"，回答的是"若业务算法改用湿口径，判定会改变多少"。
   与之相对，端点密度高度惩罚（step01/step02）基于 GHCNh 观测，是独立
   且更硬的证据。
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avwx import gfs, recon3d as R3, risk as RK, thermo as TH

BBOX = (100, 130, 15, 40)
TIMES = [("20261019", "06"), ("20261019", "12"), ("20261019", "18"),
         ("20261020", "00"), ("20261020", "06"), ("20261018", "12")]
OUT = "data/step04_risk.json"
RI_CRIT = 0.25
RH_CLEAR = 80.0
N2_FLOOR = RK.N2_FLOOR
PHASE_HPA = (850.0, 500.0, 250.0)          # 爬升/下降、低巡航、主巡航
BINS = [(0, 1e-6), (1e-6, 5e-6), (5e-6, 1e-5), (1e-5, 3e-5),
        (3e-5, 5e-5), (5e-5, 1e-4), (1e-4, 2e-4), (2e-4, 1e9)]


def route_delta(diag, lat, lon, lev, p0, p1, hpa, n=60):
    """沿大圆在**固定气压层**上取 δ 的中位（用于分飞行阶段比较）。"""
    qlat, qlon = RK.great_circle(p0, p1, n)
    qlon = qlon % 360.0
    qlev = np.full(n, float(hpa))
    d = RK.interpolate_field(diag["delta"], lat, lon, lev, qlat, qlon, qlev)
    nd = RK.interpolate_field(diag["Nd2"], lat, lon, lev, qlat, qlon, qlev)
    ok = np.isfinite(d) & (nd > N2_FLOOR)
    return float(np.nanmedian(d[ok])) if ok.any() else np.nan


def main():
    t0 = time.time()
    per_time, per_level, bins, routes = [], {}, {b: [0, 0] for b in BINS}, []
    ident_err = []

    for i, (dte, cyc) in enumerate(TIMES):
        print(f"[step04] ({i+1}/{len(TIMES)}) 载入 {dte} {cyc}Z …", flush=True)
        try:
            nat = gfs.load(gfs.fetch(date=dte, cycle=cyc, bbox=BBOX))
        except Exception as e:
            print(f"   跳过（{repr(e)[:70]}）", flush=True)
            continue
        lat, lon, lev = nat["lat"], nat["lon"], nat["level"]
        f = dict(T=nat["T"], q=nat["q"], u=nat["u"], v=nat["v"], z=nat.get("z"))
        diag = RK.stability_fields(f, lat, lon, lev)
        del nat

        # ---- 地形遮蔽：剔除位于地面以下的等压面 ----
        # NOMADS 子集服务的等压面场**不做地形遮蔽**（本项目实测 1000 hPa
        # 位势高度全域仅 43–178 m，连海拔 ~1900 m 的昆明也只报 85 m），
        # 高原地区低层是地下外推值。判据：p ≤ p_sfc 才在地面之上。
        try:
            ps, _, _ = gfs.load_sfc_pres(gfs.fetch_sfc_pres(dte, cyc, bbox=BBOX))
            ag = gfs.above_ground_mask(lev, ps)
        except Exception as e:
            print("   地面气压不可用(%r)，退化为不遮蔽" % (e,), flush=True)
            ag = np.ones_like(diag["Nd2"], bool)
        n_sub = int((~ag).sum())

        # 遮蔽后的工作副本：把地下点置为"必然不合格"的哨兵值，
        # 于是 valid_mask / rank_turnover / emergence_vs_alpha 会自动排除它们，
        # 无需改动 risk 模块的接口。
        dg = dict(diag)
        dg["Nd2"] = np.where(ag, diag["Nd2"], -1.0)
        dg["Nv2"] = np.where(ag, diag["Nv2"], -1.0)
        dg["delta"] = np.where(ag, diag["delta"], np.nan)
        dg["S2"] = np.where(ag, diag["S2"], 0.0)
        dg["rh"] = np.where(ag, diag["rh"], 999.0)

        # ---- 解析恒等式 (*) 的数值核验 ----
        ana = 0.608 * TH.G * dg["dqdz"] / (1 + 0.608 * f["q"])
        num = dg["Nv2"] - dg["Nd2"]
        m = np.isfinite(num) & (np.abs(num) > 1e-9) & ag
        ident_err.append(float(np.median(np.abs(ana[m] - num[m]) / np.abs(num[m]))))

        # ---- 主指标 A：被漏判的虚温不稳定层 ----
        A = RK.missed_virtual_instability(dg, RH_CLEAR)
        base = (dg["Nd2"] > 0) & (dg["rh"] < RH_CLEAR)
        ok = RK.valid_mask(dg, RH_CLEAR, N2_FLOOR)
        dl = dg["delta"][ok]
        rs, amp = RK.ri_threshold_equivalent(dl, RI_CRIT)
        to = RK.rank_turnover(dg, fracs=(0.01, 0.05, 0.10),
                              rh_clear=RH_CLEAR, n2_floor=N2_FLOOR)
        ev = RK.emergence_vs_alpha(dg, RI_CRIT, RH_CLEAR, n2_floor=N2_FLOOR)

        per_time.append(dict(
            time=f"{dte}{cyc}", n_clear_stable=int(base.sum()),
            n_missed=int(A.sum()), missed_rate=float(A.sum() / max(base.sum(), 1)),
            n_subground=int(n_sub), subground_frac=float(n_sub / ag.size),
            delta_p50=float(np.percentile(dl, 50)), delta_p75=float(np.percentile(dl, 75)),
            delta_p95=float(np.percentile(dl, 95)),
            ri_star_p50=float(np.percentile(rs, 50)), ri_star_p95=float(np.percentile(rs, 95)),
            turnover_1=to[0.01][0], turnover_5=to[0.05][0], turnover_10=to[0.10][0],
            jaccard_5=to[0.05][1],
            emergence_alpha1=ev[0][1], emergence_peak=max(e[1] for e in ev),
            alpha_peak=max(ev, key=lambda e: e[1])[0],
            emergence_curve=[[float(a), float(e), float(dd)] for a, e, dd in ev],
            n2_floor=N2_FLOOR))

        # ---- 按 N_d² 分箱统计漏判率 ----
        for lo, hi in BINS:
            m = (dg["Nd2"] > lo) & (dg["Nd2"] <= hi) & (dg["rh"] < RH_CLEAR)
            bins[(lo, hi)][0] += int(m.sum())
            bins[(lo, hi)][1] += int(A[m].sum())

        # ---- 逐层 ----
        for k in range(len(lev)):
            key = f"{lev[k]:.0f}"
            rec = per_level.setdefault(key, dict(missed=[], delta=[], n=[]))
            mb = (dg["Nd2"][k] > 0) & (dg["rh"][k] < RH_CLEAR)
            mk = mb & (dg["Nv2"][k] < 0)
            ob = mb & (dg["Nd2"][k] > N2_FLOOR)
            rec["missed"].append(float(mk.sum() / max(mb.sum(), 1)))
            rec["delta"].append(float(np.median(dg["delta"][k][ob])) if ob.any() else np.nan)
            rec["n"].append(int(mb.sum()))

        # ---- 航迹（区域内真实机场对）----
        for a, b in R3.ROUTE_PAIRS:
            p0, p1 = R3.AIRPORTS[a], R3.AIRPORTS[b]
            rd = {f"delta_{int(h)}": route_delta(dg, lat, lon, lev, p0, p1, h)
                  for h in PHASE_HPA}
            routes.append(dict(time=f"{dte}{cyc}", dep=a, arr=b, **rd))

        print("     漏判率=%.4f%%  δ中位=%.4f  翻转(前1%%)=%.2f%%  涌现(α=1)=%.4f%%  "
              "地下层剔除=%.1f%%"
              % (per_time[-1]["missed_rate"] * 100, per_time[-1]["delta_p50"],
                 per_time[-1]["turnover_1"] * 100, per_time[-1]["emergence_alpha1"] * 100,
                 per_time[-1]["subground_frac"] * 100), flush=True)
        del diag, dg, f

    # ------------------------------------------------------------------
    def med(k):
        return float(np.median([r[k] for r in per_time]))

    print("\n" + "=" * 78)
    print("【自检】解析恒等式  N_v²−N_d² = 0.608·g·(dq/dz)/(1+0.608q)")
    print("  中位相对偏差 = %.3e   （应≈0，机器精度量级）" % float(np.median(ident_err)))
    print("=" * 78)

    print("\n【数据有效性】地形遮蔽")
    print("  地下外推层（p > p_sfc）占比 = %.2f%%   —— NOMADS 等压面场不做地形遮蔽，"
          % (med("subground_frac") * 100))
    print("  实测 1000 hPa 位势高度全域仅 43–178 m（昆明海拔 ~1900 m 也只报 85 m），")
    print("  故全部统计均按 p ≤ p_sfc 剔除地下层后再计算。")

    print("\n【主指标 A】被干口径漏判的虚温不稳定层（晴空+地面以上，%d 个时次）" % len(per_time))
    print("  判定：干口径 N_d²>0（稳定）但 虚温口径 N_v²<0（不稳定）")
    print("  ★ 漏判率中位 = %.4f%%   （占晴空+干判稳定格点的比例）" % (med("missed_rate") * 100))
    print("  对应格点数中位 = %d / %d" % (np.median([r["n_missed"] for r in per_time]),
                                    np.median([r["n_clear_stable"] for r in per_time])))
    print("\n  按干稳定度 N_d² 分箱（干口径越强，湿口径越难翻转符号）:")
    print(f"  {'N_d² 区间 [s-2]':>22s} {'格点数':>9s} {'漏判率%':>10s}")
    for lo, hi in BINS:
        n, k = bins[(lo, hi)]
        if n > 10:
            print(f"  [{lo:9.1e},{hi:9.1e}) {n:9d} {100*k/n:10.4f}")

    print("\n【指标 B】湿度对静力稳定度的相对削弱 δ（门槛 N_d²>%.0e，晴空）" % N2_FLOOR)
    print("  δ 分位 [50, 75, 95] = %.4f  %.4f  %.4f"
          % (med("delta_p50"), med("delta_p75"), med("delta_p95")))
    print("\n  逐层结构（漏判率 % / δ 中位）:")
    print(f"  {'层(hPa)':>8s} {'格点数':>8s} {'漏判率%':>10s} {'δ中位':>9s}")
    for key in sorted(per_level, key=lambda s: -float(s)):
        r = per_level[key]
        print(f"  {key:>8s} {int(np.median(r['n'])):8d} "
              f"{np.median(r['missed'])*100:10.4f} {np.nanmedian(r['delta']):9.4f}")

    print("\n【指标 C】阈值等价误差：干口径+0.25 ⟺ 湿口径+0.25/(1−δ)")
    print("  等效阈值 Ri* 分位 [50, 95] = %.4f  %.4f   （判据 0.25 被放宽到多少）"
          % (med("ri_star_p50"), med("ri_star_p95")))
    print("  即：在最湿的 5%% 空域，干算法的有效阈值比 0.25 松 %.1f%%"
          % ((med("ri_star_p95") / RI_CRIT - 1) * 100))

    print("\n【指标 D】风险排序翻转率（CAT 通告按最危险前 p% 圈定）")
    print("  前 1%% 落选 %.2f%%    前 5%% 落选 %.2f%%（Jaccard %.3f）    前 10%% 落选 %.2f%%"
          % (med("turnover_1") * 100, med("turnover_5") * 100, med("jaccard_5"),
             med("turnover_10") * 100))

    print("\n【指标 E】K–H 涌现率对次网格切变放大 α 的敏感性（门槛内）")
    print("  α=1 涌现率 = %.4f%%   峰值 = %.4f%%（出现在 α≈%.0f）"
          % (med("emergence_alpha1") * 100, med("emergence_peak") * 100, med("alpha_peak")))
    print("  ※ 网格 Ri 被高估 1–2 个量级，α=1 的数字不代表真实湍流发生率。")

    if routes:
        import collections
        by = collections.defaultdict(list)
        for r in routes:
            by[(r["dep"], r["arr"])].append(r)
        print("\n【航迹】真实机场对的 δ（按 850 hPa 降序，前 12）")
        print(f"  {'航线':>13s} {'n':>3s} {'δ@850':>8s} {'δ@500':>8s} {'δ@250':>8s}")
        rows = sorted(by.items(), key=lambda kv: -np.nanmedian([x["delta_850"] for x in kv[1]]))
        for (a, b), rs in rows[:12]:
            print(f"  {a+'-'+b:>13s} {len(rs):3d} "
                  + " ".join("%8.4f" % np.nanmedian([x[f"delta_{int(h)}"] for x in rs])
                            for h in PHASE_HPA))
        allr = {}
        for h in PHASE_HPA:
            v = [x[f"delta_{int(h)}"] for x in routes]
            v = np.array(v, float)
            allr[f"delta_{int(h)}"] = [float(np.nanpercentile(v, 25)),
                                       float(np.nanmedian(v)),
                                       float(np.nanpercentile(v, 75))]
        print("\n  全部 %d 条航迹 δ 分位 [25, 50, 75]：" % len(by))
        for h in PHASE_HPA:
            print("    %4d hPa : %s" % (h, "  ".join("%.4f" % x for x in allr[f"delta_{int(h)}"])))

    out = dict(
        bbox=list(BBOX), times=[f"{d}{c}" for d, c in TIMES],
        ri_crit=RI_CRIT, rh_clear=RH_CLEAR, n2_floor=N2_FLOOR,
        identity_check_median_rel_err=float(np.median(ident_err)),
        subground_frac=med("subground_frac"),
        missed_rate=med("missed_rate"),
        missed_by_n2_bin={f"{lo:.1e}_{hi:.1e}": (bins[(lo, hi)][0], bins[(lo, hi)][1])
                          for lo, hi in BINS},
        delta_p50=med("delta_p50"), delta_p75=med("delta_p75"), delta_p95=med("delta_p95"),
        ri_star_p50=med("ri_star_p50"), ri_star_p95=med("ri_star_p95"),
        turnover_1=med("turnover_1"), turnover_5=med("turnover_5"),
        turnover_10=med("turnover_10"), jaccard_5=med("jaccard_5"),
        emergence_alpha1=med("emergence_alpha1"), emergence_peak=med("emergence_peak"),
        alpha_peak=med("alpha_peak"),
        emergence_curve=[[float(a), float(e), float(dd)]
                         for a, e, dd in np.median(np.array(
                             [r["emergence_curve"] for r in per_time]), axis=0)],
        per_level={k: dict(missed=float(np.median(v["missed"])),
                           delta=float(np.nanmedian(v["delta"])),
                           n=int(np.median(v["n"]))) for k, v in per_level.items()},
        route_phase_delta=allr if routes else {},
        per_time=per_time, routes=routes)
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[step04] 已写出 {OUT}   用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
