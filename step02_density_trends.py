# -*- coding: utf-8 -*-
"""
step02 —— 湿度在"空气密度—起飞性能"趋势中的份额（本项目头条结果）
==================================================================

科学问题
--------
Coffel, Thompson & Horton (2017, *Climatic Change* 144:251) 是全球机场
"升温 → 空气密度下降 → 起飞限重"研究的基准文献。该方法只使用温度与气压：

        ρ = p / (R_d T)          ← 干空气

本项目在 step01 用 179 站 / 5019 站年 / 1976–2023 的逐小时**露点**观测证明，
真实空气密度应含湿：

        ρ = p / (R_d T (1 + 0.608 q))

核心可证伪推论（avwx.thermo.cc_humidity_ratio）
----------------------------------------------
在克劳修斯-克拉珀龙标度下（d ln q/dT ≈ 7 %/K）：

        R ≡ c_q / c_T = 0.608 T (d ln q/dT) q/(1 + 0.608 q) ≈ 0.0426 · q · T

含义：只算温度的密度趋势，在湿热机场**低估 R 倍**；
忽略湿度的比例 = R/(1+R)。

本脚本用观测逐条检验该解析式，并给出：
  1. 观测 R_obs vs 解析 R_pred 的吻合度（对 CC 标度的直接检验）
  2. 湿度份额的全球分布与纬度/干旱度依赖
  3. 年际变化被"干空气算法"漏掉的密度高度英尺数
  4. 折算为起飞重量与旅客数
  5. 干/湿机场排序倒置（对照父项目"干球 vs 湿球城市排序倒置"）
"""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avwx import thermo as TH

D = pd.read_csv("data/step01_station_year.csv")
META = pd.read_csv(r"D:\2026codes\全球气候变化\station_panel.csv")

OUT = "data/step02_trends.csv"


# ----------------------------------------------------------------------------
# Theil-Sen 斜率与 Mann-Kendall
# ----------------------------------------------------------------------------
def theil_sen(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = len(y)
    if n < 8:
        return np.nan
    if n <= 250:
        i, j = np.triu_indices(n, 1)
        s = (y[j] - y[i]) / (x[j] - x[i])
    else:                                   # 采样近似，保证可复现
        rng = np.random.default_rng(20261020)
        i = rng.integers(0, n, 20000); j = rng.integers(0, n, 20000)
        k = i != j
        i, j = i[k], j[k]
        s = (y[j] - y[i]) / (x[j] - x[i])
    return float(np.median(s))


def mann_kendall(y):
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    n = len(y)
    if n < 8:
        return np.nan, np.nan
    s = 0.0
    for i in range(n):
        s += np.sign(y[i + 1:] - y[i]).sum()
    # 无结点方差
    var = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var)
    elif s < 0:
        z = (s + 1) / np.sqrt(var)
    else:
        z = 0.0
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


def bh_fdr(p):
    """Benjamini-Hochberg 控制错误发现率，返回 q 值。"""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full(len(p), np.nan)
    pv = p[ok]
    n = len(pv)
    if n == 0:
        return q
    idx = np.argsort(pv)
    ranked = pv[idx]
    qq = ranked * n / (np.arange(n) + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    out = np.empty(n); out[idx] = np.clip(qq, 0, 1)
    q[ok] = out
    return q


# ----------------------------------------------------------------------------
def main():
    d = D.copy()
    # 主口径：8 个世界时正点的采样一致年均值（见 step01 的采样频率漂移说明）
    d = d[d.n_syn_bins >= 6]
    # 站级样本量门槛：至少 25 个可用年
    cnt = d.groupby("station")["year"].size()
    keep = cnt[cnt >= 25].index
    d = d[d.station.isin(keep)].copy()
    print(f"[step02] 可用站年 {len(d)}, 站点 {d.station.nunique()}")

    rows = []
    for st, g in d.groupby("station"):
        g = g.sort_values("year")
        yr = g["year"].to_numpy(float)
        # 采样一致年序列（8 个世界时正点）
        T_c = g["t_c_syn"].to_numpy(float)
        q = g["q_syn_gkg"].to_numpy(float) / 1000.0
        da_d = g["da_dry_syn"].to_numpy(float)
        da_m = g["da_moist_syn"].to_numpy(float)
        dq = g["dq_syn"].to_numpy(float)
        p = g["p_mean"].to_numpy(float)

        ok = np.isfinite(T_c) & np.isfinite(q) & np.isfinite(da_d) & np.isfinite(dq)
        if ok.sum() < 25:
            continue
        yr, T_c, q, da_d, da_m, dq, p = [a[ok] for a in (yr, T_c, q, da_d, da_m, dq, p)]
        Tk = T_c + 273.15

        # ---- 斜率（每十年）----
        sT = theil_sen(T_c, yr) * 10.0          # °C/decade
        sq = theil_sen(q, yr) * 10.0            # kg/kg per decade
        sda_d = theil_sen(da_d, yr) * 10.0      # ft/decade
        sda_m = theil_sen(da_m, yr) * 10.0
        sdq = theil_sen(dq, yr) * 10.0
        sp = theil_sen(p, yr) * 10.0

        # ---- 对数分解（精确）----
        cT = theil_sen(np.log(Tk), yr) * 10.0          # Δ ln T /decade
        cq = theil_sen(np.log1p(0.608 * q), yr) * 10.0  # Δ ln(1+0.608q)/decade
        cp = -theil_sen(np.log(p), yr) * 10.0

        # ---- 湿度份额的守卫条件（防止"分母近零"伪信号）----
        # 上一版曾出现 share_q = 1.50、0.995 等荒谬值：那些站的温度趋势≈0
        # （关岛 -0.013 °C/dec、查卡柳塔 -0.0002 °C/dec），比值因此爆炸。
        # 湿度份额只在"确实存在增温"时才有物理意义。
        # 判据：温度趋势 > +0.02 °C/dec 且 Mann-Kendall z > 1.0。
        zT, pT = mann_kendall(T_c)
        valid_share = (sT > 0.02) and (zT > 1.0) and np.isfinite(cT) and (cT > 1e-6)
        denom = cT + cq
        share_q = (cq / denom) if (valid_share and np.isfinite(denom)
                                   and abs(denom) > 1e-12) else np.nan
        ratio_obs = (cq / cT) if (valid_share and abs(cT) > 1e-12) else np.nan

        # ---- 解析 CC 预测 ----
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from avwx import thermo as TH
        ratio_pred = float(TH.cc_humidity_ratio(float(np.mean(q)), float(np.mean(Tk))))

        # ---- 显著性 ----
        z_dq, p_dq = mann_kendall(dq)
        z_da, p_da = mann_kendall(da_m)
        z_q, p_q = mann_kendall(q)

        # ---- 逐小时气候态的"不可调度底线" ----
        floors = g["dq_syn_min"].to_numpy(float)
        fl = float(np.nanmean(floors[np.isfinite(floors)])) if np.isfinite(floors).any() else np.nan

        rows.append(dict(
            station=st, lat=g.lat.iloc[0], lon=g.lon.iloc[0], elev=g.elev.iloc[0],
            n_years=int(ok.sum()), y0=int(yr.min()), y1=int(yr.max()),
            T_c=float(np.mean(T_c)), q_gkg=float(np.mean(q) * 1000),
            rh=float(np.nanmean(g.rh_mean)), p_hpa=float(np.mean(p) / 100),
            da_dry=float(np.mean(da_d)), da_moist=float(np.mean(da_m)), dq=float(np.mean(dq)),
            dq_floor=fl, z_T=zT,
            slope_T_c_dec=sT, slope_q_gkg_dec=sq * 1000, slope_da_dry_dec=sda_d,
            slope_da_moist_dec=sda_m, slope_dq_dec=sdq, slope_p_hpa_dec=sp / 100,
            cT=cT, cq=cq, cp=cp, share_q=share_q,
            ratio_obs=ratio_obs, ratio_pred=ratio_pred,
            mk_z_dq=z_dq, mk_p_dq=p_dq, mk_z_da=z_da, mk_p_da=p_da, mk_z_q=z_q, mk_p_q=p_q,
            obs_per_day=float(np.nanmean(g.obs_per_day)),
        ))

    R = pd.DataFrame(rows)
    R["share_q_bh"] = bh_fdr(R.mk_p_dq.to_numpy())
    # 排序倒置
    R["rank_dry"] = R.da_dry.rank(ascending=False)
    R["rank_moist"] = R.da_moist.rank(ascending=False)
    R["rank_shift"] = R.rank_dry - R.rank_moist

    # 载重折算（B737-800 口径，m=2 与 Coffel 一致；m=1 作保守下界）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from avwx import thermo as TH
    m2 = TH.TakeoffModel(exponent_m=2)
    m1 = TH.TakeoffModel(exponent_m=1)
    pk, pax = np.zeros(len(R)), np.zeros(len(R))
    pk1 = np.zeros(len(R))
    for i, r in R.iterrows():
        p_pa = r.p_hpa * 100.0
        pk[i] = float(m2.penalty_kg(p_pa, r.T_c, r.q_gkg / 1000.0))
        pk1[i] = float(m1.penalty_kg(p_pa, r.T_c, r.q_gkg / 1000.0))
        pax[i] = pk[i] / 95.0
    R["penalty_kg_m2"] = pk
    R["penalty_kg_m1"] = pk1
    R["penalty_pax_m2"] = pax
    # 惩罚的年增幅（kg/decade）
    R["penalty_kg_dec"] = np.where(np.isfinite(R.slope_dq_dec),
                                   R.slope_dq_dec * (pk / np.maximum(R.dq, 1e-9)), np.nan)

    # 站名
    nm = META[["station", "name", "country"]].drop_duplicates("station")
    R = R.merge(nm, on="station", how="left")

    os.makedirs("data", exist_ok=True)
    R.to_csv(OUT, index=False, encoding="utf8")
    print(f"[step02] 已写出 {OUT} ({len(R)} 站)")

    # ---------------- 汇总 ----------------
    def rep(col, label, unit=""):
        x = R[col].dropna()
        if len(x) == 0:
            return
        print(f"  {label:34s} n={len(x):3d}  中位={x.median():9.3f}  "
              f"IQR=[{x.quantile(.25):8.3f},{x.quantile(.75):8.3f}] {unit}")

    print("\n" + "=" * 88)
    print("【表 1】湿度惩罚的量级（1976–2023 站气候态）")
    print("=" * 88)
    rep("dq", "ΔDA_q 湿度惩罚", "ft")
    rep("dq_floor", "ΔDA_q 最凉时次（不可调度底线）", "ft")
    rep("da_dry", "干空气密度高度（现行算法）", "ft")
    rep("da_moist", "含湿密度高度（正确算法）", "ft")
    print(f"  底线保留比例 中位 = {float((R.dq_floor / R.dq).median()):.4f}")

    print("\n" + "=" * 88)
    print("【表 2】趋势（每十年）")
    print("=" * 88)
    rep("slope_T_c_dec", "干球温度", "°C/dec")
    rep("slope_q_gkg_dec", "比湿", "g/kg/dec")
    rep("slope_da_dry_dec", "干空气密度高度（现行算法）", "ft/dec")
    rep("slope_da_moist_dec", "含湿密度高度（正确算法）", "ft/dec")
    rep("slope_dq_dec", "湿度惩罚 ΔDA_q", "ft/dec")
    rep("slope_p_hpa_dec", "场面气压", "hPa/dec")

    print("\n" + "=" * 88)
    print("【表 3】湿度份额 —— 头条结果")
    print("=" * 88)
    rep("share_q", "观测湿度份额 c_q/(c_T+c_q)")
    rep("ratio_obs", "观测比值 R_obs = c_q/c_T")
    rep("ratio_pred", "解析预测 R_pred = 0.0426·q·T")
    m = R.dropna(subset=["ratio_obs", "ratio_pred"])
    if len(m) > 5:
        r = np.corrcoef(m.ratio_obs, m.ratio_pred)[0, 1]
        print(f"  R_obs 与 R_pred 的相关 r = {r:.4f}  (n={len(m)})")
        d_ = m.ratio_obs - m.ratio_pred
        print(f"  差值 R_obs - R_pred: 中位 {np.median(d_):+.4f}  "
              f"IQR [{np.percentile(d_,25):+.4f},{np.percentile(d_,75):+.4f}]")
        ok = (m.ratio_pred > 1e-6)
        print(f"  比值 R_obs/R_pred: 中位 {np.median(m.ratio_obs[ok]/m.ratio_pred[ok]):.3f}")

    print("\n" + "=" * 88)
    print("【表 4】运行折算（B737-800, MTOW 79 t, 95 kg/人）")
    print("=" * 88)
    rep("penalty_kg_m2", "被遗漏的载重限制 (m=2)", "kg")
    rep("penalty_kg_m1", "被遗漏的载重限制 (m=1, 保守)", "kg")
    rep("penalty_pax_m2", "折算旅客数 (m=2)", "人")
    rep("penalty_kg_dec", "惩罚的年际增幅 (m=2)", "kg/dec")

    print("\n" + "=" * 88)
    print("【表 5】排序倒置：按干空气 vs 含湿密度高度")
    print("=" * 88)
    big = R[R.rank_shift.abs() >= 20].sort_values("rank_shift")
    print(f"  位移 ≥ 20 位的站: {len(big)} / {len(R)} ({100*len(big)/len(R):.0f}%)")
    print(R.reindex(R.rank_shift.abs().sort_values(ascending=False).index)
          [["station", "name", "lat", "q_gkg", "da_dry", "da_moist", "dq",
            "rank_dry", "rank_moist", "rank_shift"]].head(12).to_string(index=False))

    print("\n" + "=" * 88)
    print("【表 6】湿度惩罚最高的 12 站（现行算法漏得最多）")
    print("=" * 88)
    print(R.sort_values("dq", ascending=False)
          [["station", "name", "lat", "elev", "T_c", "q_gkg", "rh",
            "dq", "dq_floor", "penalty_kg_m2", "penalty_pax_m2"]].head(12).to_string(index=False))

    print("\n" + "=" * 88)
    print("【表 7】湿度份额最高的 12 站（趋势被低估最严重）")
    print("=" * 88)
    print(R.sort_values("share_q", ascending=False)
          [["station", "name", "lat", "q_gkg", "slope_T_c_dec", "slope_q_gkg_dec",
            "slope_da_dry_dec", "slope_da_moist_dec", "share_q", "mk_z_dq"]].head(12).to_string(index=False))

    climatology_and_composite(R)
    return R


# ----------------------------------------------------------------------------
def climatology_and_composite(R):
    """全网络气候态（179 站）+ 网络合成序列趋势（69 长序列站）。

    GHCNh 网络实测构成（本项目已核实）：
        1976–1980 约 160 站；1981–1999 约 68 站；
        2000–2004 约 166 站；2005–2018 约 70 站；2019–2023 约 169 站。
    即：约 160 个站只有三段 5 年期，约 70 个站拥有连续长序列。
    因此 **气候态用全部站（空间覆盖最优），趋势只用长序列站（时间一致性最优）**，
    两者口径不同，必须分别报告——混用会制造假信号。
    """
    D = pd.read_csv("data/step01_station_year.csv")
    nm = META[["station", "name", "country"]].drop_duplicates("station")

    # ---------- A. 全网络气候态 ----------
    a = D[D.n_syn_bins >= 6]
    ag = a.groupby("station").agg(
        n=("dq_syn", "size"), lat=("lat", "first"), lon=("lon", "first"),
        elev=("elev", "first"), dq=("dq_syn", "mean"), floor=("dq_syn_min", "mean"),
        q=("q_syn_gkg", "mean"), t=("t_c_syn", "mean"), rh=("rh_mean", "mean"),
        da_dry=("da_dry_syn", "mean"), da_moist=("da_moist_syn", "mean"),
        p=("p_mean", "mean")).reset_index()
    ag = ag[ag.n >= 3].merge(nm, on="station", how="left")
    m2 = TH.TakeoffModel(exponent_m=2)
    m1 = TH.TakeoffModel(exponent_m=1)
    ag["kg_m2"] = [float(m2.penalty_kg(r.p, r.t, r.q / 1000.0)) for r in ag.itertuples()]
    ag["kg_m1"] = [float(m1.penalty_kg(r.p, r.t, r.q / 1000.0)) for r in ag.itertuples()]
    ag["pax_m2"] = ag.kg_m2 / 95.0
    ag["floor_ratio"] = ag.floor / ag.dq
    ag.to_csv("data/step02_climatology_all_stations.csv", index=False, encoding="utf8")

    print("\n" + "=" * 88)
    print(f"【表 8】全网络气候态 —— {len(ag)} 站（含仅有 5 年期的站，空间覆盖最优）")
    print("=" * 88)
    for c, lab, u in (("dq", "ΔDA_q 湿度惩罚", "ft"),
                      ("floor", "最凉时次惩罚（不可调度底线）", "ft"),
                      ("floor_ratio", "底线保留比例", ""),
                      ("kg_m2", "遗漏载重 (m=2)", "kg"),
                      ("pax_m2", "折算旅客 (m=2)", "人")):
        x = ag[c].dropna()
        print(f"  {lab:30s} n={len(x):3d} 中位={x.median():9.3f} "
              f"IQR=[{x.quantile(.25):8.3f},{x.quantile(.75):8.3f}]  "
              f"max={x.max():8.2f} {u}")

    # 纬度带
    ag["band"] = pd.cut(ag.lat.abs(), [0, 15, 30, 45, 90],
                        labels=["热带 0-15", "亚热带 15-30", "温带 30-45", "高纬 >45"])
    print("\n  按纬度带（绝对值纬度）:")
    bt = ag.groupby("band", observed=True).agg(
        n=("dq", "size"), dq=("dq", "median"), q=("q", "median"),
        kg=("kg_m2", "median"), pax=("pax_m2", "median"), fr=("floor_ratio", "median"))
    print(bt.to_string())

    # ---------- B. 网络合成序列（69 长序列站）----------
    tr_st = set(R.station)
    b = D[(D.n_syn_bins >= 6) & (D.station.isin(tr_st))].copy()
    # 各站相对自身 1991–2020 基准的距平
    comp = {}
    base = b[(b.year >= 1991) & (b.year <= 2020)]
    for col in ("t_c_syn", "q_syn_gkg", "dq_syn", "da_dry_syn", "da_moist_syn"):
        ref = base.groupby("station")[col].mean()
        b[col + "_an"] = b[col] - b.station.map(ref)
        gmean = base[col].mean()
        s = b.groupby("year")[col + "_an"].median() + gmean
        comp[col] = s
    yrs = sorted(set(b.year))
    T = np.array([comp["t_c_syn"].get(y, np.nan) for y in yrs])
    q = np.array([comp["q_syn_gkg"].get(y, np.nan) / 1000.0 for y in yrs])
    dq = np.array([comp["dq_syn"].get(y, np.nan) for y in yrs])
    da_d = np.array([comp["da_dry_syn"].get(y, np.nan) for y in yrs])
    da_m = np.array([comp["da_moist_syn"].get(y, np.nan) for y in yrs])
    yy = np.array(yrs, float)
    ok = np.isfinite(T) & np.isfinite(q)

    print("\n" + "=" * 88)
    print(f"【表 9】网络合成序列趋势（{len(tr_st)} 长序列站，{int(yy[ok].min())}–{int(yy[ok].max())}）")
    print("=" * 88)
    Tk = T[ok] + 273.15
    sT = theil_sen(T[ok], yy[ok]) * 10
    sq = theil_sen(q[ok], yy[ok]) * 10
    sdq = theil_sen(dq[ok], yy[ok]) * 10
    sda_d = theil_sen(da_d[ok], yy[ok]) * 10
    sda_m = theil_sen(da_m[ok], yy[ok]) * 10
    zT, pT = mann_kendall(T[ok]); zq, pq = mann_kendall(q[ok]); zdq, pdq = mann_kendall(dq[ok])
    zdd, pdd = mann_kendall(da_d[ok]); zdm, pdm = mann_kendall(da_m[ok])
    cT = theil_sen(np.log(Tk), yy[ok]) * 10
    cq = theil_sen(np.log1p(0.608 * q[ok]), yy[ok]) * 10
    share = cq / (cT + cq)
    ratio = cq / cT
    ratio_pred = float(TH.cc_humidity_ratio(float(np.mean(q[ok])), float(np.mean(Tk))))
    print(f"  干球温度        {sT:+.4f} °C/dec   MK z={zT:+.2f} p={pT:.2e}")
    print(f"  比湿            {sq*1000:+.4f} g/kg/dec MK z={zq:+.2f} p={pq:.2e}")
    print(f"  干空气密度高度  {sda_d:+.3f} ft/dec   MK z={zdd:+.2f} p={pdd:.2e}")
    print(f"  含湿密度高度    {sda_m:+.3f} ft/dec   MK z={zdm:+.2f} p={pdm:.2e}")
    print(f"  湿度惩罚 ΔDA_q  {sdq:+.4f} ft/dec   MK z={zdq:+.2f} p={pdq:.2e}")
    print()
    print(f"  >>> 网络合成的湿度份额 c_q/(c_T+c_q) = {share*100:.2f} %")
    print(f"  >>> 观测比值 R_obs = {ratio:.4f}   解析预测 R_pred = {ratio_pred:.4f}   "
          f"比值 = {ratio/ratio_pred:.3f}")
    print(f"  >>> 忽略湿度导致的密度高度趋势低估 = {100*(sda_m-sda_d)/sda_d:.2f} %")
    print(f"  >>> 相对增湿速率 dlnq/dT = {theil_sen(np.log(q[ok]), yy[ok])/theil_sen(T[ok], yy[ok])*100:.2f} %/K"
          f"（克劳修斯-克拉珀龙参考值 ≈ 7 %/K）")

    np.save("data/step02_composite_series.npy",
            np.column_stack([yy, T, q * 1000, dq, da_d, da_m]))
    with open("data/step02_composite.json", "w", encoding="utf8") as f:
        json.dump(dict(n_stations=len(tr_st), years=[int(y) for y in yy[ok].tolist()],
                       slope_T_c_dec=sT, slope_q_gkg_dec=sq * 1000,
                       slope_da_dry_dec=sda_d, slope_da_moist_dec=sda_m,
                       slope_dq_dec=sdq, mk_z=dict(T=zT, q=zq, dq=zdq, da_dry=zdd, da_moist=zdm),
                       share_q=share, ratio_obs=ratio, ratio_pred=ratio_pred,
                       dlnq_dT_pct_per_K=theil_sen(np.log(q[ok]), yy[ok]) / theil_sen(T[ok], yy[ok]) * 100,
                       underestimation_pct=100 * (sda_m - sda_d) / sda_d),
                  f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
