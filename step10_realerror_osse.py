# -*- coding: utf-8 -*-
"""
step10 —— **真实预报误差 OSSE**（poor-man's OSSE / NMC 型）

为什么必须再做一次
------------------
step03 / step09 的背景场是"真值粗化 + 相关噪声"。这一构造隐含一个致命假设：
**背景误差在四个变量之间互相独立**，于是试验在数学上不可能给多变量耦合留下
任何可提取的信号：

    （1）湿度平流耦合 M_{V→q} 的输入是"风的分析增量"，而它要去逼近的对象
         —— 真值侧的湿度误差 —— 与风误差**毫无关系**（人为随机场），
         所以 δq = q_b(x − δV·τ) − q_b(x) 再正确也只能推出噪声；
    （2）静力—地转耦合同理：随机构造的背景风误差并不满足地转/静力平衡，
         强行施加平衡只是把**没有**结构的误差解释成**有**结构。

实测证实了这一点：step09 把 σ_b 标定到观测误差之上、并改用多轮 OI 之后，
单变量分析已把可榨取的局地信息抽干（T 改善 12.6%），而 C/D/E 三个耦合配置
与 B **完全相同到小数点后两位** —— 这不是"耦合无效"，而是
**试验本身没有给耦合留下任何可以提取的东西**。

真实大气不是这样。6 小时预报误差是**动力学协调**的：
    * 锋面位置偏了 ⇒ T、q、V 的误差同时出现且满足热成风关系；
    * 水汽被输送偏了 ⇒ q 误差 = 沿气流的水汽梯度 × 风误差位移，
      这**恰恰就是** δq = q_b(x − δV·τ) − q_b(x) 要描述的东西。

因此本脚本把背景换成 GFS **6 小时预报**（同一有效时刻）：
    真值  nat : GFS f000 分析
    背景  bg  : GFS f006 预报（6 小时前起报）
    误差  ε   : f006 − f000 —— 真实的、多变量协调的短时预报误差

这正是业务上 NMC 法定义背景误差协方差所用的同一种差值（24h 减 12h），
只是这里把较短的 6h 差直接当作"真背景误差"用于 OSSE。

新增的关键对照：配置 F
----------------------
即便证明"风 ──平流──▶ 湿度"有效，决策者真正要问的是：
**它值不值得？** 即相对于"干脆给全机队加装湿度传感器"能拿到的信息，
耦合算子白捡回来多少？因此加入配置 F：假定全部飞机都报湿度（WVSS-II 精度），
给出直报湿度的**收益上限**；D 相对 F 的比值就是耦合的"性价比"。

其余与 step09 一致：谷地填充、地形遮蔽、标定/检验时次严格分离。
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from avwx import aws_gfs as AG
from avwx import recon3d as R3

BBOX = (100, 130, 15, 40)
OERR = dict(T=0.5, q=5e-4, u=1.5, v=1.5)
OERR_Q_AC = 5e-4            # 假想的机载湿度观测误差（WVSS-II 量级）
IMPACT_R = 150e3
IMPACT_LNP = 0.5
NATIVE_ROOT = "data/aws/"

# 真值（分析）时次：第 1 个用于标定耦合系数，其余用于独立检验（严格分离）
TRUTH_TIMES = [("20261001", "00"), ("20261008", "12"),
               ("20261012", "18"), ("20261016", "00")]
# 预报时效。f006 实测 T 误差仅 0.145 K < 观测误差 0.5 K ⇒ K≈0.08，无法检验，
# 故采用 NMC 惯例的较长时效；两个时效可检验"耦合收益随背景误差单调增长"。
LEADS = (12, 24)
KEYS = ("T", "q", "u", "v")


def bg_cycle(dte, cyc, lead):
    """给定真值时刻与时效，返回背景场的起报时刻。"""
    import datetime as dt
    t = dt.datetime.strptime(dte + cyc, "%Y%m%d%H") - dt.timedelta(hours=lead)
    return t.strftime("%Y%m%d"), t.strftime("%H")


# --------------------------------------------------------------- 工具
def native_path(dte, cyc, fhr):
    return os.path.join(NATIVE_ROOT,
                        "gfs_%s_%s_f%03d_L21_1000_100.grib2" % (dte, cyc, fhr))


def prepare(d, tag=""):
    """谷地填充 + 返回地面以上掩码。与 step09 口径一致。"""
    lat, lon, lev = d["lat"], d["lon"], d["level"]
    psfc = np.asarray(d["p_sfc"], float)
    play = np.asarray(lev, float)[:, None, None] * 100.0
    above = play <= psfc[None, ...]
    out = {v: np.asarray(d[v], float).copy() for v in ("T", "q", "u", "v", "z")}
    kk = np.arange(play.shape[0])[:, None, None]
    k0 = np.argmax(above, axis=0)
    fill = kk < k0[None]
    for v in ("T", "q", "u", "v"):
        a = out[v]
        vals = np.take_along_axis(a, k0[None], axis=0)
        a[fill] = np.broadcast_to(vals, a.shape)[fill]
    return dict(f=out, lat=lat, lon=lon, lev=np.asarray(lev, float),
                above=above, nbelow=int((~above).sum()))


def load_pair(tspec, bspec, lead):
    """读取 (真值分析, 背景预报) 一对，做谷地填充，检查网格一致。"""
    nat = prepare(AG.load(native_path(tspec[0], tspec[1], 0), bbox=BBOX), "nat")
    bg = prepare(AG.load(native_path(bspec[0], bspec[1], lead), bbox=BBOX), "bg")
    for a, b in ((nat["lat"], bg["lat"]), (nat["lon"], bg["lon"]),
                 (nat["lev"], bg["lev"])):
        if a.shape != b.shape or not np.allclose(a, b):
            raise RuntimeError("真值与背景网格不一致")
    lat, lon, lev = nat["lat"], nat["lon"], nat["lev"]
    above = nat["above"] & bg["above"]
    nb = int((~above).sum())
    b = {v: bg["f"][v] for v in ("T", "q", "u", "v")}
    b["z"] = bg["f"]["z"]

    sim = R3.ObsSim(dict(lat=lat, lon=lon, level=lev,
                         **{v: nat["f"][v] for v in ("T", "q", "u", "v")}),
                    seed=7)
    obs = dict(sfc=sim.surface(n=150), raob=sim.raob(n=12), ac=sim.aircraft())

    # ---- 观测不得落在地面以下（物理约束；step09 未做，此处补上）----
    def _filter(o, p):
        sel = np.zeros(len(p), bool)
        for i, (pv, pla, plo) in enumerate(p):
            ia = int(np.clip(np.searchsorted(lat, pla), 0, len(lat) - 1))
            io = int(np.clip(np.searchsorted(lon, plo), 0, len(lon) - 1))
            ik = int(np.clip(np.searchsorted(-lev, -pv), 0, len(lev) - 1))
            sel[i] = above[ik, ia, io]
        return {k: v[sel] for k, v in o.items()}, p[sel]

    for tag in ("sfc", "raob", "ac"):
        obs[tag] = _filter(obs[tag][0], obs[tag][1])

    dg = R3.diagnose_background_error(nat["f"], b, lat, lon, lev)
    # σ_b 逐变量逐层**诊断**（真实预报误差，非人为假设）
    sig_b, so = {}, {}
    for v in KEYS:
        e = b[v] - nat["f"][v]
        s = np.array([float(np.sqrt(np.nanmean((e[k][above[k]]) ** 2)))
                      for k in range(len(lev))])
        sig_b[v] = np.maximum(s, 1e-9)
        so[v] = np.full(len(lev), OERR[v])
    sigma = {v: (sig_b[v], so[v]) for v in KEYS}
    return dict(nat=nat["f"], bg=b, obs=obs, lat=lat, lon=lon, lev=lev,
                dg=dg, sigma=sigma, above=above, nbelow=nb, sig_b=sig_b)


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


def make_rec(lat, lon, lev, dg, sigma, max_obs=900):
    return R3.Reconstructor(lat, lon, lev, L_h=dg["L_h1"], L_z=dg["L_z"],
                            L_h2=dg["L_h2"], w2=dg["w2"], sigma=sigma,
                            max_obs=max_obs, vert_corr=dg.get("vert_corr"))


NAME_F = "F  +飞机直报湿度(上限)"
NAMES = ["A  地面+探空       无飞机", "B  +飞机 单变量(湿度盲)",
         "C  B+T2V 静力-地转", "D  B+V2q 水汽平流", "E  B+全耦合", NAME_F]


def med_of(rows, key):
    v = [r[key] for r in rows if key in r]
    return float(np.median(v)) if v else float("nan")


def med_pct(rows, key, bgk="bg|q"):
    """**逐时次**算改善百分比，再跨时次取中位 —— 唯一正确的聚合口径。

    ⚠ 一度踩坑：原实现是"先跨时次中位 RMSE、再算百分比"
      （med_of(rows, key) 后相减）。但各时次的背景误差相差可达 3 倍
      （bg q 从 2.8e-4 到 8.7e-4 g/kg），两个来自**不同时次**的 RMSE
      相减得到的百分比不对应任何真实时次，属于 Simpson 型聚合谬误。
      正确做法是先在每个时次内部归一化，再聚合。
    """
    v = [100 * (r[bgk] - r[key]) / r[bgk]
         for r in rows if (key in r and bgk in r and r[bgk] > 0)]
    if not v:
        return float("nan"), float("nan"), float("nan"), 0
    return (float(np.median(v)), float(np.min(v)), float(np.max(v)), len(v))


def boot_ci(vals, n=5000, seed=11):
    """中位数的 bootstrap 95% 置信区间（样本小，不能用正态近似）。"""
    vals = np.asarray([v for v in vals if np.isfinite(v)], float)
    if vals.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n, vals.size))
    bs = np.median(vals[idx], axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def print_tables(rows, lead):
    for title, pre, bgk in (
            ("RMSE 降幅 —— 飞机观测影响区 ∩ 地面以上", "", "bg|"),
            ("对照 —— 全部地面以上区域（含无观测区）", "ALL|", "bgALL|")):
        print("\n" + "=" * 92)
        print("【f%03d】%s      （逐时次百分比 → 跨时次中位）" % (lead, title))
        print("=" * 92)
        print("%-26s %11s %11s %11s %11s" % ("配置", *KEYS))
        print("%-26s " % "(背景场绝对值)" +
              "  ".join("%11.4g" % med_of(rows, bgk + k) for k in KEYS))
        for nm in NAMES:
            if pre:
                continue
            print("%-26s " % nm +
                  "  ".join("%10.2f%%" % med_pct(rows, nm + "|" + k,
                                                 bgk + k)[0]
                            for k in KEYS))


def attribution(rows, lead, nb=None):
    """核心归因。全部采用 **逐时次归一化 → 跨时次中位** 口径（见 med_pct 说明）。"""
    # 逐时次的改善百分比序列
    seq_D = [100 * (r["bg|q"] - r[NAMES[3] + "|q"]) / r["bg|q"] for r in rows]
    seq_F = [100 * (r["bg|q"] - r[NAME_F + "|q"]) / r["bg|q"] for r in rows]
    seq_B = [100 * (r["bg|q"] - r[NAMES[1] + "|q"]) / r["bg|q"] for r in rows]
    seq_A = [100 * (r["bg|q"] - r[NAMES[0] + "|q"]) / r["bg|q"] for r in rows]
    g_D, g_lo, g_hi, n = med_pct(rows, NAMES[3] + "|q")
    g_F = med_pct(rows, NAME_F + "|q")[0]
    g_B = med_pct(rows, NAMES[1] + "|q")[0]
    g_A = med_pct(rows, NAMES[0] + "|q")[0]
    ci_D, ci_F = boot_ci(seq_D), boot_ci(seq_F)
    print("\n" + "=" * 92)
    print("【f%03d】核心归因：湿度盲观测到底榨出了多少湿度信息" % lead)
    print("=" * 92)
    print("  样本：%d 个独立检验时次，口径 = 逐时次百分比 → 跨时次中位" % n)
    print("  内部一致性  |B−A| 中位 = %.4f 个百分点   （飞机不报湿度 ⇒ 必须 ≈ 0）"
          % abs(g_B - g_A))
    print("  V2q 耦合相对背景的改善     = %+.4f 个百分点  95%%CI[%+.2f, %+.2f]  范围[%+.2f, %+.2f]"
          % (g_D, ci_D[0], ci_D[1], g_lo, g_hi))
    print("  同一批飞机**直报**湿度的改善 = %+.4f 个百分点  95%%CI[%+.2f, %+.2f]  ← 硬件上限"
          % (g_F, ci_F[0], ci_F[1]))
    if abs(g_F) > 1e-9:
        print("  耦合 / 直报 = %.1f%%  ← 不花钱的软件方案能兑现直报湿度的比例"
              % (100 * g_D / g_F))
    a_pct = [100 * (r["bg|q"] - r[k + "|q"]) / r["bg|q"]
             for k in (NAMES[0], NAMES[1], NAMES[3], NAME_F)]
    return dict(q_A=g_A, q_B=g_B, q_D=g_D, q_F=g_F,
                ci_gain=ci_D, ci_direct=ci_F, gain_lo=g_lo, gain_hi=g_hi,
                n_times=n, rel_BA=abs(g_B - g_A),
                gain_pp=g_D, direct_pp=g_F,
                coupling_over_direct=(100 * g_D / g_F
                                      if abs(g_F) > 1e-9 else float("nan")))


def run_lead(lead):
    pairs = [(t, bg_cycle(t[0], t[1], lead)) for t in TRUTH_TIMES]
    print("\n" + "#" * 92)
    print("#  预报时效 f%03d   真值 = GFS f000 分析   背景 = 提前 %d h 起报" % (lead, lead))
    print("#" * 92)
    print("\n[标定] %s %sz  ← 背景 %s %sz"
          % (pairs[0][0][0], pairs[0][0][1], pairs[0][1][0], pairs[0][1][1]),
          flush=True)
    C = load_pair(pairs[0][0], pairs[0][1], lead)
    lat, lon, lev = C["lat"], C["lon"], C["lev"]
    nz = len(lev)
    print("    网格 %dx%d  层 %d  地面以上 %.1f%%  观测：地面 %d / 探空 %d / 飞机 %d"
          % (len(lat), len(lon), nz, 100.0 * C["above"].sum() / C["above"].size,
             len(C["obs"]["sfc"][1]), len(C["obs"]["raob"][1]),
             len(C["obs"]["ac"][1])), flush=True)

    print("\n    背景误差（真实 %dh 预报误差）与卡尔曼增益 K = sb²/(sb²+so²)" % lead)
    print("    %-8s %9s %6s | %9s %6s | %10s %6s"
          % ("层[hPa]", "s_b(T)", "K(T)", "s_b(u)", "K(u)", "s_b(q)", "K(q)"))
    Kgain = {}
    for v in KEYS:
        Kgain[v] = [C["sig_b"][v][k] ** 2 /
                    (C["sig_b"][v][k] ** 2 + OERR[v] ** 2) for k in range(nz)]
    for k in range(0, nz, 3):
        print("    %-8.0f %9.3f %6.2f | %9.3f %6.2f | %10.3g %6.2f"
              % (lev[k], C["sig_b"]["T"][k], Kgain["T"][k],
                 C["sig_b"]["u"][k], Kgain["u"][k],
                 C["sig_b"]["q"][k], Kgain["q"][k]))
    print("    全域 RMSE: " + "  ".join(
        "%s=%.4g" % (v, rmse(C["bg"][v], C["nat"][v], C["above"])) for v in KEYS))
    print("    相关长度: L_h1=%.0f km  L_h2=%.0f km  w2=%.2f  L_z=%.2f"
          % (C["dg"]["L_h1"] / 1e3, C["dg"]["L_h2"] / 1e3, C["dg"]["w2"],
             C["dg"]["L_z"]), flush=True)

    rec = make_rec(lat, lon, lev, C["dg"], C["sigma"])
    msk = impact_mask(C["obs"]["ac"][1], lat, lon, lev) & C["above"]
    uni = rec.univariate(C["bg"], C["obs"], use=("sfc", "raob", "ac"))
    q0 = rmse(uni["q"], C["nat"]["q"], msk)
    u0 = rmse(uni["u"], C["nat"]["u"], msk)
    bi = {v: rmse(C["bg"][v], C["nat"][v], msk) for v in KEYS}
    best, scan = None, []
    for aT in (0.0, 0.3, 0.6, 1.0):
        for aq in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
            for tau in (1.0, 2.0, 3.0, 6.0, 12.0):
                rec.alpha_T2V, rec.alpha_V2q, rec.tau_h = aT, aq, tau
                r = rec.apply_coupling(uni, C["bg"], coupling=("T2V", "V2q"))
                s = ((q0 - rmse(r["an"]["q"], C["nat"]["q"], msk)) / bi["q"] +
                     (u0 - rmse(r["an"]["u"], C["nat"]["u"], msk)) / bi["u"])
                scan.append(dict(alpha_T2V=aT, alpha_V2q=aq, tau_h=tau, score=s))
                if best is None or s > best[0]:
                    best = (s, aT, aq, tau)
                del r
    _, A_T2V, A_V2Q, TAU = best
    print("    标定: alpha_T2V=%.2f  alpha_V2q=%.2f  tau_h=%.1f h  (评分 %+.3f%%)"
          % (A_T2V, A_V2Q, TAU, best[0] * 100), flush=True)
    SB = {v: [float(x) for x in C["sig_b"][v]] for v in KEYS}
    del C, uni, rec, msk
    gc.collect()

    print("\n[检验] 独立时次 …", flush=True)
    rows = []
    for i in range(1, len(pairs)):
        ts, bs = pairs[i]
        try:
            X = load_pair(ts, bs, lead)
        except Exception as e:
            print("    %s %sz 跳过: %s" % (ts[0], ts[1], str(e)[:80]), flush=True)
            continue
        am = impact_mask(X["obs"]["ac"][1], lat, lon, lev) & X["above"]
        rec = make_rec(lat, lon, lev, X["dg"], X["sigma"])
        # ⚠ 属性名是 alpha_V2q（小写 q）——曾误写成 alpha_V2Q，静默创建了新属性，
        #   使检验沿用未标定的默认值 1.0（本项目实测踩坑：结果"看起来合理"但与标定不符）
        rec.alpha_T2V, rec.alpha_V2q, rec.tau_h = A_T2V, A_V2Q, TAU
        uniA = rec.univariate(X["bg"], X["obs"], use=("sfc", "raob"))
        uniB = rec.univariate(X["bg"], X["obs"], use=("sfc", "raob", "ac"))

        # ---- 配置 F：假定同一批飞机也报湿度（收益上限）----
        acp = X["obs"]["ac"][1]
        aq = np.array([X["nat"]["q"][
            int(np.clip(np.searchsorted(-lev, -pv), 0, nz - 1)),
            int(np.clip(np.searchsorted(lat, pla), 0, len(lat) - 1)),
            int(np.clip(np.searchsorted(lon, plo), 0, len(lon) - 1))]
            for pv, pla, plo in acp])
        rng = np.random.default_rng(2027)
        aq = aq + rng.normal(0.0, OERR_Q_AC, aq.size)
        aq, acp2, cnt = R3.superob(aq, acp, lat, lon, lev)
        # superob 后误差按 n^(-1/4) 保守缩小（与 Reconstructor._oi 同一口径）：
        # sigma_o 必须是**逐观测**数组，oi_univariate 内部再按 oi_lev 索引。
        so_obs = np.full(len(acp2), OERR_Q_AC) * cnt
        _, anF_q = R3.oi_univariate(
            X["bg"]["q"], aq, acp2, lat, lon, lev,
            L_h=X["dg"]["L_h1"], L_z=X["dg"]["L_z"], L_h2=X["dg"]["L_h2"],
            w2=X["dg"]["w2"],
            sigma_b=X["sigma"]["q"][0], sigma_o=so_obs,
            max_obs=900, vert_corr=X["dg"].get("vert_corr"))
        anF = dict(uniB)
        anF["q"] = anF_q
        Ans = [uniA, uniB,
               rec.apply_coupling(uniB, X["bg"], coupling=("T2V",))["an"],
               rec.apply_coupling(uniB, X["bg"], coupling=("V2q",))["an"],
               rec.apply_coupling(uniB, X["bg"], coupling=("T2V", "V2q"))["an"],
               anF]

        entry = dict(time="%s%sz" % ts, lead=lead, n_impact=int(am.sum()),
                     n_above=int(X["above"].sum()))
        for nm, an in zip(NAMES, Ans):
            for v in KEYS:
                entry["%s|%s" % (nm, v)] = rmse(an[v], X["nat"][v], am)
                entry["ALL|%s|%s" % (nm, v)] = rmse(an[v], X["nat"][v], X["above"])
        for v in KEYS:
            entry["bg|%s" % v] = rmse(X["bg"][v], X["nat"][v], am)
            entry["bgALL|%s" % v] = rmse(X["bg"][v], X["nat"][v], X["above"])
        rows.append(entry)
        print("    %s %sz 完成（影响区 %d 格点）" % (ts[0], ts[1], entry["n_impact"]),
              flush=True)
        del X, uniA, uniB, anF, rec, Ans
        gc.collect()

    print_tables(rows, lead)
    att = attribution(rows, lead)
    return dict(pairs=pairs, rows=rows, kgain={v: Kgain[v] for v in KEYS},
                sig_b=SB,
                calibration=dict(alpha_T2V=A_T2V, alpha_V2q=A_V2Q, tau_h=TAU,
                                 score=best[0], scan=scan),
                attribution=att)


def main():
    t0 = time.time()
    leads = [int(a) for a in sys.argv[1:]] or list(LEADS)
    print("=" * 92)
    print("step10  真实预报误差 OSSE：真值 = GFS f000 分析，背景 = GFS f0xx 预报")
    print("        时效 %s  区域 %s" % (leads, list(BBOX)))
    print("=" * 92, flush=True)
    res = {}
    for ld in leads:
        try:
            res[ld] = run_lead(ld)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("f%03d 失败: %s" % (ld, str(e)[:120]), flush=True)

    print("\n" + "#" * 92)
    print("#  跨时效汇总：湿度信息提取（相对背景 q RMSE 的百分点）")
    print("#" * 92)
    print("%-8s %7s %9s %9s %9s %9s %9s"
          % ("时效", "n", "bg q", "V2q耦合", "[95% CI]", "飞机直报", "耦合/直报"))
    for ld in leads:
        if ld not in res:
            continue
        a = res[ld]["attribution"]
        bg = float(np.median([r["bg|q"] for r in res[ld]["rows"]]))
        print("%-8s %7d %9.3g %8.3f%% [%+5.2f,%+5.2f] %8.3f%% %8.1f%%"
              % ("f%03d" % ld, a["n_times"], bg * 1e3, a["gain_pp"],
                 a["ci_gain"][0], a["ci_gain"][1], a["direct_pp"],
                 a["coupling_over_direct"]))
        res[ld]["attribution"]["bg_q"] = bg

    out = dict(meta=dict(bbox=list(BBOX), leads=list(leads), oerr=OERR,
                         oerr_q_ac=OERR_Q_AC, impact_r=IMPACT_R,
                         truth=[list(t) for t in TRUTH_TIMES],
                         note="truth = GFS f000 analysis; background = GFS forecast"),
               results={str(ld): dict(calibration=res[ld]["calibration"],
                                      kgain=res[ld]["kgain"],
                                      sig_b=res[ld]["sig_b"],
                                      attribution=res[ld]["attribution"],
                                      rows=res[ld]["rows"])
                        for ld in res})
    with open("data/step10_realerror_osse.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step10_realerror_osse.json   用时 %.0f s"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
