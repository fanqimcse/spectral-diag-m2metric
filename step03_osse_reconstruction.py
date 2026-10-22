# -*- coding: utf-8 -*-
"""
step03 —— 观测系统模拟试验（OSSE）：湿度盲的飞机观测能贡献多少湿度信息
======================================================================

科学命题
--------
航空业已拥有极高密度的高空观测，但它们几乎全部是**湿度盲**的：
AMDAR / Mode-S EHS / EMADDC（de Haan et al., *AMT* 18:3341, 2025）每天在
欧洲空域产出约 5500 万条风观测、3200 万条温度观测，**比湿为 0 条**。
而湿度恰恰决定密度高度、对流起始、积冰与湿理查森数。

本试验回答一个可量化的问题：

    完全不报湿度的密集飞机观测，能否通过**物理耦合**约束三维湿度场？
    若能，贡献有多大？

耦合链条
--------
    温度观测 ──[静力 + 地转平衡]──▶ 风场增量        (M_{T→V})
    风观测   ──[水汽平流位移]──────▶ 湿度场增量      (M_{V→q})  ★核心

        δq(x) = q_b(x − δV_a·τ) − q_b(x)

受控对照（本试验的设计核心）
----------------------------
    A: 地面 + 探空                    —— 无飞机
    B: 地面 + 探空 + 飞机，单变量      —— 飞机是湿度盲 ⇒ 对 q 的增益**必须为零**
    C: B + 水汽平流耦合               —— 飞机经 M_{V→q} 间接贡献湿度

    C 相对 B 的改善 = **从湿度盲观测中榨取的湿度信息量**。

    B 相对 A 恒等（对 q 而言）是一个**内部一致性检验**：
    若出现差异，说明实现有误（例如飞机湿度未被正确屏蔽）。

为什么用 OSSE
-------------
探空（IGRA）单层下载实测 7.7 KB/s，全球长序列不可得；OSSE 是资料同化领域的
标准验证手段，能精确控制观测类型、数量与误差，从而逐项归因。

⚠ 检验指标的选择（本项目踩过的大坑）
-------------------------------------
全域 RMSE **不是**衡量重构精度的合适指标。n_obs ≪ n_grid 时，
全域分析误差方差的理论下界为

    σ_b² − (n_obs − tr(R C⁻¹)) / n_grid

本项目 150 个地面观测 vs 12221 个格点，全域 RMSE 的改善上限仅约 1%，
而观测影响区（<30 km）的改善可达 40%（实测 +41.4%）。
用全域 RMSE 会把后者完全稀释，甚至得出"重构无效"的错误结论。
本试验因此以**观测影响区**为主口径，全域指标仅作对照。

时次划分（杜绝循环论证）
------------------------
    标定 : 2026-10-19 的 06/12/18 UTC   （仅用于确定耦合系数 α）
    检验 : 2026-10-20 的 00/06 UTC、2026-10-18 的 12 UTC  （最终数字只在此报告）
"""
import os, sys, json, time
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avwx import gfs, recon3d as R3

BBOX = (100, 130, 15, 40)
CAL_TIMES = [("20261019", "06"), ("20261019", "12"), ("20261019", "18")]
VAL_TIMES = [("20261020", "00"), ("20261020", "06"), ("20261018", "12")]
OUT = "data/step03_osse.json"

# 观测误差（业务典型值；σ_o **不参与标定**，是给定的输入）
OERR = dict(T=0.5, q=5e-4, u=1.5, v=1.5)
IMPACT_R = 150e3          # "观测影响区"水平半径
IMPACT_LNP = 0.5          # 垂直影响半径（ln p）

CONFIGS = [
    ("A  sfc+raob            无飞机",        ("sfc", "raob"),        ()),
    ("B  +ac 单变量(湿度盲)",                 ("sfc", "raob", "ac"),  ()),
    ("C  B+T2V 静力-地转",                    ("sfc", "raob", "ac"),  ("T2V",)),
    ("D  B+V2q 水汽平流 ★",                   ("sfc", "raob", "ac"),  ("V2q",)),
    ("E  B+全耦合",                           ("sfc", "raob", "ac"),  ("T2V", "V2q")),
]


def _xy(latv, lonv):
    return np.column_stack([np.asarray(latv, float) * 111195.0,
                            np.asarray(lonv, float) * 111195.0 *
                            np.cos(np.deg2rad(np.asarray(latv, float)))])


def impact_mask(ac_pos, lat, lon, lev, r=IMPACT_R, dlnp=IMPACT_LNP):
    """飞机观测影响区：水平 < r 且 |Δln p| < dlnp 的格点。"""
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


def build_case(dte, cyc, seed=20261020):
    nat = gfs.load(gfs.fetch(date=dte, cycle=cyc, bbox=BBOX))
    bg = R3.make_background(nat, coarse=4, noise=1.2, seed=seed)
    sim = R3.ObsSim(nat, seed=7)
    obs = dict(sfc=sim.surface(n=150), raob=sim.raob(n=12), ac=sim.aircraft())
    lat, lon, lev = nat["lat"], nat["lon"], nat["level"]
    dg = R3.diagnose_background_error(nat, bg, lat, lon, lev)
    sig_b = {v: np.array([float(np.sqrt(np.nanmean((bg[v][k] - nat[v][k]) ** 2)))
                          for k in range(len(lev))]) for v in ("T", "q", "u", "v")}
    sigma = {v: (sig_b[v], np.full(len(lev), OERR[v])) for v in sig_b}
    return nat, bg, obs, lat, lon, lev, dg, sigma


def make_rec(lat, lon, lev, dg, sigma, **kw):
    return R3.Reconstructor(lat, lon, lev, L_h=dg["L_h1"], L_z=dg["L_z"],
                            L_h2=dg["L_h2"], w2=dg["w2"],
                            sigma=sigma, max_obs=900,
                            vert_corr=dg.get("vert_corr"), **kw)


def _score(an, nat, mask, keys=("T", "q", "u", "v")):
    return {v: float(np.sqrt(np.nanmean((an[v] - nat[v]) ** 2))) for v in keys}, \
           {v: float(np.sqrt(np.nanmean((an[v][mask] - nat[v][mask]) ** 2))) for v in keys}


def run_case(nat, bg, obs, lat, lon, lev, dg, sigma, alpha_T2V, alpha_V2q, tau_h):
    """一个时次下各配置的评估。

    效率关键：单变量 OI **与耦合系数无关**，故 A、B 各只做一次，
    其余配置 (C/D/E) 复用 B 的单变量结果、仅重跑纯算子（几乎免费）。
    """
    rec = make_rec(lat, lon, lev, dg, sigma, alpha_T2V=alpha_T2V,
                   alpha_V2q=alpha_V2q, tau_h=tau_h)
    mask = impact_mask(obs["ac"][1], lat, lon, lev)
    base = _score(bg, nat, mask, ("T", "q", "u", "v"))
    rows = []
    uni_cache = {}
    for name, use, coup in CONFIGS:
        key = tuple(use)
        if key not in uni_cache:
            uni_cache[key] = rec.univariate(bg, obs, use=use)
        r = rec.apply_coupling(uni_cache[key], bg, coupling=coup)
        g, gi = _score(r["an"], nat, mask)
        rows.append(dict(name=name, use=list(use), coupling=list(coup),
                         rmse=g, rmse_impact=gi))
        del r
    return dict(base=base[0], base_impact=base[1], n_impact=int(mask.sum()),
                rows=rows, L_h1=dg["L_h1"], L_h2=dg["L_h2"],
                w2=dg["w2"], L_z=dg["L_z"])


def summarize(cases):
    """跨时次取中位。"""
    keys = ("T", "q", "u", "v")
    out = []
    for i, (name, _, _) in enumerate(CONFIGS):
        d = {}
        for k in keys:
            g = [c["rows"][i]["rmse"][k] for c in cases]
            gi = [c["rows"][i]["rmse_impact"][k] for c in cases]
            d[k] = float(np.median(g))
            d[k + "_imp"] = float(np.median(gi))
        out.append(dict(name=name, **d))
    base = {k: float(np.median([c["base"][k] for c in cases])) for k in keys}
    base_i = {k: float(np.median([c["base_impact"][k] for c in cases])) for k in keys}
    return out, base, base_i


def main():
    t0 = time.time()
    print("[step03] 载入标定时次 …", flush=True)
    cal = [build_case(d, c) for d, c in CAL_TIMES]
    lat, lon, lev = cal[0][3], cal[0][4], cal[0][5]
    print(f"[step03] 网格 {len(lat)}x{len(lon)} 层数 {len(lev)}", flush=True)

    # ---- 在标定时次上标定耦合系数 ----
    # 效率关键：单变量 OI **与 α 无关**，故整轮扫描只算一次 OI，
    # 其余仅重跑纯耦合算子（含水汽平流位移，成本可忽略）。
    print("\n[step03] 标定耦合系数（仅标定时次；单变量 OI 只算一次）", flush=True)
    nat, bg, obs, _, _, _, dg, sg = cal[0]
    rec_cal = make_rec(lat, lon, lev, dg, sg)
    uni_cal = rec_cal.univariate(bg, obs, use=("sfc", "raob", "ac"))
    msk = impact_mask(obs["ac"][1], lat, lon, lev)
    bi = {v: float(np.sqrt(np.nanmean((bg[v][msk] - nat[v][msk]) ** 2)))
          for v in ("T", "q", "u", "v")}
    q0 = float(np.sqrt(np.nanmean((uni_cal["q"][msk] - nat["q"][msk]) ** 2)))
    u0 = float(np.sqrt(np.nanmean((uni_cal["u"][msk] - nat["u"][msk]) ** 2)))
    best, scan = None, []
    for aT in (0.0, 0.3, 0.6, 1.0):
        for aq in (0.0, 0.5, 1.0, 1.5):
            for tau in (1.0, 2.0, 3.0):
                rec_cal.alpha_T2V, rec_cal.alpha_V2q, rec_cal.tau_h = aT, aq, tau
                rr = rec_cal.apply_coupling(uni_cal, bg, coupling=("T2V", "V2q"))
                qc = float(np.sqrt(np.nanmean((rr["an"]["q"][msk] - nat["q"][msk]) ** 2)))
                uc = float(np.sqrt(np.nanmean((rr["an"]["u"][msk] - nat["u"][msk]) ** 2)))
                sc = (q0 - qc) / bi["q"] + (u0 - uc) / bi["u"]
                scan.append(dict(alpha_T2V=aT, alpha_V2q=aq, tau_h=tau, score=sc))
                if best is None or sc > best[0]:
                    best = (sc, aT, aq, tau)
                del rr
    _, A_T2V, A_V2Q, TAU = best
    print(f"[step03] 标定结果: alpha_T2V={A_T2V:.2f}  alpha_V2q={A_V2Q:.2f}  tau_h={TAU:.1f}"
          f"   (评分 {best[0]*100:+.3f}%)", flush=True)
    del cal, nat, bg, obs, uni_cal, rec_cal, msk
    import gc; gc.collect()

    # ---- 独立检验时次 ----
    print("\n[step03] 载入检验时次（最终数字只在此报告）…", flush=True)
    val = []
    for d, c in VAL_TIMES:
        try:
            val.append(run_case(*build_case(d, c), alpha_T2V=A_T2V,
                                alpha_V2q=A_V2Q, tau_h=TAU))
            print(f"    {d}{c}Z 完成", flush=True)
        except Exception as e:
            print(f"    {d}{c}Z 跳过 {repr(e)[:50]}", flush=True)

    rows, base, base_i = summarize(val)
    n_imp = int(np.median([c["n_impact"] for c in val]))
    ngrid = len(lat) * len(lon) * len(lev)

    def show(title, base_d, suf=""):
        print("\n" + "=" * 86)
        print(title)
        print("=" * 86)
        print(f"{'配置':30s} {'T':>12s} {'q':>12s} {'u':>12s} {'v':>12s}")
        print(f"{'(背景场绝对值)':30s} " +
              "  ".join(f"{base_d[k]:12.4g}" for k in ("T", "q", "u", "v")))
        for r in rows:
            s = "  ".join(f"{100*(base_d[k]-r[k+suf])/base_d[k]:11.2f}%" for k in ("T", "q", "u", "v"))
            print(f"{r['name']:30s} {s}")

    show(f"【表】相对背景场的 RMSE 降幅 —— 观测影响区（{n_imp} 格点，占全域 {100*n_imp/ngrid:.1f}%）",
         base_i, "_imp")
    show("【对照】全域 RMSE 降幅（观测数≪格点数，理论上限约 1%，仅作参考）", base)

    # ---- 核心结果 ----
    g = lambda nm, k, suf="": next(r[nm if False else k + suf] for r in rows if r["name"].startswith(nm))
    qA = g("A", "q", "_imp"); qB = g("B", "q", "_imp"); qD = g("D", "q", "_imp")
    qE = g("E", "q", "_imp"); qC = g("C", "q", "_imp")
    print("\n" + "=" * 86)
    print("【核心结果】湿度盲的飞机观测，通过水汽平流耦合榨出多少湿度信息？")
    print("=" * 86)
    print(f"  A 无飞机（地面+探空）         q RMSE = {qA:.6g}  相对背景 {100*(base_i['q']-qA)/base_i['q']:+.3f}%")
    print(f"  B 加飞机·单变量（湿度盲）     q RMSE = {qB:.6g}  相对背景 {100*(base_i['q']-qB)/base_i['q']:+.3f}%")
    print(f"     ↳ 内部一致性 |B−A|/A = {abs(qB-qA)/qA*100:.4f}%  （应≈0，飞机不报湿度）")
    print(f"  D 加飞机·加水汽平流耦合 ★    q RMSE = {qD:.6g}  相对背景 {100*(base_i['q']-qD)/base_i['q']:+.3f}%")
    print(f"     ↳ ★ 从湿度盲观测榨取的湿度信息 = {100*(qB-qD)/base_i['q']:+.4f} 个百分点")
    print(f"     ↳  等价地，相对单变量分析的再改善 = {100*(qB-qD)/qB:+.3f}%")
    print(f"  E 全耦合                      q RMSE = {qE:.6g}  相对背景 {100*(base_i['q']-qE)/base_i['q']:+.3f}%")

    print("\n" + "=" * 86)
    print("【耦合项对风/温场的贡献】")
    print("=" * 86)
    for k, nmU, nmX in (("u", "B", "C"), ("v", "B", "C")):
        a = g(nmU, k, "_imp"); b = g(nmX, k, "_imp")
        print(f"  {k}: 单变量 {a:.6g} → 加静力-地转 {b:.6g}   "
              f"额外降幅 {100*(a-b)/base_i[k]:+.3f}%")
    for k in ("T", "u", "v"):
        a = g("B", k, "_imp"); b = g("D", k, "_imp")
        print(f"  {k}: 单变量 {a:.6g} → 加水汽平流 {b:.6g}   "
              f"（水汽平流不应影响 {k}：{100*(a-b)/base_i[k]:+.4f}%）")

    out = dict(bbox=list(BBOX), grid=[len(lat), len(lon), len(lev)],
               levels=lev.tolist(),
               cal_times=[f"{d}{c}" for d, c in CAL_TIMES],
               val_times=[f"{d}{c}" for d, c in VAL_TIMES],
               alpha_T2V=A_T2V, alpha_V2q=A_V2Q, tau_h=TAU,
               impact_radius_km=IMPACT_R / 1e3, n_impact=n_imp, n_grid=ngrid,
               base=base, base_impact=base_i, rows=rows,
               headline=dict(
                   q_A=qA, q_B=qB, q_C=qC, q_D=qD, q_E=qE,
                   internal_consistency_pct=abs(qB - qA) / qA * 100,
                   gain_from_blind_obs_pct=100 * (qB - qD) / base_i["q"],
                   gain_rel_univariate_pct=100 * (qB - qD) / qB,
               ))
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[step03] 已写出 {OUT}   用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
