# -*- coding: utf-8 -*-
"""step13 —— 端到端航迹风险产品：把 δ、主指标 A 与 EDR 校准权重接进评分。

为什么不能沿用"0–100 分"的指数
------------------------------
step04 给出的是 cat_dry_frac / emerge_frac 一类**无量纲比例**。它们能说明物理，
但没法回答签派真正要问的问题："这条航路上，遭遇 light-or-greater 颠簸的
**概率**是多少？"——比例不是概率，也没有经过观测校准。

本脚本做四件事，把研究成果落到产品上：

  1. **校准**：用 step11b 那批客观 EDR 观测，对每个诊断**单独**拟合一元
     逻辑回归  logit P(EDR ≥ t) = a + b·(−log10 Ri)。
     三个诊断（干式 / 虚温式 / DK 饱和式）各自标定后比 Brier / AUC，
     这就是"用 EDR 把 GTG 当年用 PIREP 做的权重标定重做一遍"。

  2. **越界处理**：Ri ≤ 0（湿式层结翻转）超出 −log10 的支撑集，
     不靠外推，改用训练样本中该层的**实测频率** P(EDR ≥ t | Ri ≤ 0)。

  3. **评估**：Brier / BSS / 可靠性图。校准过的概率必须落在可靠性图对角线上，
     否则不能直接给用户看。

  4. **应用**：沿航迹逐点给出标定概率，并按物理状态选择口径
     （晴空 ⇒ 虚温式；云内 ⇒ DK 饱和式），与"全程干式"的业务基线对照，
     给出 ΔP —— 这是 H2 在**产品端**的表述。

ICAO 中型机 ε^{1/3} 阈值：0.1 light、0.22 moderate。
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from avwx import risk as RK
from avwx import verify as VF
from avwx import calib as CB

EDR_LGT, EDR_MOD = 0.10, 0.22
RH_CLEAR = 80.0
N2_FLOOR = RK.N2_FLOOR


# ------------------------------------------------------------------ 逻辑回归
def fit_logit(X, y, ridge=1e-6, iters=80):
    """IRLS / Newton–Raphson。X 已含常数列。"""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    p = X.shape[1]
    beta = np.zeros(p)
    for _ in range(iters):
        eta = X @ beta
        pr = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = np.clip(pr * (1 - pr), 1e-8, None)
        g = X.T @ (y - pr)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(p)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = X @ beta
    pr = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    W = np.clip(pr * (1 - pr), 1e-8, None)
    cov = np.linalg.pinv(X.T @ (X * W[:, None]) + ridge * np.eye(p))
    return beta, np.sqrt(np.diag(cov))


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability(p, y, nb=10):
    """可靠性图数据：观测频率 vs 预报概率。"""
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, nb + 1)
    out = []
    for i in range(nb):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() < 5:
            out.append(dict(center=0.5 * (edges[i] + edges[i + 1]),
                            n=int(m.sum()), fcst=None, obs=None))
            continue
        out.append(dict(center=0.5 * (edges[i] + edges[i + 1]), n=int(m.sum()),
                        fcst=float(p[m].mean()), obs=float(y[m].mean())))
    return out


def main():
    pkl = "data/madis/step11b_matched.pkl"
    if not os.path.exists(pkl):
        raise SystemExit("缺少 %s，先跑 step11b" % pkl)
    with open(pkl, "rb") as fh:
        rows = pickle.load(fh)

    def stack(k):
        return np.concatenate([r[k] for r in rows])

    p = stack("p")
    rh = stack("rh")
    S2 = stack("S2")
    Nd2, Nv2, Ndk2 = stack("Nd2"), stack("Nv2"), stack("Ndk2")
    med = np.where(np.isfinite(stack("medEDR")), stack("medEDR"), 0.0)
    mx = np.where(np.isfinite(stack("maxEDR")), stack("maxEDR"), 0.0)

    def ri(n2):
        return n2 / np.maximum(S2, 1e-12)

    x = lambda z: -np.log10(np.maximum(np.asarray(z, float), 1e-6))
    XD, XV, XK = x(ri(Nd2)), x(ri(Nv2)), x(ri(Ndk2))
    good = (np.isfinite(XD) & np.isfinite(XV) & np.isfinite(XK)
            & np.isfinite(rh) & np.isfinite(p))
    print("样本 %d（EDR 有效且诊断量有限 %d）" % (len(p), int(good.sum())))

    res = {}
    calib = {}
    # 产品默认用 maxEDR（"航段内峰值"对应"遭遇"语义）；medEDR 作为稳健性对照
    for tag, thr in (("light", EDR_LGT), ("moderate", EDR_MOD)):
        for ename, esrc in (("max", mx), ("med", med)):
            y = (esrc >= thr).astype(float)
            m = good & np.isfinite(y)
            for nm, sub in (("全部", np.ones(int(m.sum()), bool)),
                            ("晴空+高层", (rh[m] < RH_CLEAR) & (p[m] <= 400))):
                mm = np.where(m)[0][sub]
                key = "%s|%s|%s" % (tag, nm, ename)
                if mm.size < 400 or y[mm].sum() < 30:
                    print("  %-22s 样本不足 (%d)" % (key, mm.size))
                    continue
                clim = float(y[mm].mean())
                ent = dict(n=int(mm.size), clim=clim, thr=thr, edr=ename)
                cblk = {}

                # ---- 三个**单诊断**模型：避开共线性 ----
                # ⚠ 踩坑：曾把 x_dry / x_virt / x_DK 与 log S² 同时放进一个模型。
                #   但 Ri = N²/S² ⟹ −log10 Ri = −log10 N² + log10 S²，
                #   三个 x 之间、以及它们与 logS2 之间几乎完全共线，
                #   晴空子集直接解出 β_dry=+41 / β_virt=−41 这类病态系数。
                single = {}
                for sname, sv, raw in (("dry", XD[mm], ri(Nd2)[mm]),
                                       ("virt", XV[mm], ri(Nv2)[mm]),
                                       ("DK", XK[mm], ri(Ndk2)[mm])):
                    X1 = np.column_stack([np.ones(mm.size), sv])
                    b1, s1 = fit_logit(X1, y[mm])
                    pr1 = 1 / (1 + np.exp(-np.clip(X1 @ b1, -30, 30)))
                    bs1 = brier(pr1, y[mm])
                    # Ri ≤ 0 层的实测频率（不推断、不外推）
                    negm = raw <= 0
                    pneg = (float(y[mm][negm].mean()) if negm.sum() >= 20
                            else None)
                    single[sname] = dict(
                        beta=[float(b1[0]), float(b1[1])],
                        se=[float(s1[0]), float(s1[1])],
                        z=float(b1[1] / max(s1[1], 1e-12)),
                        p_neg=pneg, n_neg=int(negm.sum()),
                        brier=bs1, bss=1 - bs1 / (clim * (1 - clim)),
                        auc=VF.auc_score(pr1, y[mm]))
                    cblk[sname] = [float(b1[0]), float(b1[1]), pneg]
                ent["single"] = single
                ent["best_single"] = max(single, key=lambda k: single[k]["bss"])
                calib[key] = cblk

                # ---- 多元岭回归：仅作参考，附条件数 ----
                Xd = np.column_stack([np.ones(mm.size), XD[mm], XV[mm], XK[mm]])
                names = ["const", "x_dry", "x_virt", "x_DK"]
                cond = float(np.linalg.cond(Xd))
                lam = 0.05 * mm.size
                beta, se = fit_logit(Xd, y[mm], ridge=lam)
                z = beta / np.maximum(se, 1e-12)
                prM = 1 / (1 + np.exp(-np.clip(Xd @ beta, -30, 30)))
                bsM = brier(prM, y[mm])
                ent.update(
                    cond=cond, ridge=lam,
                    beta=dict(zip(names, beta.tolist())),
                    se=dict(zip(names, se.tolist())),
                    z=dict(zip(names, z.tolist())),
                    brier_multi=bsM, bss_multi=1 - bsM / (clim * (1 - clim)),
                    auc_multi=VF.auc_score(prM, y[mm]),
                    reliability=reliability(prM, y[mm]))
                res[key] = ent

                print("\n[%-22s] n=%d 气候率=%.4f 条件数=%.0f"
                      % (key, mm.size, clim, cond))
                print("   单诊断 BSS/AUC/β : " + "  ".join(
                    "%s %.4f/%.3f(β=%+.3f,z=%+.1f)"
                    % (k, single[k]["bss"], single[k]["auc"],
                       single[k]["beta"][1], single[k]["z"])
                    for k in ("dry", "virt", "DK")))
                print("   最优单诊断 = %s ；多元岭回归 BSS %.4f / AUC %.3f"
                      % (ent["best_single"], ent["bss_multi"],
                         ent["auc_multi"]))

    # ---------------- 应用：沿航迹标定概率 ----------------
    print("\n" + "=" * 96)
    print("【应用】沿航迹输出标定概率（东京 → 新加坡，巡航 200 hPa）")
    print("=" * 96)
    g = sorted(glob.glob("data/aws/gfs_*_f000_L21_1000_100.grib2"))
    if not g:
        print("无本地 GFS 场，跳过应用部分")
    else:
        from avwx import aws_gfs as AG
        d = AG.load(g[0], bbox=(95.0, 150.0, -10.0, 45.0))
        lat, lon, lev = d["lat"], d["lon"], d["level"]
        fld = dict(T=d["T"], q=d["q"], u=d["u"], v=d["v"], z=d["z"])
        diag = RK.stability_fields(fld, lat, lon, lev)
        p0, p1 = (35.55, 139.78), (1.36, 103.99)
        route = RK.assess_route(fld, diag, lat, lon, lev, p0, p1,
                                cruise_hpa=200, n=120)
        cruise = route["lev"] <= 200 * 1.15
        rh_r = route["rh"]

        app = {}
        for tag in ("light", "moderate"):
            for pol in ("dry", "switch"):
                try:
                    r = CB.route_probability(
                        route["Ri_dry"], route["Ri_moist"], route["Ri_dk"]
                        if "Ri_dk" in route else route["Ri_moist"],
                        rh_r, {"light|全部|max": calib.get("light|全部|max"),
                               "moderate|全部|max": calib.get("moderate|全部|max")},
                        tag=tag, edr="max", subset_policy=pol)
                except Exception as e:
                    print("  %s/%s 失败：%s" % (tag, pol, e))
                    continue
                app["%s|%s" % (tag, pol)] = {
                    k: (None if isinstance(v, np.ndarray) else v)
                    for k, v in r.items()}
                print("  [%s | %-6s] 单点P中位 %.4f 最大 %.4f | "
                      "至少一次 %.3f | 期望遭遇点 %.2f/%d"
                      % (tag, pol, float(np.nanmedian(r["p_point"])),
                         r["p_max"], r["p_encounter"], r["n_expect"],
                         r["n_valid"]))

        # 晴空段：干式 vs 虚温式 的 ΔP —— H2 在产品端的表述
        clr = cruise & np.isfinite(rh_r) & (rh_r < RH_CLEAR)
        dP = {}
        for tag in ("light", "moderate"):
            mk = calib.get("%s|全部|max" % tag)
            if not mk:
                continue
            pd = CB.p_from_ri(route["Ri_dry"], *CB.models_for(
                {("%s|全部|max" % tag): mk}, tag, "全部", "max")["dry"])
            pv = CB.p_from_ri(route["Ri_moist"], *CB.models_for(
                {("%s|全部|max" % tag): mk}, tag, "全部", "max")["virt"])
            d = np.where(clr & np.isfinite(pd) & np.isfinite(pv), pv - pd, np.nan)
            if np.isfinite(d).any():
                dP[tag] = dict(median=float(np.nanmedian(d)),
                               mean=float(np.nanmean(d)),
                               p90=float(np.nanpercentile(d, 90)))
                print("  晴空段 ΔP = P(virt) − P(dry) [%s]：中位 %+.4f "
                      "均值 %+.4f  P90 %+.4f"
                      % (tag, dP[tag]["median"], dP[tag]["mean"],
                         dP[tag]["p90"]))

        # 主指标 A：航迹上被干口径**漏判的虚温不稳定层**
        nd2r = RK.interpolate_field(diag["Nd2"], lat, lon, lev,
                                    route["lat"], route["lon"] % 360.0,
                                    route["lev"])
        nv2r = RK.interpolate_field(diag["Nv2"], lat, lon, lev,
                                    route["lat"], route["lon"] % 360.0,
                                    route["lev"])
        A = clr & (nd2r > 0) & (nv2r < 0)
        print("  主指标 A（干判稳定、虚温判不稳定）航迹晴空段占比 = %.3f%%"
              % (100 * float(np.mean(A)) if clr.any() else float("nan")))
        print("  δ(湿度对稳定度削弱) 巡航段中位 = %.3f%%"
              % (100 * route.get("delta_median", np.nan)))

        res["route_demo"] = dict(
            app=app, dP_clear=dP,
            A_route_clear_frac=float(np.mean(A)) if clr.any() else None,
            delta_median=route.get("delta_median", None),
            cat_dry_frac=route.get("cat_dry_frac", None),
            emerge_frac=route.get("emerge_frac", None),
            emerge_clear_frac=route.get("emerge_clear_frac", None),
            n2_weaken_median=route.get("n2_weaken_median", None))

    out = dict(res=res, calib=calib)
    os.makedirs("data", exist_ok=True)
    with open(CB.CALIB_FILE, "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 %s" % CB.CALIB_FILE)
    c = CB.load()
    print("标定块可被 avwx.calib 读回：%s"
          % ("是" if c else "否"))


if __name__ == "__main__":
    main()
