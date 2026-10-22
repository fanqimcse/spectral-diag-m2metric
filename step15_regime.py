# -*- coding: utf-8 -*-
"""step15 —— 分"层结状态 × 高度"核验：把 H2 的机制从"偏差"改判为"伪方差注入"。

出发点（一个必须自己先回答的质疑）
--------------------------------
step06 的 δ 气候学显示：δ 随高度单调下降，200 hPa 只有 **0.11%**。
于是最自然的审稿意见是：

    "你主张湿度修正会改变晴空湍流诊断，但在商用机巡航高度上
     δ 只有千分之一 —— 这个机制在它最该起作用的地方几乎为零。
     整个工作是不是只在中低层有意义？"

这个质疑是对的，本脚本就是回答它。答案分两半，而且**两半方向相反**：

  第一半（对虚温式）：质疑成立。
      在巡航层，真实湿度修正确实小到可以忽略；EDR 检验里 virt 与 dry 的
      AUC/BSS 在所有高度层上都不分彼此（H1 为零假设，再次确认）。
      ⇒ 所以本项目的结论**不是**"把湿度加进去就能报得更准"。

  第二半（对饱和式）：质疑不成立，而且恰恰因为 δ→0 才更严重。
      饱和式（Durran–Klemp）的湿度修正只通过 Γ_m(T,p) 与 r_s(T,p) 进入，
      两者**与实际水汽无关**。于是当 q→0（上对流层）时：
          真实修正 δ      → 0          （虚温式，∝ dq/dz）
          饱和式修正 δ_sat → 5–13%     （只依赖 T,p）
      它注入的是一个**有量级、但与真实湿度效应不相关**的扰动：
          · 幅度比真实效应大 2 个数量级；
          · 与真实效应相关性≈0。
      这不是"系统性偏差"（那可以通过重新标定阈值吸收掉），
      而是**向诊断量里注入噪声**——重新标定救不了它。
      可观测后果：晴空+高层子集上 DK 的 BSS 从 0.0224 塌到 0.0001。

本脚本给出的三个可直接写进论文的数：
  (1) 分层 EDR 检验：dry / virt / DK 的 AUC 与 BSS，按高度 × 晴云分列；
  (2) 伪方差比 σ(δ_sat)/σ(δ) 与相关系数 corr(δ, δ_sat)，按高度分列；
  (3) 一个"可标定性"判据：把 DK 重新标定后是否还能追上 dry（答案：不能）。
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from avwx import verify as VF

RH_CLEAR = 80.0
EDR_LGT = 0.10

BANDS = [("高层 p≤300", 0.0, 300.0),
         ("中高层 300–500", 300.0, 500.0),
         ("中层 500–700", 500.0, 700.0),
         ("低层 p>700", 700.0, 1e9)]


def fit_logit1(x, y, ridge=1e-8, iters=80):
    """一元逻辑回归 logit P = a + b x（含常数）。"""
    X = np.column_stack([np.ones(x.size), x])
    p = 2
    beta = np.zeros(p)
    for _ in range(iters):
        pr = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        W = np.clip(pr * (1 - pr), 1e-8, None)
        g = X.T @ (y - pr)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(p)
        try:
            st = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        beta = beta + st
        if np.max(np.abs(st)) < 1e-9:
            break
    return beta


def brier(p, y):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def diag_scores(xs, y, boot=400, seed=7):
    """单诊断 → (AUC, BSS, β)。"""
    x = np.asarray(xs, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, yy = x[ok], np.asarray(y, float)[ok]
    if x.size < 200 or not (0 < yy.mean() < 1):
        return None
    b = fit_logit1(x, yy)
    pr = 1.0 / (1.0 + np.exp(-np.clip(b[0] + b[1] * x, -30, 30)))
    clim = float(yy.mean())
    bs = brier(pr, yy)
    return dict(n=int(x.size), clim=clim, auc=VF.auc_score(x, yy),
                bss=1 - bs / (clim * (1 - clim)), beta=float(b[1]),
                brier=bs)


def main():
    pkl = "data/madis/step11b_matched.pkl"
    with open(pkl, "rb") as fh:
        rows = pickle.load(fh)

    def stack(k):
        return np.concatenate([r[k] for r in rows])

    p = stack("p")
    rh = stack("rh")
    S2 = stack("S2")
    Nd2, Nv2, Ndk2 = stack("Nd2"), stack("Nv2"), stack("Ndk2")
    mx = np.where(np.isfinite(stack("maxEDR")), stack("maxEDR"), 0.0)
    y = (mx >= EDR_LGT).astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(Nd2 > 1e-5, (Nd2 - Nv2) / Nd2, np.nan)
        dsat = np.where(Nd2 > 1e-5, (Nd2 - Ndk2) / Nd2, np.nan)
    xr = lambda n2: -np.log10(np.maximum(n2 / np.maximum(S2, 1e-12), 1e-6))
    XD, XV, XK = xr(Nd2), xr(Nv2), xr(Ndk2)
    good = np.isfinite(XD) & np.isfinite(XV) & np.isfinite(XK) & np.isfinite(rh)

    out = dict(bands={}, spurious={})

    # ---------------- (1) 分层 EDR 检验 ----------------
    print("=" * 100)
    print("(1) 分层 EDR 检验   y = 1[maxEDR ≥ 0.10]")
    print("=" * 100)
    print("%-16s %-6s %6s %7s | %-15s | %-15s | %-15s | %9s"
          % ("高度层", "子集", "n", "气候率", "dry AUC/BSS", "virt AUC/BSS",
             "DK AUC/BSS", "DK N²≤0"))
    for bname, lo, hi in BANDS:
        for sname, smask in (("全部", np.ones(good.sum(), bool)),
                             ("晴空", rh < RH_CLEAR)):
            m = good & (p > lo) & (p <= hi) & smask & np.isfinite(y)
            if m.sum() < 400 or y[m].sum() < 30:
                print("%-16s %-6s %6d  样本/正例不足" % (bname, sname, m.sum()))
                continue
            sc = {k: diag_scores(v[m], y[m])
                  for k, v in (("dry", XD), ("virt", XV), ("DK", XK))}
            fneg = float(np.mean(Ndk2[m] <= 0)) if m.any() else float("nan")
            for k, v in sc.items():
                if v:
                    v["fN2_le0"] = fneg
            out["bands"]["%s|%s" % (bname, sname)] = {
                k: v for k, v in sc.items() if v}
            print("%-16s %-6s %6d %6.3f | %-15s | %-15s | %-15s | %8.1f%%"
                  % (bname, sname, int(m.sum()), float(y[m].mean()),
                     ("%.3f/%.4f" % (sc["dry"]["auc"], sc["dry"]["bss"]))
                     if sc["dry"] else "—",
                     ("%.3f/%.4f" % (sc["virt"]["auc"], sc["virt"]["bss"]))
                     if sc["virt"] else "—",
                     ("%.3f/%.4f" % (sc["DK"]["auc"], sc["DK"]["bss"]))
                     if sc["DK"] else "—", 100 * fneg))

    # ---------------- (2) 伪方差比与相关性 ----------------
    print("\n" + "=" * 100)
    print("(2) 晴空区：饱和式注入的伪修正 vs 真实湿度修正")
    print("=" * 100)
    print("%-16s %7s | %10s %10s | %8s %9s | %8s %8s | %10s"
          % ("高度层", "n", "δ 中位%", "δ_sat 中位%", "σ(δ)%", "σ(δ_sat)%",
             "比值", "corr", "正交噪声/信号"))
    clr = good & (rh < RH_CLEAR) & np.isfinite(delta) & np.isfinite(dsat)
    for bname, lo, hi in BANDS:
        m = clr & (p > lo) & (p <= hi)
        # 比值型统计量在近中性层会发散，故用 5–95 分位内样本
        d, ds = delta[m], dsat[m]
        keep = np.isfinite(d) & np.isfinite(ds)
        if keep.sum() < 200:
            continue
        d, ds = d[keep], ds[keep]
        lo5, hi5 = np.percentile(d, [5, 95])
        k2 = (d >= lo5) & (d <= hi5)
        sd, sds = float(np.std(d[k2])), float(np.std(ds[k2]))
        cc = float(np.corrcoef(d[k2], ds[k2])[0, 1])
        # 关键统计量：把 δ_sat **正交分解**为"与真实信号平行"和"正交"两部分。
        #   平行分量可以通过重新标定吸收掉（只是系数不同）；
        #   正交分量是纯噪声，无论怎么标定都消不掉。
        #   σ_⊥ = σ(δ_sat)·√(1−r²)，与信号 σ(δ) 之比即"标定救不了的噪声倍数"。
        sd_orth = sds * np.sqrt(max(1.0 - cc * cc, 0.0))
        orth_ratio = sd_orth / sd if sd > 0 else float('nan')
        out["spurious"][bname] = dict(
            n=int(keep.sum()), delta_med=float(np.median(d)),
            dsat_med=float(np.median(ds)), sd=sd, sd_sat=sds,
            ratio=(sds / sd if sd > 0 else None), corr=cc,
            sd_orth=sd_orth, orth_ratio=orth_ratio)
        print("%-16s %7d | %9.4f%% %9.4f%% | %7.4f%% %8.4f%% | %7.1f %8.3f | %9.1f×"
              % (bname, int(keep.sum()), 100 * np.median(d),
                 100 * np.median(ds), 100 * sd, 100 * sds,
                 (sds / sd if sd > 0 else float('nan')), cc, orth_ratio))

    # ---------------- (3) 可标定性判据 ----------------
    print("\n" + "=" * 100)
    print("(3) 可标定性判据：把 DK 单独重新标定后，能否追上 dry？")
    print("=" * 100)
    print("（逻辑：系统性偏差可被重新标定吸收；伪方差不能。"
          "若重新标定后 DK 仍显著劣于 dry，则问题不是标定而是口径。）")
    band_hi = good & (p <= 300) & (rh < RH_CLEAR) & np.isfinite(y)
    for bname, m in (("高层+晴空", band_hi),
                     ("全部高度+晴空", good & (rh < RH_CLEAR) & np.isfinite(y))):
        if m.sum() < 400 or y[m].sum() < 30:
            continue
        sc = {k: diag_scores(v[m], y[m])
              for k, v in (("dry", XD), ("DK", XK))}
        if not sc["dry"] or not sc["DK"]:
            continue
        # 配对 bootstrap：ΔBSS = BSS(DK) − BSS(dry)
        rng = np.random.default_rng(11)
        idx = np.where(m)[0]
        dbs = []
        for _ in range(300):
            s = rng.choice(idx, size=idx.size, replace=True)
            a = diag_scores(XD[s], y[s], boot=0)
            b = diag_scores(XK[s], y[s], boot=0)
            if a and b:
                dbs.append(b["bss"] - a["bss"])
        ci = (float(np.percentile(dbs, 2.5)), float(np.percentile(dbs, 97.5))) \
            if len(dbs) > 20 else (float('nan'), float('nan'))
        out.setdefault("recalib", {})[bname] = dict(
            dry_bss=sc["dry"]["bss"], dk_bss=sc["DK"]["bss"],
            dbss=sc["DK"]["bss"] - sc["dry"]["bss"], ci=list(ci),
            nboot=len(dbs))
        print("  %-14s n=%5d  BSS dry %.5f  DK %.5f  ΔBSS %+.5f  95%%CI [%+.5f, %+.5f]"
              % (bname, int(m.sum()), sc["dry"]["bss"], sc["DK"]["bss"],
                 sc["DK"]["bss"] - sc["dry"]["bss"], ci[0], ci[1]))

    with open("data/step15_regime.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step15_regime.json")


if __name__ == "__main__":
    main()
