# -*- coding: utf-8 -*-
"""step17 —— 样本外验证、bootstrap 置信区间与检验功效。

审稿人会立刻问的三件事，本脚本一次性回答：

  1. **所有已报的 BSS/AUC 都是样本内的**（系数在同一批数据上拟合与评估）。
     逻辑回归只有 2 个参数、样本 1 万，过拟合风险不大，但"不大"不是"没有"，
     而且必须给出数字。这里做**分组留出**：按 region 留一、按 time 前后半留一。

  2. **头条数字没有置信区间**。step11b 的 ΔAUC 只给了点估计。
     这里对每个高度层的 ΔAUC(DK − dry) 与 ΔBSS 做**配对** bootstrap
     （同一个重抽样上同时重算两个诊断，差值才是对的）。

  3. **样本够不够？** AUC ≈ 0.6 的差异若置信区间宽到 ±0.05，
     "DK 更差"就可能只是抽样噪声。这里给出**最小可探测 ΔAUC**（80% 功效）。

分组留出的必要性
----------------
EDR 样本是"航段"而非独立点：同一架飞机连续报的几十条记录高度相关
（同一条航路、同一个天气系统）。逐点随机拆分会把相关样本分到两边，
乐观地抬高样本外表现。因此必须按**组**（这里的一步 = 一个 region×time
组合）整块留出。
"""
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from avwx import verify as VF

RH_CLEAR = 80.0
EDR_LGT = 0.10
BANDS = [("高层 p≤300", 0.0, 300.0),
         ("中高层 300–500", 300.0, 500.0),
         ("中层 500–700", 500.0, 700.0)]


def fit_logit1(x, y, ridge=1e-8, iters=80):
    X = np.column_stack([np.ones(x.size), x])
    beta = np.zeros(2)
    for _ in range(iters):
        pr = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))
        W = np.clip(pr * (1 - pr), 1e-8, None)
        try:
            st = np.linalg.solve(X.T @ (X * W[:, None]) + ridge * np.eye(2),
                                 X.T @ (y - pr))
        except np.linalg.LinAlgError:
            break
        beta = beta + st
        if np.max(np.abs(st)) < 1e-9:
            break
    return beta


def bss_of(x, y, a=None, b=None):
    """给 x,y 算 BSS；a,b 给定时用样本外系数（不再拟合）。"""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[ok], np.asarray(y, float)[ok]
    if x.size < 100 or not (0 < y.mean() < 1):
        return None
    if a is None:
        bb = fit_logit1(x, y)
        a, b = bb[0], bb[1]
    pr = 1.0 / (1.0 + np.exp(-np.clip(a + b * x, -30, 30)))
    clim = float(y.mean())
    return dict(n=int(x.size), clim=clim, auc=VF.auc_score(x, y),
                bss=1 - float(np.mean((pr - y) ** 2)) / (clim * (1 - clim)),
                a=float(a), b=float(b))


def pair_boot_stat(x1, x2, y, fn, nboot=600, seed=23):
    """配对 bootstrap：同一重抽样上算 fn(x1)−fn(x2)。"""
    rng = np.random.default_rng(seed)
    n = y.size
    out = []
    for _ in range(nboot):
        s = rng.integers(0, n, n)
        a = fn(x1[s], y[s])
        b = fn(x2[s], y[s])
        if a and b:
            out.append(a - b)
    return out


def main():
    with open("data/madis/step11b_matched.pkl", "rb") as fh:
        rows = pickle.load(fh)

    def stack(k):
        return np.concatenate([r[k] for r in rows])

    # 分组标签：每条记录属于哪个 (region, time) 组
    gid = np.concatenate([np.full(len(r["p"]), i) for i, r in enumerate(rows)])
    greg = np.array([r["region"] for r in rows])
    gtim = np.array([r["time"] for r in rows])

    p = stack("p")
    rh = stack("rh")
    S2 = stack("S2")
    Nd2, Nv2, Ndk2 = stack("Nd2"), stack("Nv2"), stack("Ndk2")
    mx = np.where(np.isfinite(stack("maxEDR")), stack("maxEDR"), 0.0)
    y = (mx >= EDR_LGT).astype(float)
    xr = lambda n2: -np.log10(np.maximum(n2 / np.maximum(S2, 1e-12), 1e-6))
    XD, XK = xr(Nd2), xr(Ndk2)
    good = np.isfinite(XD) & np.isfinite(XK) & np.isfinite(rh) & np.isfinite(y)
    clear = rh < RH_CLEAR

    out = {}

    # ---------------- (1) 分组留出 ----------------
    print("=" * 100)
    print("(1) 分组留出：系数在留出组之外拟合，再在留出组上评估")
    print("=" * 100)
    print("%-16s %-10s %6s | %-18s | %-18s | %10s"
          % ("高度层", "留出方式", "n", "dry BSS(样本外)", "DK BSS(样本外)",
             "ΔBSS"))
    splits = [("留一 region", [("region", v) for v in sorted(set(greg))]),
              ("前半/后半 time", [("time_half", 0), ("time_half", 1)])]
    # ⚠ time 是字符串（形如 '2026100112'），不能直接取中位数；按字典序排序后
    #   按下标切成前后两半，保证两半的**样本量**接近（按值切会因为日期分布
    #   不均而切歪）。
    order = np.argsort(gtim)
    thalf = np.zeros(len(gtim), int)
    thalf[order[len(order) // 2:]] = 1
    for bname, lo, hi in BANDS:
        for sname, parts in splits:
            dbs, nbs = [], 0
            for part in parts:
                if part[0] == "region":
                    hold = np.isin(gid, np.where(greg == part[1])[0])
                else:
                    hold = np.isin(gid, np.where(thalf == part[1])[0])
                tr = good & (~hold) & (p > lo) & (p <= hi) & clear
                te = good & hold & (p > lo) & (p <= hi) & clear
                if tr.sum() < 300 or te.sum() < 200 or y[te].sum() < 15:
                    continue
                s1 = bss_of(XD[tr], y[tr])
                s2 = bss_of(XK[tr], y[tr])
                if not s1 or not s2:
                    continue
                e1 = bss_of(XD[te], y[te], s1["a"], s1["b"])
                e2 = bss_of(XK[te], y[te], s2["a"], s2["b"])
                if not e1 or not e2:
                    continue
                dbs.append(e2["bss"] - e1["bss"])
                nbs += 1
            if dbs:
                out.setdefault("holdout", {})["%s|%s" % (bname, sname)] = \
                    dict(n_split=nbs, dbss_median=float(np.median(dbs)),
                         dbss_all=[float(v) for v in dbs])
                print("%-16s %-10s %6d | %-18s | %-18s | %+9.5f"
                      % (bname, sname, nbs, "—", "—", float(np.median(dbs))))

    # ---------------- (2) 配对 bootstrap：ΔAUC / ΔBSS 的置信区间 ----------------
    print("\n" + "=" * 100)
    print("(2) 配对 bootstrap（同一重抽样上重算两个诊断）")
    print("=" * 100)
    print("%-16s %-6s %6s | %-24s | %-24s"
          % ("高度层", "子集", "n", "ΔAUC(DK−dry) [95%CI]",
             "ΔBSS(DK−dry) [95%CI]"))
    for bname, lo, hi in BANDS:
        for sname, extra in (("全部", np.ones(good.shape, bool)),
                             ("晴空", clear)):
            m = good & (p > lo) & (p <= hi) & extra
            if m.sum() < 400 or y[m].sum() < 30:
                continue
            x1, x2, yy = XD[m], XK[m], y[m]
            # ⚠ 注意别写成 bss_of(a,b)["auc"] if bss_of(a,b) else None ——
            #   那会把逻辑回归拟合**跑两遍**。
            def _f(k):
                def g(a, b):
                    r = bss_of(a, b)
                    return r[k] if r else None
                return g
            da = pair_boot_stat(x2, x1, yy, _f("auc"))
            db = pair_boot_stat(x2, x1, yy, _f("bss"))
            ci = lambda v: (float(np.percentile(v, 2.5)),
                            float(np.percentile(v, 97.5))) if len(v) > 20 \
                else (float('nan'), float('nan'))
            ca, cb = ci(da), ci(db)
            out.setdefault("boot", {})["%s|%s" % (bname, sname)] = \
                dict(n=int(m.sum()), dAUCr=float(np.median(da)),
                     dAUC_ci=list(ca), dBSS=float(np.median(db)),
                     dBSS_ci=list(cb))
            print("%-16s %-6s %6d | %+7.4f [%+.4f,%+.4f] | %+7.5f [%+.5f,%+.5f]"
                  % (bname, sname, int(m.sum()), float(np.median(da)),
                     ca[0], ca[1], float(np.median(db)), cb[0], cb[1]))

    # ---------------- (3) 检验功效 ----------------
    print("\n" + "=" * 100)
    print("(3) 检验功效：在当前样本量下能探测到多小的 ΔAUC？")
    print("=" * 100)
    print("（把观测到的 ΔAUC 与最小可探测值并列：若 |ΔAUC| > MDE，结论才站得住。）")
    for bname, lo, hi in BANDS:
        m = good & (p > lo) & (p <= hi) & clear
        if m.sum() < 400:
            continue
        x1, x2, yy = XD[m], XK[m], y[m]
        npos, nneg = int(yy.sum()), int((1 - yy).sum())
        # 配对 bootstrap 的差值标准差 → MDE(80% 功效, 双侧 α=0.05)
        def _g(a, b):
            r = bss_of(a, b)
            return r["auc"] if r else None
        db = pair_boot_stat(x2, x1, yy, _g, nboot=400, seed=99)
        sd = float(np.std(db)) if len(db) > 20 else float('nan')
        mde = 2.80 * sd                      # (z_{0.975}+z_{0.80})·sd
        obs = float(np.median(db))
        out.setdefault("power", {})[bname] = dict(
            n=int(m.sum()), n_pos=npos, n_neg=nneg, sd_delta=sd,
            mde80=mde, observed=obs,
            detectable=bool(abs(obs) > mde) if np.isfinite(mde) else None)
        print("  %-16s n=%5d  正例 %4d/%4d  sd(ΔAUC)=%.4f  "
              "MDE(80%%)=%.4f  实测 %+.4f  → %s"
              % (bname, int(m.sum()), npos, nneg, sd, mde, obs,
                 "可探测" if abs(obs) > mde else "**样本不足，不能下结论**"))

    with open("data/step17_holdout.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step17_holdout.json")


if __name__ == "__main__":
    main()
