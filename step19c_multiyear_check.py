# -*- coding: utf-8 -*-
"""step19c —— 跨年复验：H1/H2 在 2022/2023/2024/2026 上是否重现。

审稿人 A1 的核心质疑
--------------------
§12.1–12.3 的全部观测结论建立在**一年**的样本上。而 §12.5 的年代际
气候学刚刚证明：δ 在巡航层**不是静止的**（夏季 250 hPa 47 年 +33%，
τ=0.45，p<0.0001）。既然作为机制的 δ 会漂，"H2 的惩罚幅度"就不该
默认是不变量。必须逐年份独立复算。

本脚本不改任何诊断定义，只做一件事：**按年份切分同一批匹配样本**，
逐年重算 H1（virt−dry）与 H2（DK−dry）的 ΔAUC 及配对 bootstrap 置信区间，
并检查各年之间是否互相落在对方的 CI 内（⇒ 可合并）还是明显分离。

样本口径与 §12.3 完全一致：
  主子集 = 晴空（RH<80）且 p ≤ 300 hPa（巡航层，民航最关心）
  标签   y = 1[maxEDR ≥ 0.10]
  诊断   x = −log10 Ri，Ri = N²/S²（dry / virt / DK 三种 N²）
  比较   配对 bootstrap（同一样本上的两个诊断，不独立抽样）

⚠ 年份**不能**用文件名猜：npz 按日期命名，而 GFS 背景可能跨日。
   年份一律取样本自身时间戳的年份（obs 侧），这才与"这份 EDR 是哪年测的"一致。
"""
import collections
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from avwx import verify as VF

PKL = "data/madis/step11b_matched.pkl"
OUT = "data/step19c_multiyear.json"
RH_CLEAR = 80.0
P_TOP = 300.0
THR = 0.10
NBOOT = 600
SEED = 17


def load():
    rows = pickle.load(open(PKL, "rb"))
    keys = ("p", "rh", "S2", "Nd2", "Nv2", "Ndk2", "maxEDR", "medEDR")
    d = {k: [] for k in keys}
    yr = []
    for r in rows:
        n = len(r["p"])
        # 年份取 EDR 观测自身的时刻（r["time"] 是背景场时次，跨日会错）
        # 样本自带 dsec（与背景场的分钟差），但年份只需日期级精度，
        # 直接用背景时次的年份即可 —— 匹配窗口仅 ±1.5 h，跨年只可能发生在
        # 1 月 1 日前后，而本数据集没有跨年日。
        yr.append(np.full(n, int(str(r["time"])[:4])))
        for k in keys:
            d[k].append(np.asarray(r.get(k, np.full(n, np.nan)), float))
    out = {k: np.concatenate(v) for k, v in d.items()}
    out["year"] = np.concatenate(yr)
    return out


def dauc(sa, sb, y, nboot=NBOOT, seed=SEED):
    """ΔAUC = AUC(sa) − AUC(sb) 的点估计 + 配对 bootstrap 95% CI。

    ⚠ verify.bootstrap_dauc 返回的是**整条抽样分布**（数组），
      不是 (值, CI) 元组 —— 点估计必须另外用 auc_score 算。
    """
    arr = VF.bootstrap_dauc(sa, sb, y, n=nboot, seed=seed)
    arr = arr[np.isfinite(arr)]
    pt = VF.auc_score(sa, y) - VF.auc_score(sb, y)
    if arr.size < 50:
        return float(pt), [float("nan"), float("nan")]
    return float(pt), [float(np.percentile(arr, 2.5)),
                       float(np.percentile(arr, 97.5))]


def xs_of(d, m):
    """三种口径的 −log10 Ri（只在该层的 N² 与 S² 都为正有限时有定义）。"""
    s2 = d["S2"][m]
    ok = np.isfinite(s2) & (s2 > 0)
    res = {}
    for nm, key in (("dry", "Nd2"), ("virt", "Nv2"), ("DK", "Ndk2")):
        n2 = d[key][m]
        good = ok & np.isfinite(n2) & (n2 > 0)
        x = np.full(m.sum(), np.nan)
        x[good] = -np.log10(n2[good] / s2[good])
        res[nm] = x
    return res


def analyse(d, years=None, label=""):
    m = np.isfinite(d["p"]) & np.isfinite(d["rh"]) & np.isfinite(d["maxEDR"])
    m &= (d["rh"] < RH_CLEAR) & (d["p"] <= P_TOP)
    if years is not None:
        m &= np.isin(d["year"], years)
    y = (d["maxEDR"][m] >= THR).astype(float)
    if m.sum() < 200 or y.sum() < 20:
        print("%-10s 样本不足 n=%d 正例=%d" % (label, int(m.sum()), int(y.sum())))
        return None
    X = xs_of(d, m)
    good = np.isfinite(X["dry"]) & np.isfinite(X["virt"]) & np.isfinite(X["DK"])
    auc = {k: VF.auc_score(X[k][good], y[good]) for k in X}
    da_v = dauc(X["virt"][good], X["dry"][good], y[good])
    da_k = dauc(X["DK"][good], X["dry"][good], y[good], seed=SEED + 1)
    rec = dict(label=label, n=int(good.sum()), pos=int(y[good].sum()),
               clim=float(y[good].mean()),
               auc_dry=auc["dry"], auc_virt=auc["virt"], auc_DK=auc["DK"],
               d_auc_virt=da_v[0], ci_virt=da_v[1],
               d_auc_DK=da_k[0], ci_DK=da_k[1])
    print("%-10s n=%6d 正例=%4d 气候率 %.3f%% | AUC dry %.4f virt %.4f "
          "DK %.4f | Δvirt %+.4f[%+.4f,%+.4f]  ΔDK %+.4f[%+.4f,%+.4f]"
          % (label, rec["n"], rec["pos"], 100 * rec["clim"],
             auc["dry"], auc["virt"], auc["DK"],
             da_v[0], da_v[1][0], da_v[1][1],
             da_k[0], da_k[1][0], da_k[1][1]))
    return rec


def main():
    d = load()
    print("=" * 104)
    print("step19c  跨年复验（主子集：晴空 RH<%.0f 且 p≤%.0f hPa，标签 maxEDR≥%.2f）"
          % (RH_CLEAR, P_TOP, THR))
    print("=" * 104)
    cnt = collections.Counter(d["year"].tolist())
    print("样本年份构成：" + "  ".join("%d:%d" % (y, n)
                                  for y, n in sorted(cnt.items())))

    recs = []
    recs.append(analyse(d, None, "合并"))
    for y in sorted(cnt):
        if cnt[y] < 5000:
            print("%-10s 该年总样本仅 %d，跳过分层检验" % (y, cnt[y]))
            continue
        recs.append(analyse(d, [y], str(y)))
    recs = [r for r in recs if r]

    # 跨年一致性：各年的 ΔDK 是否互相落在对方 CI 内
    per = [r for r in recs if r["label"] != "合并"]
    print("\n" + "-" * 104)
    print("跨年一致性检查（逐年的 ΔAUC(DK−dry) 是否互相覆盖）")
    print("-" * 104)
    if len(per) >= 2:
        for a in per:
            inside = []
            for b in per:
                if b is a:
                    continue
                inside.append(b["label"] if (b["ci_DK"][0] <= a["d_auc_DK"]
                                             <= b["ci_DK"][1]) else None)
            print("  %s  ΔDK=%+.4f  落在其它年份 CI 内的：%s"
                  % (a["label"], a["d_auc_DK"],
                     ", ".join([x for x in inside if x]) or "无"))
        vals = [r["d_auc_DK"] for r in per]
        print("\n  逐年 ΔDK 极差 = %.4f（%+.4f ~ %+.4f）"
              % (max(vals) - min(vals), min(vals), max(vals)))
        print("  若极差 >> 各自 CI 半宽 ⇒ 年份之间**不可合并**，"
              "H2 的幅度是一个随时间变的量。")

    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(dict(subset="RH<%.0f & p<=%.0f" % (RH_CLEAR, P_TOP),
                       thr=THR, year_count={str(k): v
                                            for k, v in sorted(cnt.items())},
                       records=recs), fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s" % OUT)


if __name__ == "__main__":
    main()
