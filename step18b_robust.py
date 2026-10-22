# -*- coding: utf-8 -*-
"""step18b —— 两件事：δ 口径矛盾的定位，以及 ΔAUC 的留一稳健性。

(1) δ 的符号矛盾（必须解决，否则 step06 与 step18 不能同时成立）
--------------------------------------------------------------
step06（全域全层中位，共享层比较）：L11 的 δ **低于** L21，比值 2.1–2.9×。
step18（沿航迹分层中位，插值到样本气压）：L11 的 δ **高于** L21，比值 1.36/1.46×。

怀疑根因：δ 随气压**强烈凸增**（上对流层 0.08% → 700 hPa 以下 8.3%）。
step18 的做法是"在各自层组上算 δ，再**垂直插值**到样本气压"。
对一个凸函数做线性插值必然**高估**，而且层组越粗（Δz 越大）高估越多。
于是粗层组的 δ 被系统性抬升 —— 这正是 1.36/1.46 的来源。

检验方法：把样本**限制在 L11 的共享层附近**（|p − p_L11| ≤ 20 hPa），
此时两组都在"自己的层上"求值，不存在跨层插值，可比。
若此时比值回到 <1，则确认是插值假象，step18 的 δ 表必须撤下，
只保留 AUC 表（AUC 那部分两组用**完全相同**的插值流程，仍然可比）。

(2) ΔAUC 的留一稳健性
--------------------
500–700 hPa 晴空的 ΔAUC = −0.1148，CI 下界 −0.2138、上界 −0.0252，
**刚刚**排除 0。这种"擦边"结果必须做留一检验：
  - 留一区域（NA / NATL / EUR / EA）
  - 留一时次（9 个）
看中位 ΔAUC 是否稳定为负、以及是否由单个时次/区域支配。
"""
import json
import pickle
import sys

import numpy as np

sys.path.insert(0, ".")
from step18_ops_levelset_skill import (LEVEL_SETS, RH_CLEAR, EDR_THR,
                                       analyse, stack, score)
from avwx import verify as VF

CACHE = "data/step18_diag.pkl"
OUT = "data/step18b_robust.json"
L11 = LEVEL_SETS["L11"]


def main():
    recs = pickle.load(open(CACHE, "rb"))
    p = stack(recs, "p")
    y = (stack(recs, "maxEDR") >= EDR_THR).astype(int)
    rh21 = stack(recs, "L21_rh")
    S2_21 = stack(recs, "L21_S2")
    S2_11 = stack(recs, "L11_S2")
    d21 = stack(recs, "L21_delta")
    d11 = stack(recs, "L11_delta")
    reg = np.concatenate([np.full(len(r["lat"]), r["region"]) for r in recs])
    tim = np.concatenate([np.full(len(r["lat"]), r["time"]) for r in recs])

    res = {}

    # ---------------- (1) δ：共享层 vs 插值 ----------------
    print("=" * 96)
    print("(1) δ 的层组比值：插值口径 vs 共享层口径")
    print("=" * 96)
    ok = (np.isfinite(p) & np.isfinite(rh21) & np.isfinite(d21)
          & np.isfinite(d11) & (rh21 < RH_CLEAR))
    near = np.zeros(p.size, bool)
    for lv in L11:
        near |= np.abs(p - lv) <= 20.0
    print("%-10s %8s %10s %10s %8s | %8s %10s %10s %8s"
          % ("高度层", "n(插值)", "δ21", "δ11", "11/21", "n(共享)",
             "δ21", "δ11", "11/21"))
    res["delta"] = {}
    BANDS = (("p ≤ 300", 0.0, 300.0), ("300–500", 300.0, 500.0),
             ("500–700", 500.0, 700.0), ("p > 700", 700.0, 1e9))
    for bname, lo, hi in BANDS:
        row = {}
        cells = []
        for tag, m0 in (("interp", ok), ("shared", ok & near)):
            m = m0 & (p > lo) & (p <= hi)
            if m.sum() < 60:
                cells += ["—", "—", "—", "—"]
                row[tag] = dict(n=int(m.sum()))
                continue
            a, b = float(np.median(d21[m])), float(np.median(d11[m]))
            row[tag] = dict(n=int(m.sum()), d21=a, d11=b,
                            ratio=(b / a if a > 0 else None))
            cells += ["%d" % int(m.sum()), "%.4f%%" % (100 * a),
                      "%.4f%%" % (100 * b), "%.2f" % (b / a if a > 0
                                                      else float("nan"))]
        res["delta"][bname] = row
        print("%-10s %8s %10s %10s %8s | %8s %10s %10s %8s"
              % (bname, *cells))

    # ---------------- (2) ΔAUC 留一稳健性 ----------------
    print("\n" + "=" * 96)
    print("(2) ΔAUC(L11 − L21) 的留一稳健性")
    print("=" * 96)
    good = (np.isfinite(p) & np.isfinite(rh21) & np.isfinite(S2_21)
            & np.isfinite(S2_11))
    res["loo"] = {}
    for bname, lo, hi in BANDS:
        for sname, smask in (("全部", np.ones(p.size, bool)),
                             ("晴空", rh21 < RH_CLEAR)):
            m = good & (p > lo) & (p <= hi) & smask
            if m.sum() < 300:
                continue
            yy = y[m]
            if not (0 < yy.mean() < 1) or yy.sum() < 20:
                continue
            a21 = score(stack(recs, "L21_Nd2")[m], S2_21[m])
            a11 = score(stack(recs, "L11_Nd2")[m], S2_11[m])
            full = VF.auc_score(a11, yy) - VF.auc_score(a21, yy)
            rr, tt = reg[m], tim[m]
            loo_r, loo_t = [], []
            for g in sorted(set(rr)):
                k = rr != g
                if y[m][k].sum() < 10:
                    continue
                loo_r.append(float(VF.auc_score(a11[k], yy[k])
                                   - VF.auc_score(a21[k], yy[k])))
            for g in sorted(set(tt)):
                k = tt != g
                if y[m][k].sum() < 10:
                    continue
                loo_t.append(float(VF.auc_score(a11[k], yy[k])
                                   - VF.auc_score(a21[k], yy[k])))
            res["loo"]["%s|%s" % (bname, sname)] = dict(
                n=int(m.sum()), pos=int(yy.sum()), full=float(full),
                loo_region=loo_r, loo_time=loo_t,
                loo_region_med=(float(np.median(loo_r)) if loo_r else None),
                loo_time_med=(float(np.median(loo_t)) if loo_t else None),
                loo_time_max=(float(np.max(loo_t)) if loo_t else None))
            print("%-10s %-6s n=%5d pos=%4d  全样本 %+.4f | "
                  "留一区域 中位 %+.4f [%d 折] | 留一时次 中位 %+.4f "
                  "最差 %+.4f [%d 折]"
                  % (bname, sname, int(m.sum()), int(yy.sum()), full,
                     float(np.median(loo_r)) if loo_r else float("nan"),
                     len(loo_r),
                     float(np.median(loo_t)) if loo_t else float("nan"),
                     float(np.max(loo_t)) if loo_t else float("nan"),
                     len(loo_t)))

    with open(OUT, "w", encoding="utf8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n已写出 " + OUT)


if __name__ == "__main__":
    main()
