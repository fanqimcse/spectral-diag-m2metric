# -*- coding: utf-8 -*-
"""step18 —— 把"业务层组失明"从抽象的 δ 换算成**湍流预报技巧**。

为什么要做这一步
----------------
前面对"业务层组失明"的量化一直停留在**场本身的量**上：
    δ = (N²_dry − N²_v) / N²_dry    —— 湿度对静力稳定度的相对削弱
以及"L11 相对 L21 把 δ 低估 1.2–3.9 倍"（step06）。
但 δ 是一个**中间量**：它变小 3.9 倍，究竟让颠簸预报损失多少技巧？
这个问题此前没有人回答，而审稿人一定会问（"所以呢"）。

本步把它换算成决策货币：用 step11 建成的 10 278 条**客观 EDR** 报告，
比较同一个诊断量在 **L21 原生层组** 与 **L11 业务层组** 上对
`y = 1[maxEDR ≥ 0.10]` 的 AUC。ΔAUC(L11 − L21) 就是"业务垂直分辨率"
在湍流可预报性上的**净代价**，且以客观观测为标尺。

关键设计（为了可比性，必须逐条满足）
------------------------------------
1. **同一批样本、同一时刻、同一区域**：两组诊断都从**同一份** GFS f000
   场算出，只是垂直层组不同；样本集合完全一致 ⇒ 严格配对比较。
2. **晴空/云内与高度层的划分用 L21 的 RH 定义**，然后同时施加到两组，
   避免"两组的样本集不同"造成的伪差异。
3. **配对 bootstrap** 给 ΔAUC 的置信区间（EDR 是航段，点与点不独立）。
4. **不做**任何"再标定"：这里问的是**信息量**损失，不是标定损失。
   若 L11 只是"数值偏了"，重新标定能救；若 CI 仍不含 0，说明是**结构性**损失。

一个必须说明的口径
------------------
L11 的垂直导数是从 11 层差分算的，L21 从 21 层算。这**同时**包含
"真实层结被平滑"和"差分步长变大"两个效应 —— 它们在实际业务里是
**不可分离**的（业务系统只有 11 层，没有更密的真值可差分）。
所以这个 ΔAUC 是"业务层组的**总**代价"，而不是其中某一部分的代价。
"""
import gc
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from avwx import aws_gfs as AG
from avwx import risk as RK
from avwx import verify as VF

PKL = "data/madis/step11b_matched.pkl"
OUT = "data/step18_ops_levelset_skill.json"
GFS_GLOB = "data/aws/gfs_*_f000_L21_1000_100.grib2"

# 与 step06 完全一致（含那次"廓线上下颠倒"的踩坑守卫）
LEVEL_SETS = {
    "L21": [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
            550, 500, 450, 400, 350, 300, 250, 200, 150, 100],
    "L11": [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100],
    "L07": [1000, 850, 700, 500, 300, 200, 100],
}

REGIONS = {
    "NA": (235.0, 295.0, 25.0, 55.0),
    "NATL": (295.0, 350.0, 25.0, 60.0),
    "EUR": (350.0, 30.0, 35.0, 65.0),
    "EA": (100.0, 145.0, 15.0, 45.0),
}

RH_CLEAR = 80.0
EDR_THR = 0.10
N2_FLOOR = 1e-5

BANDS = (("p ≤ 300", 0.0, 300.0), ("300–500", 300.0, 500.0),
         ("500–700", 500.0, 700.0), ("p > 700", 700.0, 1e9))


def level_index(lev, want):
    """want 各层在 lev 中的索引，保证 lev[idx] **降序**（气压随索引递减）。

    ⚠ 踩坑记录（从 step06 原样抄来，务必保留此断言）：
      曾写成 sorted(set(idx))[::-1]。lev 本身降序 1000→100，索引升序
      已对应降序气压；再 [::-1] 会把整条廓线上下颠倒，所有垂直导数反号
      而**不报错** —— 实测把 δ 从 +26% 变成 −0.6%，隐蔽性极强。
    """
    lev = np.asarray(lev, float)
    idx = []
    for w in want:
        k = int(np.argmin(np.abs(lev - w)))
        if abs(lev[k] - w) > 0.51 * min(
                abs(np.diff(lev).min()), abs(np.diff(want).min()) if len(want) > 1
                else 1e9):
            # 该层在 lev 中不存在足够近的代表层 → 跳过（宁缺勿错）
            continue
        idx.append(k)
    idx = sorted(set(idx))
    assert np.all(np.diff(lev[idx]) < 0), "层组必须严格降序（气压递减）"
    return np.asarray(idx, int)


def find_gfs_times():
    rx = re.compile(r"gfs_(\d{8})_(\d{2})_f000")
    out = {}
    for p in sorted(glob.glob(GFS_GLOB)):
        m = rx.search(os.path.basename(p))
        if m:
            out["%s_%sz" % (m.group(1), m.group(2))] = p
    return out


def diag_at(d, want_levels, qlat, qlon, ptar):
    """在指定层组上诊断，并插值到样本位置。"""
    idx = level_index(d["level"], want_levels)
    lev = np.asarray(d["level"], float)[idx]
    f = {v: np.asarray(d[v], float)[idx] for v in ("T", "q", "u", "v", "z")}
    dg = RK.stability_fields(f, d["lat"], d["lon"], lev)
    res = {}
    for k in ("S2", "Nd2", "Nv2", "Ndk2", "rh", "delta", "delta_sat"):
        if k not in dg:
            continue
        res[k] = VF.vert_interp(
            VF.bilinear_cols(np.asarray(dg[k], float), d["lat"],
                             np.asarray(d["lon"], float), qlat, qlon),
            lev, ptar)
    return res


def build():
    rows = pickle.load(open(PKL, "rb"))
    gtimes = find_gfs_times()
    print("EDR 组 %d 个；本地 GFS f000 时次 %d 个" % (len(rows), len(gtimes)),
          flush=True)
    out = []
    for i, r in enumerate(rows):
        tk = r["time"]
        if tk not in gtimes:
            print("  %s %-4s 缺 GFS，跳过" % (tk, r["region"]), flush=True)
            continue
        bb = REGIONS[r["region"]]
        try:
            d = AG.load(gtimes[tk], bbox=bb)
        except Exception as e:
            print("  %s %-4s 载入失败: %s" % (tk, r["region"], str(e)[:70]),
                  flush=True)
            continue
        lo = np.asarray(d["lon"], float)
        qlat = np.asarray(r["lat"], float)
        qlon = np.asarray(r["lon"], float)
        qlon = np.where(lo.max() <= 180.0,
                        np.where(qlon > 180, qlon - 360, qlon), qlon)
        ptar = np.asarray(r["p"], float)
        rec = dict(time=tk, region=r["region"], lat=qlat, lon=qlon, p=ptar,
                   maxEDR=np.asarray(r["maxEDR"], float))
        for tag, lv in LEVEL_SETS.items():
            dd = diag_at(d, lv, qlat, qlon, ptar)
            for k, v in dd.items():
                rec["%s_%s" % (tag, k)] = v
        out.append(rec)
        print("  [%2d/%d] %s %-4s n=%d" % (i + 1, len(rows), tk, r["region"],
                                           len(ptar)), flush=True)
        del d
        gc.collect()
    return out


def stack(rows, k):
    return np.concatenate([r[k] for r in rows])


def score(n2, s2):
    return -np.log10(np.maximum(n2 / np.maximum(s2, 1e-12), 1e-6))


def analyse(recs):
    p = stack(recs, "p")
    y = (stack(recs, "maxEDR") >= EDR_THR).astype(int)
    rh21 = stack(recs, "L21_rh")
    S2_21, S2_11 = stack(recs, "L21_S2"), stack(recs, "L11_S2")

    out = dict(bands={}, delta={})
    good = (np.isfinite(p) & np.isfinite(rh21) & np.isfinite(S2_21)
            & np.isfinite(S2_11))

    print("\n" + "=" * 96)
    print("业务层组（L11）相对原生层组（L21）的湍流技巧代价   "
          "y = 1[maxEDR ≥ %.2f]" % EDR_THR)
    print("=" * 96)
    print("%-10s %-6s %6s %6s | %-24s | %-24s | %-24s"
          % ("高度层", "子集", "n", "正例", "dry  A21→A11  Δ[95%CI]",
             "virt A21→A11  Δ[95%CI]", "DK   A21→A11  Δ[95%CI]"))

    for bname, lo, hi in BANDS:
        for sname, smask in (("全部", np.ones(good.size, bool)),
                             ("晴空", rh21 < RH_CLEAR)):
            m = good & (p > lo) & (p <= hi) & smask
            if m.sum() < 300:
                print("%-10s %-6s %6d   样本不足" % (bname, sname, int(m.sum())))
                continue
            yy = y[m]
            if not (0 < yy.mean() < 1):
                print("%-10s %-6s %6d   正例缺失" % (bname, sname, int(m.sum())))
                continue
            rec = {}
            cells = []
            for k, n2k in (("dry", "Nd2"), ("virt", "Nv2"), ("DK", "Ndk2")):
                a21 = score(stack(recs, "L21_%s" % n2k)[m], S2_21[m])
                a11 = score(stack(recs, "L11_%s" % n2k)[m], S2_11[m])
                A21 = VF.auc_score(a21, yy)
                A11 = VF.auc_score(a11, yy)
                bs = VF.bootstrap_dauc(a11, a21, yy, n=1500)
                d = A11 - A21
                lo_ci, hi_ci = np.nanpercentile(bs, [2.5, 97.5])
                rec[k] = dict(auc21=float(A21), auc11=float(A11),
                              d=float(d), ci=[float(lo_ci), float(hi_ci)])
                cells.append("%.3f→%.3f %+.4f [%.4f,%.4f]"
                             % (A21, A11, d, lo_ci, hi_ci))
            out["bands"]["%s|%s" % (bname, sname)] = dict(
                n=int(m.sum()), pos=int(yy.sum()), clim=float(yy.mean()), **rec)
            print("%-10s %-6s %6d %6d | %-24s | %-24s | %-24s"
                  % (bname, sname, int(m.sum()), int(yy.sum()), *cells))

    # ---------------- δ 的交叉校验（与 step06 口径对照） ----------------
    print("\n" + "-" * 96)
    print("δ 的层组依赖（交叉校验 step06；样本集与上表一致）")
    print("-" * 96)
    print("%-10s %8s %10s %10s %10s" % ("高度层", "n", "δ(L21)", "δ(L11)",
                                        "L11/L21"))
    d21 = stack(recs, "L21_delta")
    d11 = stack(recs, "L11_delta")
    okd = good & np.isfinite(d21) & np.isfinite(d11) & (rh21 < RH_CLEAR)
    for bname, lo, hi in BANDS:
        m = okd & (p > lo) & (p <= hi)
        if m.sum() < 100:
            continue
        v21, v11 = float(np.median(d21[m])), float(np.median(d11[m]))
        out["delta"][bname] = dict(n=int(m.sum()), d21=v21, d11=v11,
                                   ratio=(v11 / v21 if v21 > 0 else None))
        print("%-10s %8d %9.4f%% %9.4f%% %9.2f"
              % (bname, int(m.sum()), 100 * v21, 100 * v11,
                 (v11 / v21 if v21 > 0 else float("nan"))))
    return out


def main():
    t0 = time.time()
    cache = "data/step18_diag.pkl"
    if os.path.exists(cache):
        recs = pickle.load(open(cache, "rb"))
        print("复用缓存 %s（%d 组）" % (cache, len(recs)), flush=True)
    else:
        recs = build()
        with open(cache, "wb") as fh:
            pickle.dump(recs, fh)
    res = analyse(recs)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
