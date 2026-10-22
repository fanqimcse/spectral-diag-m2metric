# -*- coding: utf-8 -*-
"""step11b —— 用客观机载 EDR 复验 step08 的 H1 / H2。

step08 在 3706 份 PIREP 上得到：
    H1（虚温修正提升判别技巧） 被**否证**   ΔAUC = -0.0024 [-0.0035,-0.0013]
    H2（饱和式在晴空是物理误用）被**强支持** ΔAUC = -0.1241 [-0.1449,-0.1052]
但 PIREP 是飞行员主观报告，存在三条已知偏倚（航线选择性取样 / nil-report 缺失 /
强度分级主观），审稿人有充分理由要求用**客观湍流观测**复核。

本脚本用 AMDAR 的 EDR（机上惯导反演、机型无关、连续量）做同一件事，并且：

  (1) 标签是**连续量** ⟹ 除二分类 AUC 外可直接用 Spearman 秩相关，
      完全不需要人为阈值 —— PIREP 做不到这一点；
  (2) 同时给出**多阈值 AUC 曲线**（0.02/0.05/0.10/0.15），展示结论对阈值的稳健性；
  (3) 背景场用**原生 21 层**（step08 用的是业务 11 层），与 step06 的 δ 结论同源；
  (4) 每天 ≥2 个 GFS 循环 ⟹ 时间错配 ≤ 1.5 h，优于 step08 的 ±3 h。

一处 simplification：AMDAR 报告自带气压高度，已在采集阶段换算成气压，
因此诊断量可直接**在气压上插值到观测层**，省掉 step08 里"高度→气压"那次
额外插值带来的误差。
"""
import datetime as dt
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
from avwx import madis_amdar as MA
from avwx import risk as RK
from avwx import verify as VF

WIN_H = 1.5             # 时间匹配窗口（小时）
RH_CLEAR = 80.0         # 干/晴空判据，与 step08 完全一致
EDR_THR = (0.02, 0.05, 0.10, 0.15)
REGIONS = {
    "NA": (235.0, 295.0, 25.0, 55.0),
    "NATL": (295.0, 350.0, 25.0, 60.0),
    "EUR": (350.0, 30.0, 35.0, 65.0),
    "EA": (100.0, 145.0, 15.0, 45.0),
}
GFS_GLOB = "data/aws/gfs_*_f000_L21_1000_100.grib2"
EDR_DIR = "data/madis/edr"
PKL = "data/madis/step11b_matched.pkl"


# ------------------------------------------------------------------ 观测
def load_obs():
    paths = sorted(glob.glob(os.path.join(EDR_DIR, "*.npz")))
    if not paths:
        raise SystemExit("无 EDR 数据：先跑 step11a_fetch_edr.py")
    bags = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        bags.append({k: z[k] for k in z.files})
    keys = bags[0].keys()
    out = {}
    for k in keys:
        if k == "n_all":
            continue
        out[k] = np.concatenate([b[k] for b in bags])
    # 只要"EDR 有效"的：有无标签这件事本身由机载算法决定，不由人决定。
    m = np.isfinite(out["medEDR"]) & np.isfinite(out["T"]) & \
        np.isfinite(out["wspd"]) & np.isfinite(out["p"])
    out = {k: v[m] for k, v in out.items()}
    return out, len(paths)


def find_gfs_times():
    """扫描本地 GFS f000 分析，返回 [(datetime, path)]。"""
    rx = re.compile(r"gfs_(\d{8})_(\d{2})_f000")
    res = []
    for p in sorted(glob.glob(GFS_GLOB)):
        m = rx.search(os.path.basename(p))
        if m:
            res.append((dt.datetime.strptime(m.group(1) + m.group(2),
                                             "%Y%m%d%H"), p))
    return sorted(res)


# ------------------------------------------------------------------ 主流程
def build_matched():
    obs, nday = load_obs()
    print("载入 EDR 数据集：%d 天，EDR 有效样本 %d 条" % (nday, obs["time"].size),
          flush=True)
    gtimes = find_gfs_times()
    print("可用 GFS f000 分析 %d 个时次" % len(gtimes), flush=True)
    rows = []
    done = set()
    if os.path.exists(PKL):                       # 断点续算：每个时次即时落盘
        with open(PKL, "rb") as fh:
            rows = pickle.load(fh)
        done = set(r["time"] for r in rows)
        print("断点续算：已有 %d 个时次" % len(done), flush=True)
    for ct, path in gtimes:
        if ct.strftime("%Y%m%d_%Hz") in done:
            continue
        tk = ct.timestamp()
        dsec = np.abs(obs["time"] - tk)
        mt = dsec <= WIN_H * 3600.0
        if int(mt.sum()) < 20:
            continue
        # ⚠ 踩坑：hit 必须在**每个时次**开头清零。它在该时次的区域循环里累加，
        #   用于报告"该时次匹配 N 条"。此前忘了初始化——因为一直有断点续算的
        #   缓存（PKL 存在 ⇒ 走 `continue` 从不到这行），bug 潜伏了整个项目；
        #   一旦删掉缓存强制重算就 UnboundLocalError。
        hit = 0
        new_rows = []
        for rname, bb in REGIONS.items():
            mb = MA.in_bbox(obs["lat"], obs["lon"], bb)
            m = mb & mt
            if int(m.sum()) < 20:
                continue
            idx = np.where(m)[0]
            qlat = obs["lat"][idx].astype(float)
            qlon = obs["lon"][idx].astype(float)
            try:
                d = AG.load(path, bbox=bb)
            except Exception as e:
                print("   %s %s 加载失败: %s" % (rname, path, str(e)[:70]),
                      flush=True)
                continue
            lo = np.asarray(d["lon"], float)
            qlon_q = np.where(lo.max() <= 180.0, np.where(qlon > 180, qlon - 360,
                                                          qlon), qlon)
            diag = RK.stability_fields(dict(T=d["T"], q=d["q"], u=d["u"],
                                            v=d["v"], z=d["z"]),
                                       d["lat"], d["lon"], d["level"])
            ptar = obs["p"][idx].astype(float) / 100.0        # Pa -> hPa
            rec = dict(time=ct.strftime("%Y%m%d_%Hz"), region=rname,
                       lat=qlat, lon=qlon_q, p=ptar, alt=obs["alt"][idx],
                       medEDR=obs["medEDR"][idx], maxEDR=obs["maxEDR"][idx],
                       To=obs["T"][idx], q=obs["q"][idx], dsec=dsec[idx] / 60.0)
            for k in ("S2", "Nd2", "Nv2", "Ndk2", "rh", "dqdz",
                      "delta", "delta_sat"):
                if k in diag:
                    rec[k] = VF.vert_interp(
                        VF.bilinear_cols(np.asarray(diag[k], float),
                                         d["lat"], lo, qlat, qlon_q),
                        d["level"], ptar)
            new_rows.append(rec)
            hit += int(idx.size)
            del diag, d
            gc.collect()
        rows.extend(new_rows)
        with open(PKL, "wb") as fh:
            pickle.dump(rows, fh)
        print("  %s  匹配 %6d 条  → 已落盘（累计 %d 组）"
              % (ct.strftime("%Y%m%d_%Hz"), hit, len(rows)), flush=True)
        del new_rows
        gc.collect()
    with open("data/madis/step11b_matched.pkl", "wb") as fh:
        pickle.dump(rows, fh)
    tot = int(sum(len(r["lat"]) for r in rows))
    print("\n累计匹配 %d 条 → data/madis/step11b_matched.pkl" % tot, flush=True)
    return rows


def stack(rows, k):
    return np.concatenate([r[k] for r in rows])


def season_of(time_str):
    mm = int(time_str[4:6])
    return {1: "冬", 7: "夏", 10: "秋"}.get(mm, "?")


def analyse(rows):
    med_edr = stack(rows, "medEDR")
    max_edr = stack(rows, "maxEDR")
    p = stack(rows, "p")
    rh = stack(rows, "rh")
    S2 = stack(rows, "S2")
    Nd2, Nv2, Ndk2 = stack(rows, "Nd2"), stack(rows, "Nv2"), stack(rows, "Ndk2")
    reg = np.concatenate([np.full(len(r["lat"]), r["region"]) for r in rows])
    sea = np.concatenate([np.full(len(r["lat"]), season_of(r["time"]))
                          for r in rows])
    dsec = stack(rows, "dsec")

    # 观测 EDR 为 NaN（无 EDR 报告的机队）已在 load 阶段剔除；
    # 这里把 EDR 缺失的残余 NaN 置 0 之前必须先确认那确实是"平滑"而非"未报"。
    med = np.where(np.isfinite(med_edr), med_edr, 0.0)
    mx = np.where(np.isfinite(max_edr), max_edr, 0.0)

    def ri(n2):
        return n2 / np.maximum(S2, 1e-12)

    rid, riv, rik = ri(Nd2), ri(Nv2), ri(Ndk2)
    sc = lambda x: -np.log10(np.maximum(np.asarray(x, float), 1e-6))
    SD, SV, SK = sc(rid), sc(riv), sc(rik)

    good = np.isfinite(SD) & np.isfinite(SV) & np.isfinite(SK) & \
        np.isfinite(rh) & np.isfinite(p)
    print("可用样本 %d / %d（诊断量有限）" % (int(good.sum()), len(p)))

    SUBSETS = [
        ("全部", np.ones(len(p), bool)),
        ("晴空 RH<80%", rh < RH_CLEAR),
        ("晴空+高层 p<=400hPa", (rh < RH_CLEAR) & (p <= 400)),
        ("晴空+高层+层结稳定", (rh < RH_CLEAR) & (p <= 400) & (Nd2 > 0)),
        ("北美", reg == "NA"),
        ("冬季", sea == "冬"),
        ("夏季", sea == "夏"),
        ("秋季", sea == "秋"),
    ]

    out = dict(meta=dict(win_h=WIN_H, rh_clear=RH_CLEAR, edr_thr=list(EDR_THR),
                         n_times=len(set(r["time"] for r in rows))),
               subsets={})

    # ---------- 表 I：连续标签的秩相关（不需要阈值） ----------
    print("\n" + "=" * 96)
    print("【表 I】连续 EDR 与三种 Ri 的 Spearman 秩相关（越高越好；无阈值）")
    print("=" * 96)
    print("%-22s %8s %9s %9s %9s %11s %11s"
          % ("子集", "n", "ρ_dry", "ρ_virt", "ρ_DK", "Δρ_virt", "Δρ_DK"))
    for nm, m in SUBSETS:
        m = m & good
        if int(m.sum()) < 300:
            print("%-22s %8d   (样本不足)" % (nm, int(m.sum())))
            continue
        y = med[m]
        rD = VF.spearman(SD[m], y)
        rV = VF.spearman(SV[m], y)
        rK = VF.spearman(SK[m], y)
        bV = VF.bootstrap_stat(SV[m], SD[m], y, n=600)
        bK = VF.bootstrap_stat(SK[m], SD[m], y, n=600)
        cV, cK = VF.ci_of(bV), VF.ci_of(bK)
        ent = dict(n=int(m.sum()), rho_dry=rD, rho_virt=rV, rho_dk=rK,
                   drho_virt=rV - rD, drho_virt_ci=cV,
                   drho_dk=rK - rD, drho_dk_ci=cK)
        out["subsets"][nm] = ent
        print("%-22s %8d %9.4f %9.4f %9.4f  %+6.4f[%+6.4f,%+6.4f] %+6.4f[%+6.4f,%+6.4f]"
              % (nm, int(m.sum()), rD, rV, rK,
                 rV - rD, cV[0], cV[1], rK - rD, cK[0], cK[1]))

    # ---------- 表 II：多阈值 AUC ----------
    for tag, ylab in (("medEDR", med), ("maxEDR", mx)):
        print("\n" + "=" * 96)
        print("【表 II-%s】多阈值 AUC（标签 = %s ≥ t）" % (tag, tag))
        print("=" * 96)
        print("%-22s %7s %6s %8s %8s %8s %8s %8s"
              % ("子集", "n", "t", "AUC_dry", "AUC_virt", "AUC_DK",
                 "Δvirt", "ΔDK"))
        slot = out["subsets"]
        for nm, m in SUBSETS:
            m = m & good
            if int(m.sum()) < 300:
                continue
            tab = slot[nm].setdefault("auc_" + tag, {})
            for t in EDR_THR:
                y = (ylab[m] >= t).astype(int)
                if y.sum() < 20 or (1 - y).sum() < 20:
                    continue
                aD = VF.auc_score(SD[m], y)
                aV = VF.auc_score(SV[m], y)
                aK = VF.auc_score(SK[m], y)
                bV = VF.bootstrap_dauc(SV[m], SD[m], y, n=800)
                bK = VF.bootstrap_dauc(SK[m], SD[m], y, n=800)
                cV, cK = VF.ci_of(bV), VF.ci_of(bK)
                tab[str(t)] = dict(n=int(m.sum()), pos=int(y.sum()),
                                   pos_rate=float(y.mean()),
                                   auc_dry=aD, auc_virt=aV, auc_dk=aK,
                                   d_auc_virt=aV - aD, d_auc_virt_ci=cV,
                                   d_auc_dk=aK - aD, d_auc_dk_ci=cK)
                print("%-22s %7d %6.2f %8.4f %8.4f %8.4f %+7.4f %+7.4f"
                      % (nm, int(m.sum()), t, aD, aV, aK, aV - aD, aK - aD))

    # ---------- 与 PIREP 并列 ----------
    pp = "data/pirep/step08_pirep_verification.json"
    if os.path.exists(pp):
        with open(pp, encoding="utf8") as f:
            pjson = json.load(f)
        out["pirep_reference"] = {k: dict(auc_dry=v["auc_dry"],
                                          auc_virt=v["auc_virt"],
                                          auc_dk=v["auc_dk"],
                                          d_auc_virt=v["d_auc_virt"],
                                          d_auc_dk=v["d_auc_dk"],
                                          n=v["n"])
                                  for k, v in pjson["subsets"].items()}
    return out


def main():
    t0 = time.time()
    pkl = PKL
    if os.path.exists(pkl):
        with open(pkl, "rb") as fh:
            rows = pickle.load(fh)
        print("复用已有匹配 %d 组" % len(rows))
    else:
        rows = build_matched()
    res = analyse(rows)
    with open("data/madis/step11b_edr_verification.json", "w",
              encoding="utf8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/madis/step11b_edr_verification.json   用时 %.0f s"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
