# -*- coding: utf-8 -*-
"""
观测检验：用真实飞行员报告（PIREP）判定三种稳定度口径的判别能力。

这是把本项目从「模式气候学统计」升级为「观测检验」的关键一步。

样本
----
AWC 的 PIREP（2026-09-21 ~ 2026-10-21，30 天，北美为主）：
  湍流强度 tbInt1 ∈ {NEG, LGT, LGT-MOD, MOD, MOD-SEV, SEV, EXTM}
  → 阳性 = LGT 及以上（含组合）；阴性 = NEG（飞行员明确报告无颠簸）
  另有 tbType1 ∈ {CAT, CHOP, LLWS}，可单独取出晴空湍流子集。

判别对象
--------
同一份分析场、同一 S²，只换稳定度口径：
    N_d²   干位温（现行业务）              → Ri_dry
    N_v²   未饱和虚温（本项目主张）        → Ri_virt
    N_DK²  Durran–Klemp 饱和式（GTG「湿」诊断的事实标准）→ Ri_DK
以 score = -log10(Ri) 作判别量，AUC 越高表示越能区分颠簸/不颠簸。

核心假设
--------
H1  虚温口径的 AUC 高于干口径（湿度确实携带判别信息）
H2  晴空子集内，饱和式 DK 的 AUC **低于**干口径 ⟹ 业务若在晴空区套用
    GTG 的 Rimoist，不但无增益，反而损失技巧。这是「晴空区误用饱和式」
    最强的观测证据（而非仅理论论证）。

时间匹配
--------
PIREP 按最近 GFS 分析时次归集，默认窗口 ±3 h（主口径）。
"""
import datetime as dt
import gc
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from avwx import aws_gfs as AG
from avwx import risk as RK

WIN_H = 3.0                              # 主口径 ±3 h
BBOX = (230.0, 300.0, 12.0, 58.0)        # AWS 经度约定 0..360
RH_CLEAR = 80.0
POS_SET = ("LGT", "LGT-MOD", "MOD", "MOD-SEV", "SEV", "EXTM")
NEG_SET = ("NEG",)


# ------------------------------------------------------------------ 工具
def bilinear_cols(field, lat1d, lon1d, qlat, qlon):
    """把 (nz,ny,nx) 场双线性插到散点。返回 (npts, nz)。"""
    yi = np.clip(np.searchsorted(lat1d, qlat) - 1, 0, len(lat1d) - 2)
    xi = np.clip(np.searchsorted(lon1d, qlon) - 1, 0, len(lon1d) - 2)
    ty = (qlat - lat1d[yi]) / np.maximum(lat1d[yi + 1] - lat1d[yi], 1e-9)
    tx = (qlon - lon1d[xi]) / np.maximum(lon1d[xi + 1] - lon1d[xi], 1e-9)
    ty = np.clip(ty, 0, 1)
    tx = np.clip(tx, 0, 1)
    nz = field.shape[0]
    out = np.empty((len(qlat), nz))
    for k in range(nz):
        f0 = field[k]
        out[:, k] = ((1 - ty) * (1 - tx) * f0[yi, xi] +
                     (1 - ty) * tx * f0[yi, xi + 1] +
                     ty * (1 - tx) * f0[yi + 1, xi] +
                     ty * tx * f0[yi + 1, xi + 1])
    return out


def vert_interp(col, lev_desc, ptar):
    """按气压垂插。col: (npts,nz)，列顺序与 lev_desc（降序）一致。"""
    xp = lev_desc[::-1]                       # 升序
    out = np.empty(len(ptar))
    for i in range(len(ptar)):
        out[i] = np.interp(ptar[i], xp, col[i][::-1])
    return out


def _ranks(s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(ss):                        # 并列取平均秩
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def auc_score(score, label):
    """AUC = P(score_pos > score_neg) + 0.5 P(=)。"""
    m = np.isfinite(score) & np.isfinite(label)
    s = np.asarray(score)[m]
    y = np.asarray(label)[m].astype(int)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = _ranks(s)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def bootstrap_dauc(sa, sb, y, n=2000, seed=20261021):
    rng = np.random.default_rng(seed)
    m = np.isfinite(sa) & np.isfinite(sb) & np.isfinite(y)
    sa, sb, yy = np.asarray(sa)[m], np.asarray(sb)[m], np.asarray(y)[m].astype(int)
    N = len(yy)
    out = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, N, N)
        out[b] = auc_score(sa[idx], yy[idx]) - auc_score(sb[idx], yy[idx])
    return np.nan_to_num(out, nan=np.nan)


# ------------------------------------------------------------------ 主流程
def main():
    t0 = time.time()
    print("=" * 84)
    print("PIREP 观测检验：干 / 虚温 / Durran–Klemp 三口径判别能力对照")
    print("=" * 84)

    df = pd.read_json("data/pirep/pirep_30days.json")
    df["ot"] = pd.to_datetime(df["obsTime"], unit="s", utc=True)
    df["tb1"] = df["tbInt1"].fillna("")
    df["tbt"] = df["tbType1"].fillna("")
    df = df[df["tb1"].isin(list(POS_SET) + list(NEG_SET))].copy()
    df["pos"] = df["tb1"].isin(list(POS_SET)).astype(int)
    df = df[(df["fltLvl"].fillna(0) > 0) & df["lat"].notna() & df["lon"].notna()]
    print("可用于判别的 PIREP：%d 条（阳性 %d / 阴性 %d）"
          % (len(df), int(df["pos"].sum()), int((1 - df["pos"]).sum())), flush=True)

    cyc = []
    for pth in sorted(glob.glob("data/aws/gfs_*_f000_L21_1000_100.grib2")):
        b = os.path.basename(pth).split("_")
        cyc.append((dt.datetime.strptime(b[1] + b[2], "%Y%m%d%H").replace(tzinfo=dt.timezone.utc), pth))
    cyc = sorted(cyc)
    print("可用 GFS 分析时次：%d 个" % len(cyc))

    rows, tot = [], 0
    for ct, pth in cyc:
        dtv = (df["ot"] - ct).dt.total_seconds().abs() / 3600.0
        sub = df[dtv <= WIN_H]
        if len(sub) < 50:
            continue
        d = AG.load(pth, bbox=BBOX)
        lat, lon, lev = d["lat"], d["lon"], d["level"]
        diag = RK.stability_fields(dict(T=d["T"], q=d["q"], u=d["u"], v=d["v"], z=d["z"]),
                                   lat, lon, lev)
        qlat = sub["lat"].to_numpy(float)
        qlon = np.mod(sub["lon"].to_numpy(float), 360.0)
        if float(lon.max()) <= 180.0:
            qlon = np.where(qlon > 180, qlon - 360, qlon)
        alt = sub["fltLvl"].to_numpy(float) * 30.48          # FL(百英尺) → m

        Zcol = bilinear_cols(np.asarray(diag["Z"], float), lat, lon, qlat, qlon)
        pgrid = np.broadcast_to(lev[None, :], Zcol.shape)
        ptar = np.array([np.interp(alt[i], Zcol[i], pgrid[i]) for i in range(len(alt))])
        del Zcol

        rec = dict(time=ct.strftime("%Y%m%d_%Hz"), lat=qlat, lon=qlon, p=ptar,
                   pos=sub["pos"].to_numpy(int), tbtype=sub["tbt"].to_numpy(object))
        for k in ("S2", "Nd2", "Nv2", "Ndk2", "rh", "dqdz", "delta", "delta_sat"):
            if k in diag:
                rec[k] = vert_interp(bilinear_cols(np.asarray(diag[k], float),
                                                   lat, lon, qlat, qlon), lev, ptar)
        rows.append(rec)
        tot += len(qlat)
        print("  %s  匹配 %5d 条（阳性 %4d）"
              % (rec["time"], len(qlat), int(rec["pos"].sum())), flush=True)
        del diag, d
        gc.collect()

    with open("data/pirep/step08_matched.pkl", "wb") as fh:
        pickle.dump(rows, fh)
    print("\n累计匹配样本 %d 条 → data/pirep/step08_matched.pkl" % tot)

    def stack(k):
        return np.concatenate([r[k] for r in rows])

    pos = stack("pos").astype(float)
    p, rh, S2 = stack("p"), stack("rh"), stack("S2")
    Nd2, Nv2, Ndk2 = stack("Nd2"), stack("Nv2"), stack("Ndk2")
    ttype = np.concatenate([r["tbtype"] for r in rows])

    def ri(n2):
        return n2 / np.maximum(S2, 1e-12)

    rid, riv, rik = ri(Nd2), ri(Nv2), ri(Ndk2)
    sc = lambda x: -np.log10(np.maximum(x, 1e-6))

    SUBSETS = [
        ("全部（含低层）", np.ones(len(pos), bool)),
        ("晴空 RH<80%", rh < RH_CLEAR),
        ("晴空 + 高层 p<=400hPa", (rh < RH_CLEAR) & (p <= 400)),
        ("晴空+高层+层结稳定", (rh < RH_CLEAR) & (p <= 400) & (Nd2 > 0)),
        ("CAT 类型子集", (rh < RH_CLEAR) & (p <= 400) & (ttype == "CAT")),
    ]

    print("\n" + "=" * 84)
    print("【表 I】三口径对 PIREP 颠簸的判别能力（AUC，越高越好）")
    print("=" * 84)
    print("%-24s %7s %7s %9s %9s %9s" % ("子集", "n", "阳性率", "Ri_dry", "Ri_virt", "Ri_DK"))
    res = {}
    for nm, m in SUBSETS:
        if int(m.sum()) < 200:
            print("%-24s %7d %7s %9s %9s %9s   (样本不足)"
                  % (nm, int(m.sum()), "-", "-", "-", "-"))
            continue
        a1 = auc_score(sc(rid[m]), pos[m])
        a2 = auc_score(sc(riv[m]), pos[m])
        a3 = auc_score(sc(rik[m]), pos[m])
        res[nm] = dict(n=int(m.sum()), pos_rate=float(pos[m].mean()),
                       auc_dry=a1, auc_virt=a2, auc_dk=a3)
        print("%-24s %7d %6.1f%% %9.4f %9.4f %9.4f"
              % (nm, int(m.sum()), 100 * pos[m].mean(), a1, a2, a3))

    print("\n" + "=" * 84)
    print("【表 II】ΔAUC 与 95%% 置信区间（配对 bootstrap 2000 次）")
    print("=" * 84)
    print("%-24s %26s %26s" % ("子集", "虚温 − 干", "DK − 干"))
    for nm, m in SUBSETS:
        if int(m.sum()) < 200:
            continue
        d1 = bootstrap_dauc(sc(riv[m]), sc(rid[m]), pos[m])
        d2 = bootstrap_dauc(sc(rik[m]), sc(rid[m]), pos[m])
        l1, h1 = np.nanpercentile(d1, [2.5, 97.5])
        l2, h2 = np.nanpercentile(d2, [2.5, 97.5])
        res[nm]["d_auc_virt"] = float(np.nanmedian(d1))
        res[nm]["d_auc_virt_ci"] = [float(l1), float(h1)]
        res[nm]["d_auc_dk"] = float(np.nanmedian(d2))
        res[nm]["d_auc_dk_ci"] = [float(l2), float(h2)]
        print("%-24s %+8.4f [%+.4f,%+.4f] %+9.4f [%+.4f,%+.4f]"
              % (nm, np.nanmedian(d1), l1, h1, np.nanmedian(d2), l2, h2))

    print("\n" + "=" * 84)
    print("【表 III】观测层面的「湿度涌现」：干口径判安全处，实际颠簸率")
    print("=" * 84)
    m = (rh < RH_CLEAR) & (p <= 400) & (Nd2 > 0)
    emg_tab = []
    for thr in (0.25, 0.5, 1.0):
        safe = m & (rid > thr)
        emg = safe & (riv <= thr)
        oth = safe & (riv > thr)
        r1 = float(pos[emg].mean()) if emg.sum() else np.nan
        r2 = float(pos[oth].mean()) if oth.sum() else np.nan
        ratio = (r1 / r2) if (np.isfinite(r1) and np.isfinite(r2) and r2 > 0) else np.nan
        emg_tab.append(dict(ri_crit=thr, n_dry_safe=int(safe.sum()),
                            n_emerged=int(emg.sum()), rate_emerged=r1,
                            n_dry_and_moist_safe=int(oth.sum()),
                            rate_both_safe=r2, ratio=ratio))
        print("  Ri_dry>%.2f：干+湿均判安全 n=%6d 颠簸率 %5.2f%%  ‖  湿口径翻转判危险 n=%6d 颠簸率 %5.2f%%  比 %s"
              % (thr, int(oth.sum()), 100 * r2 if np.isfinite(r2) else np.nan,
                 int(emg.sum()), 100 * r1 if np.isfinite(r1) else np.nan,
                 "%.2f 倍" % ratio if np.isfinite(ratio) else "—"))

    with open("data/pirep/step08_pirep_verification.json", "w", encoding="utf8") as fh:
        json.dump(dict(window_h=WIN_H, rh_clear=RH_CLEAR, n_total=int(len(pos)),
                       n_pos=int(pos.sum()),
                       times=[r["time"] for r in rows],
                       n_per_time=[int(len(r["lat"])) for r in rows],
                       subsets=res, emergence=emg_tab),
                  fh, ensure_ascii=False, indent=1)
    print("\n已写出 data/pirep/step08_pirep_verification.json   用时 %.0f s"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
