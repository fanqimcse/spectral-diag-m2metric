# -*- coding: utf-8 -*-
"""step14b —— 1979–2025 年代际气候学：业务层组"失明"的时间演变。

任务 3 要回答的其实是一个**两问**的问题：
    (i)   决定"湿度—稳定度耦合"的那批物理量，47 年来变了多少？
    (ii)  业务层组（L11）相对**可获得的最细层组**漏掉了多少，这个漏掉的量
          是不是也随时间在变？
只做 (ii) 而没有 (i) 就没法解释趋势；只做 (i) 而不做 (ii) 就与"层组失明"
这个标题无关。两者一起做，趋势才可以被归因。

数据口径（必须交代）
--------------------
NCEP/DOE Reanalysis 2，**没有 shum**，只有 rhum。因此 q 由
    e = (RH/100)·e_s(T)；  q = ε e / (p − (1−ε)e)
反算。这引入一个已知偏差：R2 的 rhum 在低于 0°C 时仍是相对**水面**定义的，
高空的 q 因此被轻微高估。由于 δ ∝ dq/dz（比值型量），
该偏差在很大程度上被抵消，但绝对值不可与 GFS 侧直接比 ——
本脚本只报**趋势与相对量**，不报跨数据集的绝对对比。

可分辨性限制（必须交代）
------------------------
R2 在 1000–100 hPa 内只有 **12 层**，而业务 L11 是 11 层。
两者之差仅在于 600 hPa 一层。所以这里的"L11 失明"是一个**下界**，
不是真实值；真实值需要用 GFS 21 层（step06 已做，给出的是当代值）。
本脚本贡献的是这个下界的**时间演变**，step06 贡献的是当代的**完整幅度**。
"""
import json
import os
import sys

import numpy as np
from scipy.io import netcdf_file

sys.path.insert(0, ".")
from avwx import risk as RK

YEARS = list(range(1979, 2026))
MONTHS = ("01", "07")
VARS = ("air", "rhum", "uwnd", "vwnd", "hgt")
OPS11 = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]
RH_CLEAR = 80.0
N2_FLOOR = RK.N2_FLOOR
EPS = 0.622


def es_water(t_c):
    """Tetens 水面饱和水汽压 [hPa]。"""
    return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))


def read(var, year, month):
    p = "data/reanalysis/%s_EA_%d_%s.nc" % (var, year, month)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        f = netcdf_file(fh, "r", mmap=False)
        d = f.variables[var][:].astype(float)
        lv = f.variables["level"][:].astype(float)
        lat = f.variables["lat"][:].astype(float)
        lon = f.variables["lon"][:].astype(float)
        sc = getattr(f.variables[var], "scale_factor", None)
        ad = getattr(f.variables[var], "add_offset", None)
    if sc is not None:
        d = d * float(sc) + (float(ad) if ad is not None else 0.0)
    d = np.where(np.abs(d) > 1e29, np.nan, d)
    return dict(d=d, lev=lv, lat=lat, lon=lon)


def year_records(year, month):
    """返回该年该月逐层的中位数诊断量。"""
    a = read("air", year, month)
    r = read("rhum", year, month)
    u = read("uwnd", year, month)
    v = read("vwnd", year, month)
    z = read("hgt", year, month)
    if any(x is None for x in (a, r, u, v, z)):
        return None

    lev = a["lev"]
    keep = lev >= 100.0
    lev = lev[keep]
    T = a["d"][:, keep]
    RH = np.clip(r["d"][:, keep], 0.0, 100.0)
    U = u["d"][:, keep]
    V = v["d"][:, keep]
    Z = z["d"][:, keep]

    p = lev[None, :, None, None] * 100.0                    # Pa
    e = (RH / 100.0) * es_water(T - 273.15) * 100.0         # Pa
    q = EPS * e / np.maximum(p - (1.0 - EPS) * e, 1.0)
    q = np.clip(q, 0.0, 0.06)

    lat, lon = a["lat"], a["lon"]

    def diag_on(idx):
        """逐时次诊断，再跨时次聚合 → 每层一个 dict。

        ⚠ 第一版返回的是 [时次][层] 的嵌套列表，下游按层取最小值时
          拿到的是"某一层某个时次"，类型就不对了。诊断聚合必须在
          本函数内部完成：逐时次算**层内中位**，再跨时次取中位。
        """
        nz = len(idx)
        nt = T.shape[0]
        acc = {k: np.full((nt, nz), np.nan)
               for k in ("n", "delta", "dsat", "missed", "nd2", "s2")}
        for k in range(nt):
            fk = dict(T=T[k, idx], q=q[k, idx], u=U[k, idx], v=V[k, idx],
                      z=Z[k, idx])
            d = RK.stability_fields(fk, lat, lon, lev[idx])
            clr = d["rh"] < RH_CLEAR
            ok = clr & (d["Nd2"] > N2_FLOOR) & (d["S2"] > 0) \
                & np.isfinite(d["delta"]) & np.isfinite(d["delta_sat"])
            missed = clr & (d["Nd2"] > 0) & (d["Nv2"] < 0)
            for j in range(nz):
                m = ok[j]
                if m.sum() < 50:
                    continue
                acc["n"][k, j] = int(m.sum())
                acc["delta"][k, j] = float(np.median(d["delta"][j][m]))
                acc["dsat"][k, j] = float(np.median(d["delta_sat"][j][m]))
                acc["nd2"][k, j] = float(np.median(d["Nd2"][j][m]))
                acc["s2"][k, j] = float(np.median(d["S2"][j][m]))
                cj = clr[j]
                if cj.sum() > 50:
                    acc["missed"][k, j] = float(np.mean(missed[j][cj]))
        out = []
        for j in range(nz):
            r = dict(lev=float(lev[idx][j]))
            for k in acc:
                v = acc[k][:, j]
                r[k] = float(np.nansum(v)) if k == "n" \
                    else (float(np.nanmedian(v)) if np.isfinite(v).any()
                          else np.nan)
            if r["n"] < 50:
                r["n"] = 0
            out.append(r)
        return out

    idx_all = np.arange(len(lev))
    idx_ops = np.array([int(np.argmin(np.abs(lev - L))) for L in OPS11])
    idx_ops = np.array(sorted(set(idx_ops)))
    return dict(lev=lev.tolist(), native=diag_on(idx_all),
                ops=diag_on(idx_ops), ops_lev=lev[idx_ops].tolist())


def series_of(rows, key, lev_target):
    """取某一层（最接近 lev_target）的年序列。"""
    out = []
    for r in rows:
        if r is None:
            out.append(np.nan)
            continue
        best = min(r["native"], key=lambda x: abs(x["lev"] - lev_target))
        out.append(best.get(key, np.nan) if best.get("n", 0) >= 50 else np.nan)
    return np.array(out, float)


def theil_sen(x, y):
    """Theil–Sen 斜率（对离群点稳健）+ Mann–Kendall 近似 p。"""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x, float)[m], np.asarray(y, float)[m]
    n = x.size
    if n < 8:
        return dict(slope=np.nan, n=n, tau=np.nan, p=np.nan)
    sl = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[j] != x[i]:
                sl.append((y[j] - y[i]) / (x[j] - x[i]))
    slope = float(np.median(sl))
    # Kendall tau
    s = sum(np.sign(y[j] - y[i]) for i in range(n) for j in range(i + 1, n))
    tau = 2.0 * s / (n * (n - 1))
    var = n * (n - 1) * (2 * n + 5) / 18.0
    z = s / np.sqrt(var) if var > 0 else 0.0
    from math import erfc, sqrt
    p = erfc(abs(z) / sqrt(2.0))
    return dict(slope=slope, n=int(n), tau=float(tau), p=float(p))


def main():
    print("=" * 100)
    print("step14b  1979–2025 年代际气候学（东亚 100–145E, 15–45N）")
    print("=" * 100)
    for mi, month in enumerate(MONTHS):
        rows = []
        for y in YEARS:
            r = year_records(y, month)
            rows.append(r)
            if r is None:
                print("  %d-%s 缺失" % (y, month))
        print("\n【%s 月】" % month)
        print("%-8s %10s %10s %10s %10s %10s"
              % ("层[hPa]", "δ 1979s", "δ 2020s", "趋势/十年", "τ", "p"))
        table = {}
        for lev_target in (200, 250, 300, 400, 500, 700):
            s = series_of(rows, "delta", lev_target)
            yr = np.array(YEARS, float)
            ts = theil_sen(yr, s)
            e1 = s[(yr <= 1998)]
            e2 = s[(yr >= 2007)]
            table[lev_target] = dict(
                series=s.tolist(),
                early=float(np.nanmedian(e1)) if np.isfinite(e1).any() else None,
                late=float(np.nanmedian(e2)) if np.isfinite(e2).any() else None,
                slope_per_decade=ts["slope"] * 10, tau=ts["tau"], p=ts["p"])
            print("%-8d %9.4f%% %9.4f%% %+9.4f%% %+9.3f %10.4f"
                  % (lev_target,
                     100 * (table[lev_target]["early"] or np.nan),
                     100 * (table[lev_target]["late"] or np.nan),
                     100 * ts["slope"] * 10, ts["tau"], ts["p"]))

        # ---- L11 失明的下界及其趋势 ----
        print("\n  L11 业务层组 vs 最细层组（R2 12 层）——失明下界")
        print("  %-8s %12s %12s %12s" % ("层[hPa]", "δ 细", "δ L11", "相对损失"))
        blind = {}
        for lev_target in (200, 250, 300, 400, 500, 700):
            sn = series_of(rows, "delta", lev_target)
            so = []
            for r in rows:
                if r is None:
                    so.append(np.nan)
                    continue
                best = min(r["ops"], key=lambda x: abs(x["lev"] - lev_target))
                so.append(best.get("delta", np.nan)
                          if best.get("n", 0) >= 50 else np.nan)
            so = np.array(so, float)
            rel = np.where(np.isfinite(sn) & np.isfinite(so) & (np.abs(sn) > 1e-9),
                           1.0 - so / np.where(sn == 0, np.nan, sn), np.nan)
            ts = theil_sen(np.array(YEARS, float), rel)
            blind[lev_target] = dict(
                rel_median=float(np.nanmedian(rel)) if np.isfinite(rel).any() else None,
                slope_per_decade=ts["slope"] * 10, tau=ts["tau"], p=ts["p"],
                rel_series=[None if not np.isfinite(v) else v for v in rel])
            print("  %-8d %11.4f%% %11.4f%% %+11.2f%%"
                  % (lev_target, 100 * np.nanmedian(sn), 100 * np.nanmedian(so),
                     100 * np.nanmedian(rel)))

        with open("data/step14b_decadal_%s.json" % month, "w",
                  encoding="utf8") as f:
            json.dump(dict(years=YEARS, delta=table, blind=blind),
                      f, ensure_ascii=False, indent=1)
        print("  已写出 data/step14b_decadal_%s.json" % month)


if __name__ == "__main__":
    main()
