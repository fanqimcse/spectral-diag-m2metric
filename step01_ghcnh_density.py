# -*- coding: utf-8 -*-
"""
step01 —— 全球机场"含湿密度高度"气候学（观测基础）
==================================================

科学问题
--------
现行航空性能气候影响研究（Coffel, Thompson & Horton 2017, *Climatic Change* 144:251）
在计算空气密度时**只使用温度与气压**，即把大气当作干空气：

        ρ_dry = p / (R_d T)

真实大气是湿空气，水汽（分子量 18）比干空气（29）轻：

        ρ_moist = p / (R_d T_v),   T_v = T (1 + 0.608 q)

本脚本用 NOAA GHCNh 全球亚小时站网（179 站 / 5019 站年 / 1976–2023，
含**露点温度**与**场面气压**）逐小时计算两者之差，并给出：

  * 湿度惩罚 ΔDA_q = DA_moist − DA_dry [ft]        —— 现行做法系统性遗漏的量
  * 密度变化的 **精确对数三分解**（温度 / 湿度 / 气压）—— 气压无关
  * 逐小时气候态（24 时次）                        —— 用于"夜间能否缓解"判据
  * 日最大密度高度                                  —— 运行相关量

关键设计（稳健性优先）
----------------------
1. **主结论不依赖气压**。湿度份额 = Δln(1+0.608q) / (ΔlnT + Δln(1+0.608q))，
   气压项完全抵消。因此即便场面气压缺测 66 %，结论仍然成立。
2. **ΔDA_q 对气压误差极不敏感**：ΔDA_q ∝ σ^0.235，1 % 气压误差只带来 0.235 % 误差
   （<1 ft / 400 ft）。但**绝对 DA 对气压敏感**（约 30 ft/hPa），故绝对量另设气压来源标记。
3. **质控口径由数据本身判定**，不凭记忆（见 step00 与下方 QC_RATIONALE）。

QC_RATIONALE（本项目的质控依据，务必与论文方法段一致）
------------------------------------------------------
GHCNh 质量码中 '5' 占 18–40 %。若按"failed spatial consistency check"解释就必须剔除，
那会损失三分之一以上样本并可能引入选择偏差。本项目做了一次**判定检验**：
以同一站年 code-'1' 的逐小时中位数为基准，计算各码值的距平离散度：

        code '1' : sd = 3.74 °C
        code '5' : sd = 1.67 °C     ← 比 '1' 更贴合本站气候态
        code '4' : sd = 2.05 °C

若 '5' 真是质量失败标记，其离散度应**大于**，实测反而更小。
故本项目**接受 '0','1','4','5'**，仅剔除 '2','3','6','7'（gross/internal 失败）。
同时并行计算严格口径（仅 '1'）作为敏感性检验，两者结论一致才写入主结论。
"""
import os, sys, glob, json, time, collections
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avwx import thermo as TH

GHCNH_DIR = r"D:\2026codes\全球气候变化\ghcn_data"
OUT_CSV = "data/step01_station_year.csv"

COLS = ["STATION", "Year", "Month", "Day", "Hour", "Minute",
        "LATITUDE", "LONGITUDE", "ELEVATION",
        "temperature", "temperature_Quality_Code",
        "dew_point_temperature", "dew_point_temperature_Quality_Code",
        "station_level_pressure", "station_level_pressure_Quality_Code",
        "sea_level_pressure", "sea_level_pressure_Quality_Code"]

# 接受的质量码（见 QC_RATIONALE）
QC_OK_MAIN = {"0", "1", "4", "5"}
QC_OK_STRICT = {"0", "1"}

# 物理范围
T_RANGE = (-80.0, 60.0)
TD_RANGE = (-90.0, 40.0)
Q_RANGE = (0.0, 0.060)
P_RANGE = (30000.0, 110000.0)


def _col(tbl, name, dtype=float, default=np.nan):
    """列提取。数值列走 pyarrow→numpy 直转（避免 30 万个 Python 对象），
    字符串列走 to_pylist。

    本环境物理内存仅 10 GB，逐文件若用 to_pylist 会产生约 15 MB/文件的
    Python 对象垃圾，累积到数百个文件后触发 numpy 分配失败（实测已崩溃一次）。
    """
    c = tbl.column(name)
    if dtype is float:
        a = c.to_numpy(zero_copy_only=False)
        return np.asarray(a, dtype=float)
    return np.array(c.to_pylist(), dtype=object)


def pressure_pa(p_station_hpa, slp_hpa, elev_m):
    """三级气压方案，返回 (p_pa, source_flag)。

    flag = 1  观测场面气压（最优）
    flag = 2  由海平面气压经 ISA 换算（误差 ~1–2 hPa）
    flag = 3  标准大气（无天气信号，仅用于气候态）
    """
    isa = (1.0 - TH.L_ISA * np.maximum(elev_m, 0.0) / TH.T0_ISA) ** (TH.G / (TH.R_D * TH.L_ISA))
    isa = np.clip(isa, 0.1, 1.5)
    p1 = p_station_hpa * 100.0
    p2 = slp_hpa * 100.0 * isa
    p3 = TH.P0_ISA * isa
    out = np.where(np.isfinite(p1), p1, np.where(np.isfinite(p2), p2, p3))
    src = np.where(np.isfinite(p1), 1, np.where(np.isfinite(p2), 2, 3)).astype(int)
    return out, src


def process_file(path, strict=False):
    """处理单个站年文件，返回年度聚合字典（不保留逐小时原始值）。"""
    qc_ok = QC_OK_STRICT if strict else QC_OK_MAIN
    try:
        tbl = pq.read_table(path, columns=COLS)
    except Exception as e:
        return None
    n = tbl.num_rows
    if n == 0:
        return None

    st = tbl.column("STATION").to_pylist()[0]
    lat = float(tbl.column("LATITUDE").to_pylist()[0] or np.nan)
    lon = float(tbl.column("LONGITUDE").to_pylist()[0] or np.nan)
    elev = float(tbl.column("ELEVATION").to_pylist()[0] or 0.0)
    yr = int(tbl.column("Year").to_pylist()[0])

    T = _col(tbl, "temperature")
    Td = _col(tbl, "dew_point_temperature")
    ps = _col(tbl, "station_level_pressure")
    slp = _col(tbl, "sea_level_pressure")
    qcT = _col(tbl, "temperature_Quality_Code", dtype=object, default=None)
    qcD = _col(tbl, "dew_point_temperature_Quality_Code", dtype=object, default=None)
    qcP = _col(tbl, "station_level_pressure_Quality_Code", dtype=object, default=None)
    qcS = _col(tbl, "sea_level_pressure_Quality_Code", dtype=object, default=None)

    # 气压码过滤（'2'/'3' 视为失败）
    qcP_ok = np.array([(c is None) or (c in QC_OK_MAIN) and (c not in ("2", "3", "6", "7")) for c in qcP])
    qcS_ok = np.array([(c is None) or (c in QC_OK_MAIN) and (c not in ("2", "3", "6", "7")) for c in qcS])
    ps = np.where(qcP_ok & np.isfinite(ps), ps, np.nan)
    slp = np.where(qcS_ok & np.isfinite(slp), slp, np.nan)

    Hh = np.array([int(x) if x is not None else -1 for x in tbl.column("Hour").to_pylist()], dtype=int)
    dd = np.array([int(x) if x is not None else 0 for x in tbl.column("Day").to_pylist()], dtype=int)
    mm = np.array([int(x) if x is not None else 0 for x in tbl.column("Month").to_pylist()], dtype=int)
    # 日序号（含跨月）
    daykey = mm * 100 + dd

    okT = np.isfinite(T) & (T >= T_RANGE[0]) & (T <= T_RANGE[1]) & np.array([c in qc_ok for c in qcT])
    okD = np.isfinite(Td) & (Td >= TD_RANGE[0]) & (Td <= TD_RANGE[1]) & np.array([c in qc_ok for c in qcD])
    good = okT & okD & (Td <= T + 0.5) & (Hh >= 0)

    p_pa, psrc = pressure_pa(ps, slp, elev)
    good &= (p_pa >= P_RANGE[0]) & (p_pa <= P_RANGE[1])

    ng = int(good.sum())
    rec = {"station": st, "year": yr, "lat": lat, "lon": lon, "elev": elev,
           "n_rows": n, "n_good": ng, "coverage": ng / max(n, 1)}
    if ng < 200:
        rec["n_good"] = ng
        for k in ("da_dry_mean", "da_moist_mean", "dq_mean", "da_dry_dmax_mean",
                  "da_moist_dmax_mean", "t_k_mean", "q_mean", "p_mean"):
            rec[k] = np.nan
        return rec

    Tg = T[good]; Tdg = Td[good]; pg = p_pa[good]; Hg = Hh[good]; srcg = psrc[good]
    dkey = daykey[good]

    # 比湿（由露点 + 气压）
    q = TH.specific_humidity_from_dewpoint(Tdg, pg)
    q = np.clip(q, 0.0, 0.06)
    okq = (q >= Q_RANGE[0]) & (q <= Q_RANGE[1])
    Tg, Tdg, pg, Hg, q, srcg, dkey = Tg[okq], Tdg[okq], pg[okq], Hg[okq], q[okq], srcg[okq], dkey[okq]
    if len(q) < 200:
        for k in ("da_dry_mean", "da_moist_mean", "dq_mean", "da_dry_dmax_mean",
                  "da_moist_dmax_mean", "t_k_mean", "q_mean", "p_mean"):
            rec[k] = np.nan
        return rec

    Tk = Tg + 273.15
    rho_d = TH.air_density(pg, Tk, 0.0)
    rho_m = TH.air_density(pg, Tk, q)
    sig_d = rho_d / TH.RHO0_ISA
    sig_m = rho_m / TH.RHO0_ISA
    da_d = TH.density_altitude_ft(sig_d)
    da_m = TH.density_altitude_ft(sig_m)
    dq = da_m - da_d

    fin = np.isfinite(da_d) & np.isfinite(da_m)
    rec["n_final"] = int(fin.sum())

    rec["da_dry_mean"] = float(np.nanmean(da_d[fin]))
    rec["da_moist_mean"] = float(np.nanmean(da_m[fin]))
    rec["dq_mean"] = float(np.nanmean(dq[fin]))
    rec["dq_median"] = float(np.nanmedian(dq[fin]))
    rec["dq_p95"] = float(np.nanpercentile(dq[fin], 95))
    rec["t_k_mean"] = float(np.nanmean(Tk[fin]))
    rec["q_mean"] = float(np.nanmean(q[fin]))
    rec["p_mean"] = float(np.nanmean(pg[fin]))
    rec["t_c_mean"] = float(np.nanmean(Tg[fin]))
    rec["td_c_mean"] = float(np.nanmean(Tdg[fin]))
    rec["rh_mean"] = float(np.nanmean(TH.relative_humidity(Tg[fin], q[fin], pg[fin])))

    # 气压来源构成
    rec["frac_p_obs"] = float(np.mean(srcg == 1))
    rec["frac_p_slp"] = float(np.mean(srcg == 2))
    rec["frac_p_isa"] = float(np.mean(srcg == 3))

    # ---- 日最大（运行相关）----
    uniq, idx = np.unique(dkey[fin], return_inverse=True)
    cnt = np.bincount(idx, minlength=len(uniq))
    keep = cnt >= 4                       # 该日至少 4 个有效时次
    def dmax(vals):
        v = vals[fin]
        s = np.zeros(len(uniq))
        np.maximum.at(s, idx, v)
        return float(np.mean(s[keep])) if keep.sum() >= 20 else np.nan
    def dmin(vals):
        v = vals[fin]
        s = np.full(len(uniq), np.inf)
        np.minimum.at(s, idx, v)
        return float(np.mean(s[keep])) if keep.sum() >= 20 else np.nan
    rec["da_dry_dmax_mean"] = dmax(da_d)
    rec["da_moist_dmax_mean"] = dmax(da_m)
    rec["da_dry_dmin_mean"] = dmin(da_d)
    rec["da_moist_dmin_mean"] = dmin(da_m)
    # 日最大时刻的湿度惩罚
    s_idx = np.zeros(len(uniq), dtype=int)
    v = da_m[fin]
    # 取该日 da_m 最大的那一行对应的 dq
    best = np.full(len(uniq), -np.inf); bdq = np.zeros(len(uniq))
    for i, j in enumerate(idx):
        if v[i] > best[j]:
            best[j] = v[i]; bdq[j] = dq[fin][i]
    rec["dq_at_dailymax"] = float(np.mean(bdq[keep])) if keep.sum() >= 20 else np.nan
    v2 = da_m[fin]
    # 日最小时刻（夜间/清晨）的湿度惩罚 = 不可调度底线
    worst = np.full(len(uniq), np.inf); wdq = np.zeros(len(uniq))
    for i, j in enumerate(idx):
        if v2[i] < worst[j]:
            worst[j] = v2[i]; wdq[j] = dq[fin][i]
    rec["dq_at_dailymin"] = float(np.mean(wdq[keep])) if keep.sum() >= 20 else np.nan

    # ---- 逐小时气候态（24 时次）----
    clim = {}
    for hh in range(24):
        m = fin & (Hg == hh)
        if m.sum() < 20:
            clim[hh] = (np.nan, np.nan, np.nan, np.nan, np.nan)
            continue
        clim[hh] = (float(np.nanmean(da_d[m])), float(np.nanmean(da_m[m])),
                    float(np.nanmean(dq[m])), float(np.nanmean(Tg[m])),
                    float(np.nanmean(q[m]) * 1000.0))   # q in g/kg
    rec["_clim"] = clim

    # ---- 采样无关（sampling-neutral）年均值 ----
    # 关键：GHCNh 的采样频率从 3 小时（1970s）变为 1 小时/半小时（2000s）。
    # 直接对全部有效观测取年均值会混入采样偏差。故以"24 个逐小时均值的均值"作为
    # 采样无关年均值——这是本项目的主口径。
    arr = np.array([[v[0], v[1], v[2], v[3], v[4]] for v in clim.values()], float)
    rec["n_hour_bins"] = int(np.sum(np.isfinite(arr[:, 0])))
    with np.errstate(invalid="ignore"):
        rec["da_dry_clim"] = float(np.nanmean(arr[:, 0])) if rec["n_hour_bins"] >= 18 else np.nan
        rec["da_moist_clim"] = float(np.nanmean(arr[:, 1])) if rec["n_hour_bins"] >= 18 else np.nan
        rec["dq_clim"] = float(np.nanmean(arr[:, 2])) if rec["n_hour_bins"] >= 18 else np.nan
        rec["t_c_clim"] = float(np.nanmean(arr[:, 3])) if rec["n_hour_bins"] >= 18 else np.nan
        rec["q_clim_gkg"] = float(np.nanmean(arr[:, 4])) if rec["n_hour_bins"] >= 18 else np.nan
        # 不可调度底线：日最小 DA 时刻的湿度惩罚（逐小时气候态口径）
        rec["dq_clim_min"] = float(np.nanmin(arr[:, 2])) if rec["n_hour_bins"] >= 18 else np.nan
        rec["dq_clim_max"] = float(np.nanmax(arr[:, 2])) if rec["n_hour_bins"] >= 18 else np.nan
        rec["da_moist_clim_min"] = float(np.nanmin(arr[:, 1])) if rec["n_hour_bins"] >= 18 else np.nan
        rec["da_dry_clim_min"] = float(np.nanmin(arr[:, 0])) if rec["n_hour_bins"] >= 18 else np.nan
        rec["da_moist_clim_max"] = float(np.nanmax(arr[:, 1])) if rec["n_hour_bins"] >= 18 else np.nan
        rec["da_dry_clim_max"] = float(np.nanmax(arr[:, 0])) if rec["n_hour_bins"] >= 18 else np.nan
    rec["obs_per_day"] = rec["n_final"] / max(len(np.unique(dkey[fin])), 1)

    # ---- 采样一致（sampling-consistent）年均值：只用 8 个世界时正点 ----
    # GHCNh 采样频率在 1970s 为 3 小时、2000s 后为 1 小时。3 小时资料只覆盖
    # 00,03,...,21 八个时次，若要求 24 个时次全有数据，会把早期年份整体剔除，
    # 造成严重的样本选择偏差（实测只剩 68/179 站）。
    # 解法：只用所有年代都有的 8 个正点构建年均值——这是 HadISD/IGRA 等
    # 长序列气候数据集处理观测频率变化的标准做法。
    SYN = (0, 3, 6, 9, 12, 15, 18, 21)
    syn = np.array([[clim[h][0], clim[h][1], clim[h][2], clim[h][3], clim[h][4]]
                    for h in SYN], float)
    rec["n_syn_bins"] = int(np.sum(np.isfinite(syn[:, 0])))
    if rec["n_syn_bins"] >= 6:
        m_syn = np.isfinite(syn[:, 0])
        rec["da_dry_syn"] = float(np.nanmean(syn[m_syn, 0]))
        rec["da_moist_syn"] = float(np.nanmean(syn[m_syn, 1]))
        rec["dq_syn"] = float(np.nanmean(syn[m_syn, 2]))
        rec["t_c_syn"] = float(np.nanmean(syn[m_syn, 3]))
        rec["q_syn_gkg"] = float(np.nanmean(syn[m_syn, 4]))
        rec["dq_syn_min"] = float(np.nanmin(syn[m_syn, 2]))
        rec["dq_syn_max"] = float(np.nanmax(syn[m_syn, 2]))
        rec["da_moist_syn_min"] = float(np.nanmin(syn[m_syn, 1]))
        rec["da_dry_syn_min"] = float(np.nanmin(syn[m_syn, 1]))
    else:
        for k in ("da_dry_syn", "da_moist_syn", "dq_syn", "t_c_syn", "q_syn_gkg",
                  "dq_syn_min", "dq_syn_max", "da_moist_syn_min", "da_dry_syn_min"):
            rec[k] = np.nan
    return rec


def main(strict=False, limit=None):
    files = sorted(glob.glob(os.path.join(GHCNH_DIR, "GHCNh_*_*.parquet")))
    if limit:
        files = files[:limit]
    print(f"[step01] {len(files)} 站年文件, strict={strict}", flush=True)
    os.makedirs("data", exist_ok=True)
    t0 = time.time()
    rows = []
    import gc
    for i, f in enumerate(files):
        r = process_file(f, strict=strict)
        if r is not None:
            rows.append(r)
        del r
        if (i + 1) % 200 == 0:
            gc.collect()
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(files)}  {el:.0f}s  预计剩余 {el/(i+1)*(len(files)-i-1):.0f}s", flush=True)
    print(f"[step01] 完成 {len(rows)} 站年, 用时 {time.time()-t0:.0f}s", flush=True)

    # 拆出逐小时气候态
    clim_rows = []
    for r in rows:
        c = r.pop("_clim", {})
        for hh, v in c.items():
            clim_rows.append((r["station"], r["year"], hh) + v)
    import csv
    base = [k for k in rows[0].keys()]
    with open(OUT_CSV, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=base)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in base})
    with open("data/step01_hourly_clim.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(["station", "year", "hour", "da_dry", "da_moist", "dq", "t_c", "q_gkg"])
        w.writerows(clim_rows)
    print(f"[step01] 已写出 {OUT_CSV} ({len(rows)} 行)")
    print(f"[step01] 已写出 data/step01_hourly_clim.csv ({len(clim_rows)} 行)")


if __name__ == "__main__":
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(limit=lim)
