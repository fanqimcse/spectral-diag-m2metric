# -*- coding: utf-8 -*-
"""
step00 —— GHCNh 数据侦察
========================
目的：在写主流水线之前，先摸清 181 个站×48 年数据的真实可用性：

  1. 各变量（干球/露点/场面气压）的缺测率
  2. 质量码分布（决定 QC 规则）
  3. 物理一致性（露点 ≤ 干球）
  4. 时间覆盖（逐小时 vs 逐 3 小时）
  5. 场面气压缺测时的备选方案可行性

这是父项目"先侦察后建模"传统的延续：不先侦察就下结论，是假信号的主要来源。
"""
import os, sys, glob, json, collections
import numpy as np
import pyarrow.parquet as pq

GHCNH_DIR = r"D:\2026codes\全球气候变化\ghcn_data"

COLS = ["STATION", "Year", "Month", "Day", "Hour", "Minute",
        "LATITUDE", "LONGITUDE", "ELEVATION",
        "temperature", "temperature_Quality_Code",
        "dew_point_temperature", "dew_point_temperature_Quality_Code",
        "station_level_pressure", "station_level_pressure_Quality_Code",
        "sea_level_pressure", "sea_level_pressure_Quality_Code"]


def main():
    files = sorted(glob.glob(os.path.join(GHCNH_DIR, "GHCNh_*_*.parquet")))
    print(f"发现 {len(files)} 个站年文件")

    # 站点 → 年份清单
    st_years = collections.defaultdict(list)
    for f in files:
        b = os.path.basename(f)[:-8]           # GHCNh_STATION_YEAR
        st, yr = b[len("GHCNh_"):].rsplit("_", 1)
        st_years[st].append(int(yr))
    print(f"站点数: {len(st_years)}")
    spans = {s: (min(v), max(v), len(v)) for s, v in st_years.items()}
    nfull = sum(1 for s, (a, b, n) in spans.items() if n >= 40)
    print(f"年份数 ≥ 40 的站: {nfull}")

    # 均匀抽样若干个站年做侦察（跨纬度、跨时段）
    sample = []
    sts = sorted(st_years.keys())
    pick = [sts[int(i * len(sts) / 24)] for i in range(24)]
    for s in pick:
        ys = sorted(st_years[s])
        for y in (ys[0], ys[len(ys) // 2], ys[-1]):
            p = os.path.join(GHCNH_DIR, f"GHCNh_{s}_{y}.parquet")
            if os.path.exists(p):
                sample.append(p)
    print(f"抽样 {len(sample)} 个站年做细查")

    qc_t = collections.Counter(); qc_td = collections.Counter(); qc_p = collections.Counter()
    miss = collections.Counter(); total = 0
    phys_viol = 0; phys_n = 0
    hour_mods = collections.Counter()
    slp_fallback_needed = 0; p_present = 0
    elev_seen = {}

    for f in sample:
        t = pq.read_table(f, columns=COLS)
        d = t.to_pydict()
        n = t.num_rows; total += n
        st = d["STATION"][0]
        elev_seen[st] = d["ELEVATION"][0]
        T = np.array([np.nan if v is None else v for v in d["temperature"]], float)
        Td = np.array([np.nan if v is None else v for v in d["dew_point_temperature"]], float)
        P = np.array([np.nan if v is None else v for v in d["station_level_pressure"]], float)
        SLP = np.array([np.nan if v is None else v for v in d["sea_level_pressure"]], float)
        Hh = np.array([int(v) if v is not None else -1 for v in d["Hour"]], int)
        for c, key in ((d["temperature_Quality_Code"], "T"),
                       (d["dew_point_temperature_Quality_Code"], "Td"),
                       (d["station_level_pressure_Quality_Code"], "P")):
            pass
        qc_t.update(d["temperature_Quality_Code"])
        qc_td.update(d["dew_point_temperature_Quality_Code"])
        qc_p.update(d["station_level_pressure_Quality_Code"])
        miss["T"] += int(np.isnan(T).sum())
        miss["Td"] += int(np.isnan(Td).sum())
        miss["P"] += int(np.isnan(P).sum())
        miss["SLP"] += int(np.isnan(SLP).sum())
        ok = (~np.isnan(T)) & (~np.isnan(Td))
        phys_n += int(ok.sum())
        phys_viol += int((Td[ok] > T[ok] + 0.5).sum())
        hour_mods.update([h for h in Hh.tolist() if h >= 0])
        p_present += int(np.sum(~np.isnan(P)))
        slp_fallback_needed += int(np.sum(np.isnan(P) & ~np.isnan(SLP)))

    print(f"\n样本总行数 {total:,}")
    print(f"\n缺测率:")
    for k in ("T", "Td", "P", "SLP"):
        print(f"  {k:4s}: {100*miss[k]/total:6.2f} %")
    print(f"  场面气压缺但海压可用（可用作回退）: {slp_fallback_needed:,} 行 "
          f"({100*slp_fallback_needed/total:.2f} %)")
    print(f"\n质量码分布 (temperature): {dict(qc_t.most_common())}")
    print(f"质量码分布 (dew point)  : {dict(qc_td.most_common())}")
    print(f"质量码分布 (station P)  : {dict(qc_p.most_common())}")
    print(f"\n物理一致性: 露点 > 干球+0.5 的比例 = {100*phys_viol/max(phys_n,1):.3f} % (n={phys_n:,})")
    print(f"\n观测时次 (Hour 分布, 前 12): {sorted(hour_mods.items())[:12]}")
    print(f"海拔范围: {min(elev_seen.values()):.1f} … {max(elev_seen.values()):.1f} m")

    out = {
        "n_files": len(files), "n_stations": len(st_years),
        "stations_ge40yr": nfull,
        "sample_rows": total,
        "missing_pct": {k: 100 * miss[k] / total for k in ("T", "Td", "P", "SLP")},
        "qc_temperature": dict(qc_t), "qc_dewpoint": dict(qc_td), "qc_pressure": dict(qc_p),
        "td_gt_t_pct": 100 * phys_viol / max(phys_n, 1),
        "hours": dict(hour_mods),
        "station_years": {s: sorted(v) for s, v in st_years.items()},
    }
    os.makedirs("work", exist_ok=True)
    with open("work/step00_recon.json", "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 work/step00_recon.json")

    # 打印若干站的年份覆盖，便于判断可用样本
    cnt = sorted(spans.items(), key=lambda kv: -kv[1][2])[:15]
    print("\n年份覆盖最好的 15 站:")
    for s, (a, b, n) in cnt:
        print(f"  {s}  {a}-{b}  n={n}")


if __name__ == "__main__":
    main()
