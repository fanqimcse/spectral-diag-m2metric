# -*- coding: utf-8 -*-
"""step29 —— 用独立探空观测验证"误差谱决定诊断精度"（把 OSSE 结论外推）。

背景与科学问题
--------------
step20–27 的全部结论（"同化移除大尺度、平滑移除小尺度""场精度不是诊断
精度的代理量"）都建立在一个 OSSE 上，其真值、背景、观测、集合成员**全是
GFS 一家**。审稿人会追问的最尖锐的一刀是：

    "你们证明的是 GFS 预报误差的谱结构，还是真实大气的误差谱结构？
     OSSE 的自洽性会不会把结论变成循环论证？"

本脚本用 **IGRA2 探空（独立观测，非模式）** 来回答。核心检验：

  命题 V（独立观测版）：对真实大气的背景误差（GFS f024 预报 vs 探空），
  · 同化（用 GFS f000 分析 vs 探空）移除的误差**主要在垂直廓线的大尺度
    （整层系统性偏差）**，而对**层间小尺度梯度（诊断量所需）**的修正有限；
  · 因此"场 RMSE 的改善"与"诊断量（垂直梯度）误差的改善"不成比例，
    与 OSSE 的谱—诊断对应原理一致。

方法（诚实声明其边界）
--------------------
探空是**单点垂直廓线**，无法直接做"水平误差谱分解"。但可以做等价检验：
  对每个站点、每个时次，把探空廓线插值到 GFS 气压层，得到：
    · 场误差 e = T_gfs − T_raob（每层）
    · 垂直梯度误差 e' = (dT/dz)_gfs − (dT/dz)_raob（层间差分）
  检验：同化（f024 → f000）对"场误差 e"的相对改善，是否**系统性小于**
  对"梯度误差 e'"的相对改善——即同化对诊断量的作用更弱。

这等价于 OSSE 里"同化对 M0 有效、对 M2 几乎无效"的结论，但现在是
**独立观测**在验证。

数据
----
- 探空：IGRA2，东亚 18 个非中国站点（2022–2024 有实测）
- GFS：项目已有的 data/aws 缓存（f000 分析 + f024 预报，同日历日）
"""
import gc
import json
import os
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, ".")
import step10_realerror_osse as S10
from avwx import thermo as TH

OUT = "data/step29_raob_verify.json"

# 目标日期（与 step19 跨年一致的日历日）
TARGET_DAYS = [
    (2022, 1, 15), (2022, 1, 16), (2022, 7, 15), (2022, 7, 16),
    (2023, 1, 15), (2023, 1, 16), (2023, 7, 15), (2023, 7, 16),
]
# 有 2022-2023 实测的站点（从 igra_fetch.log 确认）
SITES = {
    "JAM00047909": (28.39, 129.49, "名濑/日本"),
    "JAM00047918": (24.34, 124.16, "石垣/日本"),
    "KSM00047122": (37.10, 127.03, "乌山/韩国"),
    "KSM00047138": (36.03, 129.38, "浦项/韩国"),
    "KSM00047155": (35.17, 128.57, "昌原/韩国"),
    "KSM00047169": (34.69, 125.45, "黑山岛/韩国"),
    "KSM00047230": (37.26, 126.10, "德积/韩国"),
    "HKM00045004": (22.31, 114.17, "香港"),
    "VMM00048820": (21.03, 105.80, "河内/越南"),
    "VMM00048845": (18.67, 105.68, "荣市/越南"),
    "VMM00048855": (16.07, 108.35, "岘港/越南"),
    "RPM00098223": (18.18, 120.53, "拉瓦格/菲律宾"),
    "RPM00098233": (17.65, 121.76, "土格加劳/菲律宾"),
    "THM00048378": (16.80, 100.28, "彭世洛/泰国"),
    "THM00048381": (16.46, 102.79, "孔敬/泰国"),
    "THM00048407": (15.25, 104.87, "乌汶/泰国"),
    "PFM00059981": (16.83, 112.33, "西沙"),
}

# GFS bbox 已覆盖东亚（100-130E, 15-40N），这些站点都在 bbox 内


def parse_igra_sounding(sid, target_days):
    """解析一个站点的探空，只保留目标日期的探空。

    target_days: set of (year, mon, day)。
    返回 {datetime: [(p, gph, T, rh, wdir, wspd), ...]}，只含目标日。
    按 IGRA2 数据行固定宽度解析（见 igra2-data-format.txt）。
    """
    fn = "data/igra_%s.zip" % sid
    if not os.path.exists(fn):
        return {}
    z = zipfile.ZipFile(fn)
    txt = z.read(z.namelist()[0]).decode("utf8", "replace")
    lines = txt.split("\n")
    soundings = {}
    cur_key = None
    cur = []
    keep_cur = False

    def conv(s):
        s = s.strip()
        if s in ("", "-9999", "-8888"):
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    for L in lines:
        if L.startswith("#"):
            # 收尾上一条
            if cur_key is not None and keep_cur and cur:
                soundings[cur_key] = cur
            cur = []
            keep_cur = False
            p = L.split()
            if len(p) < 6:
                cur_key = None
                continue
            try:
                year, mon, day, hour = int(p[1]), int(p[2]), int(p[3]), int(p[4])
            except (ValueError, IndexError):
                cur_key = None
                continue
            cur_key = (year, mon, day, hour)
            keep_cur = (year, mon, day) in target_days
        else:
            if cur_key is None or not keep_cur:
                continue
            if len(L) < 51:
                continue
            press = L[9:15].strip()
            temp = L[22:27].strip()
            rh = L[28:33].strip()
            wdir = L[40:45].strip()
            wspd = L[46:51].strip()
            pv = conv(press)
            if not np.isfinite(pv) or pv < 1000:
                continue
            cur.append(dict(
                p=pv,              # Pa
                T=conv(temp),      # 0.1°C
                rh=conv(rh),       # 0.1%
                wdir=conv(wdir),
                wspd=conv(wspd),   # 0.1 m/s
            ))
    if cur_key is not None and keep_cur and cur:
        soundings[cur_key] = cur
    return soundings


def interp_gfs_to_raob(gfs_T, gfs_q, gfs_u, gfs_v, gfs_lev, raob):
    """把 GFS 三维场插值到探空站点的 (lat, lon, 气压层) 上。

    gfs_*: (nz, ny, nx)，gfs_lev: (nz,) 降序 hPa。
    raob: 探空记录的列表 [(p_Pa, T, ...)]。
    返回 dict 的逐层 (gfs_T, gfs_q, gfs_u, gfs_v, raob_T, raob_rh, ...)。
    """
    # 找到探空站点最近的格点（简化：用最近的纬度/经度格点，而非插值）
    # 这一步需要 lat/lon，由调用方传入
    return None


def main():
    t0 = time.time()
    print("=" * 98)
    print("step29  用独立探空观测验证误差谱命题（OSSE → 真实大气）")
    print("=" * 98)

    # ---------- 1. 解析探空 ----------
    print("\n[1] 解析探空数据（只保留目标日期）")
    print("-" * 98)
    target_day_set = {(d[0], d[1], d[2]) for d in TARGET_DAYS}
    all_sond = {}
    for sid, (lat, lon, name) in SITES.items():
        snd = parse_igra_sounding(sid, target_day_set)
        # 统计目标日期的探空数量
        n_match = sum(1 for k in snd if k[3] in (0, 12))
        all_sond[sid] = snd
        print("  %s %-12s %6.1f°N %6.1f°E  目标日探空 %d"
              % (sid, name, lat, lon, n_match))

    # ---------- 2. 匹配 GFS 并做误差分析 ----------
    print("\n[2] 匹配 GFS（f024 背景 vs f000 分析）并分析误差")
    print("-" * 98)

    # 预加载所有 GFS 场到缓存（避免循环内重复读 GRIB）
    gfs_cache = {}
    def load_gfs(date, cyc, fhr):
        key = (date, cyc, fhr)
        if key not in gfs_cache:
            p = S10.native_path(date, cyc, fhr)
            if not os.path.exists(p):
                gfs_cache[key] = None
            else:
                gfs_cache[key] = S10.prepare(S10.AG.load(p, bbox=S10.BBOX))
        return gfs_cache[key]

    rows = []
    for sid, (slat, slon, name) in SITES.items():
        snd = all_sond[sid]
        for (year, mon, day) in TARGET_DAYS:
            date = "%04d%02d%02d" % (year, mon, day)
            # 探空时次：00 和 12 UTC
            for hh in (0, 12):
                key = (year, mon, day, hh)
                if key not in snd:
                    continue
                raob = snd[key]
                cyc = "%02d" % hh
                # f000 分析 + f024 背景（起报 = date - 24h）
                nat = load_gfs(date, cyc, 0)
                if nat is None:
                    continue
                b = S10.bg_cycle(date, cyc, 24)
                bg = load_gfs(b[0], b[1], 24)
                if bg is None:
                    continue

                # 取 GFS 场
                lat = nat["lat"]; lon = nat["lon"]; lev = nat["lev"]
                # 找最近的格点
                ilat = int(np.argmin(np.abs(lat - slat)))
                ilon = int(np.argmin(np.abs(lon - slon)))
                gfs_lev = np.asarray(lev, float)

                # 探空插值到 GFS 层（按气压线性插值，log-p）
                raob_p = np.array([r["p"] for r in raob]) / 100.0  # hPa
                raob_T = np.array([r["T"] for r in raob]) / 10.0    # °C
                raob_rh = np.array([r["rh"] for r in raob]) / 10.0  # %
                raob_wdir = np.array([r["wdir"] for r in raob])
                raob_wspd = np.array([r["wspd"] for r in raob]) / 10.0

                # 有效探空层
                valid = (np.isfinite(raob_T) & (raob_p > 0) & (raob_p < 1100))
                if valid.sum() < 5:
                    continue
                raob_p = raob_p[valid]
                raob_T = raob_T[valid]
                raob_rh = raob_rh[valid]

                # GFS 温度（K → °C）
                gfs_T_f000 = nat["f"]["T"][:, ilat, ilon] - 273.15
                gfs_T_f024 = bg["f"]["T"][:, ilat, ilon] - 273.15
                gfs_u_f000 = nat["f"]["u"][:, ilat, ilon]
                gfs_u_f024 = bg["f"]["u"][:, ilat, ilon]
                gfs_v_f000 = nat["f"]["v"][:, ilat, ilon]
                gfs_v_f024 = bg["f"]["v"][:, ilat, ilon]

                # 探空 → GFS 层插值（log-p 线性）
                logp_raob = np.log(raob_p)
                logp_gfs = np.log(gfs_lev)
                # 只取探空覆盖的 GFS 层范围
                lo = logp_gfs.min()
                hi = logp_gfs.max()
                sel = (logp_raob >= lo) & (logp_raob <= hi)
                if sel.sum() < 3:
                    continue
                raob_T_gfslev = np.interp(logp_gfs, logp_raob[sel][::-1],
                                          raob_T[sel][::-1])

                # 误差（对探空）
                e_T_f024 = gfs_T_f024 - raob_T_gfslev
                e_T_f000 = gfs_T_f000 - raob_T_gfslev

                # 垂直梯度误差（诊断量所需）：dT/dlnp
                def dTdlnp(T, logp):
                    return np.gradient(T, logp)
                grad_raob = dTdlnp(raob_T_gfslev, logp_gfs)
                grad_f024 = dTdlnp(gfs_T_f024, logp_gfs)
                grad_f000 = dTdlnp(gfs_T_f000, logp_gfs)
                e_grad_f024 = grad_f024 - grad_raob
                e_grad_f000 = grad_f000 - grad_raob

                # 巡航层（200-400 hPa）掩码
                cruise = (gfs_lev >= 200) & (gfs_lev <= 400)

                # RMSE
                def rmse(e, mask):
                    m = mask & np.isfinite(e)
                    return float(np.sqrt(np.nanmean(e[m] ** 2))) if m.sum() > 0 else np.nan

                rmse_T_f024 = rmse(e_T_f024, cruise)
                rmse_T_f000 = rmse(e_T_f000, cruise)
                rmse_grad_f024 = rmse(e_grad_f024, cruise)
                rmse_grad_f000 = rmse(e_grad_f000, cruise)

                if not np.isfinite(rmse_T_f024) or not np.isfinite(rmse_grad_f024):
                    continue

                # 相对改善（同化 f024→f000）
                d_field = (rmse_T_f024 - rmse_T_f000) / rmse_T_f024
                d_grad = (rmse_grad_f024 - rmse_grad_f000) / rmse_grad_f024

                rows.append(dict(
                    sid=sid, name=name, date=date, hour=hh,
                    rmse_T_f024=rmse_T_f024, rmse_T_f000=rmse_T_f000,
                    rmse_grad_f024=rmse_grad_f024, rmse_grad_f000=rmse_grad_f000,
                    d_field=d_field, d_grad=d_grad))

        gc.collect()

    # ---------- 3. 汇总 ----------
    print("\n[3] 汇总：同化（f024→f000）对'场误差'与'梯度误差'的相对改善")
    print("-" * 98)
    if not rows:
        print("  无匹配样本！检查 GFS 缓存与探空日期。")
        json.dump(dict(rows=[]), open(OUT, "w", encoding="utf8"))
        return

    d_field = np.array([r["d_field"] for r in rows])
    d_grad = np.array([r["d_grad"] for r in rows])
    print("  匹配样本数：%d（站点×时次）" % len(rows))
    print("  场误差（T，巡航层）相对改善：中位 %+.1f%%（同化 f024→f000）"
          % (100 * np.median(d_field)))
    print("  梯度误差（dT/dlnp，巡航层）相对改善：中位 %+.1f%%"
          % (100 * np.median(d_grad)))
    print()
    print("  关键检验：同化对'梯度误差'的改善 是否 小于 对'场误差'的改善？")
    if np.median(d_grad) < np.median(d_field):
        print("  ⇒ 是：梯度误差改善 %+.1f%% < 场误差改善 %+.1f%%"
              % (100 * np.median(d_grad), 100 * np.median(d_field)))
        print("  ⇒ 独立观测验证了谱—诊断对应原理：同化对诊断量（梯度）的作用更弱。")
    else:
        print("  ⇒ 否：梯度误差改善 %+.1f%% ≥ 场误差改善 %+.1f%%"
              % (100 * np.median(d_grad), 100 * np.median(d_field)))
        print("  ⇒ 独立观测**不支持**该命题（需进一步分析原因）。")

    # 逐样本明细
    print("\n  逐样本（前 20）：")
    print("  %-14s %-10s %-4s %8s %8s %8s %8s" %
          ("站点", "日期", "时次", "T_f024", "T_f000", "grad_f024", "grad_f000"))
    for r in rows[:20]:
        print("  %-14s %-10s %-4s %8.2f %8.2f %8.2f %8.2f"
              % (r["name"], r["date"], r["hour"],
                 r["rmse_T_f024"], r["rmse_T_f000"],
                 r["rmse_grad_f024"], r["rmse_grad_f000"]))

    out = dict(rows=rows,
               median=dict(d_field=float(np.median(d_field)),
                           d_grad=float(np.median(d_grad)),
                           n=len(rows)))
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
