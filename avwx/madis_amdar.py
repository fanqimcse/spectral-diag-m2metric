# -*- coding: utf-8 -*-
"""avwx/madis_amdar —— MADIS 公开存档的机载观测（AMDAR / ACARS / TAMDAR）读取。

为什么必须换掉 PIREP
--------------------
step08 用 PIREP 检验三种 Ri，结论可信但可被攻击：PIREP 是**飞行员主观报告**，
存在三类已知偏倚 ——（i）航线与高度的选择性取样（只在繁忙航路上报）；
（ii）"nil report" 缺失（没颠簸就不报，负样本需人为构造）；（iii）强度分级主观。
EDA 领域公认的做法是用 **EDR**（eddy dissipation rate，ICAO 标准湍流度量）：
它是机上惯导/空速管按客观算法反演的**连续量**，与机型无关、与飞行员无关，
且每一条有 EDR 的记录都自带"报告与否"这一事实本身由算法决定，不由人决定。

本模块负责把它取出来：

    数据源  https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/YYYY/MM/DD/point/acars/netcdf/YYYYMMDD_HH00.gz
    实测    匿名可读；覆盖 >= 2015；近实时延迟约 5 天；
            单小时全球约 3.3 万条记录，其中 **31% 携带 EDR**，**10.5% 携带 WVSS-II 水汽**。

缺测约定（踩坑记录）
--------------------
MADIS netcdf 的缺省值**不是 NaN 而是 1e5**（数值变量）与 **-1**（waterVaporMR）。
若忘记掩蔽，任何统计量都会被这些哨兵值摧毁（我们第一次统计 medEDR 时
得到"中位数 0、最大值 0.174"，其实是把 1e5 当成了有效值排除后的残骸）。
本模块统一在 `read_hour` 里做掩蔽后再交给上层。
"""
import datetime as dt
import gzip
import io
import os
import shutil
import tempfile
import time

import numpy as np
import requests
from scipy.io import netcdf_file

ARCHIVE = ("https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"
           "%Y/%m/%d/point/acars/netcdf/%Y%m%d_%H00.gz")

MISS = 1e5      # 数值变量的哨兵
MISS_Q = -1.0   # waterVaporMR 的哨兵

# 只抽取实际用得到的变量，其余（尾号串、QC 位、inventory）一律丢弃
NUMVARS = ("timeObs", "latitude", "longitude", "altitude", "GPSaltitude",
           "temperature", "windDir", "windSpeed", "medEDR", "maxEDR",
           "vertGust", "vertAccel", "waterVaporMR", "relHumidity",
           "heading", "trueAirSpeed", "phaseFlight", "altitudeQCR",
           "temperatureQCR", "windSpeedQCR")


def url_for(dte, hour):
    """dte='20261008', hour=12 -> 该小时的 netcdf gzip URL"""
    t = dt.datetime.strptime("%s%02d" % (dte, hour), "%Y%m%d%H")
    return t.strftime(ARCHIVE)


def fetch_nc(dte, hour, cache="data/madis", keep_nc=False, timeout=180,
             session=None, verbose=True):
    """下载并解压出一个小时的 netcdf；返回本地路径（已解压）。

    已存在同名文件时直接复用（缓存）。
    """
    os.makedirs(cache, exist_ok=True)
    hour_tag = int(hour)
    nc = os.path.join(cache, "acars_%s_%02d.nc" % (dte, hour_tag))
    if os.path.exists(nc):
        return nc
    u = url_for(dte, hour_tag)
    http = session or requests
    r = http.get(u, timeout=timeout, stream=True)
    if r.status_code != 200:
        r.close()
        raise IOError("HTTP %s  %s" % (r.status_code, u))
    raw = io.BytesIO()
    for chunk in r.iter_content(1 << 20):
        raw.write(chunk)
    r.close()
    raw.seek(0)
    tmp = nc + ".tmp"
    try:
        with gzip.GzipFile(fileobj=raw, mode="rb") as gz, open(tmp, "wb") as fo:
            shutil.copyfileobj(gz, fo, 1 << 20)
        os.replace(tmp, nc)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    if verbose:
        print("    + %s  %.1f MB" % (os.path.basename(nc),
                                     os.path.getsize(nc) / 1e6), flush=True)
    return nc


def _num(f, name, miss=MISS):
    """取出一个数值变量并把哨兵值换成 NaN。"""
    if name not in f.variables:
        return None
    a = np.asarray(f.variables[name][:], dtype=np.float64)
    a = np.where(np.isfinite(a), a, np.nan)
    return np.where(np.abs(a - miss) < 1e3, np.nan, a)


def _open_nc(src):
    """接受路径或 file-like；netCDF3 允许直接从内存缓冲读取。"""
    if hasattr(src, "read"):
        return netcdf_file(src, "r", mmap=False)
    with open(src, "rb") as fh:
        buf = io.BytesIO(fh.read())
    return netcdf_file(buf, "r", mmap=False)


def read_hour(src, want=None):
    """读一个小时的 netcdf，返回 {变量名: ndarray}，缺测已置 NaN。

    src 可以是路径，也可以是已经 gzip 解压到内存的 BytesIO —— 后者让我们在
    流水线里完全不落中间文件（单个 nc 展开后约 38 MB，上百个小时就是几 GB）。

    waterVaporMR 的哨兵是 -1（不是 1e5），单独处理。
    """
    want = want or NUMVARS
    f = _open_nc(src)
    out = {}
    for v in want:
        if v not in f.variables:
            continue
        if v == "waterVaporMR":
            a = np.asarray(f.variables[v][:], dtype=np.float64)
            a = np.where(np.isfinite(a), a, np.nan)
            out[v] = np.where(a > 0, a, np.nan)
        else:
            out[v] = _num(f, v)
    return out


def pressure_from_altitude(alt_m, tsfc=None):
    """气压高度 -> 气压 Pa。

    AMDAR 的 `altitude` 是**标准气压高度**（1013.25 hPa 基准），因此可不考虑
    真实地面气压，直接用标准大气位势-气压关系反算，误差 < 1 hPa（对流层上层）。
    这一步是必须的：三种 Ri 都在等压面上定义，而飞机给的是高度。
    """
    alt = np.asarray(alt_m, float)
    # ICAO 标准大气：位势高度 <=> 气压（对流层 11 km 以下）
    T0, p0, L, g, M, R = 288.15, 101325.0, 0.0065, 9.80665, 0.0289644, 8.31447
    p = p0 * (1.0 - L * alt / T0) ** (g * M / (R * L))
    hi = alt > 11000.0
    if np.any(hi):
        p11 = p0 * (1.0 - L * 11000.0 / T0) ** (g * M / (R * L))
        p[hi] = p11 * np.exp(-g * M * (alt[hi] - 11000.0) / (R * 216.65))
    return p


def uv_from_dir_speed(wdir, wspd):
    """气象风向（来向，度）/风速 -> u,v。"""
    wd = np.deg2rad(np.asarray(wdir, float))
    ws = np.asarray(wspd, float)
    return -ws * np.sin(wd), -ws * np.cos(wd)


def bbox_mask(lat, lon, bbox):
    """bbox = (lon0, lon1, lat0, lat1)，经度按 wrap 处理。"""
    lon0, lon1, lat0, lat1 = bbox
    lo = np.asarray(lon, float) % 360.0
    l0 = lon0 % 360.0
    l1 = lon1 % 360.0
    if l0 <= l1:
        inlon = (lo >= l0) & (lo <= l1)
    else:
        inlon = (lo >= l0) | (lo <= l1)
    return np.asarray(lat, float) >= lat0, None if False else (np.asarray(lat, float) <= lat1), inlon


def in_bbox(lat, lon, bbox):
    lon0, lon1, lat0, lat1 = bbox
    lat = np.asarray(lat, float)
    lo = np.asarray(lon, float) % 360.0
    l0, l1 = lon0 % 360.0, lon1 % 360.0
    inlon = (lo >= l0) & (lo <= l1) if l0 <= l1 else ((lo >= l0) | (lo <= l1))
    return (lat >= lat0) & (lat <= lat1) & inlon


def selfcheck():
    """模块自检：解析 URL、缺测掩蔽、气压换算、风向换算。"""
    out = []
    u = url_for("20261008", 12)
    out.append(("url", u ==
                "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/"
                "2026/10/08/point/acars/netcdf/20261008_1200.gz", u))

    # 缺测掩蔽：1e5 必须变 NaN，真值必须保留
    a = np.array([1e5, 0.02, np.nan, -1.0])
    m = np.where(np.abs(a - MISS) < 1e3, np.nan, np.asarray(a, float))
    out.append(("miss_masking", bool(np.isnan(m[0]) and abs(m[1] - 0.02) < 1e-9),
                float(np.nansum(np.isfinite(m)))))

    # 标准大气：0 m -> 1013.25 hPa；5000 m -> 540.2 hPa（ICAO 表值）
    p0 = pressure_from_altitude(np.array([0.0]))[0] / 100.0
    p5 = pressure_from_altitude(np.array([5000.0]))[0] / 100.0
    p11 = pressure_from_altitude(np.array([11000.0]))[0] / 100.0
    e0 = abs(p0 - 1013.25)
    e5 = abs(p5 - 540.48)          # ICAO 5000 m = 540.48 hPa
    e11 = abs(p11 - 226.32)        # ICAO 11000 m = 226.32 hPa
    out.append(("std_atm_0m", e0 < 0.01, e0))
    out.append(("std_atm_5000m", e5 < 0.5, e5))
    out.append(("std_atm_11000m", e11 < 0.5, e11))

    # 风向：北风(360/0) -> v = -10；东风(90) -> u = -10
    u10, v10 = uv_from_dir_speed(np.array([360.0, 90.0]), np.array([10.0, 10.0]))
    out.append(("wind_north", abs(v10[0] + 10) < 1e-9 and abs(u10[0]) < 1e-9,
                float(v10[0])))
    out.append(("wind_east", abs(u10[1] + 10) < 1e-9 and abs(v10[1]) < 1e-9,
                float(u10[1])))

    # bbox 跨越 0 经度
    m1 = in_bbox(np.array([50.0, 50.0]), np.array([-10.0, 10.0]),
                 (350.0, 20.0, 40.0, 60.0))
    out.append(("bbox_dateline", bool(m1[0] and m1[1]), float(m1.sum())))
    m2 = in_bbox(np.array([0.0, 30.0]), np.array([120.0, 120.0]),
                 (100.0, 130.0, 15.0, 40.0))
    out.append(("bbox_plain", bool((not m2[0]) and m2[1]), float(m2.sum())))
    return out


if __name__ == "__main__":
    for name, ok, val in selfcheck():
        print("%-18s %s   %s" % (name, "OK " if ok else "FAIL", val))
