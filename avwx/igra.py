# -*- coding: utf-8 -*-
"""
avwx.igra —— IGRA v2 探空解析
=============================

IGRA v2 派生参数档（-drvd.txt）逐层固定宽度格式。
本模块只解析项目需要的字段：气压、位势高度、温度、露点差、比湿、
风向风速、高度——即三维风-温-湿重构的直接观测来源。

格式（IGRA v2 derived, 派生参数记录）：
  HEADER: 1-1 '#'  2-11 ID  12-17 year  18-19 month  20-21 day
          22-23 hour  24-25 minute(常缺) 26-31 释放时间  32-37 numlev
          38-45 PWAT  46-50 ... （各稳定度指数）
  DERIVED PARAM RECORD (每条一层):
          1-1  ' ' 或 '@'
          2-7  PRESSURE (Pa, 整数；海面或缺失时为 -9999)
          8-14 GEOPOTENTIAL HEIGHT (m)
          15-20 TEMPERATURE (degC *10)
          21-26 DEWPOINT DEPRESSION (degC *10)
          27-32 WIND DIRECTION (deg)
          33-38 WIND SPEED (m/s *10)
          39-44 U WIND COMPONENT (m/s*10)
          45-50 V WIND COMPONENT (m/s*10)
          51-56 SPECIFIC HUMIDITY (g/kg *1000)
          57-62 RELATIVE HUMIDITY (% *10)
          ...（其余为本研究不用的诊断量）

参考: IGRA v2 Documentation (Durre, Yin & Vose 2018, NOAA NCEI)
     https://www.ncei.noaa.gov/pub/data/igra/igra2-derived-format.txt
"""
import os, io, re, zipfile
import numpy as np

PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]

_HDR = re.compile(r"^#(?P<sid>[A-Z0-9]{11})\s")


def parse_drvd_text(text):
    """解析单个 derived 文本，返回逐次探空的列表。

    每项: dict(station, time(dict), levels(dict of arrays))
    """
    lines = text.splitlines()
    out = []
    i = 0
    n = len(lines)
    while i < n:
        L = lines[i]
        if not L.startswith("#") or len(L) < 40:
            i += 1
            continue
        sid = L[1:12]
        try:
            yr = int(L[13:17]); mo = int(L[18:20]); dy = int(L[21:23])
            hr = int(L[23:25]) if L[23:25].strip() else -9999
            nlev = int(L[31:37]) if L[31:37].strip() else 0
        except (ValueError, IndexError):
            i += 1
            continue
        i += 1
        nlev = min(nlev, 200)
        recs = []
        for j in range(nlev):
            if i + j >= n:
                break
            r = lines[i + j]
            if len(r) < 57:
                continue
            def fld(a, b, scale=1.0, miss=-9999.0):
                s = r[a - 1:b]
                s = s.strip()
                if not s or s.startswith("-9999") or s.startswith("9999"):
                    return np.nan
                try:
                    return float(s) / scale
                except ValueError:
                    return np.nan
            recs.append((
                fld(2, 7),        # pressure Pa
                fld(8, 14),       # geopotential height m
                fld(16, 21, 10.0),  # temperature degC   (注: 15 位是符号位)
                fld(22, 27, 10.0),  # dewpoint depression
                fld(28, 33, 10.0),  # wind direction
                fld(34, 39, 10.0),  # wind speed
                fld(40, 45, 10.0),  # u
                fld(46, 51, 10.0),  # v
                fld(52, 57, 1000.0),  # specific humidity g/kg -> /1000 => g/kg? 见下
            ))
        i += nlev
        if not recs:
            continue
        a = np.array(recs, float)
        lev = {
            "p_pa": a[:, 0], "z_m": a[:, 1], "t_c": a[:, 2], "dpd_c": a[:, 3],
            "wdir_deg": a[:, 4], "wspd_ms": a[:, 5], "u_ms": a[:, 6], "v_ms": a[:, 7],
            "q_gkg": a[:, 8],
        }
        # 派生：露点 = 温度 - 露点差
        lev["td_c"] = lev["t_c"] - lev["dpd_c"]
        out.append({"station": sid, "year": yr, "month": mo, "day": dy,
                    "hour": hr, "levels": lev})
    return out


def read_drvd_zip(path):
    """读取 -drvd.txt.zip，返回探空列表。"""
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        return parse_drvd_text(z.read(name).decode("utf8", errors="replace"))


def interp_to_pressure(lev, target_hpa):
    """把一次探空线性插值到目标气压层（在 ln p 上插值）。

    返回 dict of arrays，长度 = len(target_hpa)。
    """
    p = lev["p_pa"] / 100.0     # hPa
    tgt = np.asarray(target_hpa, float)
    out = {k: np.full(len(tgt), np.nan) for k in
           ("z_m", "t_c", "td_c", "u_ms", "v_ms", "q_gkg", "wspd_ms", "wdir_deg")}
    ok = np.isfinite(p) & (p > 0)
    if ok.sum() < 2:
        return out
    p = p[ok]
    lp = np.log(p)
    lt = np.log(tgt)
    # p 必须单调（探空自下而上，p 递减）；转为递增序
    order = np.argsort(lp)
    lp = lp[order]
    def itp(key):
        v = lev[key][ok][order]
        m = np.isfinite(v)
        if m.sum() < 2:
            return np.full(len(tgt), np.nan)
        return np.interp(lt, lp[m], v[m], left=np.nan, right=np.nan)
    for k in out:
        out[k] = itp(k)
    return out


def sounding_matrix(soundings, target_hpa, year_min=None, year_max=None,
                    require_levels=("t_c", "u_ms", "q_gkg")):
    """把多次探空整理成 (n_time, n_level) 的矩阵字典，便于做 OSE / 气候态。

    返回: dict(time_meta list, arrays...)
    """
    keys = ("z_m", "t_c", "td_c", "u_ms", "v_ms", "q_gkg")
    mats = {k: [] for k in keys}
    meta = []
    for s in soundings:
        if year_min is not None and s["year"] < year_min:
            continue
        if year_max is not None and s["year"] > year_max:
            continue
        r = interp_to_pressure(s["levels"], target_hpa)
        good = all(np.sum(np.isfinite(r[k])) >= max(3, len(target_hpa) // 3)
                   for k in require_levels)
        if not good:
            continue
        for k in keys:
            mats[k].append(r[k])
        meta.append((s["station"], s["year"], s["month"], s["day"], s["hour"]))
    for k in keys:
        mats[k] = np.array(mats[k], float) if mats[k] else np.zeros((0, len(target_hpa)))
    mats["meta"] = meta
    return mats
