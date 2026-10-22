# -*- coding: utf-8 -*-
"""
avwx.gfs —— GFS 背景场的实时获取（NOMADS 子集服务）
====================================================

选用 NOAA NOMADS 的 GRIB 过滤服务而非整份 GFS：
  * 整份 pgrb2.0p25 约 300 MB，本环境国际带宽 10–80 KB/s，不可行；
  * 过滤服务可按变量 + 气压层 + 子区域裁切，单次约 0.4 MB，秒级返回；
  * 数据为 NOAA 官方实时分析场，可直接作为业务化背景场。

参考: https://nomads.ncep.noaa.gov/
"""
import os, io, time, datetime as dt
import numpy as np

BASE = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
LEVELS_MB = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]

# 25 hPa 等间隔的**高垂直分辨率**层组（1000→100 hPa，共 37 层）。
# 用途：湍流与稳定度诊断。理由——11 层的业务层组层厚达 1.5–5.5 km，
# 而 K–H 判据 Ri<0.25 是针对 ∂z ~ 10–100 m 的湍流尺度切变定义的；
# 层厚越大，中心差分越平滑，S² 被系统性低估、Ri 被高估 1–2 个量级。
# 25 hPa 层组把层厚压到 200–900 m，是 NOMADS 过滤服务能给的实用上限。
LEVELS_HI = list(range(1000, 95, -25))

VARS = ["TMP", "RH", "UGRD", "VGRD", "HGT"]
CACHE = "data/gfs"


def _http(url, tries=4):
    import urllib.request
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "avwx/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            if len(b) > 2000:
                return b
            last = RuntimeError(f"too small: {len(b)}")
        except Exception as e:
            last = e
        time.sleep(2 + 3 * k)
    raise last


def find_latest_cycle(max_back=6):
    """从最近的若干天/时次中找一个可用的 GFS 分析时次。"""
    import urllib.request
    now = dt.datetime.utcnow()
    cands = []
    for dh in range(0, max_back * 24, 6):
        t = now - dt.timedelta(hours=dh)
        cyc = (t.hour // 6) * 6
        cands.append((t.date(), cyc))
    # 粗查目录是否存在
    for dte, cyc in cands:
        idx = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
        try:
            req = urllib.request.Request(idx, headers={"User-Agent": "avwx/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                html = r.read().decode("utf8", "replace")
        except Exception:
            continue
        tag = f"gfs.{dte:%Y%m%d}/"
        if tag in html:
            return f"{dte:%Y%m%d}", f"{cyc:02d}"
    return f"{now:%Y%m%d}", "00"


def build_url(date, cycle, leftlon, rightlon, bottomlat, toplat,
              levels=None, fhr=0):
    lev = levels or LEVELS_MB
    parts = [f"file=gfs.t{cycle}z.pgrb2.0p25.f{fhr:03d}"]
    for L in lev:
        parts.append(f"lev_{L}_mb=on")
    for v in VARS:
        parts.append(f"var_{v}=on")
    parts.append("subregion=")
    parts.append(f"leftlon={leftlon}&rightlon={rightlon}")
    parts.append(f"toplat={toplat}&bottomlat={bottomlat}")
    parts.append(f"dir=%2Fgfs.{date}%2F{cycle}%2Fatmos")
    return BASE + "?" + "&".join(parts)


def fetch(date=None, cycle=None, bbox=(100, 130, 15, 40), fhr=0, cache=True,
          max_back=16, levels=None):
    """bbox = (leftlon, rightlon, bottomlat, toplat)。返回 GRIB 字节。

    levels: 气压层列表 [hPa]，None 时用 LEVELS_MB。
            湍流/稳定度诊断应传 LEVELS_HI（25 hPa 间隔，37 层）。

    关键实现细节：NOMADS 上"目录存在"并不等于"该时次的 f000 已发布"
    （实测 2026-10-20 的 12z/18z 目录存在但文件 404/403，只有 00z/06z 可用）。
    因此必须逐个候选时次真实尝试，而不是只查目录。
    """
    lo, hi, la0, la1 = bbox
    os.makedirs(CACHE, exist_ok=True)
    lv = levels or LEVELS_MB
    lvtag = "L%d_%d_%d" % (len(lv), lv[0], lv[-1])
    if date and cycle:
        cands = [(date, cycle)]
    else:
        now = dt.datetime.utcnow()
        cands = []
        # 由近及远回退，跨到前一天
        for k in range(max_back):
            t = now - dt.timedelta(hours=6 * k)
            cands.append((f"{t:%Y%m%d}", f"{(t.hour // 6) * 6:02d}"))
    last = None
    for dte, cyc in cands:
        key = f"gfs_{dte}_{cyc}_f{fhr:03d}_{lo}_{hi}_{la0}_{la1}_{lvtag}.grib2"
        p = os.path.join(CACHE, key)
        if cache and os.path.exists(p) and os.path.getsize(p) > 2000:
            with open(p, "rb") as f:
                return f.read()
        url = build_url(dte, cyc, lo, hi, la0, la1, levels=levels, fhr=fhr)
        try:
            b = _http(url, tries=2)
        except Exception as e:
            last = e
            continue
        with open(p, "wb") as f:
            f.write(b)
        return b
    raise RuntimeError(f"no usable GFS cycle among {cands[:4]}... last={last}")


def load(path_or_bytes):
    """用 cfgrib 读取为 xarray，返回 dict(t, q, u, v, z) + 坐标。

    RH → 比湿 q 的转换使用 thermo 模块，与全项目一致。
    """
    import xarray as xr
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from avwx import thermo as TH
    # cfgrib 在收到字节流时无法建立消息上下文（实测 TypeError:
    # 'int' object is not subscriptable），必须落到临时文件再打开。
    if isinstance(path_or_bytes, (bytes, bytearray)):
        os.makedirs(CACHE, exist_ok=True)
        tmp = os.path.join(CACHE, "_tmp_decode.grib2")
        with open(tmp, "wb") as f:
            f.write(path_or_bytes)
        path_or_bytes = tmp
    ds = xr.open_dataset(path_or_bytes, engine="cfgrib")
    lev = np.array(ds.isobaricInhPa.values, float)
    lat = np.array(ds.latitude.values, float)
    lon = np.array(ds.longitude.values, float)
    # cfgrib 输出 lat 常为降序，统一为升序
    if lat[0] > lat[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))
        lat = np.array(ds.latitude.values, float)
    T = np.array(ds.t.values, float)                      # K, (lev, lat, lon)
    RH = np.clip(np.array(ds.r.values, float), 0, 100)    # %
    U = np.array(ds.u.values, float)
    V = np.array(ds.v.values, float)
    Z = np.array(ds.gh.values, float) if "gh" in ds else None
    P = lev[:, None, None] * 100.0
    Tc = T - 273.15
    q = np.clip(TH.specific_humidity_from_rh(Tc, RH, P), 0, 0.06)
    return dict(level=lev, lat=lat, lon=lon, T=T, q=q, u=U, v=V, z=Z,
                time=str(ds.time.values) if "time" in ds else "")


def fetch_sfc_pres(date, cycle, bbox=(100, 130, 15, 40), fhr=0, cache=True):
    """只取地面气压 PRES:surface（约 23 KB，用于等压面的地形遮蔽）。

    为什么必须有它
    --------------
    NOMADS 子集服务给出的等压面 HGT **不做地形遮蔽**：本项目实测
    1000 hPa 位势高度在本区域全域仅 43–178 m，连海拔 ~1900 m 的昆明
    也只报 85 m。也就是说，高原地区的 1000/925/850 hPa 是**地下外推值**，
    物理量上没有意义，却与真实层混在一起无法从三维场本身分辨。

    判据：气压层 p 位于地面之上 ⟺ p ≤ p_sfc（p > p_sfc 即在地下）。
    """
    lo, hi, la0, la1 = bbox
    os.makedirs(CACHE, exist_ok=True)
    key = f"gfs_{date}_{cycle}_f{fhr:03d}_{lo}_{hi}_{la0}_{la1}_sfcpres.grib2"
    p = os.path.join(CACHE, key)
    if cache and os.path.exists(p) and os.path.getsize(p) > 2000:
        with open(p, "rb") as f:
            return f.read()
    parts = [f"file=gfs.t{cycle}z.pgrb2.0p25.f{fhr:03d}",
             "lev_surface=on", "var_PRES=on", "subregion=",
             f"leftlon={lo}&rightlon={hi}", f"toplat={la1}&bottomlat={la0}",
             f"dir=%2Fgfs.{date}%2F{cycle}%2Fatmos"]
    b = _http(BASE + "?" + "&".join(parts))
    with open(p, "wb") as f:
        f.write(b)
    return b


def load_sfc_pres(path_or_bytes):
    """读取地面气压，返回 (Pa, lat, lon)，lat 统一升序。"""
    import xarray as xr
    if isinstance(path_or_bytes, (bytes, bytearray)):
        os.makedirs(CACHE, exist_ok=True)
        tmp = os.path.join(CACHE, "_tmp_sfc.grib2")
        with open(tmp, "wb") as f:
            f.write(path_or_bytes)
        path_or_bytes = tmp
    ds = xr.open_dataset(path_or_bytes, engine="cfgrib")
    lat = np.array(ds.latitude.values, float)
    lon = np.array(ds.longitude.values, float)
    if lat[0] > lat[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))
        lat = np.array(ds.latitude.values, float)
    return np.array(ds.sp.values, float), lat, lon


def above_ground_mask(lev, p_sfc):
    """气压层是否在地面之上。(nz,ny,nx) 布尔场。

    p_sfc 形状 (ny,nx) [Pa]；lev 降序 [hPa]（与本项目其余部分一致）。
    """
    lev_pa = np.asarray(lev, float)[:, None, None] * 100.0
    return lev_pa <= np.asarray(p_sfc, float)[None, ...]


def load_hi(date, cycle, bbox=(100, 130, 15, 40), levels=None, fhr=0,
            cache=True, ngroup=2):
    """高垂直分辨率下载：**交错分组**多次请求后沿气压层合并。

    为什么要分组
    ------------
    NOMADS 过滤服务对单次请求存在限制：本项目实测一次性请求 37 层
    （1000→100 hPa，25 hPa 间隔）稳定返回 HTTP 500，而 11 层正常。
    限制的具体形式（层数上限 / URL 长度 / 服务端并发）未公开，故采用
    分组规避——这是纯工程绕过，不改变数据内容与物理口径。

    为什么用**交错**分组而不是连续分段
    ----------------------------------
    levels[i::ngroup] 使每组内部仍保持均匀的 2×Δp 间隔（本例 50 hPa），
    避免某一组全是低层或全是高层，从而在任一分组失败时其余分组
    仍覆盖完整气柱（可降级为 50 hPa 分辨率而不至于缺失整段高度）。

    合并后按气压降序排列，与 load() 的输出约定一致。
    """
    levels = list(levels or LEVELS_HI)
    parts = []
    for i in range(ngroup):
        ch = levels[i::ngroup]
        if not ch:
            continue
        parts.append(load(fetch(date=date, cycle=cycle, bbox=bbox, fhr=fhr,
                                cache=cache, levels=ch)))
    if len(parts) == 1:
        return parts[0]
    lvl = np.concatenate([p["level"] for p in parts])
    idx = np.argsort(-lvl)                       # 降序
    out = dict(level=lvl[idx], lat=parts[0]["lat"], lon=parts[0]["lon"],
               time=parts[0]["time"])
    for k in ("T", "q", "u", "v", "z"):
        if parts[0].get(k) is None:
            out[k] = None
        else:
            out[k] = np.concatenate([p[k] for p in parts], axis=0)[idx]
    return out
