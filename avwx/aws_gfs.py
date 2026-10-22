# -*- coding: utf-8 -*-
"""
avwx.aws_gfs —— 经 AWS Open Data 精确抽取 GFS 任意气压层（绕开 NOMADS 限制）
=============================================================================

为什么需要它
------------
NOMADS 的 GRIB 过滤服务（avwx.gfs）在请求 11 层以上时稳定返回 HTTP 500，
且**失败形式极具误导性**：11 层成功、13/15/21/37 层失败，看起来像层数上限，
但实测同一时刻请求 PRES:surface 又正常，说明并非简单的层数限制（疑为服务端
按 URL 长度/并发的限流）。分组合并同样失败。因此 NOMADS 只能给 11 层，
而层厚 1.5–5.5 km 的 11 层组无法刻画 K–H 判据所针对的 ~10–100 m 尺度切变。

AWS Open Data（noaa-gfs-bdp-pds）保存**完整** GFS 产品并提供 HTTP Range 请求，
配合官方 `.idx` 索引可**只下载需要的 GRIB2 消息**，从而拿到：

  * 原生全部等压面层（1000→100 hPa 共 **21 层**，而非 NOMADS 的 11 层）；
  * 全球覆盖 ⇒ 一次下载可切出**任意多个区域**（东亚 / 北美 / 热带西太平洋），
    这对"多区域气候学"是决定性的；
  * 历史回溯（实测 2026-01-15 仍可用）⇒ 一次下载可取**任意季节**。

代价：每条 GRIB2 消息是**全球场**（约 0.7–1.2 MB），而本项目只用到 1% 左右的
面积。抽取 21 层 × 5 变量 ≈ 84 MB/时次，必须用多线程 Range 下载。

用法举例
--------
    import avwx.aws_gfs as AG
    p = AG.fetch(date="20261019", cycle="06", levels=AG.LEVELS_NATIVE)
    d = AG.load(p, bbox=(100, 130, 15, 40))       # 与 avwx.gfs.load 同构
"""
import io
import os
import time
import datetime as _dt
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
CACHE = "data/aws"

# GFS 在 1000→100 hPa 之间的**原生**等压面层（含 Cf 之外的 41 层中的对流层/下平流层段）。
# 注意：不存在 875/825/775 hPa，故 25 hPa 等间隔的 37 层组在本产品上并不存在；
# 21 层已是 GFS 原生垂直分辨率的上限 —— 这一点必须在报告中诚实说明。
LEVELS_NATIVE = [1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600,
                 550, 500, 450, 400, 350, 300, 250, 200, 150, 100]

# 业务层组（NOMADS 给的 11 层），用作垂直分辨率对照的低分辨率端。
LEVELS_OPS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]

# 更粗一档，用于三档分辨率对照（检验是否单调收敛）。
LEVELS_COARSE = [1000, 850, 700, 500, 300, 200, 100]

VARS = ["TMP", "RH", "UGRD", "VGRD", "HGT"]
SFC_VARS = ["PRES"]


# ---------------------------------------------------------------- 索引
def _http(url, tries=4, timeout=180, extra=None):
    h = {"User-Agent": "avwx/1.0"}
    if extra:
        h.update(extra)
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(2 + 3 * k)
    raise last


def grib_url(date, cycle, fhr=0, product="pgrb2.0p25"):
    return "%s/gfs.%s/%s/atmos/gfs.t%sz.%s.f%03d" % (
        BASE, date, cycle, cycle, product, fhr)


def read_index(date, cycle, fhr=0, product="pgrb2.0p25", cache=True):
    """返回 [(n, offset, size, var, level)]，level 形如 '500 mb' / 'surface'。

    .idx 每行: `<msgno>:<byte offset>:<date>:<var>:<level>:<ftime>:`
    单条消息长度 = 下一行 offset − 本行 offset（末条需额外探测，这里用 None）。
    """
    os.makedirs(CACHE, exist_ok=True)
    key = os.path.join(CACHE, "idx_%s_%s_f%03d.idx" % (date, cycle, fhr))
    if cache and os.path.exists(key) and os.path.getsize(key) > 1000:
        raw = io.open(key, encoding="utf8").read()
    else:
        raw = _http(grib_url(date, cycle, fhr, product) + ".idx").decode("utf8", "replace")
        io.open(key, "w", encoding="utf8").write(raw)
    lines = [L for L in raw.strip().split("\n") if L.strip()]
    total = None
    out = []
    for i, L in enumerate(lines):
        p = L.split(":")
        off = int(p[1])
        nxt = int(lines[i + 1].split(":")[1]) if i + 1 < len(lines) else None
        out.append((int(p[0]), off, None if nxt is None else nxt - off,
                    p[3], p[4].strip()))
    return out


def _total_size(date, cycle, fhr, product):
    """末条消息长度需要文件总长（HTTP HEAD）。"""
    url = grib_url(date, cycle, fhr, product)
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "avwx/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def select(records, variables=VARS, levels=LEVELS_NATIVE, surface_vars=SFC_VARS):
    """按 (变量, 层) 挑选消息，并合并**物理相邻**的字节区间以减少连接数。

    GFS 产品通常按变量成组排放，故同一变量的连续层大多是相邻字节区间，
    合并后一次 Range 可取多条消息。GRIB2 消息自带结束标志（7777），
    因此按字节拼接后 cfgrib 仍能正确切分。
    """
    want = set()
    for L in levels:
        for v in variables:
            want.add((v, "%d mb" % L))
    for v in surface_vars:
        want.add((v, "surface"))
    hit = [r for r in records if (r[3], r[4]) in want]
    # 保持文件原序
    hit.sort(key=lambda r: r[1])
    return hit


def coalesce(hit, total=None, max_gap=0):
    """把**字节上严格相邻**的消息合并成一个区间。

    max_gap 默认 0：idx 中相邻消息的 offset 差恰好等于前一条的长度，因此
    只有真正连续的消息才会被合并，不会多下载一个字节。跨变量的消息之间通常
    隔着其它变量（VVEL/DZDT/ABSV...），gap 远大于 0，会自然分成独立区间，
    交给线程池并行下载。
    """
    runs = []
    for r in hit:
        off, size = r[1], r[2]
        if off is None or size == 0:
            continue
        # 末条消息在 idx 中没有"下一条 offset"可减，长度须由文件总长推算为
        # **剩余字节数**（total − offset），而不是 total 本身。
        end = off + size if size is not None else total
        if runs and off - runs[-1][1] <= max_gap:
            runs[-1][1] = end
        else:
            runs.append([off, end])
    return [(a, b) for a, b in runs]


def _range_get(args):
    url, start, end, idx, timeout = args
    h = {"User-Agent": "avwx/1.0", "Range": "bytes=%d-%d" % (start, end - 1)}
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return idx, start, r.read()


def download_ranges(url, runs, nproc=8, timeout=240):
    """并行 Range 下载。返回 [(offset, bytes)]。"""
    jobs = [(url, a, b, i, timeout) for i, (a, b) in enumerate(runs)]
    got = []
    with ThreadPoolExecutor(max_workers=nproc) as ex:
        for idx, start, blob in ex.map(_range_get, jobs):
            got.append((start, blob))
    got.sort(key=lambda x: x[0])
    return got


def fetch(date, cycle, fhr=0, variables=VARS, levels=LEVELS_NATIVE,
          surface_vars=SFC_VARS, cache=True, product="pgrb2.0p25",
          nproc=8, verbose=True):
    """抽取指定变量/层的 GRIB2 消息，拼成一个文件并返回路径。"""
    os.makedirs(CACHE, exist_ok=True)
    lvtag = "L%d_%d_%d" % (len(levels), levels[0], levels[-1])
    out = os.path.join(CACHE, "gfs_%s_%s_f%03d_%s.grib2" % (date, cycle, fhr, lvtag))
    if cache and os.path.exists(out) and os.path.getsize(out) > 100000:
        if verbose:
            print("  [aws] cache hit  %s  (%.1f MB)" % (
                os.path.basename(out), os.path.getsize(out) / 1e6))
        return out
    rec = read_index(date, cycle, fhr, product)
    hit = select(rec, variables, levels, surface_vars)
    if not hit:
        raise RuntimeError("no messages selected for %s %s" % (date, cycle))
    total = _total_size(date, cycle, fhr, product)
    runs = coalesce(hit, total=total)
    nbytes = sum(b - a for a, b in runs)
    if verbose:
        print("  [aws] %s %sZ  消息 %d 条 → 区间 %d 个, %.1f MB, %d 线程"
              % (date, cycle, len(hit), len(runs), nbytes / 1e6, nproc),
              flush=True)
    t0 = time.time()
    got = download_ranges(grib_url(date, cycle, fhr, product), runs, nproc=nproc)
    with open(out, "wb") as f:
        for start, blob in got:
            f.write(blob)
    if verbose:
        dt_ = time.time() - t0
        print("  [aws] 完成 %.1f MB / %.0f s → %.0f KB/s" % (
            os.path.getsize(out) / 1e6, dt_, os.path.getsize(out) / dt_ / 1024),
              flush=True)
    return out


# ---------------------------------------------------------------- 读取
def _match_lon(lon, lo, hi):
    """在 lon 上匹配 [lo, hi]，兼容 -180..180 与 0..360 两种约定、以及跨 180°。

    GFS 全球场的 longitude 轴是 0..359.75，而北美既可写成 235~290（0..360 约定）
    也可写成 -125~-70（-180..180 约定）。两种都必须能命中，否则会静默给出空
    数组 —— 这是一个极隐蔽的失败模式，必须由显式异常拦住。
    """
    L = np.asarray(lon, float)
    alt = np.where(L > 180, L - 360, L)
    cands = []
    if lo <= hi:
        cands.append((L >= lo - 1e-6) & (L <= hi + 1e-6))
        cands.append((alt >= lo - 1e-6) & (alt <= hi + 1e-6))
    else:                                   # 跨 ±180
        cands.append((alt >= lo - 1e-6) | (alt <= hi + 1e-6))
        cands.append((L >= lo - 1e-6) | (L <= hi + 1e-6))
    for m in cands:
        if m.any():
            return np.where(m)[0]
    raise RuntimeError("bbox lon %s..%s 与数据经度轴不相交" % (lo, hi))


def _subset_latlon(lat, lon, bbox):
    """返回 (lat_slice, lon_slice 或 lon 索引数组)。"""
    lo, hi, la0, la1 = bbox
    ix = _match_lon(lon, lo, hi)
    if np.any(np.diff(ix) > 1):             # 跨 180° 的两段，取第一段
        ix = ix[:int(np.argmax(np.diff(ix) > 1)) + 1]
    iy = np.where((lat >= la0 - 1e-6) & (lat <= la1 + 1e-6))[0]
    if len(iy) == 0:
        raise RuntimeError("bbox lat %s..%s 与数据纬度轴不相交" % (la0, la1))
    LonSl = slice(ix[0], ix[-1] + 1) if np.all(np.diff(ix) == 1) else ix
    return slice(iy[0], iy[-1] + 1), LonSl


def load(path_or_bytes, bbox=None):
    """读取并按 bbox 裁切。返回 dict(level, lat, lon, T, q, u, v, z, p_sfc, time)。

    与 avwx.gfs.load 的约定完全一致（lat 升序、level 降序、q 由 RH 换算），
    额外给出 p_sfc 以便后续做地形遮蔽。
    """
    import xarray as xr
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from avwx import thermo as TH

    if isinstance(path_or_bytes, (bytes, bytearray)):
        os.makedirs(CACHE, exist_ok=True)
        tmp = os.path.join(CACHE, "_tmp_aws.grib2")
        with open(tmp, "wb") as f:
            f.write(path_or_bytes)
        path_or_bytes = tmp

    dsets = xr.open_dataset(path_or_bytes, engine="cfgrib",
                            backend_kwargs={"indexpath": ""})
    ds = dsets
    lev = np.array(ds.isobaricInhPa.values, float)
    lat = np.array(ds.latitude.values, float)
    lon = np.array(ds.longitude.values, float)
    if lat[0] > lat[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))
        lat = np.array(ds.latitude.values, float)

    order = np.argsort(-lev)                     # 降序
    lev = lev[order]

    def take(name, extra=False):
        a = np.array(ds[name].values, float)
        if extra:                                # 二维（地面量）无需排序
            pass
        else:
            a = a[order]
        return a

    T = take("t"); RH = np.clip(take("r"), 0, 100)
    U = take("u"); V = take("v")
    Z = take("gh") if "gh" in ds else None
    p_sfc = np.array(ds.sp.values, float) if "sp" in ds else None

    P = lev[:, None, None] * 100.0
    q = np.clip(TH.specific_humidity_from_rh(T - 273.15, RH, P), 0, 0.06)

    out = dict(level=lev, lat=lat, lon=lon, T=T, q=q, u=U, v=V, z=Z,
               p_sfc=p_sfc, time=str(ds.time.values) if "time" in ds else "")
    if bbox is None:
        return out
    return subset(out, bbox)


def subset(d, bbox):
    """对 load() 的输出按 (leftlon, rightlon, bottomlat, toplat) 裁切。"""
    siy, six = _subset_latlon(d["lat"], d["lon"], bbox)
    out = dict(level=d["level"], lat=d["lat"][siy], lon=d["lon"][six],
               time=d["time"])
    for k in ("T", "q", "u", "v", "z"):
        arr = d.get(k)
        out[k] = None if arr is None else np.ascontiguousarray(
            arr[:, siy, six] if isinstance(six, slice) else arr[:, siy][:, :, six])
    if d.get("p_sfc") is not None:
        out["p_sfc"] = np.ascontiguousarray(
            d["p_sfc"][siy, six] if isinstance(six, slice) else d["p_sfc"][siy][:, six])
    # lon 归一到 -180..180，与其它数据源一致
    if out["lon"].max() > 180:
        out["lon"] = np.where(out["lon"] > 180, out["lon"] - 360, out["lon"])
        k = np.argsort(out["lon"])
        out["lon"] = out["lon"][k]
        for vv in ("T", "q", "u", "v", "z"):
            if out[vv] is not None:
                out[vv] = np.ascontiguousarray(out[vv][:, :, k])
        if out.get("p_sfc") is not None:
            out["p_sfc"] = np.ascontiguousarray(out["p_sfc"][:, k])
    return out


def selfcheck():
    """不联网的自检：索引解析、区间合并、切片约定。"""
    ok = []

    def chk(name, cond, info=""):
        ok.append((name, bool(cond), info))

    fake = [
        (1, 0, 100, "TMP", "1000 mb"),
        (2, 100, 100, "TMP", "975 mb"),
        (3, 200, 100, "UGRD", "1000 mb"),
        (4, 5000, 100, "RH", "500 mb"),
        (5, 5100, None, "PRES", "surface"),
    ]
    # 只选 TMP/RH；UGRD 虽在 fake 数据里但不该被选中
    hit = select(fake, variables=["TMP", "RH"],
                 levels=[1000, 975, 500], surface_vars=["PRES"])
    chk("select 过滤未请求变量", all(r[3] != "UGRD" for r in hit))
    chk("select 命中请求层",
        sorted(set(r[4] for r in hit)) == ["1000 mb", "500 mb", "975 mb", "surface"])
    chk("select 按文件偏移排序", [r[1] for r in hit] == sorted(r[1] for r in hit))

    runs = coalesce(hit, total=5200)
    # 0-100 与 100-200 严格相邻 → 合并；5000-5100 与 5100-5200 相邻 → 合并
    chk("coalesce 只合并严格相邻消息", runs == [(0, 200), (5000, 5200)], str(runs))
    chk("coalesce 不吞入无关字节", sum(b - a for a, b in runs) == 400)

    # bbox 切片：构造 1 度网格 -180..179
    lat = np.arange(-90, 91, 1.0)
    lon = np.arange(-180, 180, 1.0)
    siy, six = _subset_latlon(lat, lon, (100, 130, 15, 40))
    chk("lat 切片正确", lat[siy][0] == 15.0 and lat[siy][-1] == 40.0)
    chk("lon 切片正确", lon[six][0] == 100.0 and lon[six][-1] == 130.0)

    lat2 = np.arange(-90, 91, 1.0)
    lon2 = np.arange(0, 360, 1.0)
    siy, six = _subset_latlon(lat2, lon2, (235, 290, 25, 50))
    chk("lon 0..360 数组 + 0..360 bbox 可取北美",
        lon2[six][0] == 235 and lon2[six][-1] == 290)
    siy, six = _subset_latlon(lat2, lon2, (-125, -70, 25, 50))
    chk("lon 0..360 数组 + -180..180 bbox 同样可行",
        lon2[six][0] == 235 and lon2[six][-1] == 290)
    lon3 = np.arange(100, 131, 1.0)                        # 东亚子域
    try:
        _subset_latlon(lat2, lon3, (235, 290, 25, 50))     # 子域内不含北美
        chk("不相交时抛异常", False)
    except RuntimeError:
        chk("不相交时抛异常（不静默返回空）", True)

    return ok


if __name__ == "__main__":
    for name, cond, info in selfcheck():
        print("%-32s %s  %s" % (name, "OK " if cond else "FAIL", info))
