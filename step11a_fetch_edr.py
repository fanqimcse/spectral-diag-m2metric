# -*- coding: utf-8 -*-
"""step11a —— 建立 EDR（客观湍流）检验数据集。

为什么值得为它单独写一个脚本
----------------------------
PIREP 是"人说的"，EDR 是"机器算的"。要用 EDR 复验 step08 的 H1/H2，
第一步是把足够多的机载湍流测量堆起来 —— 但单小时全球只有：
    ~3.3 万条记录 × 31% 带 EDR，其中非零 ~460 条，达到 light(≥0.1) 仅 ~13 条。
要获得统计上能用的正样本必须有**上百个小时**的量。

本脚本做三件事：
  1. 从 MADIS 公开存档（延迟约 5 天）拉取 2026 年若干天的全部 24 个小时文件；
  2. **边下载边抽取**：gzip -> netcdf -> 只保留北美/东亚 + 带 EDR 或带 WVSS-II 的记录
     -> 立即删除解压出来的 ~38 MB netcdf（不落 11 GB 中间数据）；
  3. 每天落一个紧凑 npz。

日期选择原则：必须能配到同时刻的 GFS 垂直廓线，因此优先取
工作区里已经有 GFS 原生层的那些天（跨冬/夏/秋三季）。
"""
import datetime as dt
import gzip
import io
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

sys.path.insert(0, ".")
from avwx import madis_amdar as MA

# 必须与 step10d 正在下载的 GFS f000 分析对齐 —— 那批数据是免费的配套背景场，
# 每天 4 个循环 ⇒ 时间错配 ≤ 1.5 h（远优于常规 ±3 h 窗口）。
DAYS = ["20260115", "20260116", "20260715", "20260716",
        "20261004", "20261008", "20261012", "20261016", "20261019"]
GZDIR = "data/madis/gz"
OUT = "data/madis/edr"

# 与 step06 一致的东亚/北美，另补两条 EDR 高发走廊。
# 实测单小时 EDR 有效样本的地理构成：北美 55%、北大西洋 16%、其它 27%（含欧洲）；
# 只取北美会丢掉近一半样本。bbox 仅用于**裁切**，扩区域不增加任何下载量。
REGIONS = {
    "NA": (235.0, 295.0, 25.0, 55.0),    # 北美
    "NATL": (295.0, 350.0, 25.0, 60.0),  # 北大西洋走廊（急流 CAT 高发）
    "EUR": (350.0, 30.0, 35.0, 65.0),    # 欧洲（跨 0°）
    "EA": (100.0, 145.0, 15.0, 45.0),    # 东亚
}

LOCAL = requests.Session()


def _local_session():
    return requests.Session()


_tls = {}


def get_session():
    import threading
    tid = threading.get_ident()
    if tid not in _tls:
        _tls[tid] = requests.Session()
    return _tls[tid]


def pull_hour(dte, hh):
    """下载->解压->抽取->返回精简 dict。中间 netcdf 用完即删。"""
    gzname = "acars_%s_%02d.gz" % (dte, hh)
    os.makedirs(GZDIR, exist_ok=True)
    gzpath = os.path.join(GZDIR, gzname)
    if not os.path.exists(gzpath) or os.path.getsize(gzpath) < 1000:
        u = MA.url_for(dte, hh)
        s = get_session()
        r = s.get(u, timeout=180, stream=True)
        if r.status_code != 200:
            r.close()
            return None, "HTTP %s" % r.status_code
        tmp = gzpath + ".tmp"
        try:
            with open(tmp, "wb") as fo:
                for ch in r.iter_content(1 << 20):
                    fo.write(ch)
            os.replace(tmp, gzpath)
        finally:
            r.close()
            if os.path.exists(tmp):
                os.remove(tmp)

    # gzip -> 内存 -> netcdf：全程不落中间文件。
    # （踩坑：一度用 tempfile.mkdtemp + shutil.rmtree 清理，触发了
    #   宿主安全删除策略导致整条流水线中断；改为纯内存后彻底绕开。）
    with gzip.open(gzpath, "rb") as gz:
        bio = io.BytesIO(gz.read())
    try:
        D = MA.read_hour(bio)
    finally:
        bio.close()

    lat, lon = D["latitude"], D["longitude"]
    keep = np.zeros(lat.size, bool)
    for bb in REGIONS.values():
        keep |= MA.in_bbox(lat, lon, bb)
    has_t = np.isfinite(D.get("temperature", np.full(lat.size, np.nan)))
    has_w = np.isfinite(D.get("windSpeed", np.full(lat.size, np.nan)))
    has_e = np.isfinite(D["medEDR"]) if "medEDR" in D else np.zeros(lat.size, bool)
    has_q = np.isfinite(D["waterVaporMR"]) if "waterVaporMR" in D else \
        np.zeros(lat.size, bool)
    m = keep & has_t & has_w & (has_e | has_q)
    if m.sum() == 0:
        return None, 0

    alt = D["altitude"][m]
    p = MA.pressure_from_altitude(alt)
    u10, v10 = MA.uv_from_dir_speed(D["windDir"][m], D["windSpeed"][m])
    rec = dict(
        time=D["timeObs"][m].astype(np.float64),
        lat=lat[m].astype(np.float32), lon=lon[m].astype(np.float32),
        alt=alt.astype(np.float32), p=p.astype(np.float32),
        T=D["temperature"][m].astype(np.float32),
        wspd=D["windSpeed"][m].astype(np.float32),
        wdir=D["windDir"][m].astype(np.float32),
        u=u10.astype(np.float32), v=v10.astype(np.float32),
        head=D.get("heading", np.full(int(m.sum()), np.nan))[m].astype(np.float32),
        medEDR=D["medEDR"][m].astype(np.float32) if "medEDR" in D else
        np.full(int(m.sum()), np.nan, np.float32),
        maxEDR=D["maxEDR"][m].astype(np.float32) if "maxEDR" in D else
        np.full(int(m.sum()), np.nan, np.float32),
        q=D["waterVaporMR"][m].astype(np.float32) if "waterVaporMR" in D else
        np.full(int(m.sum()), np.nan, np.float32),
        n_all=int(lat.size))
    return rec, m.sum()


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(GZDIR, exist_ok=True)
    os.makedirs("data/madis", exist_ok=True)
    t0 = time.time()
    summary = {}
    for dte in DAYS:
        out = os.path.join(OUT, "%s.npz" % dte)
        got = []
        tasks = [(dte, h) for h in range(24)]
        if os.path.exists(out):
            z = np.load(out, allow_pickle=True)
            if z["time"].size > 0:
                print("%s 已存在 (%d 条)" % (dte, z["time"].size), flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(pull_hour, d, h): (d, h) for d, h in tasks}
            for fu in as_completed(futs):
                d, h = futs[fu]
                try:
                    rec, info = fu.result()
                except Exception as e:
                    print("  %s %02dz ERR %s" % (d, h, str(e)[:90]), flush=True)
                    continue
                if rec is None:
                    continue
                got.append((h, rec))
        if not got:
            print("%s 无数据" % dte, flush=True)
            continue
        got.sort()
        keys = list(got[0][1].keys())
        merged = {}
        for k in keys:
            arrs = [g[1][k] for g in got]
            if k == "n_all":
                merged[k] = np.array([a for a in arrs], dtype=np.int64)
            else:
                merged[k] = np.concatenate(arrs)
        np.savez_compressed(out, **merged)
        ne = int(np.nansum(merged["medEDR"] > 0))
        nq = int(np.nansum(np.isfinite(merged["q"])))
        summary[dte] = dict(n=int(merged["time"].size), n_edr_pos=ne, n_wvss=nq)
        print("%s  记录 %7d   EDR>0 %6d   WVSS-II %6d   %.0fs"
              % (dte, merged["time"].size, ne, nq, time.time() - t0), flush=True)
    with open(os.path.join(OUT, "_summary.json"), "w", encoding="utf8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("总用时 %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
