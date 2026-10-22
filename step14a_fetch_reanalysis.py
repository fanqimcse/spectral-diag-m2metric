# -*- coding: utf-8 -*-
"""step14a —— 抓取 1979–present 再分析的区域子集，用于"业务层组失明"的年代际演变。

数据源：NCEP/DOE Reanalysis 2（正好从 1979 开始），NOAA PSL，匿名。
取数方式：**THREDDS 的 NCSS 子集服务**，而不是下载整年文件。

    整年日平均文件：air 116 MB / uwnd 170 MB / vwnd 174 MB …
    NCSS 区域子集（东亚 100–145E,15–45N，数年）：约 17 KB/天/变量

两者相差四个数量级。这一步决定了这个任务到底能不能做 —— 不做子集化，
47 年 × 5 个变量是 28 GB，做了是 ~150 MB。

采样设计：每年取 1 月与 7 月各 3 天（1/11/21 日），即 6 天/年。
    理由：δ 有强季节循环，只取单个季节会混淆"年代际趋势"与"季节相位"；
    冬夏各取一点可以同时给出两个季节各自的趋势与季节对比。
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from scipy.io import netcdf_file

BASE = ("https://psl.noaa.gov/thredds/ncss/grid/Datasets/"
        "ncep.reanalysis2.dailyavgs/pressure/%s.%d.nc")
VARS = ("air", "rhum", "uwnd", "vwnd", "hgt")
YEARS = list(range(1979, 2026))
# 本任务要回答的是"失明程度的**时间演变**"，不是区域差异（区域对比 step06 已做），
# 因此只取东亚一个区域，把预算全花在 47 年的连续年份上。
# 全取 3 区域需要 1410 个请求（实测 ≈3 小时）；单区域 470 个 ≈1 小时。
REGIONS = {
    "EA": dict(north=45, south=15, west=100, east=145),
}
OUT = "data/reanalysis"
DAYS = [("-01-01", "-01-22"), ("-07-01", "-07-22")]   # 1 月/7 月各约 3 周窗口


def fetch_one(var, year, reg, win):
    bb = REGIONS[reg]
    q = dict(var=var, accept="netcdf", **bb)
    q["time_start"] = "%d%sT00:00:00Z" % (year, win[0])
    q["time_end"] = "%d%sT00:00:00Z" % (year, win[1])
    dst = os.path.join(OUT, "%s_%s_%d_%s.nc" % (var, reg, year, win[0][1:3]))
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return dst, 0
    url = BASE % (var, year)
    for attempt in range(3):
        try:
            r = requests.get(url, params=q, timeout=240)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(dst, "wb") as f:
                    f.write(r.content)
                return dst, len(r.content)
        except Exception:
            pass
        time.sleep(2)
    return None, 0


def main():
    os.makedirs(OUT, exist_ok=True)
    tasks = [(v, y, r, w) for v in VARS for y in YEARS
             for r in REGIONS for w in DAYS]
    print("共 %d 个子集请求" % len(tasks), flush=True)
    t0 = time.time()
    n_ok = n_fail = 0
    nbytes = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_one, *t): t for t in tasks}
        for i, fu in enumerate(as_completed(futs), 1):
            dst, nb = fu.result()
            if dst:
                n_ok += 1
                nbytes += nb
            else:
                n_fail += 1
            if i % 50 == 0:
                print("  %d/%d  成功 %d 失败 %d  %.1f MB  %.0fs"
                      % (i, len(tasks), n_ok, n_fail, nbytes / 1e6,
                         time.time() - t0), flush=True)
    print("完成：成功 %d，失败 %d，合计 %.1f MB，用时 %.0f s"
          % (n_ok, n_fail, nbytes / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
