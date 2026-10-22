# -*- coding: utf-8 -*-
"""step10d —— 把真实误差 OSSE 的检验样本从 3 个时次扩到跨四季。

背景：审稿人必然会问"3 个时次的中位数有什么统计功效"。本脚本把样本扩大到
18 个真值时次 × 2 个预报时效，覆盖 1 月（冬）/ 7 月（夏）/ 10 月（秋），
每日 00/06/12/18 四个循环各取一部分，且**标定与检验严格分离**。

之所以敢扩：实测 AWS noaa-gfs-bdp-pds 上 2026-01 与 2026-07 的分析仍在线，
GFS 存档保留期远超预期（>9 个月）。
"""
import datetime as dt
import os
import sys
import time

sys.path.insert(0, ".")
from avwx import aws_gfs as AG

# 真值时刻：1 月(冬) / 7 月(夏) / 10 月(秋)，四个业务循环
TRUTH = []
for d in ("20260115", "20260116", "20260715", "20260716",
          "20261004", "20261008", "20261012", "20261016", "20261019"):
    for c in ("00", "06", "12", "18"):
        TRUTH.append((d, c))
# 已有的 + 需要补齐的；标定只用第一个，其余全部进入检验
LEADS = (24, 12)

t0 = time.time()
n_ok = n_skip = 0
for fhr in LEADS:
    for dte, cyc in TRUTH:
        p0716 = "data/aws/gfs_%s_%s_f000_L21_1000_100.grib2" % (dte, cyc)
        dstbg = "data/aws/gfs_%s_%s_f%03d_L21_1000_100.grib2" % (
            (dt.datetime.strptime(dte + cyc, "%Y%m%d%H") -
             dt.timedelta(hours=fhr)).strftime("%Y%m%d"),
            (dt.datetime.strptime(dte + cyc, "%Y%m%d%H") -
             dt.timedelta(hours=fhr)).strftime("%H"), fhr)
        if os.path.exists(dstbg):
            n_skip += 1
            continue
        tbac = dt.datetime.strptime(dte + cyc, "%Y%m%d%H") - dt.timedelta(hours=fhr)
        try:
            AG.fetch(date=tbac.strftime("%Y%m%d"), cycle=tbac.strftime("%H"),
                     fhr=fhr, levels=AG.LEVELS_NATIVE)
            n_ok += 1
            print("OK  %s %sz <- f%03d %s" % (dte, cyc, fhr, os.path.basename(dstbg)),
                  flush=True)
        except Exception as e:
            print("FAIL %s %sz f%03d : %s" % (dte, cyc, fhr, str(e)[:90]), flush=True)
print("完成：新增 %d 个背景场，已存在跳过 %d，用时 %.0f s"
      % (n_ok, n_skip, time.time() - t0))

# 顺带补齐缺失的分析场（作为真值）
for dte, cyc in TRUTH:
    p = "data/aws/gfs_%s_%s_f000_L21_1000_100.grib2" % (dte, cyc)
    if os.path.exists(p):
        continue
    try:
        AG.fetch(date=dte, cycle=cyc, fhr=0, levels=AG.LEVELS_NATIVE)
        print("OK  truth  %s %sz" % (dte, cyc), flush=True)
    except Exception as e:
        print("FAIL truth %s %sz : %s" % (dte, cyc, str(e)[:90]), flush=True)
print("全部用时 %.0f s" % (time.time() - t0))
