# -*- coding: utf-8 -*-
"""step19b —— 跨年 EDR 检验的配套 GFS 分析（2022/2023/2024）。

与 step11c 完全相同的取数逻辑，只是把日期换成 step19a 的同一批日子。
每天 4 个循环（00/06/12/18），保证与 EDR 的时间错配 ≤ 1.5 h。

⚠ 磁盘预算：每个 f000 经 HTTP Range 只取需要的消息，约 80–90 MB
  （整份 pgrb2.0p25.f000 约 500 MB）。12 天 × 4 时次 ≈ 4 GB。
  开工前务必确认 D: 有 ≥ 8 GB 余量。
"""
import os
import sys
import time

sys.path.insert(0, ".")
from avwx import aws_gfs as AG

YEARS = (2022, 2023, 2024)
MMDD = ("0115", "0116", "0715", "0716")
DAYS = ["%d%s" % (y, md) for y in YEARS for md in MMDD]

TARGETS = [(d, c) for d in DAYS for c in ("00", "06", "12", "18")]

t0 = time.time()
n_ok = n_skip = n_fail = 0
for dte, cyc in TARGETS:
    p = "data/aws/gfs_%s_%s_f000_L21_1000_100.grib2" % (dte, cyc)
    if os.path.exists(p) and os.path.getsize(p) > 1e6:
        n_skip += 1
        continue
    try:
        AG.fetch(date=dte, cycle=cyc, fhr=0, levels=AG.LEVELS_NATIVE)
        n_ok += 1
        print("OK  %s %sz   (%.1f MB)  %.0fs"
              % (dte, cyc, os.path.getsize(p) / 1e6, time.time() - t0),
              flush=True)
    except Exception as e:
        n_fail += 1
        print("FAIL %s %sz : %s" % (dte, cyc, str(e)[:80]), flush=True)
print("完成：新增 %d，跳过 %d，失败 %d，用时 %.0f s"
      % (n_ok, n_skip, n_fail, time.time() - t0), flush=True)
