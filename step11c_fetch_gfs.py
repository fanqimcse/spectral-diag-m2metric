# -*- coding: utf-8 -*-
"""step11c —— 只为 EDR 检验补齐配套 GFS 分析时次。

理由：step11b 的时间匹配窗口只有 ±1.5 h，而现有本地只有 15 个 f000 分析，
导致 96 万条 EDR 观测里只有约 1 万条能用上。补齐每天 00/06/12/18 四个循环后，
同一批观测的可配对数将翻倍以上，且时间错配保持在窗口内。

只下载 f000（分析），不下载 f012/f024 —— 那两组是 step10 扩大 OSSE 样本用的，
优先级低于本任务，等带宽和磁盘有余再补。
"""
import os
import sys
import time

sys.path.insert(0, ".")
from avwx import aws_gfs as AG

TARGETS = []
for d in ("20260115", "20260116", "20260715", "20260716",
          "20261004", "20261008", "20261012", "20261016", "20261019"):
    for c in ("00", "06", "12", "18"):
        TARGETS.append((d, c))

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
      % (n_ok, n_skip, n_fail, time.time() - t0))
