# -*- coding: utf-8 -*-
"""为真实误差 OSSE 下载更长的预报时效作为背景场。

选择依据（实测，东亚 20261008 12z，地面以上）：
    f006 : T 0.145 K, u 0.40 m/s   ← **小于**观测误差 ⇒ K≈0.08，无法检验
    f012 / f024 : 误差增大到与业务背景误差相当（NMC 法惯例用 24h 减 12h）
"""
import datetime as dt
import os
import sys
import time

sys.path.insert(0, ".")
from avwx import aws_gfs as AG

TRUTH = [("20261001", "00"), ("20261008", "12"),
         ("20261012", "18"), ("20261016", "00")]
LEADS = (24, 12)

t0 = time.time()
for fhr in LEADS:
    for dte, cyc in TRUTH:
        t = dt.datetime.strptime(dte + cyc, "%Y%m%d%H") - dt.timedelta(hours=fhr)
        try:
            p = AG.fetch(date=t.strftime("%Y%m%d"), cycle=t.strftime("%H"),
                         fhr=fhr, levels=AG.LEVELS_NATIVE)
            print("OK  %s %sz <- %s f%03d  %.1f MB"
                  % (dte, cyc, os.path.basename(p), fhr,
                     os.path.getsize(p) / 1e6), flush=True)
        except Exception as e:
            print("FAIL %s %sz lead %d : %s" % (dte, cyc, fhr, str(e)[:100]),
                  flush=True)
print("总用时 %.0f s" % (time.time() - t0))
