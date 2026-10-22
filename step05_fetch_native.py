# -*- coding: utf-8 -*-
"""批量抽取 GFS 原生 21 层数据（AWS 字节范围 + 并行）。

一次下载得到**全球**场，可同时切出东亚 / 北美 / 热带西太平洋三个区域，
这是"多区域气候学"得以成立的前提（NOMADS 子集服务必须按区域分别下载）。

分组：
  SEASON_TIMES —— 冬(1 月) / 夏(7 月) / 秋(10 月)，用于把 δ 与漏判率从
                  个例统计升级为季节-区域气候学；同时覆盖昼夜两个时次。
  PIREP_TIMES  —— 近 30 天内的北美分析时次，用于与 PIREP 颠簸报告做匹配检验
                  （AWC 数据库仅回溯 30 天，故必须落在 2026-09-21 之后）。
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.abspath("."))
from avwx import aws_gfs as AG

# (date, cycle)
SEASON_TIMES = [
    ("20260715", "06"), ("20260715", "18"),     # 北半球夏
    ("20260115", "06"), ("20260115", "18"),     # 北半球冬
    ("20261019", "06"), ("20261019", "18"),     # 秋（基准）
]

PIREP_TIMES = [
    ("20261001", "00"), ("20261001", "12"),
    ("20261004", "06"), ("20261004", "18"),
    ("20261008", "00"), ("20261008", "12"),
    ("20261012", "06"), ("20261012", "18"),
    ("20261016", "00"), ("20261016", "12"),
]

# PIREP 观测对照期（2026-10-08 ~ 2026-10-20，4 个分析时次/天）。
# 每个分析时次匹配其 ±1.5 h 内的 PIREP，故时次密度直接决定样本量。
VERIF_TIMES = []
for _d in range(8, 21):
    for _c in ("00", "06", "12", "18"):
        VERIF_TIMES.append(("2026%02d%02d" % (10, _d), _c))

GROUPS = {"season": SEASON_TIMES, "pirep": PIREP_TIMES, "verif": VERIF_TIMES}
GROUP = sys.argv[1] if len(sys.argv) > 1 else "season"
TIMES = GROUPS[GROUP]

print("=== 组 = %s   时次数 = %d ===" % (GROUP, len(TIMES)), flush=True)
ok, bad = [], []
for i, (d, c) in enumerate(TIMES, 1):
    try:
        p = AG.fetch(date=d, cycle=c, levels=AG.LEVELS_NATIVE,
                     variables=AG.VARS, surface_vars=AG.SFC_VARS,
                     nproc=8)
        ok.append((d, c, os.path.getsize(p)))
        print("[%d/%d] OK %s %sZ  %.1f MB" % (i, len(TIMES), d, c,
                                              os.path.getsize(p) / 1e6), flush=True)
    except Exception as e:
        bad.append((d, c))
        print("[%d/%d] FAIL %s %sZ  %r" % (i, len(TIMES), d, c, e), flush=True)
        traceback.print_exc()

print("\n=== 汇总 ===  成功 %d  失败 %d" % (len(ok), len(bad)))
for d, c, n in ok:
    print("   OK   %s %sZ  %.1f MB" % (d, c, n / 1e6))
for d, c in bad:
    print("   FAIL %s %sZ" % (d, c))
