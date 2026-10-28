# -*- coding: utf-8 -*-
"""step39 —— 把真实误差 OSSE 的检验样本从 4 时次扩到 >=20 时次。

背景：REASSESS_v7 裁定"唯一规模短板 = 时次 4 -> >=20"。本脚本补齐缺失的
**背景场**（f012/f024 预报），供 step10 分析脚本扩展 TRUTH_TIMES 后复用。

事实（2026-10-28 核查）：
- f000 分析场（真值）已齐：22 个日期 x 4 循环（1 月/7 月/10 月，跨 2022-2026）。
- f012/f024 背景场（预报误差来源）严重不足，仅 2022 年零星 + 2026-10 少量。
- 磁盘仅剩 ~8.25 GB；每个背景场 ~80-90 MB（HTTP Range 只抽 L21 层）。

策略（磁盘友好）：
- 只下载 f024 背景（24h 预报误差，NMC 惯例，误差幅度大、检验功效高），
  跳过 f012（可省一半磁盘；且 step10 注释已说明 f006 误差太小无法检验）。
- 真值时次用 20 个（覆盖冬/夏/秋，标定 1 个 + 检验 19 个），
  与 step10 现有 4 个 10 月时次合并，最终 >=20。
- 每个真值时次只需 1 个 f024 背景 = 起报时刻 (真值 - 24h)。
"""
import datetime as dt
import os
import sys
import time

sys.path.insert(0, ".")
from avwx import aws_gfs as AG

# 真值时次：跨季 20 个（含 step10 原有 4 个 10 月时次）
# 冬季 1 月 / 夏季 7 月 / 秋季 10 月；每日取 00/12 两个循环控制磁盘与样本独立性
TRUTH_TIMES = [
    # 冬（2022/2023/2024 各取 1 天 x 2 循环）
    ("20220115", "00"), ("20220115", "12"),
    ("20230115", "00"), ("20230115", "12"),
    ("20240115", "00"), ("20240115", "12"),
    # 夏（2022/2023/2024 各取 1 天 x 2 循环）
    ("20220715", "00"), ("20220715", "12"),
    ("20230715", "00"), ("20230715", "12"),
    ("20240715", "00"), ("20240715", "12"),
    # 秋（2026-10，与 step10 原有 4 个合并到 >=20）
    ("20261001", "00"), ("20261008", "12"),
    ("20261012", "18"), ("20261016", "00"),
    # 补充 2026 冬/夏各 2 个，凑满 20
    ("20260115", "00"), ("20260115", "12"),
    ("20260715", "00"), ("20260715", "12"),
]

LEAD = 24  # 只用 24h 预报误差（NMC 惯例，功效最高、磁盘减半）

t0 = time.time()
n_ok = n_skip = n_fail = 0
for dte, cyc in TRUTH_TIMES:
    # 背景 = f024 起报时刻 = 真值 - 24h
    bg = dt.datetime.strptime(dte + cyc, "%Y%m%d%H") - dt.timedelta(hours=LEAD)
    bd, bc = bg.strftime("%Y%m%d"), bg.strftime("%H")
    out = "data/aws/gfs_%s_%s_f%03d_L21_1000_100.grib2" % (bd, bc, LEAD)
    if os.path.exists(out) and os.path.getsize(out) > 1e6:
        n_skip += 1
        print("SKIP %s %sZ f%03d（已有）" % (bd, bc, LEAD), flush=True)
        continue
    try:
        AG.fetch(date=bd, cycle=bc, fhr=LEAD, levels=AG.LEVELS_NATIVE)
        n_ok += 1
        print("OK   %s %sZ f%03d -> 真值 %s %sZ  (%.1f MB)"
              % (bd, bc, LEAD, dte, cyc, os.path.getsize(out) / 1e6), flush=True)
    except Exception as e:
        n_fail += 1
        print("FAIL %s %sZ f%03d : %s" % (bd, bc, LEAD, str(e)[:100]), flush=True)

print("\n完成：新增 %d，已有 %d，失败 %d，用时 %.0f s"
      % (n_ok, n_skip, n_fail, time.time() - t0))
print("注：真值时次共 %d 个；如需 >=20 个且失败较多，可增补循环或日期。" % len(TRUTH_TIMES))
