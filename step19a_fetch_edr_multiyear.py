# -*- coding: utf-8 -*-
"""step19a —— 跨年 EDR 取数（复验的复验）。

为什么需要它
------------
审稿人 A1：本项目的观测证据只来自**一年**（2026）的 14 天。
而 §12.5 的年代际气候学刚刚证明 δ 在巡航层**不是静止的**
（夏季 250 hPa 47 年 +33%，τ=0.45）。既然 δ 会漂，
那么"H2 的惩罚幅度"会跟着漂吗？在一年的样本上回答不了。

本脚本不改 step11a 的任何逻辑，只是把 DAYS 换成 2022/2023/2024 的
**相同日历日**（1 月 15/16、7 月 15/16），保证季节构成与 2026 基准一致
⇒ 年份之间是唯一变量。

产出落到同一个 `data/madis/edr/`（step11b 用 glob 读取，会自然合并），
年份标签由样本自带的时间戳在下游切分。

⚠ 踩坑：step11a.main() 结束时会把 `OUT/_summary.json` **整个覆盖**，
只写本次 DAYS 的统计。所以本脚本先把旧的备份出来，跑完再合并回去。
"""
import json
import os
import shutil
import sys

sys.path.insert(0, ".")
import step11a_fetch_edr as S11A

# 与 2026 基准**同日历日**：20260115/16、20260715/16 ⇒ 冬/夏各两天。
YEARS = (2022, 2023, 2024)
MMDD = ("0115", "0116", "0715", "0716")
DAYS = ["%d%s" % (y, md) for y in YEARS for md in MMDD]

OUT = S11A.OUT
SUMMARY = os.path.join(OUT, "_summary.json")
BAK = os.path.join(OUT, "_summary_before_multiyear.json")


def main():
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(SUMMARY) and not os.path.exists(BAK):
        shutil.copyfile(SUMMARY, BAK)
        print("已备份旧的 _summary.json → %s" % BAK, flush=True)
    S11A.DAYS = DAYS
    print("跨年取数 %d 天：%s" % (len(DAYS), ", ".join(DAYS)), flush=True)
    S11A.main()

    # 合并：旧统计 + 本次统计
    merged = {}
    if os.path.exists(BAK):
        with open(BAK, encoding="utf8") as fh:
            merged.update(json.load(fh))
    if os.path.exists(SUMMARY):
        with open(SUMMARY, encoding="utf8") as fh:
            merged.update(json.load(fh))
    with open(SUMMARY, "w", encoding="utf8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=1)
    tot = sum(v.get("n", 0) for v in merged.values())
    pos = sum(v.get("n_edr_pos", 0) for v in merged.values())
    print("\n合并后 _summary.json：%d 天，记录 %d，EDR>0 %d"
          % (len(merged), tot, pos), flush=True)


if __name__ == "__main__":
    main()
