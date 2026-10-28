# -*- coding: utf-8 -*-
"""step40 —— 真实误差 OSSE 扩时次（4 → 20），跨季稳健性验证。

背景：REASSESS_v7 裁定"唯一规模短板 = 时次 4 → >=20"。step39 已补齐 20 个
跨季时次的 f024 背景场。本脚本复用 step10 的全部函数，仅替换 TRUTH_TIMES
与 LEADS，输出到**独立 json**（不覆盖 step10 原 4 时次结果）。

关键设计：
- 真值 = GFS f000 分析；背景 = GFS f024 预报（24h 真实预报误差，NMC 惯例）。
- 20 个时次跨冬(1月)/夏(7月)/秋(10月)、跨 2022-2026；第 1 个标定，19 个检验。
- 只用 f024（step39 只下了 f024；且 24h 误差幅度大、检验功效最高）。
- 输出 data/step40_scaleup_osse.json，与 step10 结果并存，供报告对照。
"""
import json
import sys
import time

import step10_realerror_osse as S10

# 20 个跨季时次（第 1 个用于标定，其余 19 个独立检验）
TRUTH_TIMES = [
    # 冬（1月）
    ("20220115", "00"), ("20220115", "12"),
    ("20230115", "00"), ("20230115", "12"),
    ("20240115", "00"), ("20240115", "12"),
    ("20260115", "00"), ("20260115", "12"),
    # 夏（7月）
    ("20220715", "00"), ("20220715", "12"),
    ("20230715", "00"), ("20230715", "12"),
    ("20240715", "00"), ("20240715", "12"),
    ("20260715", "00"), ("20260715", "12"),
    # 秋（10月，与 step10 原有 4 个对齐以便交叉核对）
    ("20261001", "00"), ("20261008", "12"),
    ("20261012", "18"), ("20261016", "00"),
]
LEADS = (24,)

# 复用 step10 的函数与常量，替换时次与时效
S10.TRUTH_TIMES = TRUTH_TIMES
S10.LEADS = LEADS


def main():
    t0 = time.time()
    print("=" * 92)
    print("step40  真实预报误差 OSSE 扩时次：20 个跨季时次，背景 = GFS f024")
    print("        标定 1 时次 + 检验 19 时次；输出独立 json，不覆盖 step10")
    print("=" * 92, flush=True)

    res = {}
    for ld in LEADS:
        try:
            res[ld] = S10.run_lead(ld)
        except Exception:
            import traceback
            traceback.print_exc()
            print("f%03d 失败" % ld, flush=True)

    print("\n" + "#" * 92)
    print("#  跨季汇总（20 时次，f024）：湿度信息提取（相对背景 q RMSE 百分点）")
    print("#" * 92)
    for ld in LEADS:
        if ld not in res:
            continue
        a = res[ld]["attribution"]
        bg = float(__import__("numpy").median([r["bg|q"] for r in res[ld]["rows"]]))
        print("  f%03d  n=%d  bg q=%.3g  V2q耦合=%+.3f%%  [%+5.2f,%+5.2f]  "
              "直报=%+.3f%%  耦合/直报=%.1f%%"
              % (ld, a["n_times"], bg * 1e3, a["gain_pp"],
                 a["ci_gain"][0], a["ci_gain"][1], a["direct_pp"],
                 a["coupling_over_direct"]))

    out = dict(meta=dict(bbox=list(S10.BBOX), leads=list(LEADS),
                         truth=[list(t) for t in TRUTH_TIMES],
                         note="scaleup OSSE: truth=f000, bg=f024 forecast, 20 seasonal times"),
               results={str(ld): dict(calibration=res[ld]["calibration"],
                                      attribution=res[ld]["attribution"],
                                      rows=res[ld]["rows"])
                        for ld in res})
    with open("data/step40_scaleup_osse.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step40_scaleup_osse.json   用时 %.0f s"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
