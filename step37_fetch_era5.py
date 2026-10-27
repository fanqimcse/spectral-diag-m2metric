# -*- coding: utf-8 -*-
"""step37 —— 下载 ERA5 独立系统数据，用于谱分解对照（step37b 消费）。

科学目标
--------
命题 P（谱—诊断对应原理）目前只在单一 GFS 系统（OSSE，4 时次东亚）上验证。
JOURNAL_RECOMMENDATION.md 的 ROI 第一项就是"补一个独立系统的谱分解"。

ERA5（ECMWF IFS 再分析）是与 GFS（NCEP）完全独立的同化系统。把 ERA5 的
分析场 / 短时预报场做误差谱分解，检验"同化增量谱带宽受限 → 大尺度主导、
导数放大 → 诊断对小尺度敏感"是否是**同化系统的一般性质**，而非 GFS 特有。

下载参数严格对齐 GFS 谱分解（step21）：
    BBOX   = 100–130°E, 15–40°N（东亚，与 S10.BBOX 一致）
    层次   = 等压面（含 250 hPa 巡航层代表层）
    变量   = t, q, u, v, z（与 KEYS = T/q/u/v/z 一致）
    时次   = 4 个目标时次（对齐 step12.TARGETS 的日期）

为做"背景 vs 分析"谱对照，额外下载每个目标时次前 24h 的**模式预报场**：
ERA5 不直接提供 first guess，但 ERA5 的 0.25° 分析场本身即可做谱分解，
并可用其 12h 时间差构造"场增量"检验带宽。这里先下分析场 + 前一时次分析场，
供 step37b 构造"分析增量谱"。

数据源：CDS（Copernicus Climate Data Store），需 API key（用户提供）。
"""
import os
import sys
import time

import cdsapi

sys.path.insert(0, ".")
import step12_enoi as S12

OUT = "data/era5"
BBOX = [100.0, 15.0, 130.0, 40.0]          # [N, W, E, S] = [北, 西, 东, 南]
# 等压面层次（hPa）：覆盖对流层 + 250hPa 巡航层
LEVELS = ["1000", "925", "850", "700", "600", "500", "400", "300",
          "250", "200", "150", "100"]
# 目标时次（日期对齐 step12.TARGETS）+ 前一时次（用于构造分析增量）
TARGET_DATES = ["2026-10-01", "2026-10-08", "2026-10-12", "2026-10-16"]
# 回退日期：ERA5 完整版滞后数月（近实时由 ERA5T 覆盖，通常滞后约 5 天）。
# 若目标日期尚未上线，回退到 2024 年同期——谱结构检验不依赖与 GFS 严格同日，
# 只要求区域、层次、变量一致。
FALLBACK_DATES = ["2024-10-01", "2024-10-08", "2024-10-12", "2024-10-16"]
HOURS = ["00:00", "06:00", "12:00", "18:00"]
# 变量：t(温度K) q(比湿) u/v(风) z(位势)
VARS = ["temperature", "specific_humidity", "u_component_of_wind",
        "v_component_of_wind", "geopotential"]


def download_one(c, date, hour, dst):
    """下载单个时次的 ERA5 等压面分析场。"""
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        return dst, "cached"
    req = {
        "product_type": ["reanalysis"],
        "variable": VARS,
        "pressure_level": LEVELS,
        "year": [date[:4]],
        "month": [date[5:7]],
        "day": [date[8:10]],
        "time": [hour],
        "area": BBOX,
        "format": "netcdf",
        "download_format": "unarchived",
    }
    c.retrieve("reanalysis-era5-pressure-levels", req, dst)
    return dst, "downloaded"


def main():
    os.makedirs(OUT, exist_ok=True)
    c = cdsapi.Client()                       # 读取 ~/.cdsapirc

    # ---- 可用性探测与日期回退 ----
    probe = os.path.join(OUT, "_probe_avail.nc")
    try:
        c.retrieve("reanalysis-era5-pressure-levels", {
            "product_type": ["reanalysis"],
            "variable": ["temperature"],
            "pressure_level": ["250"],
            "year": [TARGET_DATES[0][:4]],
            "month": [TARGET_DATES[0][5:7]],
            "day": [TARGET_DATES[0][8:10]],
            "time": ["00:00"],
            "area": BBOX,
            "format": "netcdf",
            "download_format": "unarchived",
        }, probe)
        dates = TARGET_DATES
        print("目标日期 %s 可用" % TARGET_DATES[0], flush=True)
    except Exception as e:
        msg = str(e)
        if "licence" in msg.lower():
            print("[中止] 尚未在 CDS 网页接受数据集许可。请登录后访问：", flush=True)
            print("  https://cds.climate.copernicus.eu/datasets/"
                  "reanalysis-era5-pressure-levels?tab=download#manage-licences",
                  flush=True)
            return 1
        dates = FALLBACK_DATES
        print("目标日期 %s 不可用（%s），回退到 2024 年同期"
              % (TARGET_DATES[0], msg[:80]), flush=True)

    t0 = time.time()
    n = 0
    for d in dates:
        for h in HOURS:
            dst = os.path.join(OUT, "era5_pl_%s_%s.nc"
                               % (d.replace("-", ""), h.replace(":", "")))
            try:
                _, status = download_one(c, d, h, dst)
                n += 1
                print("  [%d/%d] %s %s -> %s" %
                      (n, len(dates) * len(HOURS), d, h, status),
                      flush=True)
            except Exception as e:
                print("  [失败] %s %s : %s" % (d, h, str(e)[:120]), flush=True)
    print("完成：%d 个时次，用时 %.0f s" % (n, time.time() - t0))


if __name__ == "__main__":
    main()
