# -*- coding: utf-8 -*-
"""step37 —— 下载 ERA5 独立系统数据，用于谱分解对照（step37b 消费）。

科学目标
--------
命题 P（谱—诊断对应原理）目前只在单一 GFS 系统（OSSE，4 时次东亚）上验证。
本脚本下载 ECMWF IFS 再分析（ERA5），供 step37b 检验三条核心机制
（谱质心 k_c / 导数放大 / 算子准则方向）是否为**同化系统的一般性质**。

配额最小化设计（CDS 有并发与队列限制）
--------------------------------------
1. **变量只取 5 个**：temperature / specific_humidity / u_component_of_wind /
   v_component_of_wind / geopotential。诊断 Ri_dry 与 N² 只需要这五个
   （相对湿度、云水、垂直速度、涡度、散度、位涡、臭氧一律不需要，
   详见脚本末尾 NOTES 的说明）。
2. **层次与 GFS 完全对齐的 21 层**：GFS 为 1000–100 hPa 共 21 层，ERA5 的
   37 个标准等压面完整包含这 21 层，故交集 = 21 层，双方无需裁剪即可
   同口径比较垂直导数。若配额紧张，可把 LEVELS 换成 LEVELS_MINIMAL（13 层），
   数据量约降至 60%，代价是两侧垂直层次需裁剪到 13 层。
3. **1 个请求覆盖所有时次**：CDS 请求对 year×month×day×time 做笛卡尔积，
   因此把 day 设为列表、time 只留一个时刻，可在**单个请求**里拿到 4 个时次。
   这比 16 次单独请求省 16 倍队列等待。
   时刻精确对齐（GFS 用的是 00/12/18Z 混合）会让笛卡尔积膨胀到 4×3=12 个
   组合，故放弃时刻对齐——谱结构检验不依赖与 GFS 同日同时刻。
4. **年份取 2024**：2026-10 尚未进入 ERA5 完整版（滞后数月，近实时由 ERA5T
   覆盖且后续会被重算），而本检验是统计性质、不要求与 GFS 同日。
   取 2024 年 10 月：数据已稳定，且季节与 GFS 算例一致（同为秋季）。
   若需跨季节稳健性，把 MONTHS 改为 ["01","10"] 即可（请求数 +1）。

网页表单（CDS Download tab）对照
-------------------------------
| 表单项            | 选择                                             |
|-------------------|--------------------------------------------------|
| Product type      | Reanalysis                                        |
| Variable          | 见 VARS（5 个）                                    |
| Pressure level    | 见 LEVELS（21 层，与 GFS 一致）                     |
| Year / Month / Day| 2024 / 10 / 01,08,12,16                            |
| Time              | 00:00 只勾一个                                     |
| Geographical area | **Area selection**（不是 Whole available region）  |
|                   | North=40, West=100, South=15, East=130            |
| Data format       | **NetCDF4 (Experimental)**（不要 GRIB）            |
| Download format   | **Unarchived**（若返回 zip 脚本会自动解压）         |

注意："Data format" 必须选 NetCDF4 —— 本脚本与 step37b 用 netCDF4 库读取；
scipy.io.netcdf_file 只能读 NetCDF3，读不了 CDS 现在的输出。

数据源：CDS（Copernicus Climate Data Store），需 API key（~/.cdsapirc）。
下载前必须先在网页接受数据集许可，否则返回 403 required licences not accepted。
"""
import os
import sys
import time

import cdsapi

sys.path.insert(0, ".")
import step12_enoi as S12

OUT = "data/era5"
# CDS 的 area 参数顺序为 [North, West, South, East]（注意不是 [N,W,E,S]）。
# 这里 = 北 40 / 西 100 / 南 15 / 东 130，即 100–130°E, 15–40°N（东亚，与 GFS 一致）。
# 网页表单对应 "Area selection"：North=40, West=100, South=15, East=130。
BBOX = [40.0, 100.0, 15.0, 130.0]

# ---- 层次：与 GFS 的 21 层完全一致（ERA5 标准等压面 37 层包含它们）----
LEVELS = ["1000", "975", "950", "925", "900", "850", "800", "750", "700",
          "650", "600", "550", "500", "450", "400", "350", "300", "250",
          "200", "150", "100"]
# ---- 兜底：配额紧张时改用 13 层（覆盖 200–400 hPa 巡航带仍足够密）----
LEVELS_MINIMAL = ["1000", "925", "850", "800", "750", "700", "600", "500",
                  "450", "400", "350", "300", "250", "200", "150", "100"]

# ---- 变量：仅需 5 个 ----
VARS = ["temperature", "specific_humidity", "u_component_of_wind",
        "v_component_of_wind", "geopotential"]

# ---- 时间：2024 年 10 月，4 天 × 单时刻，合并成 1 个请求 ----
YEAR = "2024"
MONTHS = ["10"]
DAYS = ["01", "08", "12", "16"]      # 对齐 GFS 算例的日期
HOURS = ["00:00"]                    # 单时刻：避免笛卡尔积膨胀

DATASET = "reanalysis-era5-pressure-levels"
LICENCE_URL = ("https://cds.climate.copernicus.eu/datasets/"
               "reanalysis-era5-pressure-levels?tab=download#manage-licences")


def estimate_mb(levels, n_time):
    """粗估下载体积（MB）。

    ERA5 CDS 的 netcdf 以 int16 + scale_factor 打包，这里按 2 bytes/点估算，
    再按 netcdf 内部 deflate 压缩折半。实测这个配置总量只有几 MB——
    **真正的瓶颈不是流量，而是并发请求数（队列等待）**，所以优化方向
    是"合并请求数"而非"削减数据量"。
    """
    nlon, nlat = 121, 101            # 0.25° over 100–130E / 15–40N
    return nlon * nlat * len(levels) * len(VARS) * n_time * 2 / 2.0 / 1e6


def maybe_unzip(path):
    """CDS 在单次请求产生多个文件时会打包成 zip；是 zip 则解压出 .nc。"""
    try:
        with open(path, "rb") as f:
            head = f.read(2)
    except OSError:
        return [path]
    if head != b"PK":
        return [path]
    import zipfile
    d = os.path.dirname(path) or "."
    out = []
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.lower().endswith(".nc"):
                z.extract(n, d)
                out.append(os.path.join(d, n))
    print("    解压 zip → %d 个 nc" % len(out))
    return out or [path]


def probe(c):
    """最小探测请求：确认许可已接受、目标时段可用。"""
    try:
        c.retrieve(DATASET, {
            "product_type": ["reanalysis"],
            "variable": ["temperature"],
            "pressure_level": ["250"],
            "year": [YEAR], "month": [MONTHS[0]], "day": [DAYS[0]],
            "time": [HOURS[0]],
            "area": BBOX, "format": "netcdf", "download_format": "unarchived",
        }, os.path.join(OUT, "_probe_avail.nc"))
        return True, "可用"
    except Exception as e:
        return False, str(e)


def main():
    os.makedirs(OUT, exist_ok=True)
    levels = LEVELS
    n_time = len(DAYS) * len(HOURS)
    print("=" * 78)
    print("step37  下载 ERA5（ECMWF IFS）等压面场")
    print("=" * 78)
    print("  变量 %d 个：%s" % (len(VARS), ", ".join(VARS)))
    print("  层次 %d 层：%s … %s" % (len(levels), levels[0], levels[-1]))
    print("  区域 N/W/E/S = %s" % BBOX)
    print("  时间 %s-%s 日 %s  → %d 个时次，合并为 %d 个 CDS 请求"
          % (YEAR, "/".join(MONTHS), "/".join(DAYS), n_time, len(MONTHS)))
    print("  预估体积 ≈ %.0f MB" % estimate_mb(levels, n_time))

    c = cdsapi.Client()                       # 读取 ~/.cdsapirc
    ok, msg = probe(c)
    if not ok:
        if "licence" in msg.lower():
            print("\n[中止] 尚未在 CDS 网页接受数据集许可。请登录同一账号后访问：")
            print("  " + LICENCE_URL)
            print("  接受 'Licence to use Copernicus Products' 后重跑本脚本。")
            return 1
        print("\n[中止] 探测请求失败：%s" % msg[:300])
        return 1
    print("  探测通过：%s-%s 可用" % (YEAR, MONTHS[0]))

    t0 = time.time()
    nreq = 0
    for m in MONTHS:
        dst = os.path.join(OUT, "era5_pl_%s%s.nc" % (YEAR, m))
        if os.path.exists(dst) and os.path.getsize(dst) > 100000:
            print("  [缓存] %s 已存在，跳过" % os.path.basename(dst))
            continue
        req = {
            "product_type": ["reanalysis"],
            "variable": VARS,
            "pressure_level": levels,
            "year": [YEAR], "month": [m], "day": DAYS, "time": HOURS,
            "area": BBOX, "format": "netcdf", "download_format": "unarchived",
        }
        print("  请求 %s-%s（%d 时次）… " % (YEAR, m, n_time), flush=True)
        try:
            c.retrieve(DATASET, req, dst)
            nreq += 1
            got = maybe_unzip(dst)
            print("    → %s  %.1f MB  %.0fs"
                  % (", ".join(os.path.basename(g) for g in got),
                     sum(os.path.getsize(g) for g in got) / 1e6,
                     time.time() - t0), flush=True)
        except Exception as e:
            print("    [失败] %s" % str(e)[:200], flush=True)

    # 按月文件 → 按时次拆分，供 step37b 逐个消费
    try:
        from step37_split_era5 import split_month_file
        split_month_file()
    except Exception as e:
        print("  （拆分步骤跳过：%s）" % str(e)[:120])
    print("完成：%d 个请求，用时 %.0f s" % (nreq, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ======================================================================
# NOTES：为什么只下这 5 个变量
# ======================================================================
# 诊断链路（avwx/risk.py::stability_fields）实际读取的只有 T, q, u, v, z：
#   - T, q  → 虚温 T_v → 静力稳定度 N²（干/湿两套）
#   - u, v  → 垂直切变 S² = (∂u/∂z)² + (∂v/∂z)²
#   - z     → 垂直坐标（层厚由虚温廓线积分得到，非均匀层厚必须用）
#   - Ri_dry = N_d² / S²；Ellrod TI 只用水平风的形变与切变
#
# 下列变量**均不需要**，不要为了"以防万一"下载（每个变量直接放大体积与队列时间）：
#   Divergence / Vorticity (relative)   —— 是诊断的"输出家族"，不是输入
#   Potential vorticity                 —— 同上；PV 若要做跨诊断泛化需另立实验
#   Vertical velocity (omega)           —— 本项目的湍流诊断不使用
#   Relative humidity                   —— 由 T, q 直接算出（诊断已输出 RH）
#   Specific cloud ice/liquid water     —— Ri_dry 面向晴空湍流区
#   Specific rain/snow water content    —— 同上
#   Fraction of cloud cover             —— 同上
#   Ozone mass mixing ratio             —— 与低空飞行气象风险无关
#   "Clear all"                         —— 界面按钮，不是变量
#
# 年份：2026-10 尚未进入 ERA5 完整版；且本检验为统计性质（谱质心 / 放大倍数 /
# 算子方向），不要求与 GFS 算例同日。取 2024 年 10 月：数据已稳定、季节一致。
# ======================================================================
