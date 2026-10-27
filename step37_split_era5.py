# -*- coding: utf-8 -*-
"""把 step37 下载的 ERA5 月度合并文件拆成单时次文件，供 step37b 逐个消费。

step37 为节省 CDS 配额，把 4 个时次合并进 1 个请求，落盘为
`era5_pl_YYYYMM.nc`（时间维长度 = 4）。step37b 的 glob 是
`era5_pl_*.nc` 并逐个 load_era5（内部 np.squeeze 去掉 size-1 的 time 维），
因此先按时间维切成 `era5_pl_YYYYMMDD_HHMM.nc`。

依赖：scipy（netcdf3 读写），与项目其余脚本一致。
"""
import glob
import os
from datetime import datetime, timedelta

import numpy as np
from scipy.io import netcdf_file

OUT = "data/era5"
EPOCH = datetime(1900, 1, 1)      # ERA5 netcdf 常用 "hours since 1900-01-01"


def _stamp(tval, i):
    """把时间坐标值转成 YYYYMMDD_HHMM 字符串；失败则退回序号。"""
    try:
        t = float(np.asarray(tval)[i])
        dt = EPOCH + timedelta(hours=t)
        if 1950 < dt.year < 2100:
            return dt.strftime("%Y%m%d_%H%M")
    except Exception:
        pass
    return "t%02d" % i


def _find_time_dim(nc):
    """定位时间维：名字含 time 且长度 >1 的维度；退化时取第 0 维。"""
    dims = list(getattr(nc, "dimensions", {}).keys())
    varnames = list(nc.variables.keys())
    for d in dims:
        if "time" in d.lower():
            for v in varnames:
                if v.lower() in ("valid_time", "time", "date") \
                        and nc.variables[v].dimensions == (d,):
                    arr = np.asarray(nc.variables[v][:])
                    if arr.ndim == 1 and arr.size > 1:
                        return v, arr
    # 退而求其次：找任一变量的首维长度 >1
    for v in varnames:
        vd = nc.variables[v].dimensions
        if len(vd) >= 1:
            a = np.asarray(nc.variables[v][:])
            if a.ndim >= 1 and a.shape[0] > 1:
                return v, np.arange(a.shape[0])
    return None, None


def split_month_file():
    months = sorted(glob.glob(os.path.join(OUT, "era5_pl_[0-9][0-9][0-9][0-9]"
                                           "[0-9][0-9].nc")))
    months = [f for f in months if "_probe" not in f]
    made = []
    for mp in months:
        nc = netcdf_file(mp, "r", mmap=False, maskandscale=True)
        tname, tval = _find_time_dim(nc)
        if tname is None:
            # 单时次文件，直接改名即可
            nc.close()
            continue
        n = len(tval)
        base = os.path.basename(mp).replace(".nc", "")
        for i in range(n):
            dst = os.path.join(OUT, "%s_%s.nc" % (base, _stamp(tval, i)))
            if os.path.exists(dst) and os.path.getsize(dst) > 10000:
                made.append(dst)
                continue
            w = netcdf_file(dst, "w")
            # 复制维度
            for d, ln in nc.dimensions.items():
                if d == nc.variables[tname].dimensions[0]:
                    w.createDimension(d, 1)
                else:
                    w.createDimension(d, ln)
            for v, var in nc.variables.items():
                a = np.asarray(var[:])
                if var.dimensions and var.dimensions[0] == \
                        nc.variables[tname].dimensions[0]:
                    a = a[i:i + 1]
                nv = w.createVariable(v, a.dtype.char, var.dimensions)
                for attr in getattr(var, "_attributes", {}):
                    try:
                        setattr(nv, attr, var._attributes[attr])
                    except Exception:
                        pass
                nv[:] = a
            w.close()
            made.append(dst)
        nc.close()
    print("  拆分出 %d 个单时次文件" % len(made))
    return made


if __name__ == "__main__":
    split_month_file()
