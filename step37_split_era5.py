# -*- coding: utf-8 -*-
"""把 step37 下载的 ERA5 月度合并文件拆成单时次文件，供 step37b 逐个消费。

step37 为节省 CDS 队列等待，把 4 个时次合并进 1 个请求，落盘为
`era5_pl_YYYYMM.nc`（时间维长度 = 4）。step37b 的 glob 会挑出单时次文件，
因此先按时间维切成 `era5_pl_YYYYMM_YYYYMMDD_HHMM.nc`。

用 netCDF4 而非 scipy.io.netcdf_file：CDS 现在的 netcdf 输出是
NetCDF4/HDF5，scipy 的 netcdf_file 只支持 NetCDF3，会读失败。
"""
import glob
import os
from datetime import datetime, timedelta

import numpy as np
from netCDF4 import Dataset

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


def _arr(var):
    a = var[:]
    if hasattr(a, "filled"):
        a = a.filled(np.nan)
    return np.asarray(a)


def _find_time_dim(nc):
    """定位时间维：名字含 time 且长度 >1 的维度；退化时取第 0 维。"""
    for d in list(nc.dimensions.keys()):
        if "time" in d.lower():
            for v in nc.variables.keys():
                var = nc.variables[v]
                if var.dimensions == (d,):
                    arr = _arr(var)
                    if arr.ndim == 1 and arr.size > 1:
                        return d, arr
    for v in nc.variables.keys():
        var = nc.variables[v]
        if len(var.dimensions) >= 1:
            a = _arr(var)
            if a.ndim >= 1 and a.shape[0] > 1:
                return var.dimensions[0], np.arange(a.shape[0])
    return None, None


def split_month_file():
    months = sorted(glob.glob(os.path.join(OUT, "era5_pl_[0-9][0-9][0-9][0-9]"
                                           "[0-9][0-9].nc")))
    months = [f for f in months if "_probe" not in f]
    made = []
    for mp in months:
        nc = Dataset(mp, "r")
        tdim, tval = _find_time_dim(nc)
        if tdim is None:
            nc.close()
            continue
        n = len(tval)
        base = os.path.basename(mp).replace(".nc", "")
        for i in range(n):
            dst = os.path.join(OUT, "%s_%s.nc" % (base, _stamp(tval, i)))
            if os.path.exists(dst) and os.path.getsize(dst) > 10000:
                made.append(dst)
                continue
            w = Dataset(dst, "w")
            for d, dim in nc.dimensions.items():
                if d == tdim:
                    w.createDimension(d, 1)
                elif dim.isunlimited():
                    w.createDimension(d, None)
                else:
                    w.createDimension(d, len(dim))
            for name, var in nc.variables.items():
                a = var[:]
                if var.dimensions and var.dimensions[0] == tdim:
                    a = a[i:i + 1]
                fv = var.getncattr("_FillValue") \
                    if "_FillValue" in var.ncattrs() else None
                nv = w.createVariable(name, var.dtype, var.dimensions,
                                      fill_value=fv)
                atts = {k: var.getncattr(k) for k in var.ncattrs()
                        if k != "_FillValue"}
                if atts:
                    nv.setncatts(atts)
                nv[:] = a
            w.close()
            made.append(dst)
        nc.close()
    print("  拆分出 %d 个单时次文件" % len(made))
    return made


if __name__ == "__main__":
    split_month_file()
