# -*- coding: utf-8 -*-
"""step37b —— ERA5（ECMWF IFS）独立系统谱分解对照。

科学目标
--------
"谱—诊断对应原理"目前只在单一 GFS（NCEP）系统上验证，这是论文最主要的
规模/普适性短板。本脚本把三条核心机制推到**完全独立的 ECMWF IFS 系统**，
检验它们是"同化系统的一般性质"还是"GFS 特有"：

  [A] 谱结构   ：ERA5 各变量水平能量谱的谱质心 k_c，与 GFS 同法对比
  [B] 导数放大 ：同一统计特性的小尺度误差，分别落在 GFS 与 ERA5 场上，
                 诊断误差放大倍数是否一致
  [C] 算子准则 ：(a) 水平高斯 / (b) 水平锐利 / (c) 垂直高斯 三算子
                 在 ERA5 上的表现方向，是否与 GFS 一致

设计要点（为什么这样构造）
--------------------------
1. **ERA5 是再分析，无公开 first guess**，无法直接构造"分析增量谱"。
   因此采用"真值场 = 分析场、背景场 = 真值 + 可控小尺度误差"的口径：
   场的物理结构来自**真实独立系统**，误差扰动是统计可控的。
   这比 step36 [B] 的理想合成场强得多——那里连场都是合成的。

2. **同一代码路径、同一口径双跑 GFS 与 ERA5**。跨系统的绝对值比较没有
   意义（定义、网格、层次都不同），有意义的是**比值是否一致**。因此
   本脚本对 GFS 也重算一遍，作为同口径基准，而不是直接引用论文的 4.2×。

3. **误差场统计特性对齐**：两边都注入空间相关长度 30 km、幅度为场空间
   标准差 2% 的高斯噪声（step36 [B] 口径）。2% 保证诊断（Ri 为比值型）
   仍在线性响应区，放大倍数不受幅度影响。

4. **垂直层次取交集**：GFS 与 ERA5 的等压面层数不同，而垂直导数误差
   对 dlnp 敏感，故两边都裁剪到共有的层次集合再比较。
"""
import gc
import glob
import json
import os
import re
import sys
import time

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
import step12_enoi as S12
import step36_operator_sufficient_condition as S36
from m2metric import spectrum2d as m2_spectrum2d

OUT = "data/step37b_era5_spectral.json"
ERA5_DIR = "data/era5"
# 注意：S12.KEYS = ('T','q','u','v') 不含 z，但 stability_fields 需要 z 作为
# 垂直坐标（层厚由虚温廓线积分得到），故此处必须显式补上 'z'。
KEYS = tuple(S12.KEYS) + ("z",)
CRUISE = (200.0, 400.0)
NOISE_KM = 30.0        # 注入误差的空间相关长度（与 step36 [B] 一致）
NOISE_FRAC = 0.02      # 误差幅度 = 2% × 场空间标准差
SEED = 42
N_TIMES = 4            # 参与汇总的 ERA5 时次数（下载多少就用多少，上限 4）


# ======================================================================
# ERA5 读取
# ======================================================================
def _pick(nc, cands):
    for c in cands:
        if c in nc.variables:
            return c
    raise KeyError("找不到变量，候选=%s，实际=%s"
                   % (cands, list(nc.variables.keys())))


def _arr(var):
    """netCDF4 变量 -> float ndarray，缺失值统一转 nan。

    注意：CDS 现在的 'netcdf' 输出是 NetCDF4/HDF5，scipy.io.netcdf_file
    只能读 NetCDF3，故统一改用 netCDF4.Dataset。
    """
    a = var[:]
    if hasattr(a, "filled"):            # masked array -> nan
        a = a.filled(np.nan)
    return np.asarray(a, float)


def load_era5(path):
    """读一个 ERA5 等压面 netcdf（NetCDF3/4 均可），返回 (fld, lat, lon, lev)。"""
    nc = Dataset(path, "r")
    latn = _pick(nc, ["lat", "latitude"])
    lonn = _pick(nc, ["lon", "longitude"])
    levn = _pick(nc, ["level", "pressure_level", "plev"])
    lat = _arr(nc.variables[latn])
    lon = _arr(nc.variables[lonn])
    lev = _arr(nc.variables[levn])
    vmap = {"T": ["t", "temperature"], "q": ["q", "specific_humidity"],
            "u": ["u", "u_component_of_wind"],
            "v": ["v", "v_component_of_wind"],
            "z": ["z", "geopotential"]}
    fld = {}
    for k, cands in vmap.items():
        n = _pick(nc, cands)
        a = np.squeeze(_arr(nc.variables[n]))
        if a.ndim > 3:                  # 仍是多时次（未拆分的文件）：取首片
            a = a[0]
        fld[k] = a
    nc.close()
    return fld, lat, lon, lev


def orient(fld, lat, lon, lev, ref_lat, ref_lev):
    """把 lat / lev 的单调方向调整为与参考（GFS）一致，避免导数符号错位。"""
    if (lat[-1] > lat[0]) != (ref_lat[-1] > ref_lat[0]):
        lat = lat[::-1]
        for k in fld:
            fld[k] = fld[k][:, ::-1, :]
    if (lev[-1] > lev[0]) != (ref_lev[-1] > ref_lev[0]):
        lev = lev[::-1]
        for k in fld:
            fld[k] = fld[k][::-1]
    return fld, lat, lon, lev


def subset_levels(fld, lev, keep):
    """裁剪到指定层次集合（取最接近者）。"""
    idx = [int(np.argmin(np.abs(lev - l))) for l in keep]
    idx = sorted(set(idx))
    return {k: v[idx] for k, v in fld.items()}, np.asarray(lev)[idx]


# ======================================================================
# 通用：注入误差 / 诊断 / 相对误差
# ======================================================================
def make_noise(shape, dlat_m, rng, field):
    """空间相关长度 NOISE_KM 的高斯噪声，幅度 = NOISE_FRAC × 场标准差。"""
    g = max(NOISE_KM / dlat_m, 0.5)
    raw = rng.normal(size=shape)
    n = gaussian_filter(raw, sigma=(0, g, g), mode="nearest")
    n = n / (np.nanstd(n) + 1e-12)
    return n * (NOISE_FRAC * float(np.nanstd(field)))


def rel_field_error(truth, bg, mask):
    """场相对误差：RMSE(误差)/std(真值)，对 T/q/u/v 取中位。"""
    vals = []
    for v in ("T", "q", "u", "v"):
        e = bg[v] - truth[v]
        m = mask & np.isfinite(e)
        if m.sum() < 10:
            continue
        s = float(np.nanstd(truth[v][mask & np.isfinite(truth[v])]))
        if s <= 0:
            continue
        vals.append(float(np.sqrt(np.nanmean(e[m] ** 2))) / s)
    return float(np.median(vals)) if vals else float("nan")


def rel_diag_error(dg_truth, dg_bg, key, mask):
    """诊断相对误差：log 空间 RMSE / log 空间真值标准差。"""
    a = np.log10(np.clip(np.asarray(dg_bg[key], float), 1e-14, None))
    t = np.log10(np.clip(np.asarray(dg_truth[key], float), 1e-14, None))
    e = a - t
    m = mask & np.isfinite(e)
    if m.sum() < 10:
        return float("nan")
    s = float(np.nanstd(t[m]))
    if s <= 0:
        return float("nan")
    return float(np.sqrt(np.nanmean(e[m] ** 2))) / s


def median_runs(runs):
    """对多个时次的结果取中位（稳健汇总），保留单时次明细在 era5_runs。"""
    def med(path):
        vals = []
        for r in runs:
            x = r
            for p in path:
                x = x.get(p) if isinstance(x, dict) else None
                if x is None:
                    break
            if x is not None and np.isfinite(x):
                vals.append(float(x))
        return float(np.median(vals)) if vals else float("nan")

    out = dict(n_times=len(runs),
               rel_field_err=med(["rel_field_err"]),
               rel_diag_err=med(["rel_diag_err"]),
               amplification=med(["amplification"]),
               operators={})
    for op in ("a_hgauss", "b_hsharp", "c_vgauss"):
        out["operators"][op] = dict(
            rel_diag_err=med(["operators", op, "rel_diag_err"]),
            d_pct=med(["operators", op, "d_pct"]))
    return out


def run_system(tag, truth, lat, lon, lev, above):
    """对一个系统跑 [B] 导数放大 与 [C] 算子准则。返回 dict。"""
    cruise = ((lev[:, None, None] >= CRUISE[0])
              & (lev[:, None, None] <= CRUISE[1])) & above
    dlat = float(np.mean(np.diff(lat))) * 111195.0
    rng = np.random.default_rng(SEED)

    bg = {}
    for v in KEYS:
        if v == "z":
            bg["z"] = truth["z"]
        else:
            bg[v] = truth[v] + make_noise(truth[v].shape, dlat, rng, truth[v])

    dg_t = S36.diag(truth, lat, lon, lev)
    dg_b = S36.diag(bg, lat, lon, lev)

    fe = rel_field_error(truth, bg, above)
    de = rel_diag_error(dg_t, dg_b, "Ri_dry", cruise)
    amp = de / fe if fe > 0 else float("nan")

    out = dict(n_lev=int(len(lev)), shape=list(truth["T"].shape),
               rel_field_err=float(fe), rel_diag_err=float(de),
               amplification=float(amp))
    print("    [%s] 场相对误差 %.4f | 诊断相对误差 %.4f | 放大 %.2fx"
          % (tag, fe, de, amp))

    # ---- [C] 算子准则 ----
    g_h = max(45e3 / dlat, 0.5)          # 与 step36 [B] 一致的 45 km 尺度
    ops = {}
    for op in ("a_hgauss", "b_hsharp", "c_vgauss"):
        fo = {}
        for v in KEYS:
            f = bg[v]
            if op == "a_hgauss":
                fo[v] = S36.apply_horizontal(f, S36.gaussian_kernel_1d(g_h))
            elif op == "b_hsharp":
                k_cut = np.sqrt(np.log(2.0)) / (2 * np.pi * g_h)
                fo[v] = S36.apply_horizontal(f, S36.sinc_kernel_1d(k_cut))
            else:
                fo[v] = S36.apply_vertical(f, S36.gaussian_kernel_1d(2.0))
        fo["z"] = truth["z"]
        dg_o = S36.diag(fo, lat, lon, lev)
        de_o = rel_diag_error(dg_t, dg_o, "Ri_dry", cruise)
        ops[op] = dict(rel_diag_err=float(de_o),
                       d_pct=float(100 * (de - de_o) / de) if de > 0
                       else float("nan"))
        print("      %-10s 诊断相对误差 %.4f  下降 %+.1f%%"
              % (op, de_o, ops[op]["d_pct"]))
        del fo, dg_o
        gc.collect()
    out["operators"] = ops
    del dg_t, dg_b
    gc.collect()
    return out


# ======================================================================
# [A] 谱质心
# ======================================================================
def spectral_centroid(fld, lev, target_lev=250.0):
    """在指定层的水平能量谱质心（归一化频率 + 物理波长）。"""
    k = int(np.argmin(np.abs(np.asarray(lev, float) - target_lev)))
    out = {}
    for v in ("T", "q", "u", "v"):
        f = np.asarray(fld[v], float)[k]
        if not np.isfinite(f).all():
            f = np.where(np.isfinite(f), f, np.nanmean(f[np.isfinite(f)]))
        kk, pp = m2_spectrum2d(f)
        sel = np.isfinite(kk) & np.isfinite(pp) & (kk > 0)
        if sel.sum() < 3:
            continue
        kc = float(np.sum(kk[sel] * pp[sel]) / np.sum(pp[sel]))
        out[v] = dict(k_c=kc, wavelength_grid=float(1.0 / kc) if kc > 0 else None)
    return out


def main():
    t0 = time.time()
    print("=" * 98)
    print("step37b  ERA5（ECMWF IFS）独立系统谱分解对照")
    print("=" * 98)

    files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_pl_*.nc")))
    files = [f for f in files if "_probe" not in f]
    # 排除月度合并文件（era5_pl_YYYYMM.nc），只保留 step37_split_era5 拆出的单时次
    single = [f for f in files
              if re.search(r"era5_pl_\d{6}_.+\.nc$", os.path.basename(f))]
    if single:
        files = single
    if not files:
        print("\n未找到 ERA5 数据：%s 下无 era5_pl_*.nc" % ERA5_DIR)
        print("请先运行 step37_fetch_era5.py（需先在 CDS 网页接受数据集许可）。")
        return 1
    print("\n发现 %d 个 ERA5 时次，取首个做主对照" % len(files))

    # ---- GFS 参考（同口径基准） ----
    print("\n载入 GFS 参考算例 ...", flush=True)
    X = S12.load_case(S12.TARGETS[0])
    g_lat, g_lon, g_lev = X["lat"], X["lon"], X["lev"]
    g_truth = {v: X["nat"][v] for v in KEYS}
    print("  GFS  lev=%d  shape=%s" % (len(g_lev), g_truth["T"].shape))

    # ---- ERA5 ----
    fld, lat, lon, lev = load_era5(files[0])
    fld, lat, lon, lev = orient(fld, lat, lon, lev, g_lat, g_lev)
    print("  ERA5 lev=%d  shape=%s  lat[0,−1]=%.2f,%.2f"
          % (len(lev), fld["T"].shape, lat[0], lat[-1]))

    # ---- 层次取交集 ----
    common = [float(l) for l in lev if np.min(np.abs(g_lev - l)) < 1.0]
    if len(common) < 5:
        print("  共有层次不足（%d），改用各自全部层次" % len(common))
        e_fld, e_lev = fld, lev
        g_fld, g_lev2 = g_truth, g_lev
    else:
        common = sorted(common, reverse=True)
        e_fld, e_lev = subset_levels(fld, lev, common)
        g_fld, g_lev2 = subset_levels(g_truth, g_lev, common)
    print("  共有层次 %d 层：%s" % (len(e_lev), list(e_lev)))

    # above 掩码（参考 GFS 的口径，按层广播）
    above = np.zeros(e_fld["T"].shape, bool)
    above[:] = True

    # ---- [A] 谱结构 ----
    print("\n[A] 水平能量谱质心 k_c（250 hPa，归一化频率 cycles/grid）")
    print("-" * 98)
    spec = dict(gfs=spectral_centroid(g_fld, g_lev2),
                era5=spectral_centroid(e_fld, e_lev))
    for sysname in ("gfs", "era5"):
        for v, d in spec[sysname].items():
            print("  %-5s %-3s k_c=%.4f  波长≈%.1f 格"
                  % (sysname, v, d["k_c"], d["wavelength_grid"] or float("nan")))

    # ---- [B][C] 双系统对照 ----
    print("\n[B][C] 导数放大与算子准则（Ri_dry，巡航带 200–400 hPa）")
    print("-" * 98)
    res = dict(gfs=run_system("GFS ", g_fld, g_lat, g_lon, g_lev2, above))

    # ERA5：对每个已下载时次独立跑一遍再取中位（下载的时次全部用上）
    era5_runs = []
    for fp in files[:N_TIMES]:
        f2, la2, lo2, le2 = load_era5(fp)
        f2, la2, lo2, le2 = orient(f2, la2, lo2, le2, g_lat, g_lev)
        if len(common) >= 5:
            f2, le2 = subset_levels(f2, le2, common)
        era5_runs.append(run_system("ERA5", f2, la2, lo2, le2, above))
        del f2
        gc.collect()
    res["era5"] = median_runs(era5_runs)
    res["era5_runs"] = era5_runs
    print("  ERA5 汇总 %d 个时次（中位）" % len(era5_runs))

    amp_g = res["gfs"]["amplification"]
    amp_e = res["era5"]["amplification"]
    ratio = amp_e / amp_g if amp_g > 0 else float("nan")
    print("\n  ⇒ 导数放大倍数：GFS %.2fx  vs  ERA5 %.2fx  比值 %.3f"
          % (amp_g, amp_e, ratio))
    for op in ("a_hgauss", "b_hsharp", "c_vgauss"):
        dg_ = res["gfs"]["operators"][op]["d_pct"]
        de_ = res["era5"]["operators"][op]["d_pct"]
        same = (np.sign(dg_) == np.sign(de_))
        print("     算子 %-10s GFS %+8.1f%%  ERA5 %+8.1f%%  方向一致：%s"
              % (op, dg_, de_, "是" if same else "否"))

    out = dict(era5_file=os.path.basename(files[0]),
               n_era5_times=len(files),
               common_levels=[float(x) for x in e_lev],
               spectral=spec,
               systems=res,
               amp_ratio=float(ratio),
               operator_sign_agree={op: bool(
                   np.sign(res["gfs"]["operators"][op]["d_pct"])
                   == np.sign(res["era5"]["operators"][op]["d_pct"]))
                   for op in ("a_hgauss", "b_hsharp", "c_vgauss")})
    os.makedirs("data", exist_ok=True)
    with open(OUT, "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n完成，用时 %.0f s → %s" % (time.time() - t0, OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
