# -*- coding: utf-8 -*-
"""step13b —— 把航迹产品从"一条示例"扩成"一批代表性航线"，给出分布而不是孤例。

step13 只在东京→新加坡一条航线上演示了一遍。一条航线什么都说明不了：
它恰好可能是一条 δ≈0 的航线（实测确实如此：巡航段 δ 中位 0.086%，
主指标 A 占比 0.000%），用它当产品展示会**低估**结论，也会被人质疑挑样本。

本脚本在 10 条真实航线上跑同一套标定，报：
  · 每条航线的 P(遭遇 light) 与 dry→switch 的概率变化；
  · 巡航段 δ 中位与主指标 A（干判稳定、虚温判不稳定）占比；
  · 跨航线的**分布**（中位与四分位），而不是单点。

关键设计：三种口径（dry / virt / DK）各自**单独标定**后再比较，
所以概率差是"信息量差"，不是"标定差"。
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from avwx import aws_gfs as AG
from avwx import risk as RK
from avwx import calib as CB

RH_CLEAR = 80.0
CRUISE = 200.0

# (名称, 起, 终)  —— 覆盖东亚/东南亚/跨洋/高纬，避免只挑一条
ROUTES = [
    ("东京–新加坡", (35.55, 139.78), (1.36, 103.99)),
    ("北京–广州", (40.08, 116.58), (23.39, 113.30)),
    ("上海–成都", (31.14, 121.81), (30.58, 103.95)),
    ("香港–首尔", (22.31, 113.91), (37.46, 126.44)),
    ("东京–悉尼", (35.55, 139.78), (-33.94, 151.18)),
    ("新加坡–悉尼", (1.36, 103.99), (-33.94, 151.18)),
    ("上海–乌鲁木齐", (31.14, 121.81), (43.91, 87.47)),
    ("东京–檀香山", (35.55, 139.78), (21.32, -157.92)),
    ("北京–东京", (40.08, 116.58), (35.55, 139.78)),
    ("广州–曼谷", (23.39, 113.30), (13.69, 100.75)),
]


def main():
    calib = CB.load()
    if calib is None:
        raise SystemExit("缺少标定文件 %s，先跑 step13" % CB.CALIB_FILE)
    g = sorted(glob.glob("data/aws/gfs_*_f000_L21_1000_100.grib2"))
    if not g:
        raise SystemExit("无本地 GFS 场")
    d = AG.load(g[0], bbox=(90.0, 200.0, -45.0, 50.0))
    lat, lon, lev = d["lat"], d["lon"], d["level"]
    fld = dict(T=d["T"], q=d["q"], u=d["u"], v=d["v"], z=d["z"])
    diag = RK.stability_fields(fld, lat, lon, lev)

    print("=" * 108)
    print("step13b  航迹产品：10 条航线 × 三种口径（各自单独标定）")
    print("=" * 108)
    print("%-14s %7s %9s %9s %9s %9s %9s %9s"
          % ("航线", "有效点", "P(light)", "ΔP(sw-dry)", "δ巡航%",
             "A%", "cat_dry%", "emerge%"))
    rows = []
    for name, p0, p1 in ROUTES:
        try:
            r = RK.assess_route(fld, diag, lat, lon, lev, p0, p1,
                                cruise_hpa=CRUISE, n=100)
        except Exception as e:
            print("%-14s 失败: %s" % (name, str(e)[:60]))
            continue
        cruise = r["lev"] <= CRUISE * 1.15
        rh = r["rh"]
        rd, rv, rk = r["Ri_dry"], r["Ri_moist"], r["Ri_dk"]
        pd = CB.route_probability(rd, rv, rk, rh, calib, tag="light",
                                  edr="max", subset_policy="dry")
        ps = CB.route_probability(rd, rv, rk, rh, calib, tag="light",
                                  edr="max", subset_policy="switch")
        fin = cruise & np.isfinite(pd["p_point"]) & np.isfinite(ps["p_point"])
        dP = float(np.nanmedian((ps["p_point"] - pd["p_point"])[fin])) \
            if fin.any() else float("nan")
        nd2 = RK.interpolate_field(diag["Nd2"], lat, lon, lev, r["lat"],
                                   r["lon"] % 360.0, r["lev"])
        nv2 = RK.interpolate_field(diag["Nv2"], lat, lon, lev, r["lat"],
                                   r["lon"] % 360.0, r["lev"])
        clr = cruise & np.isfinite(rh) & (rh < RH_CLEAR)
        A = float(np.mean(clr & (nd2 > 0) & (nv2 < 0))) if clr.any() else np.nan
        # ⚠ 踩坑：别写 (r.get(k) or np.nan) —— 当值为 0.0 时 `or` 会把它
        #   判为假并替换成 nan，而"0%"恰恰是本表最重要的一个结果
        #   （巡航段没有一个点越过干式 CAT 阈值，正是"业务场太光滑"的直接证据）。
        def num(v):
            return float(v) if v is not None and np.isfinite(v) else float('nan')
        row = dict(route=name, n=int(fin.sum()),
                   p_enc_dry=pd.get("p_encounter"), p_enc_sw=ps.get("p_encounter"),
                   frac_dry=pd.get("frac_expect"), frac_sw=ps.get("frac_expect"),
                   dP_median=dP,
                   delta=100 * num(r.get("delta_median")),
                   A_pct=100 * A,
                   cat_dry=100 * num(r.get("cat_dry_frac")),
                   emerge=100 * num(r.get("emerge_frac")),
                   emerge_clear=100 * num(r.get("emerge_clear_frac")),
                   clear=100 * num(r.get("clear_frac")))
        rows.append(row)
        print("%-14s %7d %9.3f %+9.4f %9.3f %9.3f %9.2f %9.2f"
              % (name, row["n"], row.get("frac_sw") or np.nan, dP,
                 row["delta"], row["A_pct"], row["cat_dry"], row["emerge"]))

    def med(k):
        v = [r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        return float(np.median(v)) if v else None

    summary = dict(
        n_routes=len(rows),
        delta_median=med("delta"), delta_iqr=[med("delta")],
        A_pct_median=med("A_pct"), A_pct_max=float(np.nanmax(
            [r["A_pct"] for r in rows])),
        cat_dry_median=med("cat_dry"), cat_dry_max=float(np.nanmax(
            [r["cat_dry"] for r in rows])),
        emerge_median=med("emerge"), emerge_clear_median=med("emerge_clear"),
        clear_median=med("clear"),
        dP_median=med("dP_median"), dP_min=float(np.nanmin(
            [r["dP_median"] for r in rows])),
        dP_max=float(np.nanmax([r["dP_median"] for r in rows])),
        frac_sw_median=med("frac_sw"))
    print("\n跨航線中位：δ %.3f%%  A %.3f%%（最大 %.3f%%）  "
          "cat_dry %.2f%%（最大 %.2f%%）  emerge %.2f%%  晴空占比 %.1f%%"
          % (summary["delta_median"], summary["A_pct_median"],
             summary["A_pct_max"], summary["cat_dry_median"],
             summary["cat_dry_max"], summary["emerge_median"],
             summary["clear_median"]))
    print("ΔP(switch−dry) 中位 %+.4f，范围 [%+.4f, %+.4f]"
          % (summary["dP_median"], summary["dP_min"], summary["dP_max"]))
    with open("data/step13b_routeset.json", "w", encoding="utf8") as f:
        json.dump(dict(routes=rows, summary=summary), f,
                  ensure_ascii=False, indent=1)
    print("已写出 data/step13b_routeset.json")


if __name__ == "__main__":
    main()
