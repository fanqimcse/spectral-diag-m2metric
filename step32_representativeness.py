# -*- coding: utf-8 -*-
"""step32 —— P3：用探空空间合成分离代表性误差，坐实"同化对诊断改善更弱"。

step29 的探空验证有一个诚实声明的方法学限制：
    "探空是**点观测**、GFS 是 0.25° **格点均值**，两者之差混入了**代表性误差**
     （representativeness error），而代表性误差在梯度上更大、且无法被同化修正
     —— 它污染了'同化对梯度的相对改善小于对场的改善'这一具体预测。"

本脚本针对这个限制做正面处理。核心思想：

   代表性误差 ε_rep 来自"点观测包含格点内小尺度扰动（湍流/地形/局地对流），
   而 GFS 格点是面积均值"。这个小尺度扰动在垂直梯度上被导数算子放大，
   但**同化无法修正它**（它是观测自己的噪声，不是背景场的偏差）。

   如果能把多个**空间邻近**的探空站做平均，构造一个"准格点观测"，
   就能把 ε_rep 的方差压低（按 1/√N 若站点独立），从而：
   · 单点探空（ε_rep 大）→ 梯度误差被代表性误差主导 → 同化改善"看似"不小
   · 空间平均探空（ε_rep 小）→ 梯度误差反映真实背景场误差 → 同化改善"更弱"
   若后者的"同化对梯度改善 < 对场改善"浮现，就坐实了命题 P 的 OSSE 预测。

数据：复用 step29 的 17 个探空站 + GFS 缓存。韩国 5 站最密集（34-37N,125-130E），
用作空间合成的主区域；其余站点做单点对照。
"""
import gc
import json
import os
import sys
import time
import zipfile

import numpy as np

sys.path.insert(0, ".")
import step10_realerror_osse as S10

OUT = "data/step32_representativeness.json"

TARGET_DAYS = [
    (2022, 1, 15), (2022, 1, 16), (2022, 7, 15), (2022, 7, 16),
    (2023, 1, 15), (2023, 1, 16), (2023, 7, 15), (2023, 7, 16),
]

SITES = {
    "JAM00047909": (28.39, 129.49, "名濑/日本"),
    "JAM00047918": (24.34, 124.16, "石垣/日本"),
    "KSM00047122": (37.10, 127.03, "乌山/韩国"),
    "KSM00047138": (36.03, 129.38, "浦项/韩国"),
    "KSM00047155": (35.17, 128.57, "昌原/韩国"),
    "KSM00047169": (34.69, 125.45, "黑山岛/韩国"),
    "KSM00047230": (37.26, 126.10, "德积/韩国"),
    "HKM00045004": (22.31, 114.17, "香港"),
    "VMM00048820": (21.03, 105.80, "河内/越南"),
    "VMM00048845": (18.67, 105.68, "荣市/越南"),
    "VMM00048855": (16.07, 108.35, "岘港/越南"),
    "RPM00098223": (18.18, 120.53, "拉瓦格/菲律宾"),
    "RPM00098233": (17.65, 121.76, "土格加劳/菲律宾"),
    "THM00048378": (16.80, 100.28, "彭世洛/泰国"),
    "THM00048381": (16.46, 102.79, "孔敬/泰国"),
    "THM00048407": (15.25, 104.87, "乌汶/泰国"),
    "PFM00059981": (16.83, 112.33, "西沙"),
}

# 空间合成组：韩国 5 站（34-37N, 125-130E）—— 空间最密集
KOREA_SITES = ["KSM00047122", "KSM00047138", "KSM00047155",
               "KSM00047169", "KSM00047230"]


def parse_igra_sounding(sid, target_days):
    """解析一个站点的探空（复用 step29 逻辑）。返回 {(y,m,d,h): [(p,T,rh),...]}"""
    fn = "data/igra_%s.zip" % sid
    if not os.path.exists(fn):
        return {}
    z = zipfile.ZipFile(fn)
    txt = z.read(z.namelist()[0]).decode("utf8", "replace")
    lines = txt.split("\n")
    soundings = {}
    cur_key = None
    cur = []
    keep_cur = False

    def conv(s):
        s = s.strip()
        if s in ("", "-9999", "-8888"):
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    for L in lines:
        if L.startswith("#"):
            if cur_key is not None and keep_cur and cur:
                soundings[cur_key] = cur
            cur = []
            keep_cur = False
            p = L.split()
            if len(p) < 6:
                cur_key = None
                continue
            try:
                year, mon, day, hour = int(p[1]), int(p[2]), int(p[3]), int(p[4])
            except (ValueError, IndexError):
                cur_key = None
                continue
            cur_key = (year, mon, day, hour)
            keep_cur = (year, mon, day) in target_days
        else:
            if cur_key is None or not keep_cur:
                continue
            if len(L) < 51:
                continue
            press = L[9:15].strip()
            temp = L[22:27].strip()
            rh = L[28:33].strip()
            pv = conv(press)
            if not np.isfinite(pv) or pv < 1000:
                continue
            cur.append(dict(p=pv, T=conv(temp), rh=conv(rh)))
    if cur_key is not None and keep_cur and cur:
        soundings[cur_key] = cur
    return soundings


def raob_to_profile(raob, gfs_lev):
    """把探空插值到 GFS 气压层（log-p 线性），返回 (T_profile, valid_mask)。"""
    raob_p = np.array([r["p"] for r in raob]) / 100.0
    raob_T = np.array([r["T"] for r in raob]) / 10.0
    valid = np.isfinite(raob_T) & (raob_p > 0) & (raob_p < 1100)
    if valid.sum() < 5:
        return None
    raob_p = raob_p[valid]
    raob_T = raob_T[valid]
    lp_raob = np.log(raob_p)
    lp_gfs = np.log(gfs_lev)
    lo, hi = lp_gfs.min(), lp_gfs.max()
    sel = (lp_raob >= lo) & (lp_raob <= hi)
    if sel.sum() < 3:
        return None
    # 升序插值
    order = np.argsort(lp_raob[sel])
    T_gfslev = np.interp(lp_gfs, lp_raob[sel][order], raob_T[sel][order])
    return T_gfslev


def main():
    t0 = time.time()
    print("=" * 98)
    print("step32  P3：探空空间合成分离代表性误差")
    print("=" * 98)

    target_day_set = {(d[0], d[1], d[2]) for d in TARGET_DAYS}
    # 预解析所有站点探空
    all_sond = {}
    for sid in SITES:
        all_sond[sid] = parse_igra_sounding(sid, target_day_set)

    # 预加载 GFS（缓存）
    gfs_cache = {}
    def load_gfs(date, cyc, fhr):
        key = (date, cyc, fhr)
        if key not in gfs_cache:
            p = S10.native_path(date, cyc, fhr)
            if not os.path.exists(p):
                gfs_cache[key] = None
            else:
                gfs_cache[key] = S10.prepare(S10.AG.load(p, bbox=S10.BBOX))
        return gfs_cache[key]

    # 结果容器：单点（全部站点） vs 空间合成（韩国5站平均）
    rows_single = []
    rows_composite = []

    for (year, mon, day) in TARGET_DAYS:
        date = "%04d%02d%02d" % (year, mon, day)
        for hh in (0, 12):
            cyc = "%02d" % hh
            nat = load_gfs(date, cyc, 0)
            if nat is None:
                continue
            b = S10.bg_cycle(date, cyc, 24)
            bg = load_gfs(b[0], b[1], 24)
            if bg is None:
                continue
            lat = nat["lat"]; lon = nat["lon"]; lev = nat["lev"]
            gfs_lev = np.asarray(lev, float)

            # ---- 单点分析 ----
            for sid, (slat, slon, name) in SITES.items():
                key = (year, mon, day, hh)
                if key not in all_sond[sid]:
                    continue
                T_prof = raob_to_profile(all_sond[sid][key], gfs_lev)
                if T_prof is None:
                    continue
                ilat = int(np.argmin(np.abs(lat - slat)))
                ilon = int(np.argmin(np.abs(lon - slon)))
                gfs_T_f000 = nat["f"]["T"][:, ilat, ilon] - 273.15
                gfs_T_f024 = bg["f"]["T"][:, ilat, ilon] - 273.15
                cruise = (gfs_lev >= 200) & (gfs_lev <= 400)
                lp = np.log(gfs_lev)
                # 场误差与梯度误差
                e_f024 = gfs_T_f024 - T_prof
                e_f000 = gfs_T_f000 - T_prof
                grad_raob = np.gradient(T_prof, lp)
                grad_f024 = np.gradient(gfs_T_f024, lp)
                grad_f000 = np.gradient(gfs_T_f000, lp)
                eg_f024 = grad_f024 - grad_raob
                eg_f000 = grad_f000 - grad_raob
                m = cruise & np.isfinite(e_f024) & np.isfinite(eg_f024)
                if m.sum() < 3:
                    continue
                def rmse(e):
                    return float(np.sqrt(np.nanmean(e[m] ** 2)))
                r_f024 = rmse(e_f024); r_f000 = rmse(e_f000)
                rg_f024 = rmse(eg_f024); rg_f000 = rmse(eg_f000)
                d_field = (r_f024 - r_f000) / r_f024 if r_f024 > 0 else 0
                d_grad = (rg_f024 - rg_f000) / rg_f024 if rg_f024 > 0 else 0
                rows_single.append(dict(date=date, hh=hh, sid=sid, name=name,
                                        d_field=d_field, d_grad=d_grad,
                                        d_grad_minus_field=d_grad - d_field))

            # ---- 空间合成（韩国 5 站平均）----
            profs = []
            for sid in KOREA_SITES:
                key = (year, mon, day, hh)
                if key not in all_sond[sid]:
                    continue
                T_prof = raob_to_profile(all_sond[sid][key], gfs_lev)
                if T_prof is not None:
                    profs.append(T_prof)
            if len(profs) >= 3:   # 至少 3 站才能做有意义的空间平均
                T_comp = np.nanmean(np.vstack(profs), axis=0)
                # 合成观测的中心点（韩国 5 站质心）
                clat = np.mean([SITES[s][0] for s in KOREA_SITES])
                clon = np.mean([SITES[s][1] for s in KOREA_SITES])
                ilat = int(np.argmin(np.abs(lat - clat)))
                ilon = int(np.argmin(np.abs(lon - clon)))
                gfs_T_f000 = nat["f"]["T"][:, ilat, ilon] - 273.15
                gfs_T_f024 = bg["f"]["T"][:, ilat, ilon] - 273.15
                cruise = (gfs_lev >= 200) & (gfs_lev <= 400)
                lp = np.log(gfs_lev)
                e_f024 = gfs_T_f024 - T_comp
                e_f000 = gfs_T_f000 - T_comp
                grad_comp = np.gradient(T_comp, lp)
                grad_f024 = np.gradient(gfs_T_f024, lp)
                grad_f000 = np.gradient(gfs_T_f000, lp)
                eg_f024 = grad_f024 - grad_comp
                eg_f000 = grad_f000 - grad_comp
                m = cruise & np.isfinite(e_f024) & np.isfinite(eg_f024)
                if m.sum() >= 3:
                    def rmse2(e):
                        return float(np.sqrt(np.nanmean(e[m] ** 2)))
                    r_f024 = rmse2(e_f024); r_f000 = rmse2(e_f000)
                    rg_f024 = rmse2(eg_f024); rg_f000 = rmse2(eg_f000)
                    d_field = (r_f024 - r_f000) / r_f024 if r_f024 > 0 else 0
                    d_grad = (rg_f024 - rg_f000) / rg_f024 if rg_f024 > 0 else 0
                    rows_composite.append(dict(
                        date=date, hh=hh, n_sites=len(profs),
                        d_field=d_field, d_grad=d_grad,
                        d_grad_minus_field=d_grad - d_field))

        del nat, bg
        gc.collect()

    # ---- 汇总 ----
    print("\n" + "=" * 98)
    print("【汇总】代表性误差对'同化改善'的影响")
    print("=" * 98)

    def med(rows, k):
        v = [r[k] for r in rows if np.isfinite(r[k])]
        return float(np.median(v)) if v else float("nan")

    if rows_single:
        d_field_s = med(rows_single, "d_field")
        d_grad_s = med(rows_single, "d_grad")
        print("  单点探空（%d 样本，代表性误差大）：" % len(rows_single))
        print("    同化改善 场 %+.1f%%   梯度 %+.1f%%   （梯度−场 = %+.1f%%）"
              % (100 * d_field_s, 100 * d_grad_s, 100 * (d_grad_s - d_field_s)))
    if rows_composite:
        d_field_c = med(rows_composite, "d_field")
        d_grad_c = med(rows_composite, "d_grad")
        print("  空间合成探空（%d 样本，代表性误差小）：" % len(rows_composite))
        print("    同化改善 场 %+.1f%%   梯度 %+.1f%%   （梯度−场 = %+.1f%%）"
              % (100 * d_field_c, 100 * d_grad_c, 100 * (d_grad_c - d_field_c)))

    # 关键判据：空间合成后，是否"梯度改善 < 场改善"（命题 P 的预测）浮现？
    if rows_single and rows_composite:
        dmf_s = med(rows_single, "d_grad_minus_field")
        dmf_c = med(rows_composite, "d_grad_minus_field")
        print()
        print("  关键检验（d_grad − d_field 的中位）：")
        print("    单点（ε_rep 大）：%+.1f%%   →  空间合成（ε_rep 小）：%+.1f%%"
              % (100 * dmf_s, 100 * dmf_c))
        if dmf_c < dmf_s:
            print("  ⇒ 空间合成后 d_grad−d_field 更负，说明代表性误差确实掩盖了")
            print("    '同化对梯度改善更弱'的信号。命题 P 的 OSSE 预测在去除")
            print("    代表性误差后方向一致。")
        else:
            print("  ⇒ 空间合成未使 d_grad−d_field 更负，代表性误差不是主因。")

    out = dict(meta=dict(n_single=len(rows_single), n_composite=len(rows_composite),
                         korea_sites=KOREA_SITES),
               rows_single=rows_single, rows_composite=rows_composite,
               summary=dict(
                   single=dict(d_field=med(rows_single, "d_field") if rows_single else np.nan,
                               d_grad=med(rows_single, "d_grad") if rows_single else np.nan,
                               d_grad_minus_field=med(rows_single, "d_grad_minus_field") if rows_single else np.nan),
                   composite=dict(d_field=med(rows_composite, "d_field") if rows_composite else np.nan,
                                  d_grad=med(rows_composite, "d_grad") if rows_composite else np.nan,
                                  d_grad_minus_field=med(rows_composite, "d_grad_minus_field") if rows_composite else np.nan)))
    with open(OUT, "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("\n已写出 %s   用时 %.0f s" % (OUT, time.time() - t0))


if __name__ == "__main__":
    main()
