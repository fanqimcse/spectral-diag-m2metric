# -*- coding: utf-8 -*-
"""step10c —— 把 V2q 的增益与"纯平滑"彻底分开（最关键的零假设）

step10b 已经排除两类廉价解释：
    * flip（反方向）→ 显著恶化       ⟹ 位移**方向**是有效的
    * shuffle（空间打乱）→ 收益只剩 1/3 ⟹ 需要空间相干的位移
但 step10b 同时暴露了一个**危险信号**：把位移换成它的平滑版本后，
增益与真实增量几乎一模一样（f024: 5.93 vs 5.93）。这留下一个致命替代解释：

    H0' : 增益其实来自"对背景水汽场做了低通滤波"，
          即预测误差的小尺度分量本来就是噪声，抹掉它自然降 RMSE；
          与"沿着真实水汽梯度的流动依赖位移"毫无关系。

要判定 H0'，两级对照缺一不可：

    (1) randsmooth : 用**同一量级、同一平滑尺度的随机场**代替分析增量做位移。
                     若它与 real 一样好 ⟹ 增益只是"任意位移"的一般效应，
                     流动依赖机制不成立。
    (2) smoothq    : **完全不做位移**，直接把背景水汽场低通叠加回去。
                     若它与 real 一样好 ⟹ 增益就是纯平滑（常规Regularization），
                     本项目的 V2q 算子毫无增量。

只有 real 显著优于 randsmooth **且** 显著优于 smoothq，才可以说
"湿度盲的风观测，是通过流动依赖的水汽平流约束了湿度场"。
"""
import gc
import json
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
from avwx import recon3d as R3
import step10_realerror_osse as S10

LEADS = (12, 24)


def rmse(a, b, mask=None):
    dv = np.asarray(a, float) - np.asarray(b, float)
    if mask is not None:
        dv = dv[mask]
    return float(np.sqrt(np.nanmean(dv ** 2)))


def rand_smooth_like(A, lat, rng, corr_cells=8.0):
    """与 A 逐层同标准差、但**结构完全随机**的平滑场。"""
    out = np.empty_like(A)
    for k in range(A.shape[0]):
        a = A[k]
        sd = float(np.nanstd(a))
        g = rng.standard_normal(a.shape)
        g = gaussian_filter(g, sigma=corr_cells, mode="wrap")
        g = g / (np.nanstd(g) + 1e-12) * sd
        out[k] = g
    return out


def main():
    out = {}
    with open("data/step10_realerror_osse.json", encoding="utf8") as f:
        _d = json.load(f)
    CAL = {int(ld): (v["calibration"]["alpha_V2q"], v["calibration"]["tau_h"])
           for ld, v in _d["results"].items()}
    for lead in LEADS:
        alpha, tau = CAL[lead]
        print("\n" + "=" * 92)
        print("零假设对照（随机平滑场 / 纯低通）  f%03d    (alpha_V2q=%.2f, tau_h=%.1f h)"
              % (lead, alpha, tau))
        print("=" * 92)
        rows = []
        for i, t in enumerate(S10.TRUTH_TIMES[1:]):
            X = S10.load_pair(t, S10.bg_cycle(t[0], t[1], lead), lead)
            lat, lon, lev = X["lat"], X["lon"], X["lev"]
            rec = S10.make_rec(lat, lon, lev, X["dg"], X["sigma"])
            uniB = rec.univariate(X["bg"], X["obs"], use=("sfc", "raob", "ac"))
            am = S10.impact_mask(X["obs"]["ac"][1], lat, lon, lev) & X["above"]
            bgq = rmse(X["bg"]["q"], X["nat"]["q"], am)
            dU = uniB["u"] - X["bg"]["u"]
            dV = uniB["v"] - X["bg"]["v"]

            def pct(an_q):
                return 100 * (bgq - rmse(an_q, X["nat"]["q"], am)) / bgq

            dq_real = R3.moisture_displacement(dU, dV, X["bg"]["q"], lat, lon,
                                               tau_h=tau)

            # 有效平滑宽度：位移场对应的格数（= Δ essential for fair comparison）
            dlat_km = abs(lat[1] - lat[0]) * 111195.0
            shift_cells = float(np.nanmedian(np.abs(dU))) * tau * 3600.0 / dlat_km
            sig = float(np.clip(shift_cells, 0.5, 20.0))

            rng = np.random.default_rng(90210 + i)
            dU_r = rand_smooth_like(dU, lat, rng, corr_cells=max(sig, 4.0))
            dV_r = rand_smooth_like(dV, lat, rng, corr_cells=max(sig, 4.0))
            dq_rand = R3.moisture_displacement(dU_r, dV_r, X["bg"]["q"], lat,
                                               lon, tau_h=tau)
            # 纯低通：不做任何位移，仅把背景水汽场按同一有效宽度低通
            dq_lp = gaussian_filter(X["bg"]["q"], sigma=(0, sig, sig),
                                    mode="nearest") - X["bg"]["q"]

            res = dict(
                real=pct(np.clip(uniB["q"] + alpha * dq_real, 0, 0.06)),
                randsmooth=pct(np.clip(uniB["q"] + alpha * dq_rand, 0, 0.06)),
                smoothq=pct(np.clip(uniB["q"] + alpha * dq_lp, 0, 0.06)),
                effective_sigma_cells=sig)
            rows.append(dict(time="%s%sz" % t, **res))
            print("    %s %sz   real %+6.2f   randsmooth %+6.2f   smoothq %+6.2f"
                  "   [百分点]  等效低通 σ=%.1f 格"
                  % (t[0], t[1], res["real"], res["randsmooth"],
                     res["smoothq"], sig), flush=True)
            del X, uniB, rec
            gc.collect()
        med = {m: float(np.median([r[m] for r in rows]))
               for m in ("real", "randsmooth", "smoothq")}
        print("    中位：real %+6.2f   randsmooth %+6.2f   smoothq %+6.2f"
              % (med["real"], med["randsmooth"], med["smoothq"]))
        print("    real − randsmooth = %+6.2f      real − smoothq = %+6.2f"
              % (med["real"] - med["randsmooth"], med["real"] - med["smoothq"]))
        out[str(lead)] = dict(rows=rows, median=med)
    with open("data/step10c_null.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step10c_null.json")


if __name__ == "__main__":
    main()
