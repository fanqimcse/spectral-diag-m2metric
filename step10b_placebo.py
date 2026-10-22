# -*- coding: utf-8 -*-
"""step10b —— V2q 耦合的**安慰剂 / 随机化对照**

在 step10 里，把"风的分析增量"沿自身方向推移背景水汽场，使湿度 RMSE 相对
背景下降 2.7–3.9 个百分点。但这个数字本身还不能证明机制正确：

    **任何把背景场"抹一抹"的操作都可能恰好碰对一部分误差。**

尤其位移算子带有隐含的空间平滑（双线性重采样），而背景湿度误差本身在大尺度
上有相关性 —— 平滑也有可能"蒙对"。因此必须做零假设对照：

    H0 : 增益来自"随便移动一下背景场"，与增量方向/结构无关。

做法（对同一时次、同一观测、同一算子，只改动输入）：
    (1) real      真实分析增量 δV             —— step10 的 D 配置
    (2) shuffle   把 δV 的**空间位置随机打乱**（幅度谱不变，与误差的结构关系破坏）
    (3) flip      δV 整体取反（-δV）           —— 若只是"抹平滑"，取反也应改善
    (4) smooth    用同等尺度的**高斯平滑**代替位移（保留平滑效应、去掉流动依赖）

判据：只有 real 显著优于 shuffle/flip/smooth，才能说增益来自
"700 hPa у wind 增量沿着真实水汽梯度方向的位移"这一流动依赖机制。
"""
import gc
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")
from avwx import aws_gfs as AG
from avwx import recon3d as R3
import step10_realerror_osse as S10

LEADS = (12, 24)


def rmse(a, b, mask=None):
    dv = np.asarray(a, float) - np.asarray(b, float)
    if mask is not None:
        dv = dv[mask]
    return float(np.sqrt(np.nanmean(dv ** 2)))


def variant_q(uniB, bg, lat, lon, lev, tau, mode, rng):
    """按不同方式构造'伪风增量'，再施加同一个 V2q 位移算子，返回改善的个百分点。"""
    dU = uniB["u"] - bg["u"]
    dV = uniB["v"] - bg["v"]
    if mode == "shuffle":
        flat = lambda A: A.reshape(A.shape[0], -1)
        pflat = lambda A: A.reshape(A.shape[0], -1)
        dUe = np.empty_like(dU); dVe = np.empty_like(dV)
        for k in range(dU.shape[0]):
            p = rng.permutation(dU.shape[1] * dU.shape[2])
            dUe[k] = pflat(dU)[k][p].reshape(dU.shape[1:])
            qp = rng.permutation(dV.shape[1] * dV.shape[2])
            dVe[k] = pflat(dV)[k][qp].reshape(dV.shape[1:])
        dU, dV = dUe, dVe
    elif mode == "flip":
        dU, dV = -dU, -dV
    elif mode == "smooth":
        # 与位移位移量相当的等效平滑 σ：位移中位数对应的格数
        s = tau * 3600.0 * float(np.nanmedian(np.abs(dU)))
        g = s / (abs(lat[1] - lat[0]) * 111195.0)
        g = float(np.clip(g, 0.5, 12.0))
        dU = gaussian_filter(dU, sigma=(0, g, g), mode="nearest")
        dV = gaussian_filter(dV, sigma=(0, g, g), mode="nearest")
    dq = R3.moisture_displacement(dU, dV, bg["q"], lat, lon, tau_h=tau)
    return np.clip(uniB["q"] + dq, 0, 0.06)


def read_calib():
    """耦合系数一律从 step10 的标定结果读取，避免两处参数漂移。

    ⚠ 曾踩坑：本脚本一度硬编码 alpha/tau，而 step10 的标定扫描范围后来扩展，
      两边参数不再一致 —— 安慰剂对照的前提就是"除输入外条件完全相同"。
    """
    with open("data/step10_realerror_osse.json", encoding="utf8") as f:
        d = json.load(f)
    return {int(ld): (v["calibration"]["alpha_V2q"], v["calibration"]["tau_h"])
            for ld, v in d["results"].items()}


def main():
    out = {}
    CAL = read_calib()
    for lead in LEADS:
        alpha, tau = CAL[lead]
        print("\n" + "=" * 88)
        print("安慰剂对照  f%03d    (alpha_V2q=%.2f, tau_h=%.1f h，取自 step10 标定)"
              % (lead, alpha, tau))
        print("=" * 88)
        rows = []
        for i, t in enumerate(S10.TRUTH_TIMES[1:]):
            X = S10.load_pair(t, S10.bg_cycle(t[0], t[1], lead), lead)
            lat, lon, lev = X["lat"], X["lon"], X["lev"]
            rec = S10.make_rec(lat, lon, lev, X["dg"], X["sigma"])
            uniB = rec.univariate(X["bg"], X["obs"], use=("sfc", "raob", "ac"))
            am = S10.impact_mask(X["obs"]["ac"][1], lat, lon, lev) & X["above"]
            bgq = rmse(X["bg"]["q"], X["nat"]["q"], am)
            qB = rmse(uniB["q"], X["nat"]["q"], am)
            res = {}
            for mode in ("real", "shuffle", "flip", "smooth"):
                rng = np.random.default_rng(4242 + i)
                if mode == "real":
                    an = dict(uniB)
                    an["q"] = np.clip(
                        uniB["q"] + alpha * R3.moisture_displacement(
                            uniB["u"] - X["bg"]["u"], uniB["v"] - X["bg"]["v"],
                            X["bg"]["q"], lat, lon, tau_h=tau), 0, 0.06)
                else:
                    an = dict(uniB)
                    an["q"] = variant_q(uniB, X["bg"], lat, lon, lev, tau,
                                        mode, rng)
                    # 与 real 同一基准（单变量分析）与同一幅度系数 alpha
                    an["q"] = uniB["q"] + alpha * (an["q"] - uniB["q"])
                res[mode] = 100 * (bgq - rmse(an["q"], X["nat"]["q"], am)) / bgq
            rows.append(dict(time="%s%sz" % t, **res))
            print("    %s %sz  real %+6.2f  shuffle %+6.2f  flip %+6.2f  "
                  "smooth %+6.2f   [百分点，相对背景]"
                  % (t[0], t[1], res["real"], res["shuffle"], res["flip"],
                     res["smooth"]), flush=True)
            del X, uniB, rec
            gc.collect()
        med = {m: float(np.median([r[m] for r in rows])) for m in
               ("real", "shuffle", "flip", "smooth")}
        print("    中位：   real %+6.2f  shuffle %+6.2f  flip %+6.2f  smooth %+6.2f"
              % (med["real"], med["shuffle"], med["flip"], med["smooth"]))
        print("    零假设判据 real − max(shuffle, flip, smooth) = %+6.2f 个百分点"
              % (med["real"] - max(med["shuffle"], med["flip"], med["smooth"])))
        out[str(lead)] = dict(rows=rows, median=med,
                              margin=med["real"] - max(med["shuffle"],
                                                       med["flip"],
                                                       med["smooth"]))
    with open("data/step10b_placebo.json", "w", encoding="utf8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step10b_placebo.json")


if __name__ == "__main__":
    main()
