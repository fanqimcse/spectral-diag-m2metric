# -*- coding: utf-8 -*-
"""
m2metric —— M2（误差谱二阶矩）作为下游产品导向的同化评价指标。

一个轻量、自包含、无第三方重依赖（仅 numpy）的工具包，用于回答一个
方法学问题：

    当你的下游产品是导数型诊断（Ri / N² / S² / 涡度 / CAPE / 锋生函数），
    同化系统的验证指标应该用 RMSE（∝ 误差谱零阶矩 M0）还是 M2？

核心结论（本项目的发现）：

    RMSE 度量的是误差谱的零阶矩 M0 = ∫E(k)dk（"场"的总误差能量）；
    导数型诊断的误差由二阶矩 M2 = ∫k²E(k)dk 主导（导数算子在波数空间
    是高通，权重 k²）。

    因此，当同化算子只修正大尺度（低波数）误差时——例如 EnOI 的光滑
    增量——RMSE 会显著下降（M0 减小），但 M2 几乎不动，下游诊断毫无
    改善。RMSE 在这里"说谎"了：它报告了改进，但下游产品没有变好。

    反之，当尺度分离（如高斯平滑）消除小尺度噪声时，M2 显著下降而 M0
    几乎不动，RMSE 又会"掩盖"这次真正有价值的改进。

M2 指标正是为暴露这种"RMSE 盲区"而设计的。

主要 API
--------
    compute_m2(error_field, dx, dy, ...)
        对单个误差场，返回 M0、M2 及二者的估计统计量（偏倚/方差/CI）。

    discriminate(an_error, scale_error, bg_error, ...)
        对"同化 vs 尺度分离"两组误差场，返回 M0 与 M2 的判别力及
        配对 bootstrap 95% CI。

    spectrum2d(field, dx, dy, mask)
        二维水平功率谱（径向平均，含去趋势、加窗、NaN/掩码处理）。

    auto_k_cut(power, k)
        自动选择截断波数（偏倚—方差权衡的正则化）。

参考资料
--------
    step26_m2_metric.py  —— M2 的统计性质（偏倚/方差/截断权衡）
    step31_m2_ci.py      —— M2 判别力的配对 bootstrap 置信区间
    step21_spectral.py   —— 真实 GFS 误差谱分解（同化/平滑分工）

作者：风场—温度—湿度三维精细场实时重构 项目组
"""

from .core import (
    spectrum2d,
    moments,
    compute_m2,
    auto_k_cut,
    bootstrap_ci,
    discriminate,
    M0,
    M2,
)

__all__ = [
    "spectrum2d",
    "moments",
    "compute_m2",
    "auto_k_cut",
    "bootstrap_ci",
    "discriminate",
    "M0",
    "M2",
]

__version__ = "0.1.0"
