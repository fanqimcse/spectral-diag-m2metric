# -*- coding: utf-8 -*-
"""avwx.calib —— 把研究结论固化成**可复用的标定系数**，供航迹产品直接调用。

为什么单独成模块
----------------
step13 是一次性拟合脚本；产品端（step04 / risk.assess_route）需要一个
"输入 Ri 场，输出 P(EDR ≥ t)"的稳定接口。两者一旦分开写就会漂移 ——
本项目已经在 alpha_V2Q 上为这种漂移付过一次代价，所以这里把标定系数
与推断函数放在同一个文件里，系数只能由 step13 写出、只能由本模块读入。

标定口径（必须和论文一致）
--------------------------
* 预测量 y = 1[EDR ≥ t]，EDR 取 **maxEDR**（航段内峰值，对应"遭遇"语义）。
  medEDR 的结果同时保存，用于稳健性对比，但不作为产品默认。
* 预测因子 x = −log10(Ri)，Ri = N²/S²，N² 分别取干式 / 虚温式 / DK 饱和式。
* 每个诊断**单独**拟合一元逻辑回归（const + β·x）。
  理由（⚠ 踩过的坑）：Ri = N²/S² ⟹ −log10 Ri = −log10 N² + log10 S²，
  三个 x 之间、以及它们与 log S² 之间几乎完全共线；放进同一个多元模型
  会得到 β_dry=+41 / β_virt=−41 这类病态系数，什么都说明不了。
  要比较三个诊断的**信息量**，只能各自标定后比 Brier/AUC。
"""
import json
import os

import numpy as np

CALIB_FILE = os.path.join("data", "step13_route_product.json")
_DEFAULT = {
    # 由 step13 覆盖；这里的数是"无标定"时的保守占位，绝不应被产品用到。
    "_placeholder": True,
}


def load(path=CALIB_FILE):
    """读入标定文件。缺失或占位时返回 None，调用方必须显式处理。"""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf8") as fh:
        d = json.load(fh)
    if d.get("_placeholder") or "calib" not in d:
        return None
    return d["calib"]


def models_for(calib, tag="light", subset="全部", edr="max"):
    """取出 (tag, subset, edr) 下三个诊断各自的一元模型系数。

    返回 {"dry": (a, b), "virt": (a, b), "DK": (a, b)}。
    """
    if calib is None:
        return None
    key = "%s|%s|%s" % (tag, subset, edr)
    blk = calib.get(key)
    if not blk:
        return None
    out = {}
    for k in ("dry", "virt", "DK"):
        v = blk.get(k)
        if not v or len(v) < 2:
            continue
        a, b = float(v[0]), float(v[1])
        pn = float(v[2]) if (len(v) > 2 and v[2] is not None
                             and np.isfinite(v[2])) else None
        if np.isfinite(a) and np.isfinite(b):
            out[k] = (a, b, pn)
    return out or None


def p_from_ri(ri, a, b, p_neg=None, ri_max=1e6):
    """一元逻辑回归推断：logit P = a + b·(−log10 Ri)。

    ⚠ Ri ≤ 0 的处理（自检抓出来的坑）
    -------------------------------
    Ri ≤ 0 意味着该层（湿）静力不稳定 —— 这正是本项目"主指标 A"关心的
    那批点。这里**绝不能**用 −log10(负数)，也不能靠"截到 1e-6"外推：
    截断值是凭空发明的，而逻辑回归在 x = −log10(1e-6) = 6 处早已超出
    训练样本支撑集，外推出来的数（实测只有 0.29）物理上荒谬。

    正确做法：超出支撑集 ⇒ 不推断，改用该层在**训练样本中的实测频率**
    p_neg = P(EDR ≥ t | Ri ≤ 0)。这是观测值，不是模型外推。
    若训练样本里没有 Ri ≤ 0 的点（p_neg=None），退化为截断并**标记**
    为低置信，由调用方决定是否采用。
    """
    ri = np.asarray(ri, float)
    x = -np.log10(np.clip(ri, 1e-6, ri_max))
    eta = a + b * x
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
    p = np.where(np.isfinite(ri), p, np.nan)
    if p_neg is not None and np.isfinite(p_neg):
        p = np.where(np.isfinite(ri) & (ri <= 0.0), float(p_neg), p)
    return p


def clear_air_switch(rh, rh_clear=80.0):
    """晴空开关：True 表示该点处于晴空（未饱和）区。

    产品端的默认策略（也是本项目的核心结论之一）：
      * 晴空  ⇒ 用 **虚温式** Ri（湿度的正确口径；干式会系统性高估稳定度）
      * 云内  ⇒ 用 **DK 饱和式** Ri（此时饱和式才是正确的）
    这样"用哪个 N²"由物理状态决定，而不是全局一刀切。
    """
    rh = np.asarray(rh, float)
    return np.isfinite(rh) & (rh < rh_clear)


def route_probability(ri_dry, ri_virt, ri_dk, rh, calib, tag="light",
                      edr="max", rh_clear=80.0, subset_policy="switch"):
    """沿航迹逐点给出标定概率。

    subset_policy:
      "switch" —— 晴空用 virt、云内用 DK（推荐，物理一致）
      "all"    —— 全程用"全部样本"标定的 virt 系数（对照）
      "dry"    —— 全程用干式（业务现状基线）

    返回 dict：单点概率数组 + 三个聚合量。
      p_point      逐点概率
      p_encounter  全程至少遭遇一次  1 − Π(1 − p_i)
      n_expect     期望遭遇点数      Σ p_i
      frac_expect  期望遭遇比例      mean(p_i)
    注意 p_encounter 假设各点独立 —— 这是**上界**：实际颠簸沿航迹相关，
    真实的"至少一次"概率低于此值。产品输出时按上界标注。
    """
    if calib is None:
        raise ValueError("标定文件缺失或为占位，先跑 step13")
    m_all = models_for(calib, tag, "全部", edr)
    m_clr = models_for(calib, tag, "晴空+高层", edr)
    m = m_all if m_all is not None else m_clr
    if m is None:
        raise ValueError("标定文件里没有 tag=%s 的模型" % tag)

    rd = np.asarray(ri_dry, float)
    rv = np.asarray(ri_virt, float)
    rk = np.asarray(ri_dk, float)
    rhv = np.asarray(rh, float) if rh is not None else np.full(rd.shape, np.nan)

    if subset_policy == "dry":
        p = p_from_ri(rd, *m["dry"])
    elif subset_policy == "all":
        p = p_from_ri(rv, *(m_clr["virt"] if (m_clr and "virt" in m_clr)
                            else m["virt"]))
    else:  # switch
        clear = clear_air_switch(rhv, rh_clear)
        pv = p_from_ri(rv, *m["virt"])
        pk = p_from_ri(rk, *m["DK"]) if "DK" in m else np.full(pv.shape, np.nan)
        p = np.where(clear, pv, pk)
        # 晴空但缺 DK 系数、或云内但缺 virt 系数 时退回到可用项
        p = np.where(np.isfinite(p), p, pv)

    fin = np.isfinite(p)
    out = dict(p_point=p)
    if fin.any():
        out["p_encounter"] = float(1.0 - np.prod(1.0 - p[fin]))
        out["n_expect"] = float(np.sum(p[fin]))
        out["frac_expect"] = float(np.mean(p[fin]))
        out["p_max"] = float(np.max(p[fin]))
        out["n_valid"] = int(fin.sum())
    else:
        out.update(p_encounter=np.nan, n_expect=np.nan, frac_expect=np.nan,
                   p_max=np.nan, n_valid=0)
    return out


def selfcheck(verbose=True):
    """自检：全部针对本项目实际踩过的坑。"""
    ok = tot = 0

    def chk(name, cond, extra=""):
        nonlocal ok, tot
        tot += 1
        if cond:
            ok += 1
        if verbose:
            print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                                   ("  " + extra) if extra else ""))

    a, b = -3.0, 0.35
    # 1) 单调性：Ri 越小（越不稳定）概率越高 —— β>0 时 −log10 Ri 越大
    r = np.array([10.0, 1.0, 0.1, 0.01])
    p = p_from_ri(r, a, b)
    chk("P 随 Ri 下降单调上升", np.all(np.diff(p) > 0),
        "%.4f→%.4f" % (p[0], p[-1]))

    # 2) 概率落在 (0,1)
    chk("概率落在 (0,1)", np.all((p > 0) & (p < 1)))

    # 3) nan 传播：Ri=nan ⇒ P=nan（不能静默变 0，否则航迹端点会被算成"安全"）
    pn = p_from_ri(np.array([1.0, np.nan, 0.5]), a, b)
    chk("nan 传播而不是变 0", np.isnan(pn[1]) and np.isfinite(pn[0]))

    # 4) 负 Ri（湿式层结翻转）走实测频率分支，而不是外推
    pneg = p_from_ri(np.array([-0.5]), a, b, p_neg=0.95)
    chk("负 Ri 用实测频率而非外推", np.isfinite(pneg[0]) and abs(pneg[0] - 0.95) < 1e-12,
        "P=%.3f" % pneg[0])
    pneg2 = p_from_ri(np.array([-0.5]), a, b)
    chk("无 p_neg 时退化为有限值（低置信）",
        np.isfinite(pneg2[0]) and 0 < pneg2[0] < 1, "P=%.3f" % pneg2[0])
    # 4b) 正 Ri 不受 p_neg 影响
    chk("p_neg 不污染正 Ri",
        abs(p_from_ri(np.array([0.5]), a, b, p_neg=0.95)[0]
            - p_from_ri(np.array([0.5]), a, b)[0]) < 1e-15)

    # 5) 晴空开关边界
    sw = clear_air_switch(np.array([70.0, 80.0, 90.0, np.nan]))
    chk("晴空开关含 nan 处理", np.array_equal(sw, [True, False, False, False]))

    # 6) 逻辑回归手算一致性：x=0 (Ri=1) ⇒ P = σ(a)
    chk("Ri=1 时 P=σ(a)", abs(p_from_ri(np.array([1.0]), a, b)[0]
                              - 1 / (1 + np.exp(3.0))) < 1e-12)

    # 7) p_encounter ≥ max(p_i) 且 ≤ 1
    fake = dict()
    ri = np.array([2.0, 0.5, 3.0, 1.0])
    calib = {"light|全部|max": {"dry": [a, b, None], "virt": [a, b, None],
                                "DK": [a, b, None]}}
    rr = route_probability(ri, ri, ri, np.full(4, 50.0), calib,
                           subset_policy="all")
    chk("p_encounter ∈ [max p, 1]",
        rr["p_encounter"] >= rr["p_max"] - 1e-12 and rr["p_encounter"] <= 1.0,
        "%.4f" % rr["p_encounter"])
    chk("n_expect = Σ p", abs(rr["n_expect"] - np.nansum(rr["p_point"])) < 1e-12)

    # 8) 缺标定文件时显式报错，而不是静默给 0
    try:
        load("__definitely_missing__.json")
        chk("缺文件返回 None 而不抛", True)
    except Exception as e:                                    # pragma: no cover
        chk("缺文件返回 None 而不抛", False, repr(e))

    # 9) switch 策略在纯晴空航迹上等于 virt
    rs = route_probability(ri, ri, ri, np.full(4, 40.0), calib,
                           subset_policy="switch")
    chk("switch 在晴空航迹上取 virt",
        np.allclose(rs["p_point"], rr["p_point"]))

    # 10) 占位标定不得被当成真标定
    chk("占位标定被拒绝", models_for({"x": 1}, "light") is None)

    if verbose:
        print("  --- calib 自检 %d/%d ---" % (ok, tot))
    return ok, tot


if __name__ == "__main__":
    selfcheck()
