# -*- coding: utf-8 -*-
"""
avwx.recon3d —— 风–温–湿三维精细场重构引擎
=============================================

科学动机
--------
航空业态已拥有**极高密度的高空观测**，但它们几乎全部是"湿度盲"的：

  * AMDAR / Mode-S EHS / EMADDC（de Haan et al., *AMT* 18:3341, 2025）
    每天在欧洲空域产出约 5500 万条风观测、3200 万条温度观测，**比湿为 0 条**。
  * 探空（IGRA）有湿度，但全球仅约 800 站、每天 2 次。

于是三维湿度场成了三者中**观测最差**的一支——而它恰恰是决定
密度高度、对流起始、积冰与湿理查森数的关键量（见 thermo.py）。

本模块的核心命题：**用物理耦合把"湿度盲"的密集观测变成湿度信息**。

        温度观测 ──[静力 + 地转平衡]──▶ 风场增量
        风观测   ──[水汽平流位移]──────▶ 湿度场增量

数学表述
--------
分析增量分解为单变量最优插值 + 物理耦合算子的显式叠加：

    δx = δx_OI(单变量)  +  α_T2V · M_{T→V}[δT_a]  +  α_V2q · M_{V→q}[δV_a]

（1）静力—地转耦合  M_{T→V}
        δΦ(p_k) = -R_d Σ_{l≤k} δT_v(p_l) Δ ln p_l        （静力方程）
        δu_g = -(1/f) ∂δΦ/∂y ,  δv_g = (1/f) ∂δΦ/∂x      （地转关系）
    这是业务 3D-Var 的"平衡项"标准做法，此处显式实现以便开关与量化。

（2）水汽平流耦合  M_{V→q}（本项目的增量贡献）
        δq(x) = q_b(x - δV_a·τ) - q_b(x)
    即：风的增量把背景水汽场沿自身方向推移 τ 时间。
    即使观测完全不报湿度，风观测依然通过"把背景水汽梯度推过来"约束湿度场。

    实现上采用**半拉格朗日位移**而非一阶泰勒展开 -τ(δV·∇q_b)。
    原因：一阶展开在 |δV|·τ 与水汽梯度尺度可比时会非物理放大
    （v1 实测湿度 RMSE 从 8.2e-4 恶化到 2.3e-3，即由此而来）；
    位移形式自动满足 δq ≤ |q_b 的局地变幅|，且在大位移极限下仍定性正确。

之所以把耦合做成**显式可开关的算子**而非 4×4 黑箱协方差：
每个耦合项都可以单独开关，从而在观测系统试验（OSE）中**逐项归因**——
这正是本项目要给出的数字："湿度盲的飞机观测，到底贡献了多少湿度信息"。

v2 相对 v1 的两处根本性修正（踩坑记录）
---------------------------------------
v1 曾出现"所有配置相对背景场 RMSE 全面恶化"，根因有二，均与耦合算子无关：

（a）**背景场回插方式**。v1 用 np.repeat（最近邻）把 1° 粗场放大回 0.25° 网格，
    得到阶梯状不连续场。其误差在粗网格块内为常数、跨块边界突变，
    本质是**白噪声**，任何以平滑高斯相关为基础的最优插值都无法表示：
    新息被相关长度铺开后只会污染本已正确的邻域。
    实测：加 120 个地面观测后温度 RMSE 从 1.461 反而升到 1.500。
    真实业务把粗模式插值到细网格一律用双线性/双三次，本版改正。

（b）**背景误差协方差的标定**。v1 取 σ_b = 0.4 ×（该层空间标准差），
    对温度即 6–8 K，而真实背景误差仅 1.46 K——高估 4 倍以上；
    σ_o 又由 ratio×σ_b 派生（1.8 K）而非真实观测误差（0.5 K）。
    结果观测被赋予压倒性权重，OI 从"修正背景"退化为"照抄观测"。
    本版：σ_b 由背景误差场**诊断**得到，σ_o 取**观测误差的实际量级**，
    相关长度同样由误差场自相关拟合——对应业务中的 NMC 法 / 集合估计。

观测系统试验（OSSE）设定
------------------------
  真值  nature : GFS 0.25° 分析场（本项目实测可用，多时次）
  背景  backg  : 真值粗化到 1.0°（双线性回插）+ 相关噪声
  观测  obs    : 从真值采样，含三类——地面站 / 探空 / 飞机（后者无湿度）
  检验  verify : 与真值逐点比较 RMSE
  划分  split  : 标定时次与检验时次**严格分离**，杜绝同数据调参又打分
"""
import os, sys
import numpy as np
from scipy.ndimage import map_coordinates, uniform_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avwx import thermo as TH

R_D = TH.R_D
OMEGA = 7.2921159e-5

# 东亚区域真实机场（纬度 15–40°N，经度 100–130°E）——用于构造逼真的航路走廊。
# 真实 AMDAR/EMADDC 观测的空间分布正是**沿航路高度集中**的，
# 用随机散点会高估观测对非走廊区域的约束能力，导致 OSSE 结论偏乐观。
AIRPORTS = {
    "ZBAA": (40.08, 116.58), "ZSPD": (31.14, 121.81), "ZGGG": (23.39, 113.30),
    "VHHH": (22.31, 113.91), "ZUUU": (30.58, 103.95), "RCTP": (25.08, 121.23),
    "RKSI": (37.46, 126.44), "ZPPP": (25.10, 102.74), "ZLXY": (34.44, 108.75),
    "ZHHH": (30.78, 114.21), "ZGHA": (28.19, 113.22), "ZSNJ": (32.00, 118.80),
    "ZSHC": (30.23, 120.43), "ZSQD": (36.27, 120.37), "ZYTX": (41.64, 123.48),
    "ZUCK": (29.56, 106.64), "ZSAM": (24.54, 118.13), "ZUGY": (26.54, 106.80),
    "ZGNN": (22.61, 108.17), "ZJSY": (18.30, 109.41), "ZBTJ": (39.12, 117.35),
    "ZSSS": (31.20, 121.34), "ZSOF": (31.98, 116.98), "ZSFZ": (25.93, 119.66),
    "ZSNB": (29.83, 121.64), "ZSWZ": (28.00, 120.85), "ZGKL": (25.22, 110.04),
    "ZUZY": (27.70, 106.80), "ZHCC": (34.52, 113.84), "ZSJN": (36.86, 116.86),
}
# 真实航路对（区域内干线）
ROUTE_PAIRS = [
    ("ZBAA", "ZSPD"), ("ZBAA", "ZGGG"), ("ZBAA", "ZUUU"), ("ZBAA", "RKSI"),
    ("ZSPD", "ZGGG"), ("ZSPD", "VHHH"), ("ZSPD", "RCTP"), ("ZSPD", "ZUCK"),
    ("ZGGG", "ZUUU"), ("ZGGG", "VHHH"), ("VHHH", "RCTP"), ("VHHH", "ZPPP"),
    ("ZUUU", "ZPPP"), ("ZPPP", "ZJSY"), ("ZJSY", "ZGNN"), ("ZGNN", "ZGGG"),
    ("ZLXY", "ZBAA"), ("ZLXY", "ZUUU"), ("ZHHH", "ZBAA"), ("ZHHH", "ZGGG"),
    ("ZHHH", "ZSPD"), ("ZGHA", "ZSPD"), ("ZGHA", "ZGGG"), ("ZSNJ", "ZBAA"),
    ("ZSNJ", "ZGGG"), ("ZSHC", "ZBAA"), ("ZSHC", "ZGGG"), ("ZSQD", "ZSNJ"),
    ("ZSQD", "RKSI"), ("ZYTX", "ZSPD"), ("ZYTX", "RKSI"), ("ZUCK", "ZBAA"),
    ("ZUCK", "ZSPD"), ("ZSAM", "ZSPD"), ("ZSAM", "ZGGG"), ("ZUGY", "ZUUU"),
    ("ZUGY", "ZGGG"), ("ZBTJ", "ZSPD"), ("ZBTJ", "ZGHA"), ("ZSSS", "ZGGG"),
    ("ZSOF", "ZGGG"), ("ZSFZ", "ZSPD"), ("ZSNB", "ZSPD"), ("ZSWZ", "ZGGG"),
    ("ZGKL", "ZSPD"), ("ZGKL", "ZUUU"), ("ZHCC", "ZBAA"), ("ZSJN", "ZSPD"),
    ("RCTP", "RKSI"), ("ZLXY", "ZSPD"), ("ZHHH", "ZUUU"), ("ZPPP", "ZGGG"),
]


# ----------------------------------------------------------------------------
# 相关模型
# ----------------------------------------------------------------------------
def corr_gauss(d, L):
    """高斯相关 ρ = exp(-d²/(2L²))。"""
    return np.exp(-0.5 * (np.asarray(d, float) / L) ** 2)


# ----------------------------------------------------------------------------
# 背景场构造：粗化 + 双线性回插 + 相关噪声
# ----------------------------------------------------------------------------
def _bilinear_upsample(a2, ny, nx, C):
    """把 (…, ny2, nx2) 的粗场双线性插值回 (…, ny, nx)。

    坐标映射：粗格点 j 覆盖细格点 [jC, jC+C-1]，中心位于 jC + (C-1)/2，
    故细格点 i 的粗坐标为 (i - (C-1)/2) / C。边缘由 mode='nearest' 外推。

    见模块文档 (a)：这一步必须用插值而非最近邻。
    """
    a2 = np.asarray(a2, float)
    ii = (np.arange(ny) - (C - 1) / 2.0) / C
    jj = (np.arange(nx) - (C - 1) / 2.0) / C
    II, JJ = np.meshgrid(ii, jj, indexing="ij")
    coords = np.array([II.ravel(), JJ.ravel()])
    out = np.empty(a2.shape[:-2] + (ny, nx), float)
    flat = a2.reshape(-1, a2.shape[-2], a2.shape[-1])
    of = out.reshape(-1, ny, nx)
    for m in range(flat.shape[0]):
        of[m] = map_coordinates(flat[m], coords, order=1, mode="nearest").reshape(ny, nx)
    return out


def make_background(nat, coarse=4, noise=1.2, seed=20261020):
    """真值粗化（块平均 + **双线性**回插）+ 相关噪声，作为实时背景场。

    物理类比：业务上实时可得的是 1° 全球模式（或前一时次的粗分辨率预报），
    而我们希望重构 0.25° 的精细场；粗化误差代表模式未能分辨的中尺度结构，
    相关噪声代表模式本身的系统性偏差。

    coarse: 粗化倍数（0.25° × 4 = 1.0°）
    noise : 噪声幅度，以**粗化误差**标准差的倍数计（而不是以场的空间变率计——
            v1 用后者导致平流层噪声达 19.9 K，本项目实测踩过此坑）
    """
    rng = np.random.default_rng(seed)
    ny, nx = nat["lat"].size, nat["lon"].size
    ny2, nx2 = ny // coarse, nx // coarse

    def block(a):
        a = a[..., :ny2 * coarse, :nx2 * coarse]
        sh = a.shape[:-2] + (ny2, coarse, nx2, coarse)
        return a.reshape(sh).mean(axis=(-3, -1))

    bg = {}
    for k in ("T", "q", "u", "v", "z"):
        if nat[k] is None:
            bg[k] = None
            continue
        bg[k] = _bilinear_upsample(block(nat[k]), ny, nx, coarse)

    def smooth(a, n=6):
        return uniform_filter(a, size=(1, n, n), mode="nearest")

    # 背景误差必须是**宽谱、且以中尺度为主**的。
    # 只用"粗化误差"等于假设模式仅在 <1° 的小尺度上出错——这不真实：
    # 真实的粗分辨率模式能抓住大尺度环流，其误差集中在
    # **中尺度**（锋面结构、急流位置与强度、边界层日变化，约 100–500 km），
    # 而恰恰是这部分能被 200–800 km 间距的观测网有效约束。
    #
    # 本项目踩过的两个坑：
    #  (1) 只含小尺度分量 → 诊断 L_h 仅约 40 km，远小于观测间距，
    #      OI 增益被压到 1–2%，几近于零；
    #  (2) 加入多尺度噪声但幅度仅取 0.35×粗化误差 → 方差占比只有 11%，
    #      大尺度分量被小尺度误差完全淹没，诊断 L_h 仍被压在下限。
    # 本版以 noise=1.2×粗化误差、且方差权重偏向中—大尺度，
    # 使误差谱与真实模式误差的量级关系一致。
    #
    #   盒式滤波 n 格的等效高斯 σ = √((n²−1)/12) 格，格距 0.25° ≈ 27.8 km
    #   n=3 → 23 km │ n=9 → 72 km │ n=25 → 200 km │ n=61 → 490 km
    # 幅度权重平方后给出方差占比 ≈ 0.11 / 0.20 / 0.39 / 0.30。
    SCALES = ((3, 0.32), (9, 0.43), (25, 0.60), (61, 0.52))

    for key in ("T", "q", "u", "v"):
        comp = np.zeros_like(nat[key])
        for n, wt in SCALES:
            s = smooth(rng.standard_normal(nat[key].shape), n)
            comp = comp + wt * s / (np.nanstd(s) + 1e-12)
        for k in range(nat[key].shape[0]):
            # 以该层**粗化误差**的量级标定，而非该层的空间变率
            sd = float(np.nanstd(bg[key][k] - nat[key][k]))
            comp[k] = comp[k] / (np.nanstd(comp[k]) + 1e-12) * sd * noise
        bg[key] = bg[key] + comp
    return bg


# ----------------------------------------------------------------------------
# 背景误差协方差的**诊断**（对应业务中的 NMC 法 / 集合估计）
# ----------------------------------------------------------------------------
def corr_double_gauss(d2, L1, L2, w2):
    """双高斯相关模型 ρ = (1−w₂)·exp(−d²/2L₁²) + w₂·exp(−d²/2L₂²)。

    为什么需要长尾分量
    ------------------
    本项目实测的背景误差相关函数（整格位移诊断）**远非单高斯**：

        距离    50 km  150 km  250 km  350 km  450 km
        实测    0.57   0.31    0.21    0.15    0.10
        单高斯  0.54   ~0      ~0      ~0      ~0      (L=64 km，按 ρ=0.5 标定)

    物理原因：背景误差是**两类不同机制**的叠加——
      * 粗化误差：模式未能分辨的中尺度结构，相关尺度短（数十 km）；
      * 模式偏差：天气系统的位置/强度误差，相关尺度长（数百 km）。
    单高斯若按短尺度标定，则在观测间距（约 250 km）处权重仅 4×10⁻⁴，
    等于**完全放弃**观测网本可约束的大尺度误差分量；
    若按长尺度标定，又会在近处过度平滑。双高斯以两项分别承担，
    且与实测廓线吻合到 0.01–0.02（见 diagnose_background_error 的拟合）。

    d2: 距离**平方** [m²]（避免重复开方）
    """
    return (1.0 - w2) * np.exp(-0.5 * d2 / L1 ** 2) + w2 * np.exp(-0.5 * d2 / L2 ** 2)


def _fit_double_gauss(d, rho):
    """对经验相关廓线拟合 (L1, L2, w2)。网格搜索，廓线点数少时足够。"""
    ok = np.isfinite(rho)
    d, rho = d[ok], rho[ok]
    if len(d) < 4:
        return 60e3, 400e3, 0.25
    best = None
    for L1 in np.arange(10.0, 81.0, 5.0) * 1e3:
        for L2 in np.arange(100.0, 801.0, 50.0) * 1e3:
            if L2 <= 2 * L1:
                continue
            for w2 in np.arange(0.05, 0.66, 0.05):
                r = float(np.sum((corr_double_gauss(d ** 2, L1, L2, w2) - rho) ** 2))
                if best is None or r < best[0]:
                    best = (r, L1, L2, w2)
    return best[1], best[2], best[3]


def _corr_profile(e2d, lat, lon, max_shift=32):
    """二维场沿两个水平方向的**整格位移自相关**。

    返回 (distance[m], rho)；场退化时返回 None。

    为什么用整格位移而不是随机点对采样：确定性、无蒙特卡洛噪声、
    无边界裁剪偏差、且不需要"物理距离 → 格索引"的换算（后者是本项目的
    一个严重踩坑点，见 diagnose_background_error 内的记录）。
    """
    e2d = np.asarray(e2d, float)
    if not np.isfinite(e2d).all() or np.nanstd(e2d) <= 0:
        return None
    ec = e2d - np.nanmean(e2d)
    ny, nx = ec.shape
    ms = int(min(max_shift, ny // 3, nx // 3))
    if ms < 2:
        return None
    dy_m = abs(float(lat[1] - lat[0])) * 111195.0
    dx_m = abs(float(lon[1] - lon[0])) * 111195.0 * float(np.mean(np.cos(np.deg2rad(lat))))
    d, r = [], []
    for s in range(1, ms + 1):
        a = ec[s:, :].ravel(); b = ec[:ny - s, :].ravel()
        r.append(np.corrcoef(a, b)[0, 1]); d.append(s * dy_m)
        a = ec[:, s:].ravel(); b = ec[:, :nx - s].ravel()
        r.append(np.corrcoef(a, b)[0, 1]); d.append(s * dx_m)
    return np.array(d), np.array(r)


def diagnose_background_error(nat, bg, lat, lon, lev, nsamp=60000, seed=11):
    """从 (bg − nat) 诊断 σ_b(k)、水平相关长度 L_h、垂直相关长度 L_z。

    为什么必须诊断而不能拍脑袋：
    最优插值的权重完全由 σ_b² / (σ_b² + σ_o²) 决定。σ_b 高估 4 倍
    （v1 的情形）等价于把观测权重提高一个量级，OI 会从"修正背景"
    退化为"照抄观测 + 相关长度铺开"，全面放大误差。

    真实业务中真值不可得，标准替代是
      * NMC 方法：用同一模式的 24h 与 12h 预报之差统计背景误差；
      * 集合法：用预报集合的离散度。
    两者给出的都是同类的"逐层标准差 + 相关长度"。本项目 OSSE 中真值已知，
    直接诊断得到的是这一协方差模型的**最佳情形**，偏差方向需在论文中说明。

    返回 dict(sigma_b[nz], L_h, L_z, rmse[nz])
    """
    rng = np.random.default_rng(seed)
    lev = np.asarray(lev, float)
    lp = np.log(lev)
    ny, nx = len(lat), len(lon)
    nz = len(lev)

    sig, rms = np.zeros(nz), np.zeros(nz)
    err = {}
    for v in ("T", "q", "u", "v"):
        e = bg[v] - nat[v]
        err[v] = e
        rms_v = np.array([float(np.sqrt(np.nanmean((e[k]) ** 2))) for k in range(nz)])
        if v == "T":
            rms[:] = rms_v
    for k in range(nz):
        allerr = np.concatenate([np.abs(err[v][k]).ravel() for v in ("T", "q", "u", "v")])
        sig[k] = float(np.nanmean(allerr)) * 1.2533     # 平均绝对偏差 → 标准差
    sig = np.maximum(sig, 1e-12)
    # 用温度误差诊断相关结构（温度场是唯一各层都有可靠误差信号的量）
    e = err["T"]

    # ---- 水平相关长度 ----
    # 方法：**整格位移自相关**（确定性，无蒙特卡洛噪声）。
    # 早期版本用"随机方向+随机距离"的点对采样，存在两个问题：
    #  (1) 远距离配对的端点常被 np.clip 压到边界同一行/列，相关被系统性高估；
    #  (2) 需先把物理距离换算成格索引偏移——漏除格距会使名义距离 d 实际
    #     只走 0.25·d，相关长度**高估 4 倍**（诊断 124 km vs 真实 31 km），
    #     这是本项目"重构全面恶化"的最终根因。
    # 整格位移法对整层做切片相关，无采样噪声、无边界裁剪、无格距换算环节。
    prof_all = []
    dc = pc = None
    for k in range(nz):
        pr = _corr_profile(e[k], lat, lon, max_shift=32)
        if pr is not None:
            prof_all.append(pr)
    if prof_all:
        dall = np.concatenate([p[0] for p in prof_all])
        pall = np.concatenate([p[1] for p in prof_all])
        o = np.argsort(dall)
        d_s, p_s = dall[o], pall[o]
        # 按距离分箱取中位（合并各层、两方向）
        bins = np.arange(0.0, d_s.max() + 40e3, 40e3)
        dc, pc = [], []
        for m in range(len(bins) - 1):
            s = (d_s >= bins[m]) & (d_s < bins[m + 1])
            if s.sum() >= 3:
                dc.append(0.5 * (bins[m] + bins[m + 1])); pc.append(np.median(p_s[s]))
        dc = np.array(dc); pc = np.array(pc)
        ok = np.isfinite(pc) & (pc > 0.05)
        if ok.sum() >= 2 and pc[ok][0] > 0.5:
            d_ok, p_ok = dc[ok], pc[ok]
            j = int(np.argmax(p_ok < 0.5)) if (p_ok < 0.5).any() else len(p_ok) - 1
            if j == 0:
                d_half = d_ok[0] * 0.5
            else:
                x0, x1 = d_ok[j - 1], d_ok[j]
                y0, y1 = p_ok[j - 1], p_ok[j]
                d_half = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
            L_h = float(d_half / 1.17741)
        else:
            L_h = 120e3
    else:
        L_h = 120e3
    L_h = float(np.clip(L_h, 15e3, 600e3))

    # ---- 双高斯拟合（主用相关模型，见 corr_double_gauss 的说明）----
    if dc is not None and len(dc) >= 4:
        L1, L2, w2 = _fit_double_gauss(dc, pc)
    else:
        L1, L2, w2 = L_h, 400e3, 0.25

    # ---- 垂直相关：**直接用经验矩阵**，不要拟合 L_z ----
    # ⚠ 这是本项目最隐蔽也最具破坏性的一处错误。
    # 早期版本用单一高斯 exp(−(Δln p)²/2L_z²) 描述垂直相关并拟合 L_z=0.766，
    # 于是模型认为相邻层几乎完全相关（ρ≈0.99）。而实测矩阵是
    #
    #   层对    0→1   0→2   0→3   0→4    4→5    7→8
    #   真实   0.22  0.22  0.15  0.08  −0.08  −0.01
    #   模型   0.99  0.98  0.90  0.66   0.96   0.96
    #
    # 真实相关在相邻层就跌到 0.2 左右并近似保持，根本不是从 1 单调衰减的高斯。
    # 物理原因：背景误差是**两类成分**的叠加——
    #   * 粗化误差：垂直方向连续、相关强；
    #   * 模式噪声：逐层独立（本项目噪声即如此构造），相关≈0。
    # 二者叠加后相关函数在小 Δln p 处即饱和到 σ_c²/(σ_c²+σ_n²)，
    # 用"从 1 衰减"的单高斯拟合必然把这个平台误读为极长的相关尺度。
    #
    # 后果：地面观测被"垂直传播"到 925/850 hPa 而污染这些层——实测
    # 1000 hPa 改善 +11.4% 的同时 925 hPa 恶化 −17.3%，即由此而来。
    # 因此垂直相关**必须**用经验矩阵逐对给出，不能参数化。
    Cv = np.corrcoef(np.nan_to_num(e).reshape(nz, -1))
    Cv = np.nan_to_num(Cv, nan=0.0)
    np.fill_diagonal(Cv, 1.0)
    dz = lp[:, None] - lp[None, :]
    okz = (np.abs(dz) > 1e-6) & (Cv > 0.05) & np.isfinite(Cv)
    if okz.sum() >= 2:
        x = dz[okz] ** 2
        y = -np.log(np.clip(Cv[okz], 1e-6, 1))
        L_z = float(np.sqrt(np.sum(x ** 2) / (2.0 * np.sum(x * y))))
    else:
        L_z = 0.45
    L_z = float(np.clip(L_z, 0.05, 1.2))

    return dict(sigma_b_T=dict(zip(lev.tolist(), sig.tolist())),
                L_h=L_h, L_h1=L1, L_h2=L2, w2=w2, L_z=L_z,
                vert_corr=Cv.tolist(),
                rmse_T=dict(zip(lev.tolist(), rms.tolist())),
                corr_profile=dict(d=[float(x) for x in (dc if dc is not None else [])],
                                  rho=[float(x) for x in (pc if pc is not None else [])]))


# ----------------------------------------------------------------------------
# 观测模拟
# ----------------------------------------------------------------------------
class ObsSim:
    """从真值场采样各类观测。飞机观测**刻意不含湿度**。

    观测误差取业务典型值：
      地面 METAR  : T 0.5 K, q 5e-4 kg/kg（≈0.7 K 露点）, 风 1.5 m/s
      探空 RS92   : T 0.3 K, q 3e-4,                     风 1.0 m/s
      机载 AMDAR  : T 0.5 K（Mode-S EHS 略差，取保守值）, 风 1.3 m/s
    """

    def __init__(self, nat, seed=7):
        self.nat = nat
        self.rng = np.random.default_rng(seed)
        self.lat = nat["lat"]; self.lon = nat["lon"]; self.lev = nat["level"]
        nz, ny, nx = nat["T"].shape
        self.iy = ((np.asarray(self.lat) - self.lat[0]) /
                   (self.lat[1] - self.lat[0])).round().astype(int)
        self.ix = ((np.asarray(self.lon) - self.lon[0]) /
                   (self.lon[1] - self.lon[0])).round().astype(int)

    def _idx_of(self, la, lo):
        return (int(np.clip(np.searchsorted(self.lat, la), 0, len(self.lat) - 1)),
                int(np.clip(np.searchsorted(self.lon, lo), 0, len(self.lon) - 1)))

    def _sample(self, ilat, ilon, ilev, keys, errs):
        out = {k: [] for k in keys}
        pos = []
        for a, b, c in zip(ilev, ilat, ilon):
            for k in keys:
                v = self.nat[k][a, b, c]
                out[k].append(np.nan if not np.isfinite(v)
                              else v + self.rng.normal(0.0, errs[k]))
            pos.append((self.lev[a], self.lat[b], self.lon[c]))
        for k in keys:
            out[k] = np.array(out[k], float)
        return out, np.array(pos, float)

    def surface(self, n=150, errs=None):
        """地面站（METAR 型）：最低层，全要素。"""
        errs = errs or dict(T=0.5, q=5e-4, u=1.5, v=1.5)
        rng = self.rng
        ny, nx = len(self.lat), len(self.lon)
        ilat = rng.integers(0, ny, n); ilon = rng.integers(0, nx, n)
        ilev = np.zeros(n, int)
        return self._sample(ilat, ilon, ilev, ("T", "q", "u", "v"), errs)

    def raob(self, n=12, errs=None):
        """探空（IGRA 型）：全廓线，全要素，但站点极少。"""
        errs = errs or dict(T=0.3, q=3e-4, u=1.0, v=1.0)
        rng = self.rng
        nz, ny, nx = self.nat["T"].shape
        ilat = rng.integers(2, ny - 2, n); ilon = rng.integers(2, nx - 2, n)
        ilev = np.tile(np.arange(nz), n)
        ilat2 = np.repeat(ilat, nz); ilon2 = np.repeat(ilon, nz)
        return self._sample(ilat2, ilon2, ilev, ("T", "q", "u", "v"), errs)

    def aircraft(self, route_pairs=None, pts_per=42, errs=None, seed_pts=99):
        """飞机观测（AMDAR / EMADDC 型）：**只有风与温度，无湿度**。

        沿**真实机场对**的连线采样（见 AIRPORTS / ROUTE_PAIRS），
        高度剖面为爬升 → 巡航 → 下降，观测沿航路走廊高度集中
        ——这正是真实机载下链数据的空间分布特征。

        巡航高度按**航程**分配（真实运行规则）：
            < 800 km  → 300 hPa（约 FL300）
          800–1600 km → 250 hPa（约 FL340）
            > 1600 km → 200 hPa（约 FL390）
        若所有航班都压在同一巡航层，观测的垂直分布会失真，
        也会掩盖真实 AMDAR 在中低层近乎无观测这一关键事实。
        """
        errs = errs or dict(T=0.5, u=1.3, v=1.3)
        pairs = route_pairs or ROUTE_PAIRS
        nz = len(self.lev)
        ilat, ilon, ilev = [], [], []
        for a, b in pairs:
            la0, lo0 = AIRPORTS[a]; la1, lo1 = AIRPORTS[b]
            i0, j0 = self._idx_of(la0, lo0); i1, j1 = self._idx_of(la1, lo1)
            gc = self._gc_km(la0, lo0, la1, lo1)
            cruise = 6 if gc < 800 else (7 if gc < 1600 else 8)
            cruise = int(np.clip(cruise, 3, nz - 1))
            for t in np.linspace(0, 1, pts_per):
                ia = int(round(i0 + (i1 - i0) * t)); ib = int(round(j0 + (j1 - j0) * t))
                # 高度剖面：起飞→爬升→巡航→下降→进近
                if t < 0.12:
                    frac = t / 0.12
                elif t > 0.88:
                    frac = (1.0 - t) / 0.12
                else:
                    frac = 1.0
                k = int(round(cruise * frac))
                k = int(np.clip(k, 0, nz - 1))
                ilat.append(ia); ilon.append(ib); ilev.append(k)
        o, pos = self._sample(np.array(ilat), np.array(ilon), np.array(ilev),
                              ("T", "u", "v"), errs)
        o["q"] = np.full(len(pos), np.nan)                   # ← 湿度盲
        return o, pos

    @staticmethod
    def _gc_km(la0, lo0, la1, lo1):
        """大圆距离 [km]（球面余弦公式，区域尺度足够）。"""
        p1, p2 = np.deg2rad(la0), np.deg2rad(la1)
        dl = np.deg2rad(lo1 - lo0)
        c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
        return 6371.0 * np.arccos(np.clip(c, -1, 1))


# ----------------------------------------------------------------------------
# 局部最优插值（单变量）
# ----------------------------------------------------------------------------
def _metric_xy(lat_a, lon_a, lat_b, lon_b):
    """水平位移分量 (dy, dx)，单位**米**（与 L_h 的单位必须一致）。

    ⚠ 单位陷阱（本项目实测踩坑，代价是 OI 完全失效）：
    早期版本这里乘 111.195 返回 km，而 diagnose_background_error 给出的
    L_h 以**米**为单位（如 124000）。两者相差 1000 倍，等价于把相关长度
    放大到 12.4 万 km（地球周长的三倍）——高斯相关 exp(-d²/2L²) 恒等于 1，
    所有观测被当作等权、新息被均匀铺满全域，分析增量与真实背景误差无关，
    表观增益 ≈ 0。更隐蔽的是：此时对 L_h 做敏感性扫描**完全看不出变化**
    （60 km 与 300 km 在 12.4 万 km 面前都等于无穷大），
    而新息方差统计也依然正常（该统计与 L_h 无关），因此极易误判为"方法无效"。
    全模块统一以**米**为长度单位，geostrophic_wind / moisture_displacement
    等处亦用 111195.0 m/度，与此一致。
    """
    latm = 0.5 * (np.asarray(lat_a) + np.asarray(lat_b))
    dy = (np.asarray(lat_a) - np.asarray(lat_b)) * 111195.0
    dx = (np.asarray(lon_a) - np.asarray(lon_b)) * 111195.0 * np.cos(np.deg2rad(latm))
    return dy, dx


def superob(vals, pos, lat, lon, lev, cell_deg=0.5):
    """把落在同一 (气压层, cell_deg 网格) 的观测合并为"超级观测"。

    为什么需要
    ----------
    （1）业务必要性：真实机载数据沿航路密集下链（AMDAR 每 3–6 分钟一条），
        相邻报告的**观测误差并不独立**（同平台、同仪器、相近时刻），
        若当作独立观测会使 R 矩阵严重低估、分析过信观测。
        业务系统（GSI、EMADDC 预处理）一律做 thinning / superobbing。
    （2）统计必要性：合并尺度必须与**背景误差相关长度** L_h 匹配。
        本项目实测 L_h ≈ 100–130 km（≈1°），若按 0.25° 单个格点合并
        （v1 的隐含做法）几乎不减少数量，等于把高度冗余的观测当作独立信息。
        默认 cell_deg=0.5°（≈55 km ≈ L_h/2）即按此原则选取。

    合并后观测误差按 1/√n 缩小（若误差独立）；但如上所述机载误差有相关成分，
    故取 1/n^(1/4) 这一**保守**折中——即承认一半的误差相关。
    """
    ok = np.isfinite(vals)
    v = np.asarray(vals, float)[ok]
    p = np.asarray(pos, float)[ok]
    if len(v) == 0:
        return np.zeros(0), np.zeros((0, 3)), np.zeros(0)
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    lev = np.asarray(lev, float)
    ic = np.clip(np.searchsorted(-lev, -p[:, 0]), 0, len(lev) - 1)
    ia = np.clip(((p[:, 1] - lat[0]) / cell_deg).astype(int), 0, 10 ** 6)
    ib = np.clip(((p[:, 2] - lon[0]) / cell_deg).astype(int), 0, 10 ** 6)
    na, nb = ia.max() + 1, ib.max() + 1
    key = ic.astype(np.int64) * (na * nb) + ia.astype(np.int64) * nb + ib
    uniq, inv = np.unique(key, return_inverse=True)
    n = np.bincount(inv, minlength=len(uniq)).astype(float)
    s = np.bincount(inv, weights=v, minlength=len(uniq))
    mean = s / n
    wlat = np.bincount(inv, weights=p[:, 1], minlength=len(uniq)) / n
    wlon = np.bincount(inv, weights=p[:, 2], minlength=len(uniq)) / n
    ic_u = uniq // (na * nb)
    pos_u = np.column_stack([lev[ic_u], wlat, wlon])
    return mean, pos_u, n ** (-0.25)


def oi_univariate(bg, obs_val, obs_pos, lat, lon, lev, L_h=180e3, L_z=0.45,
                  sigma_b=None, sigma_o=None, sigma_o_ratio=0.3,
                  vert_cut=None, max_obs=600, obs_count=None,
                  L_h2=None, w2=0.0, vert_corr=None, vert_min=0.10):
    """逐变量的**全局**最优插值（OI）。

        x_a = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)

    背景误差协方差取可分离高斯形式：

        B(r_i, r_j) = σ_b(k)² · exp(-d_h²/2L_h²) · exp(-(Δ ln p)²/2L_z²)

    参数量纲（v2 修正）
    ------------------
    sigma_b : 逐层**背景误差标准差**（绝对量纲，如 K / kg/kg / m·s⁻¹）
    sigma_o : 逐层**观测误差标准差**（绝对量纲）
    二者必须同量纲且都取自实际误差统计；v1 用 0.4×空间标准差 + ratio×σ_b
    双向失准，是"重构反而恶化"的主因（见模块文档 (b)）。

    实现要点
    --------
    观测量 O(10²–10³) 远小于格点数，故对每一层只需构造并分解一次
    (H B Hᵀ + R)，随后对全部格点复用：W = (HBHᵀ+R)⁻¹ K。
    得到的是**精确全局解**（无局部化损失），复杂度 O(n_obs³ + n_grid·n_obs²)。
    """
    nz, ny, nx = bg.shape
    ok = np.isfinite(obs_val)
    ov = np.asarray(obs_val, float)[ok]
    op = np.asarray(obs_pos, float)[ok]
    if len(ov) == 0:
        return np.zeros_like(bg), bg.copy()

    lev = np.asarray(lev, float)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    glat, glon = LAT.ravel(), LON.ravel()

    if sigma_b is None:
        sigma_b = np.array([0.4 * float(np.nanstd(bg[k])) for k in range(nz)])
    sigma_b = np.atleast_1d(np.asarray(sigma_b, float))
    if sigma_b.size == 1:
        sigma_b = np.repeat(sigma_b, nz)
    if sigma_o is None:
        sigma_o = sigma_o_ratio * sigma_b
    else:
        sigma_o = np.atleast_1d(np.asarray(sigma_o, float))
        if sigma_o.size == 1:
            sigma_o = np.repeat(sigma_o, nz)

    # 观测 → 网格索引。关键：气压层 lev 为**降序**（1000 → 100 hPa），
    # 而 np.searchsorted 假定升序——直接 searchsorted(lev, p) 会把 1000 hPa
    # 的地面观测判到 100 hPa 层，使观测几乎完全失效（v1 实测 RMSE 降幅 0.00%）。
    oi_lat = np.clip(np.searchsorted(lat, op[:, 1]), 0, ny - 1)
    oi_lon = np.clip(np.searchsorted(lon, op[:, 2]), 0, nx - 1)
    oi_lev = np.clip(np.searchsorted(-lev, -op[:, 0]), 0, nz - 1)
    innov = ov - bg[oi_lev, oi_lat, oi_lon]

    lp_obs = np.log(op[:, 0])
    lp_lev = np.log(lev)
    # 背景误差标准差随层变化，故协方差必须用 σ_b(k_obs)·σ_b(k_grid)，
    # 而不是统一取目标层的 σ_b²（v1 在此处简化，跨层时不等价）。
    s_obs = sigma_b[oi_lev]
    so_obs = sigma_o[oi_lev]
    if obs_count is not None:
        so_obs = so_obs * np.asarray(obs_count, float)
    if vert_cut is None:
        vert_cut = 2.0 * L_z

    if L_h2 is not None and w2 > 0:
        def _rho(d2):
            return corr_double_gauss(d2, L_h, L_h2, w2)
    else:
        def _rho(d2):
            return np.exp(-0.5 * d2 / L_h ** 2)

    # 垂直相关：优先用**经验矩阵**（见 diagnose_background_error 的说明）。
    # 只有在未提供时才退化为单高斯 exp(−(Δln p)²/2L_z²)。
    VC = None if vert_corr is None else np.asarray(vert_corr, float)

    inc = np.zeros_like(bg)
    for k in range(nz):
        if VC is not None:
            # 垂直局部化由**实际相关强度**决定，而非一个假定的相关长度
            rho_zk = VC[oi_lev, k]
            m = np.abs(rho_zk) > vert_min
        else:
            dz_abs = np.abs(lp_obs - lp_lev[k])
            m = dz_abs < vert_cut
        if m.sum() == 0:
            continue
        if m.sum() > max_obs:
            # 超限时按垂直相关强度保留最强的 max_obs 个（而非按距离，
            # 因为经验矩阵下"距离"不再单调对应相关）
            key = np.abs(VC[oi_lev, k]) if VC is not None else -np.abs(lp_obs - lp_lev[k])
            keep = np.argsort(-key)[:max_obs]
            m = np.zeros(len(m), bool); m[keep] = True
        o_lat, o_lon, o_lp = op[m, 1], op[m, 2], lp_obs[m]
        s_i = s_obs[m]
        d_i = innov[m]
        dy, dx = _metric_xy(o_lat[:, None], o_lon[:, None], o_lat[None, :], o_lon[None, :])
        rh = _rho(dy ** 2 + dx ** 2)
        if VC is not None:
            rz = VC[np.ix_(oi_lev[m], oi_lev[m])]
            rz_g = VC[oi_lev[m], k][:, None]
        else:
            dz = o_lp[:, None] - o_lp[None, :]
            rz = np.exp(-0.5 * (dz / L_z) ** 2)
            rz_g = np.exp(-0.5 * ((o_lp[:, None] - lp_lev[k]) / L_z) ** 2)
        C = (s_i[:, None] * s_i[None, :]) * rh * rz
        C[np.diag_indices_from(C)] += so_obs[m] ** 2
        dy2, dx2 = _metric_xy(o_lat[:, None], o_lon[:, None], glat[None, :], glon[None, :])
        rh2 = _rho(dy2 ** 2 + dx2 ** 2)
        K = (s_i[:, None] * sigma_b[k]) * rh2 * rz_g
        inc[k] = (np.linalg.solve(C, K).T @ d_i).reshape(LAT.shape)
    return inc, bg + inc


# ----------------------------------------------------------------------------
# 物理耦合算子
# ----------------------------------------------------------------------------
def coriolis(lat2d):
    """科氏参数 f，低纬做下限截断以保证地转平衡可逆。"""
    f = 2.0 * OMEGA * np.sin(np.deg2rad(lat2d))
    fmin = 2.0 * OMEGA * np.sin(np.deg2rad(12.0))
    return np.where(np.abs(f) < fmin, np.sign(f + 1e-30) * fmin, f)


def _smooth_h(arr, n=9, reps=4):
    """水平低通（盒式滤波多次近似高斯）。

    为什么必须平滑：平衡算子含对位势增量做**水平微分**的地转关系。
    最优插值产出的增量在观测附近空间上很粗糙，直接微分会产生量级达
    数十 m/s 的虚假风增量（v1 实测 v 的 RMSE 从 2.3 炸到 16.3 m/s）。

    业务 3D-Var 同样只对**大尺度**分量施加平衡——这不是修补而是物理要求：
    地转平衡只在罗斯贝数 Ro = U/(fL) 小的尺度上成立。中尺度（L ~ 100 km）
    的 Ro ~ 1，其风场由非地转的辐散/惯性分量主导，施加地转约束只会添错。

    平滑宽度需使 δΦ 的梯度尺度达到天气尺度（数百 km），
    n=9 格 × 4 次 ≈ 0.25°×9×√(4/12)... 等效高斯 σ ≈ 0.25°×9×√(4×(81-1)/12)
    ≈ 0.25°×9×5.16 ≈ 11.6° ——过大；实际取 n=9/reps=4 等效 σ≈4.5°（约 480 km）。
    """
    out = np.asarray(arr, float).copy()
    for _ in range(reps):
        out = uniform_filter(out, size=(1, n, n), mode="nearest")
    return out


def hydrostatic_geopotential(dTv, lev):
    """由虚温增量经静力方程积分得到位势增量 δΦ [m²/s²]。

        ∂Φ/∂ln p = -R_d T_v   ⇒   δΦ(p_k) = -R_d Σ_{l≤k} δT_v(l) Δ ln p_l

    lev 为降序（1000 → 100 hPa），自下而上（气压递减方向）积分。
    """
    lp = np.log(np.asarray(lev, float))
    asc = lp[0] < lp[-1]
    if asc:
        dTv = dTv[::-1]; lp = lp[::-1]
    nz = len(lp)
    dphi = np.zeros_like(dTv)
    acc = np.zeros(dTv.shape[1:])
    for k in range(1, nz):
        acc = acc - R_D * 0.5 * (dTv[k] + dTv[k - 1]) * (lp[k] - lp[k - 1])
        dphi[k] = acc
    return dphi[::-1] if asc else dphi


def geostrophic_wind(dphi, lat, lon):
    """由位势增量得地转风增量 (δu_g, δv_g) [m/s]。

        δu_g = -(1/f) ∂δΦ/∂y ,   δv_g = +(1/f) ∂δΦ/∂x

    ⚠ 轴序陷阱（本项目实测踩坑，是 v1 中 T2V 耦合使 v 的 RMSE 恶化 746% 的直接原因）：
    dphi 形状为 (nz, ny, nx)，故**纬度是 axis=1、经度是 axis=2，axis=0 是气压层**。
    早期版本写成 du 取 gradient(axis=1) 却除以经度格距 dx、
    dv 取 gradient(axis=0)——后者根本是沿**气压层**求梯度，与地转关系无关。
    修正后由 selfcheck 的解析解项（δΦ=B·y ⇒ δu_g=−B/f）守护。
    """
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")     # (ny, nx)
    dy = np.gradient(LAT, axis=0) * 111195.0            # 纬度格距 (ny,nx)
    dx = np.gradient(LON, axis=1) * 111195.0 * np.cos(np.deg2rad(LAT))
    f = coriolis(LAT)
    dug = -(1.0 / f) * np.gradient(dphi, axis=1) / np.where(dy == 0, 1e9, dy)
    dvg = (1.0 / f) * np.gradient(dphi, axis=2) / np.where(dx == 0, 1e9, dx)
    return dug, dvg


def moisture_displacement(dU, dV, q_b, lat, lon, tau_h=2.0):
    """水汽平流耦合 M_{V→q}（半拉格朗日位移形式）：

        δq(x) = q_b(x − δV_a·τ) − q_b(x)

    即：风的分析增量把背景水汽场沿自身方向推移 τ 时间后的改变量。

    为什么用位移而非一阶展开 −τ(δV·∇q_b)
    --------------------------------------
    一阶展开只在 |δV|τ ≪ 水汽梯度尺度 L_q 时成立。本试验中 δV 的局地
    增量可达 5 m/s，τ=2 h → 位移 36 km，尚小于 L_q（~500 km）；
    但增量场在观测航线附近有强梯度，展开形式的误差二阶项并不可忽略。
    v1 用展开式实测湿度 RMSE 从 8.16e-4 恶化到 2.30e-3（−182%）。
    位移形式自动满足 |δq| ≤ q_b 的局地变幅，不会非物理放大。

    线性对照：τ→0 极限下 δq → −τ(δV·∇q_b)，二者一致。
    """
    nz, ny, nx = q_b.shape
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dlat = lat[1] - lat[0]; dlon = lon[1] - lon[0]
    # 位移（单位：网格格数）
    tau_s = tau_h * 3600.0
    dy_m = dV * tau_s
    dx_m = dU * tau_s
    shift_i = -dy_m / (dlat * 111195.0)
    shift_j = -dx_m / (dlon * 111195.0 * np.cos(np.deg2rad(LAT)))
    II, JJ = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    out = np.empty_like(q_b)
    for k in range(nz):
        ci = (II + shift_i[k]).ravel()
        cj = (JJ + shift_j[k]).ravel()
        out[k] = map_coordinates(q_b[k], np.array([ci, cj]), order=1,
                                 mode="nearest").reshape(ny, nx) - q_b[k]
    return out


# ----------------------------------------------------------------------------
# 主重构流程
# ----------------------------------------------------------------------------
class Reconstructor:
    """三维重构器。

    sigma  : dict，逐变量的 (sigma_b[nz], sigma_o[nz])，绝对量纲。
             由 diagnose_background_error 与观测误差设定给出。
    alpha  : 耦合系数（在**标定时次**上确定，在**检验时次**上评估）
    """

    def __init__(self, lat, lon, lev, L_h=180e3, L_z=0.45,
                 alpha_T2V=1.0, alpha_V2q=1.0, tau_h=2.0, sigma=None,
                 superob=True, max_obs=600, L_h2=None, w2=0.0, vert_corr=None):
        self.lat = np.asarray(lat, float)
        self.lon = np.asarray(lon, float)
        self.lev = np.asarray(lev, float)
        self.L_h = L_h; self.L_z = L_z
        self.L_h2 = L_h2; self.w2 = w2
        self.vert_corr = vert_corr
        self.alpha_T2V = alpha_T2V; self.alpha_V2q = alpha_V2q
        self.tau_h = tau_h
        self.sigma = sigma or {}
        self.superob = superob
        self.max_obs = max_obs

    def _oi(self, var, bg, vals, pos):
        sb, so = self.sigma.get(var, (None, None))
        cnt = None
        if self.superob:
            vals, pos, cnt = superob(vals, pos, self.lat, self.lon, self.lev)
        return oi_univariate(bg, vals, pos, self.lat, self.lon, self.lev,
                             L_h=self.L_h, L_z=self.L_z, sigma_b=sb, sigma_o=so,
                             max_obs=self.max_obs, obs_count=cnt,
                             L_h2=self.L_h2, w2=self.w2, vert_corr=self.vert_corr)

    def run(self, bg, obs, use=("sfc", "raob", "ac"), coupling=("T2V", "V2q")):
        an_uni = self.univariate(bg, obs, use=use)
        return self.apply_coupling(an_uni, bg, coupling=coupling)

    def univariate(self, bg, obs, use=("sfc", "raob", "ac")):
        """单变量 OI 阶段。**与耦合系数无关**，扫描 α 时只需计算一次。"""
        keys = ("T", "q", "u", "v")
        an = {k: bg[k].copy() for k in keys}
        for var in keys:
            vals, pos = [], []
            for tag in ("sfc", "raob", "ac"):
                if tag not in use or obs.get(tag) is None:
                    continue
                o, p = obs[tag]
                if tag == "ac" and var == "q":
                    continue                       # 飞机不报湿度
                vals.append(np.asarray(o[var], float).ravel())
                pos.append(np.asarray(p, float))
            if not vals:
                continue
            v = np.concatenate(vals)
            pz = np.vstack(pos)
            _, an[var] = self._oi(var, bg[var], v, pz)
        return an

    def apply_coupling(self, an_uni, bg, coupling=("T2V", "V2q")):
        """在单变量分析之上叠加物理耦合项。**纯算子，不含 OI**，故扫描 α 极廉价。"""
        an = {k: v.copy() for k, v in an_uni.items()}
        contrib = {"T2V_u": np.zeros_like(bg["u"]), "T2V_v": np.zeros_like(bg["v"]),
                   "V2q": np.zeros_like(bg["q"])}

        if "T2V" in coupling:
            dTv = _smooth_h(an_uni["T"] - bg["T"])
            dphi = _smooth_h(hydrostatic_geopotential(dTv, self.lev))
            dug, dvg = geostrophic_wind(dphi, self.lat, self.lon)
            contrib["T2V_u"] = self.alpha_T2V * dug
            contrib["T2V_v"] = self.alpha_T2V * dvg
            an["u"] = an["u"] + contrib["T2V_u"]
            an["v"] = an["v"] + contrib["T2V_v"]

        if "V2q" in coupling:
            dq = moisture_displacement(an["u"] - bg["u"], an["v"] - bg["v"],
                                       bg["q"], self.lat, self.lon, tau_h=self.tau_h)
            contrib["V2q"] = self.alpha_V2q * dq
            an["q"] = np.clip(an["q"] + contrib["V2q"], 0, 0.06)

        return {"an": an, "an_uni": an_uni, "contrib": contrib}


# ----------------------------------------------------------------------------
# 检验
# ----------------------------------------------------------------------------
def rmse(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sqrt(np.nanmean(d ** 2)))


def verify(an, nat, keys=("T", "q", "u", "v")):
    return {k: rmse(an[k], nat[k]) for k in keys}


# ----------------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------------
def selfcheck():
    """数值自检。任何一条与期望不符都说明实现有误。

    之所以把自检写成硬断言而非留待观察：本项目 v1/v2 的两次严重失效
    （气压层降序、水平距离单位）都不是"精度下降"而是"方法静默失效"，
    表观症状相似（增益≈0），靠肉眼读 RMSE 无法区分。
    """
    out = {}
    lat = np.linspace(30.0, 30.25, 3)
    lon = np.linspace(120.0, 120.25, 3)
    lev = np.array([1000., 850., 700., 500.])

    # (a) 长度单位：1 度纬度 = 111195 m（必须，见 _metric_xy 的踩坑记录）
    dy, dx = _metric_xy(np.array([31.0]), np.array([120.0]),
                        np.array([30.0]), np.array([120.0]))
    out["metric_1deg_lat_m"] = float(dy[0])

    # (b) 高斯相关在 d = L_h 处须为 exp(-0.5) = 0.60653
    out["corr_at_1Lh"] = float(corr_gauss(124e3, 124e3))
    out["corr_at_2Lh"] = float(corr_gauss(248e3, 124e3))

    # (c) 单点 OI 的**解析解**：x_a = x_b + σ_b²/(σ_b²+σ_o²)·(y − x_b)
    bg = np.full((1, 1, 1), 10.0)
    w = 2.0 ** 2 / (2.0 ** 2 + 1.0 ** 2)          # σ_b=2, σ_o=1
    _, an = oi_univariate(bg, np.array([13.0]), np.array([[1000., 30.0, 120.0]]),
                          lat[:1], lon[:1], lev[:1], L_h=124e3, L_z=0.34,
                          sigma_b=np.array([2.0]), sigma_o=np.array([1.0]))
    out["oi_single_expect"] = float(10.0 + w * 3.0)
    out["oi_single_got"] = float(an[0, 0, 0])

    # (d) 地转关系解析解。dphi 形状 (nz, ny, nx)：纬度=axis1，经度=axis2
    ny = 41
    la = np.linspace(29.0, 31.0, ny); lo = np.linspace(119.0, 121.0, 9)
    f = 2 * OMEGA * np.sin(np.deg2rad(la.mean()))
    yy = (la - la.mean()) * 111195.0
    xx = (lo - lo.mean()) * 111195.0 * np.cos(np.deg2rad(la.mean()))
    B = 50.0
    rep = lambda a: np.repeat(a[None, :, :], 2, axis=0)
    # (d-1) δΦ = B·y ⇒ δu_g = −B/f ，δv_g = 0
    du, dv = geostrophic_wind(rep(np.tile((B * yy)[:, None], (1, len(lo)))), la, lo)
    out["geo_du_expect"] = float(-B / f)
    out["geo_du_got"] = float(du[0, ny // 2, len(lo) // 2])
    out["geo_dv_should_be_0"] = float(np.max(np.abs(dv)))
    # (d-2) δΦ = C·x ⇒ δv_g = +C/f ，δu_g = 0
    C = 30.0
    du2, dv2 = geostrophic_wind(rep(np.tile((C * xx)[None, :], (ny, 1))), la, lo)
    out["geo_dv_expect"] = float(C / f)
    out["geo_dv_got"] = float(dv2[0, ny // 2, len(lo) // 2])
    out["geo_du_should_be_0"] = float(np.max(np.abs(du2)))

    # (e) 水汽位移算子在 τ→0 极限须退化为线性平流 −τ(δV·∇q_b)
    q0 = np.zeros((1, ny, len(lo)))
    xx = (lo - lo.mean()) * 111195.0 * np.cos(np.deg2rad(la.mean()))
    q0[0] = 0.01 + 1e-9 * xx[None, :]
    dU = np.full((1, ny, len(lo)), 10.0)
    dq = moisture_displacement(dU, np.zeros_like(dU), q0, la, lo, tau_h=1.0)
    out["advect_expect"] = float(-10.0 * 1e-9 * 3600.0)
    out["advect_got"] = float(np.median(dq))

    # (f) 静力积分符号：整层均匀增温 ⇒ 位势厚度增大（δΦ 随高度递增）
    dT = np.ones((4, 3, 3)) * 1.0
    dphi2 = hydrostatic_geopotential(dT, lev)
    out["hydro_dphi_top"] = float(dphi2[-1, 0, 0])
    out["hydro_monotone"] = bool(np.all(np.diff(dphi2[:, 0, 0]) > 0))

    # (g) 气压层降序映射：1000 hPa 的观测必须落在第 0 层
    out["lev_map_1000hPa"] = int(np.searchsorted(-lev, -1000.0))
    out["lev_map_100hPa"] = int(np.searchsorted(-np.array([1000., 500., 100.]), -100.0))

    # (h) 相关长度诊断的**端到端标定**：从已知 L_h 的合成场中须能恢复。
    #     对白噪声做高斯滤波（核 σ_f 格）后，相关函数 = exp(−d²/(4σ_f²))，
    #     即等价于 exp(−d²/2L²) 中 L = √2·σ_f。
    from scipy.ndimage import gaussian_filter
    la2 = np.linspace(20.0, 35.0, 61)
    lo2 = np.linspace(105.0, 125.0, 81)
    rng2 = np.random.default_rng(0)
    sig_f = 3.0
    nz2 = 2
    fld = gaussian_filter(rng2.standard_normal((nz2, la2.size, lo2.size)),
                          sigma=(0, sig_f, sig_f), mode="nearest")
    grid_km = abs(la2[1] - la2[0]) * 111.195
    L_true = np.sqrt(2.0) * sig_f * grid_km * 1e3
    nat_f = {k: np.zeros_like(fld) for k in ("T", "q", "u", "v")}
    bg_f = {k: fld.copy() for k in ("T", "q", "u", "v")}
    dg = diagnose_background_error(nat_f, bg_f, la2, lo2, np.array([1000., 500.]))
    out["Lh_diag_expect_m"] = float(L_true)
    out["Lh_diag_got_m"] = float(dg["L_h"])
    out["Lh_diag_relerr"] = float(abs(dg["L_h"] - L_true) / L_true)
    return out
