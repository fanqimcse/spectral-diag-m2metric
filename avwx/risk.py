# -*- coding: utf-8 -*-
"""
avwx.risk —— 飞行路径气象风险评估
===================================

沿四维航迹（经纬度 + 气压层 + 时间）对重构后的三维风–温–湿场做积分，
输出可业务化解释的分项风险与综合风险。

本模块与 thermo.py 共享同一科学主线：**湿度是被系统性忽略的第一阶因子**。
在巡航层，这一点体现为 **湿理查森数 vs 干理查森数**：

    Ri = N² / S²,   N² = Brunt–Väisälä 频率平方,   S² = 垂直风切变平方

现行晴空湍流（CAT）诊断（含 Prosser et al. 2023 *Nature* 所用的 21 个指标族）
普遍采用**干**的 N_d² = (g/T)(dT/dz + g/c_p)，即完全不含湿度项。

本模块的湿口径取 N_v²（**虚温形式**，见 TH.brunt_vaisala_virtual_sq）：

    N_v² = (g/T_v) [ (1+0.608q)·dT/dz + 0.608·T·dq/dz + (1+0.608q)·g/c_p ]

之所以不用更常见的饱和湿绝热式（Wallace & Hobbs 2006 式 3.75）：
CAT 按定义发生在**无云**区，气块受扰上升不凝结、比湿守恒，
浮力由密度（虚温）决定；饱和式把实际比湿 q 代入饱和比湿 q_s 的位置，
在 q ≪ q_s 的晴空区是物理误用。

物理要害：对流层中 dq/dz < 0（水汽随高度递减），
故 0.608·T·dq/dz < 0 ⟹ **N_v² < N_d²**——湿度**降低**静力稳定度，
从而降低 Ri、提高 K–H 不稳定的可能。**用 N_d² 会高估稳定度、低估 CAT 风险**。

本模块给出可直接量化的判据：
    "湿度诱发的 CAT 涌现率" = 满足 Ri_dry > 0.25 但 Ri_moist < 0.25 的格点/航段占比
即"干算法判定安全、湿算法判定危险"的那部分。
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avwx import thermo as TH

G = TH.G
R_D = TH.R_D


# ----------------------------------------------------------------------------
# 航线（大圆 + 巡航高度剖面）
# ----------------------------------------------------------------------------
ROUTES = [
    # 名称,         起点,              终点,              巡航层(hPa), 机型
    ("PEK-SIN",   (40.07, 116.58),   (1.36, 103.99),   200, "B77W"),
    ("HKG-LHR",   (22.31, 113.91),   (51.47, -0.45),   225, "B77W"),
    ("PVG-LAX",   (31.14, 121.81),   (33.94, -118.41), 200, "B77W"),
    ("DXB-JFK",   (25.25, 55.36),    (40.64, -73.78),  225, "B77W"),
    ("DEL-BOM",   (28.56, 77.10),    (19.09, 72.87),   250, "A320"),
    ("SYD-SIN",   (-33.94, 151.18),  (1.36, 103.99),   200, "A359"),
    ("JNB-CPT",   (-26.13, 28.24),   (-33.97, 18.60),  250, "A320"),
    ("MIA-BOG",   (25.79, -80.29),   (4.70, -74.15),   225, "A320"),
    ("NRT-SIN",   (35.77, 140.39),   (1.36, 103.99),   200, "B788"),
    ("FRA-GRU",   (50.03, 8.56),     (-23.43, -46.47), 225, "B77W"),
]


def great_circle(p0, p1, n=120):
    """大圆插值，返回 (lat[], lon[])。"""
    lat0, lon0 = np.deg2rad(p0); lat1, lon1 = np.deg2rad(p1)
    d = 2 * np.arcsin(np.sqrt(np.sin((lat1 - lat0) / 2) ** 2 +
                              np.cos(lat0) * np.cos(lat1) * np.sin((lon1 - lon0) / 2) ** 2))
    f = np.linspace(0, 1, n)
    if d < 1e-9:
        return np.full(n, np.rad2deg(lat0)), np.full(n, np.rad2deg(lon0))
    A = np.sin((1 - f) * d) / np.sin(d)
    B = np.sin(f * d) / np.sin(d)
    x = A * np.cos(lat0) * np.cos(lon0) + B * np.cos(lat1) * np.cos(lon1)
    y = A * np.cos(lat0) * np.sin(lon0) + B * np.cos(lat1) * np.sin(lon1)
    z = A * np.sin(lat0) + B * np.sin(lat1)
    return np.rad2deg(np.arctan2(z, np.hypot(x, y))), np.rad2deg(np.arctan2(y, x))


def flight_profile(n=120, cruise_hpa=200, sfc_hpa=1000):
    """简化的高度剖面：爬升 12 % → 巡航 76 % → 下降 12 %。"""
    f = np.linspace(0, 1, n)
    lev = np.empty(n)
    for i, t in enumerate(f):
        if t < 0.12:
            lev[i] = sfc_hpa + (cruise_hpa - sfc_hpa) * (t / 0.12)
        elif t > 0.88:
            lev[i] = cruise_hpa + (sfc_hpa - cruise_hpa) * ((t - 0.88) / 0.12)
        else:
            lev[i] = cruise_hpa
    return lev


def interpolate_field(field, lat, lon, lev, qlat, qlon, qlev):
    """三线性插值到航迹点。field: (nz, ny, nx)，lev 降序（气压递减）。"""
    out = np.full(len(qlat), np.nan)
    ilat = np.searchsorted(lat, qlat)
    ilon = np.searchsorted(lon, qlon % 360.0)
    for i in range(len(qlat)):
        a = np.clip(ilat[i], 1, len(lat) - 1)
        b = np.clip(ilon[i], 1, len(lon) - 1)
        # 垂直：lev 降序
        k = np.searchsorted(-lev, -qlev[i])
        k = np.clip(k, 1, len(lev) - 1)
        p = (lev[k - 1] - qlev[i]) / max(lev[k - 1] - lev[k], 1e-9)
        p = np.clip(p, 0, 1)
        ty = (qlat[i] - lat[a - 1]) / max(lat[a] - lat[a - 1], 1e-9)
        tx = (qlon[i] % 360.0 - lon[b - 1]) / max(lon[b] - lon[b - 1], 1e-9)
        v = 0.0
        for kk, pk in ((k - 1, 1 - p), (k, p)):
            v += pk * ((1 - ty) * (1 - tx) * field[kk, a - 1, b - 1] +
                       (1 - ty) * tx * field[kk, a - 1, b] +
                       ty * (1 - tx) * field[kk, a, b - 1] +
                       ty * tx * field[kk, a, b])
        out[i] = v
    return out


# ----------------------------------------------------------------------------
# 三维诊断场
# ----------------------------------------------------------------------------
def height_coordinate(lev, Tv=None, gh=None):
    """把降序气压层换算为**累积**几何高度 Z [m]，形状 (nz, ny, nx)，Z[0]=0。

    为什么不能像早期版本那样逐点算 Z = −(R_d T_v/g)·ln p 再求梯度
    --------------------------------------------------------------
    那个写法本身给出的 Z 是"相对 p=1（同单位）的高度"，常数偏移无害；
    **致命的是对它求 np.gradient**——梯度作用在乘积 T_v·ln p 上：

        ∂/∂z [T_v ln p] = ln p · ∂T_v/∂z  +  T_v · ∂ ln p/∂z
                          └── 伪项，量级常为真项的 3–6 倍 ──┘└─ 真项 ─┘

    因为 |ln p| ≈ 4.6–6.9 而层间 Δ ln p 仅 ≈ 0.08，伪项被放大近两个量级。
    本项目实测：1000→925 hPa 层厚被算成 2126 m，而 GFS 自带位势高度为
    688 m（偏大 3.1 倍）⇒ S² 被压低约 10 倍、Ri 虚高到 ~200。

    正确做法：厚度是**层量**，必须逐层积分（梯形），再累加成高度坐标：

        Δz_k = (R_d/g) · ½(T_v,k + T_v,k+1) · (ln p_k − ln p_{k+1})

    得到 Z 后，垂直导数按非均匀层厚做中心差分（内部）与单侧差分（边界）。
    注意 np.gradient(X, Z, axis=0) 的"坐标数组"形式只接受**一维** Z，
    而这里的 Z 逐列不同，故在 stability_fields 中手写差分。

    gh（模式自带位势高度）可用时优先使用：它是模式自身静力积分的结果，
    比用 T_v 重算更精确（含模式对虚温、地形、参考大气的处理）。
    """
    lev = np.asarray(lev, float)
    if gh is not None and np.all(np.isfinite(gh)):
        Z = np.asarray(gh, float).copy()
        Z = Z - Z[0][None, ...]                 # 以最低层为零点
        return Z
    if Tv is None:
        raise ValueError("height_coordinate 需要 gh 或 Tv 之一")
    lp = np.log(lev)
    dlp = lp[:-1] - lp[1:]                      # >0，共 nz-1 个
    Tvm = 0.5 * (Tv[:-1] + Tv[1:])              # 层平均虚温
    gap = (R_D / G) * Tvm * dlp[:, None, None]  # (nz-1, ny, nx)
    Z = np.zeros_like(Tv)
    Z[1:] = np.cumsum(gap, axis=0)
    return Z


def stability_fields(f, lat, lon, lev):
    """由三维场计算 S²、N_d²、N_v²、Ri_dry、Ri_moist、Ellrod TI、RH。

    f: dict(T, q, u, v[, z]) —— 各为 (nz, ny, nx)，lev 为降序气压 [hPa]。
       z 为模式位势高度 [m]（可选，提供时用作垂直坐标，见 height_coordinate）。

    湿度口径：N_v²（虚温形式，适用于 CAT 所在的未饱和晴空区，
    见 TH.brunt_vaisala_virtual_sq）；同时保留 N_d² 作为"现行业务算法"
    的对照，二者之差即湿度的净效应。

    由于干、湿两套 Ri **共用同一个 S²**，恒有
        Ri_moist = Ri_dry · (1 − δ),   δ ≡ (N_d² − N_v²)/N_d²
    该式与 S² 的绝对精度无关，是本模块所有"分辨率稳健"结论的基础。
    """
    T = f["T"]; q = f["q"]; u = f["u"]; v = f["v"]
    p = np.asarray(lev, float)[:, None, None] * 100.0
    Z = height_coordinate(lev, TH.virtual_temperature(T, q), f.get("z"))

    # 非均匀层厚的垂直导数：内部中心差分、边界单侧差分。
    # 注意 np.gradient 的"坐标数组"只支持一维，而这里的 Z 随每一列不同
    # （层厚依赖于该列的虚温廓线），故必须手写。
    zspan = np.maximum(Z[2:] - Z[:-2], 1.0)      # (nz-2, ny, nx) 中心差分跨度
    zgap = np.maximum(np.diff(Z, axis=0), 1.0)   # (nz-1, ny, nx) 逐层厚度

    def d_dz(X):
        X = np.asarray(X, float)
        out = np.empty_like(X)
        if X.shape[0] > 2:
            out[1:-1] = (X[2:] - X[:-2]) / zspan
        out[0] = (X[1] - X[0]) / zgap[0]
        out[-1] = (X[-1] - X[-2]) / zgap[-1]
        return out

    dTdz = d_dz(T)
    dqdz = d_dz(q)
    dudz = d_dz(u)
    dvdz = d_dz(v)

    S2 = dudz ** 2 + dvdz ** 2
    Nd2 = TH.brunt_vaisala_sq(1e5, T, np.zeros_like(q), dTdz, None, moist=False)
    Nv2 = TH.brunt_vaisala_virtual_sq(T, q, dTdz, dqdz)

    Ri_d = TH.richardson_number(S2, Nd2)
    Ri_m = TH.richardson_number(S2, Nv2)
    rh = TH.relative_humidity(T - 273.15, q, p)

    # 水平形变（用于 Ellrod TI）
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dy = np.gradient(LAT, axis=0) * 111195.0
    dx = np.gradient(LON, axis=1) * 111195.0 * np.cos(np.deg2rad(LAT))
    dux = np.gradient(u, axis=2) / np.where(dx == 0, 1e9, dx)
    duy = np.gradient(u, axis=1) / np.where(dy == 0, 1e9, dy)
    dvx = np.gradient(v, axis=2) / np.where(dx == 0, 1e9, dx)
    dvy = np.gradient(v, axis=1) / np.where(dy == 0, 1e9, dy)
    DEF2 = TH.deformation_sq(dux, duy, dvx, dvy)
    TI = TH.turbulence_index_ellrod(S2, DEF2)

    # 口径 S：Durran & Klemp (1982) **饱和**湿绝热式。
    #   · 这是业务模式湍流诊断里"湿 Ri"的事实标准实现：ASCOS (JGR 2017)
    #     明确按"晴空用干式、云内用 Durran-Klemp 湿式"三分；ICON-LES 等
    #     实现则按饱和/未饱和分段采用同一个 N_m²。
    #   · 纳入本模块**不是推荐它**，而是把"把饱和式用在晴空区"的后果定量化：
    #     其湿度修正经 Γ_m 与 r_s 只依赖 (T, p)，与实际湿度无关；
    #     而物理正确的虚温式修正 ∝ dq/dz，随实际水汽递减线性增长。
    #     （p 已在上面定义为 lev[Pa]，形状 (nz,1,1)，可自动广播）
    Ndk2 = TH.brunt_vaisala_durran_klemp_sq(T, q, p, dTdz, dqdz)

    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(Nd2 > 1e-9, (Nd2 - Nv2) / Nd2, np.nan)   # 湿度削弱率 δ（虚温式）
        # δ_sat：把饱和式当"湿润正式谕"地到处使用时的表观削弱率。
        # 它与 δ 之差即"口径选择不当"造成的静力稳定度误差。
        delta_sat = np.where(Nd2 > 1e-9, (Nd2 - Ndk2) / Nd2, np.nan)

    # Ri_dk：Durran–Klemp 饱和式对应的理查逊数。
    # 之前只输出 Ndk2，产品端做"晴空用虚温式 / 云内用饱和式"的口径切换时
    # 拿不到它，只能拿 Ri_moist 顶替 —— 那会让切换策略退化成恒等变换，
    # 结论会被静默地抹掉。这里补齐，与 Ri_dry / Ri_moist 同口径输出。
    Ri_dk = TH.richardson_number(S2, Ndk2)

    return dict(S2=S2, Nd2=Nd2, Nv2=Nv2, Ndk2=Ndk2,
                Ri_dry=Ri_d, Ri_moist=Ri_m, Ri_dk=Ri_dk,
                delta=delta, delta_sat=delta_sat,
                TI=TI, rh=rh, dTdz=dTdz, dqdz=dqdz,
                Z=Z, dz=zgap)


def cat_flags(diag, ri_crit=0.25, rh_clear=80.0):
    """CAT 判据：Ri < 0.25（K–H 不稳定必要条件）。

    返回干/湿两套判据的布尔场，用于计算"湿度诱发的 CAT 涌现"。

    为什么要区分 clear 与 all
    -------------------------
    CAT 按定义指**晴空**湍流；云内湍流机制不同（含凝结潜热释放、
    云微物理与辐射），本项目的虚温 N_v² 口径也不适用于饱和区。
    因此主口径限定在 RH < rh_clear 的**晴空**格点，
    同时给出不区分云况的全域统计作为对照。
    """
    Ri_d = diag["Ri_dry"]; Ri_m = diag["Ri_moist"]
    cat_d = (Ri_d < ri_crit) & (Ri_d > 0)
    cat_m = (Ri_m < ri_crit) & (Ri_m > 0)
    clear = diag["rh"] < rh_clear if "rh" in diag else np.ones_like(Ri_d, bool)
    return dict(cat_dry=cat_d, cat_moist=cat_m, clear=clear,
                emerge=cat_m & (~cat_d),                 # 干判安全但湿判危险
                emerge_clear=cat_m & (~cat_d) & clear)   # 主口径：晴空区内的涌现


N2_FLOOR = 1e-5
"""稳定度门槛 [s-2]（默认 1e-5，即 N > 3.2e-3 s-1，浮力周期 < 33 min）。

为什么必须有这个门槛
--------------------
δ = (N_d²−N_v²)/N_d² 的分母是 N_d²，当层结趋近中性（N_d²→0）时 δ 发散：
本项目实测弱层结处 δ 可达 10² 量级，而绝对值 |N_v²−N_d²| 仅 ~7e-6 s-2。
更糟的是，Ri = N²/S² 在 N²→0 处对 S² 极度敏感，分析场的垂直分辨率缺陷
会被放大，使"涌现率"完全由近中性层主导（实测涌现格点的 N_d² 中位
3.4e-6，比全域中位 1.3e-4 小 38 倍）——那不是 CAT，是数值与判据的失效区。

1e-5 的取值理由：对应 N = 3.16e-3 s-1，比自由大气典型值（1e-2 s-1）
低 3 倍，即"确实存在层结但偏弱"的判据；再低（<1e-6）就进入
浮力周期 >100 min 的近中性区，K–H 判据本身不再适用。
"""


def valid_mask(diag, rh_clear=80.0, n2_floor=N2_FLOOR):
    """可用于干/湿对照的格点：层结确实存在、湿度削弱率有限、有切变、晴空。"""
    ok = ((diag["Nd2"] > n2_floor) & np.isfinite(diag["delta"])
          & (diag["S2"] > 0))
    if "rh" in diag:
        ok = ok & (diag["rh"] < rh_clear)
    return ok


def missed_virtual_instability(diag, rh_clear=80.0):
    """**主指标 A**：被干口径漏判的"虚温不稳定层"。

    定义：晴空 + 干口径判定静力稳定（N_d²>0）+ 虚温口径判定不稳定（N_v²<0）。

    为什么这是本项目最稳健的指标
    ----------------------------
    由 thermo 模块可直接解析得到（本项目已数值核验到 1e-14）：

        N_v² − N_d² = 0.608 · g · (dq/dz) / (1 + 0.608 q)

    该式**完全不含 S²**。于是：
      * 它不依赖垂直风切变，因而**不受分析场垂直分辨率缺陷的影响**——
        这是本项目所有 CAT 相关量中唯一彻底绕开该问题的一个；
      * 它判定的是**符号翻转**（稳定 ↔ 不稳定），不是阈值穿越，
        对量值误差的鲁棒性远高于"Ri 是否越过 0.25"这类判据；
      * 干、湿两口径受控对照，差异 100% 归因于湿度。

    物理含义：只要水汽随高度以常见速率递减（对流层常态），
    弱层结层在虚温口径下就是**浮力不稳定**的，而干口径会判它"稳定"。
    这是定性错误（符号错），不是定量偏差。
    """
    clear = diag["rh"] < rh_clear if "rh" in diag else np.ones_like(diag["Nd2"], bool)
    stable_dry = diag["Nd2"] > 0
    return clear & stable_dry & (diag["Nv2"] < 0)


def ri_threshold_equivalent(delta, ri_crit=0.25):
    """**阈值等价误差** —— 干口径判据等价于把湿口径的阈值放宽了多少。

    由恒等式 Ri_moist = Ri_dry(1−δ)（两者共用 S²，精确成立）可直接得到：

        干判据  Ri_dry   < Ri_c
        ⟺ 湿判据 Ri_moist < Ri_c·(1−δ)

    即：业务上"用干 N² + 阈值 0.25"，在物理上等价于
    "用正确的湿 N² + 阈值 0.25/(1−δ)"，后者**恒大于** 0.25。

    含义：现行干算法不是"阈值取 0.25"，而是**取了一个偏松的阈值**——
    它系统性漏掉所有满足 0.25 < Ri_moist < 0.25/(1−δ) 的湍流。
    返回该"实际生效的阈值" Ri* = Ri_c/(1−δ)，以及相对放大倍数。

    三种情形（不做截断，保持解析行为的诚实）
    ----------------------------------------
      δ < 1 : Ri* = Ri_c/(1−δ) > Ri_c，干判据偏松，这是通常情形；
      δ ≥ 1 : 等价于 N_v² ≤ 0，该层在湿口径下**无条件不稳定**，
              Ri* 取 +inf（任何阈值都会判它危险），不存在"等价阈值"；
      δ < 0 : 水汽随高度增加，湿度反而**增强**层结，Ri* < Ri_c，
              干判据在此偏保守——同样如实返回，不做对称化处理。

    注意 δ→1⁻ 时 Ri*→∞（近中性层的固有发散），这正是本项目必须施加
    N2_FLOOR 稳定度门槛的原因；所有对外报告的分位数均在门槛后计算。
    """
    d = np.asarray(delta, float)
    om = 1.0 - d
    with np.errstate(invalid="ignore", divide="ignore"):
        amp = np.where(om > 0, 1.0 / np.where(om > 0, om, 1.0), np.inf)
    return ri_crit * amp, amp


def emergence_vs_alpha(diag, ri_crit=0.25, rh_clear=80.0, alphas=None,
                       n2_floor=N2_FLOOR):
    """涌现率对**次网格切变放大倍数 α** 的敏感性曲线。

    为什么需要 α
    ------------
    K–H 判据 Ri<0.25 是针对**湍流尺度**（∂z ~ 10–100 m）的切变定义的。
    而业务分析场（本项目 0.25° / 25 hPa 间隔）的层厚达 200–900 m，
    只能分辨真实切变的一小部分 ⇒ 网格 Ri 被系统性高估 1–2 个量级，
    全域几乎找不到 Ri<0.25 的格点（本项目实测 0.0000%）。

    把未分辨的部分参数化为 S²_true = α · S²_analysis（α>1），则

        涌现 ⟺ Ri_dry/α > Ri_c  且  Ri_dry(1−δ)/α < Ri_c
             ⟺ Ri_dry ∈ ( α·Ri_c ,  α·Ri_c/(1−δ) )

    于是对任意 α 都能算出涌现率，且**该式只依赖 δ 与 Ri_dry 的联合分布**，
    不引入任何新的物理假设——α 只是把分析场与湍流尺度之间的差距显式化。

    这是本项目把"网格上涌现率≈0"与"物理上湿度确实削弱稳定度"这两个
    事实联系起来的诚实桥梁：α 不是拟合出来的，而是作为**敏感性参数**扫描。
    """
    if alphas is None:
        alphas = [1, 2, 5, 10, 20, 50, 100, 200]
    ok = valid_mask(diag, rh_clear, n2_floor)
    rd = np.where(ok, diag["Ri_dry"], np.nan)
    dl = np.where(ok, diag["delta"], np.nan)
    out = []
    n = float(max(ok.sum(), 1))     # 分母必须是**有效**格点数，不是全域数
    for a in alphas:
        lo = a * ri_crit
        hi = lo / np.maximum(1.0 - dl, 1e-6)
        with np.errstate(invalid="ignore"):
            em = (rd > lo) & (rd < hi)
            dry = rd < lo
        out.append((float(a), float(np.count_nonzero(em) / n),
                    float(np.count_nonzero(dry) / n)))
    return out     # [(alpha, 涌现率, 干判危险率)]  均以有效格点为分母


def rank_turnover(diag, fracs=(0.01, 0.05, 0.10), rh_clear=80.0,
                  n2_floor=N2_FLOOR):
    """**风险排序翻转率**：把 CAT 通告从干口径换成湿口径，圈定区域变了多少。

    为什么这个指标比"涌现率"更稳健
    --------------------------------
    业务 CAT 通告（SIGMET / 区域预警）实质上不是绝对阈值判定，而是
    **按风险指数排序后圈定最危险的前 p% 空域**——因为阈值的绝对值
    必须随模式分辨率、区域、季节反复标定，而"前 5%"是自适应的。

    排序翻转率问的正是：换成湿口径后，被圈进/踢出的空域有多少。
    它只取决于 δ 的**空间变率**（S² 在两套口径中完全抵消），
    因此对分析场垂直分辨率的缺陷**不敏感**——这是本项目最稳健的指标。

    返回 {p: (被干口径圈定但因加湿而落选的比例, 干湿两集合的 Jaccard 相似度)}
    """
    ok = valid_mask(diag, rh_clear, n2_floor)
    rd = np.where(ok, diag["Ri_dry"], np.inf)
    rm = np.where(ok, diag["Ri_moist"], np.inf)
    n = int(ok.sum())
    if n < 100:
        return {p: (np.nan, np.nan) for p in fracs}
    out = {}
    for p in fracs:
        k = max(int(round(p * n)), 1)
        # 取最不稳定的 k 个格点（Ri 最小）
        sd = set(np.argpartition(rd.ravel(), k)[:k].tolist())
        sm = set(np.argpartition(rm.ravel(), k)[:k].tolist())
        drop = len(sd - sm) / max(len(sd), 1)          # 干口径圈定、湿口径剔除
        jac = len(sd & sm) / max(len(sd | sm), 1)
        out[p] = (float(drop), float(jac))
    return out


def icing_flag(f, rh):
    """积冰判据：-20 °C < T < 0 °C 且 RH ≥ 80 %（FAA/ICAO 常用包线）。"""
    Tc = f["T"] - 273.15
    return (Tc > -20.0) & (Tc < 0.0) & (rh >= 80.0)


def convective_proxy(f, lat, lon, lev):
    """对流潜势的轻量代理：低层比湿 × 条件不稳定度。

    严格 CAPE 需要抬升气块计算；本代理用 850–500 hPa 层结递减率与
    1000 hPa 比湿的组合，用于航迹相对比较（非绝对预报量）。
    """
    q = f["q"]; T = f["T"]
    i850 = int(np.argmin(np.abs(lev - 850)))
    i500 = int(np.argmin(np.abs(lev - 500)))
    i1000 = int(np.argmin(np.abs(lev - 1000)))
    lr = (T[i850] - T[i500]) / 3.5                    # °C/km 近似
    return np.clip((lr - 4.0) / 5.0, 0, 2) * np.clip(q[i1000] / 0.016, 0, 2)


# ----------------------------------------------------------------------------
# 航迹综合风险
# ----------------------------------------------------------------------------
def assess_route(f, diag, lat, lon, lev, p0, p1, cruise_hpa=200, n=120,
                 surface=None, ri_crit=0.25, rh_clear=80.0):
    """对一条航线做全程风险评估。

    注：Ri 是在**网格点**上算好后三线性插值到航迹点的，与业务做法一致
    （先诊断后插值）。替代方案是插值 N² 与 S² 再求比值——在层结
    强不连续处更稳健，但会改变"干/湿对照"的口径一致性，故不采用。
    """
    qlat, qlon = great_circle(p0, p1, n)
    qlon = qlon % 360.0
    qlev = flight_profile(n, cruise_hpa)

    T = interpolate_field(f["T"], lat, lon, lev, qlat, qlon, qlev)
    q = interpolate_field(f["q"], lat, lon, lev, qlat, qlon, qlev)
    u = interpolate_field(f["u"], lat, lon, lev, qlat, qlon, qlev)
    v = interpolate_field(f["v"], lat, lon, lev, qlat, qlon, qlev)
    rid = interpolate_field(diag["Ri_dry"], lat, lon, lev, qlat, qlon, qlev)
    rim = interpolate_field(diag["Ri_moist"], lat, lon, lev, qlat, qlon, qlev)
    rdk = interpolate_field(diag["Ri_dk"], lat, lon, lev, qlat, qlon, qlev) \
        if "Ri_dk" in diag else np.full(n, np.nan)
    ti = interpolate_field(diag["TI"], lat, lon, lev, qlat, qlon, qlev)
    rh = interpolate_field(diag["rh"], lat, lon, lev, qlat, qlon, qlev) \
        if "rh" in diag else np.full(n, np.nan)

    # 巡航段掩码
    cruise = qlev <= cruise_hpa * 1.15
    cat_d = (rid < ri_crit) & (rid > 0) & cruise
    cat_m = (rim < ri_crit) & (rim > 0) & cruise
    clear = (rh < rh_clear) & cruise
    emerge = cat_m & (~cat_d)
    emerge_clear = emerge & (rh < rh_clear)

    # 湿度对静力稳定度的相对削弱 δ（核心物理量，取巡航段中位）
    nd2 = interpolate_field(diag["Nd2"], lat, lon, lev, qlat, qlon, qlev)
    nv2 = interpolate_field(diag["Nv2"], lat, lon, lev, qlat, qlon, qlev)
    with np.errstate(invalid="ignore", divide="ignore"):
        weaken = np.where((nd2 > 1e-9) & cruise, (nd2 - nv2) / nd2, np.nan)
    delta = interpolate_field(diag["delta"], lat, lon, lev, qlat, qlon, qlev) \
        if "delta" in diag else weaken
    with np.errstate(invalid="ignore", divide="ignore"):
        ri_star, amp = ri_threshold_equivalent(np.where(cruise, delta, np.nan), ri_crit)

    out = dict(
        lat=qlat, lon=qlon, lev=qlev, T=T, q=q, u=u, v=v,
        Ri_dry=rid, Ri_moist=rim, Ri_dk=rdk, TI=ti, rh=rh,
        cat_dry_frac=float(np.mean(cat_d)) if cruise.any() else np.nan,
        cat_moist_frac=float(np.mean(cat_m)) if cruise.any() else np.nan,
        emerge_frac=float(np.mean(emerge)) if cruise.any() else np.nan,
        emerge_clear_frac=float(np.mean(emerge_clear)) if cruise.any() else np.nan,
        clear_frac=float(np.mean(clear)) if cruise.any() else np.nan,
        n2_weaken_median=float(np.nanmedian(weaken)) if np.isfinite(weaken).any() else np.nan,
        delta_median=float(np.nanmedian(np.where(cruise, delta, np.nan)))
        if np.isfinite(delta).any() else np.nan,
        ri_star_median=float(np.nanmedian(ri_star)) if np.isfinite(ri_star).any() else np.nan,
        amp_median=float(np.nanmedian(amp)) if np.isfinite(amp).any() else np.nan,
        ti_mean=float(np.nanmean(ti)) if cruise.any() else np.nan,
    )
    # 端点：湿密度高度惩罚
    if surface is not None:
        out["depart"] = endpoint_risk(surface["dep"])
        out["arrive"] = endpoint_risk(surface["arr"])
    return out


def selfcheck(verbose=True):
    """risk 模块自检。返回 (通过数, 总数)。

    每项都针对本项目**实际踩过的坑**设计，而非泛泛的健全性检查：
      * 层厚/垂直导数：针对"逐点算 Z=−(R_d T_v/g)ln p 再求梯度"的伪项 bug；
      * 解析恒等式 (*)：针对 δ 的解析形式是否实现正确；
      * Ri_moist = Ri_dry(1−δ)：保证"干湿共用 S²"这条推理链成立；
      * 阈值等价：保证业务结论的算术无误。
    """
    ok = 0
    tot = 0

    def chk(name, cond, extra=""):
        nonlocal ok, tot
        tot += 1
        if cond:
            ok += 1
        if verbose:
            print("  [%s] %-52s %s" % ("OK " if cond else "FAIL", name, extra))

    # ---- 1. 层厚：等虚温大气的解析解 ----
    lev = np.array([1000.0, 900.0, 800.0, 700.0])
    Tv = np.full((4, 3, 3), 280.0)
    Z = height_coordinate(lev, Tv, None)
    dz_exact = (R_D / G) * 280.0 * np.log(1000.0 / 900.0)
    e1 = abs(Z[1, 0, 0] - dz_exact) / dz_exact
    chk("层厚 vs 等虚温解析解", e1 < 1e-10, "rel=%.2e" % e1)

    # ---- 2. 垂直导数：对 Z 线性的场应给出常数导数（伪项 bug 的照妖镜）----
    # 若 Z 由"逐点乘积"算出并直接求梯度，dT/dz 会混入 ln p·dT_v/dz 项，
    # 使本应为常数的导数随高度漂移。取 T = α·Z + β 检验。
    alpha, beta = -6.5e-3, 300.0
    Tlin = alpha * Z + beta
    zspan = np.maximum(Z[2:] - Z[:-2], 1.0)
    zgap = np.maximum(np.diff(Z, axis=0), 1.0)
    dT = np.empty_like(Tlin)
    dT[1:-1] = (Tlin[2:] - Tlin[:-2]) / zspan
    dT[0] = (Tlin[1] - Tlin[0]) / zgap[0]
    dT[-1] = (Tlin[-1] - Tlin[-2]) / zgap[-1]
    e2 = np.max(np.abs(dT - alpha)) / abs(alpha)
    chk("垂直导数对 Z 线性场为常数", e2 < 1e-9, "max_rel=%.2e" % e2)

    # ---- 3. 解析恒等式 (*)：N_v²−N_d² = 0.608 g (dq/dz)/(1+0.608q) ----
    rng = np.random.default_rng(20261020)
    n = 4000
    Tk = rng.uniform(200, 310, n)
    qq = rng.uniform(0, 0.025, n)
    dtdz = rng.uniform(-0.012, 0.002, n)
    dqdz = rng.uniform(-3e-5, -1e-9, n)          # 对流层常态 dq/dz<0
    nd2 = TH.brunt_vaisala_sq(1e5, Tk, np.zeros(n), dtdz, None, moist=False)
    nv2 = TH.brunt_vaisala_virtual_sq(Tk, qq, dtdz, dqdz)
    ana = 0.608 * TH.G * dqdz / (1 + 0.608 * qq)
    m = np.abs(nv2 - nd2) > 1e-12
    e3 = float(np.max(np.abs((nv2 - nd2)[m] - ana[m]) / np.abs((nv2 - nd2)[m])))
    chk("恒等式 N_v²−N_d² = 0.608g(dq/dz)/(1+0.608q)", e3 < 1e-12, "max_rel=%.2e" % e3)

    # ---- 4. N_v² < N_d²（dq/dz<0 时），且 q≡0 时退化 ----
    chk("dq/dz<0 ⟹ N_v²<N_d²", bool(np.all(nv2 < nd2)))
    e4 = float(np.max(np.abs(TH.brunt_vaisala_virtual_sq(Tk, np.zeros(n), dtdz,
                                                         np.zeros(n)) - nd2)))
    chk("q≡0 ⟹ N_v²≡N_d²", e4 < 1e-15, "max_abs=%.2e" % e4)

    # ---- 5. Ri_moist = Ri_dry(1−δ) 精确成立（干湿共用 S²）----
    S2 = rng.uniform(1e-7, 1e-4, n)
    rid = nd2 / S2
    rim = nv2 / S2
    dl = (nd2 - nv2) / nd2
    e5 = float(np.max(np.abs(rim - rid * (1 - dl)) / np.maximum(np.abs(rim), 1e-30)))
    chk("Ri_moist = Ri_dry·(1−δ)", e5 < 1e-12, "max_rel=%.2e" % e5)

    # ---- 6. 阈值等价：干口径 0.25 ⟺ 湿口径 0.25/(1−δ) ----
    rs, amp = ri_threshold_equivalent(dl, 0.25)
    mlt = dl < 1
    e6 = (float(np.max(np.abs(rs[mlt] * (1 - dl[mlt]) - 0.25))) if mlt.any() else 0.0)
    chk("阈值等价 Ri*·(1−δ) = 0.25 （δ<1）", e6 < 1e-12, "max_abs=%.2e" % e6)
    chk("δ≥1 ⟹ Ri* = +inf（湿口径无条件不稳定）",
        bool(np.all(np.isinf(rs[~mlt]))) if (~mlt).any() else True,
        "n(δ≥1)=%d" % int((~mlt).sum()))
    chk("δ>0 ⟹ 阈值放大倍数 ≥ 1（干判据恒偏松）",
        bool(np.all(amp[dl > 0] >= 1.0)))
    # δ<0（水汽随高度增加）时干判据偏保守，必须如实返回 <1 而不是对称化
    dneg = np.array([-0.2, -0.05])
    rs2, amp2 = ri_threshold_equivalent(dneg, 0.25)
    chk("δ<0 ⟹ 放大倍数 < 1（如实反映保守性）",
        bool(np.all(amp2 < 1.0)), "amp=%s" % np.round(amp2, 4).tolist())

    if verbose:
        print("  ---- 通过 %d / %d ----" % (ok, tot))
    return ok, tot


def endpoint_risk(sfc):
    """端点（起降机场）风险：湿密度高度惩罚 + 载重折算 + 阵风/侧风。"""
    p, Tc, q = sfc["p"], sfc["T_c"], sfc["q"]
    da_d = TH.dry_density_altitude_ft(p, Tc)
    da_m = TH.moist_density_altitude_ft(p, Tc, q)
    m2 = TH.TakeoffModel(exponent_m=2)
    m1 = TH.TakeoffModel(exponent_m=1)
    return dict(
        da_dry_ft=float(da_d), da_moist_ft=float(da_m),
        dq_ft=float(da_m - da_d),
        penalty_kg_m2=float(m2.penalty_kg(p, Tc, q)),
        penalty_kg_m1=float(m1.penalty_kg(p, Tc, q)),
        penalty_pax_m2=float(m2.penalty_passengers(p, Tc, q)),
        q_gkg=float(q * 1000), T_c=float(Tc),
        gust_kt=sfc.get("gust_kt", np.nan),
        crosswind_kt=sfc.get("crosswind_kt", np.nan),
    )
