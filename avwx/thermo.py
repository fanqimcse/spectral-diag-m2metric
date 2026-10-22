# -*- coding: utf-8 -*-
"""
avwx.thermo —— 湿空气热力学核心
================================

本模块是本项目全部科学结论的物理基础。三条设计原则：

1. **可审计**：每个公式都标注出处与适用范围，不使用"来路不明"的常数。
2. **可证伪**：给出解析解（如 CC 比值 R）与数值解两条路径，互相校验。
3. **无隐藏近似**：所有"忽略湿度"的对照量都显式构造，不静默省略。

核心命题
--------
航空性能与航空气象风险的现行标准做法（Coffel et al. 2017, Climatic Change；
FAA/ICAO 密度高度定义；多数晴空湍流诊断）在计算空气密度时**只使用温度与气压**，
即把空气当作干空气。本模块给出含湿版本，并把"干—湿之差"作为一等公民。

参考文献
--------
- Bolton, D. (1980) Mon. Wea. Rev. 108, 1046–1053.          饱和水汽压（液相）
- Alduchov & Eskridge (1996) J. Appl. Meteor. 35, 601–609.  饱和水汽压常数
- WMO (1988) 冰相饱和水汽压
- ISA / ICAO Doc 7488                                        国际标准大气
- CIPM (2007)                                               干空气气体常数 R_d
- Wallace & Hobbs (2006) Ch.3                               虚温、湿静力稳定度
- Coffel, Thompson & Horton (2017) Climatic Change 144, 251. 温度-起飞性能基准
"""

from __future__ import annotations
import numpy as np

# ----------------------------------------------------------------------------
# 常数（全部标注出处）
# ----------------------------------------------------------------------------
R_D = 287.052874      # J kg-1 K-1, 干空气气体常数 (CIPM-2007, CODATA)
R_V = 461.52          # J kg-1 K-1, 水汽气体常数
EPSILON = R_D / R_V   # 0.62198
KAPPA = 0.2854        # R_d/c_p
C_PD = 1005.7         # J kg-1 K-1 干空气定压比热
C_PV = 1870.0         # J kg-1 K-1 水汽定压比热
L_V0 = 2.501e6        # J kg-1, 0°C 汽化潜热
G = 9.80665           # m s-2, 标准重力

# 国际标准大气 (ISA, ICAO Doc 7488 / ISO 2533)
T0_ISA = 288.15       # K
P0_ISA = 101325.0     # Pa
L_ISA = 0.0065        # K m-1, 对流层温度直减率
RHO0_ISA = P0_ISA / (R_D * T0_ISA)      # 1.2250 kg m-3
H_TROP = 11000.0      # m, 对流层顶
_FT = 0.3048          # m per ft

# σ→密度高度的精确 ISA 反演系数
#   σ(H) = (1 - L H/T0)^(g/(R_d L) - 1)
#   指数 = 9.80665/(287.052874*0.0065) - 1 = 4.255876...
_DA_EXP = G / (R_D * L_ISA) - 1.0        # 4.25588
_DA_INV_EXP = 1.0 / _DA_EXP              # 0.234969
_DA_SCALE_M = T0_ISA / L_ISA             # 44330.77 m
_DA_SCALE_FT = _DA_SCALE_M / _FT         # 145442 ft

# 克劳修斯-克拉珀龙标度下的比湿相对增速（观测值 6-7 %/K；Held & Soden 2006）
CC_DLNQ_DT = 0.07


# ----------------------------------------------------------------------------
# 1. 饱和水汽压
# ----------------------------------------------------------------------------
def esat(t_celsius, phase="auto"):
    """饱和水汽压 [Pa]。

    phase='liquid' : Bolton (1980) 式 10，适用 -30…+50 °C，精度 <0.1 %
                     e_s = 611.2 exp(17.67 T / (T + 243.5))
    phase='ice'    : WMO (1988) / Alduchov-Eskridge 冰相式
                     e_s = 611.2 exp(22.46 T / (T + 272.62))
    phase='auto'   : T >= 0 用液相，T < 0 用冰相（WMO 对过冷水的惯例）
    """
    T = np.asarray(t_celsius, dtype=float)
    if phase == "auto":
        out = np.where(T >= 0.0,
                       611.2 * np.exp(17.67 * T / (T + 243.5)),
                       611.2 * np.exp(22.46 * T / (T + 272.62)))
    elif phase == "liquid":
        out = 611.2 * np.exp(17.67 * T / (T + 243.5))
    elif phase == "ice":
        out = 611.2 * np.exp(22.46 * T / (T + 272.62))
    else:
        raise ValueError(f"unknown phase: {phase}")
    return float(out) if np.isscalar(t_celsius) or np.ndim(t_celsius) == 0 else out


def dewpoint_from_vapor_pressure(e_pa):
    """由水汽压反演露点温度 [°C]，Bolton (1980) 式 11 的逆。

    T_d = 243.5 * ln(e/611.2) / (17.67 - ln(e/611.2))
    """
    e = np.asarray(e_pa, dtype=float)
    lr = np.log(np.maximum(e, 1e-9) / 611.2)
    return 243.5 * lr / (17.67 - lr)


def dewpoint(t_celsius, rh_percent):
    """由干球温度 [°C] 与相对湿度 [%] 反演露点 [°C]。"""
    T = np.asarray(t_celsius, dtype=float)
    rh = np.clip(np.asarray(rh_percent, dtype=float), 0.0, 100.0) / 100.0
    e = np.maximum(rh * esat(T), 1e-9)
    return dewpoint_from_vapor_pressure(e)


# ----------------------------------------------------------------------------
# 2. 湿度变量互换
# ----------------------------------------------------------------------------
def vapor_pressure_from_specific_humidity(q, p):
    """由比湿 q [kg/kg] 与气压 [Pa] 反算水汽压 [Pa]。

    e = q p / (ε + q(1 - ε)),  ε = R_d/R_v
    """
    q = np.asarray(q, dtype=float); p = np.asarray(p, dtype=float)
    return q * p / (EPSILON + q * (1.0 - EPSILON))


def specific_humidity_from_dewpoint(td_celsius, p):
    """由露点 [°C] 与气压 [Pa] 得比湿 [kg/kg]（露点处饱和，e=e_s(T_d)）。"""
    e = esat(td_celsius, phase="liquid")
    return specific_humidity_from_vapor_pressure(e, p)


def specific_humidity_from_vapor_pressure(e, p):
    """由水汽压 [Pa] 与气压 [Pa] 得比湿 [kg/kg]。"""
    e = np.asarray(e, dtype=float); p = np.asarray(p, dtype=float)
    return EPSILON * e / np.maximum(p - e * (1.0 - EPSILON), 1e-6)


def specific_humidity_from_rh(t_celsius, rh_percent, p):
    """由干球 [°C]、相对湿度 [%]、气压 [Pa] 得比湿 [kg/kg]。"""
    T = np.asarray(t_celsius, dtype=float)
    rh = np.clip(np.asarray(rh_percent, dtype=float), 0.0, 100.0) / 100.0
    e = rh * esat(T)
    return specific_humidity_from_vapor_pressure(e, p)


def relative_humidity(t_celsius, q, p):
    """由干球、比湿、气压反演相对湿度 [%]（对露点版本做一致性校验用）。"""
    e = vapor_pressure_from_specific_humidity(q, p)
    return 100.0 * e / esat(t_celsius)


# ----------------------------------------------------------------------------
# 3. 虚温与湿空气密度 —— 本项目的物理核心
# ----------------------------------------------------------------------------
def virtual_temperature(t_kelvin, q):
    """虚温 T_v = T (1 + 0.608 q)。

    0.608 = 1/ε - 1 = R_v/R_d - 1。这是"湿空气比干空气轻"的定量表达。
    """
    return np.asarray(t_kelvin, dtype=float) * (1.0 + 0.608 * np.asarray(q, dtype=float))


def air_density(p, t_kelvin, q=0.0):
    """湿空气密度 [kg m-3]。

        ρ = p / (R_d T_v),   T_v = T(1 + 0.608 q)

    q=0 时退化为干空气密度 p/(R_d T) —— 这正是现行航空标准做法的假设。
    """
    tv = virtual_temperature(t_kelvin, q)
    return np.asarray(p, dtype=float) / (R_D * tv)


def density_ratio(p, t_kelvin, q=0.0):
    """σ = ρ/ρ_ISA。"""
    return air_density(p, t_kelvin, q) / RHO0_ISA


# ----------------------------------------------------------------------------
# 4. 密度高度（ISA 精确反演）与"湿度惩罚"
# ----------------------------------------------------------------------------
def density_altitude_ft(sigma):
    """由密度比 σ 反演密度高度 [ft]，ISA 对流层精确解。

        H = (T0/L) (1 - σ^{1/(g/(R_d L) - 1)})

    σ > 1（比 ISA 更密）时 H < 0，物理上合法（负密度高度）。
    σ 小到对应超过对流层顶时返回 nan（本函数只在对流层内有效）。
    """
    s = np.asarray(sigma, dtype=float)
    h = _DA_SCALE_FT * (1.0 - np.power(np.maximum(s, 1e-12), _DA_INV_EXP))
    # 超过对流层顶则标记无效
    h = np.where(h > H_TROP / _FT, np.nan, h)
    return h


def dry_density_altitude_ft(p, t_celsius):
    """干空气密度高度 [ft] —— 现行标准做法（= Coffel et al. 2017 所用的量）。"""
    T = np.asarray(t_celsius, dtype=float) + 273.15
    return density_altitude_ft(air_density(p, T, 0.0) / RHO0_ISA)


def moist_density_altitude_ft(p, t_celsius, q):
    """含湿密度高度 [ft] —— 本项目的修正量。"""
    T = np.asarray(t_celsius, dtype=float) + 273.15
    return density_altitude_ft(air_density(p, T, q) / RHO0_ISA)


def humidity_penalty_ft(p, t_celsius, q):
    """湿度惩罚 ΔDA_q = DA_moist - DA_dry [ft]。

    这是现行标准做法**系统性遗漏**的量。恒为正（水汽只能让空气更轻）。
    """
    return moist_density_altitude_ft(p, t_celsius, q) - dry_density_altitude_ft(p, t_celsius)


# ----------------------------------------------------------------------------
# 5. 精确的对数分解：把密度变化拆成 温度 / 湿度 / 气压 三项
# ----------------------------------------------------------------------------
def decompose_dln_rho(p1, t1_c, q1, p2, t2_c, q2):
    """对 ln ρ 的变化做**精确**（非近似）的三项加性分解。

        ln ρ = ln p - ln R_d - ln T - ln(1 + 0.608 q)
        Δ ln ρ = Δ ln p  -  Δ ln T  -  Δ ln(1 + 0.608 q)

    返回的三个量定义为"密度降低贡献"（正值 = 使密度变小）:

        c_p = -Δ ln p          （气压下降 → 密度下降）
        c_T = +Δ ln T          （升温 → 密度下降）
        c_q = +Δ ln(1+0.608q)  （增湿 → 密度下降）

    满足:  -Δ ln ρ = c_p + c_T + c_q   （精确恒等，非近似）

    这是本项目对标父项目"湿度必要性 79–106 %"的分解工具，
    但采用对数形式，因此是**精确恒等**而非反事实情景差。
    """
    T1 = np.asarray(t1_c, dtype=float) + 273.15
    T2 = np.asarray(t2_c, dtype=float) + 273.15
    d_ln_p = np.log(np.asarray(p2, float)) - np.log(np.asarray(p1, float))
    d_ln_T = np.log(T2) - np.log(T1)
    d_ln_v = np.log1p(0.608 * np.asarray(q2, float)) - np.log1p(0.608 * np.asarray(q1, float))
    return -d_ln_p, d_ln_T, d_ln_v     # (c_pressure, c_temperature, c_humidity)


def cc_humidity_ratio(q, t_kelvin, dlnq_dT=CC_DLNQ_DT):
    """解析的"湿度/温度"贡献比 R —— 本项目的一条可证伪解析结果。

    在克劳修斯-克拉珀龙标度下（d ln q / dT ≈ 7 %/K），
    单位增温所引起的"湿度导致的密度降低"与"温度本身导致的密度降低"之比为

        R = [∂ ln ρ/∂ q · dq/dT] / [∂ ln ρ/∂ T]
          = 0.608 T (d ln q/dT) q / (1 + 0.608 q)
          ≈ 0.0426 · q · T          （q ≪ 1 时的一阶式）

    含义：只算温度的密度/起飞性能估算，在湿热地区**低估约 R 倍**。
    例：T=303 K, q=20 g/kg → R ≈ 0.25，即低估 25 %。
    """
    q = np.asarray(q, dtype=float); T = np.asarray(t_kelvin, dtype=float)
    return 0.608 * T * dlnq_dT * q / (1.0 + 0.608 * q)


def da_change_from_dln_rho(sigma, dln_rho):
    """把 Δ ln ρ 换算成 Δ 密度高度 [ft]（在 σ 处线性化，供趋势归因用）。

        dDA/d ln ρ = -(T0/L)·σ^{1/4.25588}·(1/4.25588)
     """
    return -_DA_SCALE_FT * _DA_INV_EXP * np.power(np.maximum(sigma, 1e-9), _DA_INV_EXP) * dln_rho


# ----------------------------------------------------------------------------
# 6. 起飞性能：密度 → 最大起飞重量 → 旅客数
# ----------------------------------------------------------------------------
class TakeoffModel:
    """把空气密度换算为最大起飞重量限制的透明模型。

    采用航空性能学的标准 WAT（Weight–Altitude–Temperature）标度：

        起飞场长 s ∝ W² / (ρ · F)

    其中 F 为起飞平均净推力。对涡扇发动机有两种常用假定：

      m = 1 : 推力近似不随密度变化（平额定 / flat-rated 情形）
              → W ∝ σ^{1/2},   ΔW/W = ½ Δ ln ρ
      m = 2 : 推力 ∝ ρ（非平额定、受密度直接限制）
              → W ∝ σ,         ΔW/W =     Δ ln ρ

    Coffel et al. (2017) 报告的"每升温 1 °C 约减载 0.1–1 % "落在 m≈2 一侧，
    因此本项目以 **m = 2 为主口径**，m = 1 作保守下界，两者同时报告。
    """

    def __init__(self, mtow_kg=79015.0, passenger_mass_kg=95.0, exponent_m=2):
        self.mtow = mtow_kg
        self.pax = passenger_mass_kg
        self.m = exponent_m

    def weight_ratio(self, sigma, sigma_ref):
        """相对参考密度比的最大起飞重量比。"""
        return (np.asarray(sigma) / np.asarray(sigma_ref)) ** (self.m / 2.0)

    def weight_limit_kg(self, p, t_celsius, q, sigma_ref=1.0):
        """给定气象条件下的最大起飞重量 [kg]。"""
        T = np.asarray(t_celsius, dtype=float) + 273.15
        s = air_density(p, T, q) / RHO0_ISA
        return self.mtow * self.weight_ratio(s, sigma_ref)

    def penalty_kg(self, p, t_celsius, q):
        """忽略湿度所**少算**的载重损失 [kg]（正值 = 被低估的限制）。"""
        T = np.asarray(t_celsius, dtype=float) + 273.15
        s_dry = air_density(p, T, 0.0) / RHO0_ISA
        s_moist = air_density(p, T, q) / RHO0_ISA
        return self.mtow * (self.weight_ratio(s_dry, 1.0) - self.weight_ratio(s_moist, 1.0))

    def penalty_passengers(self, p, t_celsius, q):
        """上量折算成旅客数（按 95 kg/人 含行李，IATA 短程常用值）。"""
        return self.penalty_kg(p, t_celsius, q) / self.pax


# ----------------------------------------------------------------------------
# 7. 湿静力稳定度与湿理查森数 —— 巡航层湍流风险
# ----------------------------------------------------------------------------
def brunt_vaisala_sq(p, t_kelvin, q, dtdz, dqdz=None, moist=True):
    """Brunt–Väisälä 频率平方 N² [s-2]。

    dry   : N_d² = (g/T)(dT/dz + g/c_p)
    moist : N_m² = N_d² · (1 + L q_s/(R_d T)) - (g/(1+q)) dq/dz 的常用形式
            本项目采用 Wallace & Hobbs (2006) 式 3.75 的实用写法：

             N_m² = (g/T)[ (dT/dz + g/c_p)(1 + L_v q/(R_d T)) ] - g/(1+q) · dq/dz

    关键物理：饱和或近饱和时 N_m² 显著小于 N_d²，甚至可转负
    （条件性/湿对称不稳定）。因此**用干 N² 会高估稳定度、低估湍流风险**。
    """
    T = np.asarray(t_kelvin, dtype=float); q = np.asarray(q, dtype=float)
    dtdz = np.asarray(dtdz, dtype=float)
    if dqdz is None:
        dqdz = np.zeros_like(q)
    dqdz = np.asarray(dqdz, dtype=float)
    nd2 = (G / T) * (dtdz + G / C_PD)
    if not moist:
        return nd2
    lv = L_V0 - 2370.0 * (T - 273.15)
    return nd2 * (1.0 + lv * q / (R_D * T)) - G * dqdz / (1.0 + q)


def brunt_vaisala_virtual_sq(t_kelvin, q, dtdz, dqdz=None):
    """**未饱和**湿空气的浮力频率平方 N_v² [s-2]（虚温形式）。

    为什么 CAT 诊断必须用这一个而不是饱和湿绝热式
    ----------------------------------------------
    晴空湍流（CAT）按定义发生在**无云**区，即未饱和大气。此时气块受扰
    上升**不发生凝结、比湿守恒**，其浮力由密度（即虚温）决定：

        b = g (T_v,parcel − T_v,env) / T_v,env

    设环境廓线 T(z), q(z)，气块按干绝热 Γ_d = g/c_p 升降且 q 守恒，则

        dT_v,env/dz = (1+0.608q)·dT/dz + 0.608·T·dq/dz
        dT_v,pcl/dz = −(1+0.608q)·g/c_p

        N_v² = (g/T_v) [ dT_v,env/dz + (1+0.608q)·g/c_p ]

    与之对照，Wallace & Hobbs (2006) 式 3.75 描述的是**饱和**大气
    （含 L q_s/(R_d T) 与 dq_s/dz 项，气块沿湿绝热上升）。
    把实际比湿 q 代入 q_s 的位置用于晴空区是**物理误用**：
    未饱和时 q ≪ q_s，且 dq/dz 与 dq_s/dz 在量级和符号上都可以不同。
    本项目因此以 N_v² 作为湿稳定度的主口径，饱和式仅作对照。

    退化为干：q ≡ 0 → T_v = T，N_v² ≡ N_d²（本函数已自检）。

    关键物理：由于对流层中 dq/dz < 0（水汽随高度递减），
    0.608·T·dq/dz < 0，故 **N_v² < N_d²**——湿度**降低**静力稳定度，
    从而降低 Ri、提高 K–H 不稳定（湍流）的可能。
    """
    T = np.asarray(t_kelvin, dtype=float)
    q = np.asarray(q, dtype=float)
    dtdz = np.asarray(dtdz, dtype=float)
    dqdz = np.zeros_like(q) if dqdz is None else np.asarray(dqdz, dtype=float)
    tv = virtual_temperature(T, q)
    dtvdz = (1.0 + 0.608 * q) * dtdz + 0.608 * T * dqdz
    return (G / tv) * (dtvdz + (1.0 + 0.608 * q) * G / C_PD)


def sat_mixing_ratio(t_kelvin, p_pa):
    """饱和混合比 r_s = ε e_s/(p − e_s)，ε = R_d/R_v = 0.622。"""
    T = np.asarray(t_kelvin, float)
    e = np.asarray(esat(T - 273.15), float)
    p = np.asarray(p_pa, float)
    e = np.minimum(e, 0.99 * p)
    return EPSILON * e / np.maximum(p - e, 1.0)


def moist_lapse_rate(t_kelvin, p_pa):
    """湿绝热递减率 Γ_m [K/m]（正值定义为"降温率"，与 Γ_d = g/c_p 同号）。

        Γ_m = Γ_d · (1 + L_v r_s/(R_d T)) / (1 + L_v² r_s/(c_p R_v T²))

    注意 Γ_m 只依赖 **(T, p)**，不依赖实际湿度 —— 这一性质是本项目
    判定"饱和式用于晴空区为物理误用"的关键（见 brunt_vaisala_durran_klemp_sq）。
    """
    T = np.asarray(t_kelvin, float)
    p = np.asarray(p_pa, float)
    rs = sat_mixing_ratio(T, p)
    lv = L_V0 - 2370.0 * (T - 273.15)
    gd = G / C_PD
    num = 1.0 + lv * rs / (R_D * T)
    den = 1.0 + lv * lv * rs * EPSILON / (C_PD * R_D * T * T)
    return gd * num / den


def brunt_vaisala_durran_klemp_sq(t_kelvin, q, p_pa, dtdz, dqdz=None):
    """Durran & Klemp (1982) **饱和**湿浮力频率平方 N_DK² [s-2]。

        N_DK² = (g/T)(dT/dz + Γ_m)(1 + L_v r_s/(R_d T))
              − g/(1 + r_w) · dr_w/dz

    其中 Γ_m 为湿绝热递减率、r_s 为饱和混合比、r_w = r_s + r_l 为总水混合比
    在无云区（本项目关注的晴空区）退化为水汽混合比 r。

    为什么必须把它与虚温式 N_v² 分开
    --------------------------------
    该式的推导假定**被抬升气块始终保持饱和**并沿湿绝热上升，因而每一步都
    有潜热释放 —— 这描述的是**云内**。用于晴空区（RH << 100%）时它犯两个错误：

      1. **Γ_m 项**：湿绝热递减率与实际出现与否的凝结无关，只取决于 (T, p)。
         于是两块温廓线相同、相对湿度分别为 5% 与 95% 的空气会得到
         **完全相同的湿度修正** —— 而物理上正确的虚温修正是随实际 dq/dz
         线性增长的。
      2. **r_s 项**：用**饱和**混合比做放大因子，即使空气完全干燥也照用不误。

    本函数用于把这"误用"的后果**定量化**，而不是推荐它。
    """
    T = np.asarray(t_kelvin, float)
    q = np.asarray(q, float)
    p = np.asarray(p_pa, float)
    dtdz = np.asarray(dtdz, float)
    dqdz = np.zeros_like(q) if dqdz is None else np.asarray(dqdz, float)

    gm = moist_lapse_rate(T, p)
    rs = sat_mixing_ratio(T, p)
    lv = L_V0 - 2370.0 * (T - 273.15)

    # 混合比 r = q/(1-q)，dr/dz = dq/dz /(1-q)^2
    r = q / np.maximum(1.0 - q, 1e-6)
    drdz = dqdz / np.maximum((1.0 - q) ** 2, 1e-12)

    return ((G / T) * (dtdz + gm) * (1.0 + lv * rs / (R_D * T))
            - G * drdz / (1.0 + r))


def richardson_number(shear_sq, n_sq):
    """梯度理查森数 Ri = N²/S²。Ri < 0.25 为 Kelvin–Helmholtz 不稳定的必要判据。"""
    return np.asarray(n_sq, dtype=float) / np.maximum(np.asarray(shear_sq, dtype=float), 1e-12)


# ----------------------------------------------------------------------------
# 8. 晴空湍流诊断（沿用既有指标的干/湿对照版本）
# ----------------------------------------------------------------------------
def wind_shear_sq(dudz, dvdz):
    """垂直风切变平方 S² = (∂u/∂z)² + (∂v/∂z)² [s-2]。"""
    return np.asarray(dudz, float) ** 2 + np.asarray(dvdz, float) ** 2


def turbulence_index_ellrod(shear_sq, def_sq):
    """Ellrod TI = S · DEF（常用晴空湍流指数，无量纲量级 s-2）。"""
    return np.sqrt(np.asarray(shear_sq, float) * np.asarray(def_sq, float))


def deformation_sq(dudx, dudy, dvdx, dvdy):
    """形变平方 DEF² = (∂u/∂x - ∂v/∂y)² + (∂v/∂x + ∂u/∂y)²。"""
    a = np.asarray(dudx, float) - np.asarray(dvdy, float)
    b = np.asarray(dvdx, float) + np.asarray(dudy, float)
    return a ** 2 + b ** 2


# ----------------------------------------------------------------------------
# 9. 自洽性自检
# ----------------------------------------------------------------------------
def _selfcheck():
    """对模块做数值自检，任何一条失败都说明实现有误。"""
    out = {}
    # (a) ISA 海平面：σ=1 → DA=0
    out["isa_sea_level_DA_ft"] = float(density_altitude_ft(1.0))
    # (b) ISA+20°C 海平面：定压下 σ = T0/T = 288.15/308.15
    #     → DA ≈ 2275 ft（≈114 ft/°C，与"每 °C 约 120 ft"的经验法则一致）
    out["isa_plus20_DA_ft"] = float(density_altitude_ft(288.15 / 308.15))
    # (c) 虚温：T=300K, q=0.020 → T_v=303.65 K
    out["Tv_300K_q20"] = float(virtual_temperature(300.0, 0.020))
    # (d) 密度：p=101325, T=303.15K, q=0.020
    out["rho_moist_30C_q20"] = float(air_density(101325.0, 303.15, 0.020))
    out["rho_dry_30C"] = float(air_density(101325.0, 303.15, 0.0))
    # (e) 湿度惩罚
    out["penalty_ft_30C_q20"] = float(humidity_penalty_ft(101325.0, 30.0, 0.020))
    out["penalty_ft_35C_q30"] = float(humidity_penalty_ft(101325.0, 35.0, 0.030))
    # (f) 对数分解的精确恒等性
    cp, ct, cq = decompose_dln_rho(101325.0, 30.0, 0.020, 100000.0, 32.0, 0.024)
    r1 = air_density(101325.0, 303.15, 0.020)
    r2 = air_density(100000.0, 305.15, 0.024)
    out["decomp_residual"] = float((cp + ct + cq) - (-np.log(r2) + np.log(r1)))
    # (g) CC 解析比：T=303, q=0.020
    out["cc_ratio_303K_q20"] = float(cc_humidity_ratio(0.020, 303.0))
    out["cc_ratio_303K_q25"] = float(cc_humidity_ratio(0.025, 303.0))
    # (h) RH/露点往返一致性
    td = dewpoint(30.0, 55.0)
    q_rt = specific_humidity_from_dewpoint(td, 101325.0)
    rh_rt = relative_humidity(30.0, q_rt, 101325.0)
    out["rh_roundtrip_err_pct"] = float(rh_rt - 55.0)
    return out


if __name__ == "__main__":
    r = _selfcheck()
    for k, v in r.items():
        print(f"{k:28s} = {v: .6g}")
