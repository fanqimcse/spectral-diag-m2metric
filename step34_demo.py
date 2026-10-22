# -*- coding: utf-8 -*-
"""step34 —— P4-2：用 m2metric 工具包做"RMSE 会说谎、M2 会说真话"示范。

展示一个业务同化中心会遇到的实际场景：

    你开发了一个"尺度分离"后处理（高斯平滑），想评估它对下游
    飞行气象风险诊断（Ri/N²/S² 等导数型量）的价值。

    · 用 RMSE（M0）评估：改善只有 ~7%，"聊胜于无"，可能被砍掉；
    · 用 M2 评估：改善 ~38%，"立竿见影"，值得上线。

    RMSE 在这里**系统性低估**了尺度分离对下游产品的价值，因为它度量
    的是场误差（零阶矩），而下游诊断的误差由二阶矩 M2 主导。

本脚本分两部分：
  [A] 合成演示：用 m2metric 的 discriminate() 展示判别力差异；
  [B] 真实数据对照：引用 step31（EnOI）与 step33（GFS业务）的真实数字，
      生成"RMSE vs M2"对照表。

自包含：只依赖 m2metric 包 + numpy + step31/33 的既有 JSON。
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from m2metric import discriminate, compute_m2


def demo_synthetic():
    """合成演示：一个误差场，两种后处理，RMSE 与 M2 的判读分歧。"""
    n = 128
    k0 = 0.08
    kx = np.fft.fftfreq(n)
    ky = np.fft.fftfreq(n)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX ** 2 + KY ** 2)

    Eb = np.exp(-(kr / k0) ** 2)
    Ea = Eb * (1 - np.exp(-(kr / 0.02) ** 2))   # 同化：只去大尺度
    Es = Eb * np.exp(-(kr / 0.15) ** 2)          # 尺度分离：只去小尺度

    bg_list, an_list, sc_list = [], [], []
    for s in range(12):
        rng = np.random.default_rng(100 + s)

        def synth(Ek):
            amp = np.sqrt(Ek)
            phase = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            return np.fft.ifft2(amp * phase).real

        bg_list.append(synth(Eb))
        an_list.append(synth(Ea))
        sc_list.append(synth(Es))

    d = discriminate(bg_list, an_list, sc_list, dx=1.0, dy=1.0)
    return d


def main():
    print("=" * 98)
    print("step34  P4-2：m2metric 示范性收益 —— RMSE 会说谎、M2 会说真话")
    print("=" * 98)

    # ---------- [A] 合成演示 ----------
    print("\n[A] 合成演示：同化(去大尺度) vs 尺度分离(去小尺度) 的判别力")
    print("-" * 98)
    d = demo_synthetic()
    print("  M0（RMSE² 代理）判别力 = %+.1f%%   95%%CI [%+.1f%%, %+.1f%%]"
          % (100 * d["disc_M0"], 100 * d["ci_M0"][0], 100 * d["ci_M0"][1]))
    print("  M2（诊断误差代理）判别力 = %+.1f%%   95%%CI [%+.1f%%, %+.1f%%]"
          % (100 * d["disc_M2"], 100 * d["ci_M2"][0], 100 * d["ci_M2"][1]))
    print("  Δ判别力（M2−M0）       = %+.1f%%   显著=%s"
          % (100 * d["disc_diff"], d["significant"]))
    print("  ⇒ 若只信 RMSE，会严重低估尺度分离对下游诊断的价值。")

    # ---------- [B] 真实数据对照 ----------
    print("\n[B] 真实数据对照：EnOI（尺度分离型）vs GFS 业务（全尺度）")
    print("-" * 98)

    s31 = json.load(open("data/step31_m2_ci.json", encoding="utf8"))["summary"]
    s33 = json.load(open("data/step33_cosmic_verify.json", encoding="utf8"))

    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  场景 1：EnOI 同化（光滑增量，只动大尺度）—— 真实 GFS 8 配对  │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print("    同化对 M0 的改善 : %+6.1f%%    平滑对 M0 的改善 : %+6.1f%%"
          % (100 * s31["dM0_an"], 100 * s31["dM0_sbg"]))
    print("    同化对 M2 的改善 : %+6.1f%%    平滑对 M2 的改善 : %+6.1f%%"
          % (100 * s31["dM2_an"], 100 * s31["dM2_sbg"]))
    print("    → RMSE 判读：平滑仅比同化好 %+.1f%%（判别力），'聊胜于无'"
          % (100 * s31["disc_M0"]))
    print("    → M2  判读：平滑比同化好 %+.1f%%（判别力），'立竿见影'"
          % (100 * s31["disc_M2"]))
    print("    → RMSE 掩盖了 %+.1f 个百分点的真实下游收益"
          % (100 * (s31["disc_M2"] - s31["disc_M0"])))

    print("\n  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  场景 2：GFS 业务同化（混合 4DEnVar，全尺度）—— 掩星 180 样本 │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print("    场误差改善 : %+6.1f%%    梯度误差改善 : %+6.1f%%"
          % (100 * s33["d_field"]["median"], 100 * s33["d_grad"]["median"]))
    print("    → 全尺度同化下，场与梯度同步改善，RMSE 与 M2 一致（都不说谎）")

    # ---------- 汇总落盘 ----------
    out = dict(
        synthetic=d,
        real=dict(
            enoi=s31,
            gfs_business=s33,
        ),
    )
    with open("data/step34_demo.json", "w", encoding="utf8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=str)
    print("\n已写出 data/step34_demo.json")


if __name__ == "__main__":
    main()
