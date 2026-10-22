# -*- coding: utf-8 -*-
"""step12c —— 分离"流依赖"与"信息泄漏"：同日异时效成员对照。

step12b 给出两个端点：
    留一（成员全是**其它日期**）        q −2.24%   无效
    泄漏（成员含目标日**背景所用时效**）  q +27.46%  极好   ← 但这是作弊

泄漏版之所以好是平凡的：背景就是 f024，而成员里恰好有 (f024 − analysis)，
也就是"真误差的相反数"，等于把答案塞进了集合。它**不能**用来论证
"成员必须流依赖"。

真正的判据是第三种情形：
    成员 = 目标日**其它时效**的预报误差（f006、f012），而背景是 f024。
这些成员**不含** f024 的误差（无泄漏），但**共享当日的天气系统与误差结构**。
若这一组显著优于"全其它日期"，就证明：
    决定性因素是"成员与当日误差的**相关性**"（流依赖），
    而不是"集合里有没有答案"。

三种情形并列，即可把机制钉死。
"""
import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from avwx import enoi as EN
import step10_realerror_osse as S10
import step12_enoi as S12

KEYS = S10.KEYS
OERR = S10.OERR
BG_LEAD = 24
OTHER_LEADS = (6, 12)          # 同日但**不是**背景所用时效 → 无泄漏


def pair_for(tspec, lead):
    """返回 (analysis, forecast) 这一对误差样本。"""
    nat = S10.prepare(S10.AG.load(S10.native_path(tspec[0], tspec[1], 0),
                                  bbox=S10.BBOX))
    b = S10.bg_cycle(tspec[0], tspec[1], lead)
    p = S10.native_path(b[0], b[1], lead)
    if not os.path.exists(p):
        return None
    bg = S10.prepare(S10.AG.load(p, bbox=S10.BBOX))
    out = ({v: nat["f"][v] for v in ("T", "q", "u", "v")},
           {v: bg["f"][v] for v in ("T", "q", "u", "v")})
    del nat, bg
    gc.collect()
    return out


def rmse(a, b, m):
    dv = (np.asarray(a, float) - np.asarray(b, float))[m]
    return float(np.sqrt(np.nanmean(dv ** 2)))


def pct(bg, an, nat, m):
    b = rmse(bg, nat, m)
    return 100 * (b - rmse(an, nat, m)) / b


def run_case(target, X, am, lat, lon, lev, others):
    same_day, pr = [], []
    for lead in OTHER_LEADS:
        pr_ = pair_for(target, lead)
        if pr_ is not None:
            same_day.append(pr_)
    base = S12.member_pairs(others)                 # 其它日期 × (6,12,24)
    leak = base + S12.member_pairs([target])        # 含背景所用时效 → 泄漏
    noday = base                                    # 只有其它日期
    samed = base + same_day                         # 同日异时效（无泄漏）★

    eobs = S12.to_enoi_obs(X["obs"], use=("sfc", "raob", "ac"),
                           lat=lat, lon=lon, lev=lev)
    out = {}
    for tag, pz in (("A 仅其它日期", noday),
                    ("B 其它日期 + 同日异时效 ★", samed),
                    ("C 泄漏（含背景所用时效）", leak),
                    ("D 仅同日异时效", same_day)):
        if len(pz) < 3:                    # build_members 要求 N>=3
            print("  %-26s 成员不足（N=%d）" % (tag, len(pz)))
            continue
        sb = S12.sigma_from_pairs(pz)
        mem, _ = EN.build_members(pz, sb, clip={"q": 0.02})
        E = EN.EnOI(mem, lat, lon, lev, sigma_b=sb, tiles=(6, 5),
                    tile_pad_km=350.0, max_obs=1000)
        an = E.update(X["bg"], eobs, OERR)
        vals = {v: pct(X["bg"][v], an[v], X["nat"][v], am) for v in KEYS}
        out[tag] = dict(N=len(pz), **vals)
        print("  %-26s N=%2d  " % (tag, len(pz)) +
              "  ".join("%s %+6.2f%%" % (v, vals[v]) for v in KEYS))
        del E, an, mem
        gc.collect()
    return out


def main():
    t0 = time.time()
    print("=" * 96)
    print("step12c  集合协方差需要什么：跨日期样本 vs 同日流依赖结构")
    print("=" * 96)
    res = {}
    for target in S12.TARGETS:
        try:
            X = S12.load_case(target)
        except Exception as e:
            print("%s 跳过: %s" % (target, str(e)[:80]))
            continue
        lat, lon, lev = X["lat"], X["lon"], X["lev"]
        am = S10.impact_mask(X["obs"]["ac"][1], lat, lon, lev) & X["above"]
        others = [t for t in S12.TARGETS if t != target]
        print("\n目标 %s%sz" % target, flush=True)
        res["%s%sz" % target] = run_case(target, X, am, lat, lon, lev, others)
        del X
        gc.collect()
    with open("data/step12c_flowdep.json", "w", encoding="utf8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n已写出 data/step12c_flowdep.json   用时 %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
