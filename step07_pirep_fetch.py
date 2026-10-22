# -*- coding: utf-8 -*-
"""
拉取 Iowa Environmental Mesonet (IEM) 的 PIREP 存档作为**观测真值**。

为什么必须换掉 AWC 官方接口
----------------------------
1. AWC `/api/data/pirep` 对 hours 参数的响应不稳定：实测 hours=24/96/336/720
   一律只返回 3 条，而同一个时刻的**缓存文件**有 2022 条 —— 无法用于归档抓取。
2. AWC 的 `aircraftreports.cache.csv.gz` 只含"当前"滚动窗口：实测跨度
   **1.48 小时**，且绝大多数是 AIREP（自动例行报，2014/2022），真正带 `/TB`
   的湍流报告只有 3 条。靠反复轮询攒样本不可行。
3. IEM 的 pireps.py 接受 sts/ets，可按天批量返回带 TURBULENCE 字段的记录，
   这是本项目唯一能拿到**足够样本量 + 含负例**的公开湍流观测源。

注意：IEM 的 `hours` 参数会返回 HTTP 422，必须用 sts/ets。
"""
import sys
import datetime as dt

import pandas as pd

BASE = ("https://mesonet.agron.iastate.edu/cgi-bin/request/gis/pireps.py"
        "?sts=%s&ets=%s&fmt=csv")


def fetch_range(t0, t1, day_chunk=3):
    """按 day_chunk 天分段拉取后拼接（单次窗口过长会被 IEM 拒绝）。"""
    frames = []
    cur = t0
    while cur < t1:
        nxt = min(cur + dt.timedelta(days=day_chunk), t1)
        url = BASE % (cur.strftime("%Y-%m-%dT%H:%MZ"),
                      nxt.strftime("%Y-%m-%dT%H:%MZ"))
        try:
            df = pd.read_csv(url)
        except Exception as e:
            print("  [%s..%s] FAIL %r" % (cur, nxt, e), flush=True)
            cur = nxt
            continue
        print("  [%s..%s] rows=%d" % (cur.date(), nxt.date(), len(df)), flush=True)
        frames.append(df)
        cur = nxt
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates()


def main(days=30, out="data/pirep/pirep_iem.csv"):
    now = dt.datetime(2026, 10, 21, 6, 0)
    t0 = now - dt.timedelta(days=days)
    print("拉取 IEM PIREP: %s .. %s" % (t0, now))
    df = fetch_range(t0, now)
    if df.empty:
        print("未取得数据")
        return
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print("\n=== 合计 %d 条，已写 %s ===" % (len(df), out))
    print("列:", list(df.columns))
    for c in ("TURBULENCE", "ICING"):
        if c in df.columns:
            print("\n%s 分布:" % c)
            print(df[c].value_counts(dropna=False).head(12).to_string())
    if "VALID" in df.columns:
        v = pd.to_datetime(df["VALID"], format="%Y%m%d%H%M", errors="coerce")
        print("\n时间跨度:", v.min(), "->", v.max())
        print("按天计数:")
        print(v.dt.date.value_counts().sort_index().to_string())
    print("\nFL 描述:\n", df["FL"].describe().to_string() if "FL" in df.columns else "")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
