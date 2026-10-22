# -*- coding: utf-8 -*-
"""缓存 AWC PIREP/AIREP 观测报告（用于替换纯模式的气候学统计）。

为什么它是最合适的检验基准
--------------------------
PIREP 的湍流字段 `/TB` 同时含**正例**（LGT/MOD/SEV/EXTRM）与**负例**（NEG）。
负例的存在才能构造真正的**对照表**并计算 AUC / POD / FAR，而不是只报一个
"命中率"。此前本项目只有模式自身的气候学统计，缺的正是这一环。

接口的三个坑（实测，均须绕开）
------------------------------
1. 不带 bbox → HTTP 400。文档也写明"bounding box 或 center+distance 必需"。
2. 单次返回上限 400 条 ⇒ 只靠 hours/hours=720 无法取到长时间序列。
   （实测 hours=72 与 hours=144 返回的行数与时间跨度都不随 hours 变化。）
3. 唯一可靠的回溯方式是 **date 参数**（作用于窗口**终点**）+ 短窗口滚动：
   实测每次返回约 1.5 小时窗口内的全部报告（CONUS 约 70–300 条）。
   因此按 1.5 h 步长向前滚动即可拼出完整 30 天。
"""
import datetime as dt
import json
import os
import time
import urllib.request
from collections import Counter

BASE = "https://aviationweather.gov/api/data/pirep"
# CONUS + 部分周边空域（lat0, lon0, lat1, lon1）
BBOX = "18,-128,52,-62"
OUT = "data/pirep"
STEP_H = 6.0           # 实测每窗口约 70 s，6 h 步长可在可接受时间内覆盖 12 天
SLEEP = 1.0            # AWC 限流 100 次/分钟


def get_window(start_utc, hours=None, tries=3):
    ds = start_utc.strftime("%Y%m%d_%H%M")
    h = STEP_H if hours is None else hours
    u = "%s?format=json&date=%s&hours=%g&bbox=%s" % (BASE, ds, h, BBOX)
    for k in range(tries):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "avwx/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf8", "replace"))
        except Exception as e:
            if k == tries - 1:
                raise
            time.sleep(4 + 4 * k)
    return []


def main(ndays=30, outname=None):
    os.makedirs(OUT, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    seen = {}
    t_end = now
    n_win = int(ndays * 24 / STEP_H)
    t0 = time.time()
    for i in range(n_win):
        try:
            rec = get_window(t_end)
        except Exception as e:
            print("  [%3d/%3d] %s  FAIL %r" % (i + 1, n_win, t_end, e), flush=True)
            t_end -= dt.timedelta(hours=STEP_H)
            continue
        n_new = 0
        for r in rec:
            if not isinstance(r, dict):
                continue
            key = (r.get("obsTime"), r.get("lon"), r.get("lat"),
                   r.get("fltLvl"), r.get("acType"))
            if key in seen:
                continue
            seen[key] = r
            n_new += 1
        print("  [%3d/%3d] <= %s  got=%4d new=%4d total=%6d  (%.0fs)"
              % (i + 1, n_win, t_end.strftime("%Y-%m-%d %H:%M"), len(rec),
                 n_new, len(seen), time.time() - t0), flush=True)
        t_end -= dt.timedelta(hours=STEP_H)
        time.sleep(SLEEP)

    rows = sorted(seen.values(), key=lambda r: r.get("obsTime") or 0)
    p = os.path.join(OUT, outname or "pirep_%ddays.json" % ndays)
    with open(p, "w", encoding="utf8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print("\n=== 缓存完成: %s   记录数 %d ===" % (p, len(rows)))

    def cls(r):
        v = [(r.get("tbInt1") or "").strip().upper(),
             (r.get("tbInt2") or "").strip().upper()]
        if any(x in ("SEV", "EXTRM", "MOD") for x in v):
            return "MOD+"
        if any(x == "NEG" for x in v):
            return "NEG"
        if any(x == "LGT" for x in v):
            return "LGT"
        return "none"

    c = Counter(cls(r) for r in rows)
    tot = sum(c.values())
    for k in ("MOD+", "NEG", "LGT", "none"):
        print("  %-6s %7d  (%.1f%%)" % (k, c.get(k, 0), 100.0 * c.get(k, 0) / max(tot, 1)))
    ts = [r["obsTime"] for r in rows if r.get("obsTime")]
    if ts:
        a = dt.datetime.fromtimestamp(min(ts), dt.UTC)
        b = dt.datetime.fromtimestamp(max(ts), dt.UTC)
        print("  时间跨度 %s .. %s" % (a.strftime("%Y-%m-%d %H:%M"),
                                       b.strftime("%Y-%m-%d %H:%M")))
    return rows


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    main(n)
