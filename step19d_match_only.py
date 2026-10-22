# -*- coding: utf-8 -*-
"""step19d —— 只做匹配，不做核验（把跨年样本并进同一个缓存）。

为什么单独拆出来
----------------
step11b 的 `main()` 会连带跑一整套 Spearman/AUC 的 bootstrap 核验，
在 4.4 万条样本上要一个多小时。而扩充跨年样本时我们**只需要**它前半段的
匹配功能；核验交给 step19c（按年份切分）和重跑的 step11b 分析阶段。

本脚本只调用 `step11b.build_matched()` —— 它自带断点续算
（按 GFS 时次跳过已算过的），所以增量运行只补新时次。
"""
import pickle
import sys
import time

sys.path.insert(0, ".")
import step11b_edr_verification as S11B

if __name__ == "__main__":
    t0 = time.time()
    before = []
    if S11B.os.path.exists(S11B.PKL):
        with open(S11B.PKL, "rb") as fh:
            before = pickle.load(fh)
    print("匹配前：%d 组 / %d 样本"
          % (len(before), sum(len(r["p"]) for r in before)), flush=True)
    rows = S11B.build_matched()
    print("\n匹配后：%d 组 / %d 样本"
          % (len(rows), sum(len(r["p"]) for r in rows)), flush=True)

    import collections
    yr = collections.Counter()
    for r in rows:
        yr[str(r["time"])[:4]] += len(r["p"])
    print("年份构成：" + "  ".join("%s:%d" % (k, v)
                               for k, v in sorted(yr.items())))
    print("用时 %.0f s" % (time.time() - t0), flush=True)
