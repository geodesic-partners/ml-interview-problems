"""Spearman rank correlation (the information coefficient)

Given two equal-length lists `x` and `y`, neither constant, return Spearman's rank correlation: replace each list by its ranks, giving tied values the average of the ranks they span, and return the Pearson correlation of the two rank vectors.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-spearman-rank-correlation?utm_source=github&utm_medium=repo
"""
import numpy as np

def spearman(x, y):
    # Return the Spearman rank correlation as a float.
    pass


# ---- tests: python spearman_rank_correlation.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[1,2,3,4],[10,20,30,40]],\"out\":1},{\"args\":[[1,2,3,4],[1,8,27,1000]],\"out\":1},{\"args\":[[1,2,2,3],[4,3,3,1]],\"out\":-1},{\"args\":[[0.3,-1.2,0.8,0.8,2,-0.5],[0.01,-0.02,0,0.03,0.02,0]],\"out\":0.75},{\"args\":[[1,2],[2,1]],\"out\":-1},{\"args\":[[5,3,5,1,3,5,2],[1,2,3,4,5,6,7]],\"out\":-0.35552856456843185}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(spearman(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
