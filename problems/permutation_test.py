"""An exact permutation test for a difference in means

Two groups of integers, `a` and `b`. Under the null hypothesis the group labels are exchangeable. Consider every way of choosing len(a) of the pooled values to form a new first group, the rest forming the second. Return the fraction of these relabellings whose absolute difference in group means is at least the observed absolute difference |mean(a) − mean(b)|. The observed labelling is one of them.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-permutation-test?utm_source=github&utm_medium=repo
"""
from itertools import combinations
def permutation_p_value(a, b):
    # Return the exact two-sided p-value as a float.
    pass


# ---- tests: python permutation_test.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[7,9,12],[3,4,6]],\"out\":0.1},{\"args\":[[1,2],[1,2]],\"out\":1},{\"args\":[[5],[1,2,3,4]],\"out\":0.4},{\"args\":[[10,12,14,16],[11,13,15,17]],\"out\":0.6857142857142857},{\"args\":[[3,3,3],[3,3,3,3]],\"out\":1},{\"args\":[[-2,5,8,1,0],[4,9,12,7,6,10]],\"out\":0.03463203463203463},{\"args\":[[100],[0]],\"out\":1}]")

    def close(a, b, rt=0, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(permutation_p_value(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
