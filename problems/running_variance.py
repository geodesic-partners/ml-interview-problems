"""Running mean and variance in one pass

Values arrive one at a time in `xs`. After each value, record `[mean, variance]` of everything seen so far, where the variance is the sample variance (divide by n − 1), defined as 0.0 when n = 1. Return the list of records.

Use O(1) memory beyond the output: you may not keep the values. The answer must be accurate for values of size 10⁹ that differ by single units.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-running-variance?utm_source=github&utm_medium=repo
"""
def running_stats(xs):
    # Return [[mean, variance], ...], one record after each value.
    pass


# ---- tests: python running_variance.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[2,4,6]],\"out\":[[2,0],[3,2],[4,4]]},{\"args\":[[1000000004,1000000007,1000000013,1000000016]],\"out\":[[1000000004,0],[1000000005.5,4.5],[1000000008,21],[1000000010,30]]},{\"args\":[[5]],\"out\":[[5,0]]},{\"args\":[[1,1,1,1]],\"out\":[[1,0],[1,0],[1,0],[1,0]]},{\"args\":[[-3.5,2.25,0,10,-1]],\"out\":[[-3.5,0],[-0.625,16.53125],[-0.41666666666666663,8.395833333333334],[2.1875,32.723958333333336],[1.55,26.575]]},{\"args\":[[1000000000000,1000000000001,1000000000002]],\"out\":[[1000000000000,0],[1000000000000.5,0.5],[1000000000001,1]]},{\"args\":[[0.1,0.2,0.3,0.4,0.5,0.6]],\"out\":[[0.1,0],[0.15000000000000002,0.004999999999999999],[0.2,0.009999999999999995],[0.25,0.016666666666666666],[0.3,0.025],[0.35,0.034999999999999996]]}]")

    def close(a, b, rt=0.000001, at=0.000001):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(running_stats(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
