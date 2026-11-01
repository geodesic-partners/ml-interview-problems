"""The autocorrelation function of a series

Given a series `x` of length n and an integer `max_lag` with 0 ≤ max_lag < n, return the sample autocorrelations r_0, …, r_(max_lag), where

r_k = Σ_(t=0)^(n−k−1) (x_t − x̄)(x_(t+k) − x̄) / Σ_(t=0)^(n−1) (x_t − x̄)²

and x̄ is the mean of the whole series. The series is not constant.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-autocorrelation-function?utm_source=github&utm_medium=repo
"""
import numpy as np
def autocorrelation(x, max_lag):
    # Return [r_0, r_1, ..., r_max_lag].
    pass


# ---- tests: python autocorrelation_function.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[1,2,3,4,5],2],\"out\":[1,0.4,-0.1]},{\"args\":[[1,-1,1,-1,1,-1],1],\"out\":[1,-0.8333333333333334]},{\"args\":[[0.5,0.1,-0.3,0.2],0],\"out\":[1]},{\"args\":[[3,1,4,1,5,9,2,6],4],\"out\":[1,-0.17523640661938533,0.044326241134751775,0.1456855791962175,-0.41725768321513]},{\"args\":[[0.01,-0.02,0.015,0.003,-0.011,0.007],3],\"out\":[1,-0.6177268244575937,-0.07421104536489148,0.39238165680473375]},{\"args\":[[100,101,99,102,98,103],5],\"out\":[1,-0.7571428571428571,0.5142857142857142,-0.32857142857142857,0.14285714285714285,-0.07142857142857142]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(autocorrelation(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
