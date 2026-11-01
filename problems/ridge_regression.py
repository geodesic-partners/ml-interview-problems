"""Ridge regression in closed form

Given a design matrix `X` (n × d), targets `y` (length n) and a penalty `lam` ≥ 0, return the weights w that minimise ‖Xw − y‖² + lam·‖w‖². There is no intercept. When lam = 0 the tests guarantee XᵀX is invertible.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-ridge-regression?utm_source=github&utm_medium=repo
"""
import numpy as np

def ridge(X, y, lam):
    # X: n x d list of lists, y: list of n floats, lam: float >= 0.
    # Return the d weights as a list.
    pass


# ---- tests: python ridge_regression.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,0],[0,1],[1,1]],[1,2,3],0],\"out\":[1,2]},{\"args\":[[[1,0],[0,1],[1,1]],[1,2,3],1],\"out\":[0.875,1.375]},{\"args\":[[[2],[4]],[2,4],100],\"out\":[0.16666666666666666]},{\"args\":[[[1,2,3],[4,5,6]],[1,1],0.5],\"out\":[-0.24561403508771865,0.03508771929824439,0.3157894736842111]},{\"args\":[[[0.5,-1.2,3.3],[2.1,0,-0.7],[1,1,1],[-2.2,4.1,0.3],[0.9,-0.4,2.2]],[1.5,-0.3,2,0.7,1.1],0.01],\"out\":[0.2847881021380972,0.3445365781456732,0.5760548622179539]},{\"args\":[[[1,1],[1,1.0001],[1,0.9999]],[2,2.0001,1.9999],0.001],\"out\":[0.999833359440546,0.999833362772974]}]")

    def close(a, b, rt=1e-7, at=1e-9):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(ridge(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
