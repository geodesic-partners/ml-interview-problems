"""Linear regression by gradient descent

Fit ŷ = Xw + b by batch gradient descent on the mean squared error L = (1/n) Σ (x_i·w + b − y_i)². Start from w = 0 and b = 0. At each of `steps` iterations, compute both gradients from the current parameters, then update w ← w − lr·∂L/∂w and b ← b − lr·∂L/∂b.

Return the final parameters as a list `[w₁, …, w_d, b]`.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-linear-regression-gradient-descent?utm_source=github&utm_medium=repo
"""
import numpy as np

def fit_linear_gd(X, y, lr, steps):
    # X: n x d, y: n targets. Return [w_1, ..., w_d, b].
    pass


# ---- tests: python linear_regression_gradient_descent.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1],[2],[3]],[2,4,6],0.1,1],\"out\":[1.8666666666666665,0.8]},{\"args\":[[[1],[2],[3]],[3,5,7],0.05,2000],\"out\":[2.00000000000139,0.9999999999968399]},{\"args\":[[[1,2],[3,4]],[1,2],0.01,0],\"out\":[0,0,0]},{\"args\":[[[1,2],[3,4],[5,7]],[1,2,4],0.01,50],\"out\":[0.2518683508893995,0.36642500236775866,0.02669013589260803]},{\"args\":[[[0.5,-1,2],[1.5,0.3,-0.2],[-0.7,0.8,0.1],[0.2,-0.4,1]],[1,0.5,-0.2,0.9],0.05,300],\"out\":[0.2265648057924438,-0.4064040875340351,0.1343023686923952,0.3332602583776953]}]")

    def close(a, b, rt=1e-7, at=1e-9):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(fit_linear_gd(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
