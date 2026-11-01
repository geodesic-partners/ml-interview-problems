"""Logistic regression with an L2 penalty

Train p = σ(Xw + b) by batch gradient descent on

> L = −(1/n) Σ [y_i log p_i + (1 − y_i) log(1 − p_i)] + (lam/2)‖w‖²,

starting from w = 0, b = 0, for `steps` iterations with learning rate `lr`. The bias is not penalised. Compute both gradients from the current parameters, then update both. Return `[w₁, …, w_d, b]`. Logits in the tests can exceed 700 in magnitude.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-logistic-regression?utm_source=github&utm_medium=repo
"""
import numpy as np

def fit_logistic(X, y, lr, steps, lam):
    # X: n x d, y: n labels in {0, 1}. Return [w_1, ..., w_d, b].
    pass


# ---- tests: python logistic_regression.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1],[-1]],[1,0],1,1,0],\"out\":[0.5,0]},{\"args\":[[[0,0],[1,1],[2,2],[3,3]],[0,0,1,1],0.5,500,0.1],\"out\":[1.0111758567463955,1.0111758567463955,-3.0334297813217925]},{\"args\":[[[1000],[-1000]],[1,0],0.002,3,0],\"out\":[1,0]},{\"args\":[[[0.5,1.2],[1.5,-0.3],[-1,0.4],[2,2],[-0.5,-1.5],[0.1,0.9]],[1,1,0,1,0,0],0.2,200,0.01],\"out\":[2.6967502647885864,0.5402731237957973,-1.200155278413129]},{\"args\":[[[1,0],[0,1]],[1,1],0.1,0,0.5],\"out\":[0,0,0]}]")

    def close(a, b, rt=1e-7, at=1e-9):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(fit_logistic(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
