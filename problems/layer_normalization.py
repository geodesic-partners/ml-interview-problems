"""Layer normalisation

Given `X` (N × D), and `gamma`, `beta` of length D, return Y with Y_n = γ ⊙ (X_n − μ_n) / √(σ²_n + eps) + β, where μ_n and σ²_n are the mean and the biased variance (divide by D) of row n.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-layer-normalization?utm_source=github&utm_medium=repo
"""
import numpy as np

def layer_norm(X, gamma, beta, eps):
    # X: N x D. Normalise each row over its D features.
    pass


# ---- tests: python layer_normalization.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,2,3]],[1,1,1],[0,0,0],0],\"out\":[[-1.224744871391589,0,1.224744871391589]]},{\"args\":[[[5,5,5,5]],[1,1,1,1],[0.5,0.5,0.5,0.5],0.00001],\"out\":[[0.5,0.5,0.5,0.5]]},{\"args\":[[[1,2],[10,30]],[2,1],[0,1],0],\"out\":[[-2,2],[-2,2]]},{\"args\":[[[0.3,-1.2,4.4,0,2.5],[1000000,1000001,1000002,1000003,1000004]],[1,0.5,2,1,1],[0,0,0,0.1,-0.1],0.00001],\"out\":[[-0.4506759575171558,-0.6009012766895411,3.204806809010886,-0.5009012766895411,0.5509763830803363],[-1.4142100268524473,-0.35355250671311184,0,0.8071050134262236,1.3142100268524473]]}]")

    def close(a, b, rt=1e-9, at=1e-9):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(layer_norm(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
