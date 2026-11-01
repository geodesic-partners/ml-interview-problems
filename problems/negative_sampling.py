"""Skip-gram with negative sampling: loss and gradients

Given the centre-word vector `v`, the true context vector `u` and k negative vectors `U_neg` (k × d), with s = u·v and s_j = u_j·v, the loss is

L = −log σ(s) − Σ_j log σ(−s_j).

Return `[L, dL/dv, dL/du, dL/dU_neg]`, the last a k × d list. Dot products may reach ±1000.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-negative-sampling?utm_source=github&utm_medium=repo
"""
import numpy as np

def sgns_loss_grad(v, u, U_neg):
    # Return [loss, grad_v, grad_u, grad_U_neg].
    pass


# ---- tests: python negative_sampling.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[1,0],[1,0],[[0,1]]],\"out\":[1.0064088680781682,[-0.2689414213699951,0.5],[-0.2689414213699951,0],[[0.5,0]]]},{\"args\":[[10,0],[-100,0],[[100,0]]],\"out\":[2000,[200,0],[-10,0],[[10,0]]]},{\"args\":[[0.5,-1,2],[1,1,1],[[1,0,0],[0,1,0],[-1,-1,2]]],\"out\":[5.999799694529676,[-0.5489792499739086,-0.9024971598057681,1.7956005909324573],[-0.09121276190317817,0.18242552380635635,-0.3648510476127127],[[0.3112296656009273,-0.6224593312018546,1.2449186624037092],[0.13447071068499755,-0.2689414213699951,0.5378828427399902],[0.4945065286847034,-0.9890130573694068,1.9780261147388136]]]},{\"args\":[[0,0],[3,4],[[1,1],[2,2]]],\"out\":[2.0794415416798357,[0,-0.5],[0,0],[[0,0],[0,0]]]},{\"args\":[[30,0],[30,0],[[-30,0],[0,30]]],\"out\":[0.6931471805599453,[0,15],[0,0],[[0,0],[15,0]]]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(sgns_loss_grad(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
