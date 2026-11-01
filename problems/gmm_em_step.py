"""One EM step for a Gaussian mixture

A 1-D Gaussian mixture has weights `pi`, means `mu` and variances `var` (each of length K). Given data `x`, perform one EM iteration and return `[new_pi, new_mu, new_var, loglik]`:

        - **E-step:** r_(ik) = π_kN(x_i; μ_k, σ²_k) / Σ_j π_jN(x_i; μ_j, σ²_j).

        - **M-step:** N_k = Σ_i r_(ik); new π_k = N_k/n; new μ_k = Σ_i r_(ik)x_i/N_k; new σ²_k = Σ_i r_(ik)(x_i − new μ_k)²/N_k.

        - **loglik:** Σ_i log Σ_k π_kN(x_i; μ_k, σ²_k), under the parameters you were given.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-gmm-em-step?utm_source=github&utm_medium=repo
"""
import numpy as np

def em_step(x, pi, mu, var):
    # Return [new_pi, new_mu, new_var, loglik].
    pass


# ---- tests: python gmm_em_step.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[-1,1],[0.5,0.5],[0,0],[1,1]],\"out\":[[0.49999999999999994,0.49999999999999994],[0,0],[1,1],-2.8378770664093453]},{\"args\":[[0,0.2,10,10.4],[0.5,0.5],[0,10],[1,1]],\"out\":[[0.5,0.5],[0.1,10.2],[0.010000000000000002,0.04000000000000007],-6.548342855058472]},{\"args\":[[0,1,2,9,10,11],[0.3,0.7],[1,9],[2,2]],\"out\":[[0.4999975742412612,0.5000024257587389],[0.9999953541328674,9.999960982376587],[0.6666661739040113,0.6669765040928223],-14.025001332788866]},{\"args\":[[1.2,-0.4,3.3,2.8,0.1,5.5,4.9,6.2],[0.2,0.5,0.3],[0,3,6],[1,0.5,2]],\"out\":[[0.3475367004464585,0.2820130339515119,0.3704502656020296],[0.2335372914492443,2.962516607668044,5.488912041495864],[0.42869723851744684,0.5665444275380438,0.4486773779224253],-18.373298400448466]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(em_step(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
