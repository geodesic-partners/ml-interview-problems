"""Binary cross-entropy from logits

Given logits `z` and labels `y` (each 0 or 1) of the same length, return the mean binary cross-entropy

> L = −(1/n) Σ [ y_i log σ(z_i) + (1 − y_i) log(1 − σ(z_i)) ],   σ(t) = 1 / (1 + e^(−t))

as a float. It must be accurate for |z| up to 10⁴, where σ(z) rounds to exactly 0 or 1.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-bce-with-logits?utm_source=github&utm_medium=repo
"""
import numpy as np

def bce_with_logits(z, y):
    # z: list of floats (logits), y: list of 0/1 labels. Return the mean loss as a float.
    pass


# ---- tests: python bce_with_logits.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[0],[1]],\"out\":0.6931471805599453},{\"args\":[[2,-1],[1,0]],\"out\":0.2200948492805977},{\"args\":[[-800],[1]],\"out\":800},{\"args\":[[800],[0]],\"out\":800},{\"args\":[[800],[1]],\"out\":0},{\"args\":[[-10000,10000,0.5],[0,1,0]],\"out\":0.32469232806003556},{\"args\":[[0.1,-0.2,3.5,-4.25,37.5,-37.5],[1,0,1,1,0,1]],\"out\":13.422741567443216},{\"args\":[[1e-9,-1e-9],[1,1]],\"out\":0.6931471805599454}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(bce_with_logits(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
