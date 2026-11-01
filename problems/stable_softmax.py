"""Softmax that never overflows

Given a list of real numbers `x` (logits), return the list of softmax probabilities `p`, where p_i = e^(x_i) / Σ_j e^(x_j). Your function must return finite, correct values for any finite input, including logits in the thousands.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-stable-softmax?utm_source=github&utm_medium=repo
"""
import numpy as np

def softmax(x):
    # x: list of floats. Return a list (or 1-D array) of probabilities.
    pass


# ---- tests: python stable_softmax.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[1,2,3]],\"out\":[0.09003057317038046,0.24472847105479764,0.6652409557748218]},{\"args\":[[0,0,0,0]],\"out\":[0.25,0.25,0.25,0.25]},{\"args\":[[1000,1001,1002]],\"out\":[0.09003057317038046,0.24472847105479764,0.6652409557748218]},{\"args\":[[-1000,-1000.5]],\"out\":[0.6224593312018546,0.37754066879814546]},{\"args\":[[5]],\"out\":[1]},{\"args\":[[3.2,-1.1,0,7.5,2.2]],\"out\":[0.013311678975991446,0.00018062030173721444,0.0005426133737138924,0.981067994325816,0.00489709302274137]},{\"args\":[[-10000,0,10000]],\"out\":[0,0,1]},{\"args\":[[709,710,711]],\"out\":[0.09003057317038046,0.24472847105479764,0.6652409557748218]},{\"args\":[[-745,-746,-800]],\"out\":[0.7310585786300049,0.2689414213699951,9.500701493799415e-25]},{\"args\":[[0.1,0.2,0.30000000000000004]],\"out\":[0.3006096053557273,0.3322249935333472,0.3671654011109255]}]")

    def close(a, b, rt=1e-9, at=1e-300):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(softmax(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
