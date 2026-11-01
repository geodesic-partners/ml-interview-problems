"""Linear warm-up with cosine decay

For each step in `steps` (0-based), return the learning rate:

        - if step < warmup: base_lr × (step + 1) / warmup;

        - otherwise, with progress = min(1, (step − warmup) / (total − warmup)): min_lr + ½(base_lr − min_lr)(1 + cos(π · progress)).

The arguments are `steps, base_lr, min_lr, warmup, total`, with 0 ≤ warmup < total.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-learning-rate-schedule?utm_source=github&utm_medium=repo
"""
import math

def lr_schedule(steps, base_lr, min_lr, warmup, total):
    # Return the learning rate for each step, as a list.
    pass


# ---- tests: python learning_rate_schedule.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[0,1,2,3],1,0,4,10],\"out\":[0.25,0.5,0.75,1]},{\"args\":[[4,7,10],1,0,4,10],\"out\":[1,0.5,0]},{\"args\":[[0,50,100,150],0.0003,0.00003,0,100],\"out\":[0.0003,0.00016499999999999997,0.00003,0.00003]},{\"args\":[[0,99,100,101,5000,10000,20000],0.001,0.00001,100,10000],\"out\":[0.00001,0.001,0.001,0.0009999999750767566,0.00051285365209823,0.00001,0.00001]}]")

    def close(a, b, rt=1e-12, at=1e-18):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(lr_schedule(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
