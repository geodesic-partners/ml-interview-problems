"""Expected calibration error

Given predicted probabilities `p` (each in [0, 1]), binary labels `y` and a number of bins `m`, put each prediction in bin b = min(⌊p·m⌋, m − 1), so the bins are [0, 1/m), [1/m, 2/m), …, with the last bin also containing 1. For every non-empty bin, take the absolute difference between the mean label and the mean prediction, weight it by the bin's share of all predictions, and return the sum.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-expected-calibration-error?utm_source=github&utm_medium=repo
"""
import numpy as np

def expected_calibration_error(p, y, m):
    # Return ECE as a float.
    pass


# ---- tests: python expected_calibration_error.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[0.1,0.2,0.8,0.9],[0,0,1,1],2],\"out\":0.14999999999999997},{\"args\":[[1,1,0],[1,1,0],10],\"out\":0},{\"args\":[[0.55,0.55,0.55,0.55],[1,0,0,0],5],\"out\":0.30000000000000004},{\"args\":[[0,0.25,0.5,0.75,1],[0,1,0,1,1],4],\"out\":0.3},{\"args\":[[0.3],[1],1],\"out\":0.7}]")

    def close(a, b, rt=1e-12, at=1e-14):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(expected_calibration_error(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
