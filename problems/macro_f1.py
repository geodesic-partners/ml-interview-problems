"""Macro-averaged F1 across classes

Given true labels `y_true` and predictions `y_pred` (integers), let the classes be every label that appears in either list. For each class c, F1 = 2TP / (2TP + FP + FN), counting TP, FP and FN for c against all other classes. Return the unweighted mean of the per-class F1 scores.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-macro-f1?utm_source=github&utm_medium=repo
"""
import numpy as np

def macro_f1(y_true, y_pred):
    # Return macro-F1 as a float.
    pass


# ---- tests: python macro_f1.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[0,0,1,1,2],[0,1,1,1,0]],\"out\":0.43333333333333335},{\"args\":[[5,5,5],[5,5,5]],\"out\":1},{\"args\":[[0,1],[1,0]],\"out\":0},{\"args\":[[1,2,3,4],[1,2,3,9]],\"out\":0.6},{\"args\":[[0,0,0,1],[0,0,0,0]],\"out\":0.42857142857142855}]")

    def close(a, b, rt=1e-12, at=1e-15):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(macro_f1(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
