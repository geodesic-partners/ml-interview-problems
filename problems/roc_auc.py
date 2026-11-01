"""ROC AUC from scores, with ties

Given real `scores` and binary `labels` (1 positive, 0 negative), return the ROC AUC: the probability that a randomly chosen positive has a higher score than a randomly chosen negative, with ties counting one half. There is at least one positive and one negative. Aim for O(n log n).

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-roc-auc?utm_source=github&utm_medium=repo
"""
import numpy as np

def roc_auc(scores, labels):
    # Return the AUC as a float.
    pass


# ---- tests: python roc_auc.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[0.9,0.8,0.3,0.1],[1,1,0,0]],\"out\":1},{\"args\":[[0.5,0.5,0.5,0.5],[1,0,1,0]],\"out\":0.5},{\"args\":[[0.1,0.4,0.35,0.8],[0,0,1,1]],\"out\":0.75},{\"args\":[[0.1,0.9],[1,0]],\"out\":0},{\"args\":[[3,1,2,2,3,1,2],[1,0,1,0,0,1,1]],\"out\":0.5},{\"args\":[[0.2,0.2,0.7,0.7,0.7,0.1],[0,1,1,0,1,0]],\"out\":0.7222222222222222}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(roc_auc(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
