"""The InfoNCE contrastive loss

Given queries `Q` and keys `K` (both n × d, no zero rows) and a temperature `tau` > 0: normalise every row to unit length, form the logits L_(ij) = q̂_i·k̂_j / τ, and return the mean over i of log Σ_j exp(L_(ij)) − L_(ii). Key i is the positive for query i; the other keys are its negatives.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-info-nce-loss?utm_source=github&utm_medium=repo
"""
import numpy as np

def info_nce(Q, K, tau):
    # Return the loss as a float.
    pass


# ---- tests: python info_nce_loss.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,0],[0,1]],[[1,0],[0,1]],1],\"out\":0.3132616875182228},{\"args\":[[[1,0],[0,1]],[[1,0],[0,1]],0.001],\"out\":0},{\"args\":[[[1,2],[3,4],[5,6]],[[1,2],[4,3],[-5,6]],0.1],\"out\":3.2476224441131634},{\"args\":[[[1,0,0],[0,1,0],[0,0,1]],[[0,1,0],[0,0,1],[1,0,0]],0.5],\"out\":2.2395447662218846},{\"args\":[[[2,2]],[[-1,3]],0.07],\"out\":0}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(info_nce(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
