"""Inverted dropout, forward and backward

Given activations `X`, a 0/1 `mask` of the same shape (1 keeps a unit), the drop probability `p` and the upstream gradient `dout`, return `[out, dX]` for inverted dropout in training mode: out = X ⊙ mask / (1 − p), and dX the gradient of the loss with respect to X.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-inverted-dropout?utm_source=github&utm_medium=repo
"""
import numpy as np

def dropout(X, mask, p, dout):
    # Return [out, dX].
    pass


# ---- tests: python inverted_dropout.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,2],[3,4]],[[1,0],[1,1]],0.5,[[1,1],[1,1]]],\"out\":[[[2,0],[6,8]],[[2,0],[2,2]]]},{\"args\":[[[5,-5,5]],[[1,1,1]],0.2,[[0.1,0.2,0.3]]],\"out\":[[[6.25,-6.25,6.25]],[[0.125,0.25,0.375]]]},{\"args\":[[[2]],[[0]],0.9,[[7]]],\"out\":[[[0]],[[0]]]},{\"args\":[[[0.3,-1.2,4,2.2],[1.1,0,-0.5,3.3]],[[0,1,1,0],[1,1,0,1]],0.25,[[1,-1,2,0.5],[0.3,0.3,0.3,0.3]]],\"out\":[[[0,-1.5999999999999999,5.333333333333333,0],[1.4666666666666668,0,0,4.3999999999999995]],[[0,-1.3333333333333333,2.6666666666666665,0],[0.39999999999999997,0.39999999999999997,0,0.39999999999999997]]]}]")

    def close(a, b, rt=1e-12, at=1e-15):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(dropout(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
