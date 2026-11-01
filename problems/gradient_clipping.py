"""Gradient clipping by global norm

Given a list `grads` of gradient arrays (any shapes) and `max_norm`, compute the total norm N = √(Σ over every element of every array of g²). If max_norm / (N + 10⁻⁶) < 1, multiply every array by that factor; otherwise leave them unchanged. Return `[clipped, N]`.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-gradient-clipping?utm_source=github&utm_medium=repo
"""
import numpy as np

def clip_grad_norm(grads, max_norm):
    # grads: list of arrays of any shape. Return [clipped_grads, total_norm].
    pass


# ---- tests: python gradient_clipping.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[3],[4]],1],\"out\":[[[0.599999880000024],[0.799999840000032]],5]},{\"args\":[[[0.1,0.2],[[0.1],[0.2]]],10],\"out\":[[[0.1,0.2],[[0.1],[0.2]]],0.31622776601683794]},{\"args\":[[[[1,1],[1,1]],[2,2,2,2]],3],\"out\":[[[[0.6708202432499704,0.6708202432499704],[0.6708202432499704,0.6708202432499704]],[1.3416404864999407,1.3416404864999407,1.3416404864999407,1.3416404864999407]],4.47213595499958]},{\"args\":[[[[0.5,-2,1.5]],[[0]],[3,-3]],2.5],\"out\":[[[[0.2525380851176548,-1.0101523404706192,0.7576142553529643]],[[0]],[1.5152285107059287,-1.5152285107059287]],4.949747468305833]},{\"args\":[[[1e-9]],1],\"out\":[[[1e-9]],1e-9]}]")

    def close(a, b, rt=1e-12, at=1e-15):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(clip_grad_norm(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
