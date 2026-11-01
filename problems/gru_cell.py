"""One step of a GRU cell, as PyTorch computes it

Compute one GRU step in PyTorch's convention. Inputs: `x` (length d), `h` (length H), `W_ih` (3H × d), `W_hh` (3H × H), `b_ih` and `b_hh` (length 3H). The three blocks of H rows are, in order, reset r, update z and new n. With a = W_ih·x + b_ih and u = W_hh·h + b_hh:

        - r = σ(a₀ + u₀), z = σ(a₁ + u₁), n = tanh(a₂ + r ⊙ u₂);

        - h′ = (1 − z) ⊙ n + z ⊙ h.

Return h′ as a list.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-gru-cell?utm_source=github&utm_medium=repo
"""
import numpy as np

def gru_cell(x, h, W_ih, W_hh, b_ih, b_hh):
    # Return h_new as a list.
    pass


# ---- tests: python gru_cell.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[1],[0],[[1],[1],[1]],[[1],[1],[1]],[0,0,0],[0,0,0]],\"out\":[0.20482421480982513]},{\"args\":[[0],[1],[[0],[0],[0]],[[0],[50],[0]],[0,0,0],[0,0,0]],\"out\":[1]},{\"args\":[[1,2],[0.5,-0.5],[[1,0],[0,1],[1,1],[-1,0],[0,-1],[1,-1]],[[1,0],[0,1],[0.5,0.5],[1,-1],[2,0],[0,2]],[0.1,0.2,0.3,0.4,0.5,0.6],[-0.1,-0.2,-0.3,-0.4,-0.5,-0.6]],\"out\":[0.43847344928785426,-0.7182081868233642]},{\"args\":[[600],[0.3],[[1],[-1],[1]],[[0],[0],[1]],[0,0,0],[0,0,0]],\"out\":[1]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(gru_cell(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
