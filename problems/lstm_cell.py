"""One step of an LSTM cell

Compute one step of an LSTM. Inputs: `x` (length d), `h` and `c` (length H), `W` (4H × (d + H)) and `b` (length 4H). With z = W·[x, h] + b, the four blocks of H rows are, in order, the input gate i, the forget gate f, the candidate g and the output gate o:

        - i = σ(z₀), f = σ(z₁), g = tanh(z₂), o = σ(z₃);

        - c′ = f ⊙ c + i ⊙ g and h′ = o ⊙ tanh(c′).

Return `[h′, c′]`. Pre-activations can be large in either direction; the sigmoid must not overflow.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-lstm-cell?utm_source=github&utm_medium=repo
"""
import numpy as np

def lstm_cell(x, h, c, W, b):
    # Return [h_new, c_new] as lists.
    pass


# ---- tests: python lstm_cell.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[1],[0],[0],[[0.5,0],[0.5,0],[1,0],[1,0]],[0,0,0,0]],\"out\":[[0.3227440813830522],[0.47406138896346633]]},{\"args\":[[0],[1],[2],[[0,0],[0,50],[0,0],[0,50]],[-50,0,0,0]],\"out\":[[0.9640275800758169],[2]]},{\"args\":[[1,-1],[0.5],[-0.5],[[1,2,3],[0,-1,1],[2,0,-2],[-1,1,0]],[0.1,-0.2,0.3,0]],\"out\":[[0.019313146337673198],[0.16345950051017266]]},{\"args\":[[800],[0],[1],[[1,0],[-1,0],[1,0],[1,0]],[0,0,0,0]],\"out\":[[0.7615941559557649],[1]]},{\"args\":[[-800],[0],[1],[[1,0],[-1,0],[1,0],[1,0]],[0,0,0,0]],\"out\":[[0],[1]]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(lstm_cell(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
