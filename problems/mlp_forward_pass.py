"""Forward pass of a two-layer network

Given a batch `X` (N × D) and parameters `W1` (D × H), `b1` (H), `W2` (H × C), `b2` (C), return the N × C matrix of class probabilities P = softmax(max(0, XW1 + b1)W2 + b2), with the softmax taken over each row.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-mlp-forward-pass?utm_source=github&utm_medium=repo
"""
import numpy as np

def mlp_forward(X, W1, b1, W2, b2):
    # Return the N x C matrix of probabilities.
    pass


# ---- tests: python mlp_forward_pass.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,2]],[[1,0],[0,1]],[0,0],[[1,0],[0,1]],[0,0]],\"out\":[[0.2689414213699951,0.7310585786300049]]},{\"args\":[[[-1,-1]],[[1,0],[0,1]],[0,0],[[1,0],[0,1]],[0,0]],\"out\":[[0.5,0.5]]},{\"args\":[[[1,0],[0,1]],[[2,-1],[-1,2]],[0.5,0.5],[[1,-1],[-1,1]],[0,0.1]],\"out\":[[0.9926084586557181,0.007391541344281971],[0.0060598014915841155,0.9939401985084158]]},{\"args\":[[[0.5,-0.2,1],[2,0.3,-1.1],[-0.4,0.9,0]],[[0.1,-0.3,0.8,0.2],[0.7,0.5,-0.6,0],[-0.2,0.4,0.3,1.1]],[0,0.1,-0.1,0.2],[[1,-0.5,0.2],[0.3,0.8,-1],[-0.7,0.1,0.5],[0.4,0.4,0.4]],[0,0,0.3]],\"out\":[[0.1876236239748439,0.3782055973223067,0.4341707787028495],[0.2205915539249805,0.18929616880358657,0.5901122772714329],[0.5183104256842569,0.29903910181676135,0.18265047249898184]]},{\"args\":[[[300,200]],[[1,0],[0,1]],[0,0],[[1,0],[0,1]],[0,0]],\"out\":[[1,3.720075976020836e-44]]}]")

    def close(a, b, rt=1e-9, at=1e-15):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(mlp_forward(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
