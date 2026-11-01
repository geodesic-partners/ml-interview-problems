"""Backpropagation through a two-layer network

For the network A = XW1 + b1, H = max(0, A), Z = HW2 + b2 and the loss L = −(1/N) Σ_n log softmax(Z_n)_(yn), return `[dW1, db1, dW2, db2]`, the gradients of L with respect to each parameter, each with the same shape as the parameter. `y` holds integer labels in 0…C−1.

The tests are generated so that no hidden pre-activation is exactly zero.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-mlp-backpropagation?utm_source=github&utm_medium=repo
"""
import numpy as np

def mlp_grads(X, y, W1, b1, W2, b2):
    # Return [dW1, db1, dW2, db2], each with its parameter's shape.
    pass


# ---- tests: python mlp_backpropagation.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,2]],[0],[[1,0],[0,1]],[0,0],[[1,0],[0,1]],[0,0]],\"out\":[[[-0.7310585786300049,0.7310585786300049],[-1.4621171572600098,1.4621171572600098]],[-0.7310585786300049,0.7310585786300049],[[-0.7310585786300049,0.7310585786300049],[-1.4621171572600098,1.4621171572600098]],[-0.7310585786300049,0.7310585786300049]]},{\"args\":[[[-1,-1]],[1],[[1,0],[0,1]],[0,0],[[1,0],[0,1]],[0,0]],\"out\":[[[0,0],[0,0]],[0,0],[[0,0],[0,0]],[0.5,-0.5]]},{\"args\":[[[1,0],[0,1]],[0,1],[[2,-1],[-1,2]],[0.5,0.5],[[1,-1],[-1,1]],[0,0.1]],\"out\":[[[-0.0073915413442819265,0],[0,-0.006059801491584182]],[-0.0073915413442819265,-0.006059801491584182],[[-0.009239426680352353,0.009239426680352464],[0.007574751864480145,-0.007574751864480311]],[-0.0006658699263488834,0.0006658699263488608]]},{\"args\":[[[0.5,-0.2,1],[2,0.3,-1.1],[-0.4,0.9,0]],[2,0,1],[[0.1,-0.3,0.8,0.2],[0.7,0.5,-0.6,0],[-0.2,0.4,0.3,1.1]],[0,0.1,-0.1,0.2],[[1,-0.5,0.2],[0.3,0.8,-1],[-0.7,0.1,0.5],[0.4,0.4,0.4]],[0,0,0.3]],\"out\":[[[-0.6247321792517738,0.2325036061699476,0.51030934722927,-4.869546326528957e-18],[0.19599288328044914,-0.23802320475858532,0.11105253929010789,1.2217381508394281e-17],[0.2772124941749263,0.30822692878248303,-0.44065314074806106,1.3631375931936526e-18]],[0.049762298084382095,0.11225157433571781,0.16104702668995116,1.524092540100811e-17],[[-0.06174138995785027,-0.09810344786061707,0.15984483781846734],[0.1313912970673877,-0.12503080081739776,-0.006360496249989934],[-0.21217511745079393,0.1532370790625372,0.05893803838825672],[0.10829010821563076,0.14845750948974692,-0.25674761770537763]],[-0.024491465471972923,-0.04448637735244845,0.06897784282442138]]}]")

    def close(a, b, rt=0.00001, at=0.000001):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(mlp_grads(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
