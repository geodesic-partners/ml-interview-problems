"""Cross-entropy loss and its gradient from logits

You are given a batch of logits `logits` of shape (N, C) and integer class labels `labels` of length N, each in 0…C−1. Return a list `[loss, grad]` where `loss` is the mean over the batch of −log softmax(logits_n)_(labelsn), and `grad` is the N × C matrix ∂loss/∂logits.

The function must stay finite for logits of any size.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-cross-entropy-from-logits?utm_source=github&utm_medium=repo
"""
import numpy as np

def cross_entropy(logits, labels):
    # logits: N x C list of lists, labels: list of N ints.
    # Return [loss, grad] with loss a float and grad an N x C list.
    pass


# ---- tests: python cross_entropy_from_logits.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[2,1,0]],[0]],\"out\":[0.4076059644443804,[[-0.3347590442251782,0.24472847105479764,0.09003057317038043]]]},{\"args\":[[[0,0],[0,0]],[0,1]],\"out\":[0.6931471805599453,[[-0.25,0.25],[0.25,-0.25]]]},{\"args\":[[[1000,0]],[1]],\"out\":[1000,[[1,-1]]]},{\"args\":[[[1,2,3],[3,2,1],[0,0,5]],[2,2,2]],\"out\":[0.9428659435367367,[[0.030010191056793478,0.08157615701826589,-0.1115863480750594],[0.22174698525827394,0.08157615701826589,-0.30332314227653984],[0.002216118159622001,0.002216118159622001,-0.004432236319244017]]]},{\"args\":[[[-500,500,0,1],[10,10,10,10]],[0,3]],\"out\":[500.69314718055995,[[-0.5,0.5,3.562288203370643e-218,9.683303290956438e-218],[0.125,0.125,0.125,-0.375]]]},{\"args\":[[[0.5,-1.5,2.25,0,-0.75]],[3]],\"out\":[2.5519384522981765,[[0.12848559882884594,0.01738863490932766,0.7393835708486755,-0.9220695449787673,0.03681174039191839]]]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(cross_entropy(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
