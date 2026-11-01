"""NDCG at k

`rels` lists the relevance grades (non-negative integers) of the items a system returned, in the order it returned them. Compute NDCG@k:

        - DCG@k = Σ over the first k positions i = 0, 1, … of (2^(rel_i) − 1) / log₂(i + 2).

        - IDCG@k is DCG@k of the same grades sorted from highest to lowest.

        - Return DCG@k / IDCG@k, or 0.0 if IDCG@k is 0. If k exceeds the list length, use the whole list.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-ndcg-at-k?utm_source=github&utm_medium=repo
"""
import numpy as np

def ndcg_at_k(rels, k):
    # Return NDCG@k as a float.
    pass


# ---- tests: python ndcg_at_k.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[3,2,3,0,1,2],6],\"out\":0.9488107485678985},{\"args\":[[0,0,0],3],\"out\":0},{\"args\":[[1,0,0],1],\"out\":1},{\"args\":[[0,0,1],3],\"out\":0.5},{\"args\":[[2,1],10],\"out\":1},{\"args\":[[0,3,2,1],2],\"out\":0.4966392596877323}]")

    def close(a, b, rt=1e-12, at=1e-15):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(ndcg_at_k(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
