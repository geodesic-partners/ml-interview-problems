"""Top-k search by cosine similarity

Given a query `q` (length d, not all zero) and a matrix `M` (n × d), return the indices of the `k` rows with the highest cosine similarity to q, most similar first.

        - A row that is entirely zero has similarity 0.

        - Similarities are compared after rounding to 12 decimal places; equal values are ordered by row index, lower first.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-cosine-top-k?utm_source=github&utm_medium=repo
"""
import numpy as np

def cosine_top_k(q, M, k):
    # Return a list of k row indices, most similar first.
    pass


# ---- tests: python cosine_top_k.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[1,0],[[1,0],[0,1],[1,1],[-1,0]],2],\"out\":[0,2]},{\"args\":[[1,1],[[2,2],[1,1],[0,0],[3,3]],3],\"out\":[0,1,3]},{\"args\":[[0,1],[[0,0],[0,0],[1,0]],3],\"out\":[0,1,2]},{\"args\":[[1,2,3],[[3,2,1],[1,2,3],[-1,-2,-3],[2,4,6],[1,0,0]],5],\"out\":[1,3,0,4,2]},{\"args\":[[-1,1],[[1,-1],[-2,2],[0,5],[5,0]],1],\"out\":[1]}]")

    def close(a, b, rt=0, at=0):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(cosine_top_k(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
