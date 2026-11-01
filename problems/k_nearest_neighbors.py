"""k-nearest neighbours classifier

Given training points `X_train` (n × d) with integer labels `y_train`, predict a label for each row of `X_test` using Euclidean distance and the `k` nearest training points.

        - Training points at equal distance are ordered by their index in X_train, lower first.

        - The prediction is the label with the most votes among the k. If several labels tie, choose the one whose nearest member comes first in that order.

Coordinates are integers, so equal distances are exactly equal.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-k-nearest-neighbors?utm_source=github&utm_medium=repo
"""
import numpy as np

def knn_predict(X_train, y_train, X_test, k):
    # Return a list of predicted labels, one per row of X_test.
    pass


# ---- tests: python k_nearest_neighbors.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[0,0],[1,0],[5,5],[6,5]],[0,0,1,1],[[0.5,0],[5.5,5]],1],\"out\":[0,1]},{\"args\":[[[0,0],[2,0],[4,0]],[1,2,2],[[1,0]],3],\"out\":[2]},{\"args\":[[[0,0],[2,0],[10,0]],[7,3,3],[[1,0]],2],\"out\":[7]},{\"args\":[[[1,1],[1,-1],[-1,1],[-1,-1],[3,3]],[0,1,2,3,0],[[0,0],[2,2],[-2,0]],4],\"out\":[0,0,2]},{\"args\":[[[0],[1],[2],[3],[4],[5]],[1,1,2,2,3,3],[[2.5],[0],[10]],3],\"out\":[2,1,3]}]")

    def close(a, b, rt=0, at=0):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(knn_predict(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
