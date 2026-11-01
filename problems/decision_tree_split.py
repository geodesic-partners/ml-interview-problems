"""Best decision-tree split by Gini impurity

Given features `X` (n × d) and class labels `y`, find the split "x_f ≤ t goes left" with the lowest weighted Gini impurity (n_LG_L + n_RG_R)/n, where G = 1 − Σ_c p_c².

        - Candidate thresholds for a feature are the midpoints between consecutive distinct values of that feature.

        - Treat impurities within 10⁻¹² as equal. On a tie, prefer the lower feature index, then the lower threshold.

        - Return `[feature, threshold, impurity]`. If no feature has two distinct values, return `[-1, 0.0, G]` with G the impurity of the whole set.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-decision-tree-split?utm_source=github&utm_medium=repo
"""
import numpy as np

def best_split(X, y):
    # Return [feature, threshold, weighted_gini].
    pass


# ---- tests: python decision_tree_split.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1],[2],[3],[4]],[0,0,1,1]],\"out\":[0,2.5,0]},{\"args\":[[[1,5],[2,5],[3,6],[4,6]],[0,1,0,1]],\"out\":[0,1.5,0.3333333333333333]},{\"args\":[[[7],[7],[7]],[0,1,1]],\"out\":[-1,0,0.4444444444444444]},{\"args\":[[[2,3],[1,1],[3,2],[2,4],[5,1],[4,4]],[0,0,1,0,1,1]],\"out\":[0,2.5,0]},{\"args\":[[[1,1],[1,2],[2,1],[2,2],[3,3],[3,3]],[2,2,0,1,1,1]],\"out\":[0,1.5,0.25]}]")

    def close(a, b, rt=1e-9, at=1e-9):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(best_split(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
