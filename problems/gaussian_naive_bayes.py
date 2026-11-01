"""Gaussian naive Bayes classifier

Fit a Gaussian naive Bayes model on `X_train` (n × d) with integer labels `y_train`, and return the predicted label for each row of `X_test`.

        - Prior of class c: the fraction of training rows with label c.

        - For each class and feature: the mean and the biased variance (divide by the class count), plus ε = 10⁻⁹ × the largest variance of any feature over the whole training set.

        - Predict the class with the largest log prior plus Σ_j log N(x_j; μ_(cj), σ²_(cj)).

Every feature varies within every class in the tests.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-gaussian-naive-bayes?utm_source=github&utm_medium=repo
"""
import numpy as np

def gaussian_nb(X_train, y_train, X_test):
    # Return the predicted label for each row of X_test, as a list.
    pass


# ---- tests: python gaussian_naive_bayes.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[0,0],[1,1],[0,1],[5,5],[6,6],[5,6]],[0,0,0,1,1,1],[[0.5,0.5],[5.5,5.5]]],\"out\":[0,1]},{\"args\":[[[0],[2],[10],[12]],[0,0,1,1],[[6.1]]],\"out\":[1]},{\"args\":[[[0],[1],[2],[0],[20]],[0,0,0,1,1],[[1]]],\"out\":[0]},{\"args\":[[[1.2,3.4],[0.8,2.9],[1.5,3.1],[4.1,0.2],[3.8,0.5],[4.4,0.1],[2.5,2.5],[2.7,2.2]],[3,3,3,7,7,7,9,9],[[1,3],[4,0],[2.6,2.4],[2,2]]],\"out\":[3,7,9,3]}]")

    def close(a, b, rt=0, at=0):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(gaussian_nb(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
