"""Regression coefficients with standard errors and t-statistics

Given a design matrix `X` (n × p, full column rank, n > p; include a column of ones yourself if you want an intercept) and targets `y`, fit ordinary least squares and return `[beta, se, t]`: the p coefficients, their standard errors under homoskedastic errors, and the t-statistics beta / se.

Estimate the error variance as RSS / (n − p). The fit is never exact in the tests.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-ols-standard-errors?utm_source=github&utm_medium=repo
"""
import numpy as np
def ols_inference(X, y):
    # Return [beta, se, t], each a list of p floats.
    pass


# ---- tests: python ols_standard_errors.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,0],[1,1],[1,2],[1,3]],[1,3,2,5]],\"out\":[[1.0999999999999999,1.1],[0.9721111047611788,0.519615242270663],[1.1315578997220075,2.1169509870286287]]},{\"args\":[[[1,1],[1,2],[1,3],[1,4],[1,5]],[2.1,3.9,6.2,7.8,10.1]],\"out\":[[0.04999999999999786,1.9900000000000007],[0.19807406022327423,0.05972157622389635],[0.25243083290985485,33.3212906595011]]},{\"args\":[[[1],[1],[1]],[2,4,9]],\"out\":[[5],[2.081665999466133],[2.4019223070763065]]},{\"args\":[[[1,0.5,-1.2],[1,1.5,0.3],[1,-0.7,2.2],[1,2.4,0.9],[1,0.1,-0.4],[1,-1.3,1.1],[1,0.8,0]],[1.2,3.1,0.4,4.4,0.9,-0.5,2]],\"out\":[[0.8900813150832596,1.3697964696164915,0.25831118782164186],[0.04102469438289523,0.03037730749271544,0.034601258779443415],[21.696232683073166,45.09275451568485,7.465369669588563]]},{\"args\":[[[1,100],[1,100.1],[1,100.2],[1,100.3],[1,100.4]],[5,5.3,5.1,5.6,5.7]],\"out\":[[-164.9999999999872,1.6999999999998723],[52.70436351321751,0.5259911279353204],[-3.130670574526672,3.2319936776745712]]}]")

    def close(a, b, rt=1e-7, at=1e-9):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(ols_inference(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
