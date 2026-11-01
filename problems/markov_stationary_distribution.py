"""Stationary distribution of a Markov chain

Given a row-stochastic transition matrix `P` (n × n, each row sums to 1) of an irreducible, aperiodic Markov chain, return its stationary distribution π: the unique probability vector with πP = π.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-markov-stationary-distribution?utm_source=github&utm_medium=repo
"""
import numpy as np

def stationary(P):
    # Return the stationary distribution as a list.
    pass


# ---- tests: python markov_stationary_distribution.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[0.9,0.1],[0.5,0.5]]],\"out\":[0.8333333333333334,0.16666666666666663]},{\"args\":[[[0,1],[0.5,0.5]]],\"out\":[0.3333333333333333,0.6666666666666666]},{\"args\":[[[0.5,0.25,0.25],[0.25,0.5,0.25],[0.25,0.25,0.5]]],\"out\":[0.33333333333333337,0.3333333333333333,0.3333333333333333]},{\"args\":[[[1]]],\"out\":[1]},{\"args\":[[[0.2,0.3,0.5,0],[0.1,0.1,0.4,0.4],[0.6,0,0.2,0.2],[0.25,0.25,0.25,0.25]]],\"out\":[0.3284823284823285,0.158004158004158,0.33887733887733884,0.17463617463617467]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(stationary(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
