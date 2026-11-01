"""The Viterbi algorithm for a hidden Markov model

A hidden Markov model has S states with initial probabilities `start` (length S), transition matrix `trans` (S × S, row i gives the probabilities of moving from i) and emission matrix `emit` (S × V, row i gives the probabilities of each observation from state i). Given a sequence of observation indices `obs`, return `[path, logp]`: the most likely sequence of hidden states and the natural log of its joint probability with the observations.

All probabilities are positive, and the best path is unique in the tests.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-viterbi-algorithm?utm_source=github&utm_medium=repo
"""
import numpy as np

def viterbi(start, trans, emit, obs):
    # Return [path (list of states), logp (float)].
    pass


# ---- tests: python viterbi_algorithm.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[0.6,0.4],[[0.7,0.3],[0.4,0.6]],[[0.5,0.4,0.1],[0.1,0.3,0.6]],[0,1,2]],\"out\":[[0,0,1],-4.19173690823075]},{\"args\":[[0.5,0.5],[[0.9,0.1],[0.1,0.9]],[[0.9,0.1],[0.2,0.8]],[0,0,1,1,1]],\"out\":[[0,0,1,1,1],-4.191965505785752]},{\"args\":[[0.3,0.7],[[0.5,0.5],[0.5,0.5]],[[0.8,0.2],[0.3,0.7]],[0]],\"out\":[[0],-1.4271163556401458]},{\"args\":[[0.2,0.3,0.5],[[0.6,0.3,0.1],[0.2,0.5,0.3],[0.1,0.2,0.7]],[[0.7,0.2,0.1],[0.1,0.6,0.3],[0.2,0.2,0.6]],[0,0,1,2,2,1,0]],\"out\":[[0,0,1,2,2,2,2],-11.062936560711798]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(viterbi(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
