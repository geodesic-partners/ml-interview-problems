"""Beam search decoding

Tokens are 0 to V − 1. `start[v]` is the log-probability that a sequence begins with v, and `trans[a][b]` the log-probability that b follows a. A sequence's score is the sum of its start term and its transition terms. Run beam search of width `B` for sequences of length `T`:

        - Keep the B best one-token sequences.

        - Repeat T − 1 times: extend every kept sequence by every token and keep the B best extensions.

"Best" means highest score; among equal scores, the lexicographically smaller sequence wins. Return `[sequence, score]` for the best sequence after the last step.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-beam-search?utm_source=github&utm_medium=repo
"""
import numpy as np

def beam_search(start, trans, T, B):
    # Return [best_sequence, best_score].
    pass


# ---- tests: python beam_search.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[-0.1,-2.5],[[-2.5,-0.1],[-0.1,-2.5]],3,1],\"out\":[[0,1,0],-0.30000000000000004]},{\"args\":[[-0.7,-0.7],[[-0.7,-0.7],[-0.7,-0.7]],2,2],\"out\":[[0,0],-1.4]},{\"args\":[[-0.5,-1,-2],[[-3,-0.2,-2],[-0.1,-3,-3],[-0.01,-0.02,-0.03]],4,2],\"out\":[[0,1,0,1],-1]},{\"args\":[[-0.2,-0.3],[[-5,-0.01],[-0.01,-5]],5,4],\"out\":[[0,1,0,1,0],-0.24000000000000005]},{\"args\":[[-1],[[-1]],6,3],\"out\":[[0,0,0,0,0,0],-6]}]")

    def close(a, b, rt=1e-12, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(beam_search(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
