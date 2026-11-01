"""Principal component analysis

Given data `X` (n × d) and `k` ≤ min(n, d), return `[components, projected]`:

        - `components` (k × d): the top k principal directions of the centred data, as unit-length rows, in decreasing order of variance;

        - each row's sign fixed so that its entry of largest absolute value is positive (the tests avoid ties between entries of opposite sign);

        - `projected` (n × k): the centred data expressed in those directions.

The tests guarantee the top k variances are distinct.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-pca?utm_source=github&utm_medium=repo
"""
import numpy as np

def pca(X, k):
    # Return [components (k x d), projected (n x k)].
    pass


# ---- tests: python pca.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,1],[2,2],[3,3]],1],\"out\":[[[0.7071067811865476,0.7071067811865474]],[[-1.414213562373095],[0],[1.414213562373095]]]},{\"args\":[[[0,0],[2,0],[0,1],[2,1]],2],\"out\":[[[1,0],[0,1]],[[-1,-0.5],[1,-0.5],[-1,0.5],[1,0.5]]]},{\"args\":[[[1,-2],[-1,2],[3,-6]],1],\"out\":[[[-0.447213595499958,0.894427190999916]],[[0],[4.47213595499958],[-4.47213595499958]]]},{\"args\":[[[2.5,2.4,0.5],[0.5,0.7,1.1],[2.2,2.9,0.3],[1.9,2.2,0.9],[3.1,3,0.2],[2.3,2.7,0.8],[2,1.6,1.4],[1,1.1,1]],2],\"out\":[[[0.6660598134482155,0.6964993112222387,-0.2669326401478005],[0.5946433249560608,-0.27978653773634055,0.7537365649792455]],[[0.6744273972524941,0.036278690154180865],[-2.001900642810423,-0.22512890661861445],[0.876245636858709,-0.4327548891966566],[0.028718590879996984,0.036944628719510576],[1.572042664099107,-0.0009282070077606209],[0.6700854358851825,0.05953503333584001],[-0.4560413345824249,0.6411491663465437],[-1.3635777475826396,-0.11509551573304494]]]}]")

    def close(a, b, rt=1e-7, at=1e-9):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(pca(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
