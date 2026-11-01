"""k-means clustering (Lloyd's algorithm)

Run k-means on `X` (n × d) from the starting centroids `init` (k × d):

        - Assign every point to its nearest centroid (squared Euclidean distance; ties go to the lower centroid index).

        - Repeat up to `max_iter` times: move each centroid to the mean of its assigned points (a centroid with no points stays where it is); reassign every point; stop early if no assignment changed.

Return `[centroids, labels]`, where the labels are the assignment to the returned centroids.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-k-means-clustering?utm_source=github&utm_medium=repo
"""
import numpy as np

def kmeans(X, init, max_iter):
    # Return [centroids (k x d), labels (n)].
    pass


# ---- tests: python k_means_clustering.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[0,0],[0,1],[10,10],[10,11]],[[0,0],[10,10]],10],\"out\":[[[0,0.5],[10,10.5]],[0,0,1,1]]},{\"args\":[[[0,0],[1,0],[2,0]],[[0,0],[100,100]],10],\"out\":[[[1,0],[100,100]],[0,0,0]]},{\"args\":[[[0,0],[4,0]],[[1,0],[3,0]],0],\"out\":[[[1,0],[3,0]],[0,1]]},{\"args\":[[[1,1],[1,2],[2,1],[8,8],[8,9],[9,8],[1,8],[2,9]],[[0,0],[5,5],[10,10]],50],\"out\":[[[1.3333333333333333,1.3333333333333333],[1.5,8.5],[8.333333333333334,8.333333333333334]],[0,0,0,2,2,2,1,1]]},{\"args\":[[[0],[1],[2],[10],[11],[12],[20]],[[0],[1],[2]],100],\"out\":[[[0],[1.5],[13.25]],[0,1,1,2,2,2,2]]}]")

    def close(a, b, rt=1e-9, at=1e-9):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(kmeans(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
