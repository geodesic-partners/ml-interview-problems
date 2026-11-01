"""A one-dimensional Kalman filter

A level x_t follows a random walk: x_t = x_(t−1) + w_t with w_t ~ N(0, q). You observe y_t = x_t + v_t with v_t ~ N(0, r), all noises independent. Before the first observation your belief about x_1 is N(m0, p0).

Process `ys` in order. After each observation, record `[mean, variance]` of the level given everything seen so far. There is no prediction step before the first observation. Return the records.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-kalman-filter-1d?utm_source=github&utm_medium=repo
"""
def kalman_filter(ys, q, r, m0, p0):
    # Return [[mean, variance], ...], one record after each observation.
    pass


# ---- tests: python kalman_filter_1d.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[10,12,11],1,4,8,9],\"out\":[[9.384615384615385,2.769230769230769],[10.653465346534654,1.9405940594059405],[10.800285306704708,1.6947218259629102]]},{\"args\":[[5,5,5,5],0,1,0,100],\"out\":[[4.9504950495049505,0.990099009900991],[4.975124378109452,0.49751243781094556],[4.983388704318936,0.3322259136212626],[4.987531172069825,0.24937655860349137]]},{\"args\":[[3],2,1,0,1],\"out\":[[1.5,0.5]]},{\"args\":[[1.5,2.25,1.75,3,2.5,4],0.5,2,0,10],\"out\":[[1.25,1.6666666666666663],[1.77,1.0399999999999998],[1.7612994350282487,0.8700564971751413],[2.2648784576697403,0.8130762782900252],[2.358064516129032,0.7926628716002532],[3.0026701500278543,0.7851777858885454]]},{\"args\":[[100,101,99,102],0.1,0.5,100,0.25],\"out\":[[100,0.16666666666666669],[100.34782608695652,0.1739130434782609],[99.87078651685393,0.17696629213483145],[100.6297903109183,0.17823571945047]]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(kalman_filter(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
