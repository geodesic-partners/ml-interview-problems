"""2-D convolution with stride and padding

Given a single-channel `image` (H × W) and a `kernel` (k_h × k_w), pad the image with `padding` zeros on every side, then slide the kernel with step `stride` in both directions. Each output entry is the sum of the elementwise product of the kernel with the window under it (no kernel flip). Return the output, of size ⌊(H + 2p − k_h)/s⌋ + 1 by ⌊(W + 2p − k_w)/s⌋ + 1.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-conv2d?utm_source=github&utm_medium=repo
"""
import numpy as np

def conv2d(image, kernel, stride, padding):
    # Cross-correlation with zero padding and stride. Return a 2-D list or array.
    pass


# ---- tests: python conv2d.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,2,3],[4,5,6],[7,8,9]],[[1,0],[0,-1]],1,0],\"out\":[[-4,-4],[-4,-4]]},{\"args\":[[[1,2],[3,4]],[[1,1,1],[1,1,1],[1,1,1]],1,1],\"out\":[[10,10],[10,10]]},{\"args\":[[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]],[[1,1],[1,1]],2,0],\"out\":[[14,22],[46,54]]},{\"args\":[[[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]],[[0,1,0],[1,-4,1],[0,1,0]],2,1],\"out\":[[4,2,-6],[-10,0,-16]]},{\"args\":[[[3]],[[2]],1,2],\"out\":[[0,0,0,0,0],[0,0,0,0,0],[0,0,6,0,0],[0,0,0,0,0],[0,0,0,0,0]]},{\"args\":[[[1,2,3]],[[1,-1]],3,0],\"out\":[[-1]]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(conv2d(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
