"""Scaled dot-product attention with a causal mask

Given queries `Q` (n × d), keys `K` (m × d) and values `V` (m × d_v), return softmax(QKᵀ/√d)V as an n × d_v matrix, with the softmax over each row. If `causal` is true (then n = m), query i may only attend to keys j ≤ i.

Hints, the reference solution and a live editor:
https://geodesicpartners.co.uk/ml-scaled-dot-product-attention?utm_source=github&utm_medium=repo
"""
import numpy as np

def attention(Q, K, V, causal):
    # Q: n x d, K: m x d, V: m x dv, causal: bool. Return n x dv.
    pass


# ---- tests: python scaled_dot_product_attention.py ----
if __name__ == "__main__":
    import json
    import numpy as np
    TESTS = json.loads("[{\"args\":[[[1,0]],[[1,0],[0,1]],[[10,0],[0,10]],false],\"out\":[[6.697615493266569,3.302384506733431]]},{\"args\":[[[1,0],[0,1]],[[1,0],[0,1]],[[1,2],[3,4]],true],\"out\":[[1,2],[2.3395230986533138,3.3395230986533138]]},{\"args\":[[[0,0]],[[1,1],[2,2],[3,3]],[[1],[2],[3]],false],\"out\":[[2]]},{\"args\":[[[0.5,-1,2],[1.5,0.3,-0.2],[-0.7,0.8,0.1],[0.2,-0.4,1]],[[1,0,0.5],[-0.5,1.2,0.3],[0.8,-0.8,0],[0.1,0.1,0.9]],[[1,0],[0,1],[1,1],[-1,2]],true],\"out\":[[1,0],[0.7442350393954856,0.2557649606045144],[0.35388634691118503,0.7951970318060668],[0.21674361198287245,1.0307787634658856]]},{\"args\":[[[100,0],[0,100]],[[1,0],[0,1],[1,1]],[[1,2,3],[4,5,6],[7,8,9]],false],\"out\":[[4,5,6],[5.5,6.5,7.5]]}]")

    def close(a, b, rt=1e-9, at=1e-12):
        if isinstance(b, list):
            a = a.tolist() if hasattr(a, "tolist") else list(a)
            return len(a) == len(b) and all(close(x, y, rt, at) for x, y in zip(a, b))
        return abs(float(a) - float(b)) <= at + rt * abs(float(b))

    passed = 0
    for i, t in enumerate(TESTS):
        try:
            ok = close(attention(*t["args"]), t["out"])
        except Exception as e:
            ok = False
            print(i, "error:", e)
        passed += ok
        print(i, "pass" if ok else "FAIL")
    print(f"{passed}/{len(TESTS)} passed")
