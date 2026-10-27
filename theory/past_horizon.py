"""
Margin of the optimal update past its design horizon.

For t >= k, sigma_t(v*) = D * sum_i w_i (1 - eta h_i)^(t-k) with weights
w_i = alpha_i^2 (1 - eta h_i)^(2k) / (h_i S_k) summing to one, so by Jensen
sigma_t(v*) >= D (1 - eta hbar_k)^(t-k), hbar_k = ||A^k a||^2 / S_k.

We check the identity and the bound on random instances, and compare the
margin the optimal update keeps at t = 60 with that of an update of the same
retained cost built from the readout gradient alone (v proportional to a,
the direction a first-order unlearning step takes).
"""
import json
import os
import numpy as np

RNG = np.random.default_rng(60)
OUT_DIR = os.environ.get("OUT_DIR", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "past_horizon.json")
T = 60
rows = []
viol, maxerr = 0, 0.0
for _ in range(500):
    d = int(RNG.integers(8, 40))
    h = np.exp(RNG.uniform(np.log(1e-3), 0.0, size=d))
    eta = 0.5 / h.max()
    al = RNG.standard_normal(d)
    al /= np.linalg.norm(al)
    k = int(RNG.integers(0, 17))
    D = 1.0
    b = al * (1 - eta * h) ** k
    S = float(np.sum(b * b / h))
    v = (D / S) * b / h
    U = 0.5 * float(np.sum(h * v * v))
    hbar = float(np.sum(b * b)) / S
    for t in range(k, k + 200):
        sig = float(np.sum(al * (1 - eta * h) ** t * v))
        w = al ** 2 * (1 - eta * h) ** (2 * k) / (h * S)
        ident = D * float(np.sum(w * (1 - eta * h) ** (t - k)))
        maxerr = max(maxerr, abs(sig - ident))
        if sig < D * (1 - eta * hbar) ** (t - k) - 1e-12:
            viol += 1
    # same-cost gradient-direction update
    g = al.copy()
    g *= np.sqrt(2 * U / float(np.sum(h * g * g)))
    sig_opt = float(np.sum(al * (1 - eta * h) ** T * v))
    sig_grad = float(np.sum(al * (1 - eta * h) ** T * g))
    sig_grad0 = float(np.sum(al * g))
    rows.append(dict(k=k, hbar_over_hmax=hbar / h.max(), opt_T=sig_opt, grad_T=sig_grad,
                     grad_0=sig_grad0, bound_T=(1 - eta * hbar) ** max(T - k, 0)))

r = dict(instances=len(rows), identity_max_abs_err=maxerr, bound_violations=viol,
         median_opt_margin_at_60=float(np.median([x["opt_T"] for x in rows])),
         median_grad_margin_at_60=float(np.median([x["grad_T"] for x in rows])),
         frac_opt_beats_grad_at_60=float(np.mean([x["opt_T"] > x["grad_T"] for x in rows])),
         median_hbar_over_hmax=float(np.median([x["hbar_over_hmax"] for x in rows])))
json.dump(r, open(OUT, "w"), indent=2)
for k_, v_ in r.items():
    print(f"{k_:32s} {v_}")
