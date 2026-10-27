"""
An adversary that knows the defense.

The update of Theorem 1 is designed against a stated adversary. A recipient who
knows the released weights were produced this way can pick a different one. In
the preconditioned model of Corollary 2 the adversary's freedom is the diagonal
P, so we let it choose P in a box [1/C, C], restricted to the stable range
eta p_i h_i <= 1, to minimise the margin the defender designed for.

Writing H = sum_i h_i u_i u_i' and a = sum_i alpha_i u_i, a diagonal P in that
basis gives

    sigma_k(v) = sum_i alpha_i (1 - eta p_i h_i)^k v_i,

which is separable and, on the stable range, decreasing in every p_i for which
alpha_i v_i > 0. The defender's update has v_i proportional to
alpha_i (1-eta h_i)^k / h_i, so alpha_i v_i >= 0 for every i and the adversary's
best response is the coordinate-wise upper bound. We verify that against random
search rather than assuming it, then ask what the defender should have done:
propagate with the fastest preconditioner the class allows, which is
Corollary 2 applied at the worst case rather than at P = I.
"""
import json
import os
import numpy as np

OUT_DIR = os.environ.get("OUT_DIR", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "adaptive_adversary.json")
RNG = np.random.default_rng(7)
CS = [1.0, 2.0, 5.0, 10.0]
NINST, K, D = 60, 8, 1.0


def instance(d=12):
    h = np.exp(RNG.uniform(np.log(1e-2), np.log(1.0), size=d))
    al = RNG.standard_normal(d)
    al /= np.linalg.norm(al)
    eta = 0.5 / h.max()
    return h, al, eta


def margin(v, h, al, eta, p):
    return float(np.sum(al * (1.0 - eta * p * h) ** K * v))


def design(h, al, eta, pdef):
    """Theorem 1 in the eigenbasis, propagating with preconditioner pdef."""
    b = al * (1.0 - eta * pdef * h) ** K
    S = float(np.sum(b * b / h))
    return (D / S) * b / h, S


rows = []
for C in CS:
    naive, robust, costs_n, costs_r, search_gap = [], [], [], [], []
    for _ in range(NINST):
        h, al, eta = instance()
        hi = np.minimum(C, 1.0 / (eta * h))          # stable upper bound on p_i
        lo = np.full_like(h, 1.0 / C)

        v_n, S_n = design(h, al, eta, np.ones_like(h))
        v_r, S_r = design(h, al, eta, hi)

        # adversary's best response, claimed and searched
        m_n = margin(v_n, h, al, eta, hi)
        best = m_n
        for _ in range(400):
            p = lo + (hi - lo) * RNG.random(len(h))
            best = min(best, margin(v_n, h, al, eta, p))
        search_gap.append(m_n - best)

        naive.append(m_n / D)
        robust.append(margin(v_r, h, al, eta, hi) / D)
        costs_n.append(D * D / (2 * S_n))
        costs_r.append(D * D / (2 * S_r))

    rows.append(dict(C=C,
                     margin_naive_median=float(np.median(naive)),
                     margin_naive_worst=float(np.min(naive)),
                     margin_robust_median=float(np.median(robust)),
                     cost_ratio_median=float(np.median(np.array(costs_r) / np.array(costs_n))),
                     search_gap_max=float(np.max(search_gap))))

json.dump(dict(instances=NINST, k=K, rows=rows), open(OUT, "w"), indent=2)
print(f"{'C':>6} {'naive med':>10} {'naive worst':>12} {'robust med':>11} "
      f"{'cost x':>8} {'search gap':>11}")
for r in rows:
    print(f"{r['C']:>6} {r['margin_naive_median']:>10.3f} {r['margin_naive_worst']:>12.3f} "
          f"{r['margin_robust_median']:>11.3f} {r['cost_ratio_median']:>8.2f} "
          f"{r['search_gap_max']:>11.2e}")
