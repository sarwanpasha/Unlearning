"""
Certified removal at non-vacuous epsilon, under global and retain sensitivity.

Same construction as certified.py (ridge regression, exact removal, Gaussian
output perturbation), scaled so that the certificate is meaningful:
eps in {1, 2, 5, 10} at delta = 1e-5. The design is axis-aligned, so the
retained Hessian is diagonal and n can be large without materialising X.

Two calibrations of the noise are compared.

  global sensitivity  gamma = 2Y(1 + 1/lambda)/(lambda n), the same for every
                      design that shares n, lambda and the data bounds;
  retain sensitivity  the largest movement of the minimiser over deletions of
                      one point z from R u {z} with the retained set R held
                      fixed (Heinzler et al.), which depends on R and therefore
                      on the curvature of the retained loss.

For each design we report both sensitivities, the epsilon each gives at a
common noise scale, the noise needed to reach a target epsilon under each, and
the recovery horizon, which does not depend on the noise at all.
"""
import json
import os
import numpy as np
from scipy.optimize import minimize

OUT_DIR = os.environ.get("OUT_DIR", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "certified_removal.json")
RNG = np.random.default_rng(20261026)

d, n, lam, Y = 20, 10_000_000, 1e-2, 1.0
delta, c = 1e-5, 0.5
P = [0.4, 0.04, 0.004, 0.0004]
NC = 2
EPS = [1.0, 2.0, 5.0, 10.0]
Z = np.sqrt(2.0 * np.log(1.25 / delta))


def design(p):
    """Counts and signs of the axis-aligned rows (row j is s_j * e_{a_j})."""
    k = max(1, int(round(p * n)))
    axis = np.empty(n, dtype=int)
    sign = np.ones(n)
    axis[:k] = 0
    j = np.arange(k, n)
    axis[k:] = 1 + (j % NC)
    sign[k:] = np.where((j // NC) % 2 == 0, 1.0, -1.0)
    return axis, sign


def stats(axis, sign, y, m):
    """Normalised Gram diagonal and moment vector of the first m rows."""
    G = np.bincount(axis[:m], minlength=d) / m
    b = np.bincount(axis[:m], weights=sign[:m] * y[:m], minlength=d) / m
    return G, b


def retain_sensitivity(GR, bR, mR):
    """max over z=(x,y), ||x||<=1, |y|<=Y of ||A(R u z) - A(R)||.

    A(D) = argmin (1/|D|) sum (x'th - y)^2/2 + lam/2 ||th||^2. With |R| = mR the
    retained problem has Gram GR (normalised by mR); adding z renormalises by
    mR + 1. Sherman-Morrison gives the new minimiser in closed form.
    """
    th_R = bR / (GR + lam)
    m1 = mR + 1
    D0 = GR * mR / m1 + lam                       # diagonal part after renorm
    b0 = bR * mR / m1

    def move(x, yv):
        Dinv = 1.0 / D0
        rhs = b0 + x * yv / m1
        t = Dinv * rhs
        u = Dinv * x
        th = t - u * (x @ t) / (m1 + x @ u)
        return np.linalg.norm(th - th_R)

    best = 0.0
    cands = [np.eye(d)[i] for i in range(d)]
    cands += [v / np.linalg.norm(v) for v in RNG.standard_normal((400, d))]
    for x in cands:
        for yv in (Y, -Y):
            best = max(best, move(x, yv), move(-x, yv))
    # local refinement on the sphere from the best few starts
    for x0 in cands[:d]:
        for yv in (Y, -Y):
            f = lambda w: -move(w / max(np.linalg.norm(w), 1e-12), yv)
            r = minimize(f, x0 + 1e-3 * RNG.standard_normal(d), method="Nelder-Mead",
                         options=dict(maxiter=4000, xatol=1e-10, fatol=1e-14))
            best = max(best, -r.fun)
    return best, max(move(np.eye(d)[0], yv) for yv in (Y, -Y))


gamma = 2.0 * Y * (1.0 + 1.0 / lam) / (lam * n)
u = np.zeros(d)
u[0] = 1.0

# one adversary for every design, admissible for all of them
lmax = []
for p in P:
    a_, s_ = design(p)
    lmax.append((np.bincount(a_, minlength=d) / n + lam).max())
ETA = 0.5 / max(lmax)

th_true = RNG.standard_normal(d)
th_true /= np.linalg.norm(th_true)

rows = []
for p in P:
    axis, sign = design(p)
    y = np.clip(sign * th_true[axis] + 0.05 * RNG.standard_normal(n), -Y, Y)
    # the deleted record is row 0, the one on the readout direction
    perm = np.r_[np.arange(1, n), 0]
    axis, sign, y = axis[perm], sign[perm], y[perm]
    G, b = stats(axis, sign, y, n)
    GR, bR = stats(axis, sign, y, n - 1)
    th0 = b / (G + lam)
    thr = bR / (GR + lam)
    assert np.linalg.norm(th0 - thr) <= gamma
    rs, rs_u = retain_sensitivity(GR, bR, n - 1)
    assert rs <= gamma + 1e-12

    kappa = float(GR[0] + lam)
    k_half = int(np.ceil(np.log(1.0 / c) / -np.log1p(-ETA * kappa)))

    # measured horizon: run the adversary from both models along u
    def run(th, K=20000):
        out = []
        for _ in range(K + 1):
            out.append(th[0])
            th = th - ETA * ((GR + lam) * th - bR)
        return np.array(out)

    noise = RNG.standard_normal(d) * (Z * gamma / 1.0)   # eps = 1, global
    sig = run(thr + noise) - run(th0)
    sig = np.abs(sig)
    frac = 1.0 - sig / sig[0]
    k_meas = int(np.argmax(frac >= c))

    # utility at eps = 1 under each calibration: excess risk on the retained data
    def excess(s):
        e = s * s * float(np.sum(GR + lam)) / 2.0   # E[ (1/2) w'(G+lam)w ], w ~ N(0, s^2 I)
        return e

    s_glob = {e: Z * gamma / e for e in EPS}
    s_ret = {e: Z * rs / e for e in EPS}
    risk0 = float(0.5 * np.mean((sign * thr[axis] - y) ** 2) + 0.5 * lam * thr @ thr)
    rows.append(dict(
        p=p, kappa=kappa, k_half_pred=k_half, k_half_meas=k_meas,
        global_sens=gamma, retain_sens=rs, sens_ratio=rs / gamma,
        readout_sens=rs_u, eps_readout_at_eps_retain1=rs_u / rs,
        eps_retain_at_global_noise={str(e): e * rs / gamma for e in EPS},
        sigma_global={str(e): s_glob[e] for e in EPS},
        sigma_retain={str(e): s_ret[e] for e in EPS},
        retained_risk=risk0,
        excess_risk_global={str(e): excess(s_glob[e]) for e in EPS},
        excess_risk_retain={str(e): excess(s_ret[e]) for e in EPS},
    ))

res = dict(d=d, n=n, lam=lam, Y=Y, delta=delta, eta=ETA, gamma=gamma, rows=rows)
json.dump(res, open(OUT, "w"), indent=2)
print(f"gamma={gamma:.4g} eta={ETA:.4f}")
for r in rows:
    print(f"p={r['p']:<7} kappa={r['kappa']:.4f} k1/2 pred={r['k_half_pred']:>4} "
          f"meas={r['k_half_meas']:>4}  RS={r['retain_sens']:.3e} RS/GS={r['sens_ratio']:.3f} "
          f"RSu={r['readout_sens']:.3e} epsU/epsRS={r['eps_readout_at_eps_retain1']:.3f} "
          f"eps_RS@eps_GS=1: {r['eps_retain_at_global_noise']['1.0']:.3f}  "
          f"risk={r['retained_risk']:.4f} exc(eps=1,GS)={r['excess_risk_global']['1.0']:.2e} "
          f"exc(eps=1,RS)={r['excess_risk_retain']['1.0']:.2e}")
