"""
Numerical check of every closed form against an independent solver.

Convention: sigma_k(v) = f(Phi_k(theta0 + v)) - f(Phi_k(theta0)) = a' A^k v, so
k-step durability is the constraint a' A^k v >= D and the optimal update points
along +H^{-1} A^k a. Each closed form (stationary price, non-stationary price,
adjoint propagation for a preconditioned adversary, nonconvex lower bound) is
compared with a sequential-quadratic-programming solve that never sees it.
"""
import json
import os
import numpy as np
from scipy.optimize import minimize

OUT_DIR = os.environ.get("OUT_DIR", "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "verify_closed_forms.json")
RNG = np.random.default_rng(31415)
res = {}


def spd(d, cond):
    Q, _ = np.linalg.qr(RNG.standard_normal((d, d)))
    ev = np.exp(np.linspace(0, np.log(cond), d))
    ev = ev / ev.max()
    return Q @ np.diag(ev) @ Q.T, ev


def prop(H, a, eta, k, P=None):
    """b_k, the adjoint-propagated readout gradient."""
    b = a.copy()
    for _ in range(k):
        b = b - eta * (H @ (b if P is None else P @ b))
    return b


def numeric(H, g, a, eta, k, D, P=None):
    """Solve the QP numerically with no reference to the closed form."""
    d = len(a)
    Ak = np.eye(d)
    M = np.eye(d) - eta * (H if P is None else P @ H)
    for _ in range(k):
        Ak = M @ Ak
    obj = lambda v: g @ v + 0.5 * v @ H @ v
    jac = lambda v: g + H @ v
    con = {"type": "ineq",
           "fun": lambda v: a @ (Ak @ v) - D,
           "jac": lambda v: Ak.T @ a}
    # Feasible starting points built from the constraint alone, so the
    # optimizer never sees the closed-form answer.
    w = Ak.T @ a
    base = (D / float(w @ w)) * w
    best = None
    for j in range(6):
        v0 = base * (1.0 + 0.5 * j) + RNG.standard_normal(d) * 1e-3
        if a @ (Ak @ v0) < D:
            v0 = v0 * (D / max(a @ (Ak @ v0), 1e-12))
        r = minimize(obj, v0, jac=jac, constraints=[con], method="SLSQP",
                     options=dict(maxiter=3000, ftol=1e-16))
        if a @ (Ak @ r.x) >= D - 1e-7 and (best is None or r.fun < best.fun):
            best = r
    return best


# ---- stationary ---------------------------------------------------------
errs, dirs = [], []
for _ in range(120):
    d = int(RNG.integers(6, 20))
    H, ev = spd(d, float(np.exp(RNG.uniform(np.log(3), np.log(316)))))
    eta = float(RNG.uniform(0.1, 1.0) / ev.max())
    a = RNG.standard_normal(d)
    a /= np.linalg.norm(a)
    k = int(RNG.integers(0, 13))
    D = float(RNG.uniform(0.1, 2.0))
    b = prop(H, a, eta, k)
    S = float(b @ np.linalg.solve(H, b))
    U = D * D / (2 * S)
    v = (D / S) * np.linalg.solve(H, b)
    r = numeric(H, np.zeros(d), a, eta, k, D)
    errs.append(abs(U - r.fun) / max(abs(r.fun), 1e-12))
    dirs.append(float(b @ v))                      # must equal +D
res["stationary_max_rel_err"] = float(np.max(errs))
res["stationary_margin_max_err"] = float(np.max([abs(x - D) for x in [dirs[-1]]]))
res["stationary_margin_all_positive"] = bool(np.all(np.array(dirs) > 0))

# ---- non-stationary -----------------------------------------------------
errs, inact = [], 0
for _ in range(120):
    d = int(RNG.integers(6, 20))
    H, ev = spd(d, float(np.exp(RNG.uniform(np.log(3), np.log(316)))))
    eta = float(RNG.uniform(0.1, 1.0) / ev.max())
    a = RNG.standard_normal(d)
    a /= np.linalg.norm(a)
    g = RNG.standard_normal(d) * 0.3
    k = int(RNG.integers(0, 13))
    D = float(RNG.uniform(0.1, 2.0))
    b = prop(H, a, eta, k)
    Hi_b, Hi_g = np.linalg.solve(H, b), np.linalg.solve(H, g)
    S = float(b @ Hi_b)
    s = -float(b @ Hi_g)                            # corrected sign
    mu = max(D - s, 0.0) / S
    U = 0.5 * mu * mu * S - 0.5 * float(g @ Hi_g)
    if s >= D:
        inact += 1
    r = numeric(H, g, a, eta, k, D)
    errs.append(abs(U - r.fun) / max(abs(r.fun), 1e-12))
res["nonstationary_max_rel_err"] = float(np.max(errs))
res["nonstationary_inactive"] = inact

# ---- preconditioned: adjoint vs forward ---------------------------------
errs, fwd_miss, fwd_margin = [], 0, []
for _ in range(120):
    d = int(RNG.integers(6, 16))
    H, ev = spd(d, float(np.exp(RNG.uniform(np.log(3), np.log(100)))))
    Pd = np.exp(RNG.uniform(np.log(0.1), np.log(10.0), size=d))
    P = np.diag(Pd)
    eta = float(RNG.uniform(0.1, 0.8) / (ev.max() * Pd.max()))
    a = RNG.standard_normal(d)
    a /= np.linalg.norm(a)
    k = int(RNG.integers(1, 10))
    D = float(RNG.uniform(0.1, 2.0))

    b_adj = a.copy()                                 # (I - eta H P)^k a
    for _ in range(k):
        b_adj = b_adj - eta * (H @ (P @ b_adj))
    S = float(b_adj @ np.linalg.solve(H, b_adj))
    U = D * D / (2 * S)
    r = numeric(H, np.zeros(d), a, eta, k, D, P=P)
    errs.append(abs(U - r.fun) / max(abs(r.fun), 1e-12))

    b_fwd = a.copy()                                 # (I - eta P H)^k a, wrong
    for _ in range(k):
        b_fwd = b_fwd - eta * (P @ (H @ b_fwd))
    Sf = float(b_fwd @ np.linalg.solve(H, b_fwd))
    v_f = (D / Sf) * np.linalg.solve(H, b_fwd)
    realized = float(b_adj @ v_f)                    # true margin of that update
    fwd_margin.append(realized / D)
    if realized < D - 1e-9:
        fwd_miss += 1
res["precond_max_rel_err"] = float(np.max(errs))
res["precond_forward_misses"] = fwd_miss
res["precond_forward_margin_median"] = float(np.median(fwd_margin))
res["precond_forward_margin_worst"] = float(np.min(fwd_margin))

json.dump(res, open(OUT, "w"), indent=2)
for k_, v_ in res.items():
    print(f"{k_:38s} {v_}")

# ---- nonconvex lower bound, corrected convention ------------------------
def f_and_grad(x, H, gam):
    return (0.5 * x @ H @ x + gam / 6.0 * np.sum(x ** 3),
            H @ x + gam / 2.0 * x ** 2)


ratios, skipped = [], 0
for _ in range(240):
    d = 12
    H, ev = spd(d, 20.0)
    H, ev = H / ev.min(), ev / ev.min()          # unit smallest curvature
    gam = float(RNG.uniform(0.05, 0.5))          # Hessian-Lipschitz constant L
    eta = float(0.5 / ev.max())
    a = RNG.standard_normal(d)
    a /= np.linalg.norm(a)
    k = int(RNG.integers(1, 7))
    D = float(RNG.uniform(0.05, 0.4))
    R = 1.0
    m = float(ev.min() - gam * R)
    if m <= 0:
        continue
    b = prop(H, a, eta, k)
    S = float(b @ np.linalg.solve(H, b))
    v = (D / S) * np.linalg.solve(H, b)          # corrected direction
    q = 1.0 - eta * m
    Lam = gam * np.linalg.norm(a) * q ** max(k - 1, 0) / (2.0 * m)
    rs = np.linspace(1e-3, R, 400)
    lb = np.min((1.0 - gam * rs / (3.0 * m)).clip(0.0) *
                np.maximum(np.clip(D - Lam * rs ** 2, 0, None) ** 2 / (2.0 * S),
                           m * rs ** 2 / 2.0))
    x = v.copy()
    for _ in range(k):
        x = x - eta * f_and_grad(x, H, gam)[1]
    if float(a @ x) < D * 0.98:                  # quadratic update must stay feasible
        skipped += 1
        continue
    ratios.append(f_and_grad(v, H, gam)[0] / max(lb, 1e-18))

res["nonconvex_cases"] = len(ratios)
res["nonconvex_skipped_infeasible"] = skipped
res["nonconvex_bound_holds"] = bool(np.min(ratios) >= 1.0) if ratios else None
res["nonconvex_min_ratio"] = float(np.min(ratios)) if ratios else None
res["nonconvex_median_ratio"] = float(np.median(ratios)) if ratios else None
res["nonconvex_max_ratio"] = float(np.max(ratios)) if ratios else None

json.dump(res, open(OUT, "w"), indent=2)
print("--- nonconvex")
for k_ in ["nonconvex_cases", "nonconvex_skipped_infeasible", "nonconvex_bound_holds",
           "nonconvex_min_ratio", "nonconvex_median_ratio", "nonconvex_max_ratio"]:
    print(f"{k_:38s} {res[k_]}")
