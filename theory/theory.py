"""Closed-form price of k-step durability under a local quadratic model.

Convention: the adversary runs theta <- theta - eta P grad L_R, the readout f
is large when the deleted content is gone, and the attributable suppression
after k steps is sigma_k(v) = b_k' v with b_k = (I - eta H P)^k a, a = grad f.
"""
import numpy as np
from scipy.optimize import minimize


def spd(d, cond, rng):
    """Random SPD matrix with eigenvalues log-spaced over [1/cond, 1]."""
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    ev = np.logspace(-np.log10(cond), 0.0, d)
    return (Q * ev) @ Q.T, ev


def propagate(H, a, eta, k, P=None):
    """Adjoint-propagated readout b = (I - eta H P)^k a.  P=None means P=I."""
    b = a.copy()
    for _ in range(k):
        b = b - eta * (H @ (b if P is None else P @ b))
    return b


def price(H, a, eta, k, D, g=None, P=None):
    """Exact minimum retained-loss increase subject to sigma_k(v) >= D.

    Returns (U, v, S, s): the optimal cost, the optimal update, the curvature
    functional S_k = b' H^{-1} b, and the suppression s_k = -b' H^{-1} g that the
    drift toward the retained optimum already supplies.
    """
    b = propagate(H, a, eta, k, P)
    Hi_b = np.linalg.solve(H, b)
    S = float(b @ Hi_b)
    if g is None:
        return D * D / (2.0 * S), (D / S) * Hi_b, S, 0.0
    Hi_g = np.linalg.solve(H, g)
    s = -float(b @ Hi_g)
    mu = max(D - s, 0.0) / S
    v = -Hi_g + mu * Hi_b
    U = 0.5 * mu * mu * S - 0.5 * float(g @ Hi_g)
    return U, v, S, s


def numeric_price(H, a, eta, k, D, g=None, P=None, seed=0):
    """The same problem solved numerically, as an independent check."""
    d = len(H)
    g = np.zeros(d) if g is None else g
    b = propagate(H, a, eta, k, P)
    obj = lambda v: float(g @ v + 0.5 * v @ H @ v)
    jac = lambda v: g + H @ v
    cons = [{"type": "ineq", "fun": lambda v: float(b @ v) - D, "jac": lambda v: b}]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(4):
        r = minimize(obj, rng.standard_normal(d) * 0.1, jac=jac,
                     constraints=cons, method="SLSQP",
                     options={"maxiter": 800, "ftol": 1e-14})
        if r.success and (best is None or r.fun < best.fun):
            best = r
    return float(best.fun), best.x


def frontier(h_ret, h_oth, rho, eta, k, D):
    """Two-direction entanglement model: closed-form price as a function of rho."""
    S = (rho ** 2) * (1 - eta * h_ret) ** (2 * k) / h_ret \
        + (1 - rho ** 2) * (1 - eta * h_oth) ** (2 * k) / h_oth
    return D * D / (2.0 * S)


def relearn_steps(kappa, eta, c=0.5):
    """Steps for a one-dimensional direction of curvature kappa to recover a fraction c."""
    return np.ceil(np.log(1.0 / c) / (-np.log(1.0 - eta * kappa)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    H, ev = spd(10, 30.0, rng)
    a = rng.standard_normal(10)
    a /= np.linalg.norm(a)
    eta = 0.5 / ev.max()
    for k in (0, 4, 8, 16):
        U, v, S, _ = price(H, a, eta, k, 1.0)
        Un, _ = numeric_price(H, a, eta, k, 1.0)
        print(f"k={k:2d}  closed form U={U:.6f}  numerical U={Un:.6f}  "
              f"margin={propagate(H, a, eta, k) @ v:.6f}")
