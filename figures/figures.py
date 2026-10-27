"""Figures for the paper.  No titles or annotations inside the axes."""
import os
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "theory"))
import theory as T

plt.rcParams.update({
    "font.size": 9.8, "axes.labelsize": 10.8, "axes.titlesize": 10.5,
    "xtick.labelsize": 9.8, "ytick.labelsize": 9.8, "legend.fontsize": 9.3,
    "lines.linewidth": 1.6, "lines.markersize": 4.5, "axes.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015, "pdf.fonttype": 42, "font.family": "serif",
    "mathtext.fontset": "cm",
})
COL, FULL = 3.34, 6.95
FIG = os.environ.get("FIG_DIR", "Figures") + "/"
os.makedirs(FIG, exist_ok=True)
RES = os.environ.get("RES_DIR", "results")
C = ["#1b1b1b", "#c0392b", "#2471a3", "#1e8449", "#8e44ad"]


def fig_price():
    rng = np.random.default_rng(3)
    ks = np.arange(0, 17)
    f, ax = plt.subplots(figsize=(COL, 2.05))
    for i, cond in enumerate([10.0, 30.0, 100.0]):
        H, ev = T.spd(24, cond, rng); eta = 0.9 / ev.max()
        a = rng.standard_normal(24); a /= np.linalg.norm(a)
        ex = [T.price(H, a, eta, int(k), 1.0)[0] for k in ks]
        nu = [T.numeric_price(H, a, eta, int(k), 1.0)[0] for k in ks[::4]]
        ax.semilogy(ks, ex, color=C[i], label=r"$\kappa(H)=%d$" % cond)
        ax.semilogy(ks[::4], nu, "o", color=C[i], mfc="none")
    ax.set_xlabel(r"resistance horizon $k$")
    ax.set_ylabel(r"price $U^\star(k)$")
    ax.legend(loc="upper left")
    f.savefig(FIG + "price.pdf"); plt.close(f)


def fig_frontier():
    rho = np.linspace(0.0, 0.99, 240)
    f, ax = plt.subplots(figsize=(COL, 2.05))
    for i, k in enumerate([0, 4, 8, 16]):
        ax.semilogy(rho, [T.frontier(0.4, 0.02, r, 1.0, k, 1.0) for r in rho],
                    color=C[i], label=r"$k=%d$" % k)
    ax.set_xlabel(r"forget-retain alignment $\rho$")
    ax.set_ylabel(r"price $U^\star(k)$")
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0)
    f.savefig(FIG + "frontier.pdf"); plt.close(f)


def fig_llm():
    d = pd.read_csv(os.path.join(RES, "first_study_runs.csv"))
    e = pd.read_csv(os.path.join(RES, "first_study_extra.csv"))
    m = d[d.tag.str.match(r"^m(ga|gagd|npo|nposam|dfu)_s\d_")].copy()
    order = ["ga", "gagd", "npo", "npo_sam", "dfu"]
    lab = {"ga": "GA", "gagd": "GA+GD", "npo": "NPO", "npo_sam": "NPO+SAM", "dfu": "DFU"}
    s7 = e[e._src == "llm7b"]

    f, axs = plt.subplots(1, 3, figsize=(FULL, 1.95))
    a0 = axs[0]
    x = np.arange(len(order)); w = 0.38
    g5 = m[m.tag.str.endswith("_all")].groupby("method")
    mu5 = [100 * g5.recovery_frac_benign.mean()[k] for k in order]
    sd5 = [100 * g5.recovery_frac_benign.std()[k] for k in order]
    g7 = s7.groupby("method")
    mu7 = [100 * g7.recovery_frac_benign.mean().get(k, np.nan) for k in order]
    sd7 = [100 * g7.recovery_frac_benign.std().get(k, np.nan) for k in order]
    a0.bar(x - w / 2, mu5, w, yerr=sd5, color=C[2], capsize=2.5, label="0.5B")
    a0.bar(x + w / 2, mu7, w, yerr=sd7, color=C[1], capsize=2.5, label="7B")
    a0.set_xticks(x); a0.set_xticklabels([lab[k] for k in order], rotation=28, ha="right")
    a0.set_ylabel("benign recovery (%)")
    a0.legend(loc="upper right")

    def scat(ax, col, xl, logx):
        for i, k in enumerate(order):
            s = m[m.method == k]
            ax.plot(s[col], 100 * s.recovery_frac_benign, "o", color=C[i],
                    mfc="none", label=lab[k])
        xs = np.asarray(m[col], float); ys = 100 * np.asarray(m.recovery_frac_benign, float)
        A = np.vstack([xs, np.ones_like(xs)]).T
        b = np.linalg.lstsq(A, ys, rcond=None)[0]
        gx = np.linspace(xs.min(), xs.max(), 50)
        ax.plot(gx, b[0] * gx + b[1], "-", color="0.45", lw=1.2, zorder=0)
        ax.set_xlabel(xl)
        if logx:
            ax.set_xscale("log")

    scat(axs[1], "kappa_dir", r"curvature $\kappa_a$", False)
    axs[1].set_ylabel("benign recovery (%)")
    scat(axs[2], "eps_proxy", r"certificate $\varepsilon$", False)
    axs[2].set_ylabel("benign recovery (%)")
    h, l = axs[1].get_legend_handles_labels()
    f.legend(h, l, loc="lower center", ncol=5, handletextpad=0.25,
             columnspacing=1.1, borderpad=0.1, bbox_to_anchor=(0.5, -0.045))
    f.tight_layout(pad=0.25, w_pad=1.1)
    f.savefig(FIG + "llm.pdf"); plt.close(f)


def fig_horizon():
    d = pd.read_csv(os.path.join(RES, "first_study_runs.csv"))
    e = pd.read_csv(os.path.join(RES, "first_study_extra.csv"))
    h = d[d.tag.str.match(r"^dfuk\d+_s\d$")].copy()
    h["k"] = h.tag.str.extract(r"dfuk(\d+)").astype(int)
    g = h.groupby("k").recovery_frac_benign.agg(["mean", "std"])
    p = e[(e._src == "llmP") & (e.precond == "adam")].copy()
    gp = p.groupby("dfu_k").recovery_frac_benign.agg(["mean", "std"])
    f, ax = plt.subplots(figsize=(COL, 2.05))
    ax.errorbar(g.index, 100 * g["mean"], yerr=100 * g["std"], marker="o",
                color=C[1], capsize=2.5, label=r"$P=I$")
    ax.errorbar(gp.index, 100 * gp["mean"], yerr=100 * gp["std"], marker="s",
                color=C[3], capsize=2.5, label="adjoint")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"requested horizon $k$")
    ax.set_ylabel("benign recovery (%)")
    ax.legend(loc="upper left")
    f.savefig(FIG + "horizon.pdf"); plt.close(f)


def fig_nonconvex():
    import json
    rng = np.random.default_rng(11)
    ks = np.arange(1, 13)
    H, ev = T.spd(16, 10.0, rng); eta = 0.4 / ev.max()
    a = rng.standard_normal(16); a /= np.linalg.norm(a)
    D, R, m = 0.3, 0.6, float(ev.min())
    f, ax = plt.subplots(figsize=(COL, 2.05))
    for i, gam in enumerate([0.0, 0.02, 0.05]):
        vals = []
        for k in ks:
            S = T.price(H, a, eta, int(k), D)[2]
            if gam == 0.0:
                vals.append(D * D / (2 * S)); continue
            mm = m - gam * R; q = 1 - eta * mm
            Lam = gam * q ** max(k - 1, 0) / (2 * mm)
            rs = np.linspace(1e-4, R, 2000)
            vals.append(np.min(np.clip(1 - gam * rs / (3 * mm), 0, None) *
                        np.maximum(np.clip(D - Lam * rs ** 2, 0, None) ** 2 / (2 * S),
                                   mm * rs ** 2 / 2)))
        ax.semilogy(ks, vals, color=C[i], marker="o",
                    label=(r"$L=0$" if gam == 0 else r"$L=%.2f$" % gam))
    ax.set_xlabel(r"resistance horizon $k$")
    ax.set_ylabel("lower bound")
    ax.legend(loc="lower right")
    f.savefig(FIG + "nonconvex.pdf"); plt.close(f)


def fig_supp():
    d = pd.read_csv(os.path.join(RES, "first_study_runs.csv"))
    f, axs = plt.subplots(1, 2, figsize=(FULL, 2.1))
    c = d[d.tag.str.startswith("cert_")].copy()
    for i, meth in enumerate(["npo", "dfu"]):
        s = c[c.method == meth].groupby("cert_noise").agg(
            e=("eps_certified", "mean"), r=("recovery_frac_benign", "mean"),
            u=("utility_cost", "mean"))
        axs[0].plot(s.e, 100 * s.r, "o-", color=C[i + 1], label=meth.upper())
    axs[0].set_xscale("log")
    axs[0].set_xlabel(r"certified $\varepsilon$")
    axs[0].set_ylabel("benign recovery (%)")
    axs[0].legend(loc="upper left")
    t = d[d.tag.str.match(r"^trdfuk\d+_s\d$")].copy()
    t["k"] = t.tag.str.extract(r"trdfuk(\d+)").astype(int)
    gt = t.groupby("k").agg(req=("trdfu_k_requested", "mean"),
                            gr=("trdfu_k_granted", "mean"))
    axs[1].plot(gt.index, gt.req, "o--", color="0.5", label="requested")
    axs[1].plot(gt.index, gt.gr, "s-", color=C[3], label="granted")
    axs[1].set_xscale("log", base=2)
    axs[1].set_xlabel(r"requested horizon $k$")
    axs[1].set_ylabel("horizon")
    axs[1].legend(loc="upper left")
    f.tight_layout(pad=0.25, w_pad=1.3)
    f.savefig(FIG + "supp.pdf"); plt.close(f)


if __name__ == "__main__":
    import os
    os.makedirs(FIG, exist_ok=True)
    fig_price(); fig_frontier(); fig_llm(); fig_horizon(); fig_nonconvex(); fig_supp()
    print("figures written")
