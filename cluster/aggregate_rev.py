"""Summaries for the revision arms: generation-level durability, the
learning-rate search, the oracle adversary, and Welch tests against DFU.

Usage: python cluster/aggregate_rev.py RESULTS_DIR ARM [THRESH]

Prints one line per method with means over seeds. Every attack on a method is
paired with the same attack, same seed, run from the never-unlearned model
(method none), which gives the counterfactual of the attributable suppression.
"""
import glob, json, os, sys
import numpy as np
from scipy.stats import ttest_ind

root, arm = sys.argv[1], sys.argv[2]
THR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
runs = {}
for f in glob.glob(f"{root}/{arm}/*.json"):
    R = json.load(open(f))
    key = R["method"]
    if R.get("dfu_k", 8) != 8 and key == "dfu":
        key = f"dfu_k{R['dfu_k']}"
    if abs(R.get("target_delta", 1.0) - 1.0) > 1e-9 and key != "none":
        key = f"{key}_D{R['target_delta']:g}"
    runs.setdefault(key, {})[int(R["seed"])] = R
ref = runs.get("none", {})


def gen_mean(g, field="rouge", idx=None):
    v = np.array(g[field])
    return float(v[idx].mean() if idx is not None else v.mean())


def gen_stats(R, Rf, tag):
    """Generation-level recovery and attributable share along one attack."""
    gu, gt = R.get("gen_unlearned"), R.get("gen_trained")
    cu = R.get(f"gen_curve_{tag}")
    if not (gu and gt and cu):
        return {}
    held = None
    if tag.startswith("probe"):
        held = slice(R["n_probe"], None)
    r_unl, r_tr = gen_mean(gu, idx=held), gen_mean(gt, idx=held)
    gap = r_tr - r_unl
    traj = [gen_mean(g, idx=held) for g in cu]
    out = dict(r_unl=r_unl, r_tr=r_tr, r_end=traj[-1], r_max=max(traj),
               extract_end=float(np.mean(np.array(cu[-1]["rouge"])[held if held else slice(None)] >= THR)),
               extract_tr=float(np.mean(np.array(gt["rouge"])[held if held else slice(None)] >= THR)))
    if "rouge_para" in cu[-1]:
        out["para_end"] = gen_mean(cu[-1], "rouge_para", held)
        out["para_unl"] = gen_mean(gu, "rouge_para", held)
        out["para_tr"] = gen_mean(gt, "rouge_para", held)
    if gap > 1e-6:
        out["gen_recovery"] = (max(traj) - r_unl) / gap
        if Rf and Rf.get(f"gen_curve_{tag}"):
            tf = [gen_mean(g, idx=held) for g in Rf[f"gen_curve_{tag}"]]
            if len(tf) == len(traj):
                out["gen_kept"] = min((b - a) / gap for a, b in zip(traj, tf))
    return out


def attributable(R, Rf, tag):
    cu, cr = R.get(f"relearn_curve_{tag}"), (Rf or {}).get(f"relearn_curve_{tag}")
    s = R.get("eval_nll_unlearned", 0) - R.get("eval_nll_trained", 0)
    if not (cu and cr) or len(cu) != len(cr) or s <= 1e-9:
        return None
    return min((a - b) / s for a, b in zip(cu, cr))


rows = {}
for key, bys in sorted(runs.items()):
    if key == "none":
        continue
    acc = {}
    for seed, R in bys.items():
        Rf = ref.get(seed)
        rec = {"U": R.get("utility_cost"), "supp": R.get("suppression")}
        for tag in ("benign", "probe", "probe_strong"):
            for k_, v_ in gen_stats(R, Rf, tag).items():
                rec[f"{tag}:{k_}"] = v_
        lrs = sorted({k.split("_lr")[1] for k in R if k.startswith("recovery_frac_probe_lr")}, key=float)
        if lrs:
            rec["lrmax_benign_rec"] = max(R[f"recovery_frac_benign_lr{l}"] for l in lrs)
            rec["lrmax_probe_rec"] = max(R[f"recovery_frac_probe_lr{l}"] for l in lrs)
            ks = [attributable(R, Rf, f"probe_lr{l}") for l in lrs]
            ks = [k for k in ks if k is not None]
            if ks:
                rec["lrmin_probe_kept"] = min(ks)
            kb = [attributable(R, Rf, f"benign_lr{l}") for l in lrs]
            kb = [k for k in kb if k is not None]
            if kb:
                rec["lrmin_benign_kept"] = min(kb)
        for tag in ("benign_oracle", "probe_oracle", "probe", "probe_strong", "benign"):
            if f"recovery_frac_{tag}" in R:
                rec[f"rec_{tag}"] = R[f"recovery_frac_{tag}"]
                k_ = attributable(R, Rf, tag)
                if k_ is not None:
                    rec[f"kept_{tag}"] = k_
        for k_, v_ in rec.items():
            if v_ is not None:
                acc.setdefault(k_, []).append(v_)
    rows[key] = acc

fields = sorted({f for a in rows.values() for f in a})
print("n_seeds", {k: len(runs[k]) for k in runs})
for f in fields:
    line = [f]
    for key, acc in rows.items():
        if f in acc:
            v = np.array(acc[f])
            line.append(f"{key}={v.mean():.3f}/{v.std(ddof=1) if len(v) > 1 else 0:.3f}")
    print(" ".join(line))
# Welch tests of DFU against every other method on the main fields
main = [f for f in fields if f.startswith(("kept_", "rec_", "lr", "probe_strong:gen", "probe:gen", "benign:gen"))]
if "dfu" in rows:
    print("welch p-values (dfu vs method)")
    for f in main:
        a = rows["dfu"].get(f)
        if not a or len(a) < 2:
            continue
        ps = []
        for key, acc in rows.items():
            if key == "dfu" or f not in acc or len(acc[f]) < 2:
                continue
            ps.append(f"{key}={ttest_ind(a, acc[f], equal_var=False).pvalue:.3g}")
        print(f, " ".join(ps))
