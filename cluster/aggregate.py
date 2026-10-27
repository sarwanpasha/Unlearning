"""Aggregate run files: every scalar field, mean, standard deviation and n per (arm, method).

Usage: python cluster/aggregate.py RESULTS_DIR OUT_JSON OUT_CSV [ARM1,ARM2,...]

RESULTS_DIR holds one subdirectory per arm (for example results/tofu), each
with the JSON files written by llm/run_llm.py. Runs with --method none are the
same attacks started from the never-unlearned model; when present, the
attributable suppression left after each attack (resid_* fields) is computed
against them, seed by seed.
"""
import csv, glob, json, math, os, sys
import numpy as np

root, out_json, out_csv = sys.argv[1:4]
ARMS = sys.argv[4].split(",") if len(sys.argv) > 4 else sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
rows = []
for arm in ARMS:
    for f in sorted(glob.glob(f"{root}/{arm}/*.json")):
        R = json.load(open(f))
        r = {"arm": arm, "file": os.path.basename(f)}
        m = R["method"]
        if m in ("rmu", "mcu"):
            m = f"{m}_a{int(R['rmu_alpha'])}_c{int(R['rmu_c'])}_l{R['rmu_layer']}"
        r["key"] = m
        for k, v in R.items():
            if isinstance(v, bool):
                r[k] = float(v)
            elif isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                r[k] = float(v)
        # learning speed: normalised area under the new-task curve, unlearned / reference
        cu, cr = R.get("plast_curve_unlearned"), R.get("plast_curve_reference")
        bu, br = R.get("plast_base_unlearned"), R.get("plast_base_reference")
        if cu and cr and bu and br:
            r["plast_speed"] = float(np.mean(np.array(cu) / bu) / np.mean(np.array(cr) / br))
        r["_curves"] = {t: R.get(f"relearn_curve_{t}") for t in
                        ("benign", "probe", "benign_strong", "probe_strong")}
        r["_supp"] = R.get("eval_nll_unlearned", 0) - R.get("eval_nll_trained", 0)
        rows.append(r)

# attributable suppression left after each attack, against the same attack run
# from the never-unlearned model (method none, same seed): Definition 1's sigma_k
ref = {(r["arm"], int(r.get("seed", -1))): r for r in rows if r["key"] == "none"}
for r in rows:
    if r["key"] == "none":
        continue
    rf = ref.get((r["arm"], int(r.get("seed", -1))))
    if rf is None:
        continue
    for t, cu in r["_curves"].items():
        cr = rf["_curves"].get(t)
        if cu and cr and len(cu) == len(cr) and r["_supp"] > 1e-9:
            diff = [(a - b) / r["_supp"] for a, b in zip(cu, cr)]
            r[f"resid_end_{t}"] = float(diff[-1])
            r[f"resid_min_{t}"] = float(min(diff))
for r in rows:
    r.pop("_curves", None)
    r.pop("_supp", None)

groups = {}
for r in rows:
    groups.setdefault(f"{r['arm']}:{r['key']}", []).append(r)
agg = {}
for g, rs in groups.items():
    keys = sorted({k for r in rs for k in r if isinstance(r[k], float)})
    a = {"n": len(rs), "seeds": sorted(int(r.get("seed", -1)) for r in rs)}
    for k in keys:
        v = [r[k] for r in rs if k in r]
        a[k] = float(np.mean(v))
        a[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    agg[g] = a
json.dump(agg, open(out_json, "w"), indent=1)

fields = ["arm", "key", "seed", "hit_target", "suppression", "utility_cost", "kappa_dir",
          "update_norm", "recovery_frac_benign", "recovery_frac_probe",
          "recovery_frac_benign_strong", "recovery_frac_probe_strong", "plast_speed",
          "plast_recovery_frac", "mmlu_trained", "mmlu_unlearned", "mmlu_retain_only",
          "rougeL_forget_trained", "rougeL_forget_unlearned", "rougeL_forget_retain_only",
          "rougeL_forget_after_probe_strong", "forget_quality_trained",
          "forget_quality_unlearned", "para_question_nll_trained", "para_question_nll_unlearned",
          "mia_auc_trained", "mia_auc_unlearned", "mia_auc_retain_only", "privleak",
          "twopass_lossdiff_auc_unlearned", "twopass_lossdiff_auc_retain_only",
          "twopass_grad_auc_unlearned", "twopass_grad_auc_retain_only",
          "twopass_hgrad_auc_unlearned", "twopass_hgrad_auc_retain_only",
          "wmdp_acc_V_trained", "wmdp_acc_V_unlearned", "wmdp_acc_V_after_probe_strong",
          "wmdp_acc_V_after_probe", "wmdp_acc_V_after_probe_ref", "wmdp_acc_V_after_probe_strong_ref",
          "resid_end_benign", "resid_end_probe", "resid_end_benign_strong", "resid_end_probe_strong",
          "resid_min_benign", "resid_min_probe", "resid_min_benign_strong", "resid_min_probe_strong",
          "resid_supp_end_probe", "resid_supp_end_probe_strong",
          "resid_supp_min_probe", "resid_supp_min_probe_strong",
          "eval_nll_trained", "eval_nll_unlearned", "seconds"]
with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(fields)
    for r in sorted(rows, key=lambda r: (r["arm"], r["key"], r.get("seed", -1))):
        w.writerow([r.get(k, "") if k in ("arm", "key") else
                    ("" if k not in r else f"{r[k]:.6g}") for k in fields])
print(len(rows), "runs", len(agg), "groups")
