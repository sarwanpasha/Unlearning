# The Exact Price of Resistant Unlearning

Code for the paper *The Exact Price of Resistant Unlearning*. A certified
unlearning guarantee constrains the distribution of the released parameters
but says nothing about how quickly a downstream fine-tuner can bring the
deleted content back. The paper adds a relearning horizon $k$ and a margin $D$
to the definition of forgetting and derives the minimum utility cost of
meeting them in a local quadratic model:

$$U^\star(k) = \frac{D^2}{2S_k}, \qquad S_k = a^\top A^k H^{-1} A^k a, \qquad v^\star = \frac{D}{S_k} H^{-1} A^k a,$$

where $H$ is the Hessian of the retained loss, $a$ the gradient of the deletion
readout, and $A = I - \eta H$ the Jacobian of the adversary's gradient step.
Directionally flattened unlearning (DFU) implements $v^\star$ with
Hessian-vector products only.

The repository has two parts.

* `theory/`: the closed forms and every synthetic experiment. CPU only, runs in
  minutes, and reproduces the numbers of the synthetic sections exactly.
* `llm/`: the language-model experiments (TOFU, MUSE-News and WMDP-Cyber with
  Qwen2.5-0.5B-Instruct), with nine unlearning methods, relearning adversaries,
  forget metrics, leakage tests and plasticity. Needs one GPU.

## Installation

```bash
git clone https://github.com/sarwanpasha/Unlearning.git
cd Unlearning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The language-model code downloads `Qwen/Qwen2.5-0.5B-Instruct` and the
`locuslab/TOFU`, `muse-bench/MUSE-News`, `cais/wmdp`, `cais/mmlu` and
`Salesforce/wikitext` datasets from the Hugging Face Hub on first use.

## Quick start (CPU, about one minute)

```bash
python theory/theory.py
```

prints the closed-form price next to an independent numerical solution of the
same quadratic program, and the margin the optimal update achieves:

```
k= 0  closed form U=0.053134  numerical U=0.053134  margin=1.000000
k= 4  closed form U=0.074425  numerical U=0.074425  margin=1.000000
k= 8  closed form U=0.094790  numerical U=0.094790  margin=1.000000
k=16  closed form U=0.142895  numerical U=0.142895  margin=1.000000
```

## Synthetic experiments

Each script writes a JSON file to `results/` (override with `OUT_DIR=...`) and
prints a summary.

| Script | What it checks | Runtime |
|---|---|---|
| `theory/verify_closed_forms.py` | Theorem 1, Corollaries 1 and 2 (adjoint vs forward propagation), and the nonconvex lower bound of Theorem 2, each against an SQP solve that never sees the closed form | ~1 min |
| `theory/certified_removal.py` | Ridge regression with exact removal and Gaussian output perturbation at $\varepsilon \in \{1,2,5,10\}$: global, retain and per-record sensitivity, and the relearning horizon of four designs with the same certificate | ~3 min |
| `theory/adaptive_adversary.py` | A recipient that picks its diagonal preconditioner to erase the margin, and the robust design | seconds |
| `theory/past_horizon.py` | The margin of the optimal update after its design horizon | seconds |

```bash
python theory/verify_closed_forms.py
python theory/certified_removal.py
python theory/adaptive_adversary.py
python theory/past_horizon.py
```

Expected output of `certified_removal.py` (the four designs share
$\varepsilon = 1$ and differ 34-fold in their horizon):

```
p=0.4     kappa=0.4100 k1/2 pred=   2 meas=   2  RS=1.007e-05 ...
p=0.04    kappa=0.0500 k1/2 pred=  14 meas=  14  RS=1.006e-05 ...
p=0.004   kappa=0.0140 k1/2 pred=  51 meas=  51  RS=1.004e-05 ...
p=0.0004  kappa=0.0104 k1/2 pred=  68 meas=  68  RS=1.003e-05 ...
```

## Language-model experiments

One call of `llm/run_llm.py` is one configuration: it fine-tunes (and caches) a
LoRA adapter on the benchmark, unlearns until the deleted targets' negative
log-likelihood has risen by 1 nat, attacks the released model, and writes every
measurement to one JSON file.

### Working example (one GPU with 16 GB, about 15 minutes)

```bash
python llm/run_llm.py --bench tofu --method dfu --seed 0 \
    --relearn_steps 20 --plast_steps 5 --out results/example/dfu.json
python llm/run_llm.py --bench tofu --method ga --seed 0 \
    --relearn_steps 20 --plast_steps 5 --out results/example/ga.json
python - <<'PY'
import json
for m in ("dfu", "ga"):
    R = json.load(open(f"results/example/{m}.json"))
    print(f"{m:4s} suppression={R['suppression']:.2f} utility_cost={R['utility_cost']:.3f} "
          f"curvature={R['kappa_dir']:.3f} recovered_benign={100*R['recovery_frac_benign']:.1f}% "
          f"recovered_probe={100*R['recovery_frac_probe']:.1f}%")
PY
```

Both methods reach the same suppression of about 1 nat; DFU leaves a smaller
share of it to the benign and probe adversaries, at a lower curvature along
its update. The first call also fine-tunes the TOFU model, which is cached
under `results/` and reused by the second.

### Main options

| Option | Meaning |
|---|---|
| `--bench {tofu,muse,wmdp}` | benchmark |
| `--method {ga,gagd,npo,npo_sam,rmu,mcu,cir,tar,dfu,none}` | unlearning method (`none` runs the attacks on the never-unlearned model) |
| `--target_delta` | suppression target in nats (default 1.0) |
| `--dfu_k`, `--precond adam` | DFU horizon and adjoint propagation with an Adam-like preconditioner |
| `--relearn_steps`, `--relearn_lr` | adversary budget in passes over its data, and its learning rate |
| `--strong` | also run the strong adversaries (200 passes at `1e-4`) |
| `--adv_full` | the adversary fine-tunes every weight instead of the LoRA subspace |
| `--metrics` | ROUGE-L, truth ratio and forget quality (TOFU), MMLU, Min-K% membership inference, two-release leakage attacks |
| `--plast_steps` | plasticity measurement on a held-out task (0 disables) |

Key output fields: `suppression`, `utility_cost`, `kappa_dir` (curvature of the
retained loss along the update), `recovery_frac_{benign,probe}[_strong]`,
`plast_recovery_frac`, `rougeL_forget_*`, `forget_quality_*`, `mmlu_*`,
`privleak`, `twopass_*_auc_*`.

### Full study on a Slurm cluster

```bash
python cluster/make_configs.py           # writes cluster/configs/*.txt
sbatch --job-name=unlearning_tofu --array=0-49 --gres=gpu:1 \
       cluster/run.sbatch cluster/configs/tofu.txt results/tofu
# likewise for muse.txt, wmdp.txt, tofu_fulladv.txt and rmu_screen.txt
python cluster/aggregate.py results agg.json runs.csv
```

Add the partition and account options your cluster requires. On a node with a
single large GPU, `cluster/run_packed.sbatch` runs several configurations at
once, for example lines 1 to 14 of a config file with 7 concurrent runs:

```bash
sbatch --job-name=unlearning_wmdp cluster/run_packed.sbatch \
       cluster/configs/wmdp.txt results/wmdp 1 14 7
```

Run one configuration per seed first (for example the `--method none` lines),
so that the fine-tuned checkpoints are cached before concurrent runs read them.
 `aggregate.py`
writes per-(arm, method) means and standard deviations, and a per-run CSV.

## Figures

```bash
python figures/figures.py               # writes Figures/*.pdf
```

The language-model panels read the first study's per-run measurements in
`results/first_study_runs.csv` and `results/first_study_extra.csv`.

## Repository layout

```
theory/    closed forms (theory.py) and synthetic experiments
llm/       language-model pipeline (llm_common.py, run_llm.py)
cluster/   Slurm array script, config generator, aggregation
figures/   figure script
results/   first-study measurements; outputs of the scripts land here
```
