#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
# One configuration per array task. Usage (from the repository root):
#   sbatch --job-name=unlearning_tofu --array=0-44 [--partition=... --gres=gpu:1 -A ...] \
#          cluster/run.sbatch cluster/configs/tofu.txt results/tofu
# Each line of the config file is the argument list of llm/run_llm.py for one run.
# Set PYTHON to the interpreter of your environment (default: python).
set -u
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
CF=$1; OD=$2; shift 2
cd "$ROOT"
export P="$ROOT"            # checkpoints are cached under $P/results/
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$CF")
TAG=$(echo "$LINE" | sed 's/--//g; s/ /_/g')
mkdir -p "$OD"
if [ -s "$OD/$TAG.json" ]; then echo "exists $TAG"; exit 0; fi
echo "CONFIG $LINE $*"
${PYTHON:-python} llm/run_llm.py $LINE "$@" --out "$OD/$TAG.json"
rc=$?; echo "RUNEXIT=$rc"; exit $rc
