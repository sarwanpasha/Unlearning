#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
# Several configurations at once on one large GPU (e.g. a 96 GB GH200).
# Usage (from the repository root):
#   sbatch --job-name=unlearning_wmdp [-p ... -A ...] cluster/run_packed.sbatch \
#          CONFIG_FILE OUT_DIR FIRST_LINE LAST_LINE CONCURRENCY
# Lines are 1-based and inclusive; each line is one llm/run_llm.py argument list.
set -u
CF=$1; OD=$2; FIRST=$3; LAST=$4; CONC=$5
ROOT=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$ROOT"
export P="$ROOT" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$OD" "$OD/logs"
run_one() {
  LINE="$1"
  TAG=$(echo "$LINE" | sed 's/--//g; s/ /_/g')
  if [ -s "$OD/$TAG.json" ]; then echo "exists $TAG"; return 0; fi
  ${PYTHON:-python} llm/run_llm.py $LINE --out "$OD/$TAG.json" > "$OD/logs/$TAG.log" 2>&1
  echo "$TAG RUNEXIT=$?"
}
export -f run_one
export OD PYTHON
sed -n "${FIRST},${LAST}p" "$CF" | xargs -d '\n' -P "$CONC" -I{} bash -c 'run_one "$@"' _ {}
