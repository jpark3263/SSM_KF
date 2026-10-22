#!/usr/bin/env bash
# Clean-condition evaluation (10- and 20-frame tracks) and Fig. 5 with the released checkpoints.
# Usage: bash scripts/reproduce.sh [mamba1_cuda|mamba1] [output_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

SSM_TYPE=${1:-mamba1_cuda}
OUT=${2:-results}
COMMON="--use_cuda --ssm_type ${SSM_TYPE}"

python run.py ${COMMON} --seq_len 10
python run.py ${COMMON} --seq_len 20
python scripts/fig5_visualize.py ${COMMON} --output_dir "${OUT}/fig5" --max_frames 100
