#!/bin/bash
#
# Run the whole ALLCAPS pipeline through Snakemake, as a single SLURM job.
#
#   sbatch jobs/run_snakemake.sh path/to/config.yaml
#
# Snakemake resolves the DAG and skips anything already up to date, so this is
# safe to resubmit after a failure — it picks up where it stopped. For a run that
# fans out across many nodes instead, use a Snakemake cluster profile
# (`--profile`) rather than this script; here everything shares one allocation.
#
# For the manual stage-by-stage route, see jobs/allcaps_slurm.sh.

#SBATCH --job-name=allcaps
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --output=logs/allcaps_%j.out
#SBATCH --error=logs/allcaps_%j.err

set -euo pipefail

CONFIG="${1:?usage: sbatch jobs/run_snakemake.sh <config.yaml>}"
CORES="${CORES:-4}"
ENV_NAME="${ENV_NAME:-all-caps}"

# ── Environment ───────────────────────────────────────────────────────────────
# Adapt these three lines to your site; everything below is portable.
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"
# Prefer the env's libstdc++ over an older system one.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# wandb has no network on most compute nodes; sync afterwards with
# `wandb sync --sync-all`.
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p logs

# Resolve the config to an absolute path before cd'ing: the Snakefile must run
# from src/ (it resolves scripts/ relative to itself), but the config paths are
# written relative to wherever you invoked this from.
CONFIG="$(realpath "${CONFIG}")"
cd "$(dirname "$0")/../src"

echo "[INFO] config: ${CONFIG}"
echo "[INFO] cores:  ${CORES}"

# Dry run first — cheap, and catches a broken config before burning the GPU.
snakemake --configfile "${CONFIG}" --cores "${CORES}" --dry-run

snakemake --configfile "${CONFIG}" --cores "${CORES}" --printshellcmds --rerun-incomplete

echo "[INFO] done. If WANDB_MODE=offline, run 'wandb sync --sync-all' from a login node."
