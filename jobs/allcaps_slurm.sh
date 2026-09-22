#!/bin/bash
#
# ALLCAPS pipeline, stage by stage, on SLURM.
#
#   sbatch jobs/allcaps_slurm.sh            # run the stages enabled below
#   LOO_SEROTYPE=19A sbatch jobs/allcaps_slurm.sh   # leave-one-serotype-out fold
#
# Snakemake (jobs/run_snakemake.sh) is the better default. Use this one when you
# need to re-run a single stage against an existing results directory — refit the
# kNN index, re-score a query set, retrain with a serotype held out — without
# Snakemake deciding to rebuild anything downstream.
#
# Every stage is idempotent and writes into ${RESULTS_DIR}. Flip the flags in the
# STAGES block, set the paths in the CONFIG block, submit.

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

# ══ STAGES ════════════════════════════════════════════════════════════════════
# Set to true to run. Order is the dependency order; each reads the previous
# stage's output from ${RESULTS_DIR}.
  RUN_LOCUS_CUT=false   # 000  cut cps loci out of raw assemblies
  RUN_LABELS=false      # 001  clean and normalise metadata
  RUN_SPLIT=false       # 002  train/test split, grouped by sample
  RUN_EMBED=false       # 003  ProkBERT chunk embeddings        (GPU, slow)
  RUN_META=false        # 004  drop metadata rows with no embedding
  RUN_TRAIN=false       # 005  train TransformerTriHeadLR       (GPU, slow)
  RUN_INFER=false       # 006  embed the training set through the trained model
  RUN_EVAL=false        # 007  capsule + serotype evaluation
  RUN_SANITY=false      # 008  training-vs-query embedding round-trip check
  RUN_KNN_FIT=false     # 009  fit the novelty index
  RUN_KNN_ID=false      # 010  calibrate: score the training set against itself
  RUN_QUERY=false       # 011  run a query FASTA through the model
  RUN_KNN_QUERY=false   # 012  the deployed novelty call on the query
  RUN_EXPORT=false      # 013  export a pickle-free index for publication
  RUN_EMBED_TEST=false  # 014  ProkBERT chunk embeddings for the TEST split (GPU, slow)
  RUN_BASELINE_LR=false # 015  LR baseline on the same test split, for comparison

# ══ CONFIG ════════════════════════════════════════════════════════════════════
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
ANALYSIS_NAME="${ANALYSIS_NAME:-allcaps-run}"
RESULTS_DIR="${RESULTS_DIR:-/path/to/results/${ANALYSIS_NAME}}"
DATA_DIR="${DATA_DIR:-${RESULTS_DIR}/data}"

# Inputs
RAW_INFILES="${RAW_INFILES:-/path/to/infiles.txt}"        # one assembly FASTA path per line
RAW_METADATA="${RAW_METADATA:-/path/to/metadata.csv}"     # sample metadata
QUERY_FASTA="${QUERY_FASTA:-/path/to/query.fasta}"        # loci to serotype / screen
FLANK_FASTA="${FLANK_FASTA:-${REPO_DIR}/assets/dexB_aliA_ATCC700669.fasta}"

# Model
BASE_MODEL="neuralbioinfo/prokbert-mini-long"
HEAD_MODEL="transformer_trihead_lr"
EPOCHS=100
BATCH_SIZE=128
LR=0.001

# Hyperparameters of the released checkpoint. See TRAINING.md before changing:
# alpha=0 disables the contrastive loss, weight_geno=0 the genogroup head, and
# dataset_name selects the embedding directory layout.
MODEL_PARAMS='{"embedding_dim": 384, "output_dim": 128, "num_layers": 1, "nhead": 4,
               "k_folds": 5, "random_state": 42, "temperature": 0.07,
               "weight_fine": 1, "weight_coarse": 0.4,
               "alpha": 0, "weight_sero": 2, "weight_geno": 0,
               "dataset_name": "multidomain_chunked"}'
QUERY_PARAMS='{"chunk_size": 4000, "stride_ratio": 0.5, "max_length": 30000, "rolling_step": 2000}'

# Novelty detection (deployed configuration)
KNN_K=1
KNN_THRESHOLD_PERCENTILE=95.0
KNN_MAX_K=5
KNN_K_GRID="1,5,10,50,100,200,500,1000"   # "" to skip the k-sweep report

# Leave-one-out: set LOO_SEROTYPE to hold a serotype out of training. It is then
# unseen by construction, which is what makes it a valid novelty query set.
LOO_SEROTYPE="${LOO_SEROTYPE:-}"
SKIP_FLAG=""
if [[ -n "${LOO_SEROTYPE}" ]]; then
    SKIP_FLAG="--skip_labels ${LOO_SEROTYPE}"
    # Slashes are not path-safe ('15B/C'), so folds live in a mangled directory.
    RESULTS_DIR="${RESULTS_DIR}/loo/${LOO_SEROTYPE//\//-}"
    echo "[INFO] LOO fold: holding out '${LOO_SEROTYPE}' -> ${RESULTS_DIR}"
fi

# ══ ENVIRONMENT ═══════════════════════════════════════════════════════════════
# Adapt to your site.
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME:-all-caps}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

# Training calls wandb.init() unconditionally; compute nodes have no network.
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p logs "${RESULTS_DIR}" "${DATA_DIR}"
cd "${REPO_DIR}/src"

stage() { echo "[STAGE $1] $2"; }

# ══ PIPELINE ══════════════════════════════════════════════════════════════════

if [[ "${RUN_LOCUS_CUT}" == true ]]; then
    stage 000 "Cutting cps loci from raw assemblies"
    python -m scripts.data_locus_cutter \
        --infiles "${RAW_INFILES}" \
        --query   "${FLANK_FASTA}" \
        --outpref "${DATA_DIR}/contigs" \
        --cutoff 0.7 \
        --max-extension 30000 \
        --save-noncbl
fi

if [[ "${RUN_LABELS}" == true ]]; then
    stage 001 "Preprocessing labels"
    python -m scripts.data_labels_preprocessing \
        --metadata "${RAW_METADATA}" \
        --output_dir "${DATA_DIR}" \
        --cbl-fasta    "${DATA_DIR}/contigs.fasta" \
        --noncbl-fasta "${DATA_DIR}/contigs_noncbl.fasta"
fi

if [[ "${RUN_SPLIT}" == true ]]; then
    stage 002 "Train/test split (grouped by sample)"
    python -m scripts.helpers.data_train_test_split \
        --fastas   "${DATA_DIR}/contigs.fasta" "${DATA_DIR}/contigs_noncbl.fasta" \
        --metadata "${DATA_DIR}/initial_metadata.csv"  "${DATA_DIR}/initial_metadata.csv" \
        --ratios 0.9 \
        --output_dir "${DATA_DIR}"
fi

if [[ "${RUN_EMBED}" == true ]]; then
    stage 003 "ProkBERT chunk embeddings"
    python -m scripts.embed_transformer \
        --fasta "${DATA_DIR}/train.fasta" \
        --out_dir "${RESULTS_DIR}/base_embeddings_chunked" \
        --model_name "${BASE_MODEL}" \
        --chunk_size 4000 --stride_ratio 0.5 --device cuda
fi

if [[ "${RUN_META}" == true ]]; then
    stage 004 "Filtering metadata to samples that have embeddings"
    python -m scripts.data_labels_postprocessing \
        --clean_labels "${DATA_DIR}/train_metadata.csv" \
        --embedding_dir "${RESULTS_DIR}/base_embeddings_chunked" \
        --output_dir "${DATA_DIR}"
fi

if [[ "${RUN_TRAIN}" == true ]]; then
    stage 005 "Training"
    python -m scripts.trihead.train_trihead_transformer \
        --embedding_dir "${RESULTS_DIR}/base_embeddings_chunked" \
        --labels "${DATA_DIR}/final_metadata.csv" \
        --output "${RESULTS_DIR}/transformer_model.pth" \
        --device cuda --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --lr "${LR}" \
        --labeled_only --hierarchical_loss \
        --model_params "${MODEL_PARAMS}" \
        --aug_noise_std 0.01 --aug_chunk_dropout 0.1 \
        --aug_spec_freq 0.5 --aug_spec_width 16 --aug_n_views 2 \
        ${SKIP_FLAG}
fi

if [[ "${RUN_INFER}" == true ]]; then
    stage 006 "Inference on the training set"
    python -m scripts.trihead.infer_trihead_transformer \
        --embeddings_dir "${RESULTS_DIR}/base_embeddings_chunked" \
        --labels "${DATA_DIR}/final_metadata.csv" \
        --model "${RESULTS_DIR}/transformer_model.pth" \
        --output "${RESULTS_DIR}/inference_results.npz" \
        --device cuda --batch_size "${BATCH_SIZE}" \
        --model_params "${MODEL_PARAMS}" \
        --labeled_only ${SKIP_FLAG}
fi

if [[ "${RUN_EVAL}" == true ]]; then
    stage 007 "Evaluation"
    python -m scripts.eval_cbl_classifier \
        --embeddings "${RESULTS_DIR}/inference_results.npz" \
        --model "${RESULTS_DIR}/transformer_model.pth" \
        --output "${RESULTS_DIR}/cbl_results.txt" \
        --device cuda --batch_size "${BATCH_SIZE}" \
        --model_params "${MODEL_PARAMS}"

    python -m scripts.eval_serotype_classifier \
        --embeddings "${RESULTS_DIR}/inference_results.npz" \
        --model "${RESULTS_DIR}/transformer_model.pth" \
        --labels "${DATA_DIR}/final_metadata.csv" \
        --output_dir "${RESULTS_DIR}" \
        --device cuda --batch_size "${BATCH_SIZE}" \
        --model_params "${MODEL_PARAMS}" ${SKIP_FLAG}
fi

if [[ "${RUN_SANITY}" == true ]]; then
    stage 008 "Round-trip sanity check (training vs query embedding paths)"
    python -m scripts.tests.sanity_check_roundtrip \
        --fasta "${DATA_DIR}/train.fasta" \
        --labels "${DATA_DIR}/final_metadata.csv" \
        --model "${RESULTS_DIR}/transformer_model.pth" \
        --inference_npz "${RESULTS_DIR}/inference_results.npz" \
        --output_dir "${RESULTS_DIR}/sanity_roundtrip" \
        --base_model "${BASE_MODEL}" --device cuda
fi

if [[ "${RUN_KNN_FIT}" == true ]]; then
    stage 009 "Fitting the kNN novelty index"
    python -m scripts.knn_ood fit \
        --embeddings "${RESULTS_DIR}/inference_results.npz" \
        --labels "${DATA_DIR}/final_metadata.csv" \
        --output "${RESULTS_DIR}/knn_index.pkl" \
        --k "${KNN_K}" --distance_metric cosine
fi

if [[ "${RUN_KNN_ID}" == true ]]; then
    stage 010 "Calibration: scoring the training set against its own index"
    # Everything here is in-distribution, so the flagged fraction IS the
    # false-positive rate. Expect it to land near (100 - percentile)%.
    python -m scripts.knn_ood predict \
        --input_type id \
        --embeddings "${RESULTS_DIR}/inference_results.npz" \
        --labels "${DATA_DIR}/final_metadata.csv" \
        --knn_index "${RESULTS_DIR}/knn_index.pkl" \
        --threshold_percentile "${KNN_THRESHOLD_PERCENTILE}" \
        ${KNN_K_GRID:+--k_grid "${KNN_K_GRID}"} \
        --output "${RESULTS_DIR}/knn_id_distances.csv"
fi

if [[ "${RUN_QUERY}" == true ]]; then
    stage 011 "Running the query FASTA through the model"
    # --energy_summary must be passed explicitly: without it the script falls
    # back to hard-coded percentiles calibrated on a different model.
    python -m scripts.trihead.process_trihead_query \
        --query "${QUERY_FASTA}" \
        --output_dir "${RESULTS_DIR}/test_output" \
        --base_model "${BASE_MODEL}" \
        --head_model "${HEAD_MODEL}" \
        --model_path "${RESULTS_DIR}/transformer_model.pth" \
        --model_params "${QUERY_PARAMS}" \
        --inference_mode eval \
        --energy_summary "${RESULTS_DIR}/energy_summary.json" \
        --device cuda
fi

if [[ "${RUN_KNN_QUERY}" == true ]]; then
    stage 012 "Novelty detection on the query (the deployed call)"
    python -m scripts.knn_ood predict \
        --input_type query \
        --embeddings "${RESULTS_DIR}/test_output/query_embeddings.npz" \
        --knn_index "${RESULTS_DIR}/knn_index.pkl" \
        --threshold_percentile "${KNN_THRESHOLD_PERCENTILE}" \
        --max_k "${KNN_MAX_K}" \
        ${KNN_K_GRID:+--k_grid "${KNN_K_GRID}"} \
        --output "${RESULTS_DIR}/test_output/knn_query_distances.csv"
fi

if [[ "${RUN_EXPORT}" == true ]]; then
    stage 013 "Exporting a pickle-free index for publication"
    python -m scripts.knn_ood export \
        --knn_index "${RESULTS_DIR}/knn_index.pkl" \
        --output "${RESULTS_DIR}/knn_index.npz" \
        --threshold_percentile "${KNN_THRESHOLD_PERCENTILE}"
fi

if [[ "${RUN_EMBED_TEST}" == true ]]; then
    stage 014 "ProkBERT chunk embeddings for the test split"
    # The training pipeline only embeds train.fasta. The LR baseline needs the
    # test contigs in the same 384-d chunk space, so embed them once here.
    python -m scripts.embed_transformer \
        --fasta "${DATA_DIR}/test.fasta" \
        --out_dir "${RESULTS_DIR}/test_embeddings_chunked" \
        --model_name "${BASE_MODEL}" \
        --chunk_size 4000 --stride_ratio 0.5 --device cuda
fi

if [[ "${RUN_BASELINE_LR}" == true ]]; then
    stage 015 "Logistic-regression baseline (holdout, same test split as ALLCAPS)"
    # Fit on the whole training split, score the held-out test split — the same
    # protocol behind the ALLCAPS test numbers, so the two are comparable.
    # Drop --test_* to fall back to grouped cross-validation on the train split,
    # which measures something else and should not go in the same table.
    python -m scripts.eval_baseline_lr \
        --embedding_dir "${RESULTS_DIR}/base_embeddings_chunked" \
        --labels "${DATA_DIR}/final_metadata.csv" \
        --test_embedding_dir "${RESULTS_DIR}/test_embeddings_chunked" \
        --test_labels "${DATA_DIR}/test_metadata.csv" \
        --output_dir "${RESULTS_DIR}/eval-lr" \
        --model_params '{"pooling": "mean"}'
fi

echo "[INFO] done -> ${RESULTS_DIR}"
