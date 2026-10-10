# Training ALLCAPS

How to train the TriHead model from scratch, and how to reproduce the released checkpoint.

The quickest route is the CLI, which runs every stage below from a manifest of assemblies:

```bash
ALLCAPS train --input manifest.csv --output run/ --device cuda
```

where `manifest.csv` has a `path` column and a `serotype` column, one row per assembly:

```csv
path,serotype
/data/ERR1788086.fasta,19A
/data/ERR714669.fasta.gz,3
```

It cuts the locus, cleans the labels, splits by sample, embeds with ProkBERT, trains,
fits the novelty index and calibrates it — then writes the three artifacts
`ALLCAPS predict` consumes into `run/`:

| File | Used by `predict` as |
|---|---|
| `transformer_model.pth` | `--model` |
| `knn_index.npz` | `--knn-index` |
| `energy_summary.json` | `--energy-summary` |

```bash
ALLCAPS predict --input samples.txt --extract align --output out/ \
    --model run/transformer_model.pth \
    --knn-index run/knn_index.npz \
    --energy-summary run/energy_summary.json
```

`--resume` skips stages whose outputs already exist. Hyperparameters not exposed as flags
go through `--model-params '{"..": ..}'`, merged over the released defaults in
`RELEASED_MODEL_PARAMS` ([src/allcaps/cli/train.py](src/allcaps/cli/train.py)); the rest of
this file documents what those values mean.

This file is about the `train_model` stage specifically — its inputs, its hyperparameters,
and the handful of choices that silently change what you get. Use the Snakemake route
(see the [README](README.md)) to reproduce the published analysis, including the
leave-one-serotype-out folds.

---

## 1. What training consumes

Training does **not** read FASTA. It reads pre-computed ProkBERT chunk embeddings plus a
metadata table, both produced by earlier pipeline stages:

| Input | Produced by | Shape / contract |
|---|---|---|
| `--embedding_dir` | `embed_base` ([embed_transformer.py](src/allcaps/embed_transformer.py)) | One `<Public_ID>#<Contig_ID>.npy` per contig, each `(n_chunks, 384)` |
| `--labels` | `labels_postprocessing` ([data_labels_postprocessing.py](src/allcaps/data_labels_postprocessing.py)) | CSV/TSV with exactly `Public_ID, Contig_ID, Serotype, Is_capsule` |

`final_metadata.csv` looks like this — one row per **contig**, not per sample:

```csv
Public_ID,Contig_ID,Serotype,Is_capsule
ERR9796441,17,33G,1
NONCBL#ERR9796441,3,Non-typeable,0
```

There is **no genogroup column**. Genogroups are derived at train time from `Serotype` by
`map_serotype_to_group` ([utils.py](src/allcaps/utils.py)), so the genogroup label set is a
function of the serotypes present in your data, not something you supply.

### ⚠️ The embedding directory layout depends on `dataset_name`

This is the most common way to get a silently bad model:

| `dataset_name` | Expected layout |
|---|---|
| `multidomain_chunked` (**used by the released model**) | **Flat**: all `.npy` directly in `--embedding_dir` |
| `contrastive_chunked` (the code default) | **Nested**: `cbl/*.npy` and `non-cbl/*.npy` subdirectories |

Pick the wrong one and every lookup misses. You get a `WARNING: N sample_ids do not have a
corresponding embedding file`, training proceeds on whatever remains, and the result is
garbage. Check that warning line on every run — it should report 0 missing.

---

## 2. Reproducing the released checkpoint

`ALLCAPS train`'s defaults **are** this recipe, so the CLI route in the header reproduces
it. The command below is the module-level equivalent, transcribed from
[jobs/new-gps-final.sh](jobs/new-gps-final.sh) — the job that produced the released
checkpoint:

```bash
cd src
WANDB_MODE=offline python -m allcaps.trihead.train_trihead_transformer \
    --embedding_dir "${RESULTS_DIR}/base_embeddings_chunked" \
    --labels        "${DATA_DIR}/final_metadata.csv" \
    --output        "${RESULTS_DIR}/transformer_model.pth" \
    --device cuda --epochs 100 --batch_size 128 --lr 0.001 \
    --labeled_only --hierarchical_loss \
    --model_params '{"temperature": 0.07, "k_folds": 5, "num_layers": 1, "alpha": 0,
                     "dataset_name": "multidomain_chunked"}' \
    --aug_noise_std     0.01 \
    --aug_chunk_dropout 0.1  \
    --aug_spec_freq     0.5  \
    --aug_spec_width    16   \
    --aug_n_views       2
```

Note what is **absent** from `--model_params`: no `weight_fine`, `weight_coarse`,
`weight_sero` or `weight_geno`. The released run passed none of them, so they resolved to
the code defaults — 1.0, 0.5, 2 and **1**. An earlier version of this section asserted
`weight_coarse: 0.4` and `weight_geno: 0`; both were wrong, and §4 covers what that means
for the genogroup head.

What *is* load-bearing, and **not** a code default:

- **`alpha: 0`** — the contrastive loss is **off**. The code default is
  `DEFAULT_CONTRASTIVE_LOSS_RATIO = 0.5`. The released model is trained by cross-entropy alone.
  (`--hierarchical_loss` still selects *which* contrastive loss would be used; with `alpha: 0`
  it is constructed and then multiplied by zero, so it has no effect.)
- **`dataset_name: multidomain_chunked`** — flat embedding directory; see §1.
- **The five `--aug_*` flags** — the module defaults are all zero (and `aug_n_views = 1`),
  which disables embedding augmentation outright via the `aug_enabled` check in
  `train_trihead_transformer.main`. The released model was trained with augmentation on, so
  omitting these flags trains a different model.
- **`--batch_size 128` and `--lr 0.001`** — the module's own defaults are
  `DEFAULT_BATCH_SIZE = 32` (an inference batch size, shared with the eval modules) and
  `DEFAULT_LR = 2e-5` (the contrastive-era learning rate). Both must be passed explicitly on
  the module route. `ALLCAPS train` defaults to the released values instead; see below.

The CLI keeps the same recipe in two dicts in
[src/allcaps/cli/train.py](src/allcaps/cli/train.py):

| | Holds |
|---|---|
| `RELEASED_MODEL_PARAMS` | what goes into `--model_params`: `temperature`, `k_folds`, `num_layers`, `nhead`, `alpha: 0`, `random_state`, `embedding_dim`, `output_dim`, `dataset_name` |
| `RELEASED_TRAIN_DEFAULTS` | the plain arguments: `epochs`, `batch_size`, `lr`, and the five `aug_*` values |

Both are the defaults of `ALLCAPS train`, and every one is overridable
(`--model-params`, `--lr`, `--aug-noise-std`, …). Keep them in step with the command above
if you ever retrain the released model.

> ⚠️ The `aug_*` values live in `RELEASED_TRAIN_DEFAULTS` because they are plain parser
> arguments, **not** `model_params` keys — `--model-params '{"aug_noise_std": …}'` is
> silently ignored. They were missing from the CLI until Oct 2026, during which
> `ALLCAPS train` trained with augmentation disabled while claiming to reproduce the
> release.

### Hyperparameters, in full

| | Released value | Source of default |
|---|---|---|
| Epochs | 100 | `--epochs` |
| Batch size | 128 | `--batch_size` |
| Learning rate | 1e-3, AdamW | `--lr` |
| Encoder layers / heads | 1 / 4 | `num_layers`, `nhead` |
| Base embedding dim | 384 (ProkBERT-mini-long) | `embedding_dim` |
| Locus embedding dim | 128, L2-normalised | `output_dim` |
| CV folds | 5, stratified | `k_folds` |
| Early-stopping patience | 10 (CV folds only) | `--early_stopping` |
| Random state | 42 | `random_state` |
| Capsule-head loss weight | 1 (implicit) | — |
| Serotype-head loss weight | 2 (code default) | `weight_sero` |
| Genogroup-head loss weight | **1** (code default — the head was trained and still collapsed; see §4) | `weight_geno` |
| Hierarchical fine / coarse weights | 1.0 / 0.5 (code defaults) | `weight_fine`, `weight_coarse` |
| Contrastive loss weight | **0** | `alpha` |
| Augmentation | noise 0.01, chunk dropout 0.1, SpecAugment p=0.5 width 16, 2 views | `--aug_*` |

Class weights are computed **per fold, from the training split only**, balanced as
`w_c = N / (C · n_c)` — serotype weights from resolved-serotype samples only, capsule and
genogroup weights from all samples in the split.

---

## 3. What the run does, and what it writes

1. **Label preparation.** Rows whose `Serotype` is missing are dropped (`--labeled_only`);
   `--skip_labels` removes named serotypes entirely (this is how the LOO folds are built).
2. **Label typing.** Each label is classified as a resolved *serotype*, *serogroup-only*
   (e.g. `Serogroup 24`) or *compound* (e.g. `15B/15C`). Only resolved serotypes contribute
   to the serotype loss; all capsulated samples contribute to the genogroup loss.
3. **Rarity demotion.** Serotypes with fewer than `MIN_SEROTYPE_COUNT` (= 5, overridable via
   `min_serotype_count`) samples are demoted to serogroup-only, so they never become a
   serotype class. This is why a genuinely rare type can surface as *novel* rather than as
   itself at inference.
4. **5-fold stratified CV**, with early stopping per fold → `<output>_cv_summary.json`.
5. **Final refit on all data** for the full `--epochs` (no early stopping), then evaluation
   on the unaugmented data → the saved checkpoint.

The CV stage exists to report honest accuracy; the shipped weights come from the refit, so
CV numbers describe the procedure, not the exact artefact.

### Outputs

| File | Contents |
|---|---|
| `transformer_model.pth` | `model_state_dict`, `serotype_to_idx`, `genogroup_to_idx`, `num_serotypes`, `num_genogroups`, `model_config` |
| `transformer_model_cv_summary.json` | Per-fold best epoch, val loss, and capsule / serotype / genogroup accuracy, plus means |

The checkpoint is self-describing: `model_config` carries everything
[load_trained_model](src/allcaps/inference.py#L95) needs to rebuild the architecture, and the
two `*_to_idx` dicts are the label vocabulary. Nothing external is required to load it.

---

## 4. The genogroup head does not work

The architecture has three heads; the third one is broken. In the released model the
genogroup head sits at **0.8% accuracy over 53 classes**, below the 1.9% you would get by
chance. It is still present in the checkpoint and still produces logits, and those logits
are **meaningless**.

> ⚠️ **The cause is not known.** This section previously said the head was deliberately
> disabled with `weight_geno: 0`. Two independent records say otherwise: the job script
> that produced the checkpoint ([jobs/new-gps-final.sh](jobs/new-gps-final.sh)) passes no
> `weight_geno`, and the W&B record of the released run
> (`offline-run-20260806_131600-wmizddb0`) shows `model_params` was only
> `{"temperature": 0.07, "k_folds": 5, "num_layers": 1, "alpha": 0, "dataset_name":
> "multidomain_chunked"}` — no `weight_geno` key at all, so it resolved to
> `DEFAULT_WEIGHT_GENO = 1` and the head *was* trained, with a non-zero weight, and still
> collapsed. Setting `weight_geno: 1` therefore will not fix it. Treat this as an open
> bug, not a design decision.

Consequences:

- **Do not use `pred_genogroup`** from `query_results.csv`. It is untrained-head noise.
- The genogroup reported by the novelty detector (`nn_genogroup` in
  `knn_query_distances.csv`) is *not* from this head — it is derived from the nearest
  neighbour's serotype via `map_serotype_to_group`, and is fine to use.
- Retraining with `weight_geno: 1` will **not** fix it — that is already what the released
  run did. Diagnosing the head is open work.

---

## 5. Footguns

**Training honours `--device`, but you want CUDA anyway.** The train and eval loops follow
the model's own device rather than calling `.cuda()`, so `--device cpu` runs — it is just far
too slow to be useful at this scale. `ALLCAPS train --device auto` (the default) picks CUDA
when it is available.

**wandb is optional and off by default.** Metrics go through
[allcaps.tracking](src/allcaps/tracking.py), which no-ops unless asked, so `wandb` need not
be installed at all. To turn it on:

```bash
pip install 'allcaps[wandb]'
ALLCAPS train --input manifest.csv --output run/ --wandb    # or --wandb on the module
export WANDB_MODE=offline     # writes to ./wandb/, sync later with `wandb sync --sync-all`
```

The project name is hard-coded as `WANDB_PROJECT_NAME` at the top of the training module.
Never `import wandb` in pipeline code — go through `tracking`.

**`--labeled_only` is not optional in practice.** Without it, unlabelled rows become a
`Non-typeable` class and pollute the serotype vocabulary.

**Splitting is done upstream, and it is grouped by sample.**
[data_train_test_split.py](src/allcaps/helpers/data_train_test_split.py) splits by
`Public_ID`, never by contig, so sibling contigs of one assembly never straddle the boundary.
Do not re-split inside training code; a contig-level split leaks near-duplicates and inflates
every metric.

**Costs.** On an NVIDIA A100 80GB, the released run took roughly
4h to train, on top of 3h to compute base embeddings.

---

## 6. After training

Training alone does not give you a usable novelty detector. The deployed system needs the
kNN index fitted on the trained model's own embeddings. `ALLCAPS train` already runs all of
this (stages `embed_chunks` → `serotype_classification` → `knn_fit` → `knn_predict_id` →
`knn_export`); the commands below are the same stages by hand, for a run you drove module by
module:

```bash
# 1. Embed the training set through the trained model
python -m allcaps.trihead.infer_trihead_transformer \
    --embeddings_dir "${RESULTS_DIR}/base_embeddings_chunked" \
    --labels "${DATA_DIR}/final_metadata.csv" \
    --model "${RESULTS_DIR}/transformer_model.pth" \
    --output "${RESULTS_DIR}/inference_results.npz" \
    --device cuda --labeled_only

# 2. Fit the novelty index (k=1, cosine)
python -m allcaps.knn_ood fit \
    --embeddings "${RESULTS_DIR}/inference_results.npz" \
    --labels "${DATA_DIR}/final_metadata.csv" \
    --output "${RESULTS_DIR}/knn_index.pkl" \
    --k 1 --distance_metric cosine

# 3. Calibrate: score the training set against its own index. The flagged
#    fraction is the false-positive rate and should land near 100 - percentile.
python -m allcaps.knn_ood predict \
    --input_type id \
    --embeddings "${RESULTS_DIR}/inference_results.npz" \
    --labels "${DATA_DIR}/final_metadata.csv" \
    --knn_index "${RESULTS_DIR}/knn_index.pkl" \
    --threshold_percentile 95.0 \
    --output "${RESULTS_DIR}/knn_id_distances.csv"
```

Then verify the two embedding paths agree before trusting any query result:

```bash
python -m allcaps.tests.sanity_check_roundtrip ...
```

To publish or share the index, export it without the pickle:

```bash
python -m allcaps.knn_ood export \
    --knn_index "${RESULTS_DIR}/knn_index.pkl" \
    --output    "${RESULTS_DIR}/knn_index.npz" \
    --threshold_percentile 95.0
```

This writes plain arrays plus a `knn_index_config.json` sidecar (override with
`--config_output`) carrying the resolved threshold, and round-trips bit-identically.
`KnnOOD.load` accepts either format.

> The published Hugging Face repo names that sidecar **`knn_config.json`**, because that is
> what the standalone `modeling_allcaps.py` there reads. Rename it on upload, or pass
> `--config_output knn_config.json`. `ALLCAPS predict` does not read it at all — it
> recomputes the threshold from the index's own leave-one-out distances via
> `--threshold-percentile`, which is why the two agree.

---

## 7. Leave-one-serotype-out training

Each LOO fold is a full retrain with one serotype removed from the label set:

```bash
python -m allcaps.trihead.train_trihead_transformer ... --skip_labels 19A
```

There is no `ALLCAPS train` flag for this — the LOO folds are a research path, driven by
the module directly or by Snakemake. That serotype's genomes then become the query set for
novelty evaluation — they are, by
construction, a serotype the model has never seen. 98 such folds back the reported novelty
numbers. Snakemake expands these from `config["serotypes"]`; on a cluster, drive them as a
job array (see [jobs/allcaps_slurm.sh](jobs/allcaps_slurm.sh)).
