# Training ALLCAPS

How to train the TriHead model from scratch, and how to reproduce the released checkpoint.

The [README](README.md) covers running the pipeline end to end; this file is about the
`train_model` stage specifically — its inputs, its hyperparameters, and the handful of
choices that silently change what you get.

---

## 1. What training consumes

Training does **not** read FASTA. It reads pre-computed ProkBERT chunk embeddings plus a
metadata table, both produced by earlier pipeline stages:

| Input | Produced by | Shape / contract |
|---|---|---|
| `--embedding_dir` | `embed_base` ([embed_transformer.py](src/scripts/embed_transformer.py)) | One `<Public_ID>#<Contig_ID>.npy` per contig, each `(n_chunks, 384)` |
| `--labels` | `labels_postprocessing` ([data_labels_postprocessing.py](src/scripts/data_labels_postprocessing.py)) | CSV/TSV with exactly `Public_ID, Contig_ID, Serotype, Is_capsule` |

`final_metadata.csv` looks like this — one row per **contig**, not per sample:

```csv
Public_ID,Contig_ID,Serotype,Is_capsule
ERR9796441,17,33G,1
NONCBL#ERR9796441,3,Non-typeable,0
```

There is **no genogroup column**. Genogroups are derived at train time from `Serotype` by
`map_serotype_to_group` ([utils.py](src/scripts/utils.py)), so the genogroup label set is a
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

```bash
cd src
WANDB_MODE=offline python -m scripts.trihead.train_trihead_transformer \
    --embedding_dir "${RESULTS_DIR}/base_embeddings_chunked" \
    --labels        "${DATA_DIR}/final_metadata.csv" \
    --output        "${RESULTS_DIR}/transformer_model.pth" \
    --device cuda --epochs 100 --batch_size 128 --lr 0.001 \
    --labeled_only --hierarchical_loss \
    --model_params '{"embedding_dim": 384, "output_dim": 128, "num_layers": 1, "nhead": 4,
                     "k_folds": 5, "random_state": 42, "temperature": 0.07,
                     "weight_fine": 1, "weight_coarse": 0.4,
                     "alpha": 0, "weight_sero": 2, "weight_geno": 0,
                     "dataset_name": "multidomain_chunked"}' \
    --aug_noise_std     0.01 \
    --aug_chunk_dropout 0.1  \
    --aug_spec_freq     0.5  \
    --aug_spec_width    16   \
    --aug_n_views       2
```

Three of those values are load-bearing and are **not** the code defaults:

- **`alpha: 0`** — the contrastive loss is **off**. The code default is
  `DEFAULT_CONTRASTIVE_LOSS_RATIO = 0.5`. The released model is trained by cross-entropy alone.
  (`--hierarchical_loss` still selects *which* contrastive loss would be used; with `alpha: 0`
  it is constructed and then multiplied by zero, so it has no effect.)
- **`weight_geno: 0`** — the **genogroup head is not trained**; see §4.
- **`dataset_name: multidomain_chunked`** — flat embedding directory; see §1.

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
| Serotype-head loss weight | 2 | `weight_sero` |
| Genogroup-head loss weight | **0** | `weight_geno` |
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
[load_trained_model](src/scripts/inference.py#L95) needs to rebuild the architecture, and the
two `*_to_idx` dicts are the label vocabulary. Nothing external is required to load it.

---

## 4. The genogroup head is dead weight — deliberately

The architecture has three heads, but the released model trains two of them. `weight_geno: 0`
was chosen because the genogroup head added nothing to serotype or novelty performance
(removing it did not measurably hurt either). It is still present in the checkpoint, still
produces logits, and those logits are **meaningless** — in the released model it sits at
**0.8% accuracy over 53 classes**, below the 1.9% you would get by chance.

Consequences:

- **Do not use `pred_genogroup`** from `query_results.csv`. It is untrained-head noise.
- The genogroup reported by the novelty detector (`nn_genogroup` in
  `knn_query_distances.csv`) is *not* from this head — it is derived from the nearest
  neighbour's serotype via `map_serotype_to_group`, and is fine to use.
- If you want a working genogroup head, retrain with `weight_geno: 1`. The code default is 1,
  so you get it by simply omitting the key.

---

## 5. Footguns

**Training requires CUDA.** `--device cuda` is not optional for training: the train and
eval loops call `.cuda()` directly on every batch, so `--device cpu` fails regardless of what
you pass. Inference and query processing do honour `--device cpu`.

**Training requires wandb.** `__main__` calls `wandb.init()` unconditionally. To run without
a network or an account:

```bash
export WANDB_MODE=offline     # writes to ./wandb/, sync later with `wandb sync --sync-all`
# or
export WANDB_MODE=disabled    # no run directory at all
```

The project name is hard-coded as `WANDB_PROJECT_NAME` at the top of the training module.

**`--labeled_only` is not optional in practice.** Without it, unlabelled rows become a
`Non-typeable` class and pollute the serotype vocabulary.

**Splitting is done upstream, and it is grouped by sample.**
[data_train_test_split.py](src/scripts/helpers/data_train_test_split.py) splits by
`Public_ID`, never by contig, so sibling contigs of one assembly never straddle the boundary.
Do not re-split inside training code; a contig-level split leaks near-duplicates and inflates
every metric.

**Costs.** On <!-- TODO: GPU model -->, the released run took roughly
<!-- TODO: hours --> h to train, on top of <!-- TODO: hours --> h to compute base embeddings,
with <!-- TODO: GB --> GB of host RAM and <!-- TODO: GB --> GB for the embedding directory.

---

## 6. After training

Training alone does not give you a usable novelty detector. The deployed system needs the
kNN index fitted on the trained model's own embeddings:

```bash
# 1. Embed the training set through the trained model
python -m scripts.trihead.infer_trihead_transformer \
    --embeddings_dir "${RESULTS_DIR}/base_embeddings_chunked" \
    --labels "${DATA_DIR}/final_metadata.csv" \
    --model "${RESULTS_DIR}/transformer_model.pth" \
    --output "${RESULTS_DIR}/inference_results.npz" \
    --device cuda --labeled_only

# 2. Fit the novelty index (k=1, cosine)
python -m scripts.knn_ood fit \
    --embeddings "${RESULTS_DIR}/inference_results.npz" \
    --labels "${DATA_DIR}/final_metadata.csv" \
    --output "${RESULTS_DIR}/knn_index.pkl" \
    --k 1 --distance_metric cosine

# 3. Calibrate: score the training set against its own index. The flagged
#    fraction is the false-positive rate and should land near 100 - percentile.
python -m scripts.knn_ood predict \
    --input_type id \
    --embeddings "${RESULTS_DIR}/inference_results.npz" \
    --labels "${DATA_DIR}/final_metadata.csv" \
    --knn_index "${RESULTS_DIR}/knn_index.pkl" \
    --threshold_percentile 95.0 \
    --output "${RESULTS_DIR}/knn_id_distances.csv"
```

Then verify the two embedding paths agree before trusting any query result:

```bash
python -m scripts.tests.sanity_check_roundtrip ...
```

To publish or share the index, export it without the pickle:

```bash
python -m scripts.knn_ood export \
    --knn_index "${RESULTS_DIR}/knn_index.pkl" \
    --output    "${RESULTS_DIR}/knn_index.npz" \
    --threshold_percentile 95.0
```

This writes plain arrays plus a `knn_index_config.json` sidecar carrying the resolved
threshold, and round-trips bit-identically. `KnnOOD.load` accepts either format.

---

## 7. Leave-one-serotype-out training

Each LOO fold is a full retrain with one serotype removed from the label set:

```bash
python -m scripts.trihead.train_trihead_transformer ... --skip_labels 19A
```

That serotype's genomes then become the query set for novelty evaluation — they are, by
construction, a serotype the model has never seen. 98 such folds back the reported novelty
numbers. Snakemake expands these from `config["serotypes"]`; on a cluster, drive them as a
job array (see [jobs/allcaps_slurm.sh](jobs/allcaps_slurm.sh)).
