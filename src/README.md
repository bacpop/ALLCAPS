# Pipeline overview

The Snakemake workflow for embedding, training, evaluating and querying pneumococcal capsular
loci. `Snakefile` is the authoritative definition — this file is a map of it.

> For serotyping an assembly, or training from a manifest, use the `ALLCAPS` CLI instead —
> see the [README](../README.md). This workflow is for reproducing the published analysis,
> including the leave-one-serotype-out folds the CLI does not cover.

Run from **this directory** (the Snakefile resolves `allcaps/` relative to itself via
`workflow.basedir`):

```bash
snakemake -n --configfile ../config.yaml            # dry run
snakemake --cores 4 --configfile ../config.yaml     # execute
snakemake --forceall --dag | dot -Tpdf > dag.pdf    # visualise
```

## Rules, in dependency order

| # | Rule | Does |
|---|---|---|
| 1 | `locus_cutting` | Cut *cps* loci from raw assemblies using the flanking genes; emits cleaned CBL and non-CBL contigs |
| 2 | `labels_preprocessing` | Clean and normalise the metadata/labels |
| 3 | `train_test_split` | Stratified split **by sample**, never by contig |
| 4 | `embed_base` | ProkBERT chunk embeddings for the training sequences |
| 5 | `labels_postprocessing` | Keep only samples that actually have an embedding; apply `skip_labels` |
| 6 | `train_model` | Train `TransformerTriHeadLR` on the chunked embeddings |
| 7 | `embed_chunks` | Inference pass → `inference_results.npz` (pooled locus embeddings + logits) |
| 8 | `visualize_embeddings` | t-SNE of the embedding space |
| 9 | `capsule_classification` | Capsule vs non-capsule performance |
| 10 | `serotype_classification` | Serotype performance, reports and confusion matrix |
| 11 | `novel_detection` | Run queries through the model → `query_results.csv` (serotype calls + **energy** novelty) and `query_embeddings.npz` |
| 12 | `knn_fit` | Build the nearest-neighbour index from the training embeddings → `knn_index.pkl` |
| 13 | `knn_predict_id` | Score training data against its own index (self-match excluded) → false-positive rate at the chosen threshold |
| 14 | `knn_predict_query` | **The deployed novelty call** on the queries, plus the top-K neighbour report |
| — | `train_model_loo`, `embed_chunks_loo`, `serotype_classification_loo` | Leave-one-serotype-out repeats; only materialise when `serotypes` is populated in config |

## Two novelty detectors

`novel_detection` (rule 11) reports the **energy** score — retained as a reference baseline.
`knn_predict_query` (rule 14) reports **kNN**, which is the deployed detector and the one to
quote. They live in different files on purpose:

- `query_results.csv` → `is_novel_energy`
- `knn_query_distances.csv` → `is_novel_knn`, plus `nn_serotype` (the closest known serotype)

Tuned by `knn_k` (default 1), `knn_threshold_percentile` (95.0) and `knn_max_k` (5) in config.

## Configuration notes

- Copy `config.yaml.template` and set the paths for metadata, `infiles`, flanking genes and
  results directory.
- `model_params` is a JSON **string** and must contain `embedding_dim` — the workflow validates
  this at load time and fails fast if it is missing.
- Populate `serotypes` only when you want LOO targets; each entry adds a full training run.

## Module index

Everything runs as a package from this directory: `python -m allcaps.<module>`.
Once `pip install .` has been run, the modules are importable from anywhere and the
`ALLCAPS` command wraps the common paths.

### `allcaps/cli/` — the ALLCAPS command

| Module | Role |
|---|---|
| [main.py](allcaps/cli/main.py) | The typer app: `predict`, `train`, `knn` |
| [predict.py](allcaps/cli/predict.py) | Assemblies → cut or scan → serotype calls → kNN novelty → one merged table |
| [train.py](allcaps/cli/train.py) | Manifest → every stage below → checkpoint + kNN index + `energy_summary.json` |
| [knn.py](allcaps/cli/knn.py) | Shells out to the two R threshold-tuning scripts |
| [artifacts.py](allcaps/cli/artifacts.py) | Resolves the checkpoint / index / energy table, downloading from the Hub when not given |
| [manifest.py](allcaps/cli/manifest.py) | Parses the input list (plain paths, or CSV/TSV with `path` + `serotype`) |
| [_invoke.py](allcaps/cli/_invoke.py) | Calls each stage in-process with a Namespace built from its own parser |

Every pipeline module exposes `build_parser()`, `parse_args(argv)` and `main(args)`, so the
CLI can drive it without a subprocess and `--help` stays the single source of truth for
defaults.

### `allcaps/` — core

| Module | Does |
|---|---|
| [consts.py](allcaps/consts.py) | Single source of truth for constants: split ratio, column names, `CONTIG_SEP`, model defaults, `MIN_SEROTYPE_COUNT` |
| [utils.py](allcaps/utils.py) | Shared helpers: `collate_fn`, `get_sample_id`, `map_serotype_to_group`, `classify_label_type`, the contrastive losses |
| [logging_config.py](allcaps/logging_config.py) | Central logger setup — use `get_logger(__name__)`, not `print` |
| [models.py](allcaps/models.py) | Model and dataset registries; `TransformerTriHeadLR` (the deployed architecture) plus older unused variants |
| [inference.py](allcaps/inference.py) | Shared inference path: `load_base_model`, `load_trained_model`, `embed_sequence`, `energy_score`. The single source of truth for chunking and pooling |
| [data_locus_cutter.py](allcaps/data_locus_cutter.py) | Cut the *cps* locus out of raw assemblies by aligning the `dexB`/`aliA` flanks (minimap2 via `mappy`) |
| [data_labels_preprocessing.py](allcaps/data_labels_preprocessing.py) | Clean and standardise raw serotype labels; the one entry point so train and eval always see identical label sets |
| [data_labels_postprocessing.py](allcaps/data_labels_postprocessing.py) | Drop metadata rows that have no embedding file; apply `skip_labels` |
| [data_augmentation.py](allcaps/data_augmentation.py) | Embedding-level augmentation used during training (noise, chunk dropout, SpecAugment) |
| [embed_transformer.py](allcaps/embed_transformer.py) | ProkBERT chunk embeddings for a FASTA → one `.npy` per contig |
| [knn_ood.py](allcaps/knn_ood.py) | **The deployed novelty detector.** `fit` / `predict` / `export`; cosine distance to the k-th nearest training locus |
| [eval_cbl_classifier.py](allcaps/eval_cbl_classifier.py) | Capsule vs non-capsule performance |
| [eval_serotype_classifier.py](allcaps/eval_serotype_classifier.py) | Closed-set serotype performance: report, confusion matrix, confidence stats |
| [eval_test_performance.py](allcaps/eval_test_performance.py) | Serotype/genogroup performance from a finished `query_results.csv` |
| [eval_visualize_embeddings.py](allcaps/eval_visualize_embeddings.py) | 2D embedding maps (t-SNE / UMAP) |
| [eval_baseline_lr.py](allcaps/eval_baseline_lr.py) | Logistic-regression baseline on raw ProkBERT embeddings — the reference the learned encoder has to beat. Pass `--test_labels`/`--test_embedding_dir` to score the held-out test split (comparable to ALLCAPS); without them it runs sample-grouped CV on the train split |
| [eval_svm_embeddings.py](allcaps/eval_svm_embeddings.py) | Linear/SVM separability probe on learned embeddings |

### `allcaps/trihead/` — the deployed model

| Module | Does |
|---|---|
| [train_trihead_transformer.py](allcaps/trihead/train_trihead_transformer.py) | Train `TransformerTriHeadLR`: 5-fold CV then a final refit. See [TRAINING.md](../TRAINING.md) |
| [infer_trihead_transformer.py](allcaps/trihead/infer_trihead_transformer.py) | Run the training set through the trained model → `inference_results.npz` (pooled embeddings) |
| [process_trihead_query.py](allcaps/trihead/process_trihead_query.py) | FASTA → serotype calls + `query_embeddings.npz`, with the energy novelty baseline |

### `allcaps/helpers/` — data prep and novelty analysis

| Module | Does |
|---|---|
| [data_train_test_split.py](allcaps/helpers/data_train_test_split.py) | **The splitter.** Merges FASTA/metadata sources and splits by sample, never by contig; asserts no `Public_ID` straddles the boundary |
| [sequence_sampler.py](allcaps/helpers/sequence_sampler.py) | Pull the sequences of chosen serotypes out of a FASTA — builds the held-out query set for each LOO fold |
| [knn_k_sweep.py](allcaps/helpers/knn_k_sweep.py) | Sweep k across LOO folds: is the AUROC ceiling set by k, or by the features? |
| [plot_knn_threshold_accuracy.py](allcaps/helpers/plot_knn_threshold_accuracy.py) | Regenerate the per-serotype threshold-accuracy plots from a finished sweep |
| [analyze_knn_nearest_neighbor.py](allcaps/helpers/analyze_knn_nearest_neighbor.py) | The k=1 diagnostic: when a novel locus isn't flagged, which known serotype did it land on? |
| [eval_novelty_performance.py](allcaps/helpers/eval_novelty_performance.py) | Novelty metrics for a single fold |
| [aggregate_novelty_performance.py](allcaps/helpers/aggregate_novelty_performance.py) | Aggregate novelty performance across all LOO folds → the summary report and tables |
| [statistical_summary.py](allcaps/helpers/statistical_summary.py) | F1 distribution plots and the ALLCAPS-vs-baseline statistical comparison |

### `allcaps/tests/`

| Module | Does |
|---|---|
| [sanity_check_roundtrip.py](allcaps/tests/sanity_check_roundtrip.py) | Compares the training and query embedding paths on the same input. Run it after any change to chunking, pooling or `load_base_model` |

## Cluster runs

Two driver scripts live in [`../jobs/`](../jobs/):

| Script | Use it when |
|---|---|
| [`run_snakemake.sh`](../jobs/run_snakemake.sh) | Normal case — submit the whole DAG as one SLURM job |
| [`allcaps_slurm.sh`](../jobs/allcaps_slurm.sh) | You need to re-run one stage against an existing results directory, or fan out LOO folds, without Snakemake rebuilding downstream targets |

Everything else in `jobs/` is local run output and is gitignored.
