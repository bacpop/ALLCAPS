# Pipeline overview

The Snakemake workflow for embedding, training, evaluating and querying pneumococcal capsular
loci. `Snakefile` is the authoritative definition — this file is a map of it.

Run from **this directory** (the Snakefile resolves `scripts/` relative to itself via
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

## Cluster runs

The LOO array, the AtB query and the k-sweep are driven by SLURM scripts in
[`../jobs/`](../jobs/) rather than by Snakemake, because they fan out across serotypes and
reuse a shared embedding directory. [`../jobs/README.md`](../jobs/README.md) is a dated log of
what was run and why.
