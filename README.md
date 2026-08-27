# ALLCAPS — pneumococcal *cps* locus embedding, serotyping and novel-serotype detection

Tools and workflows to **embed, classify, and detect novel pneumococcal capsular biosynthetic
loci (CBLs)** from genome assemblies. Given a locus, the model predicts

1. whether it is a *cps* locus at all,
2. its serotype (and genogroup), and
3. whether that serotype is **novel** — unseen during training.

## How it works

A *cps* locus is extracted from an assembly using the flanking `dexB`/`aliA` genes, split into
4 kbp chunks with 50% overlap, and embedded with **ProkBERT**. Those chunk embeddings pass
through a learned `TransformerEncoder` (with positional embeddings), are masked-mean-pooled into
a single 128-d L2-normalised locus embedding, and feed **three classification heads**: capsule
y/n, serotype, and genogroup. The model class is `TransformerTriHeadLR`
([src/scripts/models.py](src/scripts/models.py)).

**Novel-serotype detection** compares the pooled embedding to every training *cps* embedding by
**cosine distance**. A locus is called novel when the distance to its nearest training neighbour
exceeds a threshold set at the **95th percentile of the training leave-one-out 1-NN distances**
— a threshold derived without ever looking at novel data. The report also names the closest
known serotype, so a novel locus can be placed in a neighbourhood rather than just rejected.

An **energy** score (`E = −T·logsumexp(logits/T)`) is retained as a reference baseline and
reported alongside. Evaluated over 98 leave-one-serotype-out folds, kNN was the better detector
on every fold-level metric, so it is the deployed one.

> Distances are computed in **float64**. In float32, sklearn's cosine (`1 − x·y`) cancels below
> machine epsilon for the many near-identical loci in this dataset, quantising ~73% of
> in-distribution distances toward zero.

## Install

```bash
conda create -n pneumo python=3.10 -y && conda activate pneumo
pip install -r requirements.txt
pip install snakemake            # not in requirements.txt
```

## Run the pipeline

```bash
cp src/config.yaml.template config.yaml   # then edit the paths
cd src                                    # the Snakefile resolves scripts relative to itself
snakemake -n  --configfile ../config.yaml # dry-run the DAG first
snakemake --cores 4 --configfile ../config.yaml
```

`config.yaml` and `data/` are gitignored.

### Configuration

| Key | Meaning |
|---|---|
| `results_dir`, `data_dir` | Output and intermediate locations |
| `infiles` | Text file listing one raw assembly FASTA path per line |
| `metadata` | Sample metadata; needs a sample id, contig id, serotype and `Is_capsule` |
| `locus_cutter_query` | FASTA of the flanking genes used to cut the locus |
| `query_path` | Sequences to serotype / screen for novelty |
| `serotypes` | Serotypes to hold out for LOO; leave empty to skip those rules |
| `knn_k`, `knn_threshold_percentile`, `knn_max_k` | Novelty detector; defaults `1`, `95.0`, `5` |
| `model_params` | JSON of model hyperparameters (must include `embedding_dim`) |

## Pipeline rules

`locus_cutting` → `labels_preprocessing` → `train_test_split` → `embed_base` →
`labels_postprocessing` → `train_model` → `embed_chunks` → evaluation
(`visualize_embeddings`, `capsule_classification`, `serotype_classification`) →
`novel_detection` → `knn_fit` → `knn_predict_id` / `knn_predict_query`.

`train_model_loo`, `embed_chunks_loo` and `serotype_classification_loo` repeat training and
evaluation with one serotype withheld, and only materialise when `serotypes` is populated.
See [src/README.md](src/README.md) for the rule-by-rule breakdown.

## Outputs

| File | Contents |
|---|---|
| `query_results.csv` | Per-locus serotype and genogroup calls, confidence, and the **energy** novelty flag (`is_novel_energy`) |
| `knn_query_distances.csv` | The **deployed** novelty call: `is_novel_knn`, distance, and the nearest known serotype |
| `knn_query_distances_topk.csv` | Long-format top-K neighbours per locus (`rank`, neighbour id, serotype, genogroup, distance) |
| `knn_id_distances.csv` | The same scores on training data — every serotype is in-distribution here, so the flagged fraction is the false-positive rate |
| `classification_report.txt`, `confusion_matrix_df.csv` | Closed-set serotype performance |

## Data model

Every metadata row and FASTA record is one **contig**, keyed `Public_ID#Contig_ID`
(non-capsular records keep a `NONCBL#` prefix on `Public_ID`). One **sample** is one assembly
and may span several contigs — a *cps* locus is frequently split across two. Metrics in this
repo are computed per contig unless stated otherwise.

The train/test split
([src/scripts/helpers/data_train_test_split.py](src/scripts/helpers/data_train_test_split.py))
groups **by sample, never by contig**, so sibling contigs of one assembly never straddle the
boundary; a runtime assertion enforces it.

## Repository layout

- `src/Snakefile` — the workflow.
- `src/scripts/` — core modules (models, embedding, inference, evaluation, kNN novelty).
- `src/scripts/helpers/` — data preparation, the train/test splitter, novelty sweeps and plots.
- `src/scripts/trihead/` — training, inference and query processing for the deployed model.
- `src/scripts/tests/` — the round-trip sanity check comparing the training and query
  embedding paths. Run it after any change to chunking, pooling or base-model loading.
- `jobs/` — SLURM submission scripts for the cluster runs, with a dated log of what was run
  and why in [jobs/README.md](jobs/README.md).

Modules run as packages from `src/`, e.g. `python -m scripts.knn_ood predict ...`.

## Weights & Biases (optional)

```bash
export WANDB_MODE=offline    # during the run
wandb sync --sync-all        # afterwards
```

## Contributing

Issues and PRs welcome — please include the command you ran, the environment, and any logs.
