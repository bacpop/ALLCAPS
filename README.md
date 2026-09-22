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
```

Or, on a GPU machine, `conda env create -f environment.yml && conda activate all-caps` — that
route pulls a CUDA-matched PyTorch, which pip cannot do reliably.

## Serotype a locus with a trained model

The shortest useful path — no training, no Snakemake. Given a *cps* locus FASTA, a checkpoint
and a fitted kNN index:

```bash
cd src

# Serotype calls + the pooled embeddings
python -m scripts.trihead.process_trihead_query \
    --query        my_loci.fasta \
    --model_path   transformer_model.pth \
    --energy_summary energy_summary.json \
    --output_dir   out/ \
    --device       cpu

# The deployed novelty call
python -m scripts.knn_ood predict \
    --input_type query \
    --embeddings out/query_embeddings.npz \
    --knn_index  knn_index.npz \
    --threshold_percentile 95.0 \
    --max_k 5 \
    --output out/knn_query_distances.csv
```

ProkBERT is downloaded from the Hub on first run. If you are starting from whole assemblies
rather than cut loci, run `scripts.data_locus_cutter` first — it cuts between the `dexB`/`aliA`
flanks shipped in [assets/](assets/).

To train your own model instead, see **[TRAINING.md](TRAINING.md)**.

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
| `split_fastas`, `split_metadata` | Positionally aligned lists of sequence/label sources to merge before splitting |
| `split_ratios` | Train fraction, counted in **samples** not contigs |
| `serotypes` | Serotypes to hold out for LOO; leave empty to skip those rules |
| `knn_k`, `knn_threshold_percentile`, `knn_max_k` | Novelty detector; defaults `1`, `95.0`, `5` |
| `model_params` | JSON **string** of model hyperparameters (must include `embedding_dim`) — see [TRAINING.md](TRAINING.md) |

## Pipeline rules

`locus_cutting` → `labels_preprocessing` → `train_test_split` → `embed_base` →
`labels_postprocessing` → `train_model` → `embed_chunks` → evaluation
(`visualize_embeddings`, `capsule_classification`, `serotype_classification`) →
`novel_detection` → `knn_fit` → `knn_predict_id` / `knn_predict_query`.

`train_model_loo`, `embed_chunks_loo` and `serotype_classification_loo` repeat training and
evaluation with one serotype withheld, and only materialise when `serotypes` is populated.
See [src/README.md](src/README.md) for the rule-by-rule breakdown and a one-line description of
every module. On a cluster, [jobs/run_snakemake.sh](jobs/run_snakemake.sh) submits the whole DAG
and [jobs/allcaps_slurm.sh](jobs/allcaps_slurm.sh) drives it stage by stage.

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
- `assets/` — the `dexB`/`aliA` flanking genes used to cut the locus.
- `jobs/` — the two cluster driver scripts.


Modules run as packages from `src/`, e.g. `python -m scripts.knn_ood predict ...`.

## Weights & Biases (optional)

```bash
export WANDB_MODE=offline    # during the run
wandb sync --sync-all        # afterwards
```

## Contact
[Alireza Tajmirriahi](https://github.com/AlirezaT99) and [Sam Horsfield](https://github.com/samhorsfield96) — please open an issue for questions, bugs, or feature requests.

## License

**ALLCAPS — both the code in this repository and the trained weights — is released under the
[MIT License](LICENSE).** That grant covers only what we made.

> ### ⚠️ Third-party dependency notice
>
> **ALLCAPS cannot run on its own.** It requires
> [ProkBERT-mini-long](https://huggingface.co/neuralbioinfo/prokbert-mini-long) at inference time —
> every sequence is embedded by ProkBERT before ALLCAPS sees it — and those weights are licensed
> [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/), which **prohibits commercial
> use**.
>
> Our MIT license conveys **no rights whatsoever in ProkBERT**. You are responsible for complying
> with ProkBERT's license independently. In practice: although ALLCAPS itself is MIT, you cannot run
> this pipeline commercially without separate permission from the ProkBERT authors.

| Component | License | Obtained from |
|---|---|---|
| ALLCAPS source code | MIT | this repository |
| ALLCAPS trained weights | MIT | released separately |
| ProkBERT-mini-long **weights** | **CC-BY-NC-4.0** | downloaded from the Hub at runtime |
| ProkBERT source code | MIT | [nbrg-ppcu/prokbert](https://github.com/nbrg-ppcu/prokbert) |

No ProkBERT weights are contained in or redistributed by this repository or by the ALLCAPS
checkpoint. ProkBERT is loaded frozen via `AutoModel.from_pretrained` and used purely as a feature
extractor; every parameter in the ALLCAPS checkpoint was initialised and trained here.

If you use ALLCAPS, please cite both ALLCAPS and ProkBERT (Ligeti et al. 2024,
*Frontiers in Microbiology* 14:1331233,
[doi:10.3389/fmicb.2023.1331233](https://doi.org/10.3389/fmicb.2023.1331233)) — attribution is
required under ProkBERT's license.
