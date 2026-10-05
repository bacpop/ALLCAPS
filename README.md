# ALLCAPS <img src='assets/ALLCAPS-logo.png' align="right" height="100" />
**Pneumococcal *cps* locus serotyping and novel-serotype detection**

Give ALLCAPS a *Streptococcus pneumoniae* genome assembly; it tells you the **serotype**,
and whether that serotype looks **novel** — unlike anything it was trained on.

---

## Install

Pick the environment that matches your machine, then install the package into it:

```bash
# CPU — linux-64, osx-64, osx-arm64
mamba env create -f environment-cpu.yml
conda activate allcaps
pip install .
```

```bash
# NVIDIA GPU (CUDA) — linux-64 only
mamba env create -f environment-gpu.yml
conda activate allcaps-gpu
pip install .
```

| Platform | CPU | NVIDIA GPU |
|---|---|---|
| linux-64 | ✅ | ✅ |
| osx-arm64 (Apple silicon) | ✅ | — no CUDA build of PyTorch exists |
| osx-64 (Intel Mac) | ✅ | — |

Substitute `conda` for `mamba` if you prefer; `mamba` is much faster to solve.

> **`requirements.txt` is for `pip` only.** `torch` is the PyPI package name and does not
> exist on any conda channel, so `mamba create -f requirements.txt` fails with
> `torch >=2.0 does not exist`. Use the environment files above for conda/mamba.

## Serotype an assembly

```bash
ALLCAPS predict --input samples.txt --extract align --output results/
```

`samples.txt` lists one assembly FASTA per line (gzip is fine):

```
/data/ERR1788086.fasta
/data/ERR714669.fasta.gz
```

Or pass paths directly — `ALLCAPS predict ERR1788086.fasta --extract align`.

That writes `results/predictions.tsv` and prints the same table to stdout:

| sample | contig | is_cps | serotype | serotype_confidence | is_novel_knn | nn_serotype | nn_distance | note |
|---|---|---|---|---|---|---|---|---|
| ERR1788086 | 7 | True | 19A | 0.981 | False | 19A | 0.004 | |
| ERR714669 | 5 | True | 3 | 0.874 | True | 11A | 0.312 | NOVEL — unlike any training serotype; nearest known: 11A |

The model weights and the novelty index download automatically on first run and are
cached under `~/.cache/allcaps` (override with `ALLCAPS_CACHE`). ProkBERT, the base
model, is fetched from the Hub the same way.

### `--extract` — how the *cps* locus is found

Required, because it is a real choice:

| | What it does | When |
|---|---|---|
| `align` | Cuts between the `dexB`/`aliA` flanking genes with minimap2. | **Use this.** Fast, and matches how the model was trained. |
| `scan` | Embeds rolling windows across the whole assembly; no flanking genes needed. | Only when the flanks are genuinely absent. Orders of magnitude slower — hours per genome on CPU. |

Assemblies where neither flank is found appear in the output with
`note = no cps locus found`, rather than silently vanishing from the table.

### Useful options

| Option | Meaning |
|---|---|
| `--device cuda\|cpu\|auto` | Default `auto` — CUDA when available, else CPU. |
| `--model`, `--knn-index`, `--energy-summary` | Use your own artifacts instead of the released ones — e.g. from `ALLCAPS train`. |
| `--offline` | Never reach the network; requires the three flags above. |
| `--threshold-percentile` | Novelty threshold, as a percentile of training leave-one-out 1-NN distances. Default `95`. |
| `--max-k` | Neighbours reported per locus (default 5), written to `neighbours_topk.csv`. |
| `--cutoff` | Minimum alignment-length fraction for a flank hit (default `0.7`). Lower it if loci are being missed. |
| `--quiet`, `--keep-intermediates` | Suppress the stdout table; keep cut loci and raw CSVs. |

### Output columns

| Column | Meaning |
|---|---|
| `is_cps` | Does the model think this is a capsular locus at all? When `False`, the serotype call carries no weight. |
| `serotype`, `serotype_confidence` | The closed-set call and its softmax confidence. |
| `genogroup` | Predicted genogroup. See the note in [TRAINING.md](TRAINING.md) — this head does not work well. |
| `is_novel_knn` | **The deployed novelty call.** `True` means the locus is further from every training locus than the threshold. |
| `nn_serotype`, `nn_distance` | The closest known serotype and its cosine distance — where to place a novel locus, rather than just rejecting it. |
| `is_novel_energy`, `energy` | A reference baseline, reported for comparison. kNN is the deployed detector. |
| `note` | Plain-language caveats: not a *cps* locus, novel, or no locus found. |

---

## How it works

A *cps* locus is cut from the assembly using the flanking `dexB`/`aliA` genes, split into
4 kbp chunks with 50% overlap, and embedded with **ProkBERT**. Those chunk embeddings pass
through a learned `TransformerEncoder` (with positional embeddings), are masked-mean-pooled
into a single 128-d L2-normalised locus embedding, and feed **three classification heads**:
capsule y/n, serotype, and genogroup. The model class is `TransformerTriHeadLR`
([src/allcaps/models.py](src/allcaps/models.py)).

**Novel-serotype detection** compares the pooled embedding to every training *cps*
embedding by **cosine distance**. A locus is called novel when the distance to its nearest
training neighbour exceeds a threshold set at the **95th percentile of the training
leave-one-out 1-NN distances** — a threshold derived without ever looking at novel data.

An **energy** score (`E = −T·logsumexp(logits/T)`) is retained as a reference baseline.
Evaluated over 98 leave-one-serotype-out folds, kNN was the better detector on every
fold-level metric, so it is the deployed one.

> Distances are computed in **float64**. In float32, sklearn's cosine (`1 − x·y`) cancels
> below machine epsilon for the many near-identical loci in this dataset, quantising ~73%
> of in-distribution distances toward zero.

## Train your own model

```bash
ALLCAPS train --input manifest.csv --output run/ --device cuda
```

`manifest.csv` is a CSV/TSV with a `path` column and a `serotype` column, one row per
assembly:

```csv
path,serotype
/data/ERR1788086.fasta,19A
/data/ERR714669.fasta,3
```

The *cps* locus is cut from each assembly and the leftover fragments become the
non-capsular class, so there is no `is_cbl` column to supply. The run directory ends up
holding the three artifacts `predict` consumes — `transformer_model.pth`,
`knn_index.npz`, `energy_summary.json` — so a model trained here is immediately usable:

```bash
ALLCAPS predict --input samples.txt --extract align --output out/ \
    --model run/transformer_model.pth \
    --knn-index run/knn_index.npz \
    --energy-summary run/energy_summary.json
```

This is a long job — the released checkpoint took about **19 hours on one A100**. Use
`--resume` to continue a run that stopped. See **[TRAINING.md](TRAINING.md)** for
hyperparameters, what each stage consumes, and footguns.

## Tune per-serotype novelty thresholds

```bash
mamba env create -f environment-r.yml      # optional, ~325 MB, only for this command
ALLCAPS knn --input results/loo-sweep/knn_raw/ \
            --ground-truth merged_ground_truth.csv \
            --output results/threshold-tuning/
```

An analysis command: it asks whether a *per-serotype* novelty threshold would beat the
single global one ALLCAPS deploys. Implemented in R, and the only command that needs it —
nothing in `predict` or `train` does.

## The research pipeline (Snakemake)

`ALLCAPS train` covers the training path. The Snakefile remains the route for reproducing
the published analysis, including the leave-one-serotype-out folds:

```bash
cp src/config.yaml.template config.yaml   # then edit the paths
cd src                                    # the Snakefile resolves modules relative to itself
snakemake -n  --configfile ../config.yaml # dry-run the DAG first
snakemake --cores 4 --configfile ../config.yaml
```

`config.yaml` and `data/` are gitignored. `locus_cutting` → `labels_preprocessing` →
`train_test_split` → `embed_base` → `labels_postprocessing` → `train_model` →
`embed_chunks` → evaluation → `novel_detection` → `knn_fit` →
`knn_predict_id` / `knn_predict_query`. `train_model_loo`, `embed_chunks_loo` and
`serotype_classification_loo` repeat training with one serotype withheld, and only
materialise when `serotypes` is populated.

See [src/README.md](src/README.md) for the rule-by-rule breakdown and a one-line
description of every module, and [src/config.yaml.template](src/config.yaml.template) for
every config key. On a cluster, [jobs/run_snakemake.sh](jobs/run_snakemake.sh) submits the
whole DAG and [jobs/allcaps_slurm.sh](jobs/allcaps_slurm.sh) drives it stage by stage.

## Data model

Every metadata row and FASTA record is one **contig**, keyed `Public_ID#Contig_ID`
(non-capsular records keep a `NONCBL#` prefix on `Public_ID`). One **sample** is one
assembly and may span several contigs — a *cps* locus is frequently split across two.
Metrics in this repo are computed per contig unless stated otherwise.

The train/test split
([src/allcaps/helpers/data_train_test_split.py](src/allcaps/helpers/data_train_test_split.py))
groups **by sample, never by contig**, so sibling contigs of one assembly never straddle
the boundary; a runtime assertion enforces it.

## Repository layout

- `ALLCAPS.py` — run the CLI from a plain clone, without installing.
- `src/allcaps/cli/` — the `ALLCAPS` command (`predict`, `train`, `knn`).
- `src/allcaps/` — core modules (models, embedding, inference, evaluation, kNN novelty).
- `src/allcaps/helpers/` — data preparation, the train/test splitter, novelty sweeps and plots.
- `src/allcaps/trihead/` — training, inference and query processing for the deployed model.
- `src/allcaps/data/` — the `dexB`/`aliA` flanking genes used to cut the locus.
- `src/allcaps/tests/` — the round-trip sanity check comparing the training and query
  embedding paths. Run it after any change to chunking, pooling or base-model loading.
- `src/Snakefile` — the research workflow.
- `jobs/` — the two cluster driver scripts.

Modules also run standalone, e.g. `python -m allcaps.knn_ood predict ...` from `src/`.

## Weights & Biases (optional)

Tracking is **off by default** — `allcaps.tracking` no-ops unless asked, so `wandb` need
not even be installed. To turn it on:

```bash
pip install 'allcaps[wandb]'
ALLCAPS train --input manifest.csv --output run/ --wandb
export WANDB_MODE=offline    # to run detached, then: wandb sync --sync-all
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
