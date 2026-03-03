# Pneumococcal Serotyping Analysis
Tools and workflows for embedding, classifying, and detecting novel pneumococcal capsular biosynthetic loci (CBLs).

## Quick Start
- Install dependencies (Python ≥3.8, PyTorch, BioPython, transformers, pandas, numpy, scikit-learn, tqdm, snakemake).
- Copy and edit the config template: `cp src/config.yaml.template config.yaml`.
- Run a dry-run to verify the DAG: `snakemake -n --configfile config.yaml`.
- Execute the workflow: `snakemake --cores 4 --configfile config.yaml`.

## Environment Setup (recommended)
```bash
conda create -n pneumo python=3.10 -y
conda activate pneumo
pip install -r requirements.txt
# or install snakemake if not in requirements
pip install snakemake
```

## Configure Your Run
Edit `config.yaml` (from `src/config.yaml.template`). Key fields to set:
- `results_dir`: where outputs go.
- `data_dir`: folder for cleaned contigs and intermediate files.
- `metadata`: path to your metadata TSV/CSV.
- `infiles`: text file listing raw FASTA paths to process.
- `locus_cutter_query`: FASTA with flanking genes for locus cutting.
- `label_column`: column in metadata with serotype labels (default `ERR` placeholder, change to your column).
- `skip_labels`: list of labels to exclude (optional).
- `serotypes`: list used for LOO runs (empty if not running LOO).
- `base_model`, `chunk_size`, `stride_ratio`, `seq_max_len`: embedding settings.
- `model_params`: JSON string for model hyperparams (layers, heads, temperature, etc.).
- `energy_thresholds_json` / `id_energies_csv`: optional inputs for novelty thresholds when running query mode.

> Placeholder: provide your own paths for `metadata`, `infiles`, `locus_cutter_query`, and `query_path`. Leave empty or comment fields you do not use.

## Data Preparation
1) Create/collect raw assemblies. List them in `infiles` (one FASTA path per line).
2) Provide `metadata` with at least: sample ID, contig ID, serotype label (matching `label_column`), and capsule flag (`Is_capsule`).
3) Provide flanking gene FASTA for locus cutting (`locus_cutter_query`).

## Running the Pipeline
Common targets (see `src/Snakefile`):
- `locus_cutting`: cut loci and clean contigs.
- `infer_chunks_cbl` / `infer_chunks_noncbl`: embed CBL and non-CBL contigs.
- `labels_preprocessing` / `labels_postprocessing`: clean labels and assemble metadata.
- `train_model`: train transformer LR head (optionally hierarchical loss).
- `embed_chunks`: run inference to save embeddings/outputs.
- `visualize_embeddings`, `capsule_classification`, `serotype_classification`: evaluation and plots.
- `novel_detection`: query-time novelty detection (uses energy thresholds inputs if provided).
- LOO variants: `train_model_loo`, `embed_chunks_loo`, `serotype_classification_loo` when `serotypes` is populated.

Run everything via `rule all`:
```bash
snakemake --cores 4 --configfile config.yaml
```

## Re-train on Your Own Data
1) Update `config.yaml` with your metadata, `label_column`, and embedding settings.
2) Place raw FASTAs and flanking genes, update `infiles` and `locus_cutter_query` accordingly.
3) (Optional) Adjust `model_params` JSON (e.g., `{"embedding_dim": 384, "num_layers": 2, "nhead": 4, "temperature": 0.07, "weight_fine": 1, "weight_coarse": 0.5}`).
4) Run `snakemake --cores <n> --configfile config.yaml` to rebuild embeddings and re-train.

## Novel Detection / Query Mode
- Provide `query_path` in `config.yaml`.
- Supply either `energy_thresholds_json` or `id_energies_csv` to set novelty thresholds.
- Execute `snakemake --cores 1 --configfile config.yaml --targets query_results.csv` (or run `scripts/novel_detection.py` directly with matching args).

## Weights & Biases (optional)
To log offline during Snakemake runs:
```bash
export WANDB_MODE=offline
```
Later sync:
```bash
wandb sync --sync-all
```

## Contributing / Support
- Issues and PRs are welcome.
- Open an issue with environment details, command used, and any logs for troubleshooting.
