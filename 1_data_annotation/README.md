# Comprehensive database from CBLs
The first milestone is to annotate the CPS sequences.

## Directory Structure
This directory contains a reproducible pipeline that:

1. Extracts frozen Transformer embeddings from pretrained models ([`./scripts/embed_transformer.py`](./scripts/embed_transformer.py)).
3. Trains a contrastive head on top of the frozen embeddings ([`./scripts/train_contrastive.py`](./scripts/train_contrastive.py)) and infers the new representation.
3. Visualizes embeddings with t-SNE ([`./scripts/tsne_visualize.py`](./scripts/tsne_visualize.py)).

The pipeline uses [Snakemake](https://snakemake.readthedocs.io/) to ensure each step only runs when necessary (based on file timestamps) and to provide a clear, modular workflow ([`./Snakefile`](./Snakefile)).

## Running the Workflow
0. *(Optional but recommended)* Create and activate a conda environment.
1. Generate a `config.yaml` file based off the provided [template](./config.yaml.template).
2. Run Snakemake:
```bash
snakemake --cores 1  # Or more if you'd like parallel jobs
```