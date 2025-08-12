# Pnuemococcal Serotyping Analysis
This project aims to propose novel tools and datasets for analyzing pneumococcal capsular biosynthetic loci, the primary target of Pneumococcal Conjugate Vaccines (PCVs).

## How to Run

### Requirements
To run the workflow, ensure the following dependencies are installed:
1. **Python** (>3.8) and usual packages, e.g., `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tqdm`, `umap-learn`, `torch`, etc.
2. **Snakemake**: For managing the workflow ([installation guide](https://snakemake.readthedocs.io/en/stable/getting_started/installation.html)).
3. **Conda**: Recommended for creating isolated environments.

### Setup
0. *(Optional but recommended)* Create and activate a conda environment.
1. Generate a configuration file based off the provided [template](./src/config.yaml.template).
2. Run Snakemake:
```bash
snakemake --configfile <path/to/configuration> --cores 1
```

### W&B
This script benefits from [W&B](https://wandb.ai/site/models/) dashboard for monitoring each run. Check out their website for installation and login tutorials.

If you plan to run the script through the Snakemake pipeline, make sure you initialize the tool in offline mode:

```bash
export WANDB_MODE=offline
```

This ensures you do not break logging. Later on, you can sync the offline logs with your dashboard using:

```bash
wandb sync --sync-all
```
