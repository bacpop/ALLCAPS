# Comprehensive database from CBLs
This analysis aims at accurate identification of serotypes based on capsular sequences.

## Workflow Structure
This directory contains a reproducible pipeline that:

The pipeline uses [Snakemake](https://snakemake.readthedocs.io/) to ensure each step only runs when necessary (based on file timestamps) and to provide a clear, modular workflow ([`./Snakefile`](./Snakefile)).

1. **`locus_cutter`**: Extracts CPS sequences from the input FASTA file by aligning flanking genes.
2. **`embed_transformer`**: Generates base embeddings using a pretrained Nucleotide Transformer model.
3. **`kmer_sketch`**: Creates k-mer sketches from the input FASTA file for the baseline analysis.
4. **`visualize_umap`**: Visualizes the base embeddings using UMAP, for a visual comparison with (8).
5. **`train_contrastive`**: Trains a contrastive head on top of the frozen base embeddings.
6. **`embed_contrastive`**: Transforms the base embeddings using the trained contrastive head.
7. **`knn_inference`**: Performs k-Nearest Neighbors (kNN) inference on the contrastive embeddings and generates a report.
8. **`visualize_embeddings`**: Visualizes the contrastive embeddings (using UMAP, t-SNE or PCA) and saves the plot.
9. **`lda_analysis`**: Performs LDA and Random Forest classification on k-mer sketches and generates evaluation reports.
10. **`stats_summary`**: Compares class-wise F1 distributions across analysis methods and generates a summary plot.
11. **`calc_distances`**: Calculates pairwise distances among embeddings for further model fitting.
12. **`novel_detection`**: Compares query pairwise distances from known serogroups distributions to report on its novelty.

Each rule is designed to be modular and reproducible, ensuring efficient execution of the pipeline.

### Workflow DAG

The whole DAG of the rules in the workflow is visualized [here](dag.pdf).
> The graph is generated using Graphviz by `snakemake --forceall --dag | dot -Tpdf > dag.pdf`

