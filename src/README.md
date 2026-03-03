# Pipeline Overview
This directory hosts the Snakemake workflow for embedding, training, evaluating, and querying pneumococcal capsular loci. See `Snakefile` for authoritative rule definitions.

## Main Rules (ordered by dependency)
1. **`locus_cutting`**: Cut CPS loci from raw assemblies using flanking genes; produces cleaned CBL and non-CBL contigs.
2. **`infer_chunks_cbl` / `infer_chunks_noncbl`**: Generate base transformer embeddings for CBL and non-CBL contigs.
3. **`labels_preprocessing`**: Clean and normalize metadata/labels.
4. **`labels_postprocessing`**: Merge labels with embedding availability; apply skips and set serotype column.
5. **`train_model`**: Train the transformer LR head (optionally hierarchical loss) on chunked embeddings.
6. **`embed_chunks`**: Run inference to materialize embeddings/logits for downstream evaluation.
7. **`visualize_embeddings`**: Produce t-SNE plot of embeddings.
8. **`capsule_classification`**: Evaluate capsule/non-capsule performance.
9. **`serotype_classification`**: Evaluate serotype performance and export reports/matrices.
10. **`novel_detection`**: Energy-based novelty detection for query sequences (requires query inputs and thresholds).
11. **LOO variants** (`train_model_loo`, `embed_chunks_loo`, `serotype_classification_loo`): repeat training/eval leaving one serotype out (only when `serotypes` is populated in config).

## Running
- From repo root: `snakemake --cores 4 --configfile config.yaml`
- Dry run: `snakemake -n --configfile config.yaml`
- Generate DAG: `snakemake --forceall --dag | dot -Tpdf > dag.pdf`

## Configuration Notes
- Edit `config.yaml` (copy from `config.yaml.template`) to set paths for metadata, infiles, flanking genes, and results directory.
- Set `label_column` to your serotype column name; add `skip_labels` to exclude problematic classes.
- Populate `serotypes` when you want LOO targets; leave empty to skip LOO rules.
- Adjust `model_o0ding dimensions.

## Placeholders / TODOs
- Provide your own flanking gene FASTA (`locus_cutter_query`).
- Provide a metadata table with `Is_capsule`, contig IDs, and serotype labels.
- For query mode, supply `query_path` and either `energy_thresholds_json` or `id_energies_csv` in `config.yaml`.

