# GENESIS_iClust
Single-cell iterative clustering pipeline for GENESIS

The pipeline version is `__version__` in `00_shared.py`; each release is tagged `v<version>`.

## Steps

| Script | Step |
| --- | --- |
| `01_aggregate.py` | Merge per-sample h5ad files on disk |
| `02_anno.py` | Add Ensembl gene annotation, index genes by symbol |
| `03_qc.py` | Gene, cell and channel QC |
| `04_norm.py` | Shifted log1p normalisation (raw counts kept in `raw.X`) |
| `05_score.py` | QC metrics and gene-signature scores (cell cycle, gender, ...) |
| `06_iclust.py` | Iterative Leiden clustering: class > subclass > subtype |
| `07_scrublet.py` | Doublet detection and removal |
| `08_subset.py` | Remove listed clusters |
| `09_annotate.py` | Apply curated annotation and re-embed |
| `10_clean.py` | Remove lineage-mixed cells and embedding outliers |
| `11_umap.py` | Final embedding and UMAPs |

The Snakemake files chain the steps:
- `TEST_pass2.smk`: 01-07, then a second 06 pass
- `TEST_pass3.smk`: 08 and a third 06 pass
- `TEST_rc2.smk`: 09-11

## Shared code

- [SCTools](https://github.com/hoondy/SCTools) provides the general-purpose helpers: on-disk h5ad I/O, HVF selection, QC thresholds, signature scores, outlier and lineage-mixing detection. Install SCTools ≥ 0.5.0 in the pipeline environment (`py312`); HVF selection uses the `min_cells` option of `sct.pp.scanpy_hvf_h5ad`.
- HVFs (06, 09, 11) are always picked with Scanpy's `cell_ranger` flavor from protein-coding autosomal genes that are expressed in the input. Genes with no counts are dropped first, because `cell_ranger` fails with "Bin edges must be unique" on small subclusters when many genes are unexpressed.
- `00_shared.py` holds the pipeline-specific helpers that several steps reuse: shared defaults, HVF loading, the PCA/Harmony/kNN/UMAP embedding, and UMAP plotting with the GENESIS palette. Steps load it with `importlib.import_module('00_shared')`, so it must sit in the same directory as the numbered scripts.
