#!/usr/bin/env python
"""
Step 11 - Final embedding of the cleaned dataset (release step 3 of 3).

Re-selects HVFs and recomputes PCA/Harmony/kNN/UMAP after 10_clean.py
removed cells, then plots class/subclass UMAPs with the GENESIS palette
(00_shared.umap_annotated). No re-clustering is done; labels come from 09.

Snakemake: TEST_rc2.smk, rule `umap`
"""

# sys
import importlib
import sys

# etc
import argparse

# shared pipeline helpers (file name starts with a digit, so no plain `import`)
shared = importlib.import_module('00_shared')

################################################################################

parser = argparse.ArgumentParser(description='Step 11: final embedding and UMAP of the cleaned dataset')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)

parser.add_argument('--harmony_batch', help='Batch variable for Harmony', required=False, default='Source')
parser.add_argument('--hvf_n_top_genes', help='Number of highly variable features for UMAP', type=int, required=False, default=6000)

parser.add_argument('--palette', help='Color palette CSV file', required=True)

args = parser.parse_args()

################################################################################

# hvf/pca/regress/harmony/kNN/umap, figures and save
shared.umap_annotated(input_h5ad = args.input,
                      output_h5ad = args.output,
                      palette_csv = args.palette,
                      harmony_batch = args.harmony_batch,
                      hvf_n_top_genes = args.hvf_n_top_genes)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
