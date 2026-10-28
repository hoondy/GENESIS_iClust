#!/usr/bin/env python
"""
Step 04 - Normalise counts with a shifted log transform.

Raw counts are kept in raw.X. X becomes log1p(counts / cell total * median
cell total), i.e. scanpy normalize_total with target_sum=None, then log1p.

Snakemake: TEST_pass2.smk, rule `normalization`
"""

# sc
import scanpy as sc

# sys
import gc
import sys

# etc
import argparse

################################################################################

parser = argparse.ArgumentParser(description='Step 04: shifted log1p normalisation, raw counts kept in raw.X')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)

args = parser.parse_args()

################################################################################

def log1p_norm(input, output):
    """Store raw counts in raw.X and replace X with shifted log1p-normalised values."""

    adata = sc.read_h5ad(input)

    # keep raw counts
    adata.raw = adata.copy()

    # shifted log1p transform
    scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
    adata.X = sc.pp.log1p(scales_counts["X"], copy=True)

    adata.write(output, compression='gzip')

################################################################################

log1p_norm(input = args.input,
           output = args.output)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
