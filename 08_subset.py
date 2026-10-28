#!/usr/bin/env python
"""
Step 08 - Remove clusters before the next clustering pass.

Drops every cell whose obs[--cluster] label (e.g. subtype '7_1_1') is listed
in --exclude, a text file with one label per line, such as a doublet or
low-quality cluster found by reviewing pass 2. The subset is made on disk,
with raw.X kept.

Snakemake: TEST_pass3.smk, rule `subset`
"""

# sc
import SCTools as sct

# sys
import importlib
import sys

# etc
import argparse

# shared pipeline helpers (file name starts with a digit, so no plain `import`)
shared = importlib.import_module('00_shared')

################################################################################

parser = argparse.ArgumentParser(description='Step 08: remove listed clusters (on disk)')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)

parser.add_argument('--cluster', help='Cluster label', required=True)
parser.add_argument('--exclude', help='Text file containing a list of exclusion cluster per line', required=True)

args = parser.parse_args()

################################################################################

# load
adata = sct.io.read_everything_but_X(args.input)

# subset criteria
with open(args.exclude) as file:
    list_exclude = [line.rstrip() for line in file]
subset_obs = (~adata.obs[args.cluster].isin(list_exclude)).tolist()

# run ondisk_subset
sct.io.ondisk_subset(orig_h5ad = args.input,
                     new_h5ad = args.output,
                     subset_obs = subset_obs,
                     chunk_size = shared.ONDISK_CHUNK_SIZE,
                     raw = True)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
