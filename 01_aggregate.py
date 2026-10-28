#!/usr/bin/env python
"""
Step 01 - Merge the per-sample H5AD files into one H5AD, on disk.

The files listed in --manifest (one file name per line, relative to --path)
are concatenated with sct.io.concat_on_disk. It merges obs/var with
merge='same' and streams each count matrix into the output, so the full
matrix is never loaded into memory. Inputs stored as CSC are converted to CSR
through a temporary file (temp.h5ad in the working directory). Only X is
carried over; any raw is dropped.

Snakemake: TEST_pass2.smk, rule `aggregate`
"""

# sc
import SCTools as sct

# sys
import gc
import os
import sys

# etc
import argparse

################################################################################

parser = argparse.ArgumentParser(description='Step 01: merge per-sample h5ad files on disk')

parser.add_argument('--path', help='Directory containing the per-sample h5ad files', required=True)
parser.add_argument('--manifest', help='Text file listing one h5ad file name per line, relative to --path', required=True)
parser.add_argument('--output', help='Output merged h5ad file', required=True)

args = parser.parse_args()

################################################################################

# merge all h5ad on disk
with open(args.manifest) as f:
    list_h5ad_parts = [os.path.join(args.path, x.rstrip()) for x in f.readlines()]
sct.io.concat_on_disk(list_h5ad_parts, output_pth=args.output)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
