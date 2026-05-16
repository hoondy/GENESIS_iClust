#!/usr/bin/env python

# sc
import pegasus as pg
import scanpy as sc
import anndata as ad
import SCTools as sct

# plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import seaborn as sns

# data
import numpy as np
import pandas as pd
from scipy import stats, sparse
import h5py

# sys
import gc
import os
import sys
from pathlib import Path

# etc
import argparse

################################################################################

parser = argparse.ArgumentParser(description='pegasus end2end wrapper script')

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
                     chunk_size = 500000,
                     raw = True)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")