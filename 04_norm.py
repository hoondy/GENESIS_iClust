#!/usr/bin/env python

# sc
import pegasus as pg
import scanpy as sc
import anndata as ad
from anndata.experimental import read_elem, write_elem, sparse_dataset
# from anndata.experimental import read_elem, write_elem
# from anndata._core.sparse_dataset import SparseDataset

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

args = parser.parse_args()

################################################################################

def log1p_norm(input, output):
    
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