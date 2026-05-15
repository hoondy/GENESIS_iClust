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

parser.add_argument('--palette', help='Color palette CSV file', required=True)

args = parser.parse_args()

################################################################################

def save_umap(adata, color, title, save, pal=None):
    sc.set_figure_params(dpi=100, dpi_save=300, vector_friendly = True)
    if pal:
        sc.pl.umap(adata, color=color, title=title, legend_loc='on data', frameon=False, legend_fontsize=6, legend_fontoutline=2, size=1, add_outline=True, outline_width=(0.2,0.02), show=False, palette=pal)
    else:
        sc.pl.umap(adata, color=color, title=title, legend_loc='on data', frameon=False, legend_fontsize=6, legend_fontoutline=2, size=1, add_outline=True, outline_width=(0.2,0.02), show=False)
    plt.savefig(save, bbox_inches="tight")

################################################################################

# palette
pal = pd.read_csv(args.palette)
pal_class = pal[pal.category=='class']
pal_class = dict(zip(pal_class['name'],pal_class['color_hex']))
pal_subclass = pal[pal.category=='subclass']
pal_subclass = dict(zip(pal_subclass['name'],pal_subclass['color_hex']))

# load
adata = sct.io.read_everything_but_X(args.input)

# outlier detection
summary = sct.tl.detect_outliers(
    adata,
    groupby="subclass",
    use_rep="X_pca_regressed_harmony",
    method="lof"
)
print(summary)

# umap outlier
save_umap(adata[adata.obs["outlier"]==1], color='class', title='class', save=args.input.replace('.h5ad','_class_outlier.png'), pal=pal_class)
save_umap(adata[adata.obs["outlier"]==1], color='subclass', title='subclass', save=args.input.replace('.h5ad','_subclass_outlier.png'), pal=pal_subclass)

# umap output
save_umap(adata[adata.obs["outlier"]==0], color='class', title='class', save=args.output.replace('.h5ad','_class.png'), pal=pal_class)
save_umap(adata[adata.obs["outlier"]==0], color='subclass', title='subclass', save=args.output.replace('.h5ad','_subclass.png'), pal=pal_subclass)

# subset criteria
subset_obs = (adata.obs["outlier"]==0).tolist()

# run ondisk_subset
sct.io.ondisk_subset(orig_h5ad = args.input,
                     new_h5ad = args.output,
                     subset_obs = subset_obs,
                     chunk_size = 500000,
                     raw = True)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")