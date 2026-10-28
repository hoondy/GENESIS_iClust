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
    # plot an in-memory copy (obs columns + X_umap only): scanpy writes colors into .uns, which fails on backed views
    cols = [color] if isinstance(color, str) else list(color)
    plot = ad.AnnData(obs=adata.obs[cols].copy(), obsm={"X_umap": np.asarray(adata.obsm["X_umap"])})
    sc.pl.umap(plot, color=color, title=title, legend_loc='on data', frameon=False, legend_fontsize=6, legend_fontoutline=2, size=1, add_outline=True, outline_width=(0.2,0.02), show=False, palette=pal)
    plt.savefig(save, bbox_inches="tight")
    plt.close("all")
    
################################################################################

# palette
pal = pd.read_csv(args.palette)
pal_class = pal[pal.category=='class']
pal_class = dict(zip(pal_class['name'],pal_class['color_hex']))
pal_subclass = pal[pal.category=='subclass']
pal_subclass = dict(zip(pal_subclass['name'],pal_subclass['color_hex']))

# load
adata = ad.read_h5ad(args.input, backed="r")   # counts are streamed from raw.X

# lineage mixing
sct.tl.detect_lineage_mixing(adata,
                             groupby="subclass",
                             lineage_key="class",
                             known_lineage_sets=[("OPC", "Oligo")],   # keeps real OPC to Oligo transitions
                             )
mixed = adata[adata.obs["mixing_flag"] == 1].shape[0]
retained = adata[adata.obs["mixing_flag"] == 0].shape[0]
print('Found %i cells (%.2f %%) with mixed lineages. Retaining %i remaining cells.' % (mixed, mixed/(mixed+retained)*100, retained))

# outlier detection
summary = sct.tl.detect_outliers(adata[adata.obs["mixing_flag"] == 0],
                                 groupby="subclass",
                                 use_rep="X_pca_regressed_harmony",
                                 method="lof"
                                )
print(summary)

# umap mixed lineage
save_umap(adata[adata.obs["mixing_flag"]==1], color='class', title='class', save=args.input.replace('.h5ad','_class_mixed.png'), pal=pal_class)
save_umap(adata[adata.obs["mixing_flag"]==1], color='subclass', title='subclass', save=args.input.replace('.h5ad','_subclass_mixed.png'), pal=pal_subclass)

# umap outlier
save_umap(adata[adata.obs["outlier"]==1], color='class', title='class', save=args.input.replace('.h5ad','_class_outlier.png'), pal=pal_class)
save_umap(adata[adata.obs["outlier"]==1], color='subclass', title='subclass', save=args.input.replace('.h5ad','_subclass_outlier.png'), pal=pal_subclass)

# subset criteria
subset_obs = ((adata.obs["outlier"]==0)&(adata.obs["mixing_flag"]==0)).tolist()

# run ondisk_subset
sct.io.ondisk_subset(orig_h5ad = args.input,
                     new_h5ad = args.output,
                     subset_obs = subset_obs,
                     chunk_size = 500000,
                     raw = True)

# umap clean
save_umap(adata[(adata.obs["outlier"]==0)&(adata.obs["mixing_flag"]==0)], color='class', title='class', save=args.output.replace('.h5ad','_class.png'), pal=pal_class)
save_umap(adata[(adata.obs["outlier"]==0)&(adata.obs["mixing_flag"]==0)], color='subclass', title='subclass', save=args.output.replace('.h5ad','_subclass.png'), pal=pal_subclass)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")