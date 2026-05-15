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

parser.add_argument('--harmony_batch', help='Batch variable for Harmony', required=False, default='Source')
parser.add_argument('--hvf_n_top_genes', help='Number of highly variable features for UMAP', type=int, required=False, default=6000)

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

### highly variable features
# note, args_hvf_flavor = 'cell_ranger' throws error for rare cell populations
hvg = sct.pp.scanpy_hvf_h5ad(h5ad_file=args.input,
                             flavor='cell_ranger',
                             batch_key=None,
                             min_mean=0.0125,
                             max_mean=3,
                             min_disp=0.5,
                             n_top_genes=args.hvf_n_top_genes,
                             protein_coding=True,
                             autosome=True)
gc.collect()

### load data
data = pg.read_input(args.input, genome='GRCh38', modality='rna')
data.var['highly_variable_features'] = False
data.var.loc[data.var.index.isin(hvg),'highly_variable_features'] = True

### pca/regress/harmony
pg.pca(data, n_components=30, random_state=0)
pg.regress_out(data, attrs=['n_counts','percent_mito','cycle_diff'])
pg.run_harmony(data, batch=args.harmony_batch, rep='pca_regressed', max_iter_harmony=20, n_comps=30, random_state=0)

### kNN/umap
pg.neighbors(data, rep='pca_regressed_harmony', use_cache=False, dist='l2', K=100, n_comps=30, random_state=0)
pg.umap(data, rep='pca_regressed_harmony', n_neighbors=15, rep_ncomps=30, random_state=0)

### figure
save_umap(adata=data.to_anndata(), color='class', title='class', save=args.output.replace('.h5ad','_class.png'), pal=pal_class)
save_umap(adata=data.to_anndata(), color='subclass', title='subclass', save=args.output.replace('.h5ad','_subclass.png'), pal=pal_subclass)

### save
sct.io.save(data, args.output)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")