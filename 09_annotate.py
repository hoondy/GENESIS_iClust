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

parser.add_argument('--annotation', help='Annotation CSV file', required=True)

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

# note, args_hvf_flavor = 'cell_ranger' throws error for rare cell populations
def _pg_umap(args_input, args_output, args_pal,
             args_hvf_flavor = 'cell_ranger', args_hvf_batch = None, args_hvf_n_top_genes = None,
             args_hvf_min_mean = 0.0125, args_hvf_max_mean = 3, args_hvf_min_disp = 0.5,
             args_hvf_protein_coding = True, args_hvf_autosome = True,
             args_pca_n_pcs = 30,
             args_regress_var = ['n_counts','percent_mito','cycle_diff'],
             args_harmony_batch = 'Source',
             args_knn_K = 100, args_umap_n_neighbors = 15,
             genome='GRCh38', modality='rna'):
    
    ### highly variable features
    hvg = sct.pp.scanpy_hvf_h5ad(h5ad_file=args_input,
                                 flavor=args_hvf_flavor,
                                 batch_key=args_hvf_batch,
                                 min_mean=args_hvf_min_mean,
                                 max_mean=args_hvf_max_mean,
                                 min_disp=args_hvf_min_disp,
                                 n_top_genes=args_hvf_n_top_genes,
                                 protein_coding=args_hvf_protein_coding,
                                 autosome=args_hvf_autosome)
    gc.collect()

    ### load data
    data = pg.read_input(args_input, genome=genome, modality=modality)
    data.var['highly_variable_features'] = False
    data.var.loc[data.var.index.isin(hvg),'highly_variable_features'] = True

    ### pca/regress/harmony
    pg.pca(data, n_components=args_pca_n_pcs, random_state=0)
    pg.regress_out(data, attrs=args_regress_var)
    pg.run_harmony(data, batch=args_harmony_batch, rep='pca_regressed', max_iter_harmony=20, n_comps=args_pca_n_pcs, random_state=0)

    ### kNN/umap
    pg.neighbors(data, rep='pca_regressed_harmony', use_cache=False, dist='l2', K=args_knn_K, n_comps=args_pca_n_pcs, random_state=0)
    pg.umap(data, rep='pca_regressed_harmony', n_neighbors=args_umap_n_neighbors, rep_ncomps=args_pca_n_pcs, random_state=0)
    
    ### figure
    pal = pd.read_csv(args_pal)
    pal_class = pal[pal.category=='class']
    pal_class = dict(zip(pal_class['name'],pal_class['color_hex']))
    pal_subclass = pal[pal.category=='subclass']
    pal_subclass = dict(zip(pal_subclass['name'],pal_subclass['color_hex']))
    save_umap(adata=data.to_anndata(), color='class', title='class', save=args_output.replace('.h5ad','_class.png'), pal=pal_class)
    save_umap(adata=data.to_anndata(), color='subclass', title='subclass', save=args_output.replace('.h5ad','_subclass.png'), pal=pal_subclass)

    ### save
    sct.io.save(data, args_output)

    ### mem
    del data
    gc.collect()

################################################################################

# load

adata = sct.io.read_everything_but_X(args.input)
anno = pd.read_csv(args.annotation)

# subset criteria
subset_obs = (adata.obs['subtype'].isin(anno.subtype.tolist())).tolist()

# run ondisk_subset
sct.io.ondisk_subset(orig_h5ad = args.input,
                     new_h5ad = args.input.replace('.h5ad','.h5ad.tmp1'),
                     subset_obs = subset_obs,
                     chunk_size = 500000,
                     raw = True)

# add annotation
adata = sct.io.read_everything_but_X(args.input.replace('.h5ad','.h5ad.tmp1'))
subtype2class = dict(zip(anno['subtype'],anno['class']))
subtype2subclass = dict(zip(anno['subtype'],anno['subclass']))
adata.obs['class'] = [subtype2class[x] for x in adata.obs.subtype]
adata.obs['subclass'] = [subtype2subclass[x] for x in adata.obs.subtype]

# replace annotation
sct.io.write_h5ad_with_new_annotation(args.input.replace('.h5ad','.h5ad.tmp1'), adata, args.input.replace('.h5ad','.h5ad.tmp2'), raw = True)

# umap
_pg_umap(args_input = args.input.replace('.h5ad','.h5ad.tmp2'),
         args_output = args.output,
         args_pal = args.palette,
         args_hvf_flavor = 'cell_ranger', args_hvf_batch = None, args_hvf_n_top_genes = args.hvf_n_top_genes,
         args_hvf_protein_coding = True, args_hvf_autosome = True,
         args_pca_n_pcs = 30,
         args_regress_var = ['n_counts','percent_mito','cycle_diff'],
         args_harmony_batch = args.harmony_batch,
         args_knn_K = 100,
         args_umap_n_neighbors = 15)

# remove temp files
os.remove(args.input.replace('.h5ad','.h5ad.tmp1'))
os.remove(args.input.replace('.h5ad','.h5ad.tmp2'))

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")