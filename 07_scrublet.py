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

def _read_everything_but_X(pth) -> ad.AnnData:
    # read all keys but X and raw
    with h5py.File(pth) as f:
        attrs = list(f.keys())
        attrs.remove('X')
        if 'raw' in attrs:
            attrs.remove('raw')
        adata = ad.AnnData(**{k: read_elem(f[k]) for k in attrs})
        # print(adata.shape)
    return adata

def _clean_unused_categories(data):
    for obs_name in data.obs.columns:
        if data.obs[obs_name].dtype=='category':
            # print('Removing unused categories from',obs_name)
            data.obs[obs_name] = data.obs[obs_name].cat.remove_unused_categories()
    for var_name in data.var.columns:
        if data.var[var_name].dtype=='category':
            # print('Removing unused categories from',var_name)
            data.var[var_name] = data.var[var_name].cat.remove_unused_categories()
    return data

def _ondisk_subset(orig_h5ad, new_h5ad, subset_obs, subset_var = None, chunk_size = 500000, raw = False, adata = None):

    if adata is None:
        
        # read annotations only
        adata = _read_everything_but_X(orig_h5ad)

        # subset obs
        if subset_obs is not None:
            adata._inplace_subset_obs(subset_obs)

        # subset var
        if subset_var is not None:
            adata._inplace_subset_var(subset_var)

        # clean unused cat
        adata = _clean_unused_categories(adata)
        
    # new annotation
    new_uns=None
    if adata.uns:
        new_uns = adata.uns

    new_obsm=None
    if adata.obsm:
        new_obsm = adata.obsm

    new_varm=None
    if adata.varm:
        new_varm = adata.varm

    new_obsp=None
    if adata.obsp:
        new_obsp = adata.obsp

    new_varp=None
    if adata.varp:
        new_varp = adata.varp

    new_layers=None
    if adata.layers:
        new_layers = adata.layers
    
    # save obs and var first
    ad.AnnData(None, obs=adata.obs, var=adata.var, uns=new_uns, obsm=new_obsm, varm=new_varm, obsp=new_obsp, varp=new_varp, layers=new_layers).write(new_h5ad)
    
    # initialize new_h5ad
    with h5py.File(new_h5ad, "a") as target:
        dummy_X = sparse.csr_matrix((0, adata.var.shape[0]), dtype=np.float32)
        dummy_X.indptr = dummy_X.indptr.astype(np.int64) # Guarding against overflow for very large datasets
        dummy_X.indices = dummy_X.indices.astype(np.int64) # Guarding against overflow for very large datasets
        write_elem(target, "X", dummy_X)
        if raw:
            write_elem(target, "raw/X", dummy_X)
        
    # get indptr first
    with h5py.File(orig_h5ad, 'r') as f:
        csr_indptr = f['X/indptr'][:]

    # append subset of X
    for idx in [i for i in range(0, csr_indptr.shape[0]-1, chunk_size)]:
        print('Processing', idx, 'to', idx+chunk_size)
        row_start, row_end = idx, idx+chunk_size

        if sum(subset_obs[row_start:row_end])>0:
            # X
            with h5py.File(orig_h5ad, 'r') as f:
                tmp_indptr = csr_indptr[row_start:row_end+1]
                
                new_data = f['X/data'][tmp_indptr[0]:tmp_indptr[-1]]
                new_indices = f['X/indices'][tmp_indptr[0]:tmp_indptr[-1]]
                new_indptr = tmp_indptr - csr_indptr[row_start]
                
                if subset_var is not None:
                    new_shape = [tmp_indptr.shape[0]-1, len(subset_var)]
                    tmp_csr = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=new_shape)
                    tmp_csr = tmp_csr[subset_obs[row_start:row_end]][:,subset_var]
                else:
                    new_shape = [tmp_indptr.shape[0]-1, adata.shape[1]]
                    tmp_csr = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=new_shape)
                    tmp_csr = tmp_csr[subset_obs[row_start:row_end]]
                    
                tmp_csr.sort_indices()

            # append X
            with h5py.File(new_h5ad, "a") as target:
                mtx = sparse_dataset(target["X"])
                mtx.append(tmp_csr)

            # raw/X
            if raw and ('raw' in h5py.File(orig_h5ad, 'r')):
                with h5py.File(orig_h5ad, 'r') as f:
                    tmp_indptr = csr_indptr[row_start:row_end+1]
                    
                    new_data = f['raw/X/data'][tmp_indptr[0]:tmp_indptr[-1]]
                    new_indices = f['raw/X/indices'][tmp_indptr[0]:tmp_indptr[-1]]
                    new_indptr = tmp_indptr - csr_indptr[row_start]
                    
                    if subset_var is not None:
                        new_shape = [tmp_indptr.shape[0]-1, len(subset_var)]
                        tmp_csr = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=new_shape)
                        tmp_csr = tmp_csr[subset_obs[row_start:row_end]][:,subset_var]
                    else:
                        new_shape = [tmp_indptr.shape[0]-1, adata.shape[1]]
                        tmp_csr = sparse.csr_matrix((new_data, new_indices, new_indptr), shape=new_shape)
                        tmp_csr = tmp_csr[subset_obs[row_start:row_end]]

                    tmp_csr.sort_indices()

                # append raw/X
                with h5py.File(new_h5ad, "a") as target:
                    mtx = sparse_dataset(target["raw/X"])
                    mtx.append(tmp_csr)
                
def _save(data, filename):
    if '_tmp_fmat_highly_variable_features' in data.uns:
        del data.uns['_tmp_fmat_highly_variable_features']
    data.to_anndata().write(filename)
    print('Saved',filename)

def _write_h5ad_with_new_annotation(orig_h5ad, adata, new_h5ad, raw = True):

    # new annotation
    new_uns=None
    if adata.uns:
        new_uns = adata.uns

    new_obsm=None
    if adata.obsm:
        new_obsm = adata.obsm

    new_varm=None
    if adata.varm:
        new_varm = adata.varm

    new_obsp=None
    if adata.obsp:
        new_obsp = adata.obsp

    new_varp=None
    if adata.varp:
        new_varp = adata.varp

    new_layers=None
    if adata.layers:
        new_layers = adata.layers

    # save obs and var first
    ad.AnnData(None, obs=adata.obs, var=adata.var, uns=new_uns, obsm=new_obsm, varm=new_varm, obsp=new_obsp, varp=new_varp, layers=new_layers).write(new_h5ad)

    # append X
    with h5py.File(new_h5ad, "a") as target:
        # make dummy
        dummy_X = sparse.csr_matrix((0, adata.var.shape[0]), dtype=np.float32)
        dummy_X.indptr = dummy_X.indptr.astype(np.int64) # Guarding against overflow for very large datasets
        dummy_X.indices = dummy_X.indices.astype(np.int64) # Guarding against overflow for very large datasets
        
        with h5py.File(orig_h5ad, "r") as src:
            write_elem(target, "X", dummy_X)
            sparse_dataset(target["X"]).append(sparse_dataset(src["X"]))
            
            # append raw/X if needed
            if raw and ('raw' in h5py.File(orig_h5ad, 'r')):
                write_elem(target, "raw/X", dummy_X)
                sparse_dataset(target["raw/X"]).append(sparse_dataset(src["raw/X"]))

def save_umap(adata, color, title, save):
    sc.set_figure_params(dpi=100, dpi_save=300, vector_friendly = True)
    sc.pl.umap(adata, color=color, title=title, legend_loc='on data', frameon=False, legend_fontsize=6, legend_fontoutline=1, size=10, add_outline=True, outline_width=(0.2,0.02), show=False)
    plt.savefig(save, bbox_inches="tight")

def scrublet(input, output, clust_attr, n=16):

    ### !!!You can silence this warning by setting LOKY_MAX_CPU_COUNT to the number of cores you want to use
    os.environ['LOKY_MAX_CPU_COUNT'] = str(n)
    
    ### load data
    data = pg.read_input(input)
    
    ### find doublets
    pg.infer_doublets(data, channel_attr = 'Channel', clust_attr = clust_attr, plot_hist=None, n_jobs=n, raw_mat_key='raw.X', random_state=0)
    pg.mark_doublets(data)

    ### plot
    save_umap(adata=data.to_anndata(), color='demux_type', title=None, save=input.replace('.h5ad','_scrublet.png'))
    print(data.uns['pred_dbl_cluster'])

    ### stats
    dc = data.obs['demux_type'].value_counts().reset_index()
    print(dc)

    pct_dbl = dc.loc[dc['demux_type']=='doublet','count'].iloc[0] / np.sum(dc.loc[:,'count']) * 100
    print('Doublets: %.2f%%' % pct_dbl)

    ### save scrublet
    _save(data, input.replace('.h5ad','_scrublet.h5ad'))

    ### ondisk filter doublets
    _ondisk_subset(orig_h5ad = input.replace('.h5ad','_scrublet.h5ad'),
                   new_h5ad = output,
                   subset_obs = (data.obs.demux_type=='singlet').tolist(),
                   chunk_size = 500000,
                   raw = True)

################################################################################

scrublet(input = args.input,
         output = args.output,
         clust_attr = 'subtype',
         n=16)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")