#!/usr/bin/env python

# sc
import pegasus as pg
import scanpy as sc
import anndata as ad
from anndata.experimental import read_elem, write_elem, sparse_dataset
# from anndata.experimental import read_elem, write_elem
# from anndata._core.sparse_dataset import SparseDataset

# data
import numpy as np
import pandas as pd
import h5py
from scipy import stats, sparse

# plot
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import seaborn as sns

# sys
import gc
import os
import sys
from pathlib import Path
from datetime import datetime

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

################################################################################

# load
adata = _read_everything_but_X(args.input)

# subset criteria
with open(args.exclude) as file:
    list_exclude = [line.rstrip() for line in file]
subset_obs = (~adata.obs[args.cluster].isin(list_exclude)).tolist()

# run ondisk_subset
_ondisk_subset(orig_h5ad = args.input,
               new_h5ad = args.output,
               subset_obs = subset_obs,
               chunk_size = 500000,
               raw = True)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")