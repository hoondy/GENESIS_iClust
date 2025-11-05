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
parser.add_argument('--anno', help='Gene annotation csv file', required=True)

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

def gene_anno(input, output, anno):
    
    adata = _read_everything_but_X(input)
    
    # create gene_id attrib
    adata.var['gene_id'] = [x.replace("_index","") for x in adata.var.index]
    
    # load AnnData var annotation
    features = pd.read_csv(anno, low_memory=False)
    
    # remove dup
    features = features.drop_duplicates()
    
    # index
    features.index = features.ensembl_gene_id
    
    # fix gene name that is NaN
    features.loc[features.hgnc_symbol.isna(),'hgnc_symbol'] = features.loc[features.hgnc_symbol.isna(),'ensembl_gene_id']
    
    # merge gencode annotation
    adata.var['gene_name'] = list(features.loc[adata.var.gene_id,'hgnc_symbol'])
    adata.var['gene_type'] = list(features.loc[adata.var.gene_id,'gene_biotype'])
    adata.var['gene_chrom'] = list(features.loc[adata.var.gene_id,'chromosome_name'])
    adata.var['gene_start'] = list(features.loc[adata.var.gene_id,'start_position'])
    adata.var['gene_end'] = list(features.loc[adata.var.gene_id,'end_position'])
    
    # set featurekey
    adata.var['featurekey'] = adata.var.gene_name
    adata.var.index = adata.var.featurekey
    del adata.var['featurekey']
    
    # make feature name unique
    adata.var_names_make_unique(join='#')
    
    # save
    _write_h5ad_with_new_annotation(input, adata, output)

################################################################################

# add gene annotation
gene_anno(input=args.input,
          output=args.output,
          anno=args.anno)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")