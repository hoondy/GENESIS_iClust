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

parser.add_argument('--path', help='Path to h5ad files', required=True)
parser.add_argument('--manifest', help='List of h5ad files', required=True)
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
    
def _concat_on_disk(input_pths, output_pth, temp_pth='temp.h5ad'):
    """
    Params
    ------
    input_pths
        Paths to h5ad files which will be concatenated
    output_pth
        File to write as a result
    """
    annotations = ad.concat([_read_everything_but_X(pth) for pth in input_pths], merge='same')
    annotations.write_h5ad(output_pth)
    n_variables = annotations.shape[1]
    
    del annotations

    with h5py.File(output_pth, 'a') as target:
        
        # initiate empty X
        dummy_X = sparse.csr_matrix((0, n_variables), dtype=np.float32)
        dummy_X.indptr = dummy_X.indptr.astype(np.int64) # Guarding against overflow for very large datasets
        dummy_X.indices = dummy_X.indices.astype(np.int64) # Guarding against overflow for very large datasets
        write_elem(target, 'X', dummy_X)
        
        # append
        mtx = sparse_dataset(target['X'])
        for p in input_pths:
            with h5py.File(p, 'r') as src:
                
                # IF: src is in csc format, convert to csr and save to temp_pth
                if src['X'].attrs['encoding-type']=='csc_matrix':

                    # Convert to csr format
                    csc_mat = sparse.csc_matrix((src['X']['data'], src['X']['indices'], src['X']['indptr']))
                    csr_mat = csc_mat.tocsr()         
                    
                    # save to temp_pth
                    with h5py.File(temp_pth, 'w') as tmp:
                        write_elem(tmp, 'X', csr_mat)
                    
                    # read from temp_pth
                    with h5py.File(temp_pth, 'r') as tmp:
                        mtx.append(sparse_dataset(tmp['X']))
                        
                # ELSE: src is in csr format
                else:
                    mtx.append(sparse_dataset(src['X']))
                    
# merge all h5ad on disk
with open(args.manifest) as f:
    list_h5ad_parts = [os.path.join(args.path, x.rstrip()) for x in f.readlines()]
_concat_on_disk(list_h5ad_parts, output_pth=args.output)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")