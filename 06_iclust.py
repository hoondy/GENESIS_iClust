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
parser.add_argument('--prefix', help='prefix', required=True)
parser.add_argument('--batch', help='batch', required=False, default='Source')

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
                
def _sc_hvf(h5ad_file, flavor='cell_ranger', batch_key=None, min_mean=0.0125, max_mean=3.0, min_disp=0.5, n_top_genes=None, robust_protein_coding=False, protein_coding=False, autosome=False):
    
    adata=sc.read_h5ad(h5ad_file)

    ### subset robust_protein_coding if needed
    if robust_protein_coding:
        adata=adata[:,adata.var.robust_protein_coding]
    
    ### subset autosome if needed
    if autosome:
        adata=adata[:,~adata.var.gene_chrom.isin(['MT', 'X', 'Y'])]

    # seurat_v3 expects raw counts
    if flavor=='seurat_v3':
        adata.X = adata.raw.X
        if n_top_genes is None:
            raise ValueError('`n_top_genes` is mandatory if `flavor` is `seurat_v3`.')

    # find highly variable genes
    hvg = sc.pp.highly_variable_genes(adata, flavor=flavor, min_mean=min_mean, max_mean=max_mean, min_disp=min_disp, batch_key=batch_key, n_top_genes=n_top_genes, inplace=False, subset=False)
    
    return adata.var.index[hvg.highly_variable].tolist()

def _save(data, filename):
    if '_tmp_fmat_highly_variable_features' in data.uns:
        del data.uns['_tmp_fmat_highly_variable_features']
    data.to_anndata().write(filename)
    print('Saved',filename)

def save_umap(adata, color, title, save):
    sc.set_figure_params(dpi=100, dpi_save=300, vector_friendly = True)
    sc.pl.umap(adata, color=color, title=title, legend_loc='on data', frameon=False, legend_fontsize=6, legend_fontoutline=1, size=10, add_outline=True, outline_width=(0.2,0.02), show=False)
    plt.savefig(save, bbox_inches="tight")

# note, args_hvf_flavor = 'cell_ranger' throws error for rare cell populations
def _pg_leiden_clustering(args_input, args_output_prefix,
                          args_hvf_flavor = 'seurat', args_hvf_batch = None, args_hvf_n_top_genes = None,
                          args_hvf_min_mean = 0.0125, args_hvf_max_mean = 3, args_hvf_min_disp = 0.5,
                          args_hvf_protein_coding = True, args_hvf_autosome = True,
                          args_pca_n_pcs = 30, args_regress_var = ['n_counts','percent_mito','cycle_diff'], args_harmony_batch = 'Source',
                          args_knn_K = 100, args_leiden_res = 0.1, args_leiden_label = 'leiden_labels', args_leiden_label_prefix = None, args_umap_n_neighbors = 15,
                          genome='GRCh38', modality='rna'):
    
    ### highly variable features
    hvg = _sc_hvf(h5ad_file=args_input, flavor=args_hvf_flavor,
                  batch_key=args_hvf_batch, n_top_genes=args_hvf_n_top_genes,
                  min_mean=args_hvf_min_mean, max_mean=args_hvf_max_mean, min_disp=args_hvf_min_disp,
                  protein_coding=args_hvf_protein_coding, autosome=args_hvf_autosome)
    gc.collect()

    ### load data
    data = pg.read_input(args_input, genome=genome, modality=modality)
    data.var['highly_variable_features'] = False
    data.var.loc[data.var.index.isin(hvg),'highly_variable_features'] = True

    ### pca/regress/harmony
    pg.pca(data, n_components=args_pca_n_pcs, random_state=0)
    pg.regress_out(data, attrs=args_regress_var)
    pg.run_harmony(data, batch=args_harmony_batch, rep='pca_regressed', max_iter_harmony=20, n_comps=args_pca_n_pcs, random_state=0)

    ### kNN/leiden/umap
    pg.neighbors(data, rep='pca_regressed_harmony', use_cache=False, dist='l2', K=args_knn_K, n_comps=args_pca_n_pcs, random_state=0)
    pg.leiden(data, rep='pca_regressed_harmony', resolution=args_leiden_res, class_label=args_leiden_label, random_state=0)
    pg.umap(data, rep='pca_regressed_harmony', n_neighbors=args_umap_n_neighbors, rep_ncomps=args_pca_n_pcs, random_state=0)
    
    ### rename leiden label
    if args_leiden_label_prefix:
        data.obs[args_leiden_label] = data.obs[args_leiden_label].cat.rename_categories(dict(zip(data.obs[args_leiden_label].cat.categories, [args_leiden_label_prefix+'_'+x for x in data.obs[args_leiden_label].cat.categories])))
    
    ### figure
    save_umap(adata=data.to_anndata(), color=args_leiden_label, title=args_leiden_label, save=args_output_prefix+'.png')

    ### save
    _save(data, args_output_prefix+'.h5ad')

    ### mem
    del data
    gc.collect()

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

def iclust(input,
           output,
           prefix,
           args_iclust_levels = ['class','subclass','subtype'],
           args_iclust_hvfs = [3000,2000,1000],
           args_iclust_res = [0.1,0.2,0.3],
           args_regress_var = ['n_counts','percent_mito','cycle_diff'],
           args_harmony_batch = 'Source'):
    
    anno = pd.DataFrame()
    
    # base level
    base_prefix = prefix+'_'+args_iclust_levels[0]
    
    print('[%s] level clustering' % (args_iclust_levels[0]))
    _pg_leiden_clustering(args_input = input,
                          args_output_prefix = base_prefix,
                          args_hvf_n_top_genes = args_iclust_hvfs[0],
                          args_leiden_res = args_iclust_res[0],
                          args_leiden_label = args_iclust_levels[0],
                          args_regress_var = args_regress_var,
                          args_harmony_batch = args_harmony_batch)
    
    # subsequent levels
    base_adata = _read_everything_but_X(base_prefix+'.h5ad')

    for l1 in base_adata.obs[args_iclust_levels[0]].cat.categories.tolist():
        print('ondisk subset %s using [%s] level annotation' % (l1,args_iclust_levels[0]))

        l1_subset_prefix = prefix+'_'+args_iclust_levels[0]+'_'+l1

        # run ondisk_subset
        _ondisk_subset(orig_h5ad = base_prefix+'.h5ad',
                       new_h5ad = l1_subset_prefix+'.h5ad',
                       subset_obs = (base_adata.obs[args_iclust_levels[0]]==l1).tolist(),
                       chunk_size = 500000,
                       raw = True)

        # iclust
        _pg_leiden_clustering(args_input = l1_subset_prefix+'.h5ad',
                             args_output_prefix = l1_subset_prefix,
                             args_hvf_n_top_genes = args_iclust_hvfs[1],
                             args_leiden_res = args_iclust_res[1],
                             args_leiden_label = args_iclust_levels[1],
                             args_leiden_label_prefix = l1,
                             args_regress_var = args_regress_var,
                             args_harmony_batch = args_harmony_batch)

        l1_subset_adata = _read_everything_but_X(l1_subset_prefix+'.h5ad')

        for l2 in l1_subset_adata.obs[args_iclust_levels[1]].cat.categories.tolist():
            print('ondisk subset %s using [%s] level annotation' % (l2,args_iclust_levels[1]))

            l2_subset_prefix = prefix+'_'+args_iclust_levels[1]+'_'+l2

            # run ondisk_subset
            _ondisk_subset(orig_h5ad = l1_subset_prefix+'.h5ad',
                           new_h5ad = l2_subset_prefix+'.h5ad',
                           subset_obs = (l1_subset_adata.obs[args_iclust_levels[1]]==l2).tolist(),
                           chunk_size = 500000,
                           raw = True)

            # iclust
            _pg_leiden_clustering(args_input = l2_subset_prefix+'.h5ad',
                                 args_output_prefix = l2_subset_prefix,
                                 args_hvf_n_top_genes = args_iclust_hvfs[2],
                                 args_leiden_res = args_iclust_res[2],
                                 args_leiden_label = args_iclust_levels[2],
                                 args_leiden_label_prefix = l2,
                                 args_regress_var = args_regress_var,
                                 args_harmony_batch = args_harmony_batch)
            
            l2_subset_adata = _read_everything_but_X(l2_subset_prefix+'.h5ad')
            
            anno = pd.concat([anno, l2_subset_adata.obs[args_iclust_levels]])
            
    # annotation
    anno.to_csv(output.replace('.h5ad','.csv'))
    
    # ondisk update anno
    for x in args_iclust_levels:
        base_adata.obs[x] = anno.loc[base_adata.obs.index,x].astype(str)
    _write_h5ad_with_new_annotation(orig_h5ad = base_prefix+'.h5ad', adata = base_adata, new_h5ad = output)

################################################################################

iclust(input = args.input,
       output = args.output,
       prefix = args.prefix,
       args_iclust_levels = ['class','subclass','subtype'],
       args_iclust_hvfs = [3000,2000,1000],
       args_iclust_res = [0.1,0.2,0.3],
       args_harmony_batch = args.batch)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")