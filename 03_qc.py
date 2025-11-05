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
parser.add_argument('--anno', help='Human MitoCarta annotation', required=True)
parser.add_argument('--mad_k', help='mad_k', required=False, default=3, type=int)
parser.add_argument('--pct_mito', help='pct_mito', required=False, default=5, type=int)
parser.add_argument('--min_n_cells', help='min_n_cells', required=False, default=50, type=int)

args = parser.parse_args()

################################################################################

def _qc_boundary(counts, k=3):
    x = np.log1p(counts)
    mad = stats.median_abs_deviation(x)
    return np.exp(np.median(x) - k*mad), np.exp(np.median(x) + k*mad)
    
def qc(input, output, prefix, args_mad_k = 3, args_pct_mito = 5, args_min_n_cells = 50):

    data = pg.read_input(input, file_type='h5ad', genome='GRCh38', modality='rna')
    
    ### create channel
    if 'participant_id' in data.obs:
        data.obs['Channel'] = data.obs.participant_id
    else:
        data.obs['Channel'] = data.obs.individualID
    
    ### create Source
    if prefix=='GEN_A1':
        data.obs['Source'] = ['RADC' if x.startswith('R') else x.split('_')[1] for x in data.obs.individualID]
    
    ##################
    ### QC by gene ###
    ##################
    
    ### identify robust genes
    pg.identify_robust_genes(data, percent_cells=0.05)
    
    ### remove features that are not robust (expressed at least 0.05% of cells) from downstream analysis
    data._inplace_subset_var(data.var['robust'])
    
    ### add ribosomal genes
    data.var['ribo'] = [x.startswith("RP") for x in data.var.gene_name]
    
    ### add mitochondrial genes
    data.var['mito'] = [x.startswith("MT-") for x in data.var.gene_name]
    
    ### add protein_coding genes
    data.var['protein_coding'] = [x=='protein_coding' for x in data.var.gene_type]
    
    ### define mitocarta_genes
    mitocarta = pd.read_csv(args.anno)
    data.var['mitocarta'] = [True if x in list(mitocarta.Symbol) else False for x in data.var.index ]
    
    ### define robust_protein_coding genes (exclude ribosomal (RPL,RPS), mitochondrial, or mitocarta genes
    data.var['robust_protein_coding'] = data.var['robust'] & data.var['protein_coding']
    data.var.loc[data.var.ribo, 'robust_protein_coding'] = False
    data.var.loc[data.var.mito, 'robust_protein_coding'] = False
    data.var.loc[data.var.mitocarta, 'robust_protein_coding'] = False
    
    ### define robust_protein_coding_autosome genes (exclude ribosomal (RPL,RPS), mitochondrial, or mitocarta genes
    data.var['robust_protein_coding_autosome'] = data.var['robust_protein_coding'] & data.var.gene_chrom.isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'])
    
    ####################
    ### QC by counts ###
    ####################
    
    pg.qc_metrics(data, mito_prefix='MT-')

    print(args_mad_k)
    print(args_mad_k*1)
    
    ### nUMI and nGene QCs
    n_counts_lower, n_counts_upper = _qc_boundary(data.obs.n_counts, k=args_mad_k)
    print('n_UMIs lower: %d upper: %d' % (n_counts_lower, n_counts_upper))
    
    n_genes_lower, n_genes_upper = _qc_boundary(data.obs.n_genes, k=args_mad_k)
    print('n_genes lower: %d upper: %d' % (n_genes_lower, n_genes_upper))
    
    # log(n_counts)
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.histplot(np.log10(data.obs.n_counts))
        plt.axvline(np.log10(n_counts_lower), color='red')
        plt.axvline(np.log10(n_counts_upper), color='red')
        plt.xlabel('log10(n_counts)', fontsize=12)
        plt.savefig(prefix+"_histplot_n_counts.png")
    
    # log(n_genes)
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.histplot(np.log10(data.obs.n_genes))
        plt.axvline(np.log10(n_genes_lower), color='red')
        plt.axvline(np.log10(n_genes_upper), color='red')
        plt.xlabel('log10(n_genes)', fontsize=12)
        plt.savefig(prefix+"_histplot_n_genes.png")
    
    # scatter
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.scatterplot(x=data.obs.n_genes, y=data.obs.n_counts, alpha=0.5, s=0.1)
        plt.axhline(n_counts_lower, color='red')
        plt.axhline(n_counts_upper, color='red')
        plt.axvline(n_genes_lower, color='red')
        plt.axvline(n_genes_upper, color='red')
        plt.savefig(prefix+"_scatterplot_threshold.png")
    
    # percent_mito
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.histplot(data.obs.percent_mito)
        plt.axvline(args_pct_mito, color='red')
        plt.xlabel('percent_mito', fontsize=12)
        plt.savefig(prefix+"_histplot_percent_mito.png")
        
    ## apply QC filter
    pg.qc_metrics(data, 
                  min_genes=n_genes_lower, max_genes=n_genes_upper,
                  min_umis=n_counts_lower, max_umis=n_counts_upper,
                  mito_prefix='MT-', percent_mito=args_pct_mito)
    
    df = pg.get_filter_stats(data)
    df.to_csv(prefix+"_filter_stats.csv")
    
    #####################
    ### QC by n_cells ###
    #####################
    
    n_cells_before_qc = data.obs.Channel.value_counts().rename_axis('Channel').reset_index(name='counts')
    n_cells_after_qc = data.obs[data.obs.passed_qc].Channel.value_counts().rename_axis('Channel').reset_index(name='counts')
    
    print('n_cells before QC:', np.sum(n_cells_before_qc[n_cells_before_qc.counts>0].counts))
    print('n_cells after QC:', np.sum(n_cells_after_qc[n_cells_after_qc.counts>0].counts))
    
    print('mean n_cells before QC', np.mean(n_cells_before_qc[n_cells_before_qc.counts>0].counts))
    print('mean n_cells after QC', np.mean(n_cells_after_qc[n_cells_after_qc.counts>0].counts))
    
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.histplot(np.log10(n_cells_before_qc[n_cells_before_qc.counts>0].counts))
        plt.axvline(np.log10(args_min_n_cells), color='red')
        plt.xlabel('log10(n_cells)', fontsize=12)
        plt.savefig(prefix+"_histplot_n_cells.png")
    
    ### n_cells QC
    n_cells_outlier = list(n_cells_after_qc[n_cells_after_qc.counts<args_min_n_cells].Channel)
    data.obs.loc[data.obs.Channel.isin(n_cells_outlier),'passed_qc'] = False
    print('remove %i donors that have cells less than %i: %s' % (len(n_cells_outlier),args_min_n_cells,n_cells_outlier))
    
    ### filter cells
    pg.filter_data(data)
    
    ### clean unused categories
    data.obs['Channel'] = data.obs.Channel.cat.remove_unused_categories()
    
    ### save
    data.to_anndata().write(output, compression='gzip')

################################################################################

qc(input = args.input,
   output = args.output,
   prefix = args.prefix,
   args_mad_k = args.mad_k,
   args_pct_mito = args.pct_mito,
   args_min_n_cells = args.min_n_cells)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")