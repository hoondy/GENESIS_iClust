#!/usr/bin/env python
"""
Step 03 - Gene- and cell-level QC.

Genes
    Keep only robust genes, i.e. genes expressed in at least 0.05% of cells
    (pg.identify_robust_genes). Then add var flags:
        ribo, mito, protein_coding, mitocarta
        robust_protein_coding           robust & protein_coding & not ribo/mito/mitocarta
        robust_protein_coding_autosome  same, restricted to chr1-22
Cells
    n_counts and n_genes must be within median +/- mad_k * MAD on the log
    scale (sct.pp.qc_boundary), and percent_mito must not exceed pct_mito.
Channels
    Channel = obs['libraryID'], or obs['individualID'] when libraryID is
    missing. Channels with fewer than min_n_cells cells left after cell QC
    are removed completely.

QC plots and filter statistics are written to <prefix>_*.png and <prefix>_filter_stats.csv.

Snakemake: TEST_pass2.smk, rule `qc`
"""

# sc
import pegasus as pg
import SCTools as sct

# plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context
import seaborn as sns

# data
import numpy as np
import pandas as pd

# sys
import gc
import sys

# etc
import argparse

################################################################################

parser = argparse.ArgumentParser(description='Step 03: gene and cell QC')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)
parser.add_argument('--prefix', help='Output prefix for QC plots and filter stats', required=True)
parser.add_argument('--anno', help='Human MitoCarta annotation', required=True)
parser.add_argument('--mad_k', help='Keep cells within median +/- mad_k * MAD of log n_counts and log n_genes', required=False, default=3, type=int)
parser.add_argument('--pct_mito', help='Maximum percent of mitochondrial UMIs per cell', required=False, default=5, type=int)
parser.add_argument('--min_n_cells', help='Remove channels with fewer cells than this after QC', required=False, default=50, type=int)

args = parser.parse_args()

################################################################################

def _histplot(values, cutoffs, xlabel, save):
    """Save a 4x4 inch histogram of `values` with red vertical lines at `cutoffs`."""
    with rc_context({'figure.figsize': (4, 4)}):
        plt.figure(figsize=(4, 4))
        sns.histplot(values)
        for cutoff in cutoffs:
            plt.axvline(cutoff, color='red')
        plt.xlabel(xlabel, fontsize=12)
        plt.savefig(save)

def qc(input, output, prefix, anno, args_mad_k = 3, args_pct_mito = 5, args_min_n_cells = 50):
    """Run gene, cell and channel QC on `input` and write the filtered data to `output` (see module docstring)."""

    data = pg.read_input(input, file_type='h5ad', genome='GRCh38', modality='rna')

    ### create channel
    if 'libraryID' in data.obs:
        data.obs['Channel'] = data.obs.libraryID
    else:
        data.obs['Channel'] = data.obs.individualID
        print('WARNING: libraryID is missing. individualID is used as Channel.')

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
    mitocarta = pd.read_csv(anno)
    data.var['mitocarta'] = data.var.index.isin(mitocarta.Symbol)

    ### define robust_protein_coding genes (exclude ribosomal (RPL,RPS), mitochondrial, or mitocarta genes)
    data.var['robust_protein_coding'] = data.var['robust'] & data.var['protein_coding']
    data.var.loc[data.var.ribo, 'robust_protein_coding'] = False
    data.var.loc[data.var.mito, 'robust_protein_coding'] = False
    data.var.loc[data.var.mitocarta, 'robust_protein_coding'] = False

    ### define robust_protein_coding_autosome genes (robust_protein_coding on chr1-22)
    data.var['robust_protein_coding_autosome'] = data.var['robust_protein_coding'] & data.var.gene_chrom.isin([str(i) for i in range(1, 23)])

    ####################
    ### QC by counts ###
    ####################

    ### compute n_counts, n_genes and percent_mito
    pg.qc_metrics(data, mito_prefix='MT-')

    ### nUMI and nGene QCs
    n_counts_lower, n_counts_upper = sct.pp.qc_boundary(data.obs.n_counts, k=args_mad_k)
    print('n_UMIs lower: %d upper: %d' % (n_counts_lower, n_counts_upper))

    n_genes_lower, n_genes_upper = sct.pp.qc_boundary(data.obs.n_genes, k=args_mad_k)
    print('n_genes lower: %d upper: %d' % (n_genes_lower, n_genes_upper))

    # log(n_counts)
    _histplot(np.log10(data.obs.n_counts), [np.log10(n_counts_lower), np.log10(n_counts_upper)], 'log10(n_counts)', prefix+"_histplot_n_counts.png")

    # log(n_genes)
    _histplot(np.log10(data.obs.n_genes), [np.log10(n_genes_lower), np.log10(n_genes_upper)], 'log10(n_genes)', prefix+"_histplot_n_genes.png")

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
    _histplot(data.obs.percent_mito, [args_pct_mito], 'percent_mito', prefix+"_histplot_percent_mito.png")

    ### apply QC filter (sets obs['passed_qc'])
    pg.qc_metrics(data,
                  min_genes=n_genes_lower, max_genes=n_genes_upper,
                  min_umis=n_counts_lower, max_umis=n_counts_upper,
                  mito_prefix='MT-', percent_mito=args_pct_mito)

    df = pg.get_filter_stats(data)
    df.to_csv(prefix+"_filter_stats.csv")

    #####################
    ### QC by n_cells ###
    #####################

    ### number of cells per Channel
    n_cells_before_qc = data.obs.Channel.value_counts().rename_axis('Channel').reset_index(name='counts')
    n_cells_after_qc = data.obs[data.obs.passed_qc].Channel.value_counts().rename_axis('Channel').reset_index(name='counts')

    print('n_cells before QC:', np.sum(n_cells_before_qc[n_cells_before_qc.counts>0].counts))
    print('n_cells after QC:', np.sum(n_cells_after_qc[n_cells_after_qc.counts>0].counts))

    print('mean n_cells before QC', np.mean(n_cells_before_qc[n_cells_before_qc.counts>0].counts))
    print('mean n_cells after QC', np.mean(n_cells_after_qc[n_cells_after_qc.counts>0].counts))

    _histplot(np.log10(n_cells_before_qc[n_cells_before_qc.counts>0].counts), [np.log10(args_min_n_cells)], 'log10(n_cells)', prefix+"_histplot_n_cells.png")

    ### n_cells QC: drop channels with too few cells passing QC
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
   anno = args.anno,
   args_mad_k = args.mad_k,
   args_pct_mito = args.pct_mito,
   args_min_n_cells = args.min_n_cells)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
