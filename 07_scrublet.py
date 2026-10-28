#!/usr/bin/env python
"""
Step 07 - Doublet detection with Scrublet, then removal of doublets.

pg.infer_doublets runs Scrublet on the raw counts of each Channel, using the
`subtype` clusters from 06_iclust.py to flag doublet-enriched clusters.
pg.mark_doublets then sets obs['demux_type'] to singlet/doublet.

Files written:
    <input>_scrublet.h5ad/.png   all cells with doublet calls, and their UMAP
    <output>                     singlets only (subset on disk)

Snakemake: TEST_pass2.smk, rule `scrublet`
"""

# sc
import pegasus as pg
import SCTools as sct

# data
import numpy as np

# sys
import gc
import importlib
import os
import sys

# etc
import argparse

# shared pipeline helpers (file name starts with a digit, so no plain `import`)
shared = importlib.import_module('00_shared')

################################################################################

parser = argparse.ArgumentParser(description='Step 07: Scrublet doublet detection and removal')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)

args = parser.parse_args()

################################################################################

def scrublet(input, output, clust_attr, n=16):
    """Call doublets per Channel with `n` jobs, plot and save all cells, then write singlets to `output`."""

    ### silence the joblib/loky CPU count warning by setting the number of cores explicitly
    os.environ['LOKY_MAX_CPU_COUNT'] = str(n)

    ### load data
    data = pg.read_input(input)

    ### find doublets
    pg.infer_doublets(data, channel_attr = 'Channel', clust_attr = clust_attr, plot_hist=None, n_jobs=n, raw_mat_key='raw.X', random_state=0)
    pg.mark_doublets(data)

    ### plot
    shared.save_umap(adata=data.to_anndata(), color='demux_type', title=None, save=input.replace('.h5ad','_scrublet.png'),
                     size=10, legend_fontoutline=1)
    print(data.uns['pred_dbl_cluster'])

    ### stats
    dc = data.obs['demux_type'].value_counts().reset_index()
    print(dc)

    pct_dbl = dc.loc[dc['demux_type']=='doublet','count'].iloc[0] / np.sum(dc.loc[:,'count']) * 100
    print('Doublets: %.2f%%' % pct_dbl)

    ### save scrublet
    sct.io.save(data, input.replace('.h5ad','_scrublet.h5ad'))

    ### ondisk filter doublets
    sct.io.ondisk_subset(orig_h5ad = input.replace('.h5ad','_scrublet.h5ad'),
                         new_h5ad = output,
                         subset_obs = (data.obs.demux_type=='singlet').tolist(),
                         chunk_size = shared.ONDISK_CHUNK_SIZE,
                         raw = True)

################################################################################

scrublet(input = args.input,
         output = args.output,
         clust_attr = 'subtype',
         n=16)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
