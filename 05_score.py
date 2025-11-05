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

def sig_score(input, output):
    
    data = pg.read_input(input)
    
    pg.qc_metrics(data, mito_prefix="MT-")
    
    pg.calc_signature_score(data, 'cell_cycle_human', random_state=0) ## 'cycle_diff', 'cycling', 'G1/S', 'G2/M' ## cell cycle gene score based on [Tirosh et al. 2015 | https://science.sciencemag.org/content/352/6282/189]
    pg.calc_signature_score(data, 'gender_human', random_state=0) # female_score, male_score
    pg.calc_signature_score(data, 'mitochondrial_genes_human', random_state=0) # 'mito_genes' contains 13 mitocondrial genes from chrM and 'mito_ribo' contains mitocondrial ribosomal genes that are not from chrM
    pg.calc_signature_score(data, 'ribosomal_genes_human', random_state=0) # ribo_genes
    pg.calc_signature_score(data, 'apoptosis_human', random_state=0) # apoptosis

    data.to_anndata().write(output, compression='gzip')

################################################################################

sig_score(input = args.input,
          output = args.output)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")