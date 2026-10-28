#!/usr/bin/env python
"""
Step 02 - Annotate genes and switch var names from Ensembl IDs to gene symbols.

var (Ensembl gene IDs) is joined to the Ensembl 104 table
(annotation/all_genes_EnsemblInfo_104.csv) and gets these columns:

    gene_id     Ensembl gene ID (former var index)
    gene_name   HGNC symbol, or the Ensembl ID when there is no symbol
    gene_type   biotype, e.g. protein_coding (used by 03_qc.py)
    gene_chrom  chromosome, e.g. '1', 'X', 'MT' (used to find autosomes)
    gene_start, gene_end

var is then re-indexed by gene_name, and duplicate symbols are made unique
with '#' (GENE, GENE#1, ...). Only the annotations are rewritten; X is copied
from the input unchanged.

Snakemake: TEST_pass2.smk, rule `annotation`
"""

# sc
import SCTools as sct

# data
import pandas as pd

# sys
import gc
import sys

# etc
import argparse

################################################################################

parser = argparse.ArgumentParser(description='Step 02: add Ensembl gene annotation and index genes by symbol')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)
parser.add_argument('--anno', help='Gene annotation csv file', required=True)

args = parser.parse_args()

################################################################################

def gene_anno(input, output, anno):
    """Add Ensembl gene annotation to var of `input` and write `output` (see module docstring)."""

    adata = sct.io.read_everything_but_X(input)

    # create gene_id attrib
    adata.var['gene_id'] = [x.replace("_index","") for x in adata.var.index]

    # load gene annotation, indexed by Ensembl ID
    # (only exact duplicate rows are dropped; the merge below expects one row per Ensembl ID)
    features = pd.read_csv(anno, low_memory=False)
    features = features.drop_duplicates()
    features.index = features.ensembl_gene_id

    # fall back to the Ensembl ID for genes without an HGNC symbol
    features.loc[features.hgnc_symbol.isna(),'hgnc_symbol'] = features.loc[features.hgnc_symbol.isna(),'ensembl_gene_id']

    # merge annotation (in var order)
    adata.var['gene_name'] = list(features.loc[adata.var.gene_id,'hgnc_symbol'])
    adata.var['gene_type'] = list(features.loc[adata.var.gene_id,'gene_biotype'])
    adata.var['gene_chrom'] = list(features.loc[adata.var.gene_id,'chromosome_name'])
    adata.var['gene_start'] = list(features.loc[adata.var.gene_id,'start_position'])
    adata.var['gene_end'] = list(features.loc[adata.var.gene_id,'end_position'])

    # index genes by symbol (featurekey)
    adata.var['featurekey'] = adata.var.gene_name
    adata.var.index = adata.var.featurekey
    del adata.var['featurekey']

    # make feature name unique
    adata.var_names_make_unique(join='#')

    # save new annotation, reusing X from input on disk
    sct.io.write_h5ad_with_new_annotation(input, adata, output)

################################################################################

# add gene annotation
gene_anno(input=args.input,
          output=args.output,
          anno=args.anno)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
