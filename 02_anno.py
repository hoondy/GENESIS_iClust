#!/usr/bin/env python
"""
Step 02 - Annotate genes and switch var names from Ensembl IDs to gene symbols.

var (Ensembl gene IDs) is joined to the gene table
(annotation/all_genes_EnsemblInfo_104_1to1.csv) and gets these columns:

    gene_id     Ensembl gene ID (former var index)
    gene_name   gene symbol, or the Ensembl ID when the gene has no symbol of its own
    gene_type   biotype, e.g. protein_coding (used by 03_qc.py)
    gene_chrom  chromosome, e.g. '1', 'X', 'MT' (used to find autosomes)
    gene_start, gene_end

var is then re-indexed by gene_name. Every gene must get its own name: if two
Ensembl IDs would share one, the step stops with an error instead of renaming
one of them (rc2 and earlier renamed it GENE#1). Only the annotations are
rewritten; X is copied from the input unchanged.

The gene table is Ensembl release 104 (GRCh38, all_genes_EnsemblInfo_104.csv)
with two changes:
  - Only genes on the primary assembly are kept: chr1-22, X, Y, MT and the
    unplaced/unlocalized scaffolds (GL*/KI*), which hold 19 genes expressed in
    rc2. Alternate haplotypes and patches (CHR_*) are left out; they repeat
    primary-assembly genes and caused most duplicate symbols.
  - Each symbol belongs to one Ensembl ID. Ensembl 104 gives 18 symbols on the
    primary chromosomes to two IDs. TBCE goes to ENSG00000284770 (ENSG00000285053
    is the GGPS1-TBCE readthrough) and CCDC39 to ENSG00000284862 (ENSG00000145075
    is a lncRNA). For the other 16, the ID named in GENESIS rc2 keeps the symbol,
    or neither ID does if neither was in rc2. An ID that loses its symbol is named
    by its Ensembl ID, as are genes without an HGNC symbol (hgnc_symbol is filled
    in the table; external_gene_name still holds the Ensembl 104 symbol).
Extra columns record the rc2 name of each gene and are not used here:
rc2_var_name (e.g. TBCE#1), rc2_diff_symbol (1 if it differs from hgnc_symbol)
and rc2_n_cohorts (number of rc2 cohorts, out of 15, that kept the gene).

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
    # (a no-op for the 1to1 table, which already fills hgnc_symbol)
    features.loc[features.hgnc_symbol.isna(),'hgnc_symbol'] = features.loc[features.hgnc_symbol.isna(),'ensembl_gene_id']

    # merge annotation (in var order)
    adata.var['gene_name'] = list(features.loc[adata.var.gene_id,'hgnc_symbol'])
    adata.var['gene_type'] = list(features.loc[adata.var.gene_id,'gene_biotype'])
    adata.var['gene_chrom'] = list(features.loc[adata.var.gene_id,'chromosome_name'])
    adata.var['gene_start'] = list(features.loc[adata.var.gene_id,'start_position'])
    adata.var['gene_end'] = list(features.loc[adata.var.gene_id,'end_position'])

    # every gene must get its own name. var_names_make_unique would rename later copies GENE#1,
    # and the bare name then goes to whichever ID comes first in var order, not to the right gene
    # (in rc2, TBCE was the GGPS1-TBCE readthrough and the real TBCE was TBCE#1).
    shared = adata.var[adata.var.gene_name.duplicated(keep=False)]
    if len(shared) > 0:
        examples = shared.groupby('gene_name').gene_id.agg(', '.join)
        raise ValueError(f"{len(examples)} gene names are shared by more than one Ensembl ID, e.g. "
                         + "; ".join(f"{name}: {ids}" for name, ids in examples.head(5).items())
                         + f". Use a gene table that gives each Ensembl ID its own name (--anno {anno}).")

    # index genes by symbol (featurekey)
    adata.var['featurekey'] = adata.var.gene_name
    adata.var.index = adata.var.featurekey
    del adata.var['featurekey']

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
