#!/usr/bin/env python
"""
Step 05 - Pegasus gene-signature scores.

The QC metrics n_genes, n_counts and percent_mito are not recomputed here: X
is log-normalised by now, so they are kept as computed on raw counts by
03_qc.py.

Adds obs columns:
    G1/S, G2/M, cycle_diff, cycling, predicted_phase
        cell_cycle_human (Tirosh et al. 2016, https://science.sciencemag.org/content/352/6282/189)
    female_score, male_score, gender_score, predicted_gender
        gender_human
    mito_genes, mito_ribo
        mitochondrial_genes_human: the 13 chrM genes, and mitochondrial
        ribosomal genes that are not on chrM
    ribo_genes
        ribosomal_genes_human
    apoptosis
        apoptosis_human
Pegasus skips a signature when too few of its genes are present.

cycle_diff is later regressed out of the PCs (00_shared.REGRESS_VARS).

Snakemake: TEST_pass2.smk, rule `signature_score`
"""

# sc
import pegasus as pg
import SCTools as sct

# sys
import gc
import sys

# etc
import argparse

################################################################################

parser = argparse.ArgumentParser(description='Step 05: gene-signature scores')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)

args = parser.parse_args()

################################################################################

def sig_score(input, output):
    """Add signature scores to obs (see module docstring)."""

    data = pg.read_input(input)

    # cell_cycle, gender, mitochondrial, ribosomal and apoptosis scores (pegasus default random_state=0)
    sct.pp.calc_sig_scores(data)

    data.to_anndata().write(output, compression='gzip')

################################################################################

sig_score(input = args.input,
          output = args.output)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
