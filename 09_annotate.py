#!/usr/bin/env python
"""
Step 09 - Apply the curated annotation and re-embed (release step 1 of 3).

    1. Keep only cells whose subtype is listed in --annotation. Subtypes left
       out of the table are dropped.
    2. Replace the numeric iclust class/subclass labels with the curated
       names mapped from each cell's subtype.
    3. Re-select HVFs and recompute PCA/Harmony/kNN/UMAP on all remaining
       cells, then plot class/subclass UMAPs with the GENESIS palette
       (00_shared.umap_annotated).

--annotation is a CSV with columns class,subclass,subtype (see
YYMMDD_GEN_TEST_pass3_annotation.csv). The temporary files <input>_tmp1.h5ad
and <input>_tmp2.h5ad are removed at the end.

Snakemake: TEST_rc2.smk, rule `anno`
"""

# sc
import SCTools as sct

# data
import pandas as pd

# sys
import importlib
import os
import sys

# etc
import argparse

# shared pipeline helpers (file name starts with a digit, so no plain `import`)
shared = importlib.import_module('00_shared')

################################################################################

parser = argparse.ArgumentParser(description='Step 09: apply curated annotation and re-embed')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)

parser.add_argument('--annotation', help='Annotation CSV file', required=True)

parser.add_argument('--harmony_batch', help='Batch variable for Harmony', required=False, default='Source')
parser.add_argument('--hvf_n_top_genes', help='Number of highly variable features for UMAP', type=int, required=False, default=6000)

parser.add_argument('--palette', help='Color palette CSV file', required=True)

args = parser.parse_args()

################################################################################

tmp1_h5ad = args.input.replace('.h5ad','_tmp1.h5ad')   # annotated subtypes only
tmp2_h5ad = args.input.replace('.h5ad','_tmp2.h5ad')   # + curated class/subclass

# load
adata = sct.io.read_everything_but_X(args.input)
anno = pd.read_csv(args.annotation)

# subset criteria
subset_obs = (adata.obs['subtype'].isin(anno.subtype.tolist())).tolist()

# run ondisk_subset
sct.io.ondisk_subset(orig_h5ad = args.input,
                     new_h5ad = tmp1_h5ad,
                     subset_obs = subset_obs,
                     chunk_size = shared.ONDISK_CHUNK_SIZE,
                     raw = True)

# add annotation: subtype -> class / subclass
adata = sct.io.read_everything_but_X(tmp1_h5ad)
subtype2class = dict(zip(anno['subtype'],anno['class']))
subtype2subclass = dict(zip(anno['subtype'],anno['subclass']))
adata.obs['class'] = [subtype2class[x] for x in adata.obs.subtype]
adata.obs['subclass'] = [subtype2subclass[x] for x in adata.obs.subtype]

# replace annotation
sct.io.write_h5ad_with_new_annotation(tmp1_h5ad, adata, tmp2_h5ad, raw = True)

# umap (the palette colours are also stored in the output's .uns)
shared.umap_annotated(input_h5ad = tmp2_h5ad,
                      output_h5ad = args.output,
                      palette_csv = args.palette,
                      harmony_batch = args.harmony_batch,
                      hvf_n_top_genes = args.hvf_n_top_genes)

# remove temp files
os.remove(tmp1_h5ad)
os.remove(tmp2_h5ad)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
