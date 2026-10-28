#!/usr/bin/env python
"""
Step 10 - Remove lineage-mixed cells and embedding outliers (release step 2 of 3).

    1. sct.tl.detect_lineage_mixing flags cells whose raw counts look like a
       mixture of subclass profiles from different classes, such as
       heterotypic doublets or ambient contamination -> obs['mixing_flag'].
       OPC and Oligo count as one lineage, because transitions between them
       are real.
    2. sct.tl.detect_outliers runs LOF per subclass in the Harmony embedding
       on the remaining cells -> obs['outlier'].
    3. Cells with mixing_flag == 0 and outlier == 0 are written to --output
       (subset on disk).

The input is opened backed, so counts are streamed from raw.X instead of
being loaded into memory, and UMAPs are drawn with save_umap(backed=True).
UMAPs of the removed and the kept cells are written to:
    <input>_{class,subclass}_mixed.png, <input>_{class,subclass}_outlier.png
    <output>_{class,subclass}.png

Snakemake: TEST_rc2.smk, rule `clean`
"""

# sc
import anndata as ad
import SCTools as sct

# sys
import importlib
import sys

# etc
import argparse

# shared pipeline helpers (file name starts with a digit, so no plain `import`)
shared = importlib.import_module('00_shared')

################################################################################

parser = argparse.ArgumentParser(description='Step 10: remove lineage-mixed cells and outliers')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)

parser.add_argument('--palette', help='Color palette CSV file', required=True)

args = parser.parse_args()

################################################################################

# palette
palette = shared.load_palette(args.palette)

# load
adata = ad.read_h5ad(args.input, backed="r")   # counts are streamed from raw.X

# lineage mixing
sct.tl.detect_lineage_mixing(adata,
                             groupby="subclass",
                             lineage_key="class",
                             known_lineage_sets=[("OPC", "Oligo")],   # keeps real OPC to Oligo transitions
                             )
mixed = adata[adata.obs["mixing_flag"] == 1].shape[0]
retained = adata[adata.obs["mixing_flag"] == 0].shape[0]
print('Found %i cells (%.2f %%) with mixed lineages. Retaining %i remaining cells.' % (mixed, mixed/(mixed+retained)*100, retained))

# outlier detection
summary = sct.tl.detect_outliers(adata[adata.obs["mixing_flag"] == 0],
                                 groupby="subclass",
                                 use_rep="X_pca_regressed_harmony",
                                 method="lof"
                                )
print(summary)

# umap mixed lineage
shared.save_umap(adata[adata.obs["mixing_flag"]==1], color='class', title='class', save=args.input.replace('.h5ad','_class_mixed.png'), pal=palette['class'], backed=True)
shared.save_umap(adata[adata.obs["mixing_flag"]==1], color='subclass', title='subclass', save=args.input.replace('.h5ad','_subclass_mixed.png'), pal=palette['subclass'], backed=True)

# umap outlier
shared.save_umap(adata[adata.obs["outlier"]==1], color='class', title='class', save=args.input.replace('.h5ad','_class_outlier.png'), pal=palette['class'], backed=True)
shared.save_umap(adata[adata.obs["outlier"]==1], color='subclass', title='subclass', save=args.input.replace('.h5ad','_subclass_outlier.png'), pal=palette['subclass'], backed=True)

# subset criteria
keep = (adata.obs["outlier"]==0)&(adata.obs["mixing_flag"]==0)

# run ondisk_subset
sct.io.ondisk_subset(orig_h5ad = args.input,
                     new_h5ad = args.output,
                     subset_obs = keep.tolist(),
                     chunk_size = shared.ONDISK_CHUNK_SIZE,
                     raw = True)

# umap clean
shared.save_umap(adata[keep], color='class', title='class', save=args.output.replace('.h5ad','_class.png'), pal=palette['class'], backed=True)
shared.save_umap(adata[keep], color='subclass', title='subclass', save=args.output.replace('.h5ad','_subclass.png'), pal=palette['subclass'], backed=True)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
