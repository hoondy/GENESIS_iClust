#!/usr/bin/env python
"""
Step 10 - Remove lineage-mixed cells and embedding outliers (release step 2 of 3).

    1. sct.tl.detect_lineage_mixing flags nuclei that carry RNA from a cell
       of another class, such as doublets and nuclei with part of another
       cell attached -> obs['mixing_flag']. A nucleus is flagged when the RNA
       from any one other class adds up to at least a quarter of a typical
       nucleus of that class from the same library (--mixing_threshold,
       default 0.25; the amount is in obs['mixing_nuclei']). OPC and Oligo
       count as one class, because transitions between them are real.
    2. sct.tl.detect_outliers flags, among the remaining cells, those that
       sit among cells of another subclass: fewer than a third of their
       nearest cells in the Harmony embedding, i.e. at most 4 of 15, share
       their subclass (--outlier_min_agreement, default 1/3;
       --outlier_n_neighbors, default 15) -> obs['outlier']. These are the
       cells a UMAP draws inside another subclass's island.
       --outlier_method lof restores the earlier rule (local outlier factor
       per subclass), which releases up to rc2 used.
    3. Cells with mixing_flag == 0 and outlier == 0 are written to --output
       (subset on disk).

The input is opened backed, so counts are streamed from raw.X instead of
being loaded into memory, and UMAPs are drawn with save_umap(backed=True).
UMAPs of the removed and the kept cells are written to:
    <input>_{class,subclass}_mixed.png, <input>_{class,subclass}_outlier.png
    <input>_{class,subclass}_clean.png

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
parser.add_argument('--mixing_threshold', help='Flag cells carrying at least this many nucleus equivalents of RNA from any single other class', required=False, default=0.25, type=float)
parser.add_argument('--outlier_method', help='neighbor_label (default) or lof (the rule used up to rc2)', required=False, default='neighbor_label', choices=['neighbor_label', 'lof'])
parser.add_argument('--outlier_n_neighbors', help='neighbor_label: number of nearest cells compared', required=False, default=15, type=int)
parser.add_argument('--outlier_min_agreement', help='neighbor_label: flag cells with fewer than this share of neighbours in their own subclass', required=False, default=1/3, type=float)

args = parser.parse_args()

################################################################################

# palette
palette = shared.load_palette(args.palette)

# load
adata = ad.read_h5ad(args.input, backed="r")   # counts are streamed from raw.X

# lineage mixing
# Each nucleus's raw counts are modelled as a mixture of the subclass expression profiles (EM on
# subclass-specific marker genes). Each other class's share of the marker UMIs is then turned into
# an amount, in nucleus equivalents: it is divided by the typical marker UMIs of that subclass
# (median of its purest half) and by the depth factor of the nucleus's library. Subclass size and
# library depth are fitted jointly by alternating medians, so deeply sequenced glia-rich libraries
# do not inflate glial sizes. mixing_nuclei is the largest amount from any single other class.
# Ambient RNA gives a few hundredths of a nucleus, nuclei with material attached from another cell
# about 0.3-0.6, and doublets about 1; the 0.25 cutoff sits in the valley between these groups.
# The previous rule (foreign share of marker UMIs >= 0.3) depended on the size of the host nucleus.
# It missed large neurons carrying glial RNA and removed small clean nuclei with ordinary ambient RNA.
sct.tl.detect_lineage_mixing(adata,
                             groupby="subclass",
                             lineage_key="class",
                             known_lineage_sets=[("OPC", "Oligo")],   # keeps real OPC to Oligo transitions
                             library_key="Channel",                   # per-library depth and ambient pool
                             score="nuclei",                          # amount of foreign RNA, not its share
                             threshold=args.mixing_threshold,
                             )
mixed = adata[adata.obs["mixing_flag"] == 1].shape[0]
retained = adata[adata.obs["mixing_flag"] == 0].shape[0]
print('Found %i cells (%.2f %%) with mixed lineages. Retaining %i remaining cells.' % (mixed, mixed/(mixed+retained)*100, retained))

# outlier detection, on the cells kept by the mixing step only: removed cells are neither scored
# nor used as neighbours.
# neighbor_label: exact Euclidean kNN over all kept cells in the 30 Harmony components from step 09
# (across subclasses); a cell is an outlier when fewer than outlier_min_agreement of its
# outlier_n_neighbors nearest cells carry its subclass. The final UMAP (step 11) places each cell
# from its 15 nearest neighbours, so such a cell is drawn inside another subclass's island.
# Cut 1/3 (at most 4 of 15): with no removal, nuclei at or below it ended up away from their
# subclass in the re-embedded final UMAP more often than not; nuclei with 5-7 of 15 mostly did not.
# The rule leaves alone small subtypes that sit apart from the rest of their subclass, because
# their neighbours share their subclass.
# outlier_score = share of neighbours with another subclass; outlier_neighbor_label = their most
# common subclass.
# lof (up to rc2): local outlier factor fitted per subclass, with k = 4% of the subclass (100-4,000)
# and cut 1.5. Fitted within one subclass, it cannot see a cell that sits among another subclass,
# and with such a large k it trims whole small subtypes and donor-specific states instead.
if args.outlier_method == 'neighbor_label':
    outlier_options = dict(method="neighbor_label",
                           label_n_neighbors=args.outlier_n_neighbors,
                           label_min_agreement=args.outlier_min_agreement)
else:
    outlier_options = dict(method="lof")
summary = sct.tl.detect_outliers(adata[adata.obs["mixing_flag"] == 0],
                                 groupby="subclass",
                                 use_rep="X_pca_regressed_harmony",
                                 **outlier_options
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

# umap clean
shared.save_umap(adata[keep], color='class', title='class', save=args.input.replace('.h5ad','_class_clean.png'), pal=palette['class'], backed=True)
shared.save_umap(adata[keep], color='subclass', title='subclass', save=args.input.replace('.h5ad','_subclass_clean.png'), pal=palette['subclass'], backed=True)

# run ondisk_subset
sct.io.ondisk_subset(orig_h5ad = args.input,
                     new_h5ad = args.output,
                     subset_obs = keep.tolist(),
                     chunk_size = shared.ONDISK_CHUNK_SIZE,
                     raw = True)

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
