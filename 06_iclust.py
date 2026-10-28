#!/usr/bin/env python
"""
Step 06 - Iterative three-level Leiden clustering: class > subclass > subtype.

    1. cluster all cells                      -> obs['class']     '1', '2', ...
    2. subset each class on disk, re-cluster  -> obs['subclass']  '<class>_1', '<class>_2', ...
    3. subset each subclass, re-cluster       -> obs['subtype']   '<subclass>_1', ...

Every clustering re-selects its own HVFs (00_shared.read_input_with_hvf: 'cell_ranger'
on protein-coding autosomal genes expressed in the subset) and recomputes PCA, regression,
Harmony, kNN, Leiden and UMAP on its subset (00_shared.run_embedding), so
structure inside a population is not dominated by differences between
populations.

Files written (prefix = --prefix):
    <prefix>_class.h5ad/.png                 level 1, all cells
    <prefix>_class_<class>.h5ad/.png         level 2, one per class
    <prefix>_subclass_<subclass>.h5ad/.png   level 3, one per subclass
    <output> with .h5ad replaced by .csv     class/subclass/subtype of every cell
    <output>                                 <prefix>_class.h5ad with all three labels in obs (as str)

Snakemake: TEST_pass2.smk rules `pass1` and `pass2`, TEST_pass3.smk rule `pass3`
"""

# sc
import SCTools as sct

# data
import pandas as pd

# sys
import gc
import importlib
import sys

# etc
import argparse

# shared pipeline helpers (file name starts with a digit, so no plain `import`)
shared = importlib.import_module('00_shared')

################################################################################

parser = argparse.ArgumentParser(description='Step 06: iterative Leiden clustering (class > subclass > subtype)')

parser.add_argument('--input', help='Input h5ad file', required=True)
parser.add_argument('--output', help='Output h5ad file', required=True)
parser.add_argument('--prefix', help='Prefix for the per-level h5ad/png files', required=True)
parser.add_argument('--batch', help='obs column used as the Harmony batch', required=False, default='Source')
parser.add_argument('--iclust_hvfs', help='List of highly variable features for class, subclass, subtype', nargs="+", type=int, required=False, default=[3000,2000,1000])
parser.add_argument('--iclust_res', help='List of Leiden clustering resolutions for class, subclass, subtype', nargs="+", type=float, required=False, default=[0.1,0.2,0.3])

args = parser.parse_args()

################################################################################

def _pg_leiden_clustering(args_input, args_output_prefix, args_hvf_n_top_genes, args_leiden_res, args_leiden_label,
                          args_leiden_label_prefix = None, args_regress_var = shared.REGRESS_VARS, args_harmony_batch = 'Source'):
    """
    Cluster the cells of one h5ad and save the result as <args_output_prefix>.h5ad/.png.

    Leiden labels go to obs[args_leiden_label]. If args_leiden_label_prefix is
    set, labels are prefixed with it, e.g. '1' -> '3_1' when re-clustering
    class '3'.
    """

    ### highly variable features (protein-coding autosomal genes expressed in this subset) + load
    data = shared.read_input_with_hvf(args_input, n_top_genes=args_hvf_n_top_genes)

    ### pca/regress/harmony/kNN/leiden/umap
    shared.run_embedding(data,
                         harmony_batch=args_harmony_batch,
                         regress_var=args_regress_var,
                         leiden_res=args_leiden_res,
                         leiden_label=args_leiden_label)

    ### prefix leiden labels with the parent cluster label
    if args_leiden_label_prefix:
        categories = data.obs[args_leiden_label].cat.categories
        data.obs[args_leiden_label] = data.obs[args_leiden_label].cat.rename_categories([args_leiden_label_prefix+'_'+x for x in categories])

    ### figure
    shared.save_umap(adata=data.to_anndata(), color=args_leiden_label, title=args_leiden_label, save=args_output_prefix+'.png',
                     size=10, legend_fontoutline=1)

    ### save
    sct.io.save(data, args_output_prefix+'.h5ad')

    ### mem
    del data
    gc.collect()

def _subcluster(parent_h5ad, parent_obs, parent_level, parent_label, level, n_hvf, res, prefix, regress_var, harmony_batch):
    """
    Re-cluster the cells labelled `parent_label` in obs[parent_level] at the next level, `level`.

    The cells are subset on disk to <prefix>_<parent_level>_<parent_label>.h5ad,
    which the clustering then overwrites. Returns the path of that file and
    its annotations (no X).
    """
    print('ondisk subset %s using [%s] level annotation' % (parent_label, parent_level))

    subset_prefix = prefix+'_'+parent_level+'_'+parent_label

    # run ondisk_subset
    sct.io.ondisk_subset(orig_h5ad = parent_h5ad,
                         new_h5ad = subset_prefix+'.h5ad',
                         subset_obs = (parent_obs[parent_level]==parent_label).tolist(),
                         chunk_size = shared.ONDISK_CHUNK_SIZE,
                         raw = True)

    # iclust
    _pg_leiden_clustering(args_input = subset_prefix+'.h5ad',
                          args_output_prefix = subset_prefix,
                          args_hvf_n_top_genes = n_hvf,
                          args_leiden_res = res,
                          args_leiden_label = level,
                          args_leiden_label_prefix = parent_label,
                          args_regress_var = regress_var,
                          args_harmony_batch = harmony_batch)

    return subset_prefix+'.h5ad', sct.io.read_everything_but_X(subset_prefix+'.h5ad')

def iclust(input,
           output,
           prefix,
           args_iclust_levels = ['class','subclass','subtype'],
           args_iclust_hvfs = [3000,2000,1000],
           args_iclust_res = [0.1,0.2,0.3],
           args_regress_var = shared.REGRESS_VARS,
           args_harmony_batch = 'Source'):
    """
    Run the three-level clustering (see module docstring).

    args_iclust_levels, args_iclust_hvfs and args_iclust_res list, for each
    level, the obs column name, the number of HVFs and the Leiden resolution.
    """
    levels, hvfs, res = args_iclust_levels, args_iclust_hvfs, args_iclust_res
    common = dict(prefix=prefix, regress_var=args_regress_var, harmony_batch=args_harmony_batch)

    anno = pd.DataFrame()

    # base level: all cells
    base_prefix = prefix+'_'+levels[0]

    print('[%s] level clustering' % (levels[0]))
    _pg_leiden_clustering(args_input = input,
                          args_output_prefix = base_prefix,
                          args_hvf_n_top_genes = hvfs[0],
                          args_leiden_res = res[0],
                          args_leiden_label = levels[0],
                          args_regress_var = args_regress_var,
                          args_harmony_batch = args_harmony_batch)

    # subsequent levels: re-cluster each cluster of the level above
    base_adata = sct.io.read_everything_but_X(base_prefix+'.h5ad')

    for l1 in base_adata.obs[levels[0]].cat.categories.tolist():
        l1_h5ad, l1_adata = _subcluster(base_prefix+'.h5ad', base_adata.obs, levels[0], l1, levels[1], hvfs[1], res[1], **common)

        for l2 in l1_adata.obs[levels[1]].cat.categories.tolist():
            _, l2_adata = _subcluster(l1_h5ad, l1_adata.obs, levels[1], l2, levels[2], hvfs[2], res[2], **common)

            # leaf level: collect labels of all levels
            anno = pd.concat([anno, l2_adata.obs[levels]])

    # annotation
    anno.to_csv(output.replace('.h5ad','.csv'))

    # ondisk update anno: write labels into the base-level annotations, reusing its X/raw.X
    for x in levels:
        base_adata.obs[x] = anno.loc[base_adata.obs.index,x].astype(str)
    sct.io.write_h5ad_with_new_annotation(orig_h5ad = base_prefix+'.h5ad', adata = base_adata, new_h5ad = output)

################################################################################

iclust(input = args.input,
       output = args.output,
       prefix = args.prefix,
       args_iclust_levels = ['class','subclass','subtype'],
       args_iclust_hvfs = args.iclust_hvfs,
       args_iclust_res = args.iclust_res,
       args_harmony_batch = args.batch)
gc.collect()

################################################################################

print(f"Script {sys.argv[0]} completed successfully.")
