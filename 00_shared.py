#!/usr/bin/env python
"""
Shared helpers for the GENESIS_iClust pipeline steps.

General-purpose single-cell utilities (on-disk H5AD I/O, HVF selection, QC
thresholds, signature scores, outlier detection, ...) live in SCTools and are
called by each step directly (``sct.io``, ``sct.pp``, ``sct.tl``). This module
only holds the pipeline-specific code that more than one step reuses:

    GENOME, MODALITY, REGRESS_VARS, ONDISK_CHUNK_SIZE   shared defaults
    HVF_FLAVOR, HVF_MIN_CELLS                            HVF selection rule
    read_input_with_hvf()  select HVFs with Scanpy, load with Pegasus         (06, 09, 11)
    run_embedding()        PCA > regress-out > Harmony > kNN > [Leiden] > UMAP (06, 09, 11)
    save_umap()            write one UMAP scatter plot to PNG                 (06, 07, 09, 10, 11)
    load_palette()         read the GENESIS class/subclass colour palette     (09, 10, 11)
    umap_annotated()       HVF + embedding + UMAPs + save for annotated data  (09, 11)

The leading digits in the file name rule out a plain ``import`` statement, so
the steps load this module with::

    shared = importlib.import_module('00_shared')

Python puts the running script's directory on ``sys.path``, so 00_shared.py
must sit in the same directory as the numbered scripts.
"""

# pipeline version (git tag v<__version__>)
__version__ = '1.0.0'

# sc
import pegasus as pg
import scanpy as sc
import anndata as ad
import SCTools as sct

# plotting
import matplotlib.pyplot as plt

# data
import numpy as np
import pandas as pd

# sys
import gc

################################################################################
# Shared defaults
################################################################################

GENOME = 'GRCh38'
MODALITY = 'rna'

# obs covariates regressed out of the PCs before Harmony:
# n_counts and percent_mito come from pg.qc_metrics, cycle_diff from 05_score.py.
REGRESS_VARS = ['n_counts', 'percent_mito', 'cycle_diff']

# Number of cells (rows) read per chunk by sct.io.ondisk_subset.
ONDISK_CHUNK_SIZE = 500000

# Annotation levels coloured with the GENESIS palette (`category` column of the palette CSV).
PALETTE_LEVELS = ['class', 'subclass']

# HVF selection, used by every step that embeds cells (06, 09, 11).
# Scanpy's 'cell_ranger' flavor bins genes by percentiles of their mean. In a small
# subset (06 subclusters) many genes have no counts, their identical zero means give
# duplicate bin edges, and Scanpy raises "Bin edges must be unique". Dropping genes
# expressed in fewer than HVF_MIN_CELLS cells of the input avoids that. Such genes
# cannot be variable anyway; they only shifted the bin edges.
HVF_FLAVOR = 'cell_ranger'
HVF_MIN_CELLS = 1

################################################################################
# Embedding
################################################################################

def read_input_with_hvf(h5ad_file, n_top_genes):
    """
    Select highly variable features (HVFs) with Scanpy, then load the file with Pegasus.

    HVFs are the `n_top_genes` genes with the highest normalised dispersion
    (Scanpy HVF_FLAVOR), chosen among protein-coding autosomal genes
    (var['protein_coding'] from 03_qc.py; no chrMT/X/Y) that are expressed in
    at least HVF_MIN_CELLS cells of `h5ad_file`.

    sct.pp.scanpy_hvf_h5ad does the selection and reads only the needed
    matrix columns from disk. The returned Pegasus object has
    ``var['highly_variable_features']`` set, which is what pg.pca uses.

    Parameters
    ----------
    h5ad_file : str
        Log-normalised H5AD (X), with raw counts in raw.X.
    n_top_genes : int
        Number of HVFs to select. Scanpy can return a few more when genes tie
        at the cut-off.

    Returns
    -------
    pegasusio.MultimodalData
    """
    hvg = sct.pp.scanpy_hvf_h5ad(h5ad_file=h5ad_file,
                                 flavor=HVF_FLAVOR,
                                 n_top_genes=n_top_genes,
                                 protein_coding=True,
                                 autosome=True,
                                 min_cells=HVF_MIN_CELLS)
    gc.collect()

    data = pg.read_input(h5ad_file, genome=GENOME, modality=MODALITY)
    data.var['highly_variable_features'] = False
    data.var.loc[data.var.index.isin(hvg), 'highly_variable_features'] = True
    return data

def run_embedding(data, harmony_batch, regress_var=REGRESS_VARS, n_pcs=30, knn_K=100,
                  umap_n_neighbors=15, leiden_res=None, leiden_label='leiden_labels'):
    """
    Compute a batch-corrected embedding of a Pegasus object, in place.

    All steps use random_state=0.

        1. PCA on highly_variable_features            -> obsm['X_pca']
        2. regress `regress_var` out of the PCs       -> obsm['X_pca_regressed']
        3. Harmony on obs[harmony_batch]              -> obsm['X_pca_regressed_harmony']
        4. kNN graph (K=knn_K) on the Harmony space
        5. Leiden clustering, only if leiden_res set  -> obs[leiden_label]
        6. UMAP                                       -> obsm['X_umap']
    """
    ### pca/regress/harmony
    pg.pca(data, n_components=n_pcs, random_state=0)
    pg.regress_out(data, attrs=regress_var)
    pg.run_harmony(data, batch=harmony_batch, rep='pca_regressed', max_iter_harmony=20, n_comps=n_pcs, random_state=0)

    ### kNN/leiden/umap
    pg.neighbors(data, rep='pca_regressed_harmony', use_cache=False, dist='l2', K=knn_K, n_comps=n_pcs, random_state=0)
    if leiden_res is not None:
        pg.leiden(data, rep='pca_regressed_harmony', resolution=leiden_res, class_label=leiden_label, random_state=0)
    pg.umap(data, rep='pca_regressed_harmony', n_neighbors=umap_n_neighbors, rep_ncomps=n_pcs, random_state=0)

################################################################################
# Plotting
################################################################################

def save_umap(adata, color, title, save, pal=None, size=1, legend_fontoutline=2, backed=False):
    """
    Save a UMAP scatter plot coloured by an obs column, with labels drawn on the clusters.

    Parameters
    ----------
    adata : AnnData
        Needs obsm['X_umap'] and obs[color].
    color : str or list of str
        obs column(s) to colour by.
    title : str or None
        Plot title.
    save : str
        Output PNG path.
    pal : dict, optional
        {category: colour}. If None or empty, Scanpy reuses
        uns['<color>_colors'] when it has enough entries, otherwise its
        default palette.
    size, legend_fontoutline : float
        Point size and label outline width. Iterative clustering (06, 07)
        uses size=10, legend_fontoutline=1; the annotated steps (09-11) use
        the defaults.
    backed : bool
        Set True when `adata` is backed (or a view of a backed AnnData).
        Scanpy writes the plot colours to uns['<color>_colors'], which fails
        on a backed view, so the plot is drawn from a minimal in-memory
        AnnData (obs[color] + X_umap) instead and `adata` is left untouched.
        With backed=False the colours are written into adata.uns. For
        `data.to_anndata()` that dict is shared with the Pegasus object, so
        the colours end up in the H5AD saved afterwards.
    """
    sc.set_figure_params(dpi=100, dpi_save=300, vector_friendly=True)
    if backed:
        cols = [color] if isinstance(color, str) else list(color)
        adata = ad.AnnData(obs=adata.obs[cols].copy(), obsm={'X_umap': np.asarray(adata.obsm['X_umap'])})
    sc.pl.umap(adata, color=color, title=title, legend_loc='on data', frameon=False,
               legend_fontsize=6, legend_fontoutline=legend_fontoutline, size=size,
               add_outline=True, outline_width=(0.2, 0.02), show=False, palette=pal)
    plt.savefig(save, bbox_inches='tight')
    plt.close('all')

def load_palette(path, levels=PALETTE_LEVELS):
    """
    Read the GENESIS colour palette CSV into one {name: color_hex} dict per annotation level.

    The CSV (e.g. annotation/260122_GENESIS_color_palette.csv) has one row per
    cell type, with columns `category` (class, subclass, ...), `name` and
    `color_hex`.

    Returns
    -------
    dict
        {level: {name: color_hex}}, e.g. palette['class']['Astro'] == '#D2AF81'
    """
    pal = pd.read_csv(path)
    palette = {}
    for level in levels:
        rows = pal[pal.category == level]
        palette[level] = dict(zip(rows['name'], rows['color_hex']))
    return palette

def umap_annotated(input_h5ad, output_h5ad, palette_csv, harmony_batch, hvf_n_top_genes):
    """
    Re-embed an annotated dataset, plot class/subclass UMAPs and save it (no clustering).

    HVFs come from read_input_with_hvf. The embedding then runs through
    run_embedding with its defaults, and the result is written to
    `output_h5ad`. PNGs are named <output>_class.png and
    <output>_subclass.png. The palette colours are also stored in the
    output .uns (class_colors, subclass_colors), see save_umap.
    """
    data = read_input_with_hvf(input_h5ad, n_top_genes=hvf_n_top_genes)
    run_embedding(data, harmony_batch=harmony_batch)

    ### figure
    palette = load_palette(palette_csv)
    adata = data.to_anndata()
    save_umap(adata, color='class', title='class', save=output_h5ad.replace('.h5ad', '_class.png'), pal=palette['class'])
    save_umap(adata, color='subclass', title='subclass', save=output_h5ad.replace('.h5ad', '_subclass.png'), pal=palette['subclass'])

    ### save
    sct.io.save(data, output_h5ad)

    ### mem
    del data
    gc.collect()
