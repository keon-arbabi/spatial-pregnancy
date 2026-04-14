#region imports and setup #######################################################

import os
import sys
os.environ.setdefault(
    'CUPY_CACHE_DIR', '/tmp/cupy_cache')
os.environ.setdefault(
    'CUDA_PATH', os.path.join(os.path.dirname(os.__file__),
    'site-packages', 'nvidia', 'cuda_runtime'))
import torch
import scvi
import numpy as np
import scanpy as sc
import scipy.sparse as sp
os.environ['R_HOME'] = '/home/karbabi/miniforge3/lib/R'
from ryp import r, to_r, to_py
sys.path.insert(0, os.path.expanduser('~'))
from single_cell import SingleCell

working_dir = '/home/karbabi/spatial-pregnancy'
datasets = {
    'slidetags': 'sample',
    'xenium': 'sample_rep',
    'merfish': 'sample',
}
thr = dict(
    subclass_confidence=0.6,
    subclass_margin=0.2,
    min_cos_dist=0.3,
    n_spatial_candidates=1,
)
min_cells_per_sample = 10

#endregion
#region decontamination ########################################################

def print_correction(adata, name):
    cf = adata.obs['correction_fraction']
    print(f'[{name}] correction: mean={cf.mean():.1%}, '
          f'median={cf.median():.1%}')
    for cls in sorted(adata.obs['class'].unique()):
        c = cf[adata.obs['class'] == cls]
        print(f'  {cls}: {c.mean():.1%} ({c.shape[0]:,} cells)')

# SoupX for slidetags

def run_soupx(adata, sample_col, name):
    r('''
    suppressPackageStartupMessages(library(SoupX))
    suppressPackageStartupMessages(library(Matrix))
    ''')
    samples = adata.obs[sample_col].unique()
    corrected_chunks = []
    for sample in samples:
        mask = adata.obs[sample_col] == sample
        a_sub = adata[mask].copy()
        SingleCell(a_sub).skip_qc().to_sce('sce')
        to_r(str(sample), 'sample_name')
        r('''
        toc = counts(sce)
        clusters = factor(colData(sce)$subclass)
        # no raw/unfiltered matrix: estimate soup profile from filtered cells
        sc = SoupChannel(toc, toc, calcSoupProfile=FALSE)
        soupProf = data.frame(
            row.names=rownames(toc),
            est=rowSums(toc) / sum(toc),
            counts=rowSums(toc))
        sc = setSoupProfile(sc, soupProf)
        sc = setClusters(sc, clusters)
        sc = autoEstCont(sc, doPlot=FALSE)
        cat(sprintf("[%s] SoupX rho=%.3f\\n", sample_name, sc$fit$rhoEst))
        adj = adjustCounts(sc, roundToInt=TRUE)
        ''')
        adj = to_py('adj').T
        if sp.issparse(adj):
            adj = adj.tocsr()
        corrected_chunks.append(adj)
        r('rm(sce, toc, sc, soupProf, clusters, adj); gc()')

    corrected = sp.vstack(corrected_chunks, format='csr').astype(np.float32)
    assert (corrected.data >= 0).all() and \
        np.allclose(corrected.data, corrected.data.astype(int))
    orig_total = np.array(adata.X.sum(axis=1)).ravel()
    corr_total = np.array(corrected.sum(axis=1)).ravel()
    adata.obs['correction_fraction'] = np.where(
        orig_total > 0, 1 - corr_total / orig_total, 0).astype(np.float32)
    adata.X = corrected
    assert sp.issparse(adata.X) and adata.X.dtype == np.float32
    print_correction(adata, name)
    return adata

# RESOLVI for xenium and merfish

def run_resolvi(adata, sample_col, name):
    torch.set_float32_matmul_precision('medium')
    adata.obsm['X_spatial'] = adata.obs[['x_affine', 'y_affine']].values.astype(
        np.float32)
    scvi.external.RESOLVI.setup_anndata(
        adata,
        layer='counts',
        batch_key=sample_col,
        labels_key='subclass',
        prepare_data=True,
        prepare_data_kwargs={'n_neighbors': 20, 'spatial_rep': 'X_spatial'})
    model_dir = f'{working_dir}/output/{name}/resolvi_model'
    if os.path.exists(model_dir):
        print(f'[{name}] loading cached RESOLVI model')
        model = scvi.external.RESOLVI.load(model_dir, adata=adata)
    else:
        model = scvi.external.RESOLVI(
            adata, n_latent=10, n_hidden=32, mixture_k=100,
            semisupervised=True)
        model.train(max_epochs=100, lr=1e-3)
        model.save(model_dir, overwrite=True)
    corrected = model.get_normalized_expression(
        library_size=None, n_samples=30, batch_size=512,
        return_numpy=True)
    np.round(corrected, out=corrected)
    corrected = sp.csr_array(corrected.astype(np.float32))
    corrected.eliminate_zeros()
    assert (corrected.data >= 0).all() and \
        np.allclose(corrected.data, corrected.data.astype(int))
    orig_total = np.array(adata.X.sum(axis=1)).ravel()
    corr_total = np.array(corrected.sum(axis=1)).ravel()
    adata.obs['correction_fraction'] = np.where(
        orig_total > 0, 1 - corr_total / orig_total, 0).astype(np.float32)
    adata.X = corrected
    assert sp.issparse(adata.X) and adata.X.dtype == np.float32
    print_correction(adata, name)
    del adata.obsm['X_spatial'], adata.obsm['index_neighbor'], \
        adata.obsm['distance_neighbor']
    return adata

#endregion
#region process ################################################################

for name, sample_col in datasets.items():
    in_path = f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad'
    out_path = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    adata = sc.read_h5ad(in_path)
    n_total = len(adata)
    print(f'\n[{name}] {n_total:,} cells')

    # decontaminate before filtering
    if name == 'slidetags':
        adata = run_soupx(adata, sample_col, name)
    else:
        adata = run_resolvi(adata, sample_col, name)

    # cell type confidence filtering
    obs = adata.obs
    masks = {
        'subclass_confidence':
            obs['subclass_confidence'] >= thr['subclass_confidence'],
        'subclass_margin':
            obs['subclass_margin'] >= thr['subclass_margin'],
        'min_cos_dist':
            obs['min_cos_dist'] < thr['min_cos_dist'],
        'n_spatial_candidates':
            obs['n_spatial_candidates'] >= thr['n_spatial_candidates'],
    }
    for k, m in masks.items():
        n_drop = (~m).sum()
        print(f'  drop {k}: {n_drop:,} ({n_drop/n_total*100:.1f}%)')

    # drop cells of subclasses with < min_cells_per_sample in their sample
    counts = obs.groupby([sample_col, 'subclass'], observed=True).size()
    rare = counts[counts < min_cells_per_sample]
    rare_pairs = set(rare.index)
    rare_mask = np.array([
        (s, c) not in rare_pairs
        for s, c in zip(obs[sample_col].values, obs['subclass'].values)])
    n_rare_drop = (~rare_mask).sum()
    print(f'  drop rare (<{min_cells_per_sample}/sample): '
          f'{n_rare_drop:,} ({n_rare_drop/n_total*100:.1f}%)')
    masks['rare_subclass'] = rare_mask

    keep = np.logical_and.reduce(list(masks.values()))
    n_keep = keep.sum()
    n_dropped = n_total - n_keep
    print(f'  total dropped: {n_dropped:,} ({n_dropped/n_total*100:.1f}%)')
    print(f'  keep: {n_keep:,} ({n_keep/n_total*100:.1f}%)')

    adata = adata[keep].copy()
    adata.write(out_path)
    print(f'[{name}] saved {out_path}')

#endregion
