#region imports and setup #######################################################

import os
import gc
import time
os.environ.setdefault(
    'CUPY_CACHE_DIR', '/tmp/cupy_cache')
os.environ.setdefault(
    'CUDA_PATH', os.path.join(
        os.path.dirname(os.__file__),
        'site-packages', 'nvidia', 'cuda_runtime'))
_torch_cache = os.path.join(os.environ['SCRATCH'], 'torch_kernel_cache')
os.makedirs(_torch_cache, exist_ok=True)
os.environ.setdefault('PYTORCH_KERNEL_CACHE_PATH', _torch_cache)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['R_HOME'] = '/home/karbabi/miniforge3/lib/R'

import torch
print(f'GPU: {torch.cuda.get_device_name(0)} '
      f'({torch.cuda.device_count()} device(s))')

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scvi
scvi.settings.dl_num_workers = 0

working_dir = '/home/karbabi/spatial-pregnancy'
datasets = {
    'slidetags': 'sample',
    'merfish': 'sample',
    'xenium': 'sample_rep',
}
thr = dict(
    subclass_confidence=0.6,    # calibrated [0,1], global
    subclass_margin=0.2,        # calibrated [0,1], global
    cos_dist_pct=0.95,          # per-sample upper pct on min_cos_dist
    pdist_pct=0.95,             # per-sample upper pct on avg_pdist
)
min_cells_per_sample = 10

#endregion
#region decontamination ########################################################

def _as_sparse(x):
    y = x.tocsr() if sp.issparse(x) else sp.csr_array(x)
    return y.astype(np.float32, copy=False)

# Attach corrected counts + contamination; print diagnostics.
def _finalize(adata, corrected, contam, name):
    assert sp.issparse(corrected) and corrected.dtype == np.float32
    assert (corrected.data >= 0).all() and (corrected.data % 1 == 0).all()
    adata.X = corrected
    adata.layers['corrected_counts'] = corrected
    adata.obs['correction_fraction'] = contam.astype(np.float32)
    print(f'[{name}] X head (after):\n{adata.X[:3, :6].toarray()}')
    cf = adata.obs['correction_fraction']
    print(f'[{name}] correction: mean={cf.mean():.1%}, '
          f'median={cf.median():.1%}')
    for cls in sorted(adata.obs['class'].unique()):
        c = cf[adata.obs['class'] == cls]
        print(f'  {cls}: {c.mean():.1%} ({c.shape[0]:,} cells)')

# RESOLVI: spatial, cached model + corrected counts.
# Correction: f = (alpha_0 * px_rate) / mean_poisson   (true only)

def run_resolvi(adata, sample_col, name):
    print(f'[{name}] X head (before):\n{adata.X[:3, :6].toarray()}')
    torch.set_float32_matmul_precision('medium')
    adata.obsm['X_spatial'] = adata.obs[['x_affine', 'y_affine']]\
        .values.astype(np.float32)
    scvi.external.RESOLVI.setup_anndata(
        adata, layer='counts',
        batch_key=sample_col, labels_key='subclass', prepare_data=True,
        prepare_data_kwargs={'n_neighbors': 20, 'spatial_rep': 'X_spatial'})

    model_dir = f'{working_dir}/output/{name}/resolvi_model'
    model_cached = os.path.exists(model_dir)
    if model_cached:
        print(f'[{name}] loading cached RESOLVI model')
        model = scvi.external.RESOLVI.load(model_dir, adata=adata)
    else:
        model = scvi.external.RESOLVI(
            adata, n_latent=10, n_hidden=128, mixture_k=100,
            semisupervised=True)
        model.train(max_epochs=100, lr=1e-3, lr_extra=5e-2)
        model.save(model_dir, overwrite=True)

    base = f'{working_dir}/output/{name}'
    corr_path = f'{base}/resolvi_corrected_full.npz'
    cf_path = f'{base}/resolvi_contam_full.npy'
    if model_cached and os.path.exists(corr_path) and os.path.exists(cf_path):
        print(f'[{name}] loading cached corrected counts')
        corrected = _as_sparse(sp.load_npz(corr_path))
        mean_cf = np.load(cf_path)
    else:
        # Scale batch inversely with gene count; chunk=batch so each batch
        # is cached immediately (resilient to OOM / job kills mid-run).
        # Peak GPU ~ batch * n_neighbors * genes (px_rate_n is the blowup).
        n_cells = adata.n_obs
        batch_size = min(16384, max(256, 80_000_000 // adata.n_vars))
        chunk_size = batch_size
        print(f'[{name}] batch_size={batch_size} '
              f'(n_vars={adata.n_vars:,})')
        X = _as_sparse(adata.layers['counts'])
        chunk_dir = f'{base}/resolvi_chunks'
        os.makedirs(chunk_dir, exist_ok=True)
        chunks = []
        mean_cf = np.zeros(n_cells, np.float32)
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            ck_x = f'{chunk_dir}/{start:09d}_full.npz'
            ck_cf = f'{chunk_dir}/{start:09d}_full_cf.npy'
            if os.path.exists(ck_x) and os.path.exists(ck_cf):
                chunks.append(sp.load_npz(ck_x))
                mean_cf[start:end] = np.load(ck_cf)
                print(f'  [{name}] {end:,}/{n_cells:,} cached')
                continue
            t0 = time.time()
            post = model.sample_posterior(
                indices=np.arange(start, end), num_samples=30,
                return_sites=[
                    'mean_poisson', 'true_mixture_proportion', 'px_rate'],
                batch_size=batch_size, show_progress=False)
            t_post = time.time() - t0
            t0 = time.time()
            m = post['post_sample_means']
            mp = m['mean_poisson'] + 1e-8
            f = np.clip(
                (m['true_mixture_proportion'] * m['px_rate']) / mp,
                0, 1).astype(np.float32)
            X_chunk = X[start:end].toarray()
            c_chunk = np.round(X_chunk * f).astype(np.float32)
            w = X_chunk.sum(1)
            mean_cf[start:end] = np.divide(
                c_chunk.sum(1), w,
                out=np.ones_like(w, dtype=np.float32), where=(w > 0))
            sp_chunk = sp.csr_array(c_chunk)
            sp_chunk.eliminate_zeros()
            sp.save_npz(ck_x, sp_chunk)
            np.save(ck_cf, mean_cf[start:end])
            chunks.append(sp_chunk)
            t_apply = time.time() - t0
            del post, m, mp, f, X_chunk, c_chunk
            gc.collect()
            torch.cuda.empty_cache()
            print(f'  [{name}] {end:,}/{n_cells:,} '
                  f'post={t_post:.1f}s apply={t_apply:.1f}s '
                  f'GPU={torch.cuda.memory_reserved()/1e9:.1f}GB')
        corrected = _as_sparse(sp.vstack(chunks, format='csr'))
        del chunks
        sp.save_npz(corr_path, corrected)
        np.save(cf_path, mean_cf)

    _finalize(adata, corrected, 1 - mean_cf, name)
    # restore sparse counts to avoid h5ad bloat
    if not sp.issparse(adata.layers['counts']):
        adata.layers['counts'] = sp.csr_matrix(adata.layers['counts'])
    for k in ('X_spatial', 'index_neighbor', 'distance_neighbor'):
        adata.obsm.pop(k, None)
    return adata

#endregion
#region process ################################################################

for name, sample_col in datasets.items():
    in_path = f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad'
    out_path = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    adata = sc.read_h5ad(in_path)
    n_total = len(adata)
    print(f'\n[{name}] {n_total:,} cells')

    adata = run_resolvi(adata, sample_col, name)

    obs = adata.obs
    # per-sample upper-percentile cutoffs: drop cells above sample-specific
    # quantile for min_cos_dist and avg_pdist (distributions differ by
    # platform/sample, so a global threshold over- or under-filters).
    cos_cut = obs.groupby(sample_col, observed=True)['min_cos_dist']\
        .transform(lambda s: s.quantile(thr['cos_dist_pct']))
    pd_cut = obs.groupby(sample_col, observed=True)['avg_pdist']\
        .transform(lambda s: s.quantile(thr['pdist_pct']))
    masks = {
        'subclass_confidence':
            obs['subclass_confidence'] >= thr['subclass_confidence'],
        'subclass_margin':
            obs['subclass_margin'] >= thr['subclass_margin'],
        'min_cos_dist':
            obs['min_cos_dist'] < cos_cut,
        'avg_pdist':
            obs['avg_pdist'] < pd_cut,
    }
    # drop (sample, subclass) combos with < min_cells_per_sample
    rare = obs.groupby([sample_col, 'subclass'], observed=True).size()
    rare = rare[rare < min_cells_per_sample]
    obs_mi = pd.MultiIndex.from_arrays(
        [obs[sample_col].values, obs['subclass'].values])
    masks['rare_subclass'] = ~obs_mi.isin(rare.index)

    for k, m in masks.items():
        n_drop = (~m).sum()
        print(f'  drop {k}: {n_drop:,} ({n_drop/n_total*100:.1f}%)')

    keep = np.logical_and.reduce(list(masks.values()))
    n_dropped = n_total - keep.sum()
    print(f'  total dropped: {n_dropped:,} ({n_dropped/n_total*100:.1f}%)')
    print(f'  keep: {keep.sum():,} ({keep.sum()/n_total*100:.1f}%)')

    adata[keep].copy().write(out_path)
    print(f'[{name}] saved {out_path}')

#endregion
