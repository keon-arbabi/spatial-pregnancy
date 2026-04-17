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
assert torch.cuda.is_available(), 'CUDA GPU required for RESOLVI'
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
    # 'merfish': 'sample',
    # 'xenium': 'sample_rep',
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

def _as_sparse(x):
    y = x.tocsr() if sp.issparse(x) else sp.csr_array(x)
    return y.astype(np.float32, copy=False)

# Attach corrected counts (bg-only as X) + both contamination flavors.
def _finalize(adata, corrected_bg, contam_bg, corrected_full, contam_full,
              name):
    for c in (corrected_bg, corrected_full):
        assert sp.issparse(c) and c.dtype == np.float32
        assert (c.data >= 0).all() and (c.data % 1 == 0).all()
    adata.X = corrected_bg
    adata.layers['corrected_counts_bg'] = corrected_bg
    adata.layers['corrected_counts_full'] = corrected_full
    adata.obs['correction_fraction_bg'] = contam_bg.astype(np.float32)
    adata.obs['correction_fraction_full'] = contam_full.astype(np.float32)
    print(f'[{name}] X head (after, bg):\n{adata.X[:3, :6].toarray()}')
    print(f'[{name}] full head:\n{corrected_full[:3, :6].toarray()}')
    for label, cf in [('bg', adata.obs['correction_fraction_bg']),
                      ('full', adata.obs['correction_fraction_full'])]:
        print(f'[{name}] correction ({label}): mean={cf.mean():.1%}, '
              f'median={cf.median():.1%}')
        for cls in sorted(adata.obs['class'].unique()):
            c = cf[adata.obs['class'] == cls]
            print(f'  {cls}: {c.mean():.1%} ({c.shape[0]:,} cells)')

# RESOLVI: spatial, cached model + corrected counts (full + bg variants)

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

    # Two correction flavors computed from the same posterior:
    #   full: f = (alpha_0 * px_rate) / mean_poisson   (true only)
    #   bg:   f = 1 - background / mean_poisson        (true + diffusion)
    paths = {
        k: (f'{working_dir}/output/{name}/resolvi_corrected_{k}.npz',
            f'{working_dir}/output/{name}/resolvi_contam_{k}.npy')
        for k in ('full', 'bg')
    }
    all_cached = model_cached and all(
        os.path.exists(p) for t in paths.values() for p in t)
    if all_cached:
        print(f'[{name}] loading cached corrected counts (full+bg)')
        corrected_full = _as_sparse(sp.load_npz(paths['full'][0]))
        corrected_bg = _as_sparse(sp.load_npz(paths['bg'][0]))
        cf_full = np.load(paths['full'][1])
        cf_bg = np.load(paths['bg'][1])
    else:
        # Scale batch inversely with gene count; chunk=batch so each batch
        # is cached immediately (resilient to OOM / job kills mid-run).
        # Peak GPU ~ batch * n_neighbors * genes (px_rate_n is the blowup).
        # Target: 80M (batch*genes); leaves ~40 GB peak on H100 at 20 nbrs,
        # tolerant of a competing 30 GB process.
        n_cells = adata.n_obs
        batch_size = min(16384, max(256, 80_000_000 // adata.n_vars))
        chunk_size = batch_size
        print(f'[{name}] batch_size={batch_size} '
              f'(n_vars={adata.n_vars:,}, chunks=batches)')
        X = _as_sparse(adata.layers['counts'])
        chunk_dir = f'{working_dir}/output/{name}/resolvi_chunks'
        os.makedirs(chunk_dir, exist_ok=True)
        chunks = {'full': [], 'bg': []}
        cf = {'full': np.zeros(n_cells, np.float32),
              'bg': np.zeros(n_cells, np.float32)}
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            cks = {k: (f'{chunk_dir}/{start:09d}_{k}.npz',
                       f'{chunk_dir}/{start:09d}_{k}_cf.npy')
                   for k in ('full', 'bg')}
            if all(os.path.exists(p) for t in cks.values() for p in t):
                for k in ('full', 'bg'):
                    chunks[k].append(sp.load_npz(cks[k][0]))
                    cf[k][start:end] = np.load(cks[k][1])
                print(f'  [{name}] {end:,}/{n_cells:,} cached')
                continue
            t0 = time.time()
            post = model.sample_posterior(
                indices=np.arange(start, end), num_samples=30,
                return_sites=['mean_poisson', 'true_mixture_proportion',
                              'px_rate', 'background'],
                batch_size=batch_size, show_progress=False)
            t_post = time.time() - t0
            t0 = time.time()
            m = post['post_sample_means']
            mp = m['mean_poisson'] + 1e-8
            f = {
                'full': np.clip(
                    (m['true_mixture_proportion'] * m['px_rate']) / mp,
                    0, 1).astype(np.float32),
                'bg': np.clip(1 - m['background'] / mp,
                              0, 1).astype(np.float32),
            }
            X_chunk = X[start:end].toarray()
            w = X_chunk.sum(1)
            for k in ('full', 'bg'):
                c_chunk = np.round(X_chunk * f[k]).astype(np.float32)
                cf[k][start:end] = np.divide(
                    c_chunk.sum(1), w,
                    out=np.ones_like(w, dtype=np.float32), where=(w > 0))
                sp_chunk = sp.csr_array(c_chunk)
                sp_chunk.eliminate_zeros()
                sp.save_npz(cks[k][0], sp_chunk)
                np.save(cks[k][1], cf[k][start:end])
                chunks[k].append(sp_chunk)
            t_apply = time.time() - t0
            del post, m, mp, f, X_chunk, c_chunk
            gc.collect()
            torch.cuda.empty_cache()
            print(f'  [{name}] {end:,}/{n_cells:,} '
                  f'post={t_post:.1f}s apply={t_apply:.1f}s '
                  f'GPU={torch.cuda.memory_reserved()/1e9:.1f}GB')
        corrected_full = _as_sparse(sp.vstack(chunks['full'], format='csr'))
        corrected_bg = _as_sparse(sp.vstack(chunks['bg'], format='csr'))
        cf_full, cf_bg = cf['full'], cf['bg']
        del chunks
        sp.save_npz(paths['full'][0], corrected_full)
        sp.save_npz(paths['bg'][0], corrected_bg)
        np.save(paths['full'][1], cf_full)
        np.save(paths['bg'][1], cf_bg)

    _finalize(adata, corrected_bg, 1 - cf_bg,
              corrected_full, 1 - cf_full, name)
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
