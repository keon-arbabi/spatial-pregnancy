#region imports and setup #######################################################

import os
import sys
import torch
import faiss
import warnings
import scanorama
import polars as pl
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
sys.path.insert(0, os.path.expanduser('~'))
from single_cell import SingleCell

warnings.filterwarnings('ignore')
working_dir = '/home/karbabi/spatial-pregnancy'
merfish_ref_path = f'{working_dir}/input/adata_ref_zeng_raw.h5ad'

cells_joined = pd.read_csv('single-cell/ABC/metadata/cells_joined.csv')
color_mappings = {
    'class': dict(zip(
        cells_joined['class'].str.replace('/', '_'),
        cells_joined['class_color'])),
    'subclass': {k.replace('_', '/'): v for k, v in dict(zip(
        cells_joined['subclass'].str.replace('/', '_'),
        cells_joined['subclass_color'])).items()}
}
for level in color_mappings:
    color_mappings[level]['Unlabelled'] = '#d3d3d3'
del cells_joined

phase_a_config = dict(
    n_pcs=50, use_scanorama=True, harmony_kwargs=dict(theta=8))

# class -> tight radius as multiple of avg nearest-neighbor distance.
# scRNA candidates of these classes are dropped from the pool unless a
# ref cell of the same class sits within the tight radius of the query.
class_restrict = {
    '05 OB-IMN GABA': 8,
}

datasets = {
    'xenium': dict(
        sample_col='sample_rep',
        ave_dist_fold=100, alignment_shift_adjustment=0,
        n_pcs=30, use_scanorama=True,
        harmony_kwargs=dict(
            theta=12, alpha=0.05, tolerance=0.001, max_iterations=20)),
    'slidetags': dict(
        sample_col='sample',
        ave_dist_fold=30, alignment_shift_adjustment=0,
        n_pcs=30, use_scanorama=True,
        harmony_kwargs=dict(
            theta=12, alpha=0.05, tolerance=0.001, max_iterations=20)),
    'merfish': dict(
        sample_col='sample',
        ave_dist_fold=100, alignment_shift_adjustment=0,
        n_pcs=30, use_scanorama=True,
        harmony_kwargs=dict(
            theta=12, alpha=0.05, tolerance=0.001, max_iterations=20)),
}

# load scRNA-seq ref, filtered to subclasses present in MERFISH ref.
def load_scrna_filtered():
    scrna_ref = SingleCell('single-cell/ABC/zeng_combined_10Xv3.h5ad')
    scrna_ref = scrna_ref.skip_qc()
    scrna_ref = scrna_ref.make_var_names_unique(separator='~')
    merfish_ref = SingleCell(merfish_ref_path)
    ref_subclasses = merfish_ref.obs['subclass'].cast(str).unique().drop_nulls()
    scrna_ref = scrna_ref.filter_obs(
        pl.col('subclass').is_not_null() &
        pl.col('subclass').cast(str).is_in(ref_subclasses))
    scrna_ref = scrna_ref.with_columns_obs(
        pl.lit('scrna-seq').alias('batch'))
    print(f'[scrna] {scrna_ref.X.shape[0]:,} cells after subclass filter')
    del merfish_ref
    return scrna_ref

# hvg -> normalize -> pca -> harmony.
# returns (sc1, sc2) with harmony embeddings in obsm['harmony'].
def integrate_harmony(sc1, sc2, label='', batch_column=None,
                      use_scanorama=False, n_pcs=50, harmony_kwargs=None,
                      cache_dir=None):
    # workaround: SingleCell.normalize() mutates shared _uns dicts in-place
    # (line 21052), so stale normalized=True can leak across calls
    if sc1.uns.get('normalized'):
        sc1 = sc1.with_uns(normalized=False)
    if sc2.uns.get('normalized'):
        sc2 = sc2.with_uns(normalized=False)
    sc1, sc2 = sc1.hvg(sc2)
    sc1 = sc1.normalize()
    sc2 = sc2.normalize()
    if use_scanorama:
        sc1 = sc1.filter_var(pl.col('highly_variable'))
        sc2 = sc2.filter_var(pl.col('highly_variable'))
        import scipy.sparse as sp
        scanorama_dir = cache_dir if cache_dir else None
        scanorama_cached = (scanorama_dir and
            os.path.exists(f'{scanorama_dir}/scanorama_X1.npz'))
        if scanorama_cached:
            print(f'[{label}] loading cached scanorama')
            c1 = sp.load_npz(f'{scanorama_dir}/scanorama_X1.npz')
            c2 = sp.load_npz(f'{scanorama_dir}/scanorama_X2.npz')
            genes = list(np.load(f'{scanorama_dir}/scanorama_genes.npy'))
        else:
            a1, a2 = sc1.to_scanpy(), sc2.to_scanpy()
            print(f'[{label}] running scanorama ({a1.shape[1]} genes)...')
            corrected, genes = scanorama.correct(
                [a1.X, a2.X], [list(a1.var_names), list(a2.var_names)])
            c1, c2 = corrected[0], corrected[1]
            del a1, a2, corrected
            if scanorama_dir:
                sp.save_npz(f'{scanorama_dir}/scanorama_X1.npz',
                            sp.csr_matrix(c1))
                sp.save_npz(f'{scanorama_dir}/scanorama_X2.npz',
                            sp.csr_matrix(c2))
                np.save(f'{scanorama_dir}/scanorama_genes.npy',
                        np.array(genes))
        var = pl.DataFrame({sc1.var_names.name: genes})
        sc1 = SingleCell(X=c1.astype(np.float32), obs=sc1.obs, var=var
                         ).with_uns(QCed=True, normalized=True)
        sc2 = SingleCell(X=c2.astype(np.float32), obs=sc2.obs, var=var
                         ).with_uns(QCed=True, normalized=True)
        del c1, c2
        sc1, sc2 = sc1.pca(sc2, num_PCs=n_pcs, hvg_column=None)
    else:
        sc1, sc2 = sc1.pca(sc2, num_PCs=n_pcs)
    print(f'[{label}] running Harmony...')
    hkw = harmony_kwargs or {}
    sc1, sc2 = sc1.harmonize(sc2, batch_column=batch_column, **hkw)
    return sc1, sc2

# integration quality diagnostics on harmony embeddings.
def eval_integration_from_arrays(h1, h2, label='', k=50, n_per_batch=50_000):
    from sklearn.metrics import silhouette_score
    n1, n2 = h1.shape[0], h2.shape[0]
    # stratified subsample: equal from each batch
    rng = np.random.default_rng(0)
    idx1 = rng.choice(n1, min(n_per_batch, n1), replace=False)
    idx2 = rng.choice(n2, min(n_per_batch, n2), replace=False)
    harmony = np.vstack([h1[idx1], h2[idx2]])
    batch = np.array([0] * len(idx1) + [1] * len(idx2))
    # L2-normalize for cosine-equivalent kNN
    norms = np.linalg.norm(harmony, axis=1, keepdims=True).clip(min=1e-8)
    harmony_norm = harmony / norms
    tree = cKDTree(harmony_norm)
    _, nn_idx = tree.query(harmony_norm, k=k + 1)
    nn_idx = nn_idx[:, 1:]
    nn_batch = batch[nn_idx]
    # ilisi: batch diversity in neighborhoods (higher = better mixing)
    p1 = nn_batch.mean(axis=1)
    simpson = p1**2 + (1 - p1)**2
    ilisi = (1.0 / simpson).mean()
    # neighbor ratio: fraction of sc1 neighbors for sc2 cells
    query_mask = batch == 1
    nbr_ratio = (1 - nn_batch[query_mask].mean(axis=1)).mean()
    # batch silhouette (negative = good mixing)
    batch_sil = silhouette_score(
        harmony_norm, batch, sample_size=min(10_000, len(batch)),
        random_state=0)
    print(f'[{label}] integration: iLISI={ilisi:.2f}/2.00, '
          f'nbr_ratio={nbr_ratio:.2f} (50/50 sample), '
          f'batch_sil={batch_sil:.3f}')

# umap visualization of two harmony-integrated datasets.
def plot_harmony_umap(sc1, sc2, save_path, title=None):
    # fill missing label columns so concat keeps them (query cells show grey)
    for col in ['class', 'subclass']:
        if col not in sc2.obs.columns:
            sc2 = sc2.with_columns_obs(
                pl.lit('Unlabelled').cast(pl.Categorical).alias(col))
        if col in sc1.obs.columns:
            sc1 = sc1.with_columns_obs(pl.col(col).cast(pl.Categorical))
        if col in sc2.obs.columns:
            sc2 = sc2.with_columns_obs(pl.col(col).cast(pl.Categorical))
    # align obs_names and var_names column names for concat
    for attr in ['obs', 'var']:
        name1 = getattr(sc1, attr).columns[0]
        name2 = getattr(sc2, attr).columns[0]
        if name2 != name1:
            if name1 in getattr(sc2, attr).columns:
                sc2 = (sc2.set_obs_names(name1) if attr == 'obs'
                        else sc2.set_var_names(name1))
            else:
                sc2 = (sc2.rename_obs({name2: name1}) if attr == 'obs'
                        else sc2.rename_var({name2: name1}))
    combined = sc1.concat_obs(sc2, flexible=True)
    combined = combined.neighbors(PC_key='harmony')
    combined = combined.umap(hogwild=True)
    combined = combined.to_scanpy()
    combined.obsm['X_umap'] = combined.obsm.pop('umap')
    batch_vals = [b for b in combined.obs['batch'].unique()
                  if b != 'scrna-seq']
    tab_colors = plt.cm.tab20(np.linspace(0, 1, max(len(batch_vals), 1)))
    batch_palette = {'scrna-seq': '#888888'}
    batch_palette.update(zip(batch_vals, tab_colors))
    palettes = {**color_mappings, 'batch': batch_palette}
    n1 = sc1.X.shape[0]
    umap = combined.obsm['X_umap']
    is_query = np.arange(len(combined)) >= n1
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    if title:
        fig.suptitle(title, fontsize=14)
    for ax, col in zip(axes, ['batch', 'class', 'subclass']):
        palette = palettes.get(col, {})
        labels = combined.obs[col].values
        ref_colors = [palette.get(l, '#888888') for l in labels[~is_query]]
        ax.scatter(
            umap[~is_query, 0], umap[~is_query, 1],
            c=ref_colors, s=0.1, alpha=0.3, rasterized=True)
        if col == 'batch':
            q_colors = [palette.get(l, '#333333') for l in labels[is_query]]
        else:
            q_colors = '#000000'
        ax.scatter(umap[is_query, 0], umap[is_query, 1],
                   c=q_colors, s=0.05, alpha=0.5, rasterized=True)
        ax.set_title(col)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    del combined

#endregion

#region Phase A: MERFISH ref <-> scRNA-seq bridge ##############################

# for each MERFISH ref cell, stores its k nearest scRNA-seq neighbors
# in harmony space (indices, distances, and their cell type labels).
# this creates the spatial-expression bridge for the three-hop transfer.
def prepare_reference(k_harmony=20):

    cached_path = f'{working_dir}/input/adata_ref_zeng_bridge.h5ad'
    neighbor_path = f'{working_dir}/input/ref_scrna_neighbors.npz'

    if os.path.exists(cached_path) and os.path.exists(neighbor_path):
        print('[Phase A] loading cached bridge')
        ref_adata = sc.read_h5ad(cached_path)
        nbrs = np.load(neighbor_path)
        return ref_adata, nbrs['indices'], nbrs['distances']

    print('[Phase A] integrating scRNA-seq with MERFISH ref...')
    scrna_ref = load_scrna_filtered()
    merfish_ref = SingleCell(merfish_ref_path)
    merfish_ref = merfish_ref.skip_qc()
    merfish_ref = merfish_ref.set_var_names('gene_symbol')
    merfish_ref = merfish_ref.with_columns_obs(pl.lit('merfish').alias('batch'))
    print(f'[Phase A] scRNA-seq: {scrna_ref.X.shape}, '
          f'MERFISH ref: {merfish_ref.X.shape}')

    scrna_ref, merfish_ref = integrate_harmony(
        scrna_ref, merfish_ref, label='Phase A',
        use_scanorama=phase_a_config.get('use_scanorama', True),
        n_pcs=phase_a_config.get('n_pcs', 50),
        harmony_kwargs=phase_a_config.get('harmony_kwargs'),
        cache_dir=f'{working_dir}/input')
    os.makedirs(f'{working_dir}/figures', exist_ok=True)
    plot_harmony_umap(scrna_ref, merfish_ref,
                      f'{working_dir}/figures/phase_a_umap.png')

    merfish_harmony = merfish_ref.obsm['harmony']
    scrna_harmony = scrna_ref.obsm['harmony']

    # validate harmony quality via label_transfer_from
    for level in ['class', 'subclass']:
        print(f'[Phase A] validating {level} transfer...')
        merfish_ref = merfish_ref.label_transfer_from(
            scrna_ref, level,
            num_neighbors=20,
            cell_type_column=f'{level}_transferred',
            confidence_column=f'{level}_transfer_confidence')
    ref_adata = merfish_ref.to_scanpy()
    for level in ['class', 'subclass']:
        true = ref_adata.obs[level].astype(str)
        pred = ref_adata.obs[f'{level}_transferred'].astype(str)
        conf = ref_adata.obs[f'{level}_transfer_confidence']
        match = (pred == true).mean()
        conf_q = conf.quantile([0.25, 0.5, 0.75])
        low_conf = (conf < 0.5).mean()
        n_true = true.nunique()
        n_pred = pred.nunique()
        ref_adata.obs[f'{level}_bridge_consistent'] = (pred == true)
        print(f'[Phase A] {level}: accuracy={match:.1%}, '
              f'confidence={conf_q.iloc[0]:.2f}/{conf_q.iloc[1]:.2f}/'
              f'{conf_q.iloc[2]:.2f} (Q1/Q2/Q3), '
              f'<0.5={low_conf:.1%}, '
              f'types={n_pred}/{n_true} (pred/true)')

    # [Phase A] class: accuracy=94.9%, confidence=1.00/1.00/1.00 (Q1/Q2/Q3), <0.5=0.3%, types=24/24 (pred/true)
    # [Phase A] subclass: accuracy=87.9%, confidence=0.95/1.00/1.00 (Q1/Q2/Q3), <0.5=2.1%, types=134/135 (pred/true)

    # kNN in harmony space (cosine distance) via faiss.
    # L2-normalize → inner product search = cosine similarity.
    print(f'[Phase A] computing {k_harmony}-NN (cosine, faiss) from MERFISH ref '
          f'to scRNA-seq...')
    ref_norm = np.array(merfish_harmony, dtype=np.float32, order='C')
    scrna_norm = np.array(scrna_harmony, dtype=np.float32, order='C')
    faiss.normalize_L2(ref_norm)
    faiss.normalize_L2(scrna_norm)
    index = faiss.IndexFlatIP(scrna_norm.shape[1])
    index.add(scrna_norm)
    similarities, indices = index.search(ref_norm, k_harmony)
    distances = (1.0 - similarities).astype(np.float32)
    del ref_norm, scrna_norm, similarities

    # store scRNA labels for Phase B lookup
    scrna_adata = scrna_ref.to_scanpy()
    for level in ['class', 'subclass']:
        ref_adata.uns[f'scrna_{level}_labels'] = scrna_adata.obs[level].values

    ref_adata.write(cached_path)
    np.savez(neighbor_path, indices=indices, distances=distances)
    print(f'[Phase A] saved bridge ({len(ref_adata):,} ref cells, '
          f'{k_harmony} scRNA neighbors each)')
    del scrna_ref, scrna_adata
    return ref_adata, indices, distances

#endregion

#region Phase A': query <-> scRNA-seq Harmony integration ######################

# harmony-integrate query with scRNA-seq for expression matching.
def prepare_query(name, sample_col='sample'):

    output_dir = f'{working_dir}/output/{name}'
    cache_path = f'{output_dir}/query_scrna_harmony.npz'

    if os.path.exists(cache_path):
        print(f'[{name}] loading cached query-scRNA Harmony')
        data = np.load(cache_path)
        qh, sh = data['query_harmony'], data['scrna_harmony']
        eval_integration_from_arrays(qh, sh, label=name)
        return qh, sh

    print(f'[{name}] integrating query with scRNA-seq ref...')
    scrna_ref = load_scrna_filtered()
    query = SingleCell(f'{output_dir}/01_adata_query_{name}.h5ad')
    query = query.skip_qc()
    query = query.rename_obs({'_index': 'cell_label'})
    if 'gene_symbol' in query.var.columns:
        query = query.set_var_names('gene_symbol')
    else:
        query = query.rename_var({'_index': 'gene_symbol'})
    query = query.with_columns_obs(
        pl.col(sample_col).cast(pl.String).alias('batch'))
    print(f'[{name}] scRNA-seq: {scrna_ref.X.shape}, '
          f'query: {query.X.shape}')

    cfg = datasets[name]
    scrna_ref, query = integrate_harmony(
        scrna_ref, query, label=name, batch_column='batch',
        use_scanorama=cfg.get('use_scanorama', False),
        n_pcs=cfg.get('n_pcs', 50),
        harmony_kwargs=cfg.get('harmony_kwargs'),
        cache_dir=output_dir)
    eval_integration_from_arrays(
        scrna_ref.obsm['harmony'], query.obsm['harmony'], label=name)
    os.makedirs(f'{working_dir}/figures', exist_ok=True)
    plot_harmony_umap(
        scrna_ref, query,
        f'{working_dir}/figures/{name}_query_scrna_harmony_umap.png',
        title=f'{name}: query-scRNA Harmony integration')

    # unconstrained label transfer (no spatial constraint, for comparison)
    for level in ['class', 'subclass']:
        query = query.label_transfer_from(
            scrna_ref, level, num_neighbors=20,
            cell_type_column=f'{level}_unconstrained',
            confidence_column=f'{level}_unconstrained_confidence')
    query_adata = query.to_scanpy()
    unconstrained = {
        col: query_adata.obs[col].values
        for col in query_adata.obs.columns
        if 'unconstrained' in col}
    print(f'[{name}] unconstrained transfer: '
          f'{query_adata.obs["class_unconstrained"].nunique()} classes, '
          f'{query_adata.obs["subclass_unconstrained"].nunique()} subclasses')

    query_harmony = query.obsm['harmony'].astype(np.float32)
    scrna_harmony = scrna_ref.obsm['harmony'].astype(np.float32)

    np.savez(cache_path, query_harmony=query_harmony,
             scrna_harmony=scrna_harmony, **unconstrained)
    print(f'[{name}] saved Harmony ({query_harmony.shape[0]:,} query + '
          f'{scrna_harmony.shape[0]:,} scRNA-seq, {query_harmony.shape[1]}D)')

    del scrna_ref, query, query_adata
    return query_harmony, scrna_harmony

#endregion

#region Phase B: spatial cell type transfer ####################################

# mean nearest-neighbor distance on full data (no subsampling).
def avg_nn_dist(coords):
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return dists[:, 1].mean()

# spatially-constrained expression-based cell type transfer.
# for each query cell:
# 1. Spatial hard filter: find MERFISH ref cells within adaptive radius
# 2. Bridge expansion: gather those ref cells' scRNA-seq Harmony neighbors
# 3. Expression matching: cosine distance in query-scRNA Harmony space,
#    pick top k2, IDW-weighted vote on scRNA-seq labels
def run_project(name, k2=5, k_harmony=20, k_extend=20, levels=None):

    if levels is None:
        levels = ['class', 'subclass']

    cfg = datasets[name]
    sample_col = cfg['sample_col']
    ave_dist_fold = cfg['ave_dist_fold']
    alignment_shift_adjustment = cfg['alignment_shift_adjustment']
    output_dir = f'{working_dir}/output/{name}'
    fig_dir = f'{working_dir}/figures'
    os.makedirs(fig_dir, exist_ok=True)

    output_path = f'{output_dir}/02_adata_query_{name}.h5ad'
    if os.path.exists(output_path):
        print(f'[{name}] skipping, {output_path} already exists')
        return

    # load query adata
    adata = sc.read_h5ad(f'{output_dir}/01_adata_query_{name}.h5ad')
    print(f'[{name}] query: {adata.shape[0]:,} cells')
    # load aligned query coords
    coords_ffd = torch.load(f'{output_dir}/coords_ffd.pt', weights_only=False)

    # load bridge (Phase A) and derive ref coords from it directly.
    adata_ref, scrna_indices, _ = prepare_reference(k_harmony=k_harmony)
    ref_sections = sorted(
        s for s in adata_ref.obs['sample'].unique()
        if s.startswith('C57BL6J-638850'))
    ref_global_idx = np.concatenate([
        np.where(adata_ref.obs['sample'] == s)[0] for s in ref_sections])
    ref_obs = adata_ref.obs.iloc[ref_global_idx]
    ref_coords = ref_obs[['x_raw', 'y_raw']].values
    print(f'[{name}] ref: {len(ref_obs):,} cells across '
          f'{len(ref_sections)} sections')
    assert np.allclose(
        ref_coords,
        adata_ref.obs.iloc[ref_global_idx][['x_raw', 'y_raw']].values), \
        'ref_coords and ref_global_idx are misaligned'

    # load query-scRNA Harmony embeddings (Phase A')
    query_harmony, scrna_harmony = prepare_query(name, sample_col)
    assert query_harmony.shape[0] == len(adata), \
        f'query harmony ({query_harmony.shape[0]}) != adata ({len(adata)})'

    # load unconstrained labels from Phase A' cache
    cache_data = np.load(
        f'{output_dir}/query_scrna_harmony.npz', allow_pickle=True)
    for col in cache_data.files:
        if 'unconstrained' in col:
            adata.obs[col] = cache_data[col]

    # assign aligned coords to query obs
    for s in sorted(coords_ffd.keys()):
        mask = adata.obs[sample_col] == s
        idx = adata.obs[mask].index
        adata.obs.loc[idx, 'x_ffd'] = coords_ffd[s][:, 0]
        adata.obs.loc[idx, 'y_ffd'] = coords_ffd[s][:, 1]

    query_coords = adata.obs[['x_ffd', 'y_ffd']].values

    # hop 1: spatial constraint with adaptive radius
    print(f'[{name}] computing avg nearest-neighbor dist...')
    avg_edge_dist = avg_nn_dist(query_coords)
    pdist_thres = ave_dist_fold * avg_edge_dist + alignment_shift_adjustment
    print(f'[{name}] adaptive radius: {avg_edge_dist:.4f} avg edge x '
          f'{ave_dist_fold} + {alignment_shift_adjustment} = '
          f'{pdist_thres:.4f}')

    # type-specific class restriction: per-class 1-NN allow mask
    ref_class_arr_local = adata_ref.obs['class'].astype(str).values[
        ref_global_idx]
    scrna_class_arr = np.asarray(
        adata_ref.uns['scrna_class_labels']).astype(str)
    allowed_class = {}
    for cls_name, fold in class_restrict.items():
        mask_ref = ref_class_arr_local == cls_name
        if not mask_ref.any():
            print(f'[{name}] class_restrict: {cls_name!r} not in ref, '
                  f'skipping')
            continue
        tight_radius = fold * avg_edge_dist
        cls_tree = cKDTree(ref_coords[mask_ref])
        d_cls, _ = cls_tree.query(query_coords, k=1)
        allowed = d_cls <= tight_radius
        allowed_class[cls_name] = allowed
        print(f'[{name}] class_restrict {cls_name!r}: '
              f'{allowed.sum():,}/{len(query_coords):,} cells within '
              f'{fold}x avg_nn = {tight_radius:.4f}')

    print(f'[{name}] spatial kNN (query_ball_point)...')
    tree = cKDTree(ref_coords)
    spatial_candidates = tree.query_ball_point(query_coords, r=pdist_thres)
    # fallback: precompute k_extend nearest for cells with 0 candidates
    n_empty = sum(1 for c in spatial_candidates if len(c) == 0)
    if n_empty > 0:
        print(f'[{name}] {n_empty:,} cells with 0 ref neighbors, '
              f'using k_extend={k_extend} fallback')
        fallback_dists, fallback_indices = tree.query(
            query_coords, k=k_extend)

    n_candidates_arr = np.array([len(c) for c in spatial_candidates])
    adata.obs['n_spatial_candidates'] = n_candidates_arr

    # hop 2 + 3: bridge expansion, dedup, expression matching, vote

    # pre-encode scRNA labels as integers for fast bincount
    n_query = len(adata)
    label_data = {}
    for level in levels:
        raw_labels = adata_ref.uns[f'scrna_{level}_labels']
        unique_labels, label_codes = np.unique(
            raw_labels, return_inverse=True)
        label_data[level] = (unique_labels, label_codes)

    # bridge consistency per ref cell (for QC)
    bridge_consistent = {}
    for level in levels:
        col = f'{level}_bridge_consistent'
        if col in adata_ref.obs.columns:
            bridge_consistent[level] = adata_ref.obs[col].values

    # output arrays
    results = {}
    for level in levels:
        results[level] = {
            'assigned': np.empty(n_query, dtype=object),
            'confidence': np.zeros(n_query, dtype=np.float32),
            'margin': np.zeros(n_query, dtype=np.float32),
            'bridge_consistency': np.zeros(n_query, dtype=np.float32),
        }
    avg_pdist = np.zeros(n_query, dtype=np.float32)
    min_cos_dist = np.ones(n_query, dtype=np.float32)

    # precompute normalized harmony vectors for batch cosine similarity
    eps = np.float32(1e-8)
    query_norm = query_harmony / np.linalg.norm(
        query_harmony, axis=1, keepdims=True).clip(min=eps)
    scrna_norm = scrna_harmony / np.linalg.norm(
        scrna_harmony, axis=1, keepdims=True).clip(min=eps)

    # ref-anchored pool filtering: only keep scRNA bridge candidates whose
    # subclass exists among the spatial ref neighbors. prevents bridge leakage
    # where non-X ref cells bridge to X scRNA cells through harmony proximity.
    ref_subclass_arr = adata_ref.obs['subclass'].astype(str).values
    scrna_subclass_arr = np.asarray(
        adata_ref.uns['scrna_subclass_labels']).astype(str)

    # phase 1: build per-cell scRNA pools, avg_pdist, bridge consistency
    print(f'[{name}] building candidate pools ({n_query:,} cells)...')
    cell_pools = [None] * n_query
    pool_reduction = np.zeros(n_query, dtype=np.float32)
    total_before = 0
    total_after = 0
    for i in range(n_query):
        if i % 100_000 == 0 and i > 0:
            print(f'[{name}]   {i:,} / {n_query:,}')
        ref_local = spatial_candidates[i]
        if len(ref_local) == 0:
            ref_local = fallback_indices[i]
            avg_pdist[i] = fallback_dists[i].mean()
        else:
            ref_local = np.array(ref_local)
            avg_pdist[i] = np.linalg.norm(
                ref_coords[ref_local] - query_coords[i], axis=1).mean()
        ref_global = ref_global_idx[ref_local]
        for level in levels:
            if level in bridge_consistent:
                results[level]['bridge_consistency'][i] = \
                    bridge_consistent[level][ref_global].mean()
        ref_subs_present = set(ref_subclass_arr[ref_global])
        scrna_pool = scrna_indices[ref_global].ravel()
        subclass_mask = np.isin(
            scrna_subclass_arr[scrna_pool], list(ref_subs_present))
        # drop restricted classes when query is outside their tight radius
        class_keep = np.ones(len(scrna_pool), dtype=bool)
        for cls_name, allowed in allowed_class.items():
            if not allowed[i]:
                class_keep &= scrna_class_arr[scrna_pool] != cls_name
        unfiltered_size = len(np.unique(scrna_pool))
        pool_mask = subclass_mask & class_keep
        if pool_mask.any():
            cell_pools[i] = np.unique(scrna_pool[pool_mask])
        elif class_keep.any():
            cell_pools[i] = np.unique(scrna_pool[class_keep])
        else:
            cell_pools[i] = np.array([], dtype=scrna_pool.dtype)
        total_before += unfiltered_size
        total_after += len(cell_pools[i])
        pool_reduction[i] = 1.0 - len(cell_pools[i]) / max(unfiltered_size, 1)

    print(f'[{name}] pool filtering: median {np.median(pool_reduction):.0%} '
          f'reduction, {total_after:,}/{total_before:,} candidates kept '
          f'({total_after/max(total_before,1):.0%})')

    # phase 2: chunked expression matching + vectorized voting
    chunk_size = 500
    n_scrna = scrna_norm.shape[0]
    g2l = np.empty(n_scrna, dtype=np.intp)
    n_chunks = (n_query + chunk_size - 1) // chunk_size

    print(f'[{name}] expression matching ({n_query:,} cells, '
          f'{n_chunks} chunks, k2={k2})...')
    for cs in range(0, n_query, chunk_size):
        ce = min(cs + chunk_size, n_query)
        n_chunk = ce - cs
        if cs % 50_000 < chunk_size and cs > 0:
            print(f'[{name}]   {cs:,} / {n_query:,}')

        pools = cell_pools[cs:ce]
        union_pool = np.unique(np.concatenate(pools))
        n_pool = len(union_pool)
        if n_pool == 0:
            continue

        g2l[union_pool] = np.arange(n_pool)

        # batch cosine similarity: (n_chunk x n_pool)
        sim = query_norm[cs:ce] @ scrna_norm[union_pool].T

        # per-cell pool membership mask
        pool_sizes = [len(p) for p in pools]
        mask_rows = np.repeat(np.arange(n_chunk), pool_sizes)
        mask_cols = np.concatenate([g2l[p] for p in pools])
        mask = np.zeros((n_chunk, n_pool), dtype=bool)
        mask[mask_rows, mask_cols] = True

        dist = np.where(mask, 1.0 - sim, np.inf)
        del sim, mask

        # top-k2 per cell (inf entries sort last)
        k2_eff = min(k2, n_pool)
        if k2_eff < n_pool:
            top_local = np.argpartition(dist, k2_eff, axis=1)[:, :k2_eff]
        else:
            top_local = np.broadcast_to(
                np.arange(n_pool), (n_chunk, n_pool))[:, :k2_eff].copy()
        top_dist = np.take_along_axis(dist, top_local, axis=1)
        top_global = union_pool[top_local]
        del dist

        valid = np.isfinite(top_dist)
        min_cos_dist[cs:ce] = np.where(
            valid.any(axis=1),
            np.min(np.where(valid, top_dist, np.inf), axis=1), 1.0)

        weights = np.where(valid, 1.0 / (top_dist + 1e-6), 0.0)

        # vectorized weighted vote
        r_idx = np.repeat(
            np.arange(n_chunk)[:, None], k2_eff, axis=1).ravel()
        w_flat = weights.ravel()
        for level in levels:
            ul, lc = label_data[level]
            n_labels = len(ul)
            codes = lc[top_global].ravel()
            scores = np.zeros((n_chunk, n_labels), dtype=np.float64)
            np.add.at(scores, (r_idx, codes), w_flat)

            best = np.argmax(scores, axis=1)
            total = scores.sum(axis=1)
            safe = np.maximum(total, 1e-12)
            results[level]['assigned'][cs:ce] = ul[best]
            results[level]['confidence'][cs:ce] = scores[
                np.arange(n_chunk), best] / safe
            sorted_s = np.sort(scores, axis=1)[:, ::-1]
            results[level]['margin'][cs:ce] = (
                sorted_s[:, 0] - sorted_s[:, 1]) / safe

        del top_local, top_dist, top_global, weights

    # store results
    adata.obs['avg_pdist'] = avg_pdist
    adata.obs['min_cos_dist'] = min_cos_dist
    for level in levels:
        adata.obs[level] = results[level]['assigned']
        adata.obs[f'{level}_confidence'] = results[level]['confidence']
        adata.obs[f'{level}_margin'] = results[level]['margin']
        adata.obs[f'{level}_bridge_consistency'] = \
            results[level]['bridge_consistency']

    # summary
    for level in levels:
        n_types = adata.obs[level].nunique()
        med_conf = adata.obs[f'{level}_confidence'].median()
        med_cos = adata.obs['min_cos_dist'].median()
        med_margin = adata.obs[f'{level}_margin'].median()
        med_bc = adata.obs[f'{level}_bridge_consistency'].median()
        print(f'[{name}] {level}: {n_types} types, '
              f'confidence={med_conf:.2f}, margin={med_margin:.2f}, '
              f'cos_dist={med_cos:.4f}, bridge={med_bc:.2f}')

    # save
    adata.write(output_path)
    print(f'[{name}] saved {output_path}')

    # plot spatial maps
    for level in levels:
        color_map = color_mappings.get(level)
        samples = sorted(adata.obs[sample_col].unique())
        ncols = min(5, len(samples))
        nrows = int(np.ceil(len(samples) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 5 * nrows), squeeze=False)
        fig.suptitle(f'{name}: {level}', fontsize=16)
        for i, s in enumerate(samples):
            ax = axes[i // ncols, i % ncols]
            ax.scatter(
                ref_coords[:, 0], ref_coords[:, 1],
                s=0.1, c='lightgray', alpha=0.1)
            obs_s = adata.obs[adata.obs[sample_col] == s]
            if color_map:
                colors = [color_map.get(l, '#333333') for l in obs_s[level]]
            else:
                colors = 'red'
            ax.scatter(obs_s['x_ffd'], obs_s['y_ffd'],
                       s=0.5, c=colors, alpha=0.5)
            ax.set_title(f'{s} (n={len(obs_s):,})')
            ax.set_aspect('equal')
            ax.axis('off')

        for j in range(len(samples), nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)
        plt.tight_layout()
        fig.savefig(f'{fig_dir}/{name}_spatial_{level}.png', dpi=200)
        plt.close()

    print(f'[{name}] done')

# in-place patch: reapply class_restrict to an existing 02_adata without
# rerunning the full pipeline. only reprocesses cells currently assigned
# to a restricted class that fall outside the tight radius.
def patch_class_restrict(name, k2=5, k_extend=20, levels=None):

    if not class_restrict:
        print(f'[{name}] no class_restrict, skipping')
        return
    if levels is None:
        levels = ['class', 'subclass']

    cfg = datasets[name]
    sample_col = cfg['sample_col']
    output_path = f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad'
    adata = sc.read_h5ad(output_path)

    adata_ref, scrna_indices, _ = prepare_reference()
    ref_sections = sorted(
        s for s in adata_ref.obs['sample'].unique()
        if s.startswith('C57BL6J-638850'))
    ref_global_idx = np.concatenate([
        np.where(adata_ref.obs['sample'] == s)[0] for s in ref_sections])
    ref_coords = adata_ref.obs.iloc[ref_global_idx][
        ['x_raw', 'y_raw']].values
    ref_class_arr_local = adata_ref.obs['class'].astype(str).values[
        ref_global_idx]
    ref_subclass_arr = adata_ref.obs['subclass'].astype(str).values
    scrna_class_arr = np.asarray(
        adata_ref.uns['scrna_class_labels']).astype(str)
    scrna_subclass_arr = np.asarray(
        adata_ref.uns['scrna_subclass_labels']).astype(str)

    qh, sh = prepare_query(name, sample_col)
    query_coords = adata.obs[['x_ffd', 'y_ffd']].values
    avg_edge_dist = avg_nn_dist(query_coords)
    pdist_thres = (cfg['ave_dist_fold'] * avg_edge_dist
                   + cfg['alignment_shift_adjustment'])

    allowed_class = {}
    for cls_name, fold in class_restrict.items():
        mask_ref = ref_class_arr_local == cls_name
        if not mask_ref.any():
            continue
        cls_tree = cKDTree(ref_coords[mask_ref])
        d_cls, _ = cls_tree.query(query_coords, k=1)
        allowed_class[cls_name] = d_cls <= fold * avg_edge_dist

    cur = adata.obs['class'].astype(str).values
    affected = np.zeros(len(adata), dtype=bool)
    for cls_name, allowed in allowed_class.items():
        affected |= (cur == cls_name) & ~allowed
    n_aff = int(affected.sum())
    print(f'[{name}] reprocessing {n_aff:,}/{len(adata):,} cells')
    if n_aff == 0:
        return

    aff_idx = np.where(affected)[0]
    ref_tree = cKDTree(ref_coords)
    spatial_cands = ref_tree.query_ball_point(
        query_coords[aff_idx], r=pdist_thres)
    _, fb_indices = ref_tree.query(query_coords[aff_idx], k=k_extend)

    eps = np.float32(1e-8)
    qn = qh / np.linalg.norm(qh, axis=1, keepdims=True).clip(min=eps)
    sn = sh / np.linalg.norm(sh, axis=1, keepdims=True).clip(min=eps)

    label_data = {
        level: np.unique(
            adata_ref.uns[f'scrna_{level}_labels'], return_inverse=True)
        for level in levels}

    out_label = {l: adata.obs[l].astype(str).values.copy() for l in levels}
    out_conf = {
        l: adata.obs[f'{l}_confidence'].values.copy() for l in levels}
    out_margin = {
        l: adata.obs[f'{l}_margin'].values.copy() for l in levels}
    out_min_cos = adata.obs['min_cos_dist'].values.copy()

    for j, qi in enumerate(aff_idx):
        if j % 5000 == 0 and j > 0:
            print(f'[{name}]   {j:,}/{n_aff:,}')
        rl = spatial_cands[j]
        ref_local = np.array(rl) if len(rl) else fb_indices[j]
        ref_global = ref_global_idx[ref_local]
        ref_subs_present = set(ref_subclass_arr[ref_global])
        scrna_pool = scrna_indices[ref_global].ravel()
        subclass_mask = np.isin(
            scrna_subclass_arr[scrna_pool], list(ref_subs_present))
        class_keep = np.ones(len(scrna_pool), dtype=bool)
        for cls_name, allowed in allowed_class.items():
            if not allowed[qi]:
                class_keep &= scrna_class_arr[scrna_pool] != cls_name
        pool_mask = subclass_mask & class_keep
        if pool_mask.any():
            pool = np.unique(scrna_pool[pool_mask])
        elif class_keep.any():
            pool = np.unique(scrna_pool[class_keep])
        else:
            pool = np.array([], dtype=scrna_pool.dtype)

        if len(pool) == 0:
            for level in levels:
                ul, _ = label_data[level]
                out_label[level][qi] = ul[0]
                out_conf[level][qi] = 0.0
                out_margin[level][qi] = 0.0
            out_min_cos[qi] = 1.0
            continue

        cos_dist = 1.0 - sn[pool] @ qn[qi]
        k = min(k2, len(pool))
        top = (np.argpartition(cos_dist, k - 1)[:k]
               if k < len(pool) else np.arange(len(pool)))
        top_dist = cos_dist[top]
        top_global = pool[top]
        weights = 1.0 / (top_dist + 1e-6)
        out_min_cos[qi] = top_dist.min()
        for level in levels:
            ul, lc = label_data[level]
            codes = lc[top_global]
            scores = np.bincount(codes, weights=weights, minlength=len(ul))
            sorted_s = np.sort(scores)[::-1]
            total = scores.sum()
            best = int(np.argmax(scores))
            out_label[level][qi] = ul[best]
            out_conf[level][qi] = scores[best] / total if total > 0 else 0.0
            out_margin[level][qi] = (
                (sorted_s[0] - sorted_s[1]) / total
                if total > 0 and len(sorted_s) > 1 else 0.0)

    for level in levels:
        adata.obs[level] = out_label[level]
        adata.obs[f'{level}_confidence'] = out_conf[level]
        adata.obs[f'{level}_margin'] = out_margin[level]
    adata.obs['min_cos_dist'] = out_min_cos

    adata.write(output_path)
    print(f'[{name}] patched -> {output_path}')

#endregion

#region run ####################################################################

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <dataset|all> [patch]')
        print(f'Datasets: {", ".join(datasets.keys())}')
        sys.exit(1)
    t = sys.argv[1]
    fn = (patch_class_restrict
          if len(sys.argv) > 2 and sys.argv[2] == 'patch'
          else run_project)
    if t == 'all':
        for name in datasets:
            fn(name)
    elif t in datasets:
        fn(t)
    else:
        print(f'Unknown dataset: {t}. Choose from {list(datasets.keys())} or all')
        sys.exit(1)

    # adaptive radius visualization (zoomed views per dataset)
    from matplotlib.patches import Circle
    ref = sc.read_h5ad(f'{working_dir}/input/adata_ref_zeng_bridge.h5ad')
    ref_sections = sorted(s for s in ref.obs['sample'].unique()
                        if s.startswith('C57BL6J-638850'))
    ref_obs_sub = ref.obs[ref.obs['sample'].isin(ref_sections)]
    ref_coords_full = ref_obs_sub[['x_raw', 'y_raw']].values
    ref_tree = cKDTree(ref_coords_full)
    n_samples = 5
    fig, axes = plt.subplots(len(datasets), n_samples + 1,
                            figsize=(4 * (n_samples + 1), 4 * len(datasets)),
                            squeeze=False)
    rng = np.random.default_rng(42)
    for row, (name, cfg) in enumerate(datasets.items()):
        coords_ffd = torch.load(f'{working_dir}/output/{name}/coords_ffd.pt',
                                weights_only=False)
        query_coords = np.vstack(
            [coords_ffd[s] for s in sorted(coords_ffd.keys())])
        edge = avg_nn_dist(query_coords)
        radius = (cfg['ave_dist_fold'] * edge +
                cfg['alignment_shift_adjustment'])
        ax = axes[row, 0]
        ax.scatter(ref_coords_full[:, 0], ref_coords_full[:, 1],
                s=0.05, c='lightgray', alpha=0.3, rasterized=True)
        sample_idx = rng.integers(len(query_coords), size=n_samples)
        sample_cells = query_coords[sample_idx]
        ax.scatter(sample_cells[:, 0], sample_cells[:, 1], s=20, c='red', zorder=5)
        for cell in sample_cells:
            ax.add_patch(Circle(cell, radius, fill=False, color='red',
                                linewidth=1, linestyle='--'))
        ax.set_aspect('equal')
        ax.set_title(f'{name} overview\n'
                    f'edge={edge:.5f} (fold={cfg["ave_dist_fold"]}, '
                    f'shift={cfg["alignment_shift_adjustment"]})\n'
                    f'r={radius:.4f}', fontsize=10)
        ax.axis('off')
        for col, cell in enumerate(sample_cells, start=1):
            cands = ref_tree.query_ball_point(cell, r=radius)
            zoom = max(radius * 3, 0.05)
            ax = axes[row, col]
            in_zoom = ((np.abs(ref_coords_full[:, 0] - cell[0]) < zoom * 1.5) &
                    (np.abs(ref_coords_full[:, 1] - cell[1]) < zoom * 1.5))
            ax.scatter(ref_coords_full[in_zoom, 0], ref_coords_full[in_zoom, 1],
                    s=4, c='lightgray', alpha=0.6, rasterized=True)
            if len(cands) > 0:
                ax.scatter(ref_coords_full[cands, 0], ref_coords_full[cands, 1],
                        s=8, c='steelblue', alpha=0.8, rasterized=True)
            ax.scatter(cell[0], cell[1], s=80, c='red', zorder=5,
                    edgecolors='white', linewidths=1)
            ax.add_patch(Circle(cell, radius, fill=False, color='red',
                                linewidth=2, linestyle='--'))
            ax.set_xlim(cell[0] - zoom, cell[0] + zoom)
            ax.set_ylim(cell[1] - zoom, cell[1] + zoom)
            ax.set_aspect('equal')
            ax.set_title(f'{len(cands):,} candidates', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(f'{working_dir}/figures/radius_tuning.png', dpi=200)
    plt.close()

#endregion

