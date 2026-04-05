#region imports and setup #######################################################

import os
import sys
import torch
import warnings
import scanorama
import polars as pl
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Delaunay
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

datasets = {
    'merfish': dict(
        sample_col='sample',
        ave_dist_fold=10,
        alignment_shift_adjustment=0,
        use_scanorama=False,
        harmony_kwargs=dict(
            theta=4, alpha=0.1, tolerance=0.001,
            max_iterations=20)),
    'slidetags': dict(
        sample_col='sample',
        ave_dist_fold=10, alignment_shift_adjustment=0,
        use_scanorama=False,
        harmony_kwargs=dict(
            theta=8, alpha=0.1, tolerance=0.001,
            max_iterations=20)),
    'xenium': dict(
        sample_col='sample_rep',
        ave_dist_fold=3, alignment_shift_adjustment=0,
        use_scanorama=False,
        harmony_kwargs=dict(
            theta=4, alpha=0.1, tolerance=0.001,
            max_iterations=20)),
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
                      use_scanorama=False, harmony_kwargs=None):
    # reset normalized flag in case uns dict is shared from a prior run
    sc1 = sc1.with_uns(normalized=False)
    sc2 = sc2.with_uns(normalized=False)
    sc1, sc2 = sc1.hvg(sc2)
    sc1 = sc1.normalize()
    sc2 = sc2.normalize()
    if use_scanorama:
        sc1 = sc1.filter_var(pl.col('highly_variable'))
        sc2 = sc2.filter_var(pl.col('highly_variable'))
        a1, a2 = sc1.to_scanpy(), sc2.to_scanpy()
        print(f'[{label}] running scanorama ({a1.shape[1]} genes)...')
        corrected, genes = scanorama.correct(
            [a1.X, a2.X], [list(a1.var_names), list(a2.var_names)])
        var = pd.DataFrame(index=genes)
        a1 = sc.AnnData(X=corrected[0].astype(np.float32), obs=a1.obs, var=var)
        a2 = sc.AnnData(X=corrected[1].astype(np.float32), obs=a2.obs, var=var)
        sc1 = SingleCell(a1).with_uns(QCed=True, normalized=True)
        sc2 = SingleCell(a2).with_uns(QCed=True, normalized=True)
        del corrected, a1, a2
        sc1, sc2 = sc1.pca(sc2, num_PCs=50, hvg_column=None)
    else:
        sc1, sc2 = sc1.pca(sc2, num_PCs=50)
    print(f'[{label}] running Harmony...')
    hkw = harmony_kwargs or {}
    sc1, sc2 = sc1.harmonize(sc2, batch_column=batch_column, **hkw)
    return sc1, sc2

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
    combined = sc1.concat_obs(sc2, flexible=True)
    combined = combined.neighbors(PC_key='harmony')
    combined = combined.umap(hogwild=True)
    combined = combined.to_scanpy()
    combined.obsm['X_umap'] = combined.obsm.pop('umap')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    if title:
        fig.suptitle(title, fontsize=14)
    for ax, col in zip(axes, ['batch', 'class', 'subclass']):
        sc.pl.umap(
            combined, color=col, size=0.1, ax=ax,
            show=False, palette=color_mappings.get(col),
            legend_loc='none' if col == 'subclass' else 'right margin')
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
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
        scrna_ref, merfish_ref, label='Phase A', use_scanorama=True)
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

    # kNN in harmony space (cosine distance): for each MERFISH ref cell,
    # find k nearest scRNA cells. chunked to manage memory (~344K x 4M).
    print(f'[Phase A] computing {k_harmony}-NN (cosine) from MERFISH ref '
          f'to scRNA-seq...')
    n_ref = merfish_harmony.shape[0]
    indices = np.empty((n_ref, k_harmony), dtype=np.intp)
    distances = np.empty((n_ref, k_harmony), dtype=np.float64)
    chunk_size = 20_000
    for start in range(0, n_ref, chunk_size):
        end = min(start + chunk_size, n_ref)
        cos_dists = cdist(merfish_harmony[start:end], scrna_harmony,
                          metric='cosine')
        top_k = np.argpartition(cos_dists, k_harmony, axis=1)[:, :k_harmony]
        indices[start:end] = top_k
        distances[start:end] = np.take_along_axis(cos_dists, top_k, axis=1)

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
        return data['query_harmony'], data['scrna_harmony']

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
        harmony_kwargs=cfg.get('harmony_kwargs'))
    os.makedirs(f'{working_dir}/figures', exist_ok=True)
    plot_harmony_umap(
        scrna_ref, query,
        f'{working_dir}/figures/{name}_query_scrna_harmony_umap.png',
        title=f'{name}: query-scRNA Harmony integration')

    query_harmony = query.obsm['harmony'].astype(np.float32)
    scrna_harmony = scrna_ref.obsm['harmony'].astype(np.float32)

    np.savez(cache_path, query_harmony=query_harmony,
             scrna_harmony=scrna_harmony)
    print(f'[{name}] saved Harmony ({query_harmony.shape[0]:,} query + '
          f'{scrna_harmony.shape[0]:,} scRNA-seq, {query_harmony.shape[1]}D)')

    del scrna_ref, query
    return query_harmony, scrna_harmony

#endregion

#region Phase B: spatial cell type transfer ####################################

# average Delaunay edge distance, filtering edges above `quantile`.
# reimplements CAST's average_dist without the O(n^2) pairwise matrix.
# subsamples large datasets for speed.
def compute_avg_delaunay_dist(coords, quantile=0.99, max_subsample=50_000,
                              seed=0):

    if coords.shape[0] > max_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(coords.shape[0], max_subsample, replace=False)
        coords = coords[idx]
    # deduplicate (Delaunay fails on duplicate points)
    coords = np.unique(coords, axis=0)
    tri = Delaunay(coords)
    # extract unique edges from simplices
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    edges = np.array(list(edges))
    # compute edge lengths directly
    edge_dists = np.linalg.norm(
        coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    # filter outlier edges (long-range Delaunay artifacts)
    threshold = np.percentile(edge_dists, quantile * 100)
    edge_dists = edge_dists[edge_dists <= threshold]
    return edge_dists.mean()

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
    # load ref coords from coords_raw.pt (ref section entries)
    coords_raw = torch.load(f'{output_dir}/coords_raw.pt', weights_only=False)
    ref_sections = sorted(
        s for s in coords_raw if s.startswith('C57BL6J-638850'))
    ref_coords = np.vstack([coords_raw[s] for s in ref_sections])

    # load bridge (Phase A)
    adata_ref, scrna_indices, _ = prepare_reference(k_harmony=k_harmony)
    ref_obs = pd.concat([
        adata_ref.obs[adata_ref.obs['sample'] == s] for s in ref_sections])
    ref_global_idx = np.concatenate([
        np.where(adata_ref.obs['sample'] == s)[0] for s in ref_sections])
    print(f'[{name}] ref: {len(ref_obs):,} cells across '
          f'{len(ref_sections)} sections')
    assert len(ref_obs) == ref_coords.shape[0], \
        f'ref obs ({len(ref_obs)}) != ref coords ({ref_coords.shape[0]})'

    # load query-scRNA Harmony embeddings (Phase A')
    query_harmony, scrna_harmony = prepare_query(name, sample_col)
    assert query_harmony.shape[0] == len(adata), \
        f'query harmony ({query_harmony.shape[0]}) != adata ({len(adata)})'

    # assign aligned coords to query obs
    for s in sorted(coords_ffd.keys()):
        mask = adata.obs[sample_col] == s
        idx = adata.obs[mask].index
        adata.obs.loc[idx, 'x_ffd'] = coords_ffd[s][:, 0]
        adata.obs.loc[idx, 'y_ffd'] = coords_ffd[s][:, 1]

    query_coords = adata.obs[['x_ffd', 'y_ffd']].values

    # hop 1: spatial constraint with adaptive radius

    avg_edge_dist = compute_avg_delaunay_dist(query_coords)
    pdist_thres = ave_dist_fold * avg_edge_dist + alignment_shift_adjustment
    print(f'[{name}] adaptive radius: {avg_edge_dist:.4f} avg edge x '
          f'{ave_dist_fold} + {alignment_shift_adjustment} = '
          f'{pdist_thres:.4f}')

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

    # phase 1: build per-cell scRNA pools, avg_pdist, bridge consistency
    print(f'[{name}] building candidate pools...')
    cell_pools = [None] * n_query
    for i in range(n_query):
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
        cell_pools[i] = np.unique(scrna_indices[ref_global].ravel())

    # phase 2: chunked expression matching + vectorized voting
    chunk_size = 500
    n_scrna = scrna_norm.shape[0]
    g2l = np.empty(n_scrna, dtype=np.intp)

    print(f'[{name}] running expression matching (k2={k2})...')
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
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 5 * nrows), squeeze=False)
        fig.suptitle(f'{name}: {level}', fontsize=16)

        for i, s in enumerate(samples):
            ax = axes[i // ncols, i % ncols]
            ax.scatter(ref_coords[:, 0], ref_coords[:, 1],
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

#endregion
#region run ####################################################################

if __name__ == '__main__':
    for name, cfg in datasets.items():
        prepare_query(name, cfg['sample_col'])
    prepare_reference()

# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         print(f'Usage: python {sys.argv[0]} <dataset>')
#         print(f'Datasets: {", ".join(datasets.keys())}')
#         sys.exit(1)
#     t = sys.argv[1]
#     if t not in datasets:
#         print(f'Unknown dataset: {t}. Choose from {list(datasets.keys())}')
#         sys.exit(1)
#     run_project(t)

#endregion
