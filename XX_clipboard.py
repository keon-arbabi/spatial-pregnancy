import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import zscore

working_dir = '/home/karbabi/spatial-pregnancy'
datasets = {
    'merfish':   'sample',
    'slidetags': 'sample',
    'xenium':    'sample_rep',
}

ref = sc.read_h5ad(f'{working_dir}/input/adata_ref_zeng_bridge.h5ad')
ref_sections = sorted(s for s in ref.obs['sample'].unique()
                      if s.startswith('C57BL6J-638850'))
ref_obs = ref.obs[ref.obs['sample'].isin(ref_sections)].copy()
ref_coords_all = ref_obs[['x_raw', 'y_raw']].values
ref_class_all = ref_obs['class'].astype(str).values

# per-class within-ref NN distance (localization scale)
ref_nn_by_class = {}
for c in np.unique(ref_class_all):
    pts = ref_coords_all[ref_class_all == c]
    if len(pts) < 2:
        continue
    t = cKDTree(pts)
    d, _ = t.query(pts, k=2)
    ref_nn_by_class[c] = d[:, 1].mean()

ref_tree_all = cKDTree(ref_coords_all)
ref_trees_by_class = {
    c: cKDTree(ref_coords_all[ref_class_all == c])
    for c in np.unique(ref_class_all)
    if (ref_class_all == c).sum() > 0
}

for name, sample_col in datasets.items():
    print(f'\n=== {name} ===')
    adata = sc.read_h5ad(
        f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad')
    q_coords = adata.obs[['x_ffd', 'y_ffd']].values
    q_class = adata.obs['class'].astype(str).values

    # radius matched to pipeline
    fold = 30 if name == 'slidetags' else 100
    t = cKDTree(q_coords)
    d, _ = t.query(q_coords, k=2)
    radius = fold * d[:, 1].mean()

    # 1) local-ref support
    nbrs = ref_tree_all.query_ball_point(q_coords, r=radius)
    support = np.zeros(len(adata))
    for i, nb in enumerate(nbrs):
        if not nb:
            support[i] = np.nan
            continue
        support[i] = (ref_class_all[nb] == q_class[i]).mean()

    # 2) distance to nearest same-class ref cell, normalized
    dist_ratio = np.full(len(adata), np.nan)
    for c in np.unique(q_class):
        if c not in ref_trees_by_class or c not in ref_nn_by_class:
            continue
        idx = np.where(q_class == c)[0]
        d_c, _ = ref_trees_by_class[c].query(q_coords[idx], k=1)
        dist_ratio[idx] = d_c / ref_nn_by_class[c]

    # 3) spatial dispersion ratio (per class, pooled across sections)
    rows = []
    for c in np.unique(q_class):
        if c not in ref_trees_by_class:
            continue
        q_pts = q_coords[q_class == c]
        if len(q_pts) < 5:
            continue
        r_pts = ref_coords_all[ref_class_all == c]
        if len(r_pts) < 5:
            continue
        q_disp = np.sqrt(q_pts.var(axis=0).sum())
        r_disp = np.sqrt(r_pts.var(axis=0).sum())
        if r_disp == 0:
            continue
        rows.append(dict(
            cls=c,
            n_query=len(q_pts),
            local_support=np.nanmedian(support[q_class == c]),
            dist_ratio=np.nanmedian(dist_ratio[q_class == c]),
            disp_ratio=q_disp / r_disp,
        ))

    df = pd.DataFrame(rows).dropna()
    df = df[df['n_query'] >= 50]  # ignore tiny populations
    df['z_support'] = zscore(-df['local_support'])  # invert: low = bad
    df['z_dist']    = zscore(np.log1p(df['dist_ratio']))
    df['z_disp']    = zscore(np.log1p(df['disp_ratio']))
    df['suspicion'] = df[['z_support', 'z_dist', 'z_disp']].sum(axis=1)
    df = df.sort_values('suspicion', ascending=False)

    print(df[['cls', 'n_query', 'local_support', 'dist_ratio',
              'disp_ratio', 'suspicion']].head(20).to_string(index=False))
