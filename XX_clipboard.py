import numpy as np
import scanpy as sc
import torch
from scipy.spatial import cKDTree

working_dir = '/home/karbabi/spatial-pregnancy'
target = '05 OB-IMN GABA'

datasets = {
    'merfish': 'sample',
    'slidetags': 'sample',
    'xenium': 'sample_rep',
}

ref = sc.read_h5ad(f'{working_dir}/input/adata_ref_zeng_bridge.h5ad')
nbrs = np.load(f'{working_dir}/input/ref_scrna_neighbors.npz')
scrna_class_labels = np.asarray(ref.uns['scrna_class_labels']).astype(str)
scrna_subclass_labels = np.asarray(ref.uns['scrna_subclass_labels']).astype(str)

# ref stats
ref_class = ref.obs['class'].astype(str).values
ref_subclass = ref.obs['subclass'].astype(str).values
n_ref_total = len(ref)
n_ref_target = (ref_class == target).sum()
print(f'=== REF STATS ===')
print(f'total ref cells: {n_ref_total:,}')
print(f'ref cells of class {target}: {n_ref_target:,} '
      f'({n_ref_target/n_ref_total*100:.2f}%)')
ref_target_subs = np.unique(ref_subclass[ref_class == target])
print(f'subclasses in {target}: {list(ref_target_subs)}')

# scRNA stats
n_scrna_target = (scrna_class_labels == target).sum()
print(f'\nscRNA cells of class {target}: {n_scrna_target:,} '
      f'/ {len(scrna_class_labels):,} '
      f'({n_scrna_target/len(scrna_class_labels)*100:.2f}%)')
print()

for name, sample_col in datasets.items():
    print(f'\n=== {name.upper()} ===')
    adata = sc.read_h5ad(
        f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad')
    data = np.load(
        f'{working_dir}/output/{name}/query_scrna_harmony.npz',
        allow_pickle=True)
    query_harmony = data['query_harmony']
    scrna_harmony = data['scrna_harmony']

    # load query coords via bridge indexing (post-fix)
    coords_ffd = torch.load(
        f'{working_dir}/output/{name}/coords_ffd.pt', weights_only=False)
    ref_sections = sorted(
        s for s in ref.obs['sample'].unique()
        if s.startswith('C57BL6J-638850'))
    ref_global_idx = np.concatenate([
        np.where(ref.obs['sample'] == s)[0] for s in ref_sections])
    ref_coords = ref.obs.iloc[ref_global_idx][['x_raw', 'y_raw']].values

    query_coords = adata.obs[['x_ffd', 'y_ffd']].values

    n_total = len(adata)
    target_mask = adata.obs['class'] == target
    n_target = target_mask.sum()
    print(f'assigned {target}: {n_target:,} / {n_total:,} '
          f'({n_target/n_total*100:.2f}%)')
    print(f'expected (ref proportion): '
          f'{n_ref_target/n_ref_total*100:.2f}% '
          f'= {int(n_total * n_ref_target/n_ref_total):,}')
    print(f'over-assignment ratio: '
          f'{(n_target/n_total)/(n_ref_target/n_ref_total):.1f}x')

    # unconstrained comparison
    if 'class_unconstrained' in adata.obs.columns:
        unc_mask = adata.obs['class_unconstrained'] == target
        n_unc = unc_mask.sum()
        print(f'unconstrained {target}: {n_unc:,} '
              f'({n_unc/n_total*100:.2f}%)')
        overlap = (target_mask & unc_mask).sum()
        print(f'  constrained∩unconstrained: {overlap:,} '
              f'({overlap/max(n_target,1)*100:.0f}% of constrained)')

    # QC metrics for target cells
    tgt_obs = adata.obs[target_mask]
    print(f'\n  QC metrics for {target}-assigned cells:')
    for col in ['subclass_confidence', 'subclass_margin', 'min_cos_dist',
                'n_spatial_candidates', 'subclass_bridge_consistency']:
        if col in tgt_obs.columns:
            med = tgt_obs[col].median()
            q25, q75 = tgt_obs[col].quantile([0.25, 0.75])
            print(f'    {col}: median={med:.3f} '
                  f'(Q25={q25:.3f}, Q75={q75:.3f})')

    # what subclasses within the target class are being assigned?
    tgt_subclass_counts = tgt_obs['subclass'].value_counts().head(10)
    print(f'\n  top subclasses assigned under {target}:')
    for s, c in tgt_subclass_counts.items():
        print(f'    {s}: {c:,}')

    # sample 5 spatially diverse target cells
    tgt_idx = np.where(target_mask)[0]
    if len(tgt_idx) == 0:
        continue
    tgt_y = query_coords[tgt_idx, 1]
    sample_idx = tgt_idx[np.argsort(tgt_y)[::max(len(tgt_idx)//5, 1)][:5]]

    # compute radius (same as production)
    def avg_nn_dist(c):
        t = cKDTree(c)
        d, _ = t.query(c, k=2)
        return d[:, 1].mean()
    edge = avg_nn_dist(query_coords)
    fold = {'merfish': 100, 'slidetags': 30, 'xenium': 100}[name]
    radius = fold * edge
    tree = cKDTree(ref_coords)

    print(f'\n  radius: edge={edge:.5f} * fold={fold} = {radius:.4f}')
    print(f'  ref total in class {target} across 4 ref sections: '
          f'{(ref_class[ref_global_idx] == target).sum():,}')

    # 5-cell diagnostic
    print(f'\n  {len(sample_idx)} sample {target} cells:')
    for i in sample_idx:
        qc = query_coords[i]
        cands = tree.query_ball_point(qc, r=radius)
        if len(cands) == 0:
            continue
        cands = np.array(cands)
        ref_g = ref_global_idx[cands]
        ref_classes_near = ref_class[ref_g]
        ref_subs_near = ref_subclass[ref_g]

        n_target_in_ref = (ref_classes_near == target).sum()

        # nearest target ref cell (within all 4 sections)
        target_ref_mask = ref_class[ref_global_idx] == target
        if target_ref_mask.any():
            nearest_d, _ = cKDTree(ref_coords[target_ref_mask]).query(qc, k=1)
        else:
            nearest_d = np.inf

        # scRNA pool
        scrna_pool = nbrs['indices'][ref_g].ravel()
        pool_target = (scrna_class_labels[scrna_pool] == target).sum()
        pool_unique = len(np.unique(scrna_pool))

        # ref-anchored filtering: is target subclass present in ref neighbors?
        ref_subs_present = set(ref_subs_near)
        target_sub_present = any(
            s in ref_subs_present for s in ref_target_subs)

        conf = adata.obs['subclass_confidence'].iloc[i]
        marg = adata.obs['subclass_margin'].iloc[i]
        cos = adata.obs['min_cos_dist'].iloc[i]

        print(f'    cell {i}: ({qc[0]:.2f}, {qc[1]:.2f})')
        print(f'      conf={conf:.2f}, margin={marg:.2f}, cos={cos:.3f}')
        print(f'      {len(cands):,} ref neighbors, '
              f'{n_target_in_ref:,} of class {target}')
        print(f'      nearest target ref cell: d={nearest_d:.4f} '
              f'(radius={radius:.4f})')
        print(f'      scRNA pool: {pool_unique:,} unique, '
              f'{pool_target:,} of class {target}')
        print(f'      target subclass in ref_subs_present: '
              f'{target_sub_present}')
