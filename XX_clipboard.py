import numpy as np
import scanpy as sc
import torch
from scipy.spatial import cKDTree

working_dir = '/home/karbabi/spatial-pregnancy'
target = '10 LSX GABA'
name = 'merfish'

adata = sc.read_h5ad(f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad')
ref = sc.read_h5ad(f'{working_dir}/input/adata_ref_zeng_bridge.h5ad')
nbrs = np.load(f'{working_dir}/input/ref_scrna_neighbors.npz')
scrna_class = np.asarray(ref.uns.get('scrna_class_labels', [])).astype(str)
scrna_subclass = np.asarray(ref.uns['scrna_subclass_labels']).astype(str)

coords_ffd = torch.load(f'{working_dir}/output/{name}/coords_ffd.pt',
                         weights_only=False)
coords_raw = torch.load(f'{working_dir}/output/{name}/coords_raw.pt',
                         weights_only=False)
ref_sections = sorted(s for s in coords_raw if s.startswith('C57BL6J-638850'))
ref_coords = np.vstack([coords_raw[s] for s in ref_sections])
ref_global_idx = np.concatenate([
    np.where(ref.obs['sample'] == s)[0] for s in ref_sections])

tree = cKDTree(ref_coords)
query_coords = adata.obs[['x_ffd', 'y_ffd']].values
target_mask = adata.obs['class'] == target

# pick 5 LSX-assigned query cells from spatially diverse positions
target_idx = np.where(target_mask)[0]
print(f'total query cells assigned {target}: {len(target_idx):,}')

# sort by y_ffd to get cells from different regions
target_y = query_coords[target_idx, 1]
sample_idx = target_idx[np.argsort(target_y)[::len(target_idx)//5][:5]]

# real LSX class for reference (use class column too)
ref_class = ref.obs['class'].astype(str).values
n_ref_lsx = (ref_class == target).sum()
print(f'total ref cells of class {target}: {n_ref_lsx:,}\n')

# typical radius from production
fold = 10
def avg_nn_full(coords):
    t = cKDTree(coords)
    d, _ = t.query(coords, k=2)
    return d[:, 1].mean()
edge_nn = avg_nn_full(query_coords)
print(f'avg_nn (full): {edge_nn:.5f}')
print(f'true r at fold=10: {fold * edge_nn:.4f}')

# what we actually used
from scipy.spatial import Delaunay
def avg_delaunay_sub(coords):
    rng = np.random.default_rng(0)
    if len(coords) > 50_000:
        coords = coords[rng.choice(len(coords), 50_000, replace=False)]
    coords = np.unique(coords, axis=0)
    tri = Delaunay(coords)
    edges = set()
    for s in tri.simplices:
        for i in range(3):
            a, b = int(s[i]), int(s[(i+1)%3])
            edges.add((min(a,b), max(a,b)))
    edges = np.array(list(edges))
    d = np.linalg.norm(coords[edges[:,0]] - coords[edges[:,1]], axis=1)
    return d[d <= np.percentile(d, 99)].mean()

edge_sub = avg_delaunay_sub(query_coords)
r_used = fold * edge_sub
print(f'subsampled edge: {edge_sub:.5f}, r_used = {r_used:.4f}\n')

for i in sample_idx:
    qc = query_coords[i]
    cands = tree.query_ball_point(qc, r=r_used)
    if len(cands) == 0:
        print(f'cell {i}: ({qc[0]:.3f}, {qc[1]:.3f}) — NO ref neighbors')
        continue
    ref_g = ref_global_idx[np.array(cands)]
    ref_subs = ref.obs['subclass'].iloc[ref_g].astype(str).values
    ref_classes = ref.obs['class'].iloc[ref_g].astype(str).values

    n_lsx_ref = (ref_classes == target).sum()
    nearest_lsx_d, _ = cKDTree(
        ref_coords[ref_class[ref_global_idx] == target]
    ).query(qc, k=1) if n_ref_lsx > 0 else (np.inf, None)

    print(f'cell {i}: ({qc[0]:.3f}, {qc[1]:.3f})')
    print(f'  confidence={adata.obs["subclass_confidence"].iloc[i]:.3f}, '
          f'cos_dist={adata.obs["min_cos_dist"].iloc[i]:.3f}')
    print(f'  {len(cands):,} ref neighbors, '
          f'{n_lsx_ref:,} of class {target}')
    print(f'  nearest LSX ref cell: d={nearest_lsx_d:.4f} '
          f'(radius={r_used:.4f})')

    # ref class composition
    cls_counts = {}
    for c in ref_classes:
        cls_counts[c] = cls_counts.get(c, 0) + 1
    top5 = sorted(cls_counts.items(), key=lambda x: -x[1])[:5]
    print(f'  top 5 ref classes in radius:')
    for c, n in top5:
        print(f'    {c}: {n:,}')

    # scRNA pool composition (what bridge expansion gives)
    scrna_pool = nbrs['indices'][ref_g].ravel()
    pool_classes = scrna_class[scrna_pool] if len(scrna_class) > 0 else \
        np.array(['?'] * len(scrna_pool))

    # ref-anchored filter: which subclasses are present in ref neighbors?
    ref_subs_set = set(ref_subs)
    pool_filter = np.isin(scrna_subclass[scrna_pool], list(ref_subs_set))
    n_pool_total = len(np.unique(scrna_pool))
    n_pool_filtered = len(np.unique(scrna_pool[pool_filter]))
    n_pool_lsx_class = (pool_classes == target).sum()
    print(f'  scRNA pool: {n_pool_total:,} unique total, '
          f'{n_pool_filtered:,} after ref-anchored filter')
    print(f'  scRNA pool entries of class {target}: '
          f'{n_pool_lsx_class:,}')
    print(f'  is "{target}" in ref subclasses present? '
          f'{any(target in s for s in ref_subs_set)}')
    print()
