import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Delaunay

working_dir = '/home/karbabi/spatial-pregnancy'
name = 'merfish'
sample_col = 'sample'

# load coords
coords_ffd = torch.load(f'{working_dir}/output/{name}/coords_ffd.pt',
                         weights_only=False)
coords_raw = torch.load(f'{working_dir}/output/{name}/coords_raw.pt',
                         weights_only=False)
ref_sections = sorted(s for s in coords_raw if s.startswith('C57BL6J-638850'))
ref_coords = np.vstack([coords_raw[s] for s in ref_sections])
query_coords = np.vstack([coords_ffd[s] for s in sorted(coords_ffd.keys())])

# compute avg delaunay edge dist (subsampled)
def compute_avg_delaunay_dist(coords, quantile=0.99, max_subsample=50_000,
                              seed=0):
    if coords.shape[0] > max_subsample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(coords.shape[0], max_subsample, replace=False)
        coords = coords[idx]
    coords = np.unique(coords, axis=0)
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = int(simplex[i]), int(simplex[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    edges = np.array(list(edges))
    edge_dists = np.linalg.norm(
        coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    threshold = np.percentile(edge_dists, quantile * 100)
    edge_dists = edge_dists[edge_dists <= threshold]
    return edge_dists.mean(), edge_dists

avg_dist, edge_dists = compute_avg_delaunay_dist(query_coords)
print(f'avg Delaunay edge dist: {avg_dist:.1f}')
print(f'edge dist percentiles: '
      f'p25={np.percentile(edge_dists, 25):.1f}, '
      f'p50={np.percentile(edge_dists, 50):.1f}, '
      f'p75={np.percentile(edge_dists, 75):.1f}, '
      f'p95={np.percentile(edge_dists, 95):.1f}')

# test different radii
tree = cKDTree(ref_coords)
folds = [1, 2, 3, 5]
shifts = [0, 25, 50, 100]

fig, axes = plt.subplots(len(folds), len(shifts), figsize=(20, 16))
fig.suptitle(f'{name}: n_spatial_candidates per (fold, shift)', fontsize=14)

for row, fold in enumerate(folds):
    for col, shift in enumerate(shifts):
        r = fold * avg_dist + shift
        candidates = tree.query_ball_point(query_coords, r=r)
        n_cands = np.array([len(c) for c in candidates])
        n_empty = (n_cands == 0).sum()

        ax = axes[row, col]
        ax.hist(n_cands[n_cands > 0], bins=50, alpha=0.7, color='steelblue')
        ax.set_title(f'fold={fold}, shift={shift}\n'
                     f'r={r:.0f}, empty={n_empty:,}\n'
                     f'median={np.median(n_cands):.0f}, '
                     f'p95={np.percentile(n_cands, 95):.0f}')
        ax.set_xlabel('n candidates')
        ax.set_ylabel('n cells')

plt.tight_layout()
fig.savefig(f'{working_dir}/figures/{name}_radius_sweep.png', dpi=150)
plt.close()
print(f'saved {name}_radius_sweep.png')
