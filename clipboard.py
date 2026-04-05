import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree, Delaunay

working_dir = '/home/karbabi/spatial-pregnancy'

datasets = {
    'merfish': dict(ave_dist_fold=10, alignment_shift_adjustment=0),
    'slidetags': dict(ave_dist_fold=10, alignment_shift_adjustment=0),
    'xenium': dict(ave_dist_fold=10, alignment_shift_adjustment=0),
}

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
    return edge_dists.mean()

available = []
for name, cfg in datasets.items():
    ffd_path = f'{working_dir}/output/{name}/coords_ffd.pt'
    raw_path = f'{working_dir}/output/{name}/coords_raw.pt'
    if os.path.exists(ffd_path) and os.path.exists(raw_path):
        available.append(name)
    else:
        print(f'[{name}] skipping, no coords_ffd.pt/coords_raw.pt')

n = len(available)
fig, axes = plt.subplots(2, n, figsize=(7 * n, 12), squeeze=False)
rng = np.random.default_rng(42)

for col, name in enumerate(available):
    cfg = datasets[name]
    print(f'[{name}] loading...')
    coords_ffd = torch.load(
        f'{working_dir}/output/{name}/coords_ffd.pt', weights_only=False)
    coords_raw = torch.load(
        f'{working_dir}/output/{name}/coords_raw.pt', weights_only=False)

    ref_sections = sorted(
        s for s in coords_raw if s.startswith('C57BL6J-638850'))
    ref_coords = np.vstack([coords_raw[s] for s in ref_sections])
    query_coords = np.vstack(
        [coords_ffd[s] for s in sorted(coords_ffd.keys())])

    avg_edge = compute_avg_delaunay_dist(query_coords)
    radius = cfg['ave_dist_fold'] * avg_edge + cfg['alignment_shift_adjustment']
    print(f'[{name}] avg_edge={avg_edge:.4f}, radius={radius:.4f}, '
          f'ref={ref_coords.shape[0]:,}, query={query_coords.shape[0]:,}')

    # pick a query cell near center of tissue
    center = query_coords.mean(axis=0)
    dists_to_center = np.linalg.norm(query_coords - center, axis=1)
    central_idx = np.where(
        dists_to_center < np.percentile(dists_to_center, 10))[0]
    cell_idx = rng.choice(central_idx)
    cell = query_coords[cell_idx]

    tree = cKDTree(ref_coords)
    candidates = tree.query_ball_point(cell, r=radius)

    # --- top row: full tissue with circle ---
    ax = axes[0, col]
    ax.scatter(ref_coords[:, 0], ref_coords[:, 1],
               s=0.05, c='lightgray', alpha=0.15, rasterized=True)
    ax.scatter(query_coords[:, 0], query_coords[:, 1],
               s=0.05, c='#aec7e8', alpha=0.1, rasterized=True)
    ax.scatter(cell[0], cell[1], s=30, c='red', zorder=5)
    circle = Circle(cell, radius, fill=False, color='red',
                    linewidth=1.5, linestyle='--')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title(f'{name} (full)\navg_edge={avg_edge:.4f}, '
                 f'r={radius:.4f}')
    ax.axis('off')

    # --- bottom row: zoomed to ~8x radius ---
    ax = axes[1, col]
    zoom = max(radius * 8, 0.3)
    xlo, xhi = cell[0] - zoom, cell[0] + zoom
    ylo, yhi = cell[1] - zoom, cell[1] + zoom

    # only plot points within the zoom window (+ margin)
    margin = zoom * 0.2
    ref_mask = ((ref_coords[:, 0] > xlo - margin) &
                (ref_coords[:, 0] < xhi + margin) &
                (ref_coords[:, 1] > ylo - margin) &
                (ref_coords[:, 1] < yhi + margin))
    q_mask = ((query_coords[:, 0] > xlo - margin) &
              (query_coords[:, 0] < xhi + margin) &
              (query_coords[:, 1] > ylo - margin) &
              (query_coords[:, 1] < yhi + margin))

    ax.scatter(ref_coords[ref_mask, 0], ref_coords[ref_mask, 1],
               s=3, c='lightgray', alpha=0.4, rasterized=True,
               label='ref')
    ax.scatter(query_coords[q_mask, 0], query_coords[q_mask, 1],
               s=3, c='#aec7e8', alpha=0.4, rasterized=True,
               label='query')
    if len(candidates) > 0:
        ax.scatter(ref_coords[candidates, 0], ref_coords[candidates, 1],
                   s=10, c='steelblue', alpha=0.8, zorder=3,
                   label=f'{len(candidates)} candidates')
    ax.scatter(cell[0], cell[1], s=60, c='red', zorder=5,
               edgecolors='black', linewidths=0.5, label='query cell')
    circle = Circle(cell, radius, fill=False, color='red',
                    linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect('equal')
    ax.set_title(f'{name} (zoomed)\n{len(candidates)} ref candidates '
                 f'within r={radius:.4f}')
    ax.legend(loc='upper right', fontsize=7, markerscale=2)

plt.tight_layout()
fig.savefig(f'{working_dir}/figures/radius_visualization.png', dpi=200)
plt.close()
print('saved radius_visualization.png')
