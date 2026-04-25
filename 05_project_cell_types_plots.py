#region imports and setup ######################################################

import os
import sys
import torch
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
working_dir = '/home/karbabi/spatial-pregnancy'

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
    'xenium': dict(
        sample_col='sample_rep',
        ave_dist_fold=100, alignment_shift_adjustment=0),
    'slidetags': dict(
        sample_col='sample',
        ave_dist_fold=30, alignment_shift_adjustment=0),
    'merfish': dict(
        sample_col='sample',
        ave_dist_fold=100, alignment_shift_adjustment=0),
}

def avg_nn_dist(coords):
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return dists[:, 1].mean()

def load_ref_coords():
    ref = sc.read_h5ad(f'{working_dir}/input/adata_ref_zeng_bridge.h5ad')
    ref_sections = sorted(
        s for s in ref.obs['sample'].unique()
        if s.startswith('C57BL6J-638850'))
    ref_obs_sub = ref.obs[ref.obs['sample'].isin(ref_sections)]
    return ref_obs_sub[['x_raw', 'y_raw']].values

#endregion

#region plot spatial maps ######################################################

def plot_spatial_maps(name, levels=None):
    if levels is None:
        levels = ['class', 'subclass']

    cfg = datasets[name]
    sample_col = cfg['sample_col']
    fig_dir = f'{working_dir}/figures'
    os.makedirs(fig_dir, exist_ok=True)

    output_path = f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad'
    adata = sc.read_h5ad(output_path)
    ref_coords = load_ref_coords()

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

    print(f'[{name}] spatial maps done')

#endregion

#region adaptive radius visualization ##########################################

def plot_adaptive_radius():
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
        ax.scatter(
            sample_cells[:, 0], sample_cells[:, 1],
            s=20, c='red', zorder=5)
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

#region run ####################################################################

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <dataset|all|radius>')
        print(f'Datasets: {", ".join(datasets.keys())}')
        sys.exit(1)
    t = sys.argv[1]
    if t == 'radius':
        plot_adaptive_radius()
    elif t == 'all':
        for name in datasets:
            plot_spatial_maps(name)
        plot_adaptive_radius()
    elif t in datasets:
        plot_spatial_maps(t)
    else:
        print(f'Unknown: {t}. Choose from {list(datasets.keys())}, all, radius')
        sys.exit(1)

#endregion
