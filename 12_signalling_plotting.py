
#region plot — interaction range visualization ################################

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

from scipy.spatial import KDTree
from matplotlib.colors import Normalize
from matplotlib import cm

n_exemplars = 5
fig, axes = plt.subplots(
    len(datasets), n_exemplars + 1,
    figsize=(3 * (n_exemplars + 1), 2.8 * len(datasets)),
    squeeze=False)
rng = np.random.default_rng(42)

for row, (name, cfg) in enumerate(datasets.items()):
    adata = adatas[name]
    sample = sorted(adata.obs['sample'].unique())[0]
    sub = adata.obs[adata.obs['sample'] == sample]
    coords_raw = sub[['x_raw', 'y_raw']].to_numpy(dtype=np.float64)
    coords_aff = sub[['x_affine', 'y_affine']].to_numpy(dtype=np.float64)
    sample_idx = rng.choice(min(1000, len(coords_raw)),
                            size=min(1000, len(coords_raw)), replace=False)
    cf = float(np.median(pdist(coords_raw[sample_idx])) /
               np.median(pdist(coords_aff[sample_idx])))
    r_affine = cfg['interaction_range'] / cf
    r_contact = CONTACT_RANGE / cf
    sd = cf / cfg['interaction_range']
    ax = axes[row, 0]
    ax.scatter(coords_aff[:, 0], coords_aff[:, 1], s=0.5, c='lightgray',
               alpha=0.5, linewidth=0, rasterized=True)
    exemplar_idx = rng.integers(len(coords_aff), size=n_exemplars)
    exemplars = coords_aff[exemplar_idx]
    ax.scatter(exemplars[:, 0], exemplars[:, 1], s=20, c='red', zorder=5)
    for cell in exemplars:
        ax.add_patch(Circle(cell, r_affine, fill=False, color='red',
                            linewidth=0.8, linestyle='--'))
        ax.add_patch(Circle(cell, r_contact, fill=False, color='orange',
                            linewidth=0.8, linestyle='-'))
    ax.set_aspect('equal')
    ax.set_title(f'{name} — {sample}\n'
                 f'interaction={cfg["interaction_range"]}μm, '
                 f'contact={CONTACT_RANGE}μm\n'
                 f'scale.distance={sd:.4f}',
                 fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    tree = KDTree(coords_aff)
    cmap = cm.Blues
    for col, (idx, cell) in enumerate(zip(exemplar_idx, exemplars), start=1):
        neighbor_idx = tree.query_ball_point(cell, r=r_affine)
        neighbor_idx = [i for i in neighbor_idx if i != idx]
        zoom = r_affine * 2.5
        ax = axes[row, col]
        in_zoom = ((np.abs(coords_aff[:, 0] - cell[0]) < zoom * 1.5) &
                   (np.abs(coords_aff[:, 1] - cell[1]) < zoom * 1.5))
        ax.scatter(coords_aff[in_zoom, 0], coords_aff[in_zoom, 1],
                   s=6, c='lightgray', alpha=0.7, linewidth=0,
                   rasterized=True)
        if neighbor_idx:
            nb_coords = coords_aff[neighbor_idx]
            dists_aff = np.sqrt(((nb_coords - cell) ** 2).sum(axis=1))
            dists_um = dists_aff * cf
            weights = np.exp(-sd * dists_um)
            colors = cmap(Normalize(0, 1)(weights))
            ax.scatter(nb_coords[:, 0], nb_coords[:, 1],
                       s=10, c=colors, linewidth=0, rasterized=True)
        ax.scatter(cell[0], cell[1], s=80, c='red', zorder=5,
                   edgecolors='white', linewidths=1)
        ax.add_patch(Circle(cell, r_affine, fill=False, color='red',
                            linewidth=1.2, linestyle='--'))
        ax.add_patch(Circle(cell, r_contact, fill=False, color='orange',
                            linewidth=1.2, linestyle='-'))
        ax.set_xlim(cell[0] - zoom, cell[0] + zoom)
        ax.set_ylim(cell[1] - zoom, cell[1] + zoom)
        ax.set_aspect('equal')
        ax.set_title(f'{len(neighbor_idx):,} in range', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

plt.tight_layout()
os.makedirs(f'{working_dir}/figures', exist_ok=True)
plt.savefig(f'{working_dir}/figures/cellchat_interaction_range.png',
            dpi=200, bbox_inches='tight')
plt.close()
