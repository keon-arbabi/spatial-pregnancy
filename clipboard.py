import scanpy as sc
import matplotlib.pyplot as plt

working_dir = '/home/karbabi/spatial-pregnancy'
ref = sc.read_h5ad(f'{working_dir}/input/adata_ref_zeng_raw.h5ad')
samples = sorted(ref.obs['sample'].unique())

fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 5), squeeze=False)
fig.suptitle('Reference x_raw vs y_raw (should be coronal sections)', fontsize=16)
for i, s in enumerate(samples):
    ax = axes[0, i]
    obs = ref.obs[ref.obs['sample'] == s]
    xr = obs['x_raw'].max() - obs['x_raw'].min()
    yr = obs['y_raw'].max() - obs['y_raw'].min()
    ax.scatter(obs['x_raw'], obs['y_raw'], s=0.3, alpha=0.3)
    ax.set_title(f'{s}\nx:{xr:.2f} y:{yr:.2f}')
    ax.set_aspect('equal')
    ax.axis('off')
plt.tight_layout()
plt.savefig(f'{working_dir}/figures/ref_coords_check.png', dpi=150)
plt.close()
print('saved ref_coords_check.png')
