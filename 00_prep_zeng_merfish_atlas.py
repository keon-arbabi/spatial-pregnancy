import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')

###############################################################################

# load zeng reference, output from `merfish_zeng_prep_atlas.py`
# the X matrix is log CPM
cell_joined = pd.read_csv('single-cell/ABC/metadata/cells_joined.csv')

sections = [
    'C57BL6J-638850.49', 'C57BL6J-638850.48',
    'C57BL6J-638850.47', 'C57BL6J-638850.46'
]
adata_input = ad.read_h5ad(
    'single-cell/ABC/expression_matrices/MERFISH-C57BL6J-638850/'
    '20230830/C57BL6J-638850-raw.h5ad')

adatas_processed = []
for section in sections:
    adata = adata_input[adata_input.obs['brain_section_label'] == section]
    adata.obs = adata.obs.reset_index()
    adata.obs = pd.merge(adata.obs, cell_joined, on='cell_label',
                         how='left')
    adata.obs = adata.obs.set_index('cell_label', drop=True)
    exclude = ['unassigned', 'brain-unassigned', 'fiber tracts-unassigned']
    adata = adata[~adata.obs['parcellation_division'].isin(exclude)]
    adata = adata[adata.obs['x_ccf'].notna()]
    adata.var = adata.var.reset_index()

    adata.obs['z_ccf'] *= -1
    adata.obs['y_ccf'] *= -1

    adata.obs['z'] = adata.obs['z_ccf']
    adata.obs['y'] = adata.obs['y_ccf']
    adata.obs['x'] = adata.obs['x_ccf']
    adata.obs['sample'] = section
    adata.obs['source'] = 'Zeng-ABCA-Reference'
    print(f'[{section}] {adata.shape[0]} cells')
    adatas_processed.append(adata)

adata_combined = ad.concat(adatas_processed, axis=0, merge='same')
adata_combined.var = adata_input.var.reset_index().set_index('gene_symbol')
adata_combined.var['gene_symbol'] = adata_combined.var.index
adata_combined.var = adata_combined.var.rename_axis(None)
adata_combined = adata_combined[:,
                 ~adata_combined.var.index.duplicated(keep='first')].copy()

# save as sparse matrix
adata_combined.X = sparse.csr_matrix(adata_combined.X.astype(np.float32))
adata_combined.write('spatial-pregnancy/input/adata_ref_zeng_raw')

# plot each slice (tilted 3D with z-axis)
for selection in [
    'class_color']:
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Zeng ABCA Reference', fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    # shear AP-DV plane: fix bottom, tilt top forward
    dv_vals = adata_combined.obs['y'].values
    dv_norm = (dv_vals - dv_vals.min()) / (dv_vals.max() - dv_vals.min())
    shear = 0.7 # adjust to control tilt correction
    ap_plot = adata_combined.obs['x'].values - shear * dv_norm
    ax.scatter(adata_combined.obs['z'], ap_plot,
               adata_combined.obs['y'], s=0.8, alpha=0.2,
               c=adata_combined.obs[selection])
    ax.set_xlabel('Mid-line')
    ax.set_ylabel('Anterior-Prosterior')
    ax.set_zlabel('Dorals-Ventral')
    ax.invert_yaxis()
    dv_range = adata_combined.obs['y'].max() - adata_combined.obs['y'].min()
    ap_range = adata_combined.obs['x'].max() - adata_combined.obs['x'].min()
    ax.set_box_aspect([dv_range, ap_range * 10, dv_range])
    ax.view_init(elev=8, azim=195, roll=0)
    plt.tight_layout()
    plt.savefig(
        f'spatial-pregnancy/figures/reference/zeng_reference_{selection}.png',
        dpi=200, bbox_inches='tight', pad_inches=0.5)
