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

working_dir = '/home/karbabi/spatial-pregnancy'
ref_dir = 'single-cell/ABC'
cell_joined = pd.read_csv(f'{ref_dir}/metadata/cells_joined.csv')

sections = [
    'C57BL6J-638850.49', 'C57BL6J-638850.48',
    'C57BL6J-638850.47', 'C57BL6J-638850.46'
]
data_types = [
    ('raw',
     'MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-raw.h5ad',
     'adata_ref_zeng_raw.h5ad'),
    ('imputed',
     'MERFISH-C57BL6J-638850-imputed/20240831/C57BL6J-638850-imputed-log2.h5ad',
     'adata_ref_zeng_imputed.h5ad'),
]

for data_type, input_path, output_filename in data_types:
    print(f'Processing {data_type} data...')
    adata_input = ad.read_h5ad(f'{ref_dir}/expression_matrices/{input_path}')

    adatas_processed = []
    for section in sections:
        adata = adata_input[adata_input.obs['brain_section_label'] == section]
        adata.obs = adata.obs.reset_index()
        adata.obs = pd.merge(adata.obs, cell_joined, on='cell_label',
                             how='left')
        adata.obs = adata.obs.set_index('cell_label', drop=True)
        exclude = ['unassigned', 'brain-unassigned',
                    'fiber tracts-unassigned']
        adata = adata[~adata.obs['parcellation_division'].isin(exclude)]
        adata = adata[adata.obs['x_ccf'].notna()]
        adata.var = adata.var.reset_index()

        adata.obs['z_ccf'] *= -1
        adata.obs['y_ccf'] *= -1

        adata.obs['x_raw'] = adata.obs['z_ccf']
        adata.obs['y_raw'] = adata.obs['y_ccf']
        adata.obs['z_raw'] = adata.obs['x_ccf']
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

    adata_combined.X = sparse.csr_matrix(adata_combined.X.astype(np.float32))
    adata_combined.write(f'{working_dir}/input/{output_filename}')
    print(f'Saved {data_type} data to {output_filename}')

obs = adata_combined.obs
for selection in ['class_color']:
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Zeng ABCA Reference', fontsize=16)
    ax = fig.add_subplot(111, projection='3d')
    dv_vals = obs['y_ccf'].values
    dv_norm = (dv_vals - dv_vals.min()) / (dv_vals.max() - dv_vals.min())
    shear = 0.7
    ap_plot = obs['x_ccf'].values - shear * dv_norm
    ax.scatter(obs['z_ccf'], ap_plot, obs['y_ccf'],
               s=0.8, alpha=0.2, c=obs[selection])
    ax.set_xlabel('Mid-line')
    ax.set_ylabel('Anterior-Posterior')
    ax.set_zlabel('Dorsal-Ventral')
    ax.invert_yaxis()
    dv_range = obs['y_ccf'].max() - obs['y_ccf'].min()
    ap_range = obs['x_ccf'].max() - obs['x_ccf'].min()
    ax.set_box_aspect([dv_range, ap_range * 10, dv_range])
    ax.view_init(elev=8, azim=195, roll=0)
    plt.tight_layout()
    plt.savefig(
        f'spatial-pregnancy/figures/reference_3d.png',
        dpi=200, bbox_inches='tight', pad_inches=0.5)
