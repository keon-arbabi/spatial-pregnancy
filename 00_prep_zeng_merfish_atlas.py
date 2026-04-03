import os
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import warnings
warnings.filterwarnings('ignore')

working_dir = 'patial-pregnancy-postpart'

########################################################################

# load zeng reference, output from `merfish_zeng_prep_atlas.py`
# the X matrix is log CPM
ref_dir = 'single-cell/ABC'
cell_joined = pd.read_csv(
    f'{ref_dir}/metadata/MERFISH-C57BL6J-638850/20231215/views/'
    'cells_joined.csv')

sections = [
    'C57BL6J-638850.49', 'C57BL6J-638850.48',
    'C57BL6J-638850.47', 'C57BL6J-638850.46'
]
data_type = 'raw'
input_path = 'MERFISH-C57BL6J-638850/20230830/C57BL6J-638850-raw.h5ad'
output_filename = 'adata_ref_zeng_raw.h5ad'

print(f'Processing {data_type} data...')
adata_input = ad.read_h5ad(f'{ref_dir}/expression_matrices/{input_path}')

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

    adata.obs['x'] = adata.obs['z_ccf']
    adata.obs['y'] = adata.obs['y_ccf']
    adata.obs['sample'] = section
    adata.obs['source'] = 'Zeng-ABCA-Reference'
    print(f'[{section}] {adata.shape[0]} cells')
    adatas_processed.append(adata)

adata_combined = ad.concat(adatas_processed, axis=0, merge='same')
adata_combined.var = adata_input.var.reset_index().set_index('gene_symbol')
adata_combined.var['gene_symbol'] = adata_combined.var.index
adata_combined.var = adata_combined.var.rename_axis(None)
adata_combined = adata_combined[:,
                 ~adata_combined.var.index.duplicated(keep='first')]

# save as sparse matrix
adata_combined = adata_combined.copy()
adata_combined.X = sparse.csr_matrix(adata_combined.X.astype(np.float32))
adata_combined.write(f'{working_dir}/output/data/{output_filename}')
print(f'Saved {data_type} data to {output_filename}')

# plot each slice
for selection in [
    'class_color', 'subclass_color', 'supertype_color',
    'parcellation_division_color', 'parcellation_structure_color',
    'parcellation_substructure_color']:
    fig, axes = plt.subplots(1, 4, figsize=(25, 7))
    fig.suptitle('Zeng ABCA Reference', fontsize=16)
    for ax, (sample, data) in zip(axes, adata_combined.obs.groupby('sample')):
        ax.scatter(data['x'], data['y'], s=0.8, c=data[selection])
        ax.set_title(sample)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(
        f'{working_dir}/figures/reference/zeng_reference_{selection}.png',
        dpi=200, bbox_inches='tight', pad_inches=0)
