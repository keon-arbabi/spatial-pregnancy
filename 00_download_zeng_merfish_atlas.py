import warnings
from pathlib import Path
from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache

warnings.filterwarnings("ignore")

# https://github.com/AllenInstitute/abc_atlas_access/blob/main/notebooks/merfish_ccf_registration_tutorial.ipynb
# https://github.com/AllenInstitute/abc_atlas_access/blob/main/notebooks/merfish_imputed_genes_example.ipynb

# imputed update ###############################################################

download_base = Path('single-cell/ABC')
abc_cache = AbcProjectCache.from_cache_dir(download_base)
abc_cache.load_manifest('releases/20240831/manifest.json')

abc_cache.current_manifest
abc_cache.cache.manifest_file_names
abc_cache.cache.manifest_file_names.append('releases/20240831/manifest.json')
abc_cache.load_manifest('releases/20240831/manifest.json')

cell = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850',
    file_name='cell_metadata_with_cluster_annotation')
cell.rename(columns={
    'x': 'x_section',
    'y': 'y_section',
    'z': 'z_section'},
    inplace=True
)
cell.set_index('cell_label', inplace=True)

reconstructed_coords = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850-CCF',
    file_name='reconstructed_coordinates',
    dtype={"cell_label": str}
)
reconstructed_coords.rename(columns={
    'x': 'x_reconstructed',
    'y': 'y_reconstructed',
    'z': 'z_reconstructed'},
    inplace=True)
reconstructed_coords.set_index('cell_label', inplace=True)
cell_joined = cell.join(reconstructed_coords, how='inner')

ccf_coords = abc_cache.get_metadata_dataframe(
    directory='MERFISH-C57BL6J-638850-CCF',
    file_name='ccf_coordinates',
    dtype={"cell_label": str}
)
ccf_coords.rename(columns={
    'x': 'x_ccf',
    'y': 'y_ccf',
    'z': 'z_ccf'},
    inplace=True)
ccf_coords.drop(['parcellation_index'], axis=1, inplace=True)
ccf_coords.set_index('cell_label', inplace=True)
cell_joined = cell_joined.join(ccf_coords, how='inner')

parcellation_annotation = abc_cache.get_metadata_dataframe(
    directory='Allen-CCF-2020',
    file_name='parcellation_to_parcellation_term_membership_acronym')
parcellation_annotation.set_index('parcellation_index', inplace=True)
parcellation_annotation.columns = ['parcellation_%s'% x for x in
                                   parcellation_annotation.columns]

parcellation_color = abc_cache.get_metadata_dataframe(
    directory='Allen-CCF-2020',
    file_name='parcellation_to_parcellation_term_membership_color')
parcellation_color.set_index('parcellation_index', inplace=True)
parcellation_color.columns = [
    'parcellation_%s'% x for x in  parcellation_color.columns]

cell_joined = cell_joined.join(parcellation_annotation, on='parcellation_index')
cell_joined = cell_joined.join(parcellation_color, on='parcellation_index')

cell_joined.to_csv(
    'single-cell/ABC/metadata/cells_joined.csv')

abc_cache.get_data_path(
    'MERFISH-C57BL6J-638850',
    'C57BL6J-638850/raw')

abc_cache.get_data_path(
    'MERFISH-C57BL6J-638850-imputed',
    'C57BL6J-638850-imputed/log2')

