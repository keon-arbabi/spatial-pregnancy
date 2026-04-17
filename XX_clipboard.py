












































import polars as pl
import pandas as pd
from single_cell import SingleCell, Timer

DATA_PATH = 'spatial-pregnancy/output/slidetags/03_adata_query_slidetags.h5ad'
# DATA_PATH = 'spatial-pregnancy/output/xenium/03_adata_query_xenium.h5ad'
# DATA_PATH = 'spatial-pregnancy/output/merfish/03_adata_query_merfish.h5ad'
NUM_THREADS = -1

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

with Timer('Load data'):
    sc = SingleCell(DATA_PATH, num_threads=1)
    sc = sc.set_num_threads(NUM_THREADS)

with Timer('Quality control'): sc = sc.skip_qc()

with Timer('Feature selection'): sc = sc.hvg(batch_column='sample')

with Timer('Normalization'): sc = sc.normalize()

with Timer('PCA'): sc = sc.pca()

with Timer('Neighbors'): sc = sc.neighbors()

with Timer('Shared neighbors'): sc = sc.shared_neighbors()

with Timer('Embedding'):
    sc = sc.umap(hogwild=True)
    sc = sc.cast_obs({'subclass': pl.String})
    sc.plot_umap(
        'subclass', 'scratch/tmp.png',
        first_color=None, stride=None,
        colormap={k: v for k, v in color_mappings['subclass'].items()
                  if k in sc.obs['subclass']})
