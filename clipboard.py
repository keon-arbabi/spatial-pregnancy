import sys, os
sys.path.insert(0, os.path.expanduser('~'))
import polars as pl
from single_cell import SingleCell

# check all datasets that go through set_var_names for duplicates
print('=== duplicate gene_symbol check ===')

scrna = SingleCell('single-cell/ABC/zeng_combined_10Xv3.h5ad').skip_qc()
print(f'scrna: var_names={scrna.var_names.name!r}, '
      f'n_unique={scrna.var_names.n_unique()}, len={len(scrna.var_names)}, '
      f'dupes={len(scrna.var_names) - scrna.var_names.n_unique()}')

merfish = SingleCell(
    f'{os.getcwd()}/spatial-pregnancy/input/adata_ref_zeng_raw.h5ad').skip_qc()
merfish = merfish.set_var_names('gene_symbol')
print(f'merfish: var_names={merfish.var_names.name!r}, '
      f'n_unique={merfish.var_names.n_unique()}, len={len(merfish.var_names)}, '
      f'dupes={len(merfish.var_names) - merfish.var_names.n_unique()}')

for name in ['merfish', 'slidetags', 'xenium']:
    path = f'spatial-pregnancy/output/{name}/01_adata_query_{name}.h5ad'
    if not os.path.exists(path):
        print(f'{name} query: NOT FOUND')
        continue
    q = SingleCell(path).skip_qc()
    if 'gene_symbol' in q.var.columns:
        q = q.set_var_names('gene_symbol')
    print(f'{name} query: var_names={q.var_names.name!r}, '
          f'n_unique={q.var_names.n_unique()}, len={len(q.var_names)}, '
          f'dupes={len(q.var_names) - q.var_names.n_unique()}')

# test the fix: dedup scrna, then run the failing pipeline
print('\n=== testing fix: make_var_names_unique ===')
scrna = scrna.make_var_names_unique(separator='~')
print(f'scrna after dedup: n_unique={scrna.var_names.n_unique()}, '
      f'len={len(scrna.var_names)}')

# filter to subclasses (matching load_scrna_filtered)
merfish_ref = SingleCell(
    f'{os.getcwd()}/spatial-pregnancy/input/adata_ref_zeng_raw.h5ad')
ref_subclasses = merfish_ref.obs['subclass'].cast(str).unique().drop_nulls()
scrna = scrna.filter_obs(
    pl.col('subclass').is_not_null() &
    pl.col('subclass').cast(str).is_in(ref_subclasses))
scrna = scrna.with_columns_obs(pl.lit('scrna-seq').alias('batch'))

query = SingleCell(
    'spatial-pregnancy/output/slidetags/01_adata_query_slidetags.h5ad')
query = query.skip_qc()
query = query.rename_obs({'_index': 'cell_label'})
if 'gene_symbol' in query.var.columns:
    query = query.set_var_names('gene_symbol')
else:
    query = query.rename_var({'_index': 'gene_symbol'})
query = query.make_var_names_unique(separator='~')
query = query.with_columns_obs(
    pl.col('sample').cast(pl.String).alias('batch'))

# integrate_harmony steps
scrna = scrna.with_uns(normalized=False)
query = query.with_uns(normalized=False)
scrna, query = scrna.hvg(query)
scrna = scrna.normalize()
query = query.normalize()

print(f'\nafter hvg+normalize:')
print(f'scrna: X={scrna.X.shape}, var={scrna.var.shape}, '
      f'match={scrna.var.shape[0] == scrna.X.shape[1]}')
print(f'query: X={query.X.shape}, var={query.var.shape}, '
      f'match={query.var.shape[0] == query.X.shape[1]}')

# concat_obs (where it was failing)
for col in ['class', 'subclass']:
    if col not in query.obs.columns:
        query = query.with_columns_obs(
            pl.lit('Unlabelled').cast(pl.Categorical).alias(col))
    if col in scrna.obs.columns:
        scrna = scrna.with_columns_obs(pl.col(col).cast(pl.Categorical))
    if col in query.obs.columns:
        query = query.with_columns_obs(pl.col(col).cast(pl.Categorical))

print('\ntrying concat_obs...')
try:
    combined = scrna.concat_obs(query, flexible=True)
    print(f'SUCCESS: {combined.X.shape}')
except Exception as e:
    print(f'FAILED: {type(e).__name__}: {e}')
