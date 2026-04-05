#region imports and setup #######################################################

import os
import sys
import json
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import DBSCAN
os.environ['R_HOME'] = '/home/karbabi/miniforge3/lib/R'
from ryp import r, to_r, to_py
sys.path.insert(0, os.path.expanduser('~'))
from single_cell import SingleCell

working_dir = '/home/karbabi/spatial-pregnancy'
for d in ['merfish', 'slidetags', 'xenium']:
    os.makedirs(f'{working_dir}/output/{d}', exist_ok=True)

def run_scdblfinder(adata, output_dir, samples_col):
    file = f'{output_dir}/coldata_scDblFinder.csv'
    if os.path.exists(file):
        coldata = pd.read_csv(file, index_col=0)
    else:
        SingleCell(adata).to_sce('sce')
        to_r(samples_col, 'samples_col')
        r('''
        library(scDblFinder)
        set.seed(123)
        sce = scDblFinder(sce, samples=samples_col)
        table(sce$scDblFinder.class)
        coldata = as.data.frame(colData(sce))
        rm(sce); gc()
        ''')
        coldata = to_py('coldata', format='pandas')
        coldata.to_csv(file)
    adata.obs = adata.obs.join(
        coldata[['scDblFinder.score', 'scDblFinder.class']])

def apply_qc_filters(adata, keep_func, group_col):
    total = len(adata)
    keep_idx = []
    for group in adata.obs[group_col].unique():
        cells = adata.obs.loc[adata.obs[group_col] == group]
        kept = cells.index[keep_func(cells)]
        keep_idx.extend(kept)
        print(f'[{group}] {len(kept):,} / {len(cells):,} cells kept '
              f'({len(kept)/len(cells)*100:.0f}%)')
    dropped = total - len(keep_idx)
    print(f'\ntotal kept: {len(keep_idx):,} / {total:,} '
          f'({len(keep_idx)/total*100:.1f}%), '
          f'dropped: {dropped:,} ({dropped/total*100:.1f}%)')
    return adata[keep_idx].copy()

def save_qc(adata, output_dir, thr, qc_cols):
    pd.DataFrame({c: adata.obs[c].values for c in qc_cols}).to_csv(
        f'{output_dir}/qc_metrics_data.csv', index=False)
    with open(f'{output_dir}/qc_filter_thresholds.json', 'w') as f:
        json.dump(thr, f)

def print_qc_report(adata, filters):
    total = len(adata)
    for name, mask in filters.items():
        print(f'{name}: {mask.sum()} ({mask.sum()/total*100:.1f}%)')

#endregion

#region MERFISH ################################################################

def run_merfish():
    query_dir = f'{working_dir}/input/merfish'
    output_dir = f'{working_dir}/output/merfish'
    samples = sorted(f.replace('.h5ad', '') for f in os.listdir(query_dir))

    adatas = []
    for sample in samples:
        a = sc.read_h5ad(f'{query_dir}/{sample}.h5ad')
        a.obs['sample'] = sample
        a.obs['sample_rep'] = sample
        a.obs['condition'] = sample.rsplit('_', 1)[0]
        a.obs['source'] = 'merfish'
        a.obs = a.obs[[
            'sample', 'sample_rep', 'condition', 'source', 'cell_id',
            'Custom_regions', 'Datasets', 'volume', 'center_x', 'center_y']]
        a.obs = a.obs.rename(columns={
            'Custom_regions': 'custom_regions', 'Datasets': 'datasets',
            'center_x': 'x_raw', 'center_y': 'y_raw'})
        a.obs.index = a.obs['sample'] + '_' + \
            a.obs.index.str.split('_').str[1]
        del a.layers['orig_norm']
        print(f'[{sample}] {a.shape[0]} cells')
        adatas.append(a)
    adata = sc.concat(adatas, axis=0, merge='same')
    adata.var = adata.var.rename(columns={'gene': 'gene_symbol'})

    run_scdblfinder(adata, output_dir, 'sample')
    sc.pp.calculate_qc_metrics(
        adata, percent_top=None, log1p=True, inplace=True)

    thr = {
        'volume_min': 100,
        'n_genes_by_counts': 10,
        'scDblFinder_score': 0.2
    }
    save_qc(adata, output_dir, thr,
        ['sample', 'volume', 'n_genes_by_counts', 'total_counts',
        'scDblFinder.score'])
    print_qc_report(adata, {
        'high_doublet': adata.obs['scDblFinder.score'] > thr['scDblFinder_score'],
        'low_n_genes': adata.obs['n_genes_by_counts'] < thr['n_genes_by_counts'],
        'low_volume': adata.obs['volume'] < thr['volume_min'],
    })
    adata = apply_qc_filters(adata, lambda cells: (
        (cells['scDblFinder.score'] <= thr['scDblFinder_score']) &
        (cells['volume'] >= thr['volume_min']) &
        (cells['volume'] <= 3 * np.median(cells['volume'])) &
        (cells['n_genes_by_counts'] >= thr['n_genes_by_counts'])),
        group_col='sample')

    adata.layers['counts'] = adata.X.copy()
    adata.write(f'{output_dir}/01_adata_query_merfish.h5ad')

    '''
    high_doublet: 291478 (20.3%)
    low_n_genes: 107938 (7.5%)
    low_volume: 6997 (0.5%)
    [CTRL_1] 99,346 / 143,017 cells kept (69%)
    [CTRL_2] 111,967 / 177,697 cells kept (63%)
    [CTRL_3] 103,622 / 143,627 cells kept (72%)
    [POSTPART_1] 79,912 / 120,771 cells kept (66%)
    [POSTPART_2] 151,635 / 235,016 cells kept (65%)
    [POSTPART_3] 109,092 / 154,562 cells kept (71%)
    [PREG_1] 113,572 / 152,660 cells kept (74%)
    [PREG_2] 101,508 / 138,390 cells kept (73%)
    [PREG_3] 119,993 / 173,246 cells kept (69%)

    total kept: 990,647 / 1,438,986 (68.8%), dropped: 448,339 (31.2%)
    '''

#endregion

#region SLIDE-TAGS #############################################################

def run_slidetags():
    query_dir = f'{working_dir}/input/slidetags'
    output_dir = f'{working_dir}/output/slidetags'
    samples = sorted(f.replace('.h5ad', '') for f in os.listdir(query_dir))

    adatas = []
    for sample in samples:
        a = sc.read_h5ad(f'{query_dir}/{sample}.h5ad')
        coords = a.obs[['x', 'y']]
        if sample.startswith('PREG_3'):
            outliers = DBSCAN(eps=800, min_samples=90).fit(coords)
        else:
            outliers = DBSCAN(eps=500, min_samples=110).fit(coords)
        a = a[outliers.labels_ == 0].copy()
        print(f'[{sample}] {a.shape[0]} cells after DBSCAN filtering')
        a.obs['sample_rep'] = sample
        a.obs['sample'] = sample.rsplit('_', 1)[0]
        a.obs['condition'] = sample.split('_')[0]
        a.obs['source'] = 'slidetags'
        a.obs = a.obs[[
            'sample', 'sample_rep', 'condition', 'source',
            'cell_id', 'x', 'y']]
        a.obs = a.obs.rename(columns={'x': 'x_raw', 'y': 'y_raw'})
        a.obs.index = a.obs['sample_rep'].astype(str) + '_' + \
            a.obs.index.str.split('_', n=3).str[3] + '-' + \
            a.obs.index.factorize()[0].astype(str)
        adatas.append(a)
    adata = sc.concat(adatas, axis=0, merge='same')
    adata.var = adata.var.rename(columns={'gene': 'gene_symbol'})

    run_scdblfinder(adata, output_dir, 'sample_rep')
    adata.var['mt'] = adata.var_names.str.startswith('mt-')
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=['mt'], percent_top=None, log1p=True, inplace=True)

    thr = {
        'n_genes_by_counts': 500,
        'total_counts': 750,
        'scDblFinder_score': 0.20,
        'pct_counts_mt': 10
    }
    save_qc(adata, output_dir, thr,
        ['sample', 'n_genes_by_counts', 'total_counts',
         'scDblFinder.score', 'pct_counts_mt'])
    print_qc_report(adata, {
        'high_doublet': adata.obs['scDblFinder.score'] > thr['scDblFinder_score'],
        'low_n_genes': adata.obs['n_genes_by_counts'] < thr['n_genes_by_counts'],
        'low_counts': adata.obs['total_counts'] < thr['total_counts'],
        'high_mt_pct': adata.obs['pct_counts_mt'] > thr['pct_counts_mt'],
    })
    adata = apply_qc_filters(adata, lambda cells: (
        (cells['scDblFinder.score'] <= thr['scDblFinder_score']) &
        (cells['n_genes_by_counts'] >= thr['n_genes_by_counts']) &
        (cells['total_counts'] >= thr['total_counts']) &
        (cells['pct_counts_mt'] <= thr['pct_counts_mt'])),
        group_col='sample_rep')

    adata.layers['counts'] = adata.X.copy()
    adata.write(f'{output_dir}/01_adata_query_slidetags.h5ad')

    '''
    high_doublet: 22298 (17.6%)
    low_n_genes: 6199 (4.9%)
    low_counts: 17001 (13.4%)
    high_mt_pct: 238 (0.2%)
    [CTRL_1_1] 6,916 / 10,260 cells kept (67%)
    [CTRL_1_2] 6,041 / 12,525 cells kept (48%)
    [CTRL_2_1] 8,135 / 9,923 cells kept (82%)
    [CTRL_3_1] 6,699 / 8,657 cells kept (77%)
    [CTRL_3_2] 6,737 / 8,559 cells kept (79%)
    [POSTPART_1_1] 6,194 / 10,513 cells kept (59%)
    [POSTPART_1_2] 5,693 / 10,078 cells kept (56%)
    [POSTPART_2_1] 7,702 / 9,274 cells kept (83%)
    [POSTPART_2_2] 6,209 / 7,770 cells kept (80%)
    [PREG_1_1] 6,900 / 9,985 cells kept (69%)
    [PREG_1_2] 6,073 / 8,158 cells kept (74%)
    [PREG_2_1] 5,523 / 8,493 cells kept (65%)
    [PREG_2_2] 5,410 / 7,326 cells kept (74%)
    [PREG_3_1] 1,197 / 2,209 cells kept (54%)
    [PREG_3_2] 2,062 / 3,223 cells kept (64%)

    total kept: 87,491 / 126,953 (68.9%), dropped: 39,462 (31.1%)
    '''

#endregion

#region XENIUM #################################################################

def run_xenium():
    query_dir = f'{working_dir}/input/xenium'
    output_dir = f'{working_dir}/output/xenium'
    samples = sorted(
        f.replace('.h5ad', '') for f in os.listdir(query_dir)
        if not f.startswith('PREG_1'))

    adatas = []
    ctrl1_parts = []
    for sample in samples:
        a = sc.read_h5ad(f'{query_dir}/{sample}.h5ad')
        a.X = a.layers['counts'].copy()
        del a.layers['counts'], a.layers['data'], a.layers['scale.data']
        a.var = a.var[[]]
        a.obs = a.obs[[
            'sample', 'sample_rep', 'condition', 'source',
            'cell_id', 'x_raw', 'y_raw', 'cell_area', 'nucleus_area']]
        if sample.startswith('CTRL_1'):
            ctrl1_parts.append((sample, a))
        else:
            print(f'[{sample}] {a.shape[0]} cells')
            adatas.append(a)

    # merge CTRL_1 hemispheres: rotate 180, reflect CTRL_1_1, crop, close gap
    # coords are stored pre-rotated; other reps get rotated in cast_mark
    cut = 300
    merged_parts = []
    for sample, a in ctrl1_parts:
        coords = a.obs[['x_raw', 'y_raw']].values.copy()
        coords *= -1  # rotate 180
        if sample == 'CTRL_1_1':
            coords[:, 0] *= -1  # reflect x
            mask = coords[:, 0] > cut
            coords[mask, 0] -= cut
        else:
            mask = coords[:, 0] < -cut
            coords[mask, 0] += cut
        a = a[mask].copy()
        a.obs['x_raw'] = coords[mask, 0]
        a.obs['y_raw'] = coords[mask, 1]
        a.obs['sample_rep'] = 'CTRL_1_1'
        merged_parts.append(a)
    ctrl1 = sc.concat(merged_parts, merge='same')
    n_orig = sum(a.shape[0] for _, a in ctrl1_parts)
    print(f'[CTRL_1] {ctrl1.shape[0]:,} cells (merged from {n_orig:,})')
    adatas.append(ctrl1)

    adata = sc.concat(adatas, axis=0, merge='same')

    run_scdblfinder(adata, output_dir, 'sample_rep')
    sc.pp.calculate_qc_metrics(
        adata, percent_top=None, log1p=True, inplace=True)

    thr = {
        'cell_area_min': 10,
        'n_genes_by_counts': 100,
        'total_counts': 100,
        'scDblFinder_score': 0.2
    }
    save_qc(adata, output_dir, thr,
        ['sample', 'cell_area', 'n_genes_by_counts', 'total_counts',
         'scDblFinder.score'])
    print_qc_report(adata, {
        'high_doublet': adata.obs['scDblFinder.score'] > thr['scDblFinder_score'],
        'low_n_genes': adata.obs['n_genes_by_counts'] < thr['n_genes_by_counts'],
        'low_counts': adata.obs['total_counts'] < thr['total_counts'],
        'small_cell': adata.obs['cell_area'] < thr['cell_area_min'],
    })
    adata = apply_qc_filters(adata, lambda cells: (
        (cells['scDblFinder.score'] <= thr['scDblFinder_score']) &
        (cells['cell_area'] >= thr['cell_area_min']) &
        (cells['cell_area'] <= 3 * np.median(cells['cell_area'])) &
        (cells['n_genes_by_counts'] >= thr['n_genes_by_counts']) &
        (cells['total_counts'] >= thr['total_counts'])),
        group_col='sample_rep')

    adata.layers['counts'] = adata.X.copy()
    adata.write(f'{output_dir}/01_adata_query_xenium.h5ad')

'''
high_doublet: 282232 (24.1%)
low_n_genes: 14989 (1.3%)
low_counts: 12023 (1.0%)
small_cell: 4274 (0.4%)
[CTRL_2_1] 80,487 / 116,354 cells kept (69%)
[CTRL_2_2] 87,458 / 119,829 cells kept (73%)
[CTRL_3_1] 91,709 / 121,025 cells kept (76%)
[CTRL_3_2] 87,272 / 115,711 cells kept (75%)
[PREG_2_1] 110,236 / 140,212 cells kept (79%)
[PREG_2_2] 96,181 / 133,539 cells kept (72%)
[PREG_3_1] 97,228 / 132,527 cells kept (73%)
[PREG_3_2] 108,076 / 143,472 cells kept (75%)
[CTRL_1_1] 109,057 / 147,588 cells kept (74%)

total kept: 867,704 / 1,170,257 (74.1%), dropped: 302,553 (25.9%)
'''

#endregion

#region run ####################################################################

runners = {'merfish': run_merfish, 'slidetags': run_slidetags,
           'xenium': run_xenium}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <dataset>')
        print(f'Datasets: {", ".join(runners.keys())}')
        sys.exit(1)
    t = sys.argv[1]
    if t not in runners:
        print(f'Unknown dataset: {t}. Choose from {list(runners.keys())}')
        sys.exit(1)
    runners[t]()

#endregion
