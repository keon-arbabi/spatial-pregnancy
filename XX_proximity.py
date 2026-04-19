#region imports and setup ######################################################

import os
import gc
import re
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.sparse import coo_array
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r, to_py

warnings.filterwarnings('ignore')

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
stats_coords = ('x_affine', 'y_affine')
viz_coords = ('x_ffd', 'y_ffd')
MIN_NONZERO = 5
FDR_THRESHOLD = 0.10
NOMINAL_THRESHOLD = 0.05

datasets = {
    'slidetags': {
        'path': f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
        'local_proximity': False,
    },
    'merfish': {
        'path': f'{working_dir}/output/merfish/03_adata_query_merfish.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
        'local_proximity': True,
        'd_max_scale': 20,
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
        'local_proximity': True,
        'd_max_scale': 20,
    },
}

selected_b_suffixes = [
    'Astro-NT NN', 'Astro-TE NN', 'Endo NN', 'Ependymal NN',
    'Microglia NN', 'Oligo NN', 'OPC NN', 'Peri NN', 'VLMC NN',
]

dataset_colors = {
    'slidetags': '#3a86ff',
    'merfish': '#4361ee',
    'xenium': '#4cc9f0',
}

def get_type(ct):
    if 'Glut' in ct:
        return 'Glut'
    if any(x in ct for x in ['Gaba', 'IMN', 'Chol']):
        return 'Gaba'
    return 'NN'

def compute_spatial_stats(obs_df, coords_cols, d_max_scale, b_filter=None):
    coords = obs_df[list(coords_cols)].to_numpy(dtype=np.float64)
    tree = KDTree(coords)
    d_scale = np.median(tree.query(coords, k=2)[0][:, 1])
    d_max = d_max_scale * d_scale

    pair_arr = tree.query_pairs(d_max, output_type='ndarray')
    if len(pair_arr) == 0:
        return pd.DataFrame()
    arr = np.concatenate([pair_arr, pair_arr[:, ::-1]])
    n = len(coords)
    mat = coo_array(
        (np.ones(len(arr), dtype=np.int8), arr.T),
        shape=(n, n)).tocsr()

    cell_types = obs_df[cell_type_col].to_numpy()
    all_count = np.asarray(mat.sum(axis=1)).flatten().astype(np.int32)

    unique_b = obs_df[cell_type_col].unique().tolist()
    if b_filter is not None:
        unique_b = [b for b in unique_b if b in b_filter]

    cell_ids = obs_df.index.to_numpy()
    sample_id = obs_df['sample'].iloc[0]
    condition = obs_df['condition'].iloc[0]

    blocks = []
    for ctb in unique_b:
        b_mask = cell_types == ctb
        b_count = np.asarray(mat[:, b_mask].sum(axis=1))\
            .flatten().astype(np.int32)
        blocks.append(pd.DataFrame({
            'cell_id': cell_ids,
            'cell_type_a': cell_types,
            'cell_type_b': ctb,
            'b_count': b_count,
            'all_count': all_count,
        }))
    out = pd.concat(blocks, ignore_index=True)
    out['sample_id'] = sample_id
    out['condition'] = condition
    return out

def build_contrasts_block(contrasts):
    return ',\n        '.join(
        f"{t}_vs_{c} = 'condition{t} - condition{c}'"
        for t, c in contrasts)

adatas = {}
for name, cfg in datasets.items():
    adata = sc.read_h5ad(cfg['path'])
    drop = cfg.get('drop_samples', [])
    if drop:
        adata = adata[~adata.obs['sample'].isin(drop)].copy()
        print(f'[{name}] dropped samples: {drop}')
    adatas[name] = adata
    print(f'[{name}] {adata.shape[0]:,} cells, '
          f'{adata.obs[cell_type_col].nunique()} subclasses, '
          f'{adata.obs["sample"].nunique()} samples, '
          f'{adata.obs["condition"].nunique()} conditions')

#endregion

#region global proportions (crumblr + dream) ###################################

r('''
suppressPackageStartupMessages({
    library(crumblr)
    library(variancePartition)
})
''')

global_tt_frames = []
global_props_frames = []
for name, cfg in datasets.items():
    adata = adatas[name]
    counts = pd.crosstab(
        index=adata.obs['sample'], columns=adata.obs[cell_type_col])
    counts = counts.sort_index()

    meta = adata.obs[['sample', 'condition']].drop_duplicates()
    meta['sample'] = meta['sample'].astype(str)
    meta['condition'] = meta['condition'].astype(str)
    meta = meta.sort_values('sample').reset_index(drop=True)
    assert list(meta['sample']) == list(counts.index.astype(str))

    contrast_block = build_contrasts_block(cfg['contrasts'])

    counts_df = counts.reset_index()
    counts_df['sample'] = counts_df['sample'].astype(str)
    int_cols = counts_df.select_dtypes(include='int64').columns
    counts_df[int_cols] = counts_df[int_cols].astype(np.float64)
    to_r(counts_df, 'counts_df', format='data.frame')
    to_r(meta, 'meta_df', format='data.frame')

    r(f'''
    sample_names <- as.character(counts_df$sample)
    counts_df$sample <- NULL
    counts <- data.matrix(counts_df)
    rownames(counts) <- sample_names
    storage.mode(counts) <- "integer"

    meta <- as.data.frame(meta_df)
    rownames(meta) <- as.character(meta$sample)
    meta <- meta[sample_names, , drop = FALSE]
    meta$condition <- factor(meta$condition)

    cobj <- crumblr(counts)
    form <- ~ 0 + condition
    L <- makeContrastsDream(form, meta, contrasts = c(
        {contrast_block}
    ))
    fit <- dream(cobj, form, meta, L, useWeights = TRUE)
    fit <- eBayes(fit)

    results <- list()
    for (coef in colnames(L)) {{
        tt <- topTable(fit, coef = coef, number = Inf)
        tt$cell_type <- rownames(tt)
        tt$contrast <- coef
        results[[coef]] <- tt
    }}
    tt_all <- do.call(rbind, results)
    rownames(tt_all) <- NULL

    norm_props <- as.data.frame(cobj$E)
    norm_props$cell_type <- rownames(norm_props)
    rownames(norm_props) <- NULL
    ''')

    tt = to_py('tt_all', format='pandas')
    tt['dataset'] = name
    global_tt_frames.append(tt)

    props = to_py('norm_props', format='pandas')
    props_long = props.melt(
        id_vars=['cell_type'], var_name='sample',
        value_name='normalized_prop')
    props_long = props_long.merge(meta, on='sample', how='left')
    props_long['dataset'] = name
    global_props_frames.append(props_long)

    n_sig = int((tt['adj.P.Val'] < FDR_THRESHOLD).sum())
    print(f'[{name}] global props: {tt.shape[0]} tests, '
          f'{n_sig} sig (FDR<0.10)')

global_tt = pd.concat(global_tt_frames, ignore_index=True)
global_props = pd.concat(global_props_frames, ignore_index=True)

os.makedirs(f'{working_dir}/output', exist_ok=True)
global_tt.to_csv(
    f'{working_dir}/output/proximity_global_props.csv', index=False)
global_props.to_csv(
    f'{working_dir}/output/proximity_global_norm_props.csv', index=False)

#endregion

#region local proximities — spatial stats ######################################

proximity_datasets = {
    k: v for k, v in datasets.items() if v['local_proximity']}

spatial_stats_all = {}
recomputed_stats = {}
for name, cfg in proximity_datasets.items():
    cache_path = f'{working_dir}/output/{name}/spatial_stats.pkl'
    expected_n_b = adatas[name].obs[cell_type_col].nunique()

    spatial_stats = None
    if os.path.exists(cache_path):
        spatial_stats = pd.read_pickle(cache_path)
        cached_n_b = spatial_stats['cell_type_b'].nunique()
        if cached_n_b < expected_n_b:
            print(f'[{name}] cached stats incomplete '
                  f'({cached_n_b}/{expected_n_b} b types), recomputing')
            spatial_stats = None
        else:
            print(f'[{name}] loaded cached spatial stats: '
                  f'{spatial_stats.shape[0]:,} rows')

    if spatial_stats is None:
        adata = adatas[name]
        per_sample = []
        groups = adata.obs.groupby('sample', sort=False)
        for sample, sub in tqdm(
                groups, total=len(groups),
                desc=f'[{name}] spatial stats'):
            stats = compute_spatial_stats(
                sub, stats_coords, cfg['d_max_scale'])
            if not stats.empty:
                per_sample.append(stats)
        spatial_stats = pd.concat(per_sample, ignore_index=True)
        for col in ['cell_type_a', 'cell_type_b', 'sample_id', 'condition']:
            spatial_stats[col] = spatial_stats[col].astype('category')

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        spatial_stats.to_pickle(cache_path)
        print(f'[{name}] computed spatial stats: '
              f'{spatial_stats.shape[0]:,} rows, saved to {cache_path}')
        recomputed_stats[name] = True
    else:
        recomputed_stats[name] = False

    spatial_stats_all[name] = spatial_stats

gc.collect()

#endregion

#region local proximities — differential test (crumblr + dream) ################

r('''
suppressPackageStartupMessages(library(BiocParallel))
bp <- SnowParam(
    workers = min(8L, parallel::detectCores()),
    type = 'SOCK', progressbar = FALSE)
''')

local_tt_frames = []
for name, cfg in proximity_datasets.items():
    diff_path = f'{working_dir}/output/{name}/spatial_diff.pkl'
    if os.path.exists(diff_path) and not recomputed_stats[name]:
        spatial_diff = pd.read_pickle(diff_path)
        spatial_diff['dataset'] = name
        local_tt_frames.append(spatial_diff)
        print(f'[{name}] loaded cached spatial diff: '
              f'{spatial_diff.shape[0]:,} rows')
        continue

    spatial_stats = spatial_stats_all[name]

    pair_filter = (
        spatial_stats
        .groupby(['sample_id', 'cell_type_a', 'cell_type_b'], observed=True)
        ['b_count'].apply(lambda x: int((x > 0).sum()))
        .groupby(['cell_type_a', 'cell_type_b'])
        .min()
    )
    valid_pairs = pair_filter[pair_filter >= MIN_NONZERO].index.tolist()
    valid_b_by_a = {}
    for a, b in valid_pairs:
        valid_b_by_a.setdefault(a, []).append(b)
    print(f'[{name}] testing {len(valid_pairs)} pairs '
          f'across {len(valid_b_by_a)} center types')

    contrast_block = build_contrasts_block(cfg['contrasts'])

    groups_by_a = spatial_stats.groupby(
        'cell_type_a', observed=True, sort=False)

    pair_results = []
    for ct_a in tqdm(sorted(valid_b_by_a.keys()),
                     desc=f'[{name}] dream batched per a'):
        if ct_a not in groups_by_a.groups:
            continue
        sub_a = groups_by_a.get_group(ct_a)
        valid_b = valid_b_by_a[ct_a]

        b_pivot = sub_a.pivot_table(
            index='cell_id', columns='cell_type_b', values='b_count',
            fill_value=0, observed=True)
        b_pivot.columns = b_pivot.columns.astype(str)
        avail_b = [str(b) for b in valid_b if str(b) in b_pivot.columns]
        if not avail_b:
            continue
        b_pivot = b_pivot[avail_b]

        cell_meta = sub_a.drop_duplicates('cell_id').set_index('cell_id')
        cell_meta = cell_meta.loc[b_pivot.index]
        all_count_arr = cell_meta['all_count'].astype(np.float64).values

        b_arr = b_pivot.values.astype(np.float64)
        all_arr = all_count_arr[:, None]
        other_arr = all_arr - b_arr
        clr_matrix = (
            0.5 * (np.log(b_arr + 0.5) - np.log(other_arr + 0.5))
        ).T

        n_total = all_arr + 1.0
        p = (b_arr + 0.5) / n_total
        raw_w = (4.0 * n_total * p * (1.0 - p)).T
        for i in range(raw_w.shape[0]):
            q05 = np.quantile(raw_w[i], 0.05)
            if q05 > 0:
                raw_w[i] /= q05
            raw_w[i] = np.minimum(raw_w[i], 5.0)
        weights_matrix = raw_w

        meta_pair = pd.DataFrame({
            'condition': cell_meta['condition'].astype(str).values,
            'sample_id': cell_meta['sample_id'].astype(str).values,
        })

        required = {c for pair in cfg['contrasts'] for c in pair}
        if not required.issubset(meta_pair['condition'].unique()):
            continue

        try:
            to_r(clr_matrix, 'clr_matrix')
            to_r(weights_matrix, 'weights_matrix')
            to_r(list(avail_b), 'feature_names')
            to_r(meta_pair, 'meta', format='data.frame')
            r(f'''
            sample_ids <- paste0('c', seq_len(ncol(clr_matrix)))
            colnames(clr_matrix) <- sample_ids
            rownames(clr_matrix) <- feature_names
            colnames(weights_matrix) <- sample_ids
            rownames(weights_matrix) <- feature_names
            rownames(meta) <- sample_ids

            form <- ~ 0 + condition + (1 | sample_id)
            L <- makeContrastsDream(form, meta, contrasts = c(
                {contrast_block}
            ))
            fit <- dream(clr_matrix, form, meta, L,
                         weightsMatrix = weights_matrix, BPPARAM = bp)
            fit <- eBayes(fit)

            tt_list <- list()
            for (coef in colnames(L)) {{
                tt <- topTable(fit, coef = coef, number = Inf)
                tt$contrast <- coef
                tt$cell_type_b <- rownames(tt)
                tt_list[[coef]] <- tt
            }}
            tt_a <- do.call(rbind, tt_list)
            rownames(tt_a) <- NULL
            ''')
            df = to_py('tt_a', format='pandas')
            if df is not None and len(df) > 0:
                df['cell_type_a'] = ct_a
                pair_results.append(df)
        except Exception as e:
            print(f'[{name}] {ct_a}: {type(e).__name__}: {e}')

    if not pair_results:
        continue
    spatial_diff = pd.concat(pair_results, ignore_index=True)

    fdr_frames = []
    for contrast, group in spatial_diff.groupby('contrast'):
        group = group.copy()
        group['adj.P.Val'] = fdrcorrection(group['P.Value'])[1]
        fdr_frames.append(group)
    spatial_diff = pd.concat(fdr_frames, ignore_index=True)

    spatial_diff.to_pickle(diff_path)
    spatial_diff['dataset'] = name
    local_tt_frames.append(spatial_diff)

    n_sig = int((spatial_diff['adj.P.Val'] < FDR_THRESHOLD).sum())
    print(f'[{name}] local proximities: {spatial_diff.shape[0]} tests, '
          f'{n_sig} sig (FDR<0.10)')

local_tt = pd.concat(local_tt_frames, ignore_index=True)
local_tt.to_csv(
    f'{working_dir}/output/proximity_local_diff.csv', index=False)

#endregion

#region plot — global proportions #############################################

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

global_tt = pd.read_csv(f'{working_dir}/output/proximity_global_props.csv')

type_order = ['Glut', 'Gaba', 'NN']
all_cts = sorted(
    global_tt['cell_type'].unique(),
    key=lambda c: (type_order.index(get_type(c)), c))
ct_labels = list(all_cts)

contrasts_all = ['PREG_vs_CTRL', 'POSTPART_vs_PREG', 'POSTPART_vs_CTRL']
contrast_titles = {
    'PREG_vs_CTRL': 'Pregnant vs\nNulliparous',
    'POSTPART_vs_PREG': 'Postpartum vs\nPregnant',
    'POSTPART_vs_CTRL': 'Postpartum vs\nNulliparous',
}
ds_list = list(datasets.keys())
n_ct = len(all_cts)

CELL_SIZE = 0.12
fig_w = CELL_SIZE * len(contrasts_all) * len(ds_list) + 2.5
fig_h = CELL_SIZE * n_ct + 1.2

fig, axes = plt.subplots(
    1, len(contrasts_all),
    figsize=(fig_w, fig_h), squeeze=False,
    gridspec_kw={'wspace': 0.02})

vmax = float(global_tt['logFC'].abs().quantile(0.95))
vmax = max(vmax, 0.5)

for ci, contrast in enumerate(contrasts_all):
    ax = axes[0, ci]
    sub = global_tt[global_tt['contrast'] == contrast]
    ds_in_contrast = [d for d in ds_list
                      if not sub[sub['dataset'] == d].empty]

    mat = pd.DataFrame(np.nan,
                       index=all_cts, columns=ds_in_contrast)
    sigs = pd.DataFrame('',
                        index=all_cts, columns=ds_in_contrast)
    for _, row in sub.iterrows():
        ct, ds = row['cell_type'], row['dataset']
        if ct in mat.index and ds in mat.columns:
            mat.loc[ct, ds] = row['logFC']
            if row['adj.P.Val'] < FDR_THRESHOLD:
                sigs.loc[ct, ds] = '*'
            elif row['P.Value'] < NOMINAL_THRESHOLD:
                sigs.loc[ct, ds] = '•'

    im = ax.pcolormesh(
        mat.values, cmap='PRGn', vmin=-vmax, vmax=vmax,
        edgecolors='lightgray', linewidth=0.3)
    for i in range(n_ct):
        for j in range(len(ds_in_contrast)):
            if sigs.iat[i, j]:
                ax.text(j + 0.5, i + 0.5, sigs.iat[i, j],
                        ha='center', va='center',
                        color='white', fontsize=5, fontweight='bold')

    ax.set_xlim(0, len(ds_in_contrast))
    ax.set_ylim(n_ct, 0)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(len(ds_in_contrast)) + 0.5)
    ax.set_xticklabels(ds_in_contrast, fontsize=7,
                       rotation=90, ha='center')
    ax.set_yticks(np.arange(n_ct) + 0.5)
    if ci == 0:
        ax.set_yticklabels(ct_labels, fontsize=4)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)

cbar = fig.colorbar(im, ax=axes, shrink=0.12, pad=0.015,
                    aspect=12, label='logFC')
cbar.ax.tick_params(labelsize=5)
cbar.set_label('logFC', fontsize=7)

os.makedirs(f'{working_dir}/figures', exist_ok=True)
plt.savefig(
    f'{working_dir}/figures/proximity_global_props.png',
    dpi=300, bbox_inches='tight')
plt.savefig(
    f'{working_dir}/figures/proximity_global_props.svg',
    bbox_inches='tight')
plt.close()

#endregion

#region plot — radius visualization ############################################

n_exemplars = 5
fig, axes = plt.subplots(
    len(proximity_datasets), n_exemplars + 1,
    figsize=(3 * (n_exemplars + 1), 2.8 * len(proximity_datasets)),
    squeeze=False)
rng = np.random.default_rng(42)

for row, (name, cfg) in enumerate(proximity_datasets.items()):
    adata = adatas[name]
    sample = sorted(adata.obs['sample'].unique())[0]
    sub = adata.obs[adata.obs['sample'] == sample]
    coords = sub[list(stats_coords)].to_numpy(dtype=np.float64)
    tree = KDTree(coords)
    d_scale = np.median(tree.query(coords, k=2)[0][:, 1])
    d_max = cfg['d_max_scale'] * d_scale

    ax = axes[row, 0]
    ax.scatter(coords[:, 0], coords[:, 1], s=0.5, c='lightgray',
               alpha=0.5, linewidth=0, rasterized=True)
    exemplar_idx = rng.integers(len(coords), size=n_exemplars)
    exemplars = coords[exemplar_idx]
    ax.scatter(exemplars[:, 0], exemplars[:, 1], s=20, c='red', zorder=5)
    for cell in exemplars:
        ax.add_patch(Circle(cell, d_max, fill=False, color='red',
                            linewidth=1, linestyle='--'))
    ax.set_aspect('equal')
    ax.set_title(f'{name} — {sample}\n'
                 f'd_scale={d_scale:.4f}, d_max={d_max:.4f} '
                 f'(scale={cfg["d_max_scale"]})',
                 fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for col, (idx, cell) in enumerate(zip(exemplar_idx, exemplars), start=1):
        neighbor_idx = tree.query_ball_point(cell, r=d_max)
        neighbor_idx = [i for i in neighbor_idx if i != idx]
        zoom = d_max * 2.5
        ax = axes[row, col]
        in_zoom = ((np.abs(coords[:, 0] - cell[0]) < zoom * 1.5) &
                   (np.abs(coords[:, 1] - cell[1]) < zoom * 1.5))
        ax.scatter(coords[in_zoom, 0], coords[in_zoom, 1],
                   s=6, c='lightgray', alpha=0.7, linewidth=0,
                   rasterized=True)
        if neighbor_idx:
            ax.scatter(coords[neighbor_idx, 0], coords[neighbor_idx, 1],
                       s=10, c='steelblue', alpha=0.85, linewidth=0,
                       rasterized=True)
        ax.scatter(cell[0], cell[1], s=80, c='red', zorder=5,
                   edgecolors='white', linewidths=1)
        ax.add_patch(Circle(cell, d_max, fill=False, color='red',
                            linewidth=1.5, linestyle='--'))
        ax.set_xlim(cell[0] - zoom, cell[0] + zoom)
        ax.set_ylim(cell[1] - zoom, cell[1] + zoom)
        ax.set_aspect('equal')
        ax.set_title(f'{len(neighbor_idx):,} within d_max', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

plt.tight_layout()
plt.savefig(
    f'{working_dir}/figures/proximity_radius.png',
    dpi=200, bbox_inches='tight')
plt.close()

#endregion

#region plot — local proximities heatmap ######################################

#region plot — local proximities heatmap ######################################

local_tt = pd.read_csv(f'{working_dir}/output/proximity_local_diff.csv')

DISPLAY_CAP = 30
type_order = ['Glut', 'Gaba', 'NN']
sig_mask = local_tt['P.Value'] < NOMINAL_THRESHOLD
exclude_cts = {'060 OT D3 Folh1 Gaba'}

def _select_display(col):
    counts = local_tt.loc[sig_mask, col].value_counts()
    counts = counts[~counts.index.isin(exclude_cts)]
    top = counts.head(DISPLAY_CAP).index.tolist()
    return sorted(top,
                  key=lambda c: (type_order.index(get_type(c)), c))

display_a = _select_display('cell_type_a')
display_b = _select_display('cell_type_b')

def _insert_gaps(items):
    gapped, labels, prev = [], [], None
    for item in items:
        t = get_type(item)
        if prev is not None and t != prev:
            gapped.append(None)
            labels.append('')
        gapped.append(item)
        labels.append(item)
        prev = t
    return gapped, labels

a_gapped, a_labels = _insert_gaps(display_a)
b_gapped, b_labels = _insert_gaps(display_b)

vmax = float(local_tt['logFC'].abs().quantile(0.98))
vmax = max(vmax, 0.1)
vmin = -vmax

global_tt = pd.read_csv(f'{working_dir}/output/proximity_global_props.csv')

panels = [
    ('xenium', 'PREG_vs_CTRL', 'Xenium Pregnant vs Nulliparous'),
    ('merfish', 'PREG_vs_CTRL', 'MERFISH Pregnant vs Nulliparous'),
    ('merfish', 'POSTPART_vs_PREG', 'MERFISH Postpartum vs Pregnant'),
]

CELL_SIZE = 0.11
GAP_WIDTH = 1
n_a_g, n_b_g = len(a_gapped), len(b_gapped)
n_cols_with_glob = 1 + GAP_WIDTH + n_b_g

row_gaps = [i for i, x in enumerate(a_gapped) if x is None]
col_gaps_local = [j for j, x in enumerate(b_gapped) if x is None]

row_segs = []
start = 0
for rg in row_gaps:
    row_segs.append((start, rg))
    start = rg + 1
row_segs.append((start, n_a_g))

col_segs_local = []
start = 0
for cg in col_gaps_local:
    col_segs_local.append((start, cg))
    start = cg + 1
col_segs_local.append((start, n_b_g))

def _draw_block_borders(ax, row_segs, col_segs, lw=0.8):
    for r0, r1 in row_segs:
        for c0, c1 in col_segs:
            ax.add_patch(plt.Rectangle(
                (c0, r0), c1 - c0, r1 - r0, fill=False,
                edgecolor='black', linewidth=lw, zorder=5))

ai_map = {ct: i for i, ct in enumerate(a_gapped) if ct is not None}
bi_map = {ct: j for j, ct in enumerate(b_gapped) if ct is not None}
glob_col = 0
local_offset = 1 + GAP_WIDTH

panel_w = CELL_SIZE * n_cols_with_glob
panel_h = CELL_SIZE * n_a_g
fig_w = panel_w * len(panels) + 3.0
fig_h = panel_h + 1.5

fig, axes = plt.subplots(
    1, len(panels), figsize=(fig_w, fig_h), squeeze=False,
    gridspec_kw={'wspace': 0.25})

last_im = None
for pi, (ds_name, contrast, title) in enumerate(panels):
    ax = axes[0, pi]

    full_mat = np.full((n_a_g, n_cols_with_glob), np.nan)
    full_sigs = np.full((n_a_g, n_cols_with_glob), '', dtype=object)

    glob_sub = global_tt[(global_tt['dataset'] == ds_name) &
                         (global_tt['contrast'] == contrast)]
    for _, row in glob_sub.iterrows():
        ct = row['cell_type']
        if ct in ai_map:
            full_mat[ai_map[ct], glob_col] = row['logFC']
            if row['adj.P.Val'] < FDR_THRESHOLD:
                full_sigs[ai_map[ct], glob_col] = '*'
            elif row['P.Value'] < NOMINAL_THRESHOLD:
                full_sigs[ai_map[ct], glob_col] = '•'

    local_sub = local_tt[
        (local_tt['dataset'] == ds_name) &
        (local_tt['contrast'] == contrast)]
    if not local_sub.empty:
        mat_raw = pd.DataFrame(np.nan, index=display_a, columns=display_b)
        sigs_raw = pd.DataFrame('', index=display_a, columns=display_b)
        for _, row in local_sub.iterrows():
            a, b = row['cell_type_a'], row['cell_type_b']
            if a not in mat_raw.index or b not in mat_raw.columns:
                continue
            mat_raw.loc[a, b] = row['logFC']
            if row['adj.P.Val'] < FDR_THRESHOLD:
                sigs_raw.loc[a, b] = '*'
            elif row['P.Value'] < NOMINAL_THRESHOLD:
                sigs_raw.loc[a, b] = '•'
        for ct_a in display_a:
            for ct_b in display_b:
                full_mat[ai_map[ct_a], local_offset + bi_map[ct_b]] = \
                    mat_raw.loc[ct_a, ct_b]
                full_sigs[ai_map[ct_a], local_offset + bi_map[ct_b]] = \
                    sigs_raw.loc[ct_a, ct_b]

    im = ax.pcolormesh(full_mat, cmap='PRGn', vmin=vmin, vmax=vmax)
    last_im = im

    for i in range(n_a_g):
        if a_gapped[i] is None:
            continue
        if not np.isnan(full_mat[i, glob_col]):
            ax.add_patch(plt.Rectangle(
                (glob_col, i), 1, 1, fill=False,
                edgecolor='black', linewidth=0.15))
        if full_sigs[i, glob_col]:
            ax.text(glob_col + 0.5, i + 0.5, full_sigs[i, glob_col],
                    ha='center', va='center',
                    color='white', fontsize=3.5, fontweight='bold')
        for j in range(n_b_g):
            if b_gapped[j] is None:
                continue
            jj = local_offset + j
            ax.add_patch(plt.Rectangle(
                (jj, i), 1, 1, fill=False,
                edgecolor='black', linewidth=0.15))
            if full_sigs[i, jj]:
                ax.text(jj + 0.5, i + 0.5, full_sigs[i, jj],
                        ha='center', va='center',
                        color='white', fontsize=3.5, fontweight='bold')

    _draw_block_borders(ax, row_segs, [(glob_col, glob_col + 1)])
    shifted_col_segs = [(local_offset + c0, local_offset + c1)
                        for c0, c1 in col_segs_local]
    _draw_block_borders(ax, row_segs, shifted_col_segs)

    ax.set_xlim(0, n_cols_with_glob)
    ax.set_ylim(n_a_g, 0)
    ax.set_aspect('equal')

    x_ticks = [glob_col + 0.5] + [local_offset + j + 0.5
                                    for j in range(n_b_g)]
    x_labels = ['Global\nproportions'] + b_labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=3.5)
    ax.set_yticks(np.arange(n_a_g) + 0.5)
    if pi == 0:
        ax.set_yticklabels(a_labels, fontsize=3.5)
        ax.set_ylabel('Center cell type', fontsize=6, labelpad=3)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=1)
    ax.set_title(title, fontsize=6, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

if last_im is not None:
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.25,
                        pad=0.015, aspect=12, label='logFC')
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('logFC', fontsize=6)

plt.savefig(
    f'{working_dir}/figures/proximity_local_heatmap.png',
    dpi=300, bbox_inches='tight')
plt.savefig(
    f'{working_dir}/figures/proximity_local_heatmap.svg',
    bbox_inches='tight')
plt.close()

#endregion
