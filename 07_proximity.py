#region imports and setup ######################################################

import os
import gc
import re
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.sparse import coo_array
from scipy.spatial import KDTree
from statsmodels.stats.multitest import fdrcorrection
from tqdm.auto import tqdm
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r, to_py

warnings.filterwarnings('ignore')

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
stats_coords = ('x_affine', 'y_affine')
viz_coords = ('x_ffd', 'y_ffd')
D_MAX_SCALE = 20
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
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
        'local_proximity': True,
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

    pre_path = cfg['path'].replace('03_', '01_')
    pre = sc.read_h5ad(pre_path, backed='r')
    affine = torch.load(
        f'{working_dir}/output/{name}/coords_affine.pt',
        weights_only=False)
    sample_col = 'sample_rep' if name == 'xenium' else 'sample'
    x_aff = np.empty(pre.n_obs)
    y_aff = np.empty(pre.n_obs)
    for key, coords_arr in affine.items():
        mask = pre.obs[sample_col] == key
        pos = np.where(mask)[0]
        x_aff[pos] = coords_arr[:len(pos), 0]
        y_aff[pos] = coords_arr[:len(pos), 1]
    idx = pre.obs.index.get_indexer(adata.obs.index)
    adata.obs['x_affine'] = x_aff[idx]
    adata.obs['y_affine'] = y_aff[idx]
    del pre, affine

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
                sub, stats_coords, D_MAX_SCALE)
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

selected_full_b = sorted({
    ct for ct in global_tt['cell_type'].unique()
    if any(ct.endswith(s) for s in selected_b_suffixes)
})

condition_order = ['CTRL', 'PREG', 'POSTPART']

n_cts = len(selected_full_b)
n_cols = 3
n_rows = int(np.ceil(n_cts / n_cols))
fig, axes = plt.subplots(
    n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 1.9))
axes = axes.flatten()

for idx, ct in enumerate(selected_full_b):
    ax = axes[idx]
    for ds_name in datasets:
        sub = global_props[
            (global_props['cell_type'] == ct) &
            (global_props['dataset'] == ds_name)]
        if sub.empty:
            continue
        agg = sub.groupby('condition')['normalized_prop'].agg(['mean', 'sem'])
        ordered = [c for c in condition_order if c in agg.index]
        if not ordered:
            continue
        means = agg.loc[ordered, 'mean'].values.astype(float)
        sems = agg.loc[ordered, 'sem'].fillna(0).values.astype(float)
        if len(means) > 1 and means.std() > 0:
            mu, sd = means.mean(), means.std()
            means = (means - mu) / sd
            sems = sems / sd
        xs = [condition_order.index(c) for c in ordered]
        ax.errorbar(
            xs, means, yerr=sems, fmt='o-',
            color=dataset_colors[ds_name],
            label=ds_name, capsize=2.5, markersize=4,
            linewidth=1.2, elinewidth=0.8, alpha=0.85)

        sig_rows = global_tt[
            (global_tt['cell_type'] == ct) &
            (global_tt['dataset'] == ds_name)]
        for j in range(len(ordered) - 1):
            c1, c2 = ordered[j], ordered[j + 1]
            row = sig_rows[sig_rows['contrast'] == f'{c2}_vs_{c1}']
            if row.empty:
                continue
            p = float(row['P.Value'].iloc[0])
            adj = float(row['adj.P.Val'].iloc[0])
            mark = '*' if adj < FDR_THRESHOLD else \
                ('•' if p < NOMINAL_THRESHOLD else '')
            if mark:
                y = max(means[j], means[j + 1]) + \
                    max(sems[j], sems[j + 1]) + 0.18
                ax.text((xs[j] + xs[j + 1]) / 2, y, mark,
                        ha='center', va='bottom', fontsize=9,
                        color=dataset_colors[ds_name], fontweight='bold')

    ct_label = re.sub(r'^\d+\s+', '', ct)
    ax.set_title(ct_label, fontsize=8)
    ax.set_xticks(range(len(condition_order)))
    ax.set_xlim(-0.5, len(condition_order) - 0.5)
    if idx >= n_cts - n_cols:
        ax.set_xticklabels(['N', 'P', 'PP'], fontsize=7)
    else:
        ax.set_xticklabels([])
    ax.tick_params(labelsize=7, length=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

for i in range(n_cts, len(axes)):
    axes[i].set_visible(False)

fig.text(0.02, 0.5, 'Normalized Proportion (z)', va='center',
         ha='center', rotation='vertical', fontsize=8)

legend_elements = [
    Line2D([0], [0], color=dataset_colors[d], marker='o', markersize=4,
           linewidth=1.2, label=d) for d in datasets
]
fig.legend(handles=legend_elements, loc='upper right',
           bbox_to_anchor=(0.99, 0.99), fontsize=7,
           frameon=False, ncol=3)

plt.tight_layout(rect=[0.04, 0, 1, 0.96])
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
    figsize=(3.2 * (n_exemplars + 1), 3.2 * len(proximity_datasets)),
    squeeze=False)
rng = np.random.default_rng(42)

for row, (name, cfg) in enumerate(proximity_datasets.items()):
    adata = adatas[name]
    sample = sorted(adata.obs['sample'].unique())[0]
    sub = adata.obs[adata.obs['sample'] == sample]
    coords = sub[list(stats_coords)].to_numpy(dtype=np.float64)
    tree = KDTree(coords)
    d_scale = np.median(tree.query(coords, k=2)[0][:, 1])
    d_max = D_MAX_SCALE * d_scale

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
                 f'(scale={D_MAX_SCALE})',
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

contrast_titles = {
    'PREG_vs_CTRL': 'Pregnant vs\nNulliparous',
    'POSTPART_vs_PREG': 'Postpartum vs\nPregnant',
    'POSTPART_vs_CTRL': 'Postpartum vs\nNulliparous',
}

DISPLAY_CAP = 30
type_order = ['Glut', 'Gaba', 'NN']
sig_mask = local_tt['P.Value'] < NOMINAL_THRESHOLD

display_a = set(local_tt.loc[sig_mask, 'cell_type_a'].unique())
if len(display_a) > DISPLAY_CAP:
    counts_a = local_tt.loc[sig_mask].groupby('cell_type_a')\
        .size().sort_values(ascending=False)
    display_a = set(counts_a.head(DISPLAY_CAP).index)
display_a = sorted(display_a,
                   key=lambda c: (type_order.index(get_type(c)), c))

display_b = set(local_tt.loc[sig_mask, 'cell_type_b'].unique())
if len(display_b) > DISPLAY_CAP:
    counts_b = local_tt.loc[sig_mask].groupby('cell_type_b')\
        .size().sort_values(ascending=False)
    display_b = set(counts_b.head(DISPLAY_CAP).index)
display_b = sorted(display_b,
                   key=lambda c: (type_order.index(get_type(c)), c))

vmax = float(local_tt['logFC'].abs().quantile(0.98))
vmax = max(vmax, 0.1)
vmin = -vmax

contrasts_all = ['PREG_vs_CTRL', 'POSTPART_vs_PREG', 'POSTPART_vs_CTRL']
ds_list = list(proximity_datasets.keys())

panel_h = max(0.16 * len(display_a), 4)
panel_w = max(0.30 * len(display_b), 1.6)
fig = plt.figure(
    figsize=(panel_w * len(contrasts_all) + 2,
             panel_h * len(ds_list) + 1))
outer = gridspec.GridSpec(
    len(ds_list), len(contrasts_all), figure=fig,
    hspace=0.15, wspace=0.06)

last_im = None
for di, ds_name in enumerate(ds_list):
    for ci, contrast in enumerate(contrasts_all):
        ax = fig.add_subplot(outer[di, ci])
        sub = local_tt[
            (local_tt['dataset'] == ds_name) &
            (local_tt['contrast'] == contrast)]
        if sub.empty:
            ax.axis('off')
            continue

        mat = pd.DataFrame(np.nan, index=display_a, columns=display_b)
        sigs = pd.DataFrame('', index=display_a, columns=display_b)
        for _, row in sub.iterrows():
            a, b = row['cell_type_a'], row['cell_type_b']
            if a not in mat.index or b not in mat.columns:
                continue
            mat.loc[a, b] = row['logFC']
            if row['adj.P.Val'] < FDR_THRESHOLD:
                sigs.loc[a, b] = '*'
            elif row['P.Value'] < NOMINAL_THRESHOLD:
                sigs.loc[a, b] = '•'

        im = ax.pcolormesh(
            mat.values, cmap='PRGn', vmin=vmin, vmax=vmax,
            edgecolors='lightgray', linewidth=0.2)
        last_im = im
        for i in range(len(display_a)):
            for j in range(len(display_b)):
                if sigs.iat[i, j]:
                    ax.text(j + 0.5, i + 0.5, sigs.iat[i, j],
                            ha='center', va='center',
                            color='white', fontsize=6, fontweight='bold')

        ax.set_xlim(0, len(display_b))
        ax.set_ylim(len(display_a), 0)
        ax.set_xticks(np.arange(len(display_b)) + 0.5)
        ax.set_yticks(np.arange(len(display_a)) + 0.5)
        ax.set_xticklabels(
            [re.sub(r'^\d+\s+', '', b) for b in display_b],
            rotation=45, ha='right', fontsize=5.5)
        if ci == 0:
            ax.set_yticklabels(
                [re.sub(r'^\d+\s+', '', a) for a in display_a],
                fontsize=5.5)
            ax.set_ylabel(ds_name, fontsize=10, labelpad=18)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        if di == 0:
            ax.set_title(contrast_titles[contrast], fontsize=9, pad=6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

if last_im is not None:
    cbar_ax = fig.add_axes([0.92, 0.4, 0.012, 0.2])
    cbar = fig.colorbar(last_im, cax=cbar_ax, label='logFC')
    cbar.ax.tick_params(labelsize=7)

plt.savefig(
    f'{working_dir}/figures/proximity_local_heatmap.png',
    dpi=300, bbox_inches='tight')
plt.savefig(
    f'{working_dir}/figures/proximity_local_heatmap.svg',
    bbox_inches='tight')
plt.close()

#endregion
