#region imports and setup ######################################################

import os
import gc
import re
import time
import warnings
from collections import defaultdict
from math import comb, factorial
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
N_PERM_NULL = 100

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

gprops_path = f'{working_dir}/output/proximity/global_props.csv'
gnorm_path = f'{working_dir}/output/proximity/global_norm_props.csv'
os.makedirs(f'{working_dir}/output/proximity', exist_ok=True)

def build_contrasts_block(contrasts, indent=8):
    sep = ',\n' + ' ' * indent
    return sep.join(
        f"{t}_vs_{c} = 'condition{t} - condition{c}'"
        for t, c in contrasts)

if os.path.exists(gprops_path) and os.path.exists(gnorm_path):
    global_tt = pd.read_csv(gprops_path)
    global_props = pd.read_csv(gnorm_path)
    print(f'[global] cached: {global_tt.shape[0]:,} tests across '
          f'{global_tt["dataset"].nunique()} datasets, '
          f'{global_props.shape[0]:,} normalized-prop rows', flush=True)
else:
    r('''
    suppressPackageStartupMessages({
        library(crumblr)
        library(variancePartition)
    })
    ''')

    global_tt_frames = []
    global_props_frames = []
    for name, cfg in datasets.items():
        print(f'[{name}] global props: building crumblr+dream fit '
              f'({adatas[name].obs["sample"].nunique()} samples, '
              f'{adatas[name].obs[cell_type_col].nunique()} subclasses)',
              flush=True)
        t0 = time.time()
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
              f'{n_sig} sig (FDR<{FDR_THRESHOLD}), '
              f'{time.time() - t0:.0f}s', flush=True)

    global_tt = pd.concat(global_tt_frames, ignore_index=True)
    global_props = pd.concat(global_props_frames, ignore_index=True)

    os.makedirs(f'{working_dir}/output', exist_ok=True)
    global_tt.to_csv(gprops_path, index=False)
    global_props.to_csv(gnorm_path, index=False)
    print(f'[global] wrote proximity_global_props.csv '
          f'({global_tt.shape[0]:,} rows) and '
          f'proximity_global_norm_props.csv '
          f'({global_props.shape[0]:,} rows)', flush=True)

#endregion

#region local proximities — spatial stats ######################################

def compute_spatial_stats(obs_df, coords_cols, d_max_scale,
                          n_perm=N_PERM_NULL, b_filter=None, seed=0):
    sample_id = obs_df['sample'].iloc[0]
    condition = obs_df['condition'].iloc[0]
    n = len(obs_df)
    t0 = time.time()

    coords = obs_df[list(coords_cols)].to_numpy(dtype=np.float64)
    tree = KDTree(coords)
    d_scale = float(np.median(tree.query(coords, k=2)[0][:, 1]))
    d_max = d_max_scale * d_scale

    pair_arr = tree.query_pairs(d_max, output_type='ndarray')
    if len(pair_arr) == 0:
        print(f'[stats] {sample_id}: 0 pairs at d_max={d_max:.3f}, skip',
              flush=True)
        return pd.DataFrame()
    arr = np.concatenate([pair_arr, pair_arr[:, ::-1]])
    mat = coo_array(
        (np.ones(len(arr), dtype=np.int8), arr.T),
        shape=(n, n)).tocsr()

    cell_types = obs_df[cell_type_col].to_numpy()
    all_count = np.asarray(mat.sum(axis=1)).flatten().astype(np.int32)

    unique_b = obs_df[cell_type_col].unique().tolist()
    if b_filter is not None:
        unique_b = [b for b in unique_b if b in b_filter]
    n_b = len(unique_b)
    type_to_idx = {b: j for j, b in enumerate(unique_b)}
    col_ix = np.array([type_to_idx.get(c, -1) for c in cell_types])
    keep = col_ix >= 0
    L = coo_array(
        (np.ones(keep.sum(), dtype=np.int8),
         (np.where(keep)[0], col_ix[keep])),
        shape=(n, n_b)).tocsr()

    n_iso = int((all_count == 0).sum())
    print(f'[stats] {sample_id} ({condition}): '
          f'{n:,} cells, {n_b} subclasses, '
          f'd_scale={d_scale:.3f} d_max={d_max:.3f}, '
          f'{mat.nnz:,} edges, neighbors/cell '
          f'mean={all_count.mean():.1f} '
          f'(min={all_count.min()}, max={all_count.max()}, '
          f'isolated={n_iso}); setup {time.time() - t0:.1f}s',
          flush=True)

    t_perm = time.time()
    b_count = (mat @ L).toarray().astype(np.float64)

    rng = np.random.default_rng(seed)
    sum_b = np.zeros((n, n_b), dtype=np.float64)
    sum_b_sq = np.zeros((n, n_b), dtype=np.float64)
    for _ in tqdm(range(n_perm), desc=f'[stats] {sample_id} perms',
                  leave=False):
        perm = rng.permutation(n)
        bk = (mat @ L[perm, :]).toarray().astype(np.float64)
        sum_b += bk
        sum_b_sq += bk * bk
    mean_null = sum_b / n_perm
    var_null = np.maximum(
        sum_b_sq / n_perm - mean_null * mean_null, 0.0)
    sd_null = np.sqrt(var_null)
    z = np.where(sd_null > 1e-9,
                 (b_count - mean_null) / sd_null, np.nan).astype(np.float32)

    n_cells_total = n * n_b
    n_valid_z = int((~np.isnan(z)).sum())
    med_abs_z = (float(np.nanmedian(np.abs(z)))
                 if n_valid_z else float('nan'))
    q95_abs_z = (float(np.nanquantile(np.abs(z), 0.95))
                 if n_valid_z else float('nan'))
    print(f'[stats] {sample_id}: {n_perm} perms in '
          f'{time.time() - t_perm:.1f}s, '
          f'{n_valid_z:,}/{n_cells_total:,} valid z '
          f'({100 * n_valid_z / n_cells_total:.1f}%), '
          f'median |z|={med_abs_z:.2f}, '
          f'p95 |z|={q95_abs_z:.2f}', flush=True)

    cell_ids = obs_df.index.to_numpy()

    blocks = []
    for j, ctb in enumerate(unique_b):
        blocks.append(pd.DataFrame({
            'cell_id': cell_ids,
            'cell_type_a': cell_types,
            'cell_type_b': ctb,
            'b_count': b_count[:, j].astype(np.int32),
            'all_count': all_count,
            'z': z[:, j],
        }))
    out = pd.concat(blocks, ignore_index=True)
    out['sample_id'] = sample_id
    out['condition'] = condition
    return out

proximity_datasets = {
    k: v for k, v in datasets.items() if v['local_proximity']}

spatial_stats_all = {}
for name, cfg in proximity_datasets.items():
    cache_path = (
        f'{working_dir}/output/proximity/spatial_stats_null/{name}.pkl')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    expected_n_b = adatas[name].obs[cell_type_col].nunique()

    spatial_stats = None
    if os.path.exists(cache_path):
        spatial_stats = pd.read_pickle(cache_path)
        cached_n_b = spatial_stats['cell_type_b'].nunique()
        if 'z' not in spatial_stats.columns or cached_n_b < expected_n_b:
            print(f'[{name}] cached stats incomplete, recomputing')
            spatial_stats = None
        else:
            print(f'[{name}] loaded cached spatial stats: '
                  f'{spatial_stats.shape[0]:,} rows')

    if spatial_stats is None:
        adata = adatas[name]
        per_sample = []
        groups = adata.obs.groupby('sample', sort=False)
        for si, (sample, sub) in enumerate(tqdm(
                groups, total=len(groups),
                desc=f'[{name}] spatial stats')):
            stats = compute_spatial_stats(
                sub, stats_coords, cfg['d_max_scale'],
                n_perm=N_PERM_NULL, seed=12345 + si)
            if not stats.empty:
                per_sample.append(stats)
        spatial_stats = pd.concat(per_sample, ignore_index=True)
        for col in ['cell_type_a', 'cell_type_b', 'sample_id', 'condition']:
            spatial_stats[col] = spatial_stats[col].astype('category')

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        spatial_stats.to_pickle(cache_path)
        print(f'[{name}] computed spatial stats: '
              f'{spatial_stats.shape[0]:,} rows, saved to {cache_path}')

    spatial_stats_all[name] = spatial_stats

gc.collect()

#endregion

#region local proximities — pooled differential test (limma) ###################

r('''
suppressPackageStartupMessages({
    library(limma)
    library(parallel)
})
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
           MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1")
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(1)
    RhpcBLASctl::omp_set_num_threads(1)
}
''')

_pooled_agg_cache = {}

def get_pooled_agg(name, spatial_stats):
    if name in _pooled_agg_cache:
        return _pooled_agg_cache[name]
    t0 = time.time()
    # vectorized pair filter: count samples with b_count>0, require >=MIN
    per_sample_nz = (spatial_stats
                     .assign(_nz=(spatial_stats['b_count'] > 0)
                             .astype(np.int8))
                     .groupby(['sample_id', 'cell_type_a', 'cell_type_b'],
                              observed=True)['_nz'].sum())
    pf = per_sample_nz.groupby(['cell_type_a', 'cell_type_b']).min()
    valid = set(pf[pf >= MIN_NONZERO].index)

    clean = spatial_stats[spatial_stats['all_count'] > 0]
    agg = (clean
           .groupby(['sample_id', 'condition', 'cell_type_a',
                     'cell_type_b'], observed=True)
           .agg(z_mean=('z', 'mean'), n_cells=('z', 'size'))
           .reset_index())
    agg = agg.set_index(['cell_type_a', 'cell_type_b'])
    agg = agg[agg.index.isin(valid)].reset_index()
    agg['feature'] = (agg['cell_type_a'].astype(str) + '|' +
                      agg['cell_type_b'].astype(str))
    print(f'[{name}] aggregated sample z-scores: '
          f'{len(valid):,} valid pairs, {len(agg):,} rows '
          f'({time.time() - t0:.0f}s)', flush=True)
    _pooled_agg_cache[name] = agg
    return agg

def build_pooled_inputs(agg, contrast_conds):
    sub = agg[agg['condition'].astype(str).isin(list(contrast_conds))]
    sub = sub.dropna(subset=['z_mean'])
    if sub.empty:
        return None
    y_mat = sub.pivot(index='feature', columns='sample_id',
                      values='z_mean')
    w_mat = sub.pivot(index='feature', columns='sample_id',
                      values='n_cells').fillna(0.0)
    sample_meta = (sub[['sample_id', 'condition']].drop_duplicates()
                   .set_index('sample_id').loc[y_mat.columns]
                   .reset_index())
    sample_meta['condition'] = sample_meta['condition'].astype(str)
    return y_mat, w_mat, sample_meta

def push_pooled_to_r(y_mat, w_mat, sample_meta):
    to_r(y_mat.fillna(0.0).values, 'y')
    to_r(w_mat.values, 'w')
    to_r(list(y_mat.index), 'features')
    to_r(list(y_mat.columns), 'sample_ids')
    to_r(sample_meta, 'meta', format='data.frame')
    r('''
    rownames(y) <- features; colnames(y) <- sample_ids
    rownames(w) <- features; colnames(w) <- sample_ids
    rownames(meta) <- meta$sample_id
    meta$condition <- factor(meta$condition)
    design <- model.matrix(~ 0 + condition, data = meta)
    ''')

def pooled_contrast_fit(contrast_str):
    to_r(contrast_str, 'contrast_str')
    r('''
    L  <- makeContrasts(contrasts = contrast_str, levels = design)
    fit <- lmFit(y, design, weights = w)
    cf  <- contrasts.fit(fit, L)
    eb  <- eBayes(cf)
    tt  <- topTable(eb, coef = 1, number = Inf, sort.by = "none")
    tt$feature <- rownames(tt)
    ''')
    return to_py('tt', format='pandas')

diff_path = f'{working_dir}/output/proximity/local_diff.csv'

if os.path.exists(diff_path):
    local_tt = pd.read_csv(diff_path)
    print(f'[local] cached: {local_tt.shape[0]:,} pairs across '
          f'{local_tt["dataset"].nunique()} datasets, '
          f'{local_tt["contrast"].nunique()} contrasts', flush=True)
else:
    local_tt_frames = []
    for name, cfg in proximity_datasets.items():
        agg = get_pooled_agg(name, spatial_stats_all[name])
        per_contrast = []
        for treat, ctrl in cfg['contrasts']:
            contrast = f'{treat}_vs_{ctrl}'
            t0 = time.time()
            inputs = build_pooled_inputs(agg, (treat, ctrl))
            if inputs is None:
                print(f'[{name}] {contrast}: no valid pairs, skip',
                      flush=True)
                continue
            y_mat, w_mat, sample_meta = inputs
            if sample_meta['condition'].nunique() < 2:
                print(f'[{name}] {contrast}: <2 conditions present, skip',
                      flush=True)
                continue
            print(f'[{name}] {contrast}: pooled lmFit '
                  f'({y_mat.shape[0]:,} features x '
                  f'{y_mat.shape[1]} samples)', flush=True)
            push_pooled_to_r(y_mat, w_mat, sample_meta)
            tt = pooled_contrast_fit(f'condition{treat} - condition{ctrl}')
            if tt is None or len(tt) == 0:
                print(f'[{name}] {contrast}: empty topTable, skip',
                      flush=True)
                continue
            tt[['cell_type_a', 'cell_type_b']] = \
                tt['feature'].str.split('|', n=1, expand=True)
            tt['contrast'] = contrast
            per_contrast.append(tt[[
                'cell_type_a', 'cell_type_b', 'contrast', 'logFC',
                'AveExpr', 't', 'P.Value', 'adj.P.Val', 'B']])
            print(f'[{name}] {contrast}: fit done, {len(tt):,} pairs, '
                  f'{time.time() - t0:.0f}s', flush=True)

        if not per_contrast:
            continue
        spatial_diff = pd.concat(per_contrast, ignore_index=True)
        for contrast in spatial_diff['contrast'].unique():
            m = spatial_diff['contrast'] == contrast
            spatial_diff.loc[m, 'adj.P.Val'] = fdrcorrection(
                spatial_diff.loc[m, 'P.Value'].fillna(1.0))[1]
        spatial_diff['dataset'] = name
        local_tt_frames.append(spatial_diff)

        for contrast in spatial_diff['contrast'].unique():
            sub = spatial_diff[spatial_diff['contrast'] == contrast]
            n_sig = int((sub['adj.P.Val'] < FDR_THRESHOLD).sum())
            print(f'[{name}] {contrast}: {len(sub):,} pairs, '
                  f'{n_sig} sig (FDR<{FDR_THRESHOLD})', flush=True)

    local_tt = pd.concat(local_tt_frames, ignore_index=True)
    local_tt.to_csv(diff_path, index=False)
    print(f'[local] wrote proximity_local_diff.csv '
          f'({local_tt.shape[0]:,} rows)', flush=True)

#endregion

#region local proximities — sumrank meta-analysis ##############################

SUMRANK_N_PERM = 1000
SUMRANK_N_CORES = min(50, os.cpu_count() or 1)
SUMRANK_PLATFORMS = {'PREG_vs_CTRL': ['merfish', 'xenium']}
sumrank_cache_dir = f'{working_dir}/output/proximity/perms'
os.makedirs(sumrank_cache_dir, exist_ok=True)

def irwin_hall_cdf(x, n):
    x = np.clip(np.atleast_1d(x).astype(float), 0.0, float(n))
    out = np.zeros_like(x)
    for k in range(n + 1):
        diff = x - k
        m = diff > 0
        if m.any():
            out[m] += ((-1) ** k) * comb(n, k) * diff[m] ** n
    return out / factorial(n)

def signed_norm_rank(features, logfc, pval):
    pv = np.clip(pval.to_numpy().astype(float), 1e-300, 1.0)
    lf = np.nan_to_num(logfc.to_numpy().astype(float))
    score = np.nan_to_num(-np.log10(pv) * np.sign(lf))
    order = np.argsort(-score, kind='stable')
    rank = np.empty(len(score), dtype=np.int64)
    rank[order] = np.arange(1, len(score) + 1)
    return pd.DataFrame({
        'feature': features.values,
        'nrank': (rank - 1) / max(len(score) - 1, 1),
    })

def sumrank_one(de_frame, platforms):
    out = []
    for ct_a in de_frame['cell_type_a'].unique():
        rank_dfs, active = [], []
        for plt_name in platforms:
            sub = de_frame[(de_frame['dataset'] == plt_name) &
                           (de_frame['cell_type_a'] == ct_a)]
            if sub.empty:
                continue
            rank_dfs.append(signed_norm_rank(
                sub['cell_type_b'], sub['logFC'], sub['P.Value'])
                .rename(columns={'nrank': f'nrank_{plt_name}'}))
            active.append(plt_name)
        if len(rank_dfs) < 2:
            continue
        merged = rank_dfs[0]
        for rd in rank_dfs[1:]:
            merged = merged.merge(rd, on='feature', how='outer')
        arr = merged[[f'nrank_{p}' for p in active]].to_numpy()
        d = (~np.isnan(arr)).sum(axis=1)
        s = np.nansum(arr, axis=1)
        keep = d >= 2
        if not keep.any():
            continue
        nlp_up = np.full(len(d), np.nan)
        nlp_dn = np.full(len(d), np.nan)
        for k in np.unique(d[keep]):
            idx = (d == int(k)) & keep
            cdf = irwin_hall_cdf(s[idx], int(k))
            nlp_up[idx] = -np.log10(np.clip(cdf, 1e-300, 1.0))
            nlp_dn[idx] = -np.log10(np.clip(1 - cdf, 1e-300, 1.0))
        out.append(pd.DataFrame({
            'cell_type_a': ct_a,
            'cell_type_b': merged['feature'].values,
            'D': d.astype(np.int64),
            'sum_stat': s,
            'nlp_up': nlp_up,
            'nlp_down': nlp_dn,
        }).query('D >= 2'))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def run_permutations(name, contrast, treat_ctrl, n_perm):
    final_path = f'{sumrank_cache_dir}/perm_{name}_{contrast}.parquet'
    if os.path.exists(final_path):
        df = pd.read_parquet(final_path)
        print(f'[sumrank] perm cached: {name} {contrast} '
              f'({df["perm"].nunique()} perms)', flush=True)
        return df

    t_pop = time.time()
    agg = get_pooled_agg(name, spatial_stats_all[name])
    inputs = build_pooled_inputs(agg, treat_ctrl)
    if inputs is None:
        print(f'[sumrank] {name} {contrast}: no valid pairs, skip',
              flush=True)
        return pd.DataFrame()
    y_mat, w_mat, sample_meta = inputs
    if sample_meta['condition'].nunique() < 2:
        print(f'[sumrank] {name} {contrast}: <2 conditions, skip',
              flush=True)
        return pd.DataFrame()
    push_pooled_to_r(y_mat, w_mat, sample_meta)
    treat, ctrl = treat_ctrl
    to_r(n_perm, 'n_perm')
    to_r(SUMRANK_N_CORES, 'n_cores')
    to_r(f'condition{treat} - condition{ctrl}', 'contrast_str')
    print(f'[sumrank] {name} {contrast}: '
          f'{n_perm} perms on {SUMRANK_N_CORES} cores '
          f'({y_mat.shape[0]:,} features x {y_mat.shape[1]} samples, '
          f'inputs ready in {time.time() - t_pop:.0f}s)', flush=True)

    t0 = time.time()
    r('''
    worker <- function(k) {
        set.seed(12345L + k)
        meta_k <- meta
        meta_k$condition <- factor(
            sample(as.character(meta$condition)))
        design_k <- model.matrix(~ 0 + condition, data = meta_k)
        if (qr(design_k)$rank < 2) return(NULL)
        L_k <- tryCatch(
            makeContrasts(contrasts = contrast_str, levels = design_k),
            error = function(e) NULL)
        if (is.null(L_k)) return(NULL)
        fit <- lmFit(y, design_k, weights = w)
        cf  <- contrasts.fit(fit, L_k)
        eb  <- eBayes(cf)
        tt  <- topTable(eb, coef = 1, number = Inf, sort.by = "none")
        data.frame(feature = rownames(tt), logFC = tt$logFC,
                   P.Value = tt$P.Value, perm = k,
                   stringsAsFactors = FALSE)
    }
    perms <- do.call(rbind, parallel::mclapply(
        1:n_perm, worker, mc.cores = n_cores))
    ''')
    df = to_py('perms', format='pandas')
    dt = time.time() - t0
    if df is None or len(df) == 0:
        print(f'[sumrank] {name} {contrast}: 0 rows in {dt:.0f}s',
              flush=True)
        return pd.DataFrame()
    df[['cell_type_a', 'cell_type_b']] = \
        df['feature'].str.split('|', n=1, expand=True)
    df['dataset'] = name
    df.to_parquet(final_path, index=False)
    print(f'[sumrank] {name} {contrast}: saved {len(df):,} rows '
          f'({df["perm"].nunique()}/{n_perm} perms, '
          f'{dt / 60:.1f} min)', flush=True)
    return df

sumrank_path = f'{working_dir}/output/proximity/local_sumrank.csv'

if os.path.exists(sumrank_path):
    sumrank_out = pd.read_csv(sumrank_path)
    print(f'[sumrank] cached: {sumrank_out.shape[0]:,} pairs across '
          f'{sumrank_out["contrast"].nunique()} contrasts', flush=True)
    sumrank_frames = None
else:
    sumrank_frames = []

for contrast, platforms in (SUMRANK_PLATFORMS.items()
                            if sumrank_frames is not None else []):
    print(f'[sumrank] {contrast}: meta-analyzing across {platforms}',
          flush=True)
    de_c = local_tt[local_tt['contrast'] == contrast]
    real = sumrank_one(de_c, platforms)
    if real.empty:
        print(f'[sumrank] {contrast}: no pairs across >=2 platforms',
              flush=True)
        continue
    print(f'[sumrank] {contrast}: {len(real):,} real pairs across '
          f'{real["cell_type_a"].nunique()} center types', flush=True)

    treat, ctrl = contrast.split('_vs_')
    perms_by_plt = {
        p: run_permutations(p, contrast, (treat, ctrl), SUMRANK_N_PERM)
        for p in platforms}
    available = [p for p, pf in perms_by_plt.items() if len(pf) > 0]
    if len(available) < 2:
        print(f'[sumrank] {contrast}: <2 platforms with perms, skip',
              flush=True)
        continue

    max_k = int(min(perms_by_plt[p]['perm'].max() for p in available))
    print(f'[sumrank] {contrast} null: calibrating across {max_k} perms',
          flush=True)
    null_by_a = defaultdict(list)
    t_null = time.time()
    for k in range(1, max_k + 1):
        dfs = []
        for p in available:
            pk = perms_by_plt[p][perms_by_plt[p]['perm'] == k]
            if pk.empty:
                continue
            dfs.append(pk[['cell_type_a', 'cell_type_b', 'logFC',
                           'P.Value', 'dataset']])
        if len(dfs) < 2:
            continue
        sr_k = sumrank_one(pd.concat(dfs, ignore_index=True), platforms)
        if sr_k.empty:
            continue
        for ct_a, sub in sr_k.groupby('cell_type_a'):
            u = sub['nlp_up'].to_numpy()
            d = sub['nlp_down'].to_numpy()
            null_by_a[ct_a].append(u[~np.isnan(u)])
            null_by_a[ct_a].append(d[~np.isnan(d)])
        if k % 50 == 0 or k == max_k:
            elapsed = time.time() - t_null
            eta = (max_k - k) * elapsed / max(k, 1)
            print(f'[sumrank] {contrast} null: {k}/{max_k} '
                  f'({elapsed:.0f}s elapsed, eta {eta:.0f}s)', flush=True)

    null_sorted = {
        a: np.sort(np.concatenate(arrs)) if arrs else np.array([])
        for a, arrs in null_by_a.items()}

    emp_up = np.full(len(real), np.nan)
    emp_dn = np.full(len(real), np.nan)
    for i, a in enumerate(real['cell_type_a'].to_numpy()):
        arr = null_sorted.get(a, np.array([]))
        if arr.size == 0:
            continue
        emp_up[i] = 1 - np.searchsorted(
            arr, real['nlp_up'].iat[i]) / arr.size
        emp_dn[i] = 1 - np.searchsorted(
            arr, real['nlp_down'].iat[i]) / arr.size

    real = real.assign(
        contrast=contrast,
        emp_p_up=emp_up, emp_p_down=emp_dn,
        emp_fdr_up=fdrcorrection(
            np.where(np.isnan(emp_up), 1.0, emp_up))[1],
        emp_fdr_down=fdrcorrection(
            np.where(np.isnan(emp_dn), 1.0, emp_dn))[1])
    sumrank_frames.append(real)

    n_up = int((real['emp_fdr_up'] < FDR_THRESHOLD).sum())
    n_dn = int((real['emp_fdr_down'] < FDR_THRESHOLD).sum())
    print(f'[sumrank] {contrast}: {len(real):,} pairs, '
          f'{n_up} up / {n_dn} down sig (emp_fdr<{FDR_THRESHOLD})',
          flush=True)

if sumrank_frames:
    sumrank_out = pd.concat(sumrank_frames, ignore_index=True)
    sumrank_out.to_csv(sumrank_path, index=False)
    print(f'[sumrank] wrote proximity_local_sumrank.csv '
          f'({sumrank_out.shape[0]:,} rows)', flush=True)
    print(f'[sumrank] wrote proximity_local_sumrank.csv '
          f'({sumrank_out.shape[0]:,} rows)', flush=True)

#endregion
