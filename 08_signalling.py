#region setup ##################################################################

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import liana as li
from scipy.spatial.distance import pdist

os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_py

warnings.filterwarnings('ignore')

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
LIANA_OUT = f'{working_dir}/output/liana'

datasets = {
    'slidetags': {
        'path':
            f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'), ('POSTPART', 'CTRL')],
        'bandwidth_um': 100,
    },
    'xenium': {
        'path':
            f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
        'bandwidth_um': 50,
    },
}

for _name in datasets:
    os.makedirs(f'{LIANA_OUT}/{_name}', exist_ok=True)
os.makedirs(f'{working_dir}/figures/liana', exist_ok=True)

mc = li.rs.select_resource('mouseconsensus')

r('''
suppressPackageStartupMessages(library(NeuronChat))
data("interactionDB_mouse")
nc_df <- do.call(rbind, lapply(interactionDB_mouse, function(x) {
    expand.grid(ligand = x$lig_contributor,
                receptor = x$receptor_subunit,
                stringsAsFactors = FALSE)
}))
nc_df <- unique(nc_df[, c("ligand", "receptor")])
rownames(nc_df) <- NULL
''')
nc = to_py('nc_df', format='pandas', index=False)
resource = (pd.concat([mc, nc], ignore_index=True)
              .drop_duplicates(subset=['ligand', 'receptor'])
              .reset_index(drop=True))
print(f'mouseconsensus: {len(mc):,}  neuronchat: {len(nc):,}  '
      f'union (deduped): {len(resource):,} LR pairs')

#endregion

#region load data ##############################################################

adatas = {}
for name, cfg in datasets.items():
    adata = sc.read_h5ad(cfg['path'])
    if 'gene_symbol' in adata.var.columns:
        adata.var.index = adata.var['gene_symbol']
        adata.var_names_make_unique()
        adata.var.drop(columns='gene_symbol', inplace=True)
    adata.var.index.name = None
    drop = cfg.get('drop_samples', [])
    if drop:
        adata = adata[~adata.obs['sample'].isin(drop)].copy()
        print(f'[{name}] dropped samples: {drop}')
    adatas[name] = adata
    print(f'[{name}] {adata.shape[0]:,} cells, '
          f'{adata.obs[cell_type_col].nunique()} subclasses, '
          f'{adata.var.shape[0]:,} genes')

#endregion

#region shared helpers + bandwidth sanity vis #################################

def per_sample_path(name, cond, samp):
    return (f'{LIANA_OUT}/{name}/'
            f'inflow_{cond}_{samp}_{cell_type_col}.parquet')

def per_cond_path(name, cond):
    return (f'{LIANA_OUT}/{name}/'
            f'inflow_cond_{cond}_{cell_type_col}.parquet')

SVG_PVAL_FDR = 0.05
SVG_MORAN_I = 0.01
SVI_PVAL_FDR = 0.05
SVI_MORAN_I = 0.01
N_PERMS = 1000

def affine_to_um(adata):
    rng = np.random.default_rng(0)
    n = min(1000, adata.n_obs)
    idx = rng.choice(adata.n_obs, n, replace=False)
    raw = adata.obs.iloc[idx][['x_raw', 'y_raw']].values.astype(float)
    aff = adata.obs.iloc[idx][['x_affine', 'y_affine']].values.astype(float)
    return float(np.median(pdist(raw)) / np.median(pdist(aff)))

# Render the spatial-weight kernel for one cell per platform BEFORE the heavy
# inflow loops so the bandwidth/affine conversion can be eyeballed first.
from plotnine import (
    coord_fixed, labs, theme, theme_minimal, element_text, element_blank,
    element_rect, scale_colour_gradientn)

def plot_connectivity_one(name, cfg):
    sample = sorted(adatas[name].obs[adatas[name].obs[
        cell_type_col].notna()]['sample'].unique())[0]
    a = adatas[name][(adatas[name].obs['sample'] == sample) &
                     (adatas[name].obs[cell_type_col].notna())].copy()
    a.obsm['spatial'] = a.obs[['x_affine', 'y_affine']].values.astype(float)
    cf = affine_to_um(a)
    bandwidth_aff = cfg['bandwidth_um'] / cf

    li.ut.spatial_neighbors(
        a, bandwidth=bandwidth_aff, spatial_key='spatial', set_diag=True)
    coords = a.obsm['spatial']
    cx, cy = coords.mean(axis=0)
    idx = int(np.argmin(((coords - [cx, cy]) ** 2).sum(axis=1)))
    nnz = int((a.obsp['spatial_connectivities'][:, idx] > 0).sum())
    print(f'[{name}/{sample}] bandwidth={cfg["bandwidth_um"]}µm '
          f'({bandwidth_aff:.4f} aff), centroid idx={idx}, '
          f'{nnz:,} neighbors with w>0')

    p = li.pl.connectivity(
        a, idx=idx,
        spatial_key='spatial',
        connectivity_key='spatial_connectivities',
        size=0.6, figure_size=(5.0, 5.0), return_fig=True)
    title = (f'{name} — {sample} — bandwidth = '
             f'{cfg["bandwidth_um"]} µm  ({nnz:,} cells with w>0)')
    p = (p
        + coord_fixed()
        + scale_colour_gradientn(
            colors=['#f7f7f7', '#67a9cf', '#1c6cc6', '#053061'],
            name='spatial\nweight')
        + labs(title=title, x='x (affine)', y='y (affine)')
        + theme_minimal()
        + theme(
            text=element_text(family='DejaVu Sans', size=8),
            plot_title=element_text(size=9),
            axis_title=element_text(size=8),
            axis_text=element_text(size=7),
            legend_title=element_text(size=7),
            legend_text=element_text(size=7),
            panel_background=element_rect(fill='white'),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank()))
    out = f'{working_dir}/figures/liana/connectivity_{name}_{sample}.png'
    p.save(out, dpi=300, width=5.0, height=5.0, verbose=False)
    print(f'  wrote {out}')

for name, cfg in datasets.items():
    plot_connectivity_one(name, cfg)

#endregion

#region per-sample inflow ######################################################

def run_inflow_one_sample(adata_full, name, cond, samp, bandwidth_um):
    out_path = per_sample_path(name, cond, samp)
    if os.path.exists(out_path):
        print(f'[{name}] {cond}/{samp}: cached')
        return

    adata = adata_full[(adata_full.obs['condition'] == cond) &
                       (adata_full.obs['sample'] == samp)].copy()
    n0 = adata.n_obs
    adata = adata[adata.obs[cell_type_col].notna()].copy()
    adata.obsm['spatial'] = adata.obs[['x_affine', 'y_affine']
                                      ].values.astype(float)
    cf = affine_to_um(adata)
    bandwidth_aff = bandwidth_um / cf

    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f'[{name}] {cond}/{samp}: {n0:,} → {adata.n_obs:,} cells, '
          f'{adata.n_vars:,} genes; conv={cf:.2f} µm/aff, '
          f'bandwidth={bandwidth_um}µm = {bandwidth_aff:.4f} aff')

    li.ut.spatial_neighbors(
        adata, bandwidth=bandwidth_aff, spatial_key='spatial',
        set_diag=True)

    sq.gr.spatial_autocorr(
        adata, mode='moran', use_raw=False,
        connectivity_key='spatial_connectivities',
        show_progress_bar=False)
    moran = adata.uns['moranI']
    svgs = moran.index[(moran['pval_norm_fdr_bh'] < SVG_PVAL_FDR) &
                       (moran['I'] > SVG_MORAN_I)]
    adata = adata[:, svgs].copy()
    print(f'[{name}] {cond}/{samp}: {len(svgs):,} SVGs kept')

    adata.obs[cell_type_col] = (
        adata.obs[cell_type_col].astype('category')
        .cat.remove_unused_categories())
    print(f'[{name}] {cond}/{samp}: {adata.obs[cell_type_col].nunique()} '
          f'cell types present')

    lrdata = li.mt.inflow(
        adata,
        groupby=cell_type_col,
        resource=resource,
        use_raw=False,
        connectivity_key='spatial_connectivities',
        verbose=True)
    print(f'[{name}] {cond}/{samp}: lrdata {lrdata.shape}')

    sq.gr.spatial_autocorr(
        lrdata, mode='moran', use_raw=False,
        connectivity_key='spatial_connectivities',
        show_progress_bar=False)
    moran = lrdata.uns['moranI']
    svis = moran.index[(moran['pval_norm_fdr_bh'] <= SVI_PVAL_FDR) &
                       (moran['I'] > SVI_MORAN_I)]
    lrdata = lrdata[:, svis].copy()
    print(f'[{name}] {cond}/{samp}: {len(svis):,} SVIs kept')

    li.mt.compute_global_specificity(
        lrdata, groupby=cell_type_col, use_raw=False,
        n_perms=N_PERMS, n_jobs=-1, verbose=True)
    df = lrdata.uns['global_interactions'].copy()
    df.to_parquet(out_path, index=False)
    print(f'[{name}] {cond}/{samp}: saved {out_path} '
          f'({len(df):,} rows, '
          f'{int((df["pval"] < 0.05).sum()):,} sig at pval<0.05)')

samples_per = {}
for name, cfg in datasets.items():
    adata = adatas[name]
    conditions_all = sorted({c for pair in cfg['contrasts'] for c in pair})
    for cond in conditions_all:
        samples_per[(name, cond)] = sorted(
            adata.obs[adata.obs['condition'] == cond]['sample']
            .astype(str).unique().tolist())
        for samp in samples_per[(name, cond)]:
            run_inflow_one_sample(
                adata, name, cond, samp, cfg['bandwidth_um'])

#endregion

#region per-condition aggregation ##############################################

KEY_COLS = ['source', 'target', 'ligand_complex', 'receptor_complex']

for (name, cond), samples in samples_per.items():
    out_path = per_cond_path(name, cond)
    if os.path.exists(out_path):
        print(f'[{name}] {cond}: cond cached')
        continue
    sample_dfs = []
    for samp in samples:
        p = per_sample_path(name, cond, samp)
        if not os.path.exists(p):
            raise SystemExit(
                f'[{name}] {cond}: missing per-sample parquet: {p}')
        sample_dfs.append(pd.read_parquet(p))
    df = pd.concat(sample_dfs, ignore_index=True)
    n = len(samples)
    df['_sig'] = (df['pval'] < 0.05).astype(int)
    # lr_mean: sum across samples / fixed n (zero-fill missing).
    # n_sig: count of samples where pval<0.05 (max = n).
    # n_present: count of samples where the (s,t,l,r) survived SVI (max = n).
    # pval_min: min pval across samples where present (NaN if absent in all).
    cond_df = df.groupby(KEY_COLS, as_index=False).agg(
        lr_mean=('lr_mean', 'sum'),
        n_sig=('_sig', 'sum'),
        n_present=('_sig', 'count'),
        pval_min=('pval', 'min'))
    cond_df['lr_mean'] = cond_df['lr_mean'] / n
    cond_df.to_parquet(out_path, index=False)
    print(f'[{name}] {cond}: saved {out_path} '
          f'({len(cond_df):,} LR×CT pairs, n={n} samples, '
          f'sum n_sig={int(cond_df["n_sig"].sum()):,})')

#endregion

#region differential ###########################################################

diff_frames = []
for name, cfg in datasets.items():
    for treat, ctrl in cfg['contrasts']:
        df_ctrl = pd.read_parquet(per_cond_path(name, ctrl))
        df_treat = pd.read_parquet(per_cond_path(name, treat))
        n_ctrl = len(samples_per[(name, ctrl)])
        n_treat = len(samples_per[(name, treat)])
        merged = df_ctrl.merge(
            df_treat, on=KEY_COLS, how='outer',
            suffixes=('_ctrl', '_treat'))
        for c in ('lr_mean', 'n_sig', 'n_present'):
            merged[f'{c}_ctrl'] = merged[f'{c}_ctrl'].fillna(0)
            merged[f'{c}_treat'] = merged[f'{c}_treat'].fillna(0)
        # pval_min stays NaN where absent (no test performed in that condition).
        merged['lr_mean_diff'] = (
            merged['lr_mean_treat'] - merged['lr_mean_ctrl'])
        merged['dataset'] = name
        merged['contrast'] = f'{treat}_vs_{ctrl}'
        merged['n_samples_ctrl'] = n_ctrl
        merged['n_samples_treat'] = n_treat
        diff_frames.append(merged)
        n_pos = int((merged['lr_mean_diff'] > 0).sum())
        n_neg = int((merged['lr_mean_diff'] < 0).sum())
        # coherence summary: pairs with sig in >=half of samples in either condition
        coh_ctrl = int((merged['n_sig_ctrl'] >= (n_ctrl + 1) // 2).sum())
        coh_treat = int((merged['n_sig_treat'] >= (n_treat + 1) // 2).sum())
        print(f'[{name}] {treat}_vs_{ctrl}: '
              f'{len(merged):,} pairs '
              f'(pos={n_pos:,}, neg={n_neg:,}; '
              f'sig≥maj ctrl={coh_ctrl:,} treat={coh_treat:,})')

diff = pd.concat(diff_frames, ignore_index=True)
diff = diff[['dataset', 'contrast'] + KEY_COLS +
            ['lr_mean_ctrl', 'lr_mean_treat', 'lr_mean_diff',
             'n_sig_ctrl', 'n_sig_treat',
             'n_samples_ctrl', 'n_samples_treat',
             'n_present_ctrl', 'n_present_treat',
             'pval_min_ctrl', 'pval_min_treat']]
for (ds, ctr), sub in diff.groupby(['dataset', 'contrast']):
    has_pos = bool((sub['lr_mean_diff'] > 0).any())
    has_neg = bool((sub['lr_mean_diff'] < 0).any())
    assert has_pos and has_neg, (
        f'[{ds}] {ctr}: lr_mean_diff has uniform sign '
        f'(pos={has_pos}, neg={has_neg})')

diff_path = f'{LIANA_OUT}/inflow_diff.csv'
diff.to_csv(diff_path, index=False)
print(f'wrote {diff_path}: {len(diff):,} rows')

#endregion

