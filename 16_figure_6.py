"""Figure 6: psychiatric-trait heritability mapped onto the maternal brain.
A) peak Cauchy enrichment per trait per platform;
B) trait x cell-type heatmap, Slide-tags over Xenium;
C) per-cell MDD association in situ;
D) exemplar subclasses across reproductive states.
Supplementary panels (figS_*) cover trait specificity, platform concordance,
trait correlation and the Xenium-only subclasses.
Reads the gsMap tables written by 09_gsmap.py.
"""

#region imports and setup ######################################################

import os
import re
import warnings

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/output'
gsmap_dir = f'{out_dir}/gsmap'
fig_dir = f'{working_dir}/figures/gsmap'
os.makedirs(fig_dir, exist_ok=True)
FDR = 0.10
cell_type_col = 'subclass'

TOP_TRAITS = ['MDD', 'ADHD', 'Neuroticism', 'Autism', 'PTSD']
EXEMPLAR_CELLTYPES = {
    'MDD': ['081 ACB-BST-FS D1 Gaba',
            '066 NDB-SI-ant Prdm12 Gaba',
            '119 SI-MA-LPO-LHA Skor1 Glut',
            '032 L5 NP CTX Glut',
            '029 L6b CTX Glut',
            '114 COAa-PAA-MEA Barhl2 Glut'],
}
XENIUM_ONLY_CELLTYPES = {
    'MDD': ['088 BST Tac2 Gaba',
            '090 BST-MPN Six3 Nrgn Gaba',
            '089 PVR Six3 Sox3 Gaba',
            '072 LSX Sall3 Lmo1 Gaba'],
}
COND_ORDER = {
    'slidetags': ['CTRL', 'PREG', 'POSTPART'],
    'xenium': ['CTRL', 'PREG']}
COND_COLORS = {
    'CTRL': '#7209b7', 'PREG': '#b5179e', 'POSTPART': '#f72585'}
DS_COLORS = {
    'slidetags': '#0B6FA8', 'xenium': '#1FAA6B'}
DS_LABEL = {
    'slidetags': 'Slide-tags', 'xenium': 'Xenium'}
POINT_SIZE = {
    'slidetags': 1, 'xenium': 0.2}

def _save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(f'{fig_dir}/{name}.{ext}',
                    dpi=400 if ext == 'png' else None,
                    bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {name}')

def _bh_fdr(p):
    p = np.asarray(p, dtype=float)
    valid = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if not valid.any():
        return out
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    q = (pv * n / ranks)[order]
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_final = np.empty(n)
    q_final[order] = np.clip(q, 0, 1)
    out[valid] = q_final
    return out

def _wrap_ct(ct):
    parts = ct.split()
    if len(parts) <= 3:
        return ct
    return ' '.join(parts[:-2]) + '\n' + ' '.join(parts[-2:])

#endregion

#region load tables ############################################################

cauchy = pl.read_csv(f'{gsmap_dir}/cauchy_fdr.csv')
per_ds = pl.read_csv(f'{gsmap_dir}/per_dataset.csv')
meta = pl.read_csv(f'{gsmap_dir}/meta.csv')
means = pl.read_parquet(f'{gsmap_dir}/sample_means.parquet')
print(f'cauchy: {cauchy.height}  per_ds: {per_ds.height}  '
      f'meta: {meta.height}  means: {means.height}')

#endregion

#region fig 6A — trait ranking #################################################
# Peak -log10(p_cauchy) per trait per dataset, Bonferroni lines.

def plot_trait_ranking():
    peaks = (cauchy
        .group_by(['dataset', 'trait'])
        .agg(pl.col('neg_log10_p_cauchy').max().alias('peak'))
        .pivot(on='dataset', index='trait', values='peak')
        .with_columns(
            pl.mean_horizontal('slidetags', 'xenium').alias('mean_peak'))
        .sort('mean_peak', descending=True)
        .to_pandas())

    n_tr = cauchy['trait'].n_unique()
    n_ct = {d: cauchy.filter(pl.col('dataset') == d)['cell_type'].n_unique()
            for d in ('slidetags', 'xenium')}
    # Single Bonferroni line at the more conservative (larger n_ct) threshold
    thr = -np.log10(0.05 / (n_tr * max(n_ct.values())))

    fig, ax = plt.subplots(figsize=(3.4, 3.8), facecolor='white')
    y = np.arange(len(peaks)); w = 0.38
    for i, d in enumerate(['slidetags', 'xenium']):
        offset = (i - 0.5) * w
        ax.barh(y + offset, peaks[d], height=w,
                color=DS_COLORS[d], label=DS_LABEL[d])
    ax.axvline(thr, color='black', linestyle='--', linewidth=0.8, alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(peaks['trait'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r'Peak $-\log_{10}$ p$_{\mathrm{Cauchy}}$', fontsize=10)
    ax.legend(frameon=False, loc='lower right', fontsize=8)
    sns.despine(ax=ax)
    ax.tick_params(axis='y', length=0, labelsize=8)
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    _save(fig, 'fig6A_trait_ranking')

#endregion

#region fig 6B — trait × cell-type heatmap (across-condition) ##################

def select_featured_cts(n_glut=12, n_gaba=12, n_nn=6):
    sub = cauchy.filter(pl.col('trait').is_in(TOP_TRAITS))
    totals = (sub.group_by('cell_type')
              .agg(pl.col('neg_log10_p_cauchy').sum().alias('total'),
                   pl.col('dataset').n_unique().alias('n_ds'))
              .filter(pl.col('n_ds') == 2)
              .sort('total', descending=True)
              .to_pandas())
    def type_key(ct):
        if 'Glut' in ct: return 0
        if 'Gaba' in ct or 'IMN' in ct: return 1
        return 2
    def prefix_num(ct):
        m = re.match(r'(\d+)\s', ct)
        return int(m.group(1)) if m else 9999
    totals['type'] = totals['cell_type'].map(type_key)
    picks = []
    for t, n in [(0, n_glut), (1, n_gaba), (2, n_nn)]:
        group = totals[totals['type'] == t]['cell_type']
        group_cts = group.tolist() if n is None else group.head(n).tolist()
        group_cts.sort(key=prefix_num)  # numeric prefix order within type
        picks.extend(group_cts)
    return picks

def _pivot_scores(df_ds, dataset, featured):
    sub = (df_ds.filter(pl.col('dataset') == dataset)
           .filter(pl.col('trait').is_in(TOP_TRAITS))
           .to_pandas())
    sub['fdr_ds'] = _bh_fdr(sub['p_cauchy'].values)
    agg = (sub[sub['cell_type'].isin(featured)]
           .groupby(['trait', 'cell_type'], as_index=False)
           .agg(score=('neg_log10_p_cauchy', 'max'),
                fdr=('fdr_ds', 'min')))
    scores = agg.pivot(index='trait', columns='cell_type',
                       values='score').reindex(
        index=TOP_TRAITS, columns=featured)
    fdrs = agg.pivot(index='trait', columns='cell_type',
                     values='fdr').reindex(
        index=TOP_TRAITS, columns=featured)
    return scores, fdrs

def plot_condition_heatmap(sig_fdr=0.01):
    featured = select_featured_cts()
    print(f'[heatmap] {len(featured)} cell-types, sig asterisk at FDR<{sig_fdr}')
    cmap = plt.get_cmap('viridis')
    nT = len(TOP_TRAITS); nC = len(featured)
    # col-gap between major type groups (Glut → Gaba → NN)
    def _typ(ct):
        if 'Glut' in ct: return 0
        if 'Gaba' in ct or 'IMN' in ct: return 1
        return 2
    types = [_typ(c) for c in featured]
    GAP = 0.4
    gaps_before = [0]
    for i in range(1, nC):
        gaps_before.append(
            gaps_before[-1] + (GAP if types[i] != types[i - 1] else 0))
    xpos = [j + gaps_before[j] for j in range(nC)]
    x_extent = (xpos[-1] + 1) if nC else 1

    fig = plt.figure(figsize=(0.26 * nC + 2.0, 3.8), facecolor='white')
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1, 0.025], height_ratios=[1, 1],
        hspace=0.08, wspace=0.035, left=0.22, right=0.93,
        top=0.96, bottom=0.3)

    for row, ds in enumerate(('slidetags', 'xenium')):
        ax = fig.add_subplot(gs[row, 0])
        cax = fig.add_subplot(gs[row, 1])
        scores, fdrs = _pivot_scores(cauchy, ds, featured)
        vmax = np.nanmax(scores.values)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)

        for i in range(nT):
            for j in range(nC):
                val = scores.iloc[i, j]
                ax.add_patch(mpatches.Rectangle(
                    (xpos[j], i), 1, 1,
                    facecolor=cmap(norm(val)) if np.isfinite(val)
                    else (0.93, 0.93, 0.93, 1.0),
                    edgecolor='white', linewidth=0.5))
                f = fdrs.iloc[i, j]
                if np.isfinite(f) and f < sig_fdr:
                    ax.text(xpos[j] + 0.5, i + 0.5, '*',
                            ha='center', va='center',
                            fontsize=8, color='white',
                            fontweight='bold')

        ax.set_xlim(0, x_extent); ax.set_ylim(nT, 0)
        is_last = row == 1
        if is_last:
            ax.set_xticks([x + 0.5 for x in xpos])
            ax.set_xticklabels(featured, rotation=45,
                               ha='right', fontsize=8)
        else:
            ax.set_xticks([])
        ax.set_yticks(np.arange(nT) + 0.5)
        ax.set_yticklabels(TOP_TRAITS, fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_ylabel(DS_LABEL[ds], fontsize=10, labelpad=4)

        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                          cax=cax)
        cb.set_label(r'$-\log_{10}$ p$_{\mathrm{Cauchy}}$', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.text(0.99, 0.01, f'* FDR < {sig_fdr:g}', fontsize=8,
             color='black', ha='right', va='bottom')
    _save(fig, 'fig6B_trait_ct_heatmap')


def plot_trait_specificity(ref_trait='MDD',
                           compare_traits=('PPD', 'Stroke')):
    traits = [ref_trait] + list(compare_traits)
    TR_COLORS = {'MDD': '#D62728', 'PPD': '#1F77B4',
                 'Neuroticism': '#1F77B4', 'Stroke': '#808080',
                 'AD': '#808080', 'ADHD': '#9467BD',
                 'Autism': '#2CA02C', 'PTSD': '#FF7F0E'}

    top = (meta.filter((pl.col('trait') == ref_trait) &
                       (pl.col('contrast') == 'PREG_vs_CTRL'))
           .filter(pl.col('cell_type').str.contains_any(
               ['Glut', 'Gaba', 'IMN']))
           .with_columns(meta_min_p=pl.min_horizontal('p_up', 'p_down'))
           .sort('meta_min_p').head(10).to_pandas())
    cts = top['cell_type'].tolist()
    n = len(cts)

    ds_means_all = (means.filter(
        (pl.col('trait').is_in(traits)) &
        (pl.col('cell_type').is_in(cts)))
        .to_pandas())
    pvals_all = (per_ds.filter(
        (pl.col('contrast') == 'PREG_vs_CTRL') &
        (pl.col('trait').is_in(traits)) &
        (pl.col('cell_type').is_in(cts)))
        .select(['dataset', 'trait', 'cell_type', 'p'])
        .to_pandas().set_index(['dataset', 'trait', 'cell_type'])['p'])

    fig, axes = plt.subplots(
        2, n, figsize=(1.5 * n + 1.0, 4.2), facecolor='white',
        squeeze=False, sharex='col')
    rng = np.random.default_rng(0)
    all_conds = COND_ORDER['slidetags']
    xs = np.arange(len(all_conds))

    for row, ds in enumerate(('slidetags', 'xenium')):
        conds = all_conds
        ds_means = ds_means_all[ds_means_all['dataset'] == ds]

        for col, ct in enumerate(cts):
            ax = axes[row, col]
            panel_ymin, panel_ymax = np.inf, -np.inf
            any_data = False
            for trait in traits:
                sub = ds_means[(ds_means['cell_type'] == ct) &
                               (ds_means['trait'] == trait)]
                if sub.empty: continue
                ctrl_mean = sub[sub['condition'] == 'CTRL']['mean_score'].mean()
                sub = sub.copy()
                sub['centered'] = sub['mean_score'] - ctrl_mean
                color = TR_COLORS.get(trait, '#000000')
                means_arr, sems_arr = [], []
                for i, cond in enumerate(conds):
                    v = sub[sub['condition'] == cond]['centered'].values
                    if len(v) == 0:
                        means_arr.append(np.nan); sems_arr.append(np.nan)
                        continue
                    any_data = True
                    jit = rng.normal(0, 0.035, size=len(v))
                    ax.scatter(np.full(len(v), i) + jit, v,
                               color=color, s=14, alpha=0.7,
                               edgecolor='black', linewidth=0.3, zorder=3)
                    means_arr.append(v.mean())
                    sems_arr.append(v.std(ddof=1) / np.sqrt(len(v))
                                    if len(v) > 1 else 0.0)
                    panel_ymin = min(panel_ymin, v.min())
                    panel_ymax = max(panel_ymax, v.max())
                means_arr = np.array(means_arr); sems_arr = np.array(sems_arr)
                valid = ~np.isnan(means_arr)
                if valid.sum() >= 2:
                    ax.errorbar(xs[valid], means_arr[valid],
                                yerr=sems_arr[valid], fmt='-', color=color,
                                linewidth=1.2, capsize=2, capthick=0.8,
                                marker='None', label=trait, zorder=4)

            ax.axhline(0, color='black', linewidth=0.4, alpha=0.4, zorder=0)

            if np.isfinite(panel_ymin) and np.isfinite(panel_ymax):
                rng_span = max(panel_ymax - panel_ymin, 0.4)
                pad_for_text = rng_span * 0.50
                ax.set_ylim(panel_ymin - rng_span * 0.08,
                            panel_ymax + rng_span * 0.05 + pad_for_text)

            ax.set_xticks(xs)
            if row == 1:
                ax.set_xticklabels(
                    ['N' if c == 'CTRL' else 'P' if c == 'PREG' else 'PP'
                     for c in conds], fontsize=8)
                ax.tick_params(axis='x', length=2, labelsize=8)
            else:
                ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', length=2, labelsize=8)
            sns.despine(ax=ax)

            if row == 0:
                ax.set_title(_wrap_ct(ct), fontsize=8, pad=8)
            if col == 0:
                ax.set_ylabel(f'{DS_LABEL[ds]}\n' +
                              r'$\Delta$ ' + ref_trait + ' ' +
                              r'$-\log_{10}$ p$_{\mathrm{Cauchy}}$',
                              fontsize=9)

            x0, y0 = 0.03, 0.97
            for k, trait in enumerate(traits):
                p = pvals_all.get((ds, trait, ct), np.nan)
                if not np.isfinite(p):
                    txt = f'{trait} n/a'
                else:
                    txt = f'{trait} p={p:.2g}'
                ax.text(x0, y0 - 0.085 * k, txt,
                        transform=ax.transAxes,
                        color=TR_COLORS.get(trait, '#000000'),
                        fontsize=6.5, ha='left', va='top')

    handles = [plt.Line2D([], [], color=TR_COLORS.get(t, '#000000'),
                          marker='o', markersize=5,
                          markeredgecolor='black', markeredgewidth=0.4,
                          label=t, linewidth=1.2)
               for t in traits]
    fig.legend(handles=handles, loc='lower center',
               bbox_to_anchor=(0.5, -0.01), ncol=len(traits),
               frameon=False, fontsize=9)

    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    _save(fig, 'figS_trait_specificity')


def plot_platform_concordance(traits=None):
    traits = traits or TOP_TRAITS
    n = len(traits)

    cj = pd.read_csv(
        '/home/karbabi/single-cell/ABC/metadata/cells_joined.csv',
        usecols=['class', 'subclass', 'class_color']).drop_duplicates()
    ct_to_class = dict(zip(cj['subclass'], cj['class']))
    class_color = dict(zip(cj['class'], cj['class_color']))

    peak = (cauchy.filter(pl.col('trait').is_in(traits))
            .group_by(['dataset', 'trait', 'cell_type'])
            .agg(pl.col('neg_log10_p_cauchy').max().alias('peak'))
            .to_pandas())

    fig, axes = plt.subplots(1, n, figsize=(2.5 * n + 2.4, 2.8),
                             facecolor='white', sharey=False)
    axes = np.atleast_1d(axes)
    classes_used = set()

    for ax, trait in zip(axes, traits):
        sl = peak[(peak['dataset'] == 'slidetags') &
                  (peak['trait'] == trait)].set_index('cell_type')['peak']
        xe = peak[(peak['dataset'] == 'xenium') &
                  (peak['trait'] == trait)].set_index('cell_type')['peak']
        common = sl.index.intersection(xe.index)
        x = sl.loc[common].values; y = xe.loc[common].values
        cls = [ct_to_class.get(c, 'Unlabelled') for c in common]
        colors = [class_color.get(c, '#d3d3d3') for c in cls]
        classes_used.update(cls)

        ax.scatter(x, y, c=colors, s=18, edgecolor='black',
                   linewidth=0.3, alpha=0.9, rasterized=True)

        lo = 0
        hi = max(x.max(), y.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], color='black',
                linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)

        if len(x) >= 3:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(lo, hi, 50)
            ax.plot(xs, m * xs + b, color='black', linewidth=1.0,
                    alpha=0.8, zorder=1)
            r, _ = pearsonr(x, y)
            ax.text(0.04, 0.96, f'r = {r:.2f}\nn = {len(x)}',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=8)

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(f'Slide-tags peak\n' +
                      r'$-\log_{10}$ p$_{\mathrm{Cauchy}}$', fontsize=9)
        ax.set_ylabel(r'Xenium peak $-\log_{10}$ p$_{\mathrm{Cauchy}}$',
                      fontsize=9)
        ax.set_title(trait, fontsize=10)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    legend_classes = sorted(classes_used)
    handles = [plt.Line2D([], [], marker='o', linestyle='',
                          markerfacecolor=class_color.get(c, '#d3d3d3'),
                          markeredgecolor='black', markeredgewidth=0.3,
                          markersize=6, label=c)
               for c in legend_classes]
    fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.0, 0.5),
               fontsize=7, frameon=False, title='ABC class',
               title_fontsize=8, handletextpad=0.4)

    fig.tight_layout(rect=[0, 0, 0.84, 1])
    _save(fig, 'figS_platform_concordance')


def plot_trait_correlation():
    mat = (cauchy
        .with_columns(key=pl.col('dataset') + '|' + pl.col('condition') +
                      '|' + pl.col('cell_type'))
        .select(['trait', 'key', 'neg_log10_p_cauchy'])
        .to_pandas()
        .pivot(index='key', columns='trait', values='neg_log10_p_cauchy'))

    traits = list(mat.columns)
    corr = pd.DataFrame(np.eye(len(traits)), index=traits, columns=traits)
    for i, a in enumerate(traits):
        for b in traits[i+1:]:
            sub = mat[[a, b]].dropna()
            if len(sub) < 10:
                corr.loc[a, b] = corr.loc[b, a] = np.nan
                continue
            r = sub[a].corr(sub[b])
            corr.loc[a, b] = corr.loc[b, a] = r

    d = 1 - corr.fillna(0).values
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    order = leaves_list(linkage(squareform(d, checks=False), method='average'))
    corr = corr.iloc[order, order]

    fig, ax = plt.subplots(figsize=(5.5, 5.0), facecolor='white')
    im = ax.imshow(corr.values, cmap='seismic', vmin=-1, vmax=1,
                   aspect='equal')
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)

    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.iloc[i, j]
            if pd.isna(v): continue
            c = 'white' if abs(v) > 0.5 else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6, color=c)

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label('Pearson r', fontsize=10)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    _save(fig, 'figS_trait_correlation')

#endregion

#region fig 6C — single-cell spatial map #######################################

_adata_cache = {}
def _load_adata(dataset):
    if dataset not in _adata_cache:
        path = f'{out_dir}/{dataset}/03_adata_query_{dataset}.h5ad'
        a = sc.read_h5ad(path)
        if dataset == 'xenium':
            a = a[~a.obs['sample'].isin(['CTRL_3'])].copy()
        _adata_cache[dataset] = a
    return _adata_cache[dataset]

def plot_spatial(trait='MDD', condition='PREG'):
    panels = {}
    for ds in ('slidetags', 'xenium'):
        path = (f'{gsmap_dir}/{ds}/{condition}/spatial_ldsc/'
                f'{condition}_{trait}.csv.gz')
        if not os.path.exists(path):
            panels[ds] = None; continue
        ldsc = pl.read_csv(path).with_columns(
            score=-pl.col('p').log10())
        adata = _load_adata(ds)
        obs = adata.obs[adata.obs['condition'] == condition][
            ['x_ffd', 'y_ffd']]
        obs.index = obs.index.astype(str)
        ldsc_pd = ldsc.to_pandas().rename(columns={'spot': 'idx'})
        ldsc_pd['idx'] = ldsc_pd['idx'].astype(str)
        df = obs.reset_index().rename(columns={'index': 'idx'}).merge(
            ldsc_pd[['idx', 'score']], on='idx', how='inner')
        panels[ds] = df

    fig, axes = plt.subplots(
        2, 1, figsize=(3.0, 6.4), facecolor='white',
        gridspec_kw=dict(hspace=0.18, left=0.05, right=0.95,
                         top=0.90, bottom=0.03))

    for ax, ds in zip(axes, ('slidetags', 'xenium')):
        df = panels.get(ds)
        ax.set_facecolor('black')
        ax.set_box_aspect(1)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(DS_LABEL[ds], fontsize=10, loc='center')
        if df is None or df.empty:
            continue
        df = df.sort_values('score').reset_index(drop=True)
        vmin, vmax = np.percentile(df['score'].values, [1, 99])
        sc_plt = ax.scatter(
            df['x_ffd'], df['y_ffd'], c=df['score'],
            s=POINT_SIZE[ds], cmap='viridis', vmin=vmin, vmax=vmax,
            rasterized=True, edgecolors='none')
        cax = ax.inset_axes([0.62, 0.09, 0.33, 0.03])
        cb = fig.colorbar(sc_plt, cax=cax, orientation='horizontal')
        cb.set_ticks([vmin, vmax])
        cb.set_ticklabels([f'{vmin:.0f}', f'{vmax:.0f}'])
        cb.ax.tick_params(labelsize=5, colors='white', pad=1)
        cb.outline.set_edgecolor('white')
        cb.outline.set_linewidth(0.4)
        cb.ax.text(0.5, 1.8, r'$-\log_{10}$ p',
                   transform=cb.ax.transAxes, ha='center', va='bottom',
                   color='white', fontsize=6)

    fig.suptitle(f'{trait} — {condition}', fontsize=11, y=0.98)
    _save(fig, f'fig6C_spatial_{trait}_{condition}')

#endregion

#region fig 6D — exemplar cell-type trajectories ###############################

def _load_percell_scores(trait, cts):
    cts = set(cts)
    out = {}
    for ds in ('slidetags', 'xenium'):
        adata = _load_adata(ds)
        obs = adata.obs[['sample', 'condition', cell_type_col]].astype(str)
        obs.index = obs.index.astype(str)
        obs_pl = pl.from_pandas(
            obs.reset_index().rename(columns={
                'index': 'spot', cell_type_col: 'cell_type'}))
        for cond in sorted(obs['condition'].unique()):
            path = (f'{gsmap_dir}/{ds}/{cond}/spatial_ldsc/'
                    f'{cond}_{trait}.csv.gz')
            if not os.path.exists(path):
                continue
            ldsc = (pl.read_csv(path)
                    .with_columns(score=-pl.col('p').log10())
                    .select('spot', 'score'))
            joined = (ldsc.join(obs_pl, on='spot', how='inner')
                      .filter(pl.col('cell_type').is_in(list(cts))))
            out[(ds, cond)] = joined.to_pandas()
    return out


def plot_exemplar_trajectories(trait='MDD', cts=None, max_swarm=1200):
    cts = cts or EXEMPLAR_CELLTYPES.get(trait, [])
    n = len(cts)
    if n == 0: return
    data = _load_percell_scores(trait, cts)

    fig, axes = plt.subplots(
        2, n, figsize=(1.45 * n + 0.5, 4.8), facecolor='white',
        squeeze=False, sharex='col')
    rng = np.random.default_rng(0)

    all_conds = COND_ORDER['slidetags']
    for row, ds in enumerate(('slidetags', 'xenium')):
        conds = all_conds
        for col, ct in enumerate(cts):
            ax = axes[row, col]
            sample_means_ct = []
            swarm_y = []
            swarm_x = []
            swarm_c = []
            y_all = []
            for i, cond in enumerate(conds):
                df = data.get((ds, cond))
                if df is None:
                    sample_means_ct.append(([], [])); continue
                cell_df = df[df['cell_type'] == ct]
                if cell_df.empty:
                    sample_means_ct.append(([], [])); continue

                vals = cell_df['score'].values
                if len(vals) > max_swarm:
                    idx = rng.choice(len(vals), max_swarm, replace=False)
                    vals_sub = vals[idx]
                else:
                    vals_sub = vals
                jit = rng.normal(0, 0.07, size=len(vals_sub))
                swarm_x.extend(i + jit); swarm_y.extend(vals_sub)
                swarm_c.extend([COND_COLORS[cond]] * len(vals_sub))
                y_all.extend(vals)

                sm = (cell_df.groupby('sample')['score'].mean()
                      .reset_index())
                sample_means_ct.append(
                    (sm['sample'].tolist(), sm['score'].tolist()))

            if y_all:
                ax.scatter(swarm_x, swarm_y, s=1.2, c=swarm_c,
                           alpha=0.22, rasterized=True,
                           edgecolors='none', zorder=1)

            xs = np.arange(len(conds))
            means_arr, sems_arr = [], []
            for i, cond in enumerate(conds):
                _, v = sample_means_ct[i]
                if not v:
                    means_arr.append(np.nan); sems_arr.append(np.nan)
                    continue
                v = np.asarray(v, float)
                jit = rng.normal(0, 0.035, size=len(v))
                ax.scatter(np.full(len(v), i) + jit, v,
                           color=COND_COLORS[cond], s=34,
                           edgecolor='black', linewidth=0.5, zorder=4)
                means_arr.append(v.mean())
                sems_arr.append(v.std(ddof=1) / np.sqrt(len(v))
                                if len(v) > 1 else 0.0)
            means_arr = np.array(means_arr); sems_arr = np.array(sems_arr)
            valid = ~np.isnan(means_arr)
            if valid.sum() >= 2:
                ax.errorbar(xs[valid], means_arr[valid],
                            yerr=sems_arr[valid],
                            fmt='-', color='black',
                            linewidth=1.0, capsize=3, capthick=0.8,
                            ecolor='black', marker='None', zorder=5)

            ct_stats = per_ds.filter(
                (pl.col('trait') == trait) &
                (pl.col('cell_type') == ct) &
                (pl.col('dataset') == ds)).to_pandas()
            if y_all:
                ytop = np.nanmax(y_all)
                step = max(ytop * 0.04, 0.2)
                y = ytop + step * 0.5
                for _, r in ct_stats.iterrows():
                    treat, base = r['contrast'].split('_vs_')
                    if treat not in conds or base not in conds:
                        continue
                    p = r['p']
                    if not np.isfinite(p) or p >= 0.05:
                        continue
                    star = '***' if p<1e-3 else '**' if p<1e-2 else '*'
                    x1 = conds.index(base); x2 = conds.index(treat)
                    ax.plot([x1, x1, x2, x2],
                            [y, y + step*0.4, y + step*0.4, y],
                            color='black', lw=0.6, zorder=7)
                    ax.text((x1 + x2) / 2, y + step * 0.45, star,
                            ha='center', va='bottom', fontsize=9, zorder=7)
                    y += step * 1.6

            ax.set_xticks(xs)
            if row == 1:
                ax.set_xticklabels(
                    ['N' if c == 'CTRL' else 'P' if c == 'PREG' else 'PP'
                     for c in conds],
                    fontsize=8)
                ax.tick_params(axis='x', length=2, labelsize=8)
            else:
                ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='y', length=2, labelsize=8)
            sns.despine(ax=ax)
            if col == 0:
                ax.set_ylabel(f'{DS_LABEL[ds]}\nMDD ' +
                              r'$-\log_{10}$ p$_{\mathrm{Cauchy}}$',
                              fontsize=10)
            if row == 0:
                ax.set_title(_wrap_ct(ct), fontsize=8, pad=8)

    fig.suptitle(f'{trait} enrichment across conditions',
                 fontsize=11, x=0.5, y=0.995, ha='center')
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    _save(fig, f'fig6D_trajectories_{trait}')


def plot_xenium_only_supp(trait='MDD', cts=None, max_swarm=1200):
    cts = cts or XENIUM_ONLY_CELLTYPES.get(trait, [])
    if not cts: return
    data = _load_percell_scores(trait, cts)
    de_xe = (per_ds.filter((pl.col('trait') == trait) &
                           (pl.col('contrast') == 'PREG_vs_CTRL') &
                           (pl.col('dataset') == 'xenium'))
             .to_pandas().set_index('cell_type'))
    betas = {c: de_xe.at[c, 'beta'] if c in de_xe.index else 0.0
             for c in cts}
    cts = sorted(cts, key=lambda c: (betas[c] >= 0, -abs(betas[c])))
    n = len(cts)

    fig, axes = plt.subplots(
        1, n, figsize=(1.45 * n + 0.5, 3.0), facecolor='white',
        squeeze=False)
    axes = axes[0]
    rng = np.random.default_rng(0)

    conds = COND_ORDER['xenium']
    xs = np.array([0.0, 0.5])

    for col, ct in enumerate(cts):
        ax = axes[col]
        sample_means_ct = []
        swarm_x, swarm_y, swarm_c, y_all = [], [], [], []
        for i, cond in enumerate(conds):
            df = data.get(('xenium', cond))
            if df is None:
                sample_means_ct.append(([], [])); continue
            cell_df = df[df['cell_type'] == ct]
            if cell_df.empty:
                sample_means_ct.append(([], [])); continue
            vals = cell_df['score'].values
            vals_sub = (rng.choice(vals, max_swarm, replace=False)
                        if len(vals) > max_swarm else vals)
            jit = rng.normal(0, 0.035, size=len(vals_sub))
            swarm_x.extend(xs[i] + jit); swarm_y.extend(vals_sub)
            swarm_c.extend([COND_COLORS[cond]] * len(vals_sub))
            y_all.extend(vals)
            sm = cell_df.groupby('sample')['score'].mean().reset_index()
            sample_means_ct.append(
                (sm['sample'].tolist(), sm['score'].tolist()))

        if y_all:
            ax.scatter(swarm_x, swarm_y, s=1.2, c=swarm_c,
                       alpha=0.22, rasterized=True,
                       edgecolors='none', zorder=1)

        means_arr, sems_arr = [], []
        for i, cond in enumerate(conds):
            _, v = sample_means_ct[i]
            if not v:
                means_arr.append(np.nan); sems_arr.append(np.nan); continue
            v = np.asarray(v, float)
            jit = rng.normal(0, 0.018, size=len(v))
            ax.scatter(np.full(len(v), xs[i]) + jit, v,
                       color=COND_COLORS[cond], s=34,
                       edgecolor='black', linewidth=0.5, zorder=4)
            means_arr.append(v.mean())
            sems_arr.append(v.std(ddof=1) / np.sqrt(len(v))
                            if len(v) > 1 else 0.0)
        means_arr = np.array(means_arr); sems_arr = np.array(sems_arr)
        valid = ~np.isnan(means_arr)
        if valid.sum() >= 2:
            ax.errorbar(xs[valid], means_arr[valid],
                        yerr=sems_arr[valid],
                        fmt='-', color='black',
                        linewidth=1.0, capsize=3, capthick=0.8,
                        ecolor='black', marker='None', zorder=5)

        ct_stats = per_ds.filter(
            (pl.col('trait') == trait) &
            (pl.col('cell_type') == ct) &
            (pl.col('dataset') == 'xenium') &
            (pl.col('contrast') == 'PREG_vs_CTRL')).to_pandas()
        if y_all and not ct_stats.empty:
            ytop = np.nanmax(y_all)
            step = max(ytop * 0.05, 0.2)
            y = ytop + step * 0.6
            p = float(ct_stats['p'].iloc[0])
            ax.plot([xs[0], xs[0], xs[1], xs[1]],
                    [y, y + step*0.35, y + step*0.35, y],
                    color='black', lw=0.6, zorder=7)
            ax.text((xs[0] + xs[1]) / 2, y + step * 0.5,
                    f'p = {p:.3g}', ha='center', va='bottom',
                    fontsize=7, zorder=7)

        ax.set_xticks(xs)
        ax.set_xticklabels(['N', 'P'], fontsize=8)
        ax.set_xlim(xs[0] - 0.25, xs[1] + 0.25)
        ax.tick_params(axis='x', length=2, labelsize=8)
        ax.tick_params(axis='y', length=2, labelsize=8)
        sns.despine(ax=ax)
        if col == 0:
            ax.set_ylabel(f'{DS_LABEL["xenium"]}\nMDD ' +
                          r'$-\log_{10}$ p$_{\mathrm{Cauchy}}$',
                          fontsize=10)
        ax.set_title(_wrap_ct(ct), fontsize=8, pad=8)

    fig.suptitle(f'{trait} — xenium-exclusive peripartum populations',
                 fontsize=11, x=0.5, y=0.995, ha='center')
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, f'figS_xenium_only_{trait}')

#endregion

#region main ###################################################################

if __name__ == '__main__':
    plot_trait_ranking()
    plot_condition_heatmap()
    plot_spatial(trait='MDD', condition='PREG')
    plot_exemplar_trajectories(trait='MDD')
    plot_xenium_only_supp(trait='MDD')

#endregion
