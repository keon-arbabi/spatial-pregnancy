#region imports and setup ######################################################

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial import KDTree

warnings.filterwarnings('ignore')

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
stats_coords = ('x_affine', 'y_affine')
FDR_THRESHOLD = 0.10
NOMINAL_THRESHOLD = 0.05

datasets = {
    'slidetags': {
        'path': f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
        'local_proximity': False,
    },
    'merfish': {
        'path': f'{working_dir}/output/merfish/03_adata_query_merfish.h5ad',
        'local_proximity': True,
        'd_max_scale': 20,
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'drop_samples': ['CTRL_3'],
        'local_proximity': True,
        'd_max_scale': 20,
    },
}

adatas = {}
for name, cfg in datasets.items():
    adata = sc.read_h5ad(cfg['path'])
    drop = cfg.get('drop_samples', [])
    if drop:
        adata = adata[~adata.obs['sample'].isin(drop)].copy()
    adatas[name] = adata

proximity_datasets = {
    k: v for k, v in datasets.items() if v['local_proximity']}

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
os.makedirs(f'{working_dir}/figures/proximity', exist_ok=True)

local_tt = pd.read_csv(f'{working_dir}/output/proximity_local_diff.csv')
global_tt = pd.read_csv(f'{working_dir}/output/proximity_global_props.csv')
sumrank_tt = pd.read_csv(f'{working_dir}/output/proximity_local_sumrank.csv')

#endregion

#region plot — niche forests ###################################################

# per-pair effect size ± 95% CI across three contrasts:
#   MERFISH PREG vs CTRL, Xenium PREG vs CTRL, MERFISH POSTPART vs PREG.
# a niche is a biologically-defined (A-group × B-group) with a single
# consensus direction. pairs shown are concordant across MERFISH and Xenium
# PREG data with sumrank emp_p < 0.05, sorted by mean |logFC|.

COLOR_MERFISH = '#E8A628'   # Saffron
COLOR_XENIUM  = '#1FAA6B'   # Jade
STRIPE        = '#e8e8e8'

MATERNAL_CIRCUIT = [
    '069 LSX Nkx2-1 Gaba', '073 MEA-BST Sox6 Gaba',
    '085 SI-MPO-LPO Lhx8 Gaba', '101 ZI Pax6 Gaba',
    '107 DMH Hmx2 Gaba', '115 MS-SF Bsx Glut',
    '124 MPN-MPO-PVpo Hmx2 Glut', '131 LHA-AHN-PVH Otp Trh Glut',
    '318 Astro-NT NN',
]

NICHES = [
    ('Gliovascular → STR D1/D2 Gaba (co-occurrence ↑)', 'up',
     ['326 OPC NN', '333 Endo NN',
      '334 Microglia NN', '331 Peri NN'],
     ['061 STR D1 Gaba', '062 STR D2 Gaba', '063 STR D1 Sema5a Gaba']),
    ('Gliovascular → L2/3 + L4/5 IT CTX Glut (co-occurrence ↓)', 'down',
     ['334 Microglia NN', '326 OPC NN',
      '319 Astro-TE NN', '335 BAM NN', '333 Endo NN'],
     ['006 L4/5 IT CTX Glut', '007 L2/3 IT CTX Glut']),
    ('Interneurons → L4/5 IT CTX Glut (co-occurrence ↓)', 'down',
     ['051 Pvalb chandelier Gaba', '046 Vip Gaba', '049 Lamp5 Gaba'],
     ['006 L4/5 IT CTX Glut']),
    ('Vascular/immune → TRS-BAC Sln Glut (co-occurrence ↓)', 'down',
     ['331 Peri NN', '332 SMC NN', '334 Microglia NN', '335 BAM NN'],
     ['111 TRS-BAC Sln Glut']),
    ('Maternal hypothalamic-limbic circuit (co-occurrence ↑)', 'up',
     MATERNAL_CIRCUIT, MATERNAL_CIRCUIT),
]

def _prep_modality(df):
    se = (df['logFC'] / df['t']).abs().replace([np.inf, -np.inf], np.nan)
    return df.assign(se=se)[['cell_type_a', 'cell_type_b',
                             'logFC', 'se', 'P.Value']]

def _p_stars(p):
    if not pd.notna(p):
        return ''
    if p < 1e-3:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''

def _niche_rows(merged, group_a, group_b, direction):
    sign = 1 if direction == 'up' else -1
    mask = merged['cell_type_a'].isin(group_a) \
         & merged['cell_type_b'].isin(group_b) \
         & (merged['cell_type_a'] != merged['cell_type_b']) \
         & (merged['best_p'] < 0.05) \
         & (np.sign(merged['lfc_m']) == sign) \
         & (np.sign(merged['lfc_x']) == sign)
    df = merged[mask].copy()
    if df.empty:
        return df
    # when group_a == group_b, both orderings of each physical pair match;
    # dedup to one row per unordered pair (keep smaller emp_p)
    df['pair_key'] = df.apply(
        lambda r: tuple(sorted([r['cell_type_a'], r['cell_type_b']])),
        axis=1)
    df = df.sort_values('best_p').drop_duplicates('pair_key')
    df['mean_abs'] = (np.abs(df['lfc_m']) + np.abs(df['lfc_x'])) / 2
    df = df.sort_values('mean_abs', ascending=False).reset_index(drop=True)
    df['label'] = (df['cell_type_a'] + '  →  ' + df['cell_type_b']
                   + '   ' + df['best_p'].apply(_p_stars))
    return df

loc_m = _prep_modality(local_tt[
    (local_tt['contrast'] == 'PREG_vs_CTRL') &
    (local_tt['dataset'] == 'merfish')])
loc_x = _prep_modality(local_tt[
    (local_tt['contrast'] == 'PREG_vs_CTRL') &
    (local_tt['dataset'] == 'xenium')])
loc_pp = _prep_modality(local_tt[
    (local_tt['contrast'] == 'POSTPART_vs_PREG') &
    (local_tt['dataset'] == 'merfish')])
merged = (loc_m.rename(columns={
    'logFC': 'lfc_m', 'se': 'se_m', 'P.Value': 'p_m'})
    .merge(loc_x.rename(columns={
    'logFC': 'lfc_x', 'se': 'se_x', 'P.Value': 'p_x'}),
    on=['cell_type_a', 'cell_type_b'])
          .merge(sumrank_tt[sumrank_tt['contrast'] == 'PREG_vs_CTRL'][[
              'cell_type_a', 'cell_type_b', 'emp_p_up', 'emp_p_down']],
                 on=['cell_type_a', 'cell_type_b'], how='left')
          .merge(loc_pp.rename(columns={'logFC': 'lfc_pp', 'se': 'se_pp',
                                         'P.Value': 'p_pp'}),
                 on=['cell_type_a', 'cell_type_b'], how='left')
          .dropna(subset=['lfc_m', 'lfc_x']))
merged['best_p'] = merged[['emp_p_up', 'emp_p_down']].min(axis=1)

niche_info = [(title, d, _niche_rows(merged, ga, gb, d))
              for title, d, ga, gb in NICHES]
heights = [max(len(df), 1) for *_, df in niche_info]
fig_h = 0.24 * sum(heights) + 0.38 * len(niche_info) + 0.55
fig, axes = plt.subplots(
    len(niche_info), 2, figsize=(4.5, fig_h), squeeze=False,
    gridspec_kw={'height_ratios': heights,
                 'width_ratios': [2, 1],
                 'hspace': 0.25, 'wspace': 0.06})

for ax_i, (title, direction, df) in enumerate(niche_info):
    ax_l = axes[ax_i, 0]
    ax_r = axes[ax_i, 1]
    if df.empty:
        for a in (ax_l, ax_r):
            a.set_axis_off()
        ax_l.text(0.5, 0.5, 'no concordant sig. pairs',
                  transform=ax_l.transAxes, ha='center', va='center',
                  fontsize=7, color='gray')
        ax_l.set_title(title, fontsize=7.5, loc='left', pad=3)
        continue
    nrows = len(df)
    y = np.arange(nrows)
    for i in range(nrows):
        if i % 2 == 0:
            for a in (ax_l, ax_r):
                a.axhspan(i - 0.5, i + 0.5, color=STRIPE, zorder=0)

    # left: Pregnant vs Nulliparous (MERFISH + Xenium)
    ax_l.errorbar(df['lfc_m'], y - 0.18,
                  xerr=1.96 * df['se_m'].fillna(0),
                  fmt='o', markersize=3, color=COLOR_MERFISH,
                  elinewidth=0.7, capsize=1.2, linewidth=0, zorder=3,
                  label='MERFISH')
    ax_l.errorbar(df['lfc_x'], y + 0.18,
                  xerr=1.96 * df['se_x'].fillna(0),
                  fmt='o', markersize=3, color=COLOR_XENIUM,
                  elinewidth=0.7, capsize=1.2, linewidth=0, zorder=3,
                  label='Xenium')
    x_left = np.concatenate([
        (df['lfc_m'] + 1.96 * df['se_m'].fillna(0)).values,
        (df['lfc_m'] - 1.96 * df['se_m'].fillna(0)).values,
        (df['lfc_x'] + 1.96 * df['se_x'].fillna(0)).values,
        (df['lfc_x'] - 1.96 * df['se_x'].fillna(0)).values,
    ])
    xmax_l = float(np.nanmax(np.abs(x_left))) * 1.05 if len(x_left) else 1.0
    ax_l.set_xlim(-xmax_l, xmax_l)

    # right: Postpartum vs Pregnant (MERFISH only)
    has_pp = df['lfc_pp'].notna()
    if has_pp.any():
        ax_r.errorbar(df.loc[has_pp, 'lfc_pp'], y[has_pp],
                      xerr=1.96 * df.loc[has_pp, 'se_pp'].fillna(0),
                      fmt='o', markersize=3, color=COLOR_MERFISH,
                      elinewidth=0.7, capsize=1.2, linewidth=0, zorder=3)
        x_right = np.concatenate([
            (df['lfc_pp'] + 1.96 * df['se_pp'].fillna(0)).dropna().values,
            (df['lfc_pp'] - 1.96 * df['se_pp'].fillna(0)).dropna().values,
        ])
        xmax_r = float(np.nanmax(np.abs(x_right))) * 1.05 \
            if len(x_right) else 1.0
    else:
        xmax_r = 1.0
    ax_r.set_xlim(-xmax_r, xmax_r)

    for a in (ax_l, ax_r):
        a.axvline(0, c='black', linewidth=0.5, zorder=1)
        a.set_ylim(nrows - 0.5, -0.5)
        a.tick_params(labelsize=5.5, length=0, pad=1)
        for spine in a.spines.values():
            spine.set_linewidth(0.5)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    ax_l.set_yticks(y)
    ax_l.set_yticklabels(df['label'], fontsize=6)
    pp_star_labels = [_p_stars(p) if pd.notna(p) else ''
                      for p in df['p_pp']]
    ax_r.set_yticks(y)
    ax_r.set_yticklabels(pp_star_labels, fontsize=6.5,
                         fontweight='bold')
    ax_l.set_title(title, fontsize=7.5, loc='left', pad=3)

    if ax_i == len(niche_info) - 1:
        ax_l.set_xlabel('Pregnant vs Nulliparous\nlogFC (95% CI)',
                        fontsize=6.5, labelpad=2)
        ax_r.set_xlabel('Postpartum vs Pregnant\nlogFC (95% CI)',
                        fontsize=6.5, labelpad=2)

handles = [
    plt.Line2D([0], [0], marker='o', color='none',
               markerfacecolor=COLOR_MERFISH, markersize=4,
               label='MERFISH'),
    plt.Line2D([0], [0], marker='o', color='none',
               markerfacecolor=COLOR_XENIUM, markersize=4,
               label='Xenium'),
]
axes[-1, 1].legend(
    handles=handles, loc='upper right',
    bbox_to_anchor=(1.0, -0.85),
    ncol=2, fontsize=5.5, frameon=False,
    handletextpad=0.3, columnspacing=1.0,
    title='pairs: center → surround',
    title_fontsize=5.5, alignment='right')

plt.savefig(f'{working_dir}/figures/proximity/niches.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/proximity/niches.svg',
            bbox_inches='tight')
plt.close()

#endregion

#region plot — concordance scatter #############################################
# MERFISH vs Xenium logFC, all pairs in a single scatter. niche-member pairs
# are highlighted; concordant quadrants sit on the y=x diagonal.

niche_pairs = {frozenset((r['cell_type_a'], r['cell_type_b']))
               for _, _, df in niche_info if not df.empty
               for _, r in df.iterrows()}

COLOR_NICHE = '#B91C1C'
merged['in_niche'] = [frozenset((a, b)) in niche_pairs for a, b in
                      zip(merged['cell_type_a'], merged['cell_type_b'])]
bg = merged[~merged['in_niche']]
hi = merged[merged['in_niche']]

lim = float(np.nanquantile(np.abs(np.concatenate(
    [merged['lfc_m'], merged['lfc_x']])), 0.95))

fig, ax = plt.subplots(figsize=(3.6, 3.6))
ax.scatter(bg['lfc_m'], bg['lfc_x'], s=5, c='lightgray',
           alpha=0.55, linewidth=0, rasterized=True)
ax.scatter(hi['lfc_m'], hi['lfc_x'], s=20, c=COLOR_NICHE,
           linewidth=0.4, edgecolor='white', zorder=4)

ax.plot([-lim, lim], [-lim, lim], '--', c='black', linewidth=0.4,
        zorder=1)
ax.axhline(0, c='black', linewidth=0.3, zorder=1)
ax.axvline(0, c='black', linewidth=0.3, zorder=1)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_aspect('equal')
ax.set_xlabel('MERFISH logFC', fontsize=7)
ax.set_ylabel('Xenium logFC', fontsize=7)
ax.set_title('PREG vs CTRL cross-modality logFC',
             fontsize=8, loc='left', pad=4)
ax.tick_params(labelsize=6)
ax.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='none',
               markerfacecolor=COLOR_NICHE, markersize=5,
               markeredgecolor='white', markeredgewidth=0.4,
               label=f'niche pair ({len(hi)})'),
    plt.Line2D([0], [0], marker='o', color='none',
               markerfacecolor='lightgray', markersize=4,
               label=f'other ({len(bg):,})'),
], fontsize=5.5, loc='lower right', frameon=False,
   handletextpad=0.3)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(f'{working_dir}/figures/proximity/concordance.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/proximity/concordance.svg',
            bbox_inches='tight')
plt.close()

#endregion

#region plot — radius visualization ############################################
# per-sample d_max sanity check: one overview panel per dataset plus zoomed
# panels showing neighbors within d_max for 5 random exemplar cells.

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
    ax.set_xticks([]); ax.set_yticks([])
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
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/proximity/radius.png',
            dpi=200, bbox_inches='tight')
plt.close()

#endregion

#region plot — local proximities heatmap ######################################
# full (A × B) proximity shift matrix for cell types present in both MERFISH
# and Xenium. two panels:
#   1) PREG vs CTRL meta (mean logFC across MERFISH + Xenium; sumrank glyphs)
#   2) POSTPART vs PREG MERFISH only (per-dataset logFC; per-dataset glyphs)
# each panel: single global-proportion column | gap | A × B local matrix.

def get_type(ct):
    if 'Glut' in ct:
        return 'Glut'
    if any(x in ct for x in ['Gaba', 'IMN', 'Chol']):
        return 'Gaba'
    return 'NN'


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


def _sig_glyph(fdr, p):
    if pd.notna(fdr) and fdr < FDR_THRESHOLD:
        return '*'
    if pd.notna(p) and p < NOMINAL_THRESHOLD:
        return '•'
    return ''


def _draw_blocks(ax, rs, cs, lw=0.8):
    for r0, r1 in rs:
        for c0, c1 in cs:
            ax.add_patch(plt.Rectangle(
                (c0, r0), c1 - c0, r1 - r0, fill=False,
                edgecolor='black', linewidth=lw, zorder=5))


# --- axis layout: cell types shared by merfish + xenium -----------------------
common_cts = (set(adatas['merfish'].obs[cell_type_col].unique()) &
              set(adatas['xenium'].obs[cell_type_col].unique()))
type_order = ['Glut', 'Gaba', 'NN']
display = sorted(common_cts,
                 key=lambda c: (type_order.index(get_type(c)), c))
ab_gapped, ab_labels = _insert_gaps(display)
ab_map = {ct: i for i, ct in enumerate(ab_gapped) if ct is not None}
n_ab = len(ab_gapped)
LOCAL_OFF = 2
n_cols = 1 + 1 + n_ab

gap_idx = [i for i, x in enumerate(ab_gapped) if x is None]
segs, start = [], 0
for g in gap_idx:
    segs.append((start, g))
    start = g + 1
segs.append((start, n_ab))


def _build_matrix(local_mat, local_sig, global_vec, global_sig):
    mat = np.full((n_ab, n_cols), np.nan)
    sig = np.full((n_ab, n_cols), '', dtype=object)
    for a, i in ab_map.items():
        if a in global_vec:
            mat[i, 0] = global_vec[a]
            sig[i, 0] = global_sig.get(a, '')
        for b, j in ab_map.items():
            if (a, b) in local_mat:
                mat[i, LOCAL_OFF + j] = local_mat[(a, b)]
                sig[i, LOCAL_OFF + j] = local_sig.get((a, b), '')
    return mat, sig


# --- panel 1: PREG vs CTRL meta (merfish + xenium) ----------------------------
loc_preg = local_tt[(local_tt['contrast'] == 'PREG_vs_CTRL') &
                    local_tt['dataset'].isin(['merfish', 'xenium'])]
preg_local = (loc_preg.groupby(['cell_type_a', 'cell_type_b'])['logFC']
              .mean().to_dict())
sr_preg = sumrank_tt[sumrank_tt['contrast'] == 'PREG_vs_CTRL']
preg_local_sig = {
    (r['cell_type_a'], r['cell_type_b']): _sig_glyph(
        min(r['emp_fdr_up'], r['emp_fdr_down']),
        min(r['emp_p_up'], r['emp_p_down']))
    for _, r in sr_preg.iterrows()}
glb_preg = global_tt[(global_tt['contrast'] == 'PREG_vs_CTRL') &
                     global_tt['dataset'].isin(['merfish', 'xenium'])]
preg_global = glb_preg.groupby('cell_type')['logFC'].mean().to_dict()
preg_global_sig = {}
for ct, sub in glb_preg.groupby('cell_type'):
    if (sub['adj.P.Val'] < FDR_THRESHOLD).all():
        preg_global_sig[ct] = '*'
    elif (sub['P.Value'] < NOMINAL_THRESHOLD).any():
        preg_global_sig[ct] = '•'

# --- panel 2: POSTPART vs PREG MERFISH only -----------------------------------
loc_pp = local_tt[(local_tt['contrast'] == 'POSTPART_vs_PREG') &
                  (local_tt['dataset'] == 'merfish')]
pp_local = {(r['cell_type_a'], r['cell_type_b']): r['logFC']
            for _, r in loc_pp.iterrows()}
pp_local_sig = {(r['cell_type_a'], r['cell_type_b']): _sig_glyph(
                    r['adj.P.Val'], r['P.Value'])
                for _, r in loc_pp.iterrows()}
glb_pp = global_tt[(global_tt['contrast'] == 'POSTPART_vs_PREG') &
                   (global_tt['dataset'] == 'merfish')]
pp_global = {r['cell_type']: r['logFC'] for _, r in glb_pp.iterrows()}
pp_global_sig = {r['cell_type']: _sig_glyph(r['adj.P.Val'], r['P.Value'])
                 for _, r in glb_pp.iterrows()}

panels = [
    ('PREG vs CTRL\nmeta (MERFISH + Xenium)',
     _build_matrix(preg_local, preg_local_sig,
                   preg_global, preg_global_sig)),
    ('POSTPART vs PREG\nMERFISH',
     _build_matrix(pp_local, pp_local_sig,
                   pp_global, pp_global_sig)),
]

# --- plot ---------------------------------------------------------------------
abs_vals = np.concatenate([
    np.abs(mat[~np.isnan(mat)]) for _, (mat, _) in panels
    if (~np.isnan(mat)).any()])
vmax = max(float(np.percentile(abs_vals, 98))
           if abs_vals.size else 0.0, 0.1)
vmin = -vmax

CELL_SIZE = 0.11
fig_w = CELL_SIZE * n_cols * len(panels) + 3.0
fig_h = CELL_SIZE * n_ab + 1.8

fig, axes = plt.subplots(
    1, len(panels), figsize=(fig_w, fig_h), squeeze=False,
    gridspec_kw={'wspace': 0.25})

last_im = None
for pi, (title, (mat, sig)) in enumerate(panels):
    ax = axes[0, pi]
    im = ax.pcolormesh(mat, cmap='PRGn', vmin=vmin, vmax=vmax)
    last_im = im
    for i in range(n_ab):
        if ab_gapped[i] is None:
            continue
        if not np.isnan(mat[i, 0]):
            ax.add_patch(plt.Rectangle(
                (0, i), 1, 1, fill=False,
                edgecolor='black', linewidth=0.15))
        if sig[i, 0]:
            ax.text(0.5, i + 0.5, sig[i, 0], ha='center',
                    va='center', color='white',
                    fontsize=3.5, fontweight='bold')
        for j in range(n_ab):
            if ab_gapped[j] is None:
                continue
            jj = LOCAL_OFF + j
            if not np.isnan(mat[i, jj]):
                ax.add_patch(plt.Rectangle(
                    (jj, i), 1, 1, fill=False,
                    edgecolor='black', linewidth=0.15))
            if sig[i, jj]:
                ax.text(jj + 0.5, i + 0.5, sig[i, jj],
                        ha='center', va='center', color='white',
                        fontsize=3.5, fontweight='bold')

    _draw_blocks(ax, segs, [(0, 1)])
    _draw_blocks(ax, segs,
                 [(LOCAL_OFF + c0, LOCAL_OFF + c1) for c0, c1 in segs])

    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_ab, 0)
    ax.set_aspect('equal')
    x_ticks = [0.5] + [LOCAL_OFF + j + 0.5 for j in range(n_ab)]
    x_labels = ['Global\nproportions'] + ab_labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=3.5)
    ax.set_yticks(np.arange(n_ab) + 0.5)
    if pi == 0:
        ax.set_yticklabels(ab_labels, fontsize=3.5)
        ax.set_ylabel('Center cell type', fontsize=6, labelpad=3)
    else:
        ax.set_yticklabels([])
    ax.tick_params(length=0, pad=1)
    ax.set_title(title, fontsize=7, pad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)

if last_im is not None:
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.25,
                        pad=0.015, aspect=12)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label('logFC', fontsize=6)

plt.savefig(f'{working_dir}/figures/proximity/local_heatmap.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/proximity/local_heatmap.svg',
            bbox_inches='tight')
plt.close()

#endregion
