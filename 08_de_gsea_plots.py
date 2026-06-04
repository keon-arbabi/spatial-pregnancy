#region setup #################################################################

import os

import numpy as np
import polars as pl
import scanpy as sc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy'
de_suffix = ''            # '' for subclass; '_class' for class-level
cell_type_col = 'subclass'

#endregion

#region meta de barplot #######################################################

EMP_P_THRESH = 0.05
MIN_DEGS_TO_SHOW = 5

seismic_cmap = plt.get_cmap('seismic')
UP_COLOR = seismic_cmap(0.9)
DN_COLOR = seismic_cmap(0.1)

sr_meta = pl.read_csv(
    f'{working_dir}/output/de/sumrank_results{de_suffix}.csv')

# major class label for cell-type ordering
def get_type(ct):
    if 'Glut' in ct:
        return 'Glut'
    elif any(x in ct for x in ['Gaba', 'IMN', 'Chol']):
        return 'Gaba'
    return 'NN'

contrasts = ['PREG_vs_CTRL', 'POSTPART_vs_PREG', 'POSTPART_vs_CTRL']
contrast_titles = {
    'PREG_vs_CTRL': 'Pregnant vs\nNulliparous',
    'POSTPART_vs_PREG': 'Postpartum vs\nPregnant',
    'POSTPART_vs_CTRL': 'Postpartum vs\nNulliparous',
}

deg_counts = sr_meta\
    .group_by(['cell_type', 'contrast'])\
    .agg(
        ((pl.col('emp_p_up') < EMP_P_THRESH) &
         (pl.col('D') == 3)).sum().alias('up_d3'),
        ((pl.col('emp_p_up') < EMP_P_THRESH) &
         (pl.col('D') == 2)).sum().alias('up_d2'),
        ((pl.col('emp_p_down') < EMP_P_THRESH) &
         (pl.col('D') == 3)).sum().alias('dn_d3'),
        ((pl.col('emp_p_down') < EMP_P_THRESH) &
         (pl.col('D') == 2)).sum().alias('dn_d2'))

ct_totals = deg_counts\
    .group_by('cell_type')\
    .agg((pl.sum('up_d3') + pl.sum('up_d2') +
          pl.sum('dn_d3') + pl.sum('dn_d2')).alias('total'))\
    .filter(pl.col('total') >= MIN_DEGS_TO_SHOW)\
    .with_columns(
        pl.col('cell_type').map_elements(
            get_type, return_dtype=pl.Utf8).alias('type'))\
    .with_columns(
        pl.col('type').replace_strict(
            {'Glut': 0, 'Gaba': 1, 'NN': 2}).alias('type_order'))\
    .sort(['type_order', 'total'], descending=[False, True])

groups = ct_totals.group_by('type', maintain_order=True).all()
major_types = groups['type'].to_list()
height_ratios = groups['cell_type'].list.len().to_list()

BAR_H = 0.55

fig = plt.figure(figsize=(7, 12))
outer_gs = gridspec.GridSpec(
    len(major_types), len(contrasts), figure=fig,
    height_ratios=height_ratios, hspace=0.06, wspace=0.04)

for i, group_type in enumerate(major_types):
    group_cts = groups.filter(
        pl.col('type') == group_type)['cell_type'].explode().to_list()

    for j, contrast in enumerate(contrasts):
        ax = fig.add_subplot(outer_gs[i, j])

        sub = deg_counts.filter(
            (pl.col('contrast') == contrast) &
            (pl.col('cell_type').is_in(group_cts)))
        ct_d = {r['cell_type']: r for r in sub.to_dicts()}

        all_up, all_dn = [], []
        lw = 0.4
        for idx, ct in enumerate(group_cts):
            r = ct_d.get(ct, {})
            u3 = r.get('up_d3', 0)
            u2 = r.get('up_d2', 0)
            n3 = r.get('dn_d3', 0)
            n2 = r.get('dn_d2', 0)

            # UP: D=3 (full alpha) then D=2 (lower alpha) rightward
            ax.barh(idx, u3, height=BAR_H, align='center',
                    facecolor=UP_COLOR, edgecolor=UP_COLOR,
                    alpha=1.0, linewidth=lw, zorder=5)
            ax.barh(idx, u2, left=u3, height=BAR_H, align='center',
                    facecolor=UP_COLOR, edgecolor=UP_COLOR,
                    alpha=0.45, linewidth=lw, zorder=5)

            # DOWN: D=3 then D=2 leftward
            ax.barh(idx, -n3, height=BAR_H, align='center',
                    facecolor=DN_COLOR, edgecolor=DN_COLOR,
                    alpha=1.0, linewidth=lw, zorder=5)
            ax.barh(idx, -n2, left=-n3, height=BAR_H, align='center',
                    facecolor=DN_COLOR, edgecolor=DN_COLOR,
                    alpha=0.45, linewidth=lw, zorder=5)

            all_up.append(u3 + u2)
            all_dn.append(n3 + n2)

        xlim = max(max(all_up + [1]), max(all_dn + [1])) * 1.25

        ax.axvline(0, color='grey', linewidth=0.5, zorder=0)
        ax.grid(True, 'major', 'y', ls='-', lw=0.3,
                c='lightgray', zorder=0)
        ax.set_xlim(-xlim, xlim)
        ax.set_yticks(range(len(group_cts)))
        ax.set_ylim(len(group_cts) - 0.5, -0.5)
        ax.tick_params(length=0, labelsize=7)

        if j == 0:
            ax.set_yticklabels(group_cts, fontsize=7.5)
            ax.tick_params(axis='y', pad=8)
        else:
            ax.set_yticklabels([])

        if i == 0:
            ax.set_title(contrast_titles[contrast], fontsize=9, pad=6)

        if i == len(major_types) - 1:
            ax.set_xlabel(f'meta DEGs (emp_p<{EMP_P_THRESH})',
                          fontsize=8.5)
        else:
            ax.set_xticklabels([])

legend_elements = [
    Patch(facecolor=UP_COLOR, edgecolor=UP_COLOR,
          alpha=1.0, linewidth=0.4, label='Upregulated'),
    Patch(facecolor=DN_COLOR, edgecolor=DN_COLOR,
          alpha=1.0, linewidth=0.4, label='Downregulated'),
    Patch(facecolor='lightgray', edgecolor='black',
          alpha=1.0, linewidth=0.4, label='D=3 (3 platforms)'),
    Patch(facecolor='lightgray', edgecolor='black',
          alpha=0.45, linewidth=0.4, label='D=2 (2 platforms)'),
]
fig.legend(handles=legend_elements, loc='lower right',
           bbox_to_anchor=(0.98, 0.02), fontsize=7,
           frameon=False, ncol=2)

os.makedirs(f'{working_dir}/figures/de-gsea', exist_ok=True)
plt.savefig(f'{working_dir}/figures/de-gsea/fig2A_ndeg_barplot.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/de-gsea/fig2A_ndeg_barplot.svg',
            bbox_inches='tight')
plt.close()

#endregion

#region candidate cards #######################################################

# theme membership from curated FINAL csv (membership only; all stats below
# come from sumrank_results.csv + de_results.csv)
theme_csv = pl.read_csv(
    f'{working_dir}/output/reports/figure_themes_FINAL.csv')

THEME_LABELS = {
    'T1_NEURONAL_SUPPRESSION':
        'T1  Neuronal / synaptic program suppression',
    'T2_CYTOKINE_AXIS_BRAKE':
        'T2  LIFR / IL6ST cytokine axis + microglial activation + Cd47 brake',
    'T3_APOE_CHOLESTEROL':
        'T3  Brain Apoe + cholesterol biosynthesis (PREG↓ → POSTPART↑)',
    'T4_VASCULATURE_OPC':
        'T4  Vasculature + OPC + astrocyte structural activation',
    'T5_MATERNAL_HUB':
        'T5  MPOA hormonal hub + maternal-circuit anchors',
}

# styling
PLATFORMS = ['slidetags', 'merfish', 'xenium']
PLATFORM_COLORS = {
    'slidetags': '#0B6FA8',  # deep cerulean
    'xenium':    '#1FAA6B',  # jade
    'merfish':   '#E8A628',  # saffron
}
D_COLORS = {2: '#888888', 3: '#000000'}

EMP_P_CAND_THRESH = 0.05  # legacy: kept for stars labeling

forest_contrasts = ['PREG_vs_CTRL']
contrast_short = {'PREG_vs_CTRL': 'PREG vs CTRL'}

cond_short = {'CTRL': 'Null', 'PREG': 'Preg', 'POSTPART': 'Post'}
sp_ds_labels = {'xenium': 'Xenium', 'merfish': 'MERFISH',
                'slidetags': 'Slide-tags'}

# curated 30: 6 genes per theme, ordered by row position
THEME_GENES = {
    'T1_NEURONAL_SUPPRESSION':
        ['Homer1', 'Gap43', 'Gpr88', 'Dusp7', 'Arc', 'Fkbp5'],
    'T2_CYTOKINE_AXIS_BRAKE':
        ['Lifr', 'Il6st', 'Stat3', 'Cd47', 'Mertk', 'Prl'],
    'T3_APOE_CHOLESTEROL':
        ['Apoe', 'Idi1', 'Hmgcr', 'Hmgcs1', 'Sqle', 'Msmo1',
         'Srebf1', 'Srebf2'],
    'T4_VASCULATURE_OPC':
        ['Kdr', 'Vegfa', 'Pdgfra', 'Pecam1', 'Gjb6', 'Flt1',
        'Plvap', 'Tgfb2', 'Olig2', 'Top2a', 'Mfsd2a', 'Trem2'],
    'T5_MATERNAL_HUB':
        ['Crh', 'Brs3', 'Nts', 'Pydn' 'Calcr', 'Foxp2'],
}
# layout grid: each cell = (theme_key, gene_slice, column_title)
# T4 (12 genes) is split across two columns in row 1: first half under T2,
# second half under T3. Asymmetric grids are supported (a column may have
# fewer genes than its row's max, leaving trailing cells empty).
THEME_LAYOUT = [
    [('T1_NEURONAL_SUPPRESSION', slice(0, None),
      'Synaptic suppression'),
     ('T2_CYTOKINE_AXIS_BRAKE',  slice(0, None),
      'LIFR axis / Cd47 brake'),
     ('T3_APOE_CHOLESTEROL',     slice(0, None),
      'Apoe / cholesterol')],
    [('T5_MATERNAL_HUB',         slice(0, None),
      'MPOA / maternal'),
     ('T4_VASCULATURE_OPC',      slice(0, 6),
      'Vascular / OPC'),
     ('T4_VASCULATURE_OPC',      slice(6, None),
      'Vascular / OPC (cont.)')],
]

TICK_FS = 4.5
TITLE_FS = 6

# spine width helper
def format_ax(ax, lw=0.5):
    for spine in ax.spines.values():
        spine.set_linewidth(lw)

# percentile-based color range; falls back to (0, 1) if empty
def vrange_pctl(arr, lo=5, hi=95):
    if len(arr) == 0:
        return 0, 1
    return np.percentile(arr, lo), np.percentile(arr, hi)

# load DE + meta tables
sr_cand = pl.read_csv(
    f'{working_dir}/output/de/sumrank_results{de_suffix}.csv')
de_full = pl.read_csv(
    f'{working_dir}/output/de/de_results{de_suffix}.csv')

# load spatial adatas (xenium + merfish + slidetags)
sp_adatas = {}
for ds in ['xenium', 'merfish', 'slidetags']:
    a = sc.read_h5ad(
        f'{working_dir}/output/{ds}/03_adata_query_{ds}.h5ad')
    if 'gene_symbol' in a.var.columns:
        a.var.index = a.var['gene_symbol']
        a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4)
    if ds == 'xenium':
        a = a[a.obs['sample'] != 'CTRL_3'].copy()
    sp_adatas[ds] = a

sp_coords = {}
for ds, a in sp_adatas.items():
    xv = a.obs['x_ffd'].values
    yv = a.obs['y_ffd'].values
    fov_half = max(np.ptp(xv), np.ptp(yv)) / 2 * 1.05
    sp_coords[ds] = dict(
        x=xv, y=yv,
        cond=a.obs['condition'].values,
        cell_type=a.obs[cell_type_col].values,
        midline=(xv.min() + xv.max()) / 2,
        fov_cx=(xv.min() + xv.max()) / 2,
        fov_cy=(yv.min() + yv.max()) / 2,
        fov_half=fov_half)

# significance stars from emp_p
def stars_for(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

# shortened cell-type label: drop middle TF tokens, keep number/region/class
def short_ct(ct):
    parts = ct.split(' ')
    if len(parts) <= 3:
        return ct
    return f"{parts[0]} {parts[1]} {parts[-1]}"

# pick 1-2 spatial platforms for a gene (CTRL|PREG only): prefer
# xenium+merfish; slidetags is included as the 2nd panel only when D=2 is
# the dominant pattern (i.e., one of xen/mer is missing for the gene)
def pick_spatial_pair(gene, gd):
    has_data = {p: False for p in PLATFORMS}
    for r in gd['rows']:
        cd = r['contrasts'].get('PREG_vs_CTRL')
        if cd is None:
            continue
        for p in PLATFORMS:
            if not np.isnan(cd['plat'][p]['lfc']):
                has_data[p] = True
    in_adata = {p: gene in sp_adatas[p].var_names for p in PLATFORMS}
    candidates = [p for p in ['xenium', 'merfish', 'slidetags']
                  if has_data[p] and in_adata[p]]
    return candidates[:2]

# build per-gene data restricted to cells listed in the FINAL csv for
# this (gene, theme); stats come from sumrank_results.csv + de_results.csv.
# carveout cells (T5 MPOA) are flagged so the renderer can mark them.
def build_gene_card_data(gene, cells_with_reason):
    if not cells_with_reason:
        return None
    sr_g = sr_cand.filter(pl.col('gene') == gene)
    cts_in = [c for c, _ in cells_with_reason]
    reason_lookup = {c: r for c, r in cells_with_reason}

    rows = []
    for ct in cts_in:
        ctr = 'PREG_vs_CTRL'
        plat_d = {p: dict(lfc=np.nan, lci=np.nan, uci=np.nan)
                  for p in PLATFORMS}
        de_rows = de_full.filter(
            (pl.col('contrast') == ctr) &
            (pl.col('gene') == gene) &
            (pl.col('cell_type') == ct))
        for de_r in de_rows.iter_rows(named=True):
            if de_r['dataset'] in plat_d:
                plat_d[de_r['dataset']] = dict(
                    lfc=de_r['logFC'],
                    lci=de_r['LCI'], uci=de_r['UCI'])
        valid_lfc = [plat_d[p]['lfc'] for p in PLATFORMS
                     if not np.isnan(plat_d[p]['lfc'])]
        meta_lfc = np.mean(valid_lfc) if valid_lfc else np.nan

        srr = sr_g.filter((pl.col('contrast') == ctr) &
                          (pl.col('cell_type') == ct))
        if srr.height:
            sr_row = srr.to_dicts()[0]
            ep = min(sr_row['emp_p_up'], sr_row['emp_p_down'])
            D = int(sr_row['D'])
            stars = stars_for(ep) if ep < EMP_P_CAND_THRESH else ''
        else:
            ep, D, stars = np.nan, 0, ''

        rows.append(dict(
            cell_type=ct,
            is_carveout=(reason_lookup[ct] == 'carveout'),
            contrasts={ctr: dict(
                plat=plat_d, D=D, meta_lfc=meta_lfc,
                emp_p=ep, stars=stars)}))

    # group by class (Glut, Gaba, NN); within class sort by |meta_lfc|
    class_rank = {'Glut': 0, 'Gaba': 1, 'NN': 2}

    def _abs_meta(row):
        cd = row['contrasts'].get('PREG_vs_CTRL')
        if cd and not np.isnan(cd['meta_lfc']):
            return abs(cd['meta_lfc'])
        return 0.0

    rows.sort(key=lambda r: (
        class_rank.get(get_type(r['cell_type']), 3),
        -_abs_meta(r)))
    cts = [r['cell_type'] for r in rows]

    sp_pair = pick_spatial_pair(gene, dict(rows=rows, cell_types=cts))
    sp_data, sp_vranges = [], []
    ct_set = set(cts)
    for ds in sp_pair:
        c = sp_coords[ds]
        a = sp_adatas[ds]
        if gene not in a.var_names:
            sp_data.append(None)
            sp_vranges.append((0, 1))
            continue
        expr = np.log2(np.asarray(
            a[:, gene].X.toarray()).flatten() + 1)
        ct_mask = np.isin(c['cell_type'], list(ct_set))
        mask = ct_mask & (
            ((c['cond'] == 'CTRL') & (c['x'] < c['midline'])) |
            ((c['cond'] == 'PREG') & (c['x'] >= c['midline'])))
        sd = dict(
            x=c['x'][mask], y=c['y'][mask],
            expr=expr[mask], midline=c['midline'],
            fov_cx=c['fov_cx'], fov_cy=c['fov_cy'],
            fov_half=c['fov_half'])
        sp_data.append(sd)
        nz = sd['expr'][sd['expr'] > 0]
        sp_vranges.append(vrange_pctl(nz))

    return dict(cell_types=cts, rows=rows,
                sp_pair=sp_pair, sp_data=sp_data, sp_vranges=sp_vranges)

# per-card free x-range from the 2/98 percentile of CIs + meta logFCs
def forest_xrange(gd):
    vals = []
    for r in gd['rows']:
        cd = r['contrasts'].get('PREG_vs_CTRL')
        if cd is None:
            continue
        for p in PLATFORMS:
            d = cd['plat'][p]
            for k in ('lci', 'uci', 'lfc'):
                v = d.get(k, np.nan)
                if not np.isnan(v):
                    vals.append(v)
        if not np.isnan(cd['meta_lfc']):
            vals.append(cd['meta_lfc'])
    if not vals:
        return (-1.0, 1.0)
    lo = float(np.percentile(vals, 2))
    hi = float(np.percentile(vals, 98))
    # always include 0 (the reference axvline)
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    if hi - lo < 0.4:
        mid = (hi + lo) / 2
        lo, hi = mid - 0.2, mid + 0.2
    pad = (hi - lo) * 0.10
    return (lo - pad, hi + pad)

# draw the (PREG_vs_CTRL) forest into ax; thin lines, small markers
def draw_forest(ax, gd, show_xlabel=False):
    n = len(gd['cell_types'])
    ax.set_ylim(n - 0.5, -0.5)
    ax.axvline(0, color='grey', lw=0.4, zorder=1)

    lo, hi = forest_xrange(gd)
    ax.set_xlim(lo, hi)

    JIT = 0.18
    pos = {'slidetags': -JIT, 'merfish': 0, 'xenium': +JIT}
    yticks, ylabels = [], []
    for i, r in enumerate(gd['rows']):
        cd = r['contrasts'].get('PREG_vs_CTRL')
        stars = cd['stars'] if cd else ''
        ct_label = f"{short_ct(r['cell_type'])} {stars}".strip()
        yticks.append(i)
        ylabels.append(ct_label)
        if cd is None:
            continue
        for p in PLATFORMS:
            d = cd['plat'][p]
            lci = d.get('lci', np.nan)
            uci = d.get('uci', np.nan)
            if np.isnan(lci) or np.isnan(uci):
                continue
            yp = i + pos[p]
            ax.hlines(yp, lci, uci,
                      color=PLATFORM_COLORS[p], lw=0.9, zorder=3)
        if not np.isnan(cd['meta_lfc']) and cd['D'] >= 2:
            ax.plot(cd['meta_lfc'], i, 'D',
                    mfc=D_COLORS[cd['D']], mec='black',
                    mew=0.3, ms=2.8, zorder=5)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=TICK_FS)
    ax.tick_params(axis='y', length=0, pad=1.0)
    ax.tick_params(axis='x', labelsize=TICK_FS, length=1.0, pad=1)
    if show_xlabel:
        ax.set_xlabel('logFC', fontsize=TICK_FS, labelpad=0.5)
    format_ax(ax)

# build the combined figure from THEME_LAYOUT (asymmetric grid supported)
def build_combined_figure():
    # build cards once per (theme_key, gene); same theme_key may appear in
    # multiple cells (e.g. T4 split across two columns)
    cards = {}
    for theme_row in THEME_LAYOUT:
        for theme_key, gene_slice, _ in theme_row:
            if any(k == theme_key for k in cards):
                pass  # ensure all genes built below
            sub = theme_csv.filter(pl.col('theme') == theme_key)
            for g in THEME_GENES[theme_key]:
                if (theme_key, g) in cards:
                    continue
                cells_with_reason = [
                    (r['cell_type'], r['reason'])
                    for r in sub.filter(pl.col('gene') == g)
                                .iter_rows(named=True)
                ]
                gd = build_gene_card_data(g, cells_with_reason)
                if gd is not None and gd['rows']:
                    cards[(theme_key, g)] = gd

    # resolve gene lists per cell
    layout_genes = []  # [tr_idx][col_idx] -> list of gene names
    for theme_row in THEME_LAYOUT:
        layout_genes.append(
            [THEME_GENES[tk][gs] for tk, gs, _ in theme_row])

    n_theme_rows = len(THEME_LAYOUT)
    n_theme_cols = max(len(r) for r in THEME_LAYOUT)

    # layout (inches)
    LABEL_W = 0.68
    FOREST_W = 0.55
    SP_GAP = 0.02
    SP_W = 0.55
    SP_GAP2 = 0.02
    THEME_GAP = 0.18
    M_L = 0.10
    M_R = 0.04
    M_T = 0.25
    M_B = 0.65            # 3-row legend at bottom
    BLOCK_GAP = 0.40

    CARD_W = LABEL_W + FOREST_W + SP_GAP + SP_W + SP_GAP2 + SP_W

    ROW_H_PER_CELL = 0.085
    ROW_GAP = 0.09
    GENE_TITLE_H = 0.11
    MIN_FOREST_H = 0.36

    # per-row heights (per theme block) — max cells across cols at row gri
    block_row_heights = []
    block_n_rows = []
    block_heights = []
    for tr_idx, theme_row in enumerate(THEME_LAYOUT):
        cols_genes = layout_genes[tr_idx]
        n_rows_block = max(len(g) for g in cols_genes)
        rh_list = []
        for gri in range(n_rows_block):
            m = 1
            for col_idx, (theme_key, _, _) in enumerate(theme_row):
                cgenes = cols_genes[col_idx]
                if gri >= len(cgenes):
                    continue
                gene = cgenes[gri]
                if (theme_key, gene) in cards:
                    m = max(m, len(cards[(theme_key, gene)]['rows']))
            forest_h = max(m * ROW_H_PER_CELL, MIN_FOREST_H)
            content_h = max(forest_h, SP_W)
            rh_list.append(content_h + GENE_TITLE_H)
        block_n_rows.append(n_rows_block)
        block_row_heights.append(rh_list)
        block_heights.append(sum(rh_list) + (n_rows_block - 1) * ROW_GAP)

    fig_w = (M_L + n_theme_cols * CARD_W
             + (n_theme_cols - 1) * THEME_GAP + M_R)
    fig_h = (M_T + sum(block_heights)
             + (n_theme_rows - 1) * BLOCK_GAP + M_B)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # block tops
    block_top_ys = []
    cursor = fig_h - M_T
    for tr_idx in range(n_theme_rows):
        block_top_ys.append(cursor)
        cursor -= block_heights[tr_idx]
        if tr_idx < n_theme_rows - 1:
            cursor -= BLOCK_GAP

    # theme headers per cell
    for tr_idx, theme_row in enumerate(THEME_LAYOUT):
        title_y = block_top_ys[tr_idx] + 0.20
        for col_idx, (_, _, title) in enumerate(theme_row):
            x_card = M_L + col_idx * (CARD_W + THEME_GAP)
            cx = x_card + LABEL_W + (CARD_W - LABEL_W) / 2
            fig.text(cx / fig_w, title_y / fig_h, title,
                     ha='center', va='top',
                     fontsize=TITLE_FS + 0.5)

    # cards
    for tr_idx, theme_row in enumerate(THEME_LAYOUT):
        rh_list = block_row_heights[tr_idx]
        n_rows_block = block_n_rows[tr_idx]
        block_top = block_top_ys[tr_idx]
        cols_genes = layout_genes[tr_idx]

        for gri in range(n_rows_block):
            rh = rh_list[gri]
            row_top = (block_top - sum(rh_list[:gri]) - gri * ROW_GAP)
            row_bot = row_top - rh
            content_h = rh - GENE_TITLE_H

            for col_idx, (theme_key, _, _) in enumerate(theme_row):
                cgenes = cols_genes[col_idx]
                if gri >= len(cgenes):
                    continue
                g = cgenes[gri]
                x_card = M_L + col_idx * (CARD_W + THEME_GAP)

                # gene title
                tx = x_card + LABEL_W + FOREST_W / 2
                ty = row_top - 0.02
                fig.text(tx / fig_w, ty / fig_h, g,
                         ha='center', va='top',
                         fontsize=TITLE_FS - 0.5,
                         fontstyle='italic')

                if (theme_key, g) not in cards:
                    continue
                gd = cards[(theme_key, g)]
                n = len(gd['rows'])
                forest_h = max(n * ROW_H_PER_CELL, MIN_FOREST_H)
                forest_bot = row_bot + (content_h - forest_h) / 2
                sp_bot = row_bot + (content_h - SP_W) / 2

                fx = x_card + LABEL_W
                ax_f = fig.add_axes([fx / fig_w, forest_bot / fig_h,
                                     FOREST_W / fig_w, forest_h / fig_h])
                draw_forest(ax_f, gd,
                            show_xlabel=(gri == n_rows_block - 1))

                sp_pair = gd['sp_pair']
                for si in range(2):
                    if si >= len(sp_pair):
                        continue
                    sx = (x_card + LABEL_W + FOREST_W + SP_GAP
                          + si * (SP_W + SP_GAP2))
                    ax = fig.add_axes([sx / fig_w, sp_bot / fig_h,
                                       SP_W / fig_w, SP_W / fig_h])
                    ds = sp_pair[si]
                    ax.set_title(sp_ds_labels[ds],
                                 fontsize=TICK_FS, pad=0.5)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_facecolor('black')
                    format_ax(ax)
                    sd = gd['sp_data'][si]
                    if sd is None or len(sd['x']) == 0:
                        continue
                    vmin_sp, vmax_sp = gd['sp_vranges'][si]
                    order = np.argsort(sd['expr'])
                    sp_size = 0.18 if ds == 'slidetags' else 0.10
                    ax.scatter(sd['x'][order], sd['y'][order],
                               c=sd['expr'][order], cmap='viridis',
                               s=sp_size,
                               vmin=vmin_sp, vmax=vmax_sp,
                               linewidths=0, rasterized=True)
                    c = sp_coords[ds]
                    ax.plot([sd['midline'], sd['midline']],
                            [c['y'].min(), c['y'].max()],
                            color='white', lw=0.3, zorder=2)
                    ax.set_xlim(sd['fov_cx'] - sd['fov_half'],
                                sd['fov_cx'] + sd['fov_half'])
                    ax.set_ylim(sd['fov_cy'] - sd['fov_half'],
                                sd['fov_cy'] + sd['fov_half'])
                    ax.set_aspect('equal')
                    ax.text(0.25, 0.97, 'Null',
                            transform=ax.transAxes,
                            ha='center', va='top', color='white',
                            fontsize=TICK_FS - 0.5, zorder=4)
                    ax.text(0.75, 0.97, 'Preg',
                            transform=ax.transAxes,
                            ha='center', va='top', color='white',
                            fontsize=TICK_FS - 0.5, zorder=4)

    # 3-row legend stack at bottom: datasets / meta / asterisks
    from matplotlib.legend import Legend
    ds_handles = [
        Line2D([0], [0], color=PLATFORM_COLORS['slidetags'],
               lw=1.4, label='Slide-tags'),
        Line2D([0], [0], color=PLATFORM_COLORS['merfish'],
               lw=1.4, label='MERFISH'),
        Line2D([0], [0], color=PLATFORM_COLORS['xenium'],
               lw=1.4, label='Xenium'),
    ]
    meta_handles = [
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor=D_COLORS[3], markeredgecolor='black',
               markeredgewidth=0.3, markersize=4, label='meta D = 3'),
        Line2D([0], [0], marker='D', color='w',
               markerfacecolor=D_COLORS[2], markeredgecolor='black',
               markeredgewidth=0.3, markersize=4, label='meta D = 2'),
    ]
    star_handles = [
        Line2D([0], [0], color='none', linestyle='', marker='',
               label='*   emp_p < 0.05'),
        Line2D([0], [0], color='none', linestyle='', marker='',
               label='**   emp_p < 0.01'),
        Line2D([0], [0], color='none', linestyle='', marker='',
               label='***  emp_p < 0.001'),
    ]

    ds_y    = 0.48 / fig_h   # top legend row (datasets)
    meta_y  = 0.30 / fig_h   # middle (meta D=2/3)
    star_y  = 0.12 / fig_h   # bottom (asterisks)

    leg_ds = Legend(fig, ds_handles, [h.get_label() for h in ds_handles],
                    loc='center', bbox_to_anchor=(0.5, ds_y),
                    bbox_transform=fig.transFigure,
                    fontsize=TICK_FS, frameon=False,
                    ncol=len(ds_handles),
                    handlelength=1.6, columnspacing=1.5)
    fig.add_artist(leg_ds)

    leg_meta = Legend(fig, meta_handles,
                      [h.get_label() for h in meta_handles],
                      loc='center', bbox_to_anchor=(0.5, meta_y),
                      bbox_transform=fig.transFigure,
                      fontsize=TICK_FS, frameon=False,
                      ncol=len(meta_handles),
                      handlelength=1.6, columnspacing=1.5)
    fig.add_artist(leg_meta)

    leg_star = Legend(fig, star_handles,
                      [h.get_label() for h in star_handles],
                      loc='center', bbox_to_anchor=(0.5, star_y),
                      bbox_transform=fig.transFigure,
                      fontsize=TICK_FS, frameon=False,
                      ncol=len(star_handles),
                      handlelength=0, handletextpad=0,
                      columnspacing=1.8)
    fig.add_artist(leg_star)

    out_dir = f'{working_dir}/figures/de-gsea'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/fig2_themes_combined.png',
                dpi=400, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{out_dir}/fig2_themes_combined.svg',
                bbox_inches='tight', facecolor='white')
    plt.close()
    expected = sum(len(g) for tr in layout_genes for g in tr)
    print(f'combined figure: {fig_w:.2f}" × {fig_h:.2f}", '
          f'{len(cards)} cards built / {expected} cell slots')

build_combined_figure()

#endregion
