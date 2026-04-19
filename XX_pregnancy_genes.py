#region imports and setup ######################################################

import os
import numpy as np
import polars as pl
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy'

# gene groups — one column per group
gene_groups = {
    'Hormonal & Reproductive': [
        'Esr1', 'Pgr', 'Inhbb', 'Oxtr', 'Nr3c1', 'Crh',
        'Adrb1', 'Lifr',
    ],
    'Neuropeptides & Neuromodulators': [
        'Htr2a', 'Sst', 'Gal', 'Calcr', 'Nts', 'Maob',
    ],
    'Neurotrophic & Growth Factors': [
        'Tgfb2', 'Egfr', 'Ntrk3', 'Fgf1', 'Ngf',
        'Bdnf', 'Ntrk2', 'Vegfa', 'Flt1', 'Kitl',
    ],
}
group_names = list(gene_groups.keys())
GENE_COLS = list(gene_groups.values())
GENES = [g for col in GENE_COLS for g in col]

# thresholds (unified across heatmap and spatial)
MIN_L2FC = 0.25       # min |logFC| in both Xn and MF for concordance
MAX_HEAT_CTS = 10     # max cell types (rows) per heatmap

cell_type_col = 'subclass'

# label mappings
cond_short = {'CTRL': 'Null', 'PREG': 'Preg', 'POSTPART': 'Post'}
ds_names = {'xenium': 'Xenium', 'merfish': 'MERFISH'}
sp_ds_labels = ['Xenium', 'MERFISH', 'MERFISH']

# contrasts and layout order
fig_contrasts = [
    ('xenium', 'PREG_vs_CTRL'),
    ('merfish', 'PREG_vs_CTRL'),
    ('merfish', 'POSTPART_vs_PREG'),
]
col_labels = [
    f'{ds_names[ds]} {cond_short[c.split("_vs_")[0]]} '
    f'vs {cond_short[c.split("_vs_")[1]]}'
    for ds, c in fig_contrasts]
spatial_sides = [
    ('xenium', 'CTRL', 'PREG'),
    ('merfish', 'CTRL', 'PREG'),
    ('merfish', 'PREG', 'POSTPART'),
]

#endregion

#region load data (once) #######################################################

de_sub = pl.read_csv(f'{working_dir}/output/de_results.csv')

sp_adatas = {}
for ds in ['xenium', 'merfish']:
    adata = sc.read_h5ad(
        f'{working_dir}/output/{ds}/03_adata_query_{ds}.h5ad')
    if 'gene_symbol' in adata.var.columns:
        adata.var.index = adata.var['gene_symbol']
        adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    if ds == 'xenium':
        adata = adata[adata.obs['sample'] != 'CTRL_3'].copy()
    sp_adatas[ds] = adata

sp_coords = {}
for ds, adata in sp_adatas.items():
    xv = adata.obs['x_ffd'].values
    yv = adata.obs['y_ffd'].values
    fov_half = max(np.ptp(xv), np.ptp(yv)) / 2 * 1.05
    sp_coords[ds] = dict(
        x=xv, y=yv,
        cond=adata.obs['condition'].values,
        cell_type=adata.obs[cell_type_col].values,
        midline=(xv.min() + xv.max()) / 2,
        fov_cx=(xv.min() + xv.max()) / 2,
        fov_cy=(yv.min() + yv.max()) / 2,
        fov_half=fov_half)

#endregion

#region figure #################################################################

seismic_cmap = plt.cm.seismic
os.makedirs(f'{working_dir}/figures', exist_ok=True)

# --- helpers ------------------------------------------------------------------

TICK_FS = 5   # axis tick and label font size
TITLE_FS = 7  # gene / group title font size

def format_ax(ax, lw=0.5):
    for spine in ax.spines.values():
        spine.set_linewidth(lw)

def vrange_pctl(arr, lo=5, hi=95):
    """Return (lo, hi) percentiles; fall back to (0, 1) if empty."""
    if len(arr) == 0:
        return 0, 1
    return np.percentile(arr, lo), np.percentile(arr, hi)

def optimal_leaf_order(cts, lfc_d):
    """Reorder cell types by hierarchical clustering (optimal leaf order)."""
    if len(cts) <= 1:
        return cts
    mat = np.nan_to_num(np.array([lfc_d[ct] for ct in cts]), nan=0)
    Z = linkage(pdist(mat), method='average', optimal_ordering=True)
    return [cts[i] for i in leaves_list(Z)]

# --- precompute per-gene data -------------------------------------------------

gene_data = {}
for GENE in GENES:
    de_gene = de_sub.filter(pl.col('gene') == GENE)

    lfc_d, fdr_d, pval_d = {}, {}, {}
    for ci, (ds, con) in enumerate(fig_contrasts):
        for r in de_gene.filter(
                (pl.col('dataset') == ds) &
                (pl.col('contrast') == con)).iter_rows(named=True):
            ct = r['cell_type']
            lfc_d.setdefault(ct, [np.nan] * 3)[ci] = r['logFC']
            fdr_d.setdefault(ct, [np.nan] * 3)[ci] = r['FDR']
            pval_d.setdefault(ct, [np.nan] * 3)[ci] = r['PValue']

    # concordance: Xn and MF must agree in sign, both exceed threshold
    concordant = []
    for ct, lfc in lfc_d.items():
        if np.isnan(lfc[0]) or np.isnan(lfc[1]):
            continue
        if lfc[0] * lfc[1] <= 0:
            continue
        if abs(lfc[0]) < MIN_L2FC or abs(lfc[1]) < MIN_L2FC:
            continue
        concordant.append((ct, np.nanmean(lfc), np.nanmax(np.abs(lfc))))

    total_up = sum(1 for _, m, _ in concordant if m > 0)
    total_dn = sum(1 for _, m, _ in concordant if m <= 0)

    # cap at MAX_HEAT_CTS, prioritising by max |logFC|
    concordant.sort(key=lambda x: -x[2])
    concordant = concordant[:MAX_HEAT_CTS]
    up = [ct for ct, m, _ in concordant if m > 0]
    dn = [ct for ct, m, _ in concordant if m <= 0]

    # optimal leaf ordering within up / dn blocks
    up = optimal_leaf_order(up, lfc_d)
    dn = optimal_leaf_order(dn, lfc_d)
    heat_cts = up + dn
    if not heat_cts:
        print(f'[{GENE}] no concordant cell types \u2014 skipped')
        continue

    heat_lfc = np.array([lfc_d[c] for c in heat_cts])
    heat_fdr = np.array([fdr_d[c] for c in heat_cts])
    heat_pval = np.array([pval_d[c] for c in heat_cts])

    # spatial expression — only cells from heatmap cell types
    sp_data = []
    for ds, cond_l, cond_r in spatial_sides:
        c = sp_coords[ds]
        expr = np.log2(np.asarray(
            sp_adatas[ds][:, GENE].X.toarray()).flatten() + 1)
        ct_mask = np.isin(c['cell_type'], heat_cts)
        mask = ct_mask & (
            ((c['cond'] == cond_l) & (c['x'] < c['midline'])) |
            ((c['cond'] == cond_r) & (c['x'] >= c['midline'])))
        sp_data.append(dict(
            x=c['x'][mask], y=c['y'][mask],
            expr=expr[mask], midline=c['midline'],
            fov_cx=c['fov_cx'], fov_cy=c['fov_cy'],
            fov_half=c['fov_half']))

    xn_nz = sp_data[0]['expr'][sp_data[0]['expr'] > 0]
    mf_nz = np.concatenate(
        [sp_data[i]['expr'][sp_data[i]['expr'] > 0] for i in [1, 2]])
    sp_vranges = [vrange_pctl(xn_nz), vrange_pctl(mf_nz), vrange_pctl(mf_nz)]

    gene_data[GENE] = dict(
        heat_cts=heat_cts,
        heat_lfc=heat_lfc, heat_fdr=heat_fdr, heat_pval=heat_pval,
        sp_data=sp_data, sp_vranges=sp_vranges)
    print(f'[{GENE}] {total_up}\u2191 {total_dn}\u2193 '
          f'(showing {len(heat_cts)})')

# compact GENE_COLS: drop genes without concordant data
GENE_COLS = [[g for g in col if g in gene_data] for col in GENE_COLS]

# --- layout constants (all in inches) ----------------------------------------

CELL_H = 0.11       # heatmap cell side length (square cells)
SP_S = 0.85          # spatial plot side (square)
SP_GAP = 0.04        # gap between adjacent spatial panels
SP_BREAK = 0.04      # gap before the postpartum contrast (same as SP_GAP)
SEC_GAP = 0.12       # gap between heatmap and spatial section
HEAT_LABEL_W = 0.55  # space for cell-type y-axis labels
COL_GAP = 0.80       # gap between gene columns
ROW_GAP = 0.25       # gap between gene rows
TITLE_PAD = 0.08     # vertical gap above panels for gene title
M_L = 0.04           # left margin
M_R = 0.06           # right margin
M_T = 0.55           # top margin (group labels + spatial titles)
M_B = 0.06           # bottom margin

HEAT_SPLIT = 0.04    # gap between preg-vs-null and postpart-vs-preg columns
heat_w_fixed = 3 * CELL_H + HEAT_SPLIT
gene_w = (HEAT_LABEL_W + heat_w_fixed + SEC_GAP
          + SP_S + SP_GAP + SP_S + SP_BREAK + SP_S)

n_gene_rows = max(len(col) for col in GENE_COLS)
n_gene_cols = len(GENE_COLS)

row_heights = []
for ri in range(n_gene_rows):
    mx = 0
    for ci in range(n_gene_cols):
        if ri < len(GENE_COLS[ci]):
            mx = max(mx, len(gene_data[GENE_COLS[ci][ri]]['heat_cts']))
    row_heights.append(max(mx * CELL_H, SP_S) if mx else SP_S)

fig_w = M_L + n_gene_cols * gene_w + (n_gene_cols - 1) * COL_GAP + M_R
fig_h = M_T + sum(row_heights) + (n_gene_rows - 1) * ROW_GAP + M_B
fig = plt.figure(figsize=(fig_w, fig_h))

# --- plotting loop ------------------------------------------------------------

for ri in range(n_gene_rows):
    rh = row_heights[ri]
    row_top = fig_h - M_T - sum(row_heights[:ri]) - ri * ROW_GAP
    row_bot = row_top - rh

    for ci in range(n_gene_cols):
        if ri >= len(GENE_COLS[ci]):
            continue
        gene = GENE_COLS[ci][ri]
        gd = gene_data[gene]
        n_cts = len(gd['heat_cts'])
        is_bottom = (ri == len(GENE_COLS[ci]) - 1)

        x0 = M_L + ci * (gene_w + COL_GAP)
        heat_h = n_cts * CELL_H
        heat_bot = row_bot + (rh - heat_h) / 2
        sp_bot = row_bot + (rh - SP_S) / 2

        # gene title
        title_cx = (x0 + HEAT_LABEL_W + heat_w_fixed / 2) / fig_w
        title_cy = (heat_bot + heat_h + TITLE_PAD) / fig_h
        fig.text(title_cx, title_cy, gene,
                 fontsize=TITLE_FS, fontstyle='italic',
                 ha='center', va='bottom')

        # group label — first row only, centered over spatial panels
        if ri == 0:
            sp_x0_abs = x0 + HEAT_LABEL_W + heat_w_fixed + SEC_GAP
            sp_center = sp_x0_abs + (3 * SP_S + 2 * SP_GAP) / 2
            fig.text(sp_center / fig_w,
                     1 - 0.12 / fig_h,
                     group_names[ci],
                     fontsize=TITLE_FS,
                     ha='center', va='top')

        # ---- heatmap (two blocks: preg-vs-null | postpart-vs-preg) ----------
        vh = max(np.nanpercentile(np.abs(gd['heat_lfc']), 95), 0.1)
        hx_base = x0 + HEAT_LABEL_W

        # left block — columns 0-1 (Xn & MF preg vs null)
        ax_hl = fig.add_axes([hx_base / fig_w, heat_bot / fig_h,
                              (2 * CELL_H) / fig_w, heat_h / fig_h])
        ax_hl.imshow(gd['heat_lfc'][:, :2], cmap='seismic',
                     aspect='equal', vmin=-vh, vmax=vh,
                     interpolation='nearest')
        for i in range(n_cts):
            for j in range(2):
                if (not np.isnan(gd['heat_fdr'][i, j])
                        and gd['heat_fdr'][i, j] < 0.10):
                    ax_hl.plot(j, i, '*', ms=3, mfc='white',
                               mec='none', zorder=4)
                elif (not np.isnan(gd['heat_pval'][i, j])
                      and gd['heat_pval'][i, j] < 0.05):
                    ax_hl.plot(j, i, 'o', ms=1.2, mfc='white',
                               mec='none', zorder=4)
        ax_hl.set_yticks(range(n_cts))
        ax_hl.set_yticklabels(gd['heat_cts'], fontsize=TICK_FS)
        ax_hl.set_xticks(range(2))
        if is_bottom:
            ax_hl.set_xticklabels(col_labels[:2], rotation=45,
                                  ha='right', fontsize=TICK_FS)
        else:
            ax_hl.set_xticklabels([])
        ax_hl.tick_params(length=0)
        format_ax(ax_hl)

        # right block — column 2 (MF postpart vs preg)
        hx_r = hx_base + 2 * CELL_H + HEAT_SPLIT
        ax_hr = fig.add_axes([hx_r / fig_w, heat_bot / fig_h,
                              CELL_H / fig_w, heat_h / fig_h])
        ax_hr.imshow(gd['heat_lfc'][:, 2:3], cmap='seismic',
                     aspect='equal', vmin=-vh, vmax=vh,
                     interpolation='nearest')
        for i in range(n_cts):
            if (not np.isnan(gd['heat_fdr'][i, 2])
                    and gd['heat_fdr'][i, 2] < 0.10):
                ax_hr.plot(0, i, '*', ms=3, mfc='white',
                           mec='none', zorder=4)
            elif (not np.isnan(gd['heat_pval'][i, 2])
                  and gd['heat_pval'][i, 2] < 0.05):
                ax_hr.plot(0, i, 'o', ms=1.2, mfc='white',
                           mec='none', zorder=4)
        ax_hr.set_yticks([])
        ax_hr.set_xticks([0])
        if is_bottom:
            ax_hr.set_xticklabels([col_labels[2]], rotation=45,
                                  ha='right', fontsize=TICK_FS)
        else:
            ax_hr.set_xticklabels([])
        ax_hr.tick_params(length=0)
        format_ax(ax_hr)

        # ---- spatial maps ----------------------------------------------------
        sp_x0 = x0 + HEAT_LABEL_W + heat_w_fixed + SEC_GAP
        sp_offsets = [0, SP_S + SP_GAP,
                      SP_S + SP_GAP + SP_S + SP_BREAK]
        for si in range(3):
            sx = (sp_x0 + sp_offsets[si]) / fig_w
            ax = fig.add_axes([sx, sp_bot / fig_h,
                               SP_S / fig_w, SP_S / fig_h])
            d = gd['sp_data'][si]
            vmin_sp, vmax_sp = gd['sp_vranges'][si]
            order = np.argsort(d['expr'])
            ax.scatter(d['x'][order], d['y'][order],
                       c=d['expr'][order], cmap='viridis', s=0.15,
                       vmin=vmin_sp, vmax=vmax_sp,
                       linewidths=0, rasterized=True)
            # midline clipped to brain extent (consistent across panels)
            ds_key = spatial_sides[si][0]
            c = sp_coords[ds_key]
            ax.plot([d['midline'], d['midline']],
                    [c['y'].min(), c['y'].max()],
                    color='white', lw=0.3, zorder=2)
            ax.set_facecolor('black')
            ax.set_xlim(d['fov_cx'] - d['fov_half'],
                        d['fov_cx'] + d['fov_half'])
            ax.set_ylim(d['fov_cy'] - d['fov_half'],
                        d['fov_cy'] + d['fov_half'])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            format_ax(ax)
            ax.set_title(sp_ds_labels[si], fontsize=TICK_FS, pad=2)
            _, cl, cr = spatial_sides[si]
            ax.text(0.25, 0.02, cond_short[cl],
                    transform=ax.transAxes, fontsize=4,
                    ha='center', color='white')
            ax.text(0.75, 0.02, cond_short[cr],
                    transform=ax.transAxes, fontsize=4,
                    ha='center', color='white')

# --- save ---------------------------------------------------------------------

plt.savefig(f'{working_dir}/figures/pregnancy_genes.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(f'{working_dir}/figures/pregnancy_genes.svg',
            bbox_inches='tight', facecolor='white')
plt.close()

#endregion
