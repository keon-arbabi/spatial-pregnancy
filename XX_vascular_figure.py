"""Vascular niche figure: Panel A (GSEA dotplot) + Panel B (DE dotplot)
+ gene cards + a 3-theme LIANA chord row + a CD31 imaging block. NN-only
niche columns. Orchestrates the shared building blocks in figure_common.py;
the imaging block is figure-specific and lives here.
"""
import os

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

import figure_common as fc

fc.setup_style()
working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures/vascular'
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# Config: pathways (4 bands), genes (4 bands), cells, cards, chord themes
# =============================================================================
PATHWAY_BANDS = [
    ('Angiogenic sprouting', [
        'GOBP_VASCULATURE_DEVELOPMENT',
        'GOBP_SPROUTING_ANGIOGENESIS',
        'GOBP_VASCULOGENESIS',
    ]),
    ('VEGF axis', [
        'GOBP_CELLULAR_RESPONSE_TO_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_STIMULUS',
        'GOBP_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_SIGNALING_PATHWAY',
        'GOBP_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_PRODUCTION',
    ]),
    ('Endothelial dynamics', [
        'GOBP_ENDOTHELIAL_CELL_PROLIFERATION',
        'GOBP_ENDOTHELIAL_CELL_MIGRATION',
        'GOBP_BLOOD_VESSEL_ENDOTHELIAL_CELL_MIGRATION',
    ]),
    ('Barrier & ECM', [
        'GOBP_ESTABLISHMENT_OF_ENDOTHELIAL_BARRIER',
        'GOBP_TIGHT_JUNCTION_ORGANIZATION',
        'GOBP_COLLAGEN_BIOSYNTHETIC_PROCESS',
    ]),
]
ordered_pathways = [p for _, ps in PATHWAY_BANDS for p in ps]
pathway_band = {p: b for b, ps in PATHWAY_BANDS for p in ps}

PATHWAY_LABELS = {
    'GOBP_VASCULATURE_DEVELOPMENT':                 'vasculature development',
    'GOBP_SPROUTING_ANGIOGENESIS':                  'sprouting angiogenesis',
    'GOBP_VASCULOGENESIS':                          'vasculogenesis',
    'GOBP_CELLULAR_RESPONSE_TO_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_STIMULUS':
                                                    'response to VEGF',
    'GOBP_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_SIGNALING_PATHWAY':
                                                    'VEGF signaling',
    'GOBP_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_PRODUCTION':
                                                    'VEGF production',
    'GOBP_ENDOTHELIAL_CELL_PROLIFERATION':          'endothelial proliferation',
    'GOBP_ENDOTHELIAL_CELL_MIGRATION':              'endothelial migration',
    'GOBP_BLOOD_VESSEL_ENDOTHELIAL_CELL_MIGRATION': 'blood-vessel EC migration',
    'GOBP_ESTABLISHMENT_OF_ENDOTHELIAL_BARRIER':    'endothelial barrier',
    'GOBP_TIGHT_JUNCTION_ORGANIZATION':
        'tight junction organization',
    'GOBP_COLLAGEN_BIOSYNTHETIC_PROCESS':           'collagen biosynthesis',
}

BAND_COLORS = {
    'Angiogenic sprouting': '#0072B2',
    'VEGF axis':            '#E69F00',
    'Endothelial dynamics': '#009E73',
    'Barrier & ECM':        '#CC79A7',
}

GENE_BANDS = [
    ('Angiogenic sprouting', [
        'Notch1', 'Notch3', 'Dll4', 'Hey1', 'Angpt2', 'Cspg4',
    ]),
    ('VEGF axis', [
        'Vegfa', 'Vegfc', 'Kdr', 'Flt1', 'Nrp1',
    ]),
    ('Endothelial dynamics', [
        'Rgcc', 'Id1', 'Klf4', 'Eng', 'Acvrl1', 'Cdh5', 'Pecam1', 'Pdgfrb',
    ]),
    ('Barrier & ECM', [
        'Mfsd2a', 'Slc2a1', 'Tjp1', 'Itgb1', 'Itgav', 'Col4a1', 'Icam1',
    ]),
]
ordered_genes = [g for _, gs in GENE_BANDS for g in gs]
gene_band = {g: b for b, gs in GENE_BANDS for g in gs}

ct_allowlist = {
    '318 Astro-NT NN', '319 Astro-TE NN', '323 Ependymal NN', '326 OPC NN',
    '327 Oligo NN', '334 Microglia NN', '335 BAM NN',
}
CORTICAL_GABA = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg', 'Pax6']


def assign_class(ct):
    if 'NN' in ct: return 'Non-neuronal'
    if 'Glut' in ct: return 'Glutamatergic'
    if any(t in ct for t in CORTICAL_GABA): return 'GABAergic\nCortex'
    return 'GABAergic\nSubcortex'


CLASS_ORDER = ['Glutamatergic', 'GABAergic\nCortex', 'GABAergic\nSubcortex',
               'Non-neuronal']

CARD_GENES = ['Notch1', 'Vegfa', 'Flt1', 'Rgcc', 'Pecam1', 'Slc2a1']
card_ctx = {'Notch1': ['333 Endo NN'], 'Slc2a1': ['333 Endo NN']}
max_rows_map = {}

CHORD_THEMES = ['Notch', 'VEGF', 'Ang2_Cxcl12']
CHORD_THEME_LIGANDS = {'Notch': ['Dll4', 'Jag1', 'Jag2'],
                       'VEGF': ['Vegfa', 'Vegfb', 'Vegfc'],
                       'Ang2_Cxcl12': ['Angpt2', 'Cxcl12']}
CHORD_THEME_TITLES = {'Notch': 'Notch', 'VEGF': 'VEGF',
                      'Ang2_Cxcl12': 'Angpt2 / Cxcl12'}
CANONICAL_LR_PAIRS = set()
for _r in ['Notch1', 'Notch3', 'Notch4']:
    CANONICAL_LR_PAIRS.add(('Dll4', _r))
for _l in ['Jag1', 'Jag2']:
    for _r in ['Notch1', 'Notch3']:
        CANONICAL_LR_PAIRS.add((_l, _r))
for _r in ['Kdr', 'Flt1', 'Nrp1', 'Nrp2']:
    CANONICAL_LR_PAIRS.add(('Vegfa', _r))
CANONICAL_LR_PAIRS.add(('Vegfb', 'Flt1'))
for _r in ['Flt4', 'Kdr', 'Nrp2']:
    CANONICAL_LR_PAIRS.add(('Vegfc', _r))
for _r in ['Tek', 'Tie1']:
    CANONICAL_LR_PAIRS.add(('Angpt2', _r))
for _r in ['Cxcr4', 'Ackr3']:
    CANONICAL_LR_PAIRS.add(('Cxcl12', _r))

# =============================================================================
# Data + matrices + chord
# =============================================================================
gsea_all, gsea, real_nes = fc.load_gsea(working_dir)
de_sr_all, de_sr, de_pp = fc.load_de(working_dir)
subclass_colors = fc.load_subclass_colors()

ordered_cts, ct_class = fc.select_cell_types(
    gsea, ordered_pathways, ct_allowlist, assign_class, CLASS_ORDER,
    min_hits=2, candidate_expr=pl.col('cell_type').str.contains(' NN'))

nlp_mat, nes_mat, d_mat, sig_mat_a = fc.gsea_matrices(
    gsea_all, real_nes, ordered_pathways, ordered_cts)
sp_coords, sp_expr, sp_in, pct = fc.load_spatial(
    working_dir, CARD_GENES, ordered_genes, ordered_cts)
lfc_mat, pct_mat, sig_mat_b, d_mat_b = fc.de_matrices(
    de_sr, de_sr_all, de_pp, ordered_genes, ordered_cts, pct)
cards, max_sp_n = fc.build_cards(
    CARD_GENES, card_ctx, max_rows_map, de_sr, de_sr_all, de_pp,
    ordered_cts, sp_in, sp_coords, sp_expr)
chord_cell_set, chord_edges = fc.build_chord(
    f'{working_dir}/output/liana/inflow_diff.csv',
    CANONICAL_LR_PAIRS, CHORD_THEME_LIGANDS, k_neurons=10)

# =============================================================================
# Imaging data + layout (chord 1x3 over the imaging block, both below Panel B)
# =============================================================================
img_dir = f'{working_dir}/output/vascular_imaging'
img_df = pd.read_csv(f'{img_dir}/imaging_metrics.csv')
img_stats = pd.read_csv(f'{img_dir}/imaging_stats.csv').set_index('metric')
img_meta = pd.read_csv(f'{img_dir}/montage_meta.csv', keep_default_na=False)
img_samples = img_meta.sort_values(
    ['condition', 'mouse']).reset_index(drop=True)
COND_ORDER = ['Nulliparous', 'Pregnant']
COND_COLORS = {'Nulliparous': '#7209b7', 'Pregnant': '#b5179e'}
MOUSE_MARKERS = {1: 'o', 2: 's', 3: '^', 4: 'D', 5: 'v'}
IMG_NCOL = max((img_samples.condition == c).sum() for c in COND_ORDER)

IMG_BOT_PAD, IMG_TO_CHORD_GAP, IMG_YLAB = 0.42, 0.55, 0.30
IMG_CD31_GAP, IMG_ROW_GAP, IMG_BOX_GAP = 0.07, 0.10, 0.50
IMG_BOX_LAB, IMG_BOX_W, IMG_BOX_VGAP, IMG_BOX_XLAB = 0.50, 1.00, 0.24, 0.30
CHORD_TITLE_H, B_TO_CHORD_GAP, BOT_MARGIN = 0.30, 0.45, 0.85

class_spans = fc.spans([ct_class[c] for c in ordered_cts])
band_spans = fc.spans([pathway_band[p] for p in ordered_pathways])
gene_band_spans = fc.spans([gene_band[g] for g in ordered_genes])
norm_nes, norm_lfc, nes_vmax, lfc_vmax = fc.make_norms(
    nes_mat, lfc_mat, d_mat_b)

cm = fc.card_metrics(len(ordered_pathways), len(ordered_genes),
                     len(ordered_cts), len(CARD_GENES), max_sp_n)
span = fc.chord_full_span(cm, max_sp_n)
cpw, cph = fc.chord_panel_size(span, len(CHORD_THEMES), 3,
                               fc.CHORD_PANEL_GAP_IN)
chord_h = cph + CHORD_TITLE_H

img_w = span
img_cd31 = ((img_w - IMG_YLAB - (IMG_NCOL - 1) * IMG_CD31_GAP - IMG_BOX_GAP
             - IMG_BOX_LAB - IMG_BOX_W) / IMG_NCOL)
img_stack_h = 2 * img_cd31 + IMG_ROW_GAP
img_stack_bot = IMG_BOT_PAD + IMG_BOX_XLAB
img_stack_top = img_stack_bot + img_stack_h
chord_row_bot = img_stack_top + IMG_TO_CHORD_GAP
ax_b_bot = chord_row_bot + chord_h + B_TO_CHORD_GAP + BOT_MARGIN

L = fc.core_layout(len(ordered_pathways), len(ordered_genes),
                   len(ordered_cts), len(CARD_GENES), max_sp_n, ax_b_bot)
fig = plt.figure(figsize=(L.fig_w, L.fig_h))
ax_a = fc.add_axes(fig, L, L.ax_left_in, L.ax_a_bot_in, L.ax_w_in, L.ax_h_a_in)
ax_b = fc.add_axes(fig, L, L.ax_left_in, L.ax_b_bot_in, L.ax_w_in, L.ax_h_b_in)

fc.draw_panel_a(ax_a, L, nlp_mat, nes_mat, d_mat, sig_mat_a, norm_nes,
                ordered_pathways, PATHWAY_LABELS, class_spans, band_spans)
fc.draw_panel_b(ax_b, L, lfc_mat, pct_mat, d_mat_b, sig_mat_b, norm_lfc,
                ordered_genes, class_spans, gene_band_spans)

prefix_labels = [f'{fc.numeric_prefix(ct):03d}' for ct in ordered_cts]
fc.draw_col_anno(fig, L, L.ax_a_bot_in, ordered_cts, prefix_labels,
                 subclass_colors)
fc.draw_col_anno(fig, L, L.ax_b_bot_in, ordered_cts, ordered_cts,
                 subclass_colors)
fc.draw_band_anno(fig, L, L.ax_a_bot_in, L.ax_h_a_in, len(ordered_pathways),
                  [pathway_band[p] for p in ordered_pathways], BAND_COLORS)
fc.draw_band_anno(fig, L, L.ax_b_bot_in, L.ax_h_b_in, len(ordered_genes),
                  [gene_band[g] for g in ordered_genes], BAND_COLORS)
tleg_bot = fc.draw_dot_legends(fig, L, norm_nes, norm_lfc, nes_vmax,
                               lfc_vmax, [b for b, _ in GENE_BANDS],
                               BAND_COLORS)
fc.draw_forest_legend(fig, L)
fc.draw_cards(fig, L, CARD_GENES, cards, sp_coords)

specs = fc.chord_grid_specs(CHORD_THEMES, L.ax_left_in, chord_row_bot, cpw,
                            cph, 3, fc.CHORD_PANEL_GAP_IN,
                            title_h=CHORD_TITLE_H)
fc.draw_chord_row(fig, L, specs, chord_edges, CHORD_THEME_LIGANDS,
                  CHORD_THEME_TITLES, chord_cell_set, subclass_colors,
                  ytitle_x_in=L.ax_left_in - 0.34,
                  ytitle_cy_in=chord_row_bot + cph / 2)
fc.draw_chord_legend(fig, L, tleg_bot - fc.GENE_PITCH, chord_cell_set,
                     subclass_colors)

# ---- imaging block (figure-specific): CD31 montage + morphometric boxes ----
img_block_left = L.ax_left_in + IMG_YLAB
for ci, cond in enumerate(COND_ORDER):
    rows = img_samples[img_samples.condition == cond].reset_index(drop=True)
    cell_bot = img_stack_top - (ci + 1) * img_cd31 - ci * IMG_ROW_GAP
    for j in range(len(rows)):
        srow = rows.iloc[j]
        x_in = img_block_left + j * (img_cd31 + IMG_CD31_GAP)
        axc = fc.add_axes(fig, L, x_in, cell_bot, img_cd31, img_cd31)
        arr = plt.imread(f'{img_dir}/cd31_{srow["prefix"]}.png')
        if arr.ndim == 3:
            arr = arr[..., 0]
        rgb = np.zeros(arr.shape + (3,), dtype=float)
        rgb[..., 1] = arr
        axc.imshow(rgb, aspect='auto', interpolation='nearest')
        axc.set_xticks([]); axc.set_yticks([]); axc.set_facecolor('black')
        for s in axc.spines.values():
            s.set_color('#999999'); s.set_linewidth(0.5)
        axc.text(0.04, 0.95, f'M{int(srow["mouse"])}', transform=axc.transAxes,
                 ha='left', va='top', color='white', fontsize=7.0)
        if j == 0:
            bar = 100.0 / float(srow['fov_um'])
            axc.plot([0.05, 0.05 + bar], [0.08, 0.08], color='white', lw=1.6,
                     transform=axc.transAxes, solid_capstyle='butt')
            axc.text(0.05, 0.11, '100 µm', color='white', fontsize=5.5,
                     ha='left', va='bottom', transform=axc.transAxes)
    fig.text((L.ax_left_in + IMG_YLAB * 0.40) / L.fig_w,
             (cell_bot + img_cd31 / 2) / L.fig_h, cond, rotation=90,
             ha='center', va='center', fontsize=8.5, color='black')

fig.text((L.ax_left_in - 0.34) / L.fig_w,
         (img_stack_bot + img_stack_h / 2) / L.fig_h,
         'Vascular imaging (CD31)', rotation=90, ha='center', va='center',
         fontsize=9.0)

box_left = (img_block_left + IMG_NCOL * img_cd31 + (IMG_NCOL - 1)
            * IMG_CD31_GAP + IMG_BOX_GAP + IMG_BOX_LAB)
box_h = (img_stack_h - IMG_BOX_VGAP) / 2
QUANT_METRICS = [
    ('junctions_per_mm_vessel', 'Junction density\n(per mm vessel)'),
    ('mean_vessel_diameter_um', 'Vessel diameter (µm)'),
]


def draw_box(ax, metric, label, show_x):
    ax.set_facecolor('white')
    data = [img_df[img_df.condition == c][metric].dropna().to_numpy()
            for c in COND_ORDER]
    ylo = min(d.min() for d in data); yhi = max(d.max() for d in data)
    yr = yhi - ylo if yhi > ylo else 1.0
    bp = ax.boxplot(data, positions=[0, 1], widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color='black', lw=1.1),
                    whiskerprops=dict(color='#555555', lw=0.8),
                    capprops=dict(color='#555555', lw=0.8),
                    boxprops=dict(lw=0.8), zorder=2)
    for patch, c in zip(bp['boxes'], COND_ORDER):
        patch.set_facecolor(COND_COLORS[c]); patch.set_alpha(0.20)
        patch.set_edgecolor(COND_COLORS[c])
    for xi, c in enumerate(COND_ORDER):
        sub = img_df[img_df.condition == c].reset_index(drop=True)
        v = sub[metric].to_numpy()
        xs = xi + np.linspace(-0.18, 0.18, len(v))
        for k in range(len(v)):
            ax.scatter(xs[k], v[k], s=14, c=COND_COLORS[c],
                       marker=MOUSE_MARKERS.get(int(sub.loc[k, 'mouse']), 'o'),
                       edgecolors='white', linewidths=0.3, alpha=0.95, zorder=4)
        for mouse, g in sub.groupby('mouse'):
            ax.scatter(xi, g[metric].mean(), s=46, c=COND_COLORS[c],
                       marker=MOUSE_MARKERS.get(int(mouse), 'o'),
                       edgecolors='black', linewidths=0.6, zorder=5)
    yb = yhi + 0.10 * yr
    ax.plot([0, 0, 1, 1], [yb, yb + 0.04 * yr, yb + 0.04 * yr, yb],
            color='black', lw=0.8, zorder=6)
    p = float(img_stats.loc[metric, 'p_roi'])
    ax.text(0.5, yb + 0.06 * yr, f'p = {p:.2f}', ha='center', va='bottom',
            fontsize=6.6)
    ax.set_xlim(-0.6, 1.6); ax.set_xticks([0, 1])
    ax.set_xticklabels(['Null', 'Preg'] if show_x else [], fontsize=7.0)
    ax.set_ylim(ylo - 0.10 * yr, yb + 0.18 * yr)
    ax.set_ylabel(label, fontsize=7.2, labelpad=2, linespacing=1.0)
    ax.tick_params(axis='both', labelsize=6.8, length=3, width=0.8,
                   direction='out')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_linewidth(0.8)


for qi, (met, lab) in enumerate(QUANT_METRICS):
    box_bot = img_stack_bot + (1 - qi) * (box_h + IMG_BOX_VGAP)
    axb = fc.add_axes(fig, L, box_left, box_bot, IMG_BOX_W, box_h)
    draw_box(axb, met, lab, show_x=(qi == 1))

axml = fc.add_axes(fig, L, img_block_left, 0.07,
                   IMG_NCOL * img_cd31 + (IMG_NCOL - 1) * IMG_CD31_GAP, 0.24)
axml.set_xlim(0, 1); axml.set_ylim(0, 1)
axml.set_xticks([]); axml.set_yticks([])
for s in axml.spines.values():
    s.set_visible(False)
axml.text(0.0, 0.5, 'Mouse:', ha='left', va='center', fontsize=7.0)
mx = 0.105
for mnum, mk in [(1, 'o'), (2, 's'), (3, '^')]:
    axml.scatter(mx, 0.5, s=24, c='#555555', marker=mk, edgecolors='none')
    axml.text(mx + 0.014, 0.5, str(mnum), ha='left', va='center', fontsize=7.0)
    mx += 0.058
mx += 0.03
axml.scatter(mx, 0.5, s=14, c='#555555', marker='o', edgecolors='white',
             linewidths=0.3)
axml.text(mx + 0.016, 0.5, 'ROI', ha='left', va='center', fontsize=7.0)
mx += 0.10
axml.scatter(mx, 0.5, s=46, c='#555555', marker='o', edgecolors='black',
             linewidths=0.6)
axml.text(mx + 0.016, 0.5, 'mouse mean', ha='left', va='center', fontsize=7.0)

fc.save(fig, f'{out_dir}/vascular_combined')
print(f'wrote {out_dir}/vascular_combined.png and .svg')
