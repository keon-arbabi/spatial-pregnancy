"""Lipid metabolism figure: Panel A (GSEA dotplot) + Panel B (DE dotplot)
+ gene cards + a 4-theme LIANA chord grid (2x2). Neurons + glia.
Orchestrates the shared building blocks in figure_common.py.
"""
import os

import matplotlib.pyplot as plt

import figure_common as fc

fc.setup_style()
working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures/lipid'
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# Config: pathways (4 bands), genes (4 bands), cells, cards, chord themes
# =============================================================================
PATHWAY_BANDS = [
    ('Membrane lipid', [
        'GOBP_MEMBRANE_LIPID_METABOLIC_PROCESS',
        'GOBP_MEMBRANE_LIPID_BIOSYNTHETIC_PROCESS',
        'GOBP_LIPID_TRANSLOCATION',
    ]),
    ('Sphingolipid', [
        'GOBP_SPHINGOLIPID_METABOLIC_PROCESS',
        'GOBP_CERAMIDE_METABOLIC_PROCESS',
    ]),
    ('Fatty acid catabolism', [
        'GOBP_FATTY_ACID_CATABOLIC_PROCESS',
        'GOBP_FATTY_ACID_BETA_OXIDATION',
    ]),
    ('Cholesterol & carriers', [
        'GOBP_REGULATION_OF_LIPID_LOCALIZATION',
        'GOBP_LIPOPROTEIN_METABOLIC_PROCESS',
        'GOBP_STEROID_BIOSYNTHETIC_PROCESS',
    ]),
]
ordered_pathways = [p for _, ps in PATHWAY_BANDS for p in ps]
pathway_band = {p: b for b, ps in PATHWAY_BANDS for p in ps}

PATHWAY_LABELS = {
    'GOBP_MEMBRANE_LIPID_METABOLIC_PROCESS':    'membrane lipid metabolism',
    'GOBP_MEMBRANE_LIPID_BIOSYNTHETIC_PROCESS': 'membrane lipid biosynthesis',
    'GOBP_LIPID_TRANSLOCATION':                 'lipid translocation',
    'GOBP_SPHINGOLIPID_METABOLIC_PROCESS':      'sphingolipid metabolism',
    'GOBP_CERAMIDE_METABOLIC_PROCESS':          'ceramide metabolism',
    'GOBP_FATTY_ACID_CATABOLIC_PROCESS':        'fatty acid catabolism',
    'GOBP_FATTY_ACID_BETA_OXIDATION':           'fatty acid β-oxidation',
    'GOBP_REGULATION_OF_LIPID_LOCALIZATION':    'lipid localization regulation',
    'GOBP_LIPOPROTEIN_METABOLIC_PROCESS':       'lipoprotein metabolism',
    'GOBP_STEROID_BIOSYNTHETIC_PROCESS':        'steroid biosynthesis',
}

BAND_COLORS = {
    'Membrane lipid':         '#E69F00',
    'Sphingolipid':           '#CC79A7',
    'Fatty acid catabolism':  '#009E73',
    'Cholesterol & carriers': '#0072B2',
}

GENE_BANDS = [
    ('Membrane lipid', [
        'Lpin1', 'Tecr', 'Agpat4', 'Dgat1', 'Pisd',
    ]),
    ('Sphingolipid', [
        'Cers4', 'Cers5', 'Cers6', 'St6galnac5', 'Hexa', 'Glb1',
    ]),
    ('Fatty acid catabolism', [
        'Cpt1a', 'Ivd', 'Echs1', 'Acaa2', 'Decr1', 'Acox1', 'Hacl1',
    ]),
    ('Cholesterol & carriers', [
        'Apoe', 'Clu', 'Fabp7', 'Hmgcs1', 'Hmgcr', 'Idi1', 'Sqle', 'Pcsk9',
        'Sort1', 'Sorl1', 'Cd81', 'Abca1', 'Lpl', 'Srebf1', 'Mfsd2a',
    ]),
]
ordered_genes = [g for _, gs in GENE_BANDS for g in gs]
gene_band = {g: b for b, gs in GENE_BANDS for g in gs}

ct_allowlist = {'334 Microglia NN', '327 Oligo NN', '323 Ependymal NN',
                '333 Endo NN', '119 SI-MA-LPO-LHA Skor1 Glut'}
CORTICAL_GABA = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg', 'Pax6']


def assign_class(ct):
    if 'NN' in ct: return 'Non-neuronal'
    if 'Glut' in ct: return 'Glutamatergic'
    if any(t in ct for t in CORTICAL_GABA): return 'GABAergic\nCortex'
    return 'GABAergic\nSubcortex'


CLASS_ORDER = ['Glutamatergic', 'GABAergic\nCortex', 'GABAergic\nSubcortex',
               'Non-neuronal']

CARD_GENES = ['Lpin1', 'Glb1', 'Ivd', 'Apoe', 'Hmgcr', 'Idi1', 'Mfsd2a']
card_ctx = {'Apoe': ['326 OPC NN', '334 Microglia NN']}
max_rows_map = {'Apoe': 5}

CHORD_THEMES = ['Apoe', 'Pcsk9', 'Pltp', 'Reln']
CHORD_THEME_LIGANDS = {'Apoe': ['Apoe', 'Lpl'], 'Pcsk9': ['Pcsk9'],
                       'Pltp': ['Pltp'], 'Reln': ['Reln']}
CHORD_THEME_TITLES = {'Apoe': 'ApoE / Lpl pathway', 'Pcsk9': 'Pcsk9 pathway',
                      'Pltp': 'Pltp pathway', 'Reln': 'Reln pathway'}
CANONICAL_LR_PAIRS = set()
for _r in ['Lrp1', 'Lrp2', 'Lrp4', 'Lrp8', 'Vldlr', 'Ldlr', 'Sort1',
           'Sorl1', 'Trem2', 'Abca1']:
    CANONICAL_LR_PAIRS.add(('Apoe', _r))
CANONICAL_LR_PAIRS.add(('Lpl', 'Lrp1'))
for _r in ['Lrp1', 'Lrp2']:
    CANONICAL_LR_PAIRS.add(('Clu', _r))
    CANONICAL_LR_PAIRS.add(('Apod', _r))
CANONICAL_LR_PAIRS.add(('Reln', 'Vldlr'))
CANONICAL_LR_PAIRS.add(('Reln', 'Lrp8'))
for _r in ['Ldlr', 'Lrp1', 'Lrp8', 'Sort1', 'Vldlr', 'Cd81', 'Aplp2']:
    CANONICAL_LR_PAIRS.add(('Pcsk9', _r))
CANONICAL_LR_PAIRS.add(('Pltp', 'Abca1'))
for _l in ['Sphk1', 'Sphk2']:
    for _r in ['S1pr1', 'S1pr2', 'S1pr3', 'S1pr4', 'S1pr5']:
        CANONICAL_LR_PAIRS.add((_l, _r))
for _r in ['Slc22a17', 'Lrp2']:
    CANONICAL_LR_PAIRS.add(('Lcn2', _r))
CANONICAL_LR_PAIRS.add(('A2m', 'Lrp1'))

# =============================================================================
# Data + matrices + chord
# =============================================================================
gsea_all, gsea, real_nes = fc.load_gsea(working_dir)
de_sr_all, de_sr, de_pp = fc.load_de(working_dir)
subclass_colors = fc.load_subclass_colors()

ordered_cts, ct_class = fc.select_cell_types(
    gsea, ordered_pathways, ct_allowlist, assign_class, CLASS_ORDER,
    min_hits=2)

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
# Layout: chord grid (2x2) below Panel B
# =============================================================================
class_spans = fc.spans([ct_class[c] for c in ordered_cts])
band_spans = fc.spans([pathway_band[p] for p in ordered_pathways])
gene_band_spans = fc.spans([gene_band[g] for g in ordered_genes])
norm_nes, norm_lfc, nes_vmax, lfc_vmax = fc.make_norms(
    nes_mat, lfc_mat, d_mat_b)

BOT_MARGIN, CHORD_GAP = 2.15, 1.00
PANEL_GAP, ROW_GAP, TITLE_H = 0.45, 0.30, 0.40
cm = fc.card_metrics(len(ordered_pathways), len(ordered_genes),
                     len(ordered_cts), len(CARD_GENES), max_sp_n)
cpw, cph = fc.chord_panel_size(cm.ax_w_in, len(CHORD_THEMES), 2, PANEL_GAP)
chord_h = fc.chord_band_height(cph, len(CHORD_THEMES), 2, TITLE_H, ROW_GAP,
                               extra=0.20)
ax_b_bot = BOT_MARGIN + chord_h + CHORD_GAP

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

specs = fc.chord_grid_specs(CHORD_THEMES, L.ax_left_in, BOT_MARGIN, cpw, cph,
                            2, PANEL_GAP, ROW_GAP, TITLE_H, bottom_drop=0.20)
cy = (min(s[2] for s in specs) + max(s[2] + s[4] for s in specs)) / 2
fc.draw_chord_row(fig, L, specs, chord_edges, CHORD_THEME_LIGANDS,
                  CHORD_THEME_TITLES, chord_cell_set, subclass_colors,
                  ytitle_x_in=L.ax_left_in - 0.34, ytitle_cy_in=cy)
fc.draw_chord_legend(fig, L, tleg_bot - fc.GENE_PITCH, chord_cell_set,
                     subclass_colors)

fc.save(fig, f'{out_dir}/lipid_combined')
print(f'wrote {out_dir}/lipid_combined.png and .svg')
