"""Microglia activation figure: Panel A (GSEA dotplot) + Panel B (DE dotplot)
+ gene cards + a 3-theme LIANA chord row. Neurons + glia. Orchestrates the
shared building blocks in figure_common.py.
"""
import os

import matplotlib.pyplot as plt

import figure_common as fc

fc.setup_style()
working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures/microglia'
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# Config: pathways (4 bands), genes (4 bands), cells, cards, chord themes
# =============================================================================
PATHWAY_BANDS = [
    ('Immune activation', [
        'GOBP_ADAPTIVE_IMMUNE_RESPONSE',
        'GOBP_ACTIVATION_OF_IMMUNE_RESPONSE',
        'GOBP_REGULATION_OF_INNATE_IMMUNE_RESPONSE',
    ]),
    ('Cytokine production', [
        'GOBP_CYTOKINE_PRODUCTION',
        'GOBP_INTERLEUKIN_1_PRODUCTION',
        'GOBP_TUMOR_NECROSIS_FACTOR_SUPERFAMILY_CYTOKINE_PRODUCTION',
    ]),
    ('Inflammatory response', [
        'GOBP_INFLAMMATORY_RESPONSE',
        'GOBP_ACUTE_INFLAMMATORY_RESPONSE',
        'GOBP_NEUROINFLAMMATORY_RESPONSE',
    ]),
    ('Myeloid effector', [
        'GOBP_MACROPHAGE_ACTIVATION',
        'GOBP_MYELOID_LEUKOCYTE_MIGRATION',
        'GOBP_LEUKOCYTE_CHEMOTAXIS',
    ]),
]
ordered_pathways = [p for _, ps in PATHWAY_BANDS for p in ps]
pathway_band = {p: b for b, ps in PATHWAY_BANDS for p in ps}

PATHWAY_LABELS = {
    'GOBP_ADAPTIVE_IMMUNE_RESPONSE':             'adaptive immune response',
    'GOBP_ACTIVATION_OF_IMMUNE_RESPONSE':
        'activation of immune response',
    'GOBP_REGULATION_OF_INNATE_IMMUNE_RESPONSE': 'innate immune response',
    'GOBP_CYTOKINE_PRODUCTION':                  'cytokine production',
    'GOBP_INTERLEUKIN_1_PRODUCTION':             'IL-1 production',
    'GOBP_TUMOR_NECROSIS_FACTOR_SUPERFAMILY_CYTOKINE_PRODUCTION':
                                                 'TNF production',
    'GOBP_INFLAMMATORY_RESPONSE':                'inflammatory response',
    'GOBP_ACUTE_INFLAMMATORY_RESPONSE':          'acute inflammatory response',
    'GOBP_NEUROINFLAMMATORY_RESPONSE':           'neuroinflammatory response',
    'GOBP_MACROPHAGE_ACTIVATION':                'macrophage activation',
    'GOBP_MYELOID_LEUKOCYTE_MIGRATION':          'myeloid leukocyte migration',
    'GOBP_LEUKOCYTE_CHEMOTAXIS':                 'leukocyte chemotaxis',
}

BAND_COLORS = {
    'Immune activation':     '#0072B2',
    'Cytokine production':   '#E69F00',
    'Inflammatory response': '#009E73',
    'Myeloid effector':      '#CC79A7',
}

GENE_BANDS = [
    ('Immune activation', [
        'Trem2', 'Nlrp3', 'Casp1', 'C3', 'Ifngr1', 'Stat1',
    ]),
    ('Cytokine production', [
        'Il18', 'Il10', 'Il34', 'Tnf', 'Tgfb2', 'Csf1',
    ]),
    ('Inflammatory response', [
        'Il6st', 'Il4ra', 'Il10rb', 'Tnfrsf1b', 'Tgfbr1', 'Stat3', 'Jak2',
    ]),
    ('Myeloid effector', [
        'Ccl2', 'Cxcl12', 'Cx3cr1', 'P2ry12', 'Tmem119', 'Pik3cd', 'Mef2c',
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

CARD_GENES = ['Trem2', 'Mertk', 'Nlrp3', 'Tgfb2', 'Stat3', 'Cx3cr1',
              'Tmem119']
_mg = ['334 Microglia NN']
card_ctx = {'Trem2': ['334 Microglia NN', '335 BAM NN'],
            'Mertk': ['334 Microglia NN', '335 BAM NN'],
            'Nlrp3': _mg, 'Tgfb2': _mg, 'Stat3': _mg, 'Cx3cr1': _mg,
            'Tmem119': _mg}
max_rows_map = {}

CHORD_THEMES = ['Phagocytic', 'CSF1_axis', 'Antiinflam']
CHORD_THEME_LIGANDS = {
    'Phagocytic': ['Apoe', 'Gas6', 'Adam10', 'C1qb', 'C4b'],
    'CSF1_axis':  ['Csf1', 'Il34', 'Csf1_Il34'],
    'Antiinflam': ['Cx3cl1', 'Tgfb1', 'Tgfb2', 'Cd47'],
}
CHORD_THEME_TITLES = {
    'Phagocytic': 'Phagocytic clearance',
    'CSF1_axis':  'CSF1 / IL-34',
    'Antiinflam': 'Cx3cl1 / TGF-β / Cd47',
}
CANONICAL_LR_PAIRS = {
    ('Apoe', 'Trem2'), ('Apoe', 'Lrp1'), ('Gas6', 'Axl'), ('Gas6', 'Mertk'),
    ('Adam10', 'Trem2'), ('Adam10', 'Axl'), ('C1qb', 'Lrp1'),
    ('C4b', 'Nrp1'), ('C4b', 'C3ar1'), ('Csf1_Il34', 'Csf1r'),
    ('Csf1', 'Sirpa'), ('Il34', 'Ptprz1'), ('Cx3cl1', 'Cx3cr1'),
    ('Tgfb1', 'Tgfbr1_Tgfbr2'), ('Tgfb1', 'Eng'), ('Tgfb1', 'Itgb1'),
    ('Tgfb1', 'Itgb5'), ('Cd47', 'Sirpa'), ('Sirpa', 'Cd47'),
}

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
# Layout: chord row (1x3) directly below Panel B
# =============================================================================
class_spans = fc.spans([ct_class[c] for c in ordered_cts])
band_spans = fc.spans([pathway_band[p] for p in ordered_pathways])
gene_band_spans = fc.spans([gene_band[g] for g in ordered_genes])
norm_nes, norm_lfc, nes_vmax, lfc_vmax = fc.make_norms(
    nes_mat, lfc_mat, d_mat_b)

CHORD_BOT_PAD, B_TO_CHORD_GAP, BOT_MARGIN = 0.02, 0.45, 0.85
cm = fc.card_metrics(len(ordered_pathways), len(ordered_genes),
                     len(ordered_cts), len(CARD_GENES), max_sp_n)
span = fc.chord_full_span(cm, max_sp_n)
cpw, cph = fc.chord_panel_size(span, len(CHORD_THEMES), 3,
                               fc.CHORD_PANEL_GAP_IN)
chord_h = fc.chord_band_height(cph, len(CHORD_THEMES), 3, fc.CHORD_TITLE_H_IN)
ax_b_bot = CHORD_BOT_PAD + chord_h + B_TO_CHORD_GAP + BOT_MARGIN

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

specs = fc.chord_grid_specs(CHORD_THEMES, L.ax_left_in, CHORD_BOT_PAD, cpw,
                            cph, 3, fc.CHORD_PANEL_GAP_IN,
                            title_h=fc.CHORD_TITLE_H_IN)
fc.draw_chord_row(fig, L, specs, chord_edges, CHORD_THEME_LIGANDS,
                  CHORD_THEME_TITLES, chord_cell_set, subclass_colors,
                  ytitle_x_in=L.ax_left_in - 0.34,
                  ytitle_cy_in=CHORD_BOT_PAD + cph / 2)
fc.draw_chord_legend(fig, L, tleg_bot - fc.GENE_PITCH, chord_cell_set,
                     subclass_colors)

fc.save(fig, f'{out_dir}/microglia_combined')
print(f'wrote {out_dir}/microglia_combined.png and .svg')
