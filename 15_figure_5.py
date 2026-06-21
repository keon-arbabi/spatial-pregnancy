"""Figure 5: lipid-metabolism figure. Panel A (GSEA dotplot) + Panel B (DE
dotplot) + gene cards + a single-row LIANA chord set (4 themes). Neurons + glia.
Orchestrates the shared building blocks in 12_figure_helper.py.

Configuration reflects the biological-validation audit (output/audit_fig5):
direction-coherent bands (the old "Cholesterol & carriers" band is split into a
DOWN sterol/lipoprotein-supply band and an UP lipoprotein-uptake band), Fabp7
moved to the fatty-acid band, dead Lpl/weak Pcsk9 dropped from Panel B, and the
chord themes re-selected for cross-platform, direction-coherent lipid signaling
(Pcsk9/Reln removed; prosaposin + lipoprotein-lipase uptake added).
"""
import os
import importlib

import matplotlib.pyplot as plt

# digit-prefixed module name -> import via importlib
fc = importlib.import_module('12_figure_helper')

fc.setup_style()
working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures'
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# Config: pathways (4 bands), genes (5 bands), cells, cards, chord themes
# =============================================================================
PATHWAY_BANDS = [
    ('Membrane lipid', [
        'GOBP_MEMBRANE_LIPID_METABOLIC_PROCESS',
        'GOBP_GLYCEROLIPID_METABOLIC_PROCESS',
        'GOBP_LIPID_TRANSLOCATION',
    ]),
    ('Ceramide & sphingolipid', [
        'GOBP_SPHINGOLIPID_METABOLIC_PROCESS',
        'GOBP_CERAMIDE_METABOLIC_PROCESS',
    ]),
    ('Fatty acid catabolism', [
        'GOBP_FATTY_ACID_CATABOLIC_PROCESS',
        'GOBP_FATTY_ACID_BETA_OXIDATION',
    ]),
    ('Sterol & lipoprotein supply', [
        'GOBP_REGULATION_OF_LIPID_LOCALIZATION',
        'GOBP_NEGATIVE_REGULATION_OF_LIPID_TRANSPORT',
    ]),
    ('Lipoprotein uptake & efflux', [
        'GOBP_POSITIVE_REGULATION_OF_LIPID_LOCALIZATION',
        'GOBP_POSITIVE_REGULATION_OF_CHOLESTEROL_EFFLUX',
    ]),
]
ordered_pathways = [p for _, ps in PATHWAY_BANDS for p in ps]
pathway_band = {p: b for b, ps in PATHWAY_BANDS for p in ps}

PATHWAY_LABELS = {
    'GOBP_MEMBRANE_LIPID_METABOLIC_PROCESS':          'membrane lipid metabolism',
    'GOBP_GLYCEROLIPID_METABOLIC_PROCESS':            'glycerolipid metabolism',
    'GOBP_LIPID_TRANSLOCATION':                       'lipid translocation',
    'GOBP_SPHINGOLIPID_METABOLIC_PROCESS':            'sphingolipid metabolism',
    'GOBP_CERAMIDE_METABOLIC_PROCESS':                'ceramide metabolism',
    'GOBP_FATTY_ACID_CATABOLIC_PROCESS':              'fatty acid catabolism',
    'GOBP_FATTY_ACID_BETA_OXIDATION':                 'fatty acid β-oxidation',
    'GOBP_REGULATION_OF_LIPID_LOCALIZATION':          'lipid localization regulation',
    'GOBP_NEGATIVE_REGULATION_OF_LIPID_TRANSPORT':    'lipid transport',
    'GOBP_POSITIVE_REGULATION_OF_LIPID_LOCALIZATION': 'lipid uptake',
    'GOBP_POSITIVE_REGULATION_OF_CHOLESTEROL_EFFLUX': 'cholesterol efflux',
}

# 5 bands (Panel B adds the UP lipoprotein-uptake band, which has no pathway row)
BAND_COLORS = {
    'Membrane lipid':              '#E69F00',
    'Ceramide & sphingolipid':     '#CC79A7',
    'Fatty acid catabolism':       '#009E73',
    'Sterol & lipoprotein supply': '#0072B2',
    'Lipoprotein uptake & efflux': '#D55E00',
}

GENE_BANDS = [
    ('Membrane lipid', [
        'Lpin1', 'Tecr', 'Agpat4', 'Dgat1', 'Pisd', 'Mfsd2a',
    ]),
    ('Ceramide & sphingolipid', [
        'Cers4', 'Cers5', 'Cers6', 'St6galnac5', 'Hexa', 'Glb1',
    ]),
    ('Fatty acid catabolism', [
        'Fabp7', 'Cpt1a', 'Ivd', 'Echs1', 'Acaa2', 'Decr1', 'Acox1', 'Hacl1',
    ]),
    ('Sterol & lipoprotein supply', [
        'Apoe', 'Clu', 'Hmgcs1', 'Hmgcr', 'Idi1', 'Sqle',
    ]),
    ('Lipoprotein uptake & efflux', [
        'Sort1', 'Sorl1', 'Srebf1', 'Cd81', 'Abca1', 'Abca8b',
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
CARD_GENES = [g for g in ordered_genes if g in set(CARD_GENES)]  # dotplot order
# force the IHC-validated glial Apoe populations (microglia + OPC) into the card;
# the remaining slots auto-fill with the strongest responders
card_ctx = {'Apoe': ['334 Microglia NN', '326 OPC NN']}
max_rows_map = {'Apoe': 5}

# Single-row chord set (audit-revised): Pcsk9 (drawn-direction artifact) and Reln
# (directionless, neurodevelopmental rather than lipid) removed; Psap (prosaposin/
# saposin, the strongest sphingolipid CCC; glia->neuron DOWN) and Lpl (lipoprotein-
# lipase uptake, the only strong cross-platform lipid CCC; UP) added.
CHORD_THEMES = ['Apoe', 'Psap', 'Pltp', 'Lpl']
CHORD_THEME_LIGANDS = {'Apoe': ['Apoe'], 'Psap': ['Psap'],
                       'Pltp': ['Pltp'], 'Lpl': ['Lpl']}
CHORD_THEME_TITLES = {'Apoe': 'ApoE carriers\nApoe',
                      'Psap': 'Saposin / sphingolipid\nPsap',
                      'Pltp': 'Phospholipid transfer\nPltp',
                      'Lpl':  'Lipoprotein uptake\nLpl'}
CANONICAL_LR_PAIRS = set()
for _r in ['Lrp1', 'Lrp2', 'Lrp4', 'Lrp8', 'Vldlr', 'Ldlr', 'Sort1',
           'Sorl1', 'Trem2', 'Abca1']:
    CANONICAL_LR_PAIRS.add(('Apoe', _r))
for _r in ['Gpr37l1', 'Gpr37', 'Sort1', 'Lrp1']:
    CANONICAL_LR_PAIRS.add(('Psap', _r))
CANONICAL_LR_PAIRS.add(('Pltp', 'Abca1'))
for _r in ['Lrp1', 'Vldlr']:
    CANONICAL_LR_PAIRS.add(('Lpl', _r))

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
# Layout: single chord row (4 themes) below Panel B, fitted across content width
# =============================================================================
class_spans = fc.spans([ct_class[c] for c in ordered_cts])
band_spans = fc.spans([pathway_band[p] for p in ordered_pathways])
gene_band_spans = fc.spans([gene_band[g] for g in ordered_genes])
norm_nes, norm_lfc, nes_vmax, lfc_vmax = fc.make_norms(
    nes_mat, lfc_mat, d_mat_b)

# gap (column units) between cellular-neighbourhood (class) sections
SECTION_GAP = 0.8
# cards extend down to the bottom of the Panel B cell-type labels
card_extend = (fc.COL_ANNO_GAP_IN + fc.ANNO_W_IN
               + fc.label_drop_in(ordered_cts))
# forest-plot label column sized for full cell-type names (match dotplot)
label_w = fc.text_width_in([f'{c}  ***' for c in ordered_cts], pad_in=0.10)
CHORD_NCOLS = len(CHORD_THEMES)            # one row
CHORD_ROW_GAP, CHORD_TITLE_H = 0.0, fc.CHORD_TITLE_H_IN
CHORD_PANEL_GAP = fc.CHORD_PANEL_GAP_IN
CHORD_BOT_PAD, CHORD_CLEARANCE = 0.02, 0.55
cm = fc.card_metrics(len(ordered_pathways), len(ordered_genes),
                     len(ordered_cts), len(CARD_GENES), max_sp_n,
                     label_w_in=label_w, n_sections=len(class_spans),
                     section_gap=SECTION_GAP, card_extend_in=card_extend)
span = fc.chord_full_span(cm, max_sp_n)
cpw, cph = fc.chord_panel_size(span, len(CHORD_THEMES), CHORD_NCOLS,
                               CHORD_PANEL_GAP)
chord_h = fc.chord_band_height(cph, len(CHORD_THEMES), CHORD_NCOLS,
                               CHORD_TITLE_H, row_gap=CHORD_ROW_GAP)
# reserve room below Panel B for the extended cards + column labels, then a
# clearance gap above the chord row (chords shift down with the cards)
ax_b_bot = CHORD_BOT_PAD + chord_h + card_extend + CHORD_CLEARANCE

L = fc.core_layout(len(ordered_pathways), len(ordered_genes),
                   len(ordered_cts), len(CARD_GENES), max_sp_n, ax_b_bot,
                   label_w_in=label_w, n_sections=len(class_spans),
                   section_gap=SECTION_GAP, card_extend_in=card_extend)
L.col_x = fc.column_positions(class_spans, len(ordered_cts), SECTION_GAP)
fig = plt.figure(figsize=(L.fig_w, L.fig_h))
ax_a = fc.add_axes(fig, L, L.ax_left_in, L.ax_a_bot_in, L.ax_w_in, L.ax_h_a_in)
ax_b = fc.add_axes(fig, L, L.ax_left_in, L.ax_b_bot_in, L.ax_w_in, L.ax_h_b_in)

fc.draw_panel_a(ax_a, L, nlp_mat, nes_mat, d_mat, sig_mat_a, norm_nes,
                ordered_pathways, PATHWAY_LABELS, class_spans, band_spans)
fc.draw_panel_b(ax_b, L, lfc_mat, pct_mat, d_mat_b, sig_mat_b, norm_lfc,
                ordered_genes, class_spans, gene_band_spans, CARD_GENES)

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

specs = fc.chord_grid_specs(CHORD_THEMES, L.ax_left_in, CHORD_BOT_PAD, cpw, cph,
                            CHORD_NCOLS, CHORD_PANEL_GAP, row_gap=CHORD_ROW_GAP,
                            title_h=CHORD_TITLE_H)
cy = (min(s[2] for s in specs) + max(s[2] + s[4] for s in specs)) / 2
fc.draw_chord_row(fig, L, specs, chord_edges, CHORD_THEME_LIGANDS,
                  CHORD_THEME_TITLES, chord_cell_set, subclass_colors,
                  ytitle_x_in=L.ax_left_in - 0.34, ytitle_cy_in=cy)
fc.draw_chord_legend(fig, L, tleg_bot - fc.GENE_PITCH, chord_cell_set,
                     subclass_colors)

fc.save(fig, f'{out_dir}/figure_5')
print(f'wrote {out_dir}/figure_5.png and .svg '
      f'({L.fig_w:.1f}x{L.fig_h:.1f}in, {len(ordered_cts)} cells)')
