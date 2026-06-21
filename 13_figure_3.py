"""Neuron remodeling figure: Panel A (GSEA dotplot) + Panel B (DE dotplot)
+ gene cards. Neuronal classes (Glut + GABA) plus glial context columns.
Orchestrates the shared building blocks in 12_figure_helper.py.
"""
import os
import importlib

import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

# digit-prefixed module name -> import via importlib
fc = importlib.import_module('12_figure_helper')

fc.setup_style()
working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures'
os.makedirs(out_dir, exist_ok=True)

# =============================================================================
# Config: pathways (6 bands), genes (6 bands), cells, cards
# =============================================================================
PATHWAY_BANDS = [
    ('Synaptic adhesion', [
        'GOBP_SYNAPSE_ASSEMBLY',
        'GOBP_MAINTENANCE_OF_SYNAPSE_STRUCTURE',
        'GOBP_HOMOPHILIC_CELL_CELL_ADHESION',
    ]),
    ('Excitability', [
        'GOBP_REGULATION_OF_MEMBRANE_POTENTIAL',
        'GOBP_MONOATOMIC_ION_TRANSPORT',
        'GOBP_POTASSIUM_ION_TRANSPORT',
    ]),
    ('GABA & neuropeptide', [
        'GOBP_SYNAPTIC_TRANSMISSION_GABAERGIC',
        'GOBP_NEUROPEPTIDE_SIGNALING_PATHWAY',
        'GOBP_NEUROTRANSMITTER_SECRETION',
    ]),
    ('Glucocorticoid stress', [
        'GOBP_RESPONSE_TO_CORTICOSTEROID',
        'GOBP_RESPONSE_TO_STEROID_HORMONE',
        'GOBP_CELLULAR_RESPONSE_TO_CORTICOSTEROID_STIMULUS',
    ]),
    ('Neurotrophic', [
        'GOBP_RESPONSE_TO_NERVE_GROWTH_FACTOR',
        'GOBP_NEUROTROPHIN_TRK_RECEPTOR_SIGNALING_PATHWAY',
        'GOBP_NEUROTROPHIN_SIGNALING_PATHWAY',
    ]),
    ('Neuronal development', [
        'GOBP_REGULATION_OF_NEURON_DIFFERENTIATION',
        'GOBP_NEURON_FATE_COMMITMENT',
        'GOBP_AXON_DEVELOPMENT',
    ]),
    ('Activity-dependent plasticity', [
        'GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY',
        'GOBP_REGULATION_OF_TRANS_SYNAPTIC_SIGNALING',
        'GOBP_REGULATION_OF_LONG_TERM_SYNAPTIC_POTENTIATION',
    ]),
]
ordered_pathways = [p for _, ps in PATHWAY_BANDS for p in ps]
pathway_band = {p: b for b, ps in PATHWAY_BANDS for p in ps}

PATHWAY_LABELS = {
    'GOBP_SYNAPSE_ASSEMBLY':                     'synapse assembly',
    'GOBP_MAINTENANCE_OF_SYNAPSE_STRUCTURE':     'synapse maintenance',
    'GOBP_HOMOPHILIC_CELL_CELL_ADHESION':        'homophilic cell adhesion',
    'GOBP_REGULATION_OF_MEMBRANE_POTENTIAL':     'membrane potential',
    'GOBP_MONOATOMIC_ION_TRANSPORT':             'ion transport',
    'GOBP_POTASSIUM_ION_TRANSPORT':              'potassium transport',
    'GOBP_SYNAPTIC_TRANSMISSION_GABAERGIC':      'GABAergic transmission',
    'GOBP_NEUROPEPTIDE_SIGNALING_PATHWAY':       'neuropeptide signaling',
    'GOBP_NEUROTRANSMITTER_SECRETION':           'neurotransmitter secretion',
    'GOBP_RESPONSE_TO_CORTICOSTEROID':           'corticosteroid response',
    'GOBP_RESPONSE_TO_STEROID_HORMONE':          'steroid hormone response',
    'GOBP_CELLULAR_RESPONSE_TO_CORTICOSTEROID_STIMULUS':
                                                 'corticosteroid signaling',
    'GOBP_RESPONSE_TO_NERVE_GROWTH_FACTOR':      'NGF response',
    'GOBP_NEUROTROPHIN_TRK_RECEPTOR_SIGNALING_PATHWAY':
                                                 'Trk receptor signaling',
    'GOBP_NEUROTROPHIN_SIGNALING_PATHWAY':       'neurotrophin signaling',
    'GOBP_REGULATION_OF_NEURON_DIFFERENTIATION': 'neuron differentiation',
    'GOBP_NEURON_FATE_COMMITMENT':               'neuron fate commitment',
    'GOBP_AXON_DEVELOPMENT':                     'axon development',
    'GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY':    'synaptic plasticity',
    'GOBP_REGULATION_OF_TRANS_SYNAPTIC_SIGNALING':
                                                 'trans-synaptic signaling',
    'GOBP_REGULATION_OF_LONG_TERM_SYNAPTIC_POTENTIATION':
                                                 'long-term potentiation',
}

BAND_COLORS = {
    'Synaptic adhesion':     '#0072B2',
    'Excitability':          '#E69F00',
    'GABA & neuropeptide':   '#009E73',
    'Glucocorticoid stress': '#D55E00',
    'Neurotrophic':          '#F0E442',
    'Neuronal development':  '#CC79A7',
    'Activity-dependent plasticity': '#56B4E9',
}

GENE_BANDS = [
    ('Synaptic adhesion', [
        'Cntn1', 'Sdk1', 'Nrcam', 'Ncam1', 'Cadm1', 'Robo1', 'Cdh13',
    ]),
    ('Excitability', [
        'Gria1', 'Gria2', 'Grin1', 'Grin2a', 'Kcnh1', 'Gria4', 'Camk4',
    ]),
    ('GABA & neuropeptide', [
        'Gad2', 'Gad1', 'Gabrb3', 'Gabra1', 'Tac1', 'Vamp2', 'Syt1',
    ]),
    ('Glucocorticoid stress', [
        'Fkbp5', 'Gpr83', 'Zbtb16', 'Nr3c2', 'Bcl2',
    ]),
    ('Neurotrophic', [
        'Ntrk2', 'Gfra1', 'Nrg3', 'Igf1r',
    ]),
    ('Neuronal development', [
        'Meis2', 'Foxg1', 'Sox11', 'Zfhx3', 'Nefl', 'Nefm',
        'Sema4a', 'Sema5a', 'Plxnd1',
    ]),
    ('Activity-dependent plasticity', [
        'Egr1', 'Arc', 'Homer1', 'Egr3',
    ]),
]
ordered_genes = [g for _, gs in GENE_BANDS for g in gs]
gene_band = {g: b for b, gs in GENE_BANDS for g in gs}

# MPOA parenting nuclei: force-included maternal-behavior anchors.
MPOA_ALLOWLIST = {'085 SI-MPO-LPO Lhx8 Gaba', '086 MPO-ADP Lhx8 Gaba',
                  '124 MPN-MPO-PVpo Hmx2 Glut'}
# Glial context columns: force-included so the neuronal pathway/gene signal
# reads against a fixed glial panel (shared with the glial deep-dive figure).
GLIAL_ALLOWLIST = {'318 Astro-NT NN', '319 Astro-TE NN', '323 Ependymal NN',
                   '326 OPC NN', '327 Oligo NN', '334 Microglia NN',
                   '335 BAM NN'}
# Strong-remodeling neurons whose signal sits largely outside the chosen
# pathways (so the >=3-hit rule drops them); force-included as columns.
NEURON_ALLOWLIST = {'032 L5 NP CTX Glut', '114 COAa-PAA-MEA Barhl2 Glut'}
ct_allowlist = MPOA_ALLOWLIST | GLIAL_ALLOWLIST | NEURON_ALLOWLIST

# Cellular neighbourhoods (same definition + order as 11_figure_2.py): columns
# are grouped, gapped and boxed by these brain-region blocks (class prefix).
NEIGHBOURHOODS = {
    'Pallium Glut':    ['01', '02', '03', '04'],
    'Pallium GABA':    ['06', '07'],
    'Subpallium GABA': ['05', '08', '09', '10'],
    'HY-EA':           ['11', '12', '13', '14', '15', '18', '19', '20', '24'],
    'Non-neuronal':    ['30', '31', '33', '34'],
}
NEIGHBOURHOOD_ORDER = list(NEIGHBOURHOODS)
SECTION_GAP = 0.8        # column-units of gap between neighbourhood sections
_cj = pd.read_csv(fc.SUBCLASS_CSV, usecols=['subclass', 'class'])
_sub2cls = dict(zip(_cj['subclass'], _cj['class'].str.split(' ', n=1).str[0]))
_pre2nb = {p: nb for nb, ps in NEIGHBOURHOODS.items() for p in ps}


def assign_class(ct):
    return _pre2nb.get(_sub2cls.get(ct), 'Non-neuronal')

CARD_GENES = ['Cntn1', 'Gria1', 'Gad2', 'Tac1', 'Fkbp5', 'Ntrk2', 'Meis2',
              'Egr1']
# Force the (IHC-validated) striatal population into the Fkbp5 card; it is
# significant (emp_p 0.013) but otherwise cut by the 6-row cap, which fills
# with the more-significant cortical/astrocyte populations.
card_ctx = {'Fkbp5': ['061 STR D1 Gaba']}
max_rows_map = {g: 6 for g in CARD_GENES}

# CCC chord row: 5 neuron-intrinsic (neuron<->neuron) themes, one chord each,
# mirroring the dotplot themes. Only cross-platform-validated (xenium &
# slidetags) L-R pairs are used; glia-niche / lipid / immune signalling is
# reserved for the glial + lipid figures.
CHORD_THEMES = ['Synaptic adhesion', 'Excitability', 'GABA & neuropeptide',
                'Neuronal development']
CHORD_THEME_LIGANDS = {
    'Synaptic adhesion':    ['Cntn1', 'Ncam1'],
    'Excitability':         ['Slc17a7'],
    'GABA & neuropeptide':  ['Gad2'],
    'Neuronal development': ['Sema4a', 'Sema5a'],
}
CHORD_THEME_TITLES = {
    'Synaptic adhesion':    'Synaptic adhesion\nCntn1 / Ncam1',
    'Excitability':         'Glutamatergic\nSlc17a7',
    'GABA & neuropeptide':  'GABAergic\nGad2',
    'Neuronal development': 'Axon guidance\nSema4a / Sema5a',
}
CANONICAL_LR_PAIRS = {
    ('Cntn1', 'Nrcam'), ('Ncam1', 'Robo1'),
    ('Slc17a7', 'Gria1'), ('Slc17a7', 'Grin1'),
    ('Gad2', 'Gabbr1'),
    ('Sema4a', 'Plxnd1'), ('Sema5a', 'Plxna3'),
}

# =============================================================================
# Data + matrices
# =============================================================================
gsea_all, gsea, real_nes = fc.load_gsea(working_dir)
de_sr_all, de_sr, de_pp = fc.load_de(working_dir)
subclass_colors = fc.load_subclass_colors()

ordered_cts, ct_class = fc.select_cell_types(
    gsea, ordered_pathways, ct_allowlist, assign_class, NEIGHBOURHOOD_ORDER,
    min_hits=3,
    candidate_expr=(pl.col('cell_type').str.contains(' Glut')
                    | pl.col('cell_type').str.contains(' Gaba')))

nlp_mat, nes_mat, d_mat, sig_mat_a = fc.gsea_matrices(
    gsea_all, real_nes, ordered_pathways, ordered_cts)
sp_coords, sp_expr, sp_in, pct = fc.load_spatial(
    working_dir, CARD_GENES, ordered_genes, ordered_cts)
lfc_mat, pct_mat, sig_mat_b, d_mat_b = fc.de_matrices(
    de_sr, de_sr_all, de_pp, ordered_genes, ordered_cts, pct)
cards, max_sp_n = fc.build_cards(
    CARD_GENES, card_ctx, max_rows_map, de_sr, de_sr_all, de_pp,
    ordered_cts, sp_in, sp_coords, sp_expr)

# neuron-centric pool, but keep the glial (NN) partners signalling to those
# neurons: the other figures' chords are lipid / microglia / vascular themes,
# so the glia->neuron adhesion/transmission niche shown here does not overlap.
chord_cell_set, chord_edges = fc.build_chord(
    f'{working_dir}/output/liana/inflow_diff.csv',
    CANONICAL_LR_PAIRS, CHORD_THEME_LIGANDS, k_neurons=14,
    neuron_intrinsic=True, include_nn=True, cell_allow=set(ordered_cts),
    mag_floor=False)

# =============================================================================
# Layout + draw
# =============================================================================
class_spans = fc.spans([ct_class[c] for c in ordered_cts])
band_spans = fc.spans([pathway_band[p] for p in ordered_pathways])
gene_band_spans = fc.spans([gene_band[g] for g in ordered_genes])
norm_nes, norm_lfc, nes_vmax, lfc_vmax = fc.make_norms(
    nes_mat, lfc_mat, d_mat_b)

# cards extend down to the bottom of the Panel B cell-type labels
card_extend = (fc.COL_ANNO_GAP_IN + fc.ANNO_W_IN
               + fc.label_drop_in(ordered_cts))
# forest-plot label column sized for full cell-type names (match dotplot)
label_w = fc.text_width_in([f'{c}  ***' for c in ordered_cts], pad_in=0.10)
# single chord row (5 themes) reserved below Panel B + its x-axis labels;
# CHORD_GAP clears the extended cards + column labels (chords shift down with them)
BOT_MARGIN, CHORD_GAP = 1.95, card_extend + 0.55
PANEL_GAP, ROW_GAP, TITLE_H = 0.28, 0.0, 0.45
cm = fc.card_metrics(len(ordered_pathways), len(ordered_genes),
                     len(ordered_cts), len(CARD_GENES), max_sp_n,
                     label_w_in=label_w, n_sections=len(class_spans),
                     section_gap=SECTION_GAP, card_extend_in=card_extend)
# span the row from the dotplot left y-axis to the right edge of the 2nd
# spatial plot (the 2-technology cards' rightmost spatial panel)
chord_span = (cm.cards_left_in + 2 * cm.SP_W_IN + fc.SP_GAP_IN
              - cm.ax_left_in)
NCHORD = len(CHORD_THEMES)
cpw, cph = fc.chord_panel_size(chord_span, NCHORD, NCHORD, PANEL_GAP,
                               aspect=0.96)
chord_h = fc.chord_band_height(cph, NCHORD, NCHORD, TITLE_H, ROW_GAP,
                               extra=0.20)
ax_b_bot = BOT_MARGIN + chord_h + CHORD_GAP

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
tleg_bot = fc.draw_dot_legends(fig, L, norm_nes, norm_lfc, nes_vmax, lfc_vmax,
                               [b for b, _ in GENE_BANDS], BAND_COLORS)
fc.draw_forest_legend(fig, L)
fc.draw_cards(fig, L, CARD_GENES, cards, sp_coords)

# CCC chord row (single row of 5 neuron-intrinsic themes)
specs = fc.chord_grid_specs(CHORD_THEMES, L.ax_left_in, BOT_MARGIN, cpw, cph,
                            NCHORD, PANEL_GAP, ROW_GAP, TITLE_H,
                            bottom_drop=0.20)
cy = (min(s[2] for s in specs) + max(s[2] + s[4] for s in specs)) / 2
fc.draw_chord_row(fig, L, specs, chord_edges, CHORD_THEME_LIGANDS,
                  CHORD_THEME_TITLES, chord_cell_set, subclass_colors,
                  ytitle_x_in=L.ax_left_in - 0.34, ytitle_cy_in=cy)
fc.draw_chord_legend(fig, L, tleg_bot - fc.GENE_PITCH, chord_cell_set,
                     subclass_colors)

fc.save(fig, f'{out_dir}/figure_3')
print(f'wrote {out_dir}/figure_3.png and .svg')
