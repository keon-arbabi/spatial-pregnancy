"""Neuron remodeling figure: Panel A (GSEA dotplot) + Panel B (DE dotplot)
+ gene cards + CCC chord row. Neuronal classes (Glut + GABA) plus glial
context columns. Orchestrates the shared blocks in 12_figure_helper.py.
"""
import os
import importlib

import numpy as np
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
# Config: pathways (7 bands), genes (7 bands), columns, cards, chords
# =============================================================================
PATHWAY_BANDS = [
    ('Synaptic adhesion', [
        'GOBP_SYNAPSE_ASSEMBLY',
        'GOBP_SYNAPSE_ORGANIZATION',
        'GOBP_HOMOPHILIC_CELL_CELL_ADHESION',
    ]),
    ('Excitability', [
        'GOBP_REGULATION_OF_MEMBRANE_POTENTIAL',
        'GOBP_REGULATION_OF_POSTSYNAPTIC_MEMBRANE_POTENTIAL',
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
    ]),
    ('Neurotrophic', [
        'GOBP_RESPONSE_TO_NERVE_GROWTH_FACTOR',
        'GOBP_NEUROTROPHIN_TRK_RECEPTOR_SIGNALING_PATHWAY',
        'GOBP_RESPONSE_TO_GROWTH_FACTOR',
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
    'GOBP_SYNAPSE_ORGANIZATION':                 'synapse organization',
    'GOBP_HOMOPHILIC_CELL_CELL_ADHESION':        'homophilic cell adhesion',
    'GOBP_REGULATION_OF_MEMBRANE_POTENTIAL':     'membrane potential',
    'GOBP_REGULATION_OF_POSTSYNAPTIC_MEMBRANE_POTENTIAL':
                                                 'postsynaptic potential',
    'GOBP_POTASSIUM_ION_TRANSPORT':              'potassium transport',
    'GOBP_SYNAPTIC_TRANSMISSION_GABAERGIC':      'GABAergic transmission',
    'GOBP_NEUROPEPTIDE_SIGNALING_PATHWAY':       'neuropeptide signaling',
    'GOBP_NEUROTRANSMITTER_SECRETION':           'neurotransmitter secretion',
    'GOBP_RESPONSE_TO_CORTICOSTEROID':           'corticosteroid response',
    'GOBP_RESPONSE_TO_STEROID_HORMONE':          'steroid hormone response',
    'GOBP_RESPONSE_TO_NERVE_GROWTH_FACTOR':      'NGF response',
    'GOBP_NEUROTROPHIN_TRK_RECEPTOR_SIGNALING_PATHWAY':
                                                 'Trk receptor signaling',
    'GOBP_RESPONSE_TO_GROWTH_FACTOR':            'growth factor response',
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
        'Cntn1', 'Sdk1', 'Nrcam', 'Ncam1', 'Dag1', 'Robo1', 'Cdh13',
    ]),
    ('Excitability', [
        'Gria1', 'Gria2', 'Grin1', 'Trpc5', 'Kcnh1', 'Sez6',
    ]),
    ('GABA & neuropeptide', [
        'Gad2', 'Gad1', 'Gabrb3', 'Drd1', 'Tac1', 'Ptprn2', 'Syt1',
    ]),
    ('Glucocorticoid stress', [
        'Fkbp5', 'Gpr83', 'Zbtb16', 'Ar',
    ]),
    ('Neurotrophic', [
        'Ntrk2', 'Ntrk3', 'Gfra1', 'Nrg3', 'Erbb4', 'Igf1r', 'Fgf1', 'Sort1',
    ]),
    # development ordered fate/TF (Foxg1..Sox11) then axon/structural
    ('Neuronal development', [
        'Foxg1', 'Zfhx3', 'Sox11',
        'Nefl', 'Nefm', 'Ntng1', 'Sema4a', 'Sema5a', 'Plxnd1',
    ]),
    ('Activity-dependent plasticity', [
        'Egr1', 'Arc', 'Homer1', 'Egr3', 'Nrn1',
    ]),
]
ordered_genes = [g for _, gs in GENE_BANDS for g in gs]
gene_band = {g: b for b, gs in GENE_BANDS for g in gs}

# Major theme super-groups: both dotplots' rows are split into these boxed
# super-sections with a gap between (mirrors fig-4's immune/vascular split).
THEME_GROUPS = [
    ('Synaptic & excitability remodeling',
     ['Synaptic adhesion', 'Excitability', 'GABA & neuropeptide']),
    ('Hormone & trophic signalling',
     ['Glucocorticoid stress', 'Neurotrophic']),
    ('Development & plasticity',
     ['Neuronal development', 'Activity-dependent plasticity']),
]
theme_group = {b: i for i, (_, bs) in enumerate(THEME_GROUPS) for b in bs}
GROUP_GAP = 0.9
GROUP_EXTRA = (len(THEME_GROUPS) - 1) * GROUP_GAP


def _grouped_rows(band_of_row):
    """row_y (GROUP_GAP inserted at theme-group boundaries) + row_sections
    [(lo, hi)] index ranges, one per theme-group."""
    ys, secs, gap, start = [], [], 0.0, 0
    prev = theme_group[band_of_row[0]]
    for i, band in enumerate(band_of_row):
        g = theme_group[band]
        if g != prev:
            secs.append((start, i - 1))
            gap += GROUP_GAP
            start, prev = i, g
        ys.append(i + gap)
    secs.append((start, len(band_of_row) - 1))
    return ys, secs


row_y_a, row_sections_a = _grouped_rows(
    [pathway_band[p] for p in ordered_pathways])
row_y_b, row_sections_b = _grouped_rows(
    [gene_band[g] for g in ordered_genes])

# MPOA parenting nuclei: force-included maternal-behavior anchors.
MPOA_ALLOWLIST = {'085 SI-MPO-LPO Lhx8 Gaba', '086 MPO-ADP Lhx8 Gaba',
                  '124 MPN-MPO-PVpo Hmx2 Glut'}
# Glial context columns: force-included so the neuronal pathway/gene signal
# reads against a fixed glial panel (shared with the glial deep-dive figure).
GLIAL_ALLOWLIST = {'318 Astro-NT NN', '319 Astro-TE NN', '323 Ependymal NN',
                   '326 OPC NN', '327 Oligo NN', '334 Microglia NN',
                   '335 BAM NN'}
# Strong-remodeling neurons kept as columns regardless of hit count; both now
# qualify on their own under the >=1-hit rule, retained for stability.
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

# highlighted-gene cards (spatial map + forest), reordered to match the dotplot
CARD_GENES = ['Cntn1', 'Gria1', 'Gad2', 'Ar', 'Fkbp5', 'Ntrk2', 'Nefl', 'Egr1']
CARD_GENES = [g for g in ordered_genes if g in set(CARD_GENES)]
# force the IHC-validated striatal population into the Fkbp5 card (the 6-row
# cap otherwise fills it with more-significant cortical/astrocyte populations)
card_ctx = {'Fkbp5': ['061 STR D1 Gaba']}
max_rows_map = {g: 6 for g in CARD_GENES}

# CCC chord row: 3 neuron-intrinsic themes, one chord each, mirroring the
# dotplot themes. Only cross-platform (xenium & slidetags) L-R pairs are used;
# glia-niche / lipid / immune signalling is reserved for the glial + lipid
# figures. The development chord (Sema->Plxn) was dropped as drawn-level noise.
CHORD_THEMES = ['Synaptic adhesion', 'Excitability', 'GABA & neuropeptide']
CHORD_THEME_LIGANDS = {
    'Synaptic adhesion':    ['Cntn1', 'Ncam1'],
    'Excitability':         ['Slc17a7'],
    'GABA & neuropeptide':  ['Gad2'],
}
CHORD_THEME_TITLES = {
    'Synaptic adhesion':    'Synaptic adhesion signalling\nCntn1 / Ncam1',
    'Excitability':         'Glutamatergic signalling\nSlc17a7',
    'GABA & neuropeptide':  'GABAergic signalling\nGad2',
}
CANONICAL_LR_PAIRS = {
    ('Cntn1', 'Nrcam'), ('Ncam1', 'Robo1'),
    ('Slc17a7', 'Gria1'), ('Slc17a7', 'Grin1'),
    ('Gad2', 'Gabbr1'),
}

# =============================================================================
# Data + matrices
# =============================================================================
gsea_all, gsea, real_nes = fc.load_gsea(working_dir)
de_sr_all, de_sr, de_pp = fc.load_de(working_dir)
subclass_colors = fc.load_subclass_colors()

ordered_cts, ct_class = fc.select_cell_types(
    gsea, ordered_pathways, ct_allowlist, assign_class, NEIGHBOURHOOD_ORDER,
    min_hits=1,
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

# Panel A trajectory: median NES across platforms per (contrast, pathway, ct);
# arcs track the pregnancy-responsive (sig-in-pregnancy) neuronal pairs.
_nes = (pl.read_parquet(f'{working_dir}/output/gsea/perms/real_gsea.parquet')
        .filter(pl.col('pathway').is_in(ordered_pathways))
        .group_by(['contrast', 'pathway', 'cell_type'])
        .agg(pl.col('NES').median().alias('nes')))
_nl = {(r['contrast'], r['pathway'], r['cell_type']): r['nes']
       for r in _nes.iter_rows(named=True)}
_resp = [(r['pathway'], r['cell_type'])
         for r in gsea.filter(pl.col('pathway').is_in(ordered_pathways))
         .iter_rows(named=True)
         if any(t in r['cell_type'] for t in (' Glut', ' Gaba', 'IMN'))]


def band_arc(band):
    """(preg_NES, postpart_NES) rows for a band's responsive pairs."""
    rows = [(_nl.get(('PREG_vs_CTRL', p, c)),
             _nl.get(('POSTPART_vs_CTRL', p, c)))
            for p, c in _resp if pathway_band[p] == band]
    rows = [(a, b) for a, b in rows if a is not None and b is not None]
    return np.array(rows) if rows else np.empty((0, 2))

# =============================================================================
# Layout + draw
# =============================================================================
class_spans = fc.spans([ct_class[c] for c in ordered_cts])
band_spans = fc.spans([pathway_band[p] for p in ordered_pathways])
gene_band_spans = fc.spans([gene_band[g] for g in ordered_genes])
norm_nes, norm_lfc, nes_vmax, lfc_vmax = fc.make_norms(
    nes_mat, lfc_mat, d_mat_b)

# drop of the rotated column labels below Panel B (used for chord clearance)
card_extend = (fc.COL_ANNO_GAP_IN + fc.ANNO_W_IN
               + fc.label_drop_in(ordered_cts))
# forest-plot label column sized for full cell-type names (match dotplot)
label_w = fc.text_width_in([f'{c}  ***' for c in ordered_cts], pad_in=0.10)
# single chord row reserved below Panel B + its x-axis labels; CHORD_GAP clears
# the column labels (chords shifted up toward the dotplots, titles still clear)
BOT_MARGIN, CHORD_GAP = 1.95, card_extend + 0.25
PANEL_GAP, ROW_GAP, TITLE_H = 0.28, 0.0, 0.45
# extra dotplot->cards gap to hold the rotated theme-group titles
CARDS_PAD = 0.30
cm = fc.card_metrics(len(ordered_pathways), len(ordered_genes),
                     len(ordered_cts), len(CARD_GENES), max_sp_n,
                     label_w_in=label_w, n_sections=len(class_spans),
                     section_gap=SECTION_GAP, n_path_extra=GROUP_EXTRA,
                     n_gene_extra=GROUP_EXTRA, card_extend_in=0,
                     cards_pad_in=CARDS_PAD)
# span the row across the dotplot width only (left to right of Panels A/B)
chord_span = cm.ax_w_in
NCHORD = len(CHORD_THEMES)
cpw, cph = fc.chord_panel_size(chord_span, NCHORD, NCHORD, PANEL_GAP,
                               aspect=0.96)
chord_h = fc.chord_band_height(cph, NCHORD, NCHORD, TITLE_H, ROW_GAP,
                               extra=0.20)
ax_b_bot = BOT_MARGIN + chord_h + CHORD_GAP

L = fc.core_layout(len(ordered_pathways), len(ordered_genes),
                   len(ordered_cts), len(CARD_GENES), max_sp_n, ax_b_bot,
                   label_w_in=label_w, n_sections=len(class_spans),
                   section_gap=SECTION_GAP, n_path_extra=GROUP_EXTRA,
                   n_gene_extra=GROUP_EXTRA, card_extend_in=0,
                   cards_pad_in=CARDS_PAD)
L.col_x = fc.column_positions(class_spans, len(ordered_cts), SECTION_GAP)

# reserve Panel A (trajectory arcs) above the dotplots: the cards stay top-
# anchored (float) while the dotplots + chords shift down
ARC_H, ARC_GAP, ARC_FGAP = 2.2, 1.0, 0.35
arc_bot_in = L.ax_a_top_in + ARC_GAP
L.fig_h += ARC_GAP + ARC_H
L.cards_top_in += ARC_GAP + ARC_H

fig = plt.figure(figsize=(L.fig_w, L.fig_h))
ax_a = fc.add_axes(fig, L, L.ax_left_in, L.ax_a_bot_in, L.ax_w_in, L.ax_h_a_in)
ax_b = fc.add_axes(fig, L, L.ax_left_in, L.ax_b_bot_in, L.ax_w_in, L.ax_h_b_in)

fc.draw_panel_a(ax_a, L, nlp_mat, nes_mat, d_mat, sig_mat_a, norm_nes,
                ordered_pathways, PATHWAY_LABELS, class_spans, band_spans,
                row_y=row_y_a, row_sections=row_sections_a)
fc.draw_panel_b(ax_b, L, lfc_mat, pct_mat, d_mat_b, sig_mat_b, norm_lfc,
                ordered_genes, class_spans, gene_band_spans, CARD_GENES,
                row_y=row_y_b, row_sections=row_sections_b)

prefix_labels = [f'{fc.numeric_prefix(ct):03d}' for ct in ordered_cts]
fc.draw_col_anno(fig, L, L.ax_a_bot_in, ordered_cts, prefix_labels,
                 subclass_colors)
fc.draw_col_anno(fig, L, L.ax_b_bot_in, ordered_cts, ordered_cts,
                 subclass_colors)
fc.draw_band_anno(fig, L, L.ax_a_bot_in, L.ax_h_a_in, len(ordered_pathways),
                  [pathway_band[p] for p in ordered_pathways], BAND_COLORS,
                  row_y=row_y_a)
fc.draw_band_anno(fig, L, L.ax_b_bot_in, L.ax_h_b_in, len(ordered_genes),
                  [gene_band[g] for g in ordered_genes], BAND_COLORS,
                  row_y=row_y_b)

# rotated super-group titles, one per theme-group, right of the colour strip
GROUP_LABELS = ['Synaptic & excitability\nremodeling',
                'Hormone & trophic\nsignalling', 'Development &\nplasticity']
_gx = L.anno_x_in + fc.ANNO_W_IN + 0.20   # near the dotplots; keep cards gap
fc.draw_group_labels(fig, L, L.ax_a_bot_in, L.ax_h_a_in, row_y_a,
                     row_sections_a, GROUP_LABELS, _gx)
fc.draw_group_labels(fig, L, L.ax_b_bot_in, L.ax_h_b_in, row_y_b,
                     row_sections_b, GROUP_LABELS, _gx)
tleg_bot = fc.draw_dot_legends(fig, L, norm_nes, norm_lfc, nes_vmax, lfc_vmax,
                               [b for b, _ in GENE_BANDS], BAND_COLORS)
fc.draw_forest_legend(fig, L)
fc.draw_cards(fig, L, CARD_GENES, cards, sp_coords)

# --- Panel A: peripartum trajectory arcs (spans the dotplot width) -----------
XS = np.array([0.0, 1.0, 2.0])
XLAB = ['Null', 'Preg', 'Post']
_arcs = {b: band_arc(b) for _, bs in THEME_GROUPS for b in bs}
_meds = [np.median(_arcs[b][:, j]) for b in _arcs for j in (0, 1)
         if len(_arcs[b])]
arc_ylim = (min(_meds) - 0.5, max(_meds) + 0.4)
arc_rng = arc_ylim[1] - arc_ylim[0]
arc_fw = (L.ax_w_in - 2 * ARC_FGAP) / 3.0
for k, (_, bands) in enumerate(THEME_GROUPS):
    fx = L.ax_left_in + k * (arc_fw + ARC_FGAP)
    ax = fc.add_axes(fig, L, fx, arc_bot_in, arc_fw, ARC_H)
    ax.axhline(0, color='#999', lw=0.8, ls='--', zorder=1)
    # dashed frame on the pregnant column (Panels B/C detail this preg v null)
    ax.add_patch(plt.Rectangle((0.6, arc_ylim[0] + 0.04 * arc_rng), 0.8,
                 arc_rng * 0.92, fill=False, edgecolor='black', lw=1.0,
                 ls=(0, (4, 3)), zorder=1.5))
    for b in bands:
        med = [0, np.median(_arcs[b][:, 0]), np.median(_arcs[b][:, 1])]
        col = BAND_COLORS[b]
        ax.plot(XS, med, color=col, lw=2.4, marker='o', ms=5, zorder=4,
                solid_capstyle='round')
        ax.text(2.12, med[2], b, color=col, fontsize=7.5, va='center',
                ha='left', fontweight='bold')
    ax.set_xlim(-0.15, 4.6); ax.set_ylim(arc_ylim)
    ax.set_xticks(XS); ax.set_xticklabels(XLAB, fontsize=fc.LABEL_FS)
    ax.tick_params(length=2, labelsize=fc.LABEL_FS)
    if k == 0:
        ax.set_yticks([-1, 0, 1])
        ax.set_ylabel('median NES', fontsize=fc.LABEL_FS)
    else:
        ax.set_yticks([])
    fig.text((fx + arc_fw / 2) / L.fig_w,
             (arc_bot_in + ARC_H + 0.14) / L.fig_h, GROUP_LABELS[k],
             ha='center', va='bottom', fontsize=fc.TITLE_FS)

# CCC chord row (single row of 3 neuron-intrinsic themes)
specs = fc.chord_grid_specs(CHORD_THEMES, L.ax_left_in, BOT_MARGIN, cpw, cph,
                            NCHORD, PANEL_GAP, ROW_GAP, TITLE_H,
                            bottom_drop=0.20)
# slightly shrink the chord circles (keeps titles/sector labels the same size)
CHORD_SHRINK = 0.86
specs = [(t, lft + (w - w * CHORD_SHRINK) / 2, bt + (h - h * CHORD_SHRINK) / 2,
          w * CHORD_SHRINK, h * CHORD_SHRINK)
         for (t, lft, bt, w, h) in specs]
cy = (min(s[2] for s in specs) + max(s[2] + s[4] for s in specs)) / 2
fc.draw_chord_row(fig, L, specs, chord_edges, CHORD_THEME_LIGANDS,
                  CHORD_THEME_TITLES, chord_cell_set, subclass_colors,
                  ytitle_x_in=L.ax_left_in - 0.34, ytitle_cy_in=cy)
fc.draw_chord_legend(fig, L, tleg_bot - fc.GENE_PITCH, chord_cell_set,
                     subclass_colors)

fc.save(fig, f'{out_dir}/figure_3')
print(f'wrote {out_dir}/figure_3.png and .svg')

# =============================================================================
# IF validation (Fkbp5): striatal FKBP5+ area rises in pregnancy, validating
# the Fkbp5 card (Fkbp5 up in 061 STR D1). Prism bar+scatter, mean +/- SEM,
# unpaired t-test.
# =============================================================================
IF_XLSX = f'{working_dir}/input/IF validation quantification.xlsx'
IF_PANELS = [
    ('NONP_P_FKBP5', 'Striatal FKBP5+ area (A.U.)'),   # FKBP5 / DAPI, striatum
]
_vp = fc.if_validation_figure(IF_XLSX, IF_PANELS,
                              f'{out_dir}/figure_3_validation', panel_w=2.6)
print(f'wrote {out_dir}/figure_3_validation.png/.svg  (p: FKBP5={_vp[0]:.3g})')
