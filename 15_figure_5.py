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

import numpy as np
import polars as pl
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
        'Apoe', 'Clu', 'Srebf2', 'Hmgcs1', 'Hmgcr', 'Idi1', 'Sqle',
    ]),
    ('Lipoprotein uptake & efflux', [
        'Sort1', 'Sorl1', 'Srebf1', 'Cd81', 'Abca1', 'Abca8b',
    ]),
]
ordered_genes = [g for _, gs in GENE_BANDS for g in gs]
gene_band = {g: b for b, gs in GENE_BANDS for g in gs}

# Major theme super-groups: both dotplots' rows are split into these boxed
# super-sections with a gap between (cell-intrinsic synthesis vs trafficking);
# also the facets of the peripartum trajectory panel (Panel A).
THEME_GROUPS = [
    ('Intrinsic lipid metabolism',
     ['Membrane lipid', 'Ceramide & sphingolipid', 'Fatty acid catabolism']),
    ('Cholesterol & lipoprotein trafficking',
     ['Sterol & lipoprotein supply', 'Lipoprotein uptake & efflux']),
]
theme_group = {b: i for i, (_, bs) in enumerate(THEME_GROUPS) for b in bs}
GROUP_GAP = 0.9
GROUP_EXTRA = (len(THEME_GROUPS) - 1) * GROUP_GAP
GROUP_LABELS = ['Intrinsic lipid\nmetabolism',
                'Cholesterol & lipoprotein\ntrafficking']


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
    min_hits=1)

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

# Panel A trajectory: median NES across platforms per (contrast, pathway, ct);
# arcs track the pregnancy-responsive (sig-in-pregnancy) pairs per band. The
# lipid signal is pan-cellular, so responsive pairs are not restricted to
# neurons (unlike the neuron figure).
_nes = (pl.read_parquet(f'{working_dir}/output/gsea/perms/real_gsea.parquet')
        .filter(pl.col('pathway').is_in(ordered_pathways))
        .group_by(['contrast', 'pathway', 'cell_type'])
        .agg(pl.col('NES').median().alias('nes')))
_nl = {(r['contrast'], r['pathway'], r['cell_type']): r['nes']
       for r in _nes.iter_rows(named=True)}
_resp = [(r['pathway'], r['cell_type'])
         for r in gsea.filter(pl.col('pathway').is_in(ordered_pathways))
         .iter_rows(named=True)]


def band_arc(band):
    """(preg_NES, postpart_NES) rows for a band's responsive pairs."""
    rows = [(_nl.get(('PREG_vs_CTRL', p, c)),
             _nl.get(('POSTPART_vs_CTRL', p, c)))
            for p, c in _resp if pathway_band[p] == band]
    rows = [(a, b) for a, b in rows if a is not None and b is not None]
    return np.array(rows) if rows else np.empty((0, 2))

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
# extra dotplot->cards gap to hold the rotated theme-group titles
CARDS_PAD = 0.30
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
                     section_gap=SECTION_GAP, n_path_extra=GROUP_EXTRA,
                     n_gene_extra=GROUP_EXTRA, card_extend_in=card_extend,
                     cards_pad_in=CARDS_PAD)
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
                   section_gap=SECTION_GAP, n_path_extra=GROUP_EXTRA,
                   n_gene_extra=GROUP_EXTRA, card_extend_in=card_extend,
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
_gx = L.anno_x_in + fc.ANNO_W_IN + 0.20
fc.draw_group_labels(fig, L, L.ax_a_bot_in, L.ax_h_a_in, row_y_a,
                     row_sections_a, GROUP_LABELS, _gx)
fc.draw_group_labels(fig, L, L.ax_b_bot_in, L.ax_h_b_in, row_y_b,
                     row_sections_b, GROUP_LABELS, _gx)
tleg_bot = fc.draw_dot_legends(fig, L, norm_nes, norm_lfc, nes_vmax,
                               lfc_vmax, [b for b, _ in GENE_BANDS],
                               BAND_COLORS)
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
n_grp = len(THEME_GROUPS)
arc_fw = (L.ax_w_in - (n_grp - 1) * ARC_FGAP) / n_grp


def _place_arc_labels(ax, items, min_gap):
    """Draw right-side band labels at their Post-endpoint y, nudged apart
    (order preserved) so near-coincident arcs don't overprint."""
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for (y0, txt, col), y in zip(items, ys):
        ax.text(2.12, y, txt, color=col, fontsize=7.5, va='center',
                ha='left', fontweight='bold')


for k, (_, bands) in enumerate(THEME_GROUPS):
    fx = L.ax_left_in + k * (arc_fw + ARC_FGAP)
    ax = fc.add_axes(fig, L, fx, arc_bot_in, arc_fw, ARC_H)
    ax.axhline(0, color='#999', lw=0.8, ls='--', zorder=1)
    # dashed frame on the pregnant column (Panels B/C detail this preg v null)
    ax.add_patch(plt.Rectangle((0.6, arc_ylim[0] + 0.04 * arc_rng), 0.8,
                 arc_rng * 0.92, fill=False, edgecolor='black', lw=1.0,
                 ls=(0, (4, 3)), zorder=1.5))
    labs = []
    for b in bands:
        med = [0, np.median(_arcs[b][:, 0]), np.median(_arcs[b][:, 1])]
        col = BAND_COLORS[b]
        ax.plot(XS, med, color=col, lw=2.4, marker='o', ms=5, zorder=4,
                solid_capstyle='round')
        labs.append((med[2], b, col))
    _place_arc_labels(ax, labs, arc_rng * 0.075)
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

# =============================================================================
# IF validation: the cholesterol-supply DOWN arm at the protein level. ApoE
# carriers (microglia, OPC) and HMGCR (astrocyte cholesterol synthesis) all
# fall in pregnancy, matching the Apoe/Hmgcr DE cards and the trafficking-DOWN
# band. Prism bar+scatter, mean +/- SEM, unpaired t-test.
# =============================================================================
IF_XLSX = f'{working_dir}/input/IF validation quantification.xlsx'
IF_PANELS = [
    ('NONP_P_APOE_IBA1',   'Microglial ApoE+ area (A.U.)'),
    ('NONP_P_APOE_NG2',    'OPC ApoE+ area (A.U.)'),
    ('NONP_P_HMGCR_S100b', 'Astrocytic HMGCR+ area (A.U.)'),
]
_vp = fc.if_validation_figure(IF_XLSX, IF_PANELS,
                              f'{out_dir}/figure_5_validation', panel_w=2.6)
print(f'wrote {out_dir}/figure_5_validation.png/.svg  '
      f'(p: ApoE-microglia={_vp[0]:.3g}, ApoE-OPC={_vp[1]:.3g}, '
      f'HMGCR-astro={_vp[2]:.3g})')
