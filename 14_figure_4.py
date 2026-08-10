"""Figure 4: unified microglia + vascular glial-niche figure."""
import os
import importlib

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

fc = importlib.import_module('12_figure_helper')

fc.setup_style()
working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures'
os.makedirs(out_dir, exist_ok=True)

# --- config -----------------------------------------------------------------
# The immune arm was originally split four ways, but those bands' per-cell-type
# NES profiles are largely redundant (mean pairwise r = 0.65, vs 0.42 for the
# vascular bands) and trace a single trajectory. Immune activation and
# inflammatory response (r = 0.69) are therefore merged; cytokine signalling is
# the one distinct axis (r = 0.50-0.62 against the rest) and the myeloid
# effector band is kept for its clean microglia-restricted gene set.
PATHWAY_BANDS = [
    ('Innate immune activation', [
        'GOBP_ADAPTIVE_IMMUNE_RESPONSE',
        'GOBP_ACTIVATION_OF_IMMUNE_RESPONSE',
        'GOBP_INFLAMMATORY_RESPONSE',
        'GOBP_ACUTE_INFLAMMATORY_RESPONSE',
    ]),
    ('Cytokine signalling', [
        'GOBP_CYTOKINE_PRODUCTION',
        'GOBP_CYTOKINE_MEDIATED_SIGNALING_PATHWAY',
    ]),
    ('Myeloid effector', [
        'GOBP_MYELOID_LEUKOCYTE_MIGRATION',
        'GOBP_LEUKOCYTE_DIFFERENTIATION',
    ]),
    ('Angiogenic sprouting', [
        'GOBP_VASCULATURE_DEVELOPMENT',
        'GOBP_SPROUTING_ANGIOGENESIS',
    ]),
    ('Endothelial dynamics', [
        'GOBP_ENDOTHELIAL_CELL_MIGRATION',
        'GOBP_ENDOTHELIAL_CELL_PROLIFERATION',
    ]),
    ('Barrier & ECM', [
        'GOBP_COLLAGEN_BIOSYNTHETIC_PROCESS',
        'GOBP_EXTRACELLULAR_MATRIX_ASSEMBLY',
    ]),
]
ordered_pathways = [p for _, ps in PATHWAY_BANDS for p in ps]
pathway_band = {p: b for b, ps in PATHWAY_BANDS for p in ps}

PATHWAY_LABELS = {
    'GOBP_ADAPTIVE_IMMUNE_RESPONSE': 'adaptive immune response',
    'GOBP_ACTIVATION_OF_IMMUNE_RESPONSE': 'activation of immune response',
    'GOBP_CYTOKINE_PRODUCTION': 'cytokine production',
    'GOBP_CYTOKINE_MEDIATED_SIGNALING_PATHWAY': 'cytokine-mediated signaling',
    'GOBP_INFLAMMATORY_RESPONSE': 'inflammatory response',
    'GOBP_ACUTE_INFLAMMATORY_RESPONSE': 'acute inflammatory response',
    'GOBP_MYELOID_LEUKOCYTE_MIGRATION': 'myeloid leukocyte migration',
    'GOBP_LEUKOCYTE_DIFFERENTIATION': 'leukocyte differentiation',
    'GOBP_VASCULATURE_DEVELOPMENT': 'vasculature development',
    'GOBP_SPROUTING_ANGIOGENESIS': 'sprouting angiogenesis',
    'GOBP_ENDOTHELIAL_CELL_MIGRATION': 'endothelial migration',
    'GOBP_ENDOTHELIAL_CELL_PROLIFERATION': 'endothelial proliferation',
    'GOBP_COLLAGEN_BIOSYNTHETIC_PROCESS': 'collagen biosynthesis',
    'GOBP_EXTRACELLULAR_MATRIX_ASSEMBLY': 'ECM assembly',
}

# six distinct hues, only one blue (blue also reads as "down" in the dot
# colormap); adjacent bands within each super-section are maximally separated
BAND_COLORS = {
    'Innate immune activation': '#882255',   # wine
    'Cytokine signalling':      '#E69F00',   # orange
    'Myeloid effector':         '#CC79A7',   # pink
    'Angiogenic sprouting':     '#009E73',   # green
    'Endothelial dynamics':     '#0072B2',   # blue
    'Barrier & ECM':            '#999933',   # olive
}

# Major theme super-groups: both dotplots' rows are split into these boxed
# super-sections with a gap between (immune / vascular); also the facets of the
# peripartum trajectory panel (Panel A).
THEME_GROUPS = [
    # named "inflammatory signalling" rather than "immune activation": the
    # latter collides with maternal immune activation (MIA), an established and
    # entirely different paradigm (induced inflammation, offspring brain)
    ('Microglial & inflammatory signalling',
     ['Innate immune activation', 'Cytokine signalling', 'Myeloid effector']),
    ('Vascular remodeling & barrier',
     ['Angiogenic sprouting', 'Endothelial dynamics', 'Barrier & ECM']),
]
theme_group = {b: i for i, (_, bs) in enumerate(THEME_GROUPS) for b in bs}
GROUP_GAP = 0.9
GROUP_EXTRA = (len(THEME_GROUPS) - 1) * GROUP_GAP
GROUP_LABELS = ['Microglial &\ninflammatory signalling',
                'Vascular remodeling\n& barrier']

GENE_BANDS = [
    # Cd27 (lymphocyte TNFRSF7; sig only in astrocytes at 18% detection) and
    # Isg15 (8-9% detection, BAM/SMC only) dropped as unreliable; Cd81 dropped
    # as a ubiquitous tetraspanin already carried by figure 5's uptake band.
    # P2rx7 added: microglial purinergic/inflammasome receptor, UP in
    # microglia (68%), OPC (66%) and oligodendrocytes (37%).
    ('Innate immune activation',
     ['Cd38', 'Lacc1', 'P2rx7', 'Nlrp3', 'Txnip', 'Grn', 'Ifngr1', 'Lyn']),
    ('Cytokine signalling', ['Lifr', 'Il6st', 'Il4ra', 'Il10ra', 'Stat3']),
    ('Myeloid effector', ['Mertk', 'Csf3r', 'Pik3cd', 'Trpm2', 'Adgre1']),
    ('Angiogenic sprouting',
     ['Vegfa', 'Flt1', 'Notch1', 'Eng', 'Id1', 'Bmpr2']),
    ('Endothelial dynamics', ['Rgcc', 'Kdr', 'Cxcl12', 'Pecam1', 'Cdh5']),
    ('Barrier & ECM', ['Slc2a1', 'Mfsd2a', 'Col1a1', 'Tnxb']),
]

NN_NICHE = {
    '318 Astro-NT NN', '319 Astro-TE NN', '323 Ependymal NN', '326 OPC NN',
    '327 Oligo NN', '330 VLMC NN', '331 Peri NN', '332 SMC NN', '333 Endo NN',
    '334 Microglia NN', '335 BAM NN',
}
NEURON_ALLOWLIST = {        # not shown as columns; flagged in rank_report only
    '022 L5 ET CTX Glut', '046 Vip Gaba', '004 L6 IT CTX Glut', '053 Sst Gaba',
}
ct_allowlist = set(NN_NICHE)
CLASS_ORDER = ['Non-neuronal']


def assign_class(ct):
    return 'Non-neuronal'


CARD_GENES = ['Lifr', 'Lyn', 'Mertk',
              'Vegfa', 'Flt1', 'Cxcl12', 'Slc2a1']
_mg = ['334 Microglia NN']
card_ctx = {'Lifr': _mg,
            'Lyn': _mg, 'Mertk': _mg + ['335 BAM NN'],
            'Vegfa': ['326 OPC NN', '330 VLMC NN'],
            'Flt1': ['333 Endo NN', '331 Peri NN'],
            'Cxcl12': ['331 Peri NN', '333 Endo NN'],
            'Slc2a1': ['318 Astro-NT NN', '333 Endo NN']}
max_rows_map = {g: 8 for g in CARD_GENES}

CHORD_THEMES = ['TGFb', 'Notch', 'Ang2_Cxcl12']
CHORD_THEME_LIGANDS = {
    'TGFb':        ['Tgfb1', 'Tgfb2'],
    'Notch':       ['Dll4', 'Jag1', 'Jag2'],
    'Ang2_Cxcl12': ['Angpt2', 'Cxcl12'],
}
CHORD_THEME_TITLES = {        # convention as figure 3: 'descriptor\nligand(s)'
    'TGFb':        'Homeostatic TGF-β\nTgfb1 / Tgfb2',
    'Notch':       'Vessel maturation\nDll4 / Jag1/2',
    'Ang2_Cxcl12': 'Pericyte recruitment\nAngpt2 / Cxcl12',
}
CANONICAL_LR_PAIRS = {
    ('Tgfb1', 'Tgfbr1_Tgfbr2'), ('Tgfb1', 'Eng'), ('Tgfb1', 'Itgb1'),
    ('Tgfb1', 'Itgb5'), ('Tgfb2', 'Tgfbr1_Tgfbr2'),
}
for _r in ['Notch1', 'Notch3', 'Notch4']:
    CANONICAL_LR_PAIRS.add(('Dll4', _r))
for _l in ['Jag1', 'Jag2']:
    for _r in ['Notch1', 'Notch3']:
        CANONICAL_LR_PAIRS.add((_l, _r))
for _r in ['Tek', 'Tie1']:
    CANONICAL_LR_PAIRS.add(('Angpt2', _r))
for _r in ['Cxcr4', 'Ackr3']:
    CANONICAL_LR_PAIRS.add(('Cxcl12', _r))


# --- selection report (printed on run) --------------------------------------
BAND_PATHWAY_POOLS = {
    'Innate immune activation':
        r'IMMUNE_RESPONSE|IMMUNE_SYSTEM|IMMUNE_EFFECTOR|ANTIGEN|INFLAMMAT',
    'Cytokine signalling': r'CYTOKINE|INTERLEUKIN|TUMOR_NECROSIS|INTERFERON',
    'Myeloid effector':
        r'MACROPHAGE|MYELOID|LEUKOCYTE|PHAGOCYT|MICROGLIAL|CHEMOTAXIS',
    'Angiogenic sprouting':
        r'ANGIOGEN|VASCULATURE_DEV|VASCULOGEN|BLOOD_VESSEL_DEV|SPROUT',
    'Endothelial dynamics':
        r'ENDOTHELIAL_CELL_(PROLIF|MIGRAT|DIFFER)|BLOOD_VESSEL_ENDOTHELIAL',
    'Barrier & ECM':
        r'ENDOTHELIAL_BARRIER|TIGHT_JUNCTION|COLLAGEN|BASEMENT_MEMBRANE'
        r'|EXTRACELLULAR_MATRIX',
}


def _hdr(t):
    print('\n' + '=' * 70 + f'\n[rank_report] {t}\n' + '=' * 70)


def _signed(df):
    return df.with_columns(
        pl.max_horizontal('nlp_up', 'nlp_down').alias('nlp'),
        pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
        .then(pl.col('emp_p_up')).otherwise(pl.col('emp_p_down'))
        .alias('emp_p'))


def rank_report():
    import re
    g = pl.read_csv(f'{working_dir}/output/gsea/sumrank_gsea_results.csv')
    gs = _signed(g.filter((pl.col('contrast') == 'PREG_vs_CTRL')
                          & (pl.col('D') >= 2)))
    sig = gs.filter(pl.col('emp_p') <= 0.05)
    glial = sig.filter(pl.col('cell_type').is_in(list(NN_NICHE)))
    allp = g['pathway'].unique().to_list()

    _hdr('pathways per band (NN-niche n_ct, peak nlp)')
    sc = {r['pathway']: r for r in glial.group_by('pathway').agg(
        pl.col('cell_type').n_unique().alias('n_ct'),
        pl.col('nlp').max().alias('peak')).iter_rows(named=True)}
    for band, pat in BAND_PATHWAY_POOLS.items():
        cands = sorted((p for p in allp if re.search(pat, p) and p in sc),
                       key=lambda p: (-sc[p]['n_ct'], -sc[p]['peak']))
        chosen = {p for b, ps in PATHWAY_BANDS if b == band for p in ps}
        print(f'\n{band}:')
        for p in cands[:6]:
            mark = '*' if p in chosen else ' '
            print(f'  {mark} n_ct={sc[p]["n_ct"]:2d} '
                  f'peak={sc[p]["peak"]:5.2f}  {p}')

    _hdr('neuron context (glial-theme sig hits)')
    neu = sig.filter(pl.col('pathway').is_in(ordered_pathways)
                     & ~pl.col('cell_type').str.contains(' NN'))
    for r in (neu.group_by('cell_type').agg(pl.len().alias('n'))
              .sort('n', descending=True).head(8).iter_rows(named=True)):
        mark = '*' if r['cell_type'] in NEURON_ALLOWLIST else ' '
        print(f'  {mark} {r["n"]:2d}  {r["cell_type"]}')

    _hdr('genes per band (NN-niche n_ct, peak nlp, LE freq)')
    de = _signed(pl.read_csv(f'{working_dir}/output/de/sumrank_results.csv')
                 .filter((pl.col('contrast') == 'PREG_vs_CTRL')
                         & (pl.col('D') >= 2)
                         & (pl.col('ref_pct_detected') >= 5.0)))
    de_sig = de.filter((pl.col('emp_p') <= 0.05)
                       & pl.col('cell_type').is_in(list(NN_NICHE)))
    gstat = {r['gene']: r for r in de_sig.group_by('gene').agg(
        pl.col('cell_type').n_unique().alias('n'),
        pl.col('nlp').max().alias('pk')).iter_rows(named=True)}
    le_cols = ['leading_edge_merfish', 'leading_edge_slidetags',
               'leading_edge_xenium']
    for band, paths in PATHWAY_BANDS:
        rows = g.filter((g['contrast'] == 'PREG_vs_CTRL')
                        & g['pathway'].is_in(paths)
                        & g['cell_type'].is_in(list(NN_NICHE)))
        freq = {}
        for c in le_cols:
            for v in rows[c].drop_nulls().to_list():
                for x in str(v).split(','):
                    freq[x] = freq.get(x, 0) + 1
        cand = sorted(((x, gstat[x]['n'], gstat[x]['pk'], freq[x])
                       for x in freq if x in gstat),
                      key=lambda t: (-t[1], -t[2]))
        chosen = {x for b, gs in GENE_BANDS if b == band for x in gs}
        print(f'\n{band}:')
        for x, n, pk, lf in cand[:8]:
            mark = '*' if x in chosen else ' '
            print(f'  {mark} n_ct={n:2d} peak={pk:4.2f} LE={lf:3d}  {x}')


# --- data + selection -------------------------------------------------------
gsea_all, gsea, real_nes = fc.load_gsea(working_dir)
de_sr_all, de_sr, de_pp = fc.load_de(working_dir)
subclass_colors = fc.load_subclass_colors()

# within-band gene order = glial-niche DE strength (sig cells, then peak nlp);
# cards follow this top-to-bottom order
_glial = list(NN_NICHE)
_nsig = {r['gene']: r['n'] for r in
         de_sr.filter(pl.col('cell_type').is_in(_glial))
         .group_by('gene').agg(pl.col('cell_type').n_unique().alias('n'))
         .iter_rows(named=True)}
_peak = {r['gene']: r['pk'] for r in
         de_sr_all.filter(pl.col('cell_type').is_in(_glial))
         .group_by('gene').agg(pl.col('nlp').max().alias('pk'))
         .iter_rows(named=True)}
GENE_BANDS = [(b, sorted(gs, key=lambda g: (-_nsig.get(g, 0),
                                            -_peak.get(g, 0.0), g)))
              for b, gs in GENE_BANDS]
ordered_genes = [g for _, gs in GENE_BANDS for g in gs]
gene_band = {g: b for b, gs in GENE_BANDS for g in gs}
CARD_GENES = [g for g in ordered_genes if g in set(CARD_GENES)]

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

# chord pool not restricted to dotplot columns: glia<->neuron CCC is the point
chord_cell_set, chord_edges = fc.build_chord(
    f'{working_dir}/output/liana/inflow_diff.csv',
    CANONICAL_LR_PAIRS, CHORD_THEME_LIGANDS, k_neurons=10)

# Panel A trajectory: median NES across platforms per (contrast, pathway, ct);
# arcs track the pregnancy-responsive (sig-in-pregnancy) pairs per band,
# restricted to the non-neuronal niche shown as columns in Panels B/C.
_nes = (pl.read_parquet(f'{working_dir}/output/gsea/perms/real_gsea.parquet')
        .filter(pl.col('pathway').is_in(ordered_pathways))
        .group_by(['contrast', 'pathway', 'cell_type'])
        .agg(pl.col('NES').median().alias('nes')))
_nl = {(r['contrast'], r['pathway'], r['cell_type']): r['nes']
       for r in _nes.iter_rows(named=True)}
_resp = [(r['pathway'], r['cell_type'])
         for r in gsea.filter(pl.col('pathway').is_in(ordered_pathways))
         .iter_rows(named=True) if ' NN' in r['cell_type']]


def band_arc(band):
    """(preg_NES, postpart_NES) rows for a band's responsive pairs."""
    rows = [(_nl.get(('PREG_vs_CTRL', p, c)),
             _nl.get(('POSTPART_vs_CTRL', p, c)))
            for p, c in _resp if pathway_band[p] == band]
    rows = [(a, b) for a, b in rows if a is not None and b is not None]
    return np.array(rows) if rows else np.empty((0, 2))

# --- layout + draw ----------------------------------------------------------
class_spans = fc.spans([ct_class[c] for c in ordered_cts])
band_spans = fc.spans([pathway_band[p] for p in ordered_pathways])
gene_band_spans = fc.spans([gene_band[g] for g in ordered_genes])
norm_nes, norm_lfc, nes_vmax, lfc_vmax = fc.make_norms(
    nes_mat, lfc_mat, d_mat_b)

# split dotplot rows into the boxed theme super-sections (immune / vascular)
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

# Chords sit in the right margin as a single column, beside the lower gene
# cards, with the CCC legend below them: the bottom of the figure is then free
# for the CD31 validation block, which moves up into it.
# titles are two lines drawn above each circle, so the cell must reserve the
# offset plus the text; CHORD_PAD leaves room for the sector labels, which
# overflow the polar axes
CHORD_TITLE_H = 0.58
CHORD_TITLE_OFF = 0.24
CHORD_PAD = 0.18
CHORD_MAX_W = 1.72         # sized so the numeric prefixes do not collide
CHORD_COL_GAP = 0.16       # between stacked chords
CHORD_CLEARANCE = 0.25     # below the cards, before the validation block
CHORD_LEG_GAP = 0.30       # between stacked right-column blocks
RIGHT_COL_W = 1.95         # right margin holding chords + both legends
RIGHT_COL_PAD = 0.12       # gap from the gene cards to the chord column
CARD_LEG_AT = 5            # card legend sits level with card index 5 (Cxcl12)

# CD31 validation row, below the chords: two representative fields plus the
# morphometric battery. The four metrics are one per independent dimension of
# that battery -- density, branching, caliber and vessel-associated cellularity
# -- chosen from the correlation structure of the 18 measures, in which vessel
# area, length density and inter-vessel distance are the same measurement
# (pairwise |r| = 0.96) and the eighteen collapse to six real dimensions.
VASC_METRICS = [
    ('manual_vessel_area_fraction', 'Vessel area\n(% of tissue)'),
    ('manual_junction_density_per_mm2', 'Junction density\n(mm$^{-2}$)'),
    ('manual_mean_vessel_diameter_um', 'Mean vessel\ndiameter (µm)'),
    ('perivascular_nuclei_per_mm_vessel',
     'Perivascular nuclei\n(per mm vessel)'),
]
# one image row per condition (4 nulliparous, 5 pregnant), then the metric
# battery below; all rows span the dotplot-to-cards content width
VASC_MET_H = 1.95          # height of the metric axes
VASC_LABEL_H = 0.68        # rotated condition labels below the metric axes
VASC_ROW_GAP = 0.50        # between the image block and the metric row
VASC_IMG_ROW_GAP = 0.30    # between the two image rows
VASC_TITLE_H = 0.28        # condition title above each image row
VASC_BOT_PAD = 0.02
VASC_GAP = 0.35            # clearance above the validation block
VASC_IMG_GAP = 0.07        # between fields within a row
VASC_MET_GAP = 0.78        # between metric panels
VASC_YLAB = 0.30           # rotated 'CD31 / DAPI' label column

# the image height follows from the content width and is resolved with the
# rest of the geometry, below

# extra dotplot->cards gap to hold the rotated theme-group titles
CARDS_PAD = 0.30
# cards extend down to the bottom of the Panel B cell-type labels
card_extend = (fc.COL_ANNO_GAP_IN + fc.ANNO_W_IN
               + fc.label_drop_in(ordered_cts))
# forest-plot label column sized for full cell-type names (match dotplot)
label_w = fc.text_width_in([f'{c}  ***' for c in ordered_cts], pad_in=0.10)
cm = fc.card_metrics(len(ordered_pathways), len(ordered_genes),
                     len(ordered_cts), len(CARD_GENES), max_sp_n,
                     label_w_in=label_w,
                     n_path_extra=GROUP_EXTRA, n_gene_extra=GROUP_EXTRA,
                     card_extend_in=card_extend, cards_pad_in=CARDS_PAD)
span = fc.chord_full_span(cm, max_sp_n)

# CD31 validation geometry: every row spans the content block, from the
# dotplot left edge to the right edge of the gene cards. Fields are sized by
# the longer of the two condition rows, so both conditions render at one scale.
vasc_roi, vasc_animal, vasc_stats = fc.load_vascular(working_dir)
COND_ORDER = ['Nulliparous', 'Pregnant']
vasc_rows = [(c, sorted(vasc_animal.loc[vasc_animal.condition == c, 'animal']))
             for c in COND_ORDER]
per_row = max(len(a) for _, a in vasc_rows)
vasc_span = (cm.cards_left_in + cm.card_w_max_in) - cm.ax_left_in
vasc_img_w = ((vasc_span - VASC_YLAB - (per_row - 1) * VASC_IMG_GAP)
              / per_row)
n_met = len(VASC_METRICS)
vasc_met_w = ((vasc_span - VASC_YLAB - (n_met - 1) * VASC_MET_GAP) / n_met)
vasc_met_bot = VASC_BOT_PAD + VASC_LABEL_H
# image rows stack upward from the metric row, each carrying a title above it
vasc_row_bot = [vasc_met_bot + VASC_MET_H + VASC_ROW_GAP]
for _ in range(len(vasc_rows) - 1):
    vasc_row_bot.append(vasc_row_bot[-1] + vasc_img_w + VASC_TITLE_H
                        + VASC_IMG_ROW_GAP)
vasc_row_bot = vasc_row_bot[::-1]          # top row first, matching COND_ORDER
vasc_top = vasc_row_bot[0] + vasc_img_w + VASC_TITLE_H

# reserve room below Panel B for the extended cards + column labels, then a
# clearance gap above the validation block
ax_b_bot = vasc_top + VASC_GAP + card_extend + CHORD_CLEARANCE

L = fc.core_layout(len(ordered_pathways), len(ordered_genes),
                   len(ordered_cts), len(CARD_GENES), max_sp_n, ax_b_bot,
                   label_w_in=label_w,
                   n_path_extra=GROUP_EXTRA, n_gene_extra=GROUP_EXTRA,
                   card_extend_in=card_extend, cards_pad_in=CARDS_PAD)

# reserve Panel A (trajectory arcs) as a full-width row above the gene cards.
# Figures 3/5 fit their arcs to the dotplot width, but this figure's dotplot is
# only 11 columns (~2.2 in) -- too narrow for two readable facets -- so the arc
# row spans the whole panel block (dotplot + cards) instead.
ARC_H, ARC_GAP, ARC_FGAP = 1.45, 0.50, 0.45
arc_bot_in = L.cards_top_in + ARC_GAP
L.fig_h += ARC_GAP + ARC_H
arc_fw = (span - (len(THEME_GROUPS) - 1) * ARC_FGAP) / len(THEME_GROUPS)

# Right margin, top-aligned with the gene cards and dotplots: chords stack
# from the top, then the card legend level with the Cxcl12/Slc2a1 cards, then
# the CCC legend. The rotated axis title sits to the right of the circles, so
# the margin carries the column width plus room for that label.
right_col_left = L.cards_left_in + L.card_w_max_in + RIGHT_COL_PAD
_ytitle_w = 0.34
L.fig_w = max(L.fig_w,
              right_col_left + RIGHT_COL_W + _ytitle_w + 0.10)
_cpitch = L.card_total_h_in + fc.CARD_GAP_IN
cpw = CHORD_MAX_W
cph = cpw * 0.85
_cell_h = CHORD_TITLE_H + cph + CHORD_PAD
chord_band_top = L.cards_top_in
chord_specs = []
for _k, _theme in enumerate(CHORD_THEMES):
    _cell_top = chord_band_top - _k * (_cell_h + CHORD_COL_GAP)
    chord_specs.append((_theme, right_col_left + (RIGHT_COL_W - cpw) / 2,
                        _cell_top - CHORD_TITLE_H - cph, cpw, cph))
chord_band_bot = min(s[2] for s in chord_specs) - CHORD_PAD
# card legend level with the Cxcl12 card, CCC legend below it
card_leg_top = min(chord_band_bot - CHORD_LEG_GAP,
                   L.cards_top_in - CARD_LEG_AT * _cpitch)
card_leg_h = 2.12          # the legend's spacing needs >= 2.08 in

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
tleg_bot = fc.draw_dot_legends(fig, L, norm_nes, norm_lfc, nes_vmax, lfc_vmax,
                               [b for b, _ in GENE_BANDS], BAND_COLORS)
fc.draw_forest_legend(fig, L, left_in=right_col_left,
                      top_in=card_leg_top, h_in=card_leg_h,
                      w_in=RIGHT_COL_W)
fc.draw_cards(fig, L, CARD_GENES, cards, sp_coords)

# --- Panel A: peripartum trajectory arcs (spans the dotplot width) -----------
XS = np.array([0.0, 1.0, 2.0])
XLAB = ['Null', 'Preg', 'Post']
_arcs = {b: band_arc(b) for _, bs in THEME_GROUPS for b in bs}
_meds = [np.median(_arcs[b][:, j]) for b in _arcs for j in (0, 1)
         if len(_arcs[b])]
arc_ylim = (min(_meds) - 0.5, max(_meds) + 0.4)
arc_rng = arc_ylim[1] - arc_ylim[0]
arc_ticks = [t for t in (-1, 0, 1, 2) if arc_ylim[0] <= t <= arc_ylim[1]]


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
        ax.set_yticks(arc_ticks)
        ax.set_ylabel('median NES', fontsize=fc.LABEL_FS)
    else:
        ax.set_yticks([])
    fig.text((fx + arc_fw / 2) / L.fig_w,
             (arc_bot_in + ARC_H + 0.14) / L.fig_h, GROUP_LABELS[k],
             ha='center', va='bottom', fontsize=fc.TITLE_FS)

cy = (min(s[2] for s in chord_specs)
      + max(s[2] + s[4] for s in chord_specs)) / 2
fc.draw_chord_row(fig, L, chord_specs, chord_edges, CHORD_THEME_LIGANDS,
                  CHORD_THEME_TITLES, chord_cell_set, subclass_colors,
                  title_off_in=CHORD_TITLE_OFF,
                  ytitle_x_in=right_col_left + RIGHT_COL_W + _ytitle_w / 2,
                  ytitle_cy_in=cy, ytitle_fs=fc.TITLE_FS)
fc.draw_chord_legend(fig, L, card_leg_top - card_leg_h - CHORD_LEG_GAP,
                     chord_cell_set, subclass_colors,
                     left_in=right_col_left, w_in=RIGHT_COL_W, align='left')

# --- CD31 validation block (bottom): representative fields + morphometrics ---
# n = 4 nulliparous and 5 pregnant animals, 28 fields, one representative field
# per animal. The unit of analysis is the animal: fields within an animal are
# near-replicates (vessel-area ICC 0.85), so they are drawn as grey points
# behind the black animal means rather than tested as independent samples.
# Whiskers are 95% confidence intervals, not SEM, because these are null
# results and the interval, not the p-value, is the finding.
vasc_left = L.ax_left_in + VASC_YLAB

for r, ((cond, animals), row_bot) in enumerate(zip(vasc_rows, vasc_row_bot)):
    col = fc.IF_BAR_COLORS[r]
    for i, an in enumerate(animals):
        rgb, _, box = fc.vascular_representative(
            working_dir, vasc_roi, vasc_animal, which=an)
        x_in = vasc_left + i * (vasc_img_w + VASC_IMG_GAP)
        ax = fc.add_axes(fig, L, x_in, row_bot, vasc_img_w, vasc_img_w)
        ax.imshow(rgb, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.9); s.set_color(col)
        ax.text(0.04, 0.96, an, transform=ax.transAxes, ha='left', va='top',
                color='white', fontsize=fc.LABEL_FS)
        if r == 0 and i == 0:
            bar = 100.0 * 1.7583 / box
            ax.plot([0.05, 0.05 + bar], [0.07, 0.07], color='white', lw=2.0,
                    transform=ax.transAxes, solid_capstyle='butt')
            ax.text(0.05, 0.10, '100 µm', color='white', ha='left',
                    va='bottom', fontsize=fc.LABEL_FS - 1.5,
                    transform=ax.transAxes)
    fig.text(vasc_left / L.fig_w, (row_bot + vasc_img_w + 0.05) / L.fig_h,
             cond, ha='left', va='bottom', fontsize=fc.TITLE_FS,
             color='black')
_vasc_mid = (vasc_row_bot[-1] + vasc_row_bot[0] + vasc_img_w) / 2
fig.text((L.ax_left_in + 0.02) / L.fig_w, _vasc_mid / L.fig_h, 'CD31 / DAPI',
         rotation=90, ha='center', va='center', fontsize=fc.TITLE_FS)

for j, (col_name, label) in enumerate(VASC_METRICS):
    ax = fc.add_axes(fig, L, vasc_left + j * (vasc_met_w + VASC_MET_GAP),
                     vasc_met_bot, vasc_met_w, VASC_MET_H)
    srow = vasc_stats.loc[col_name.replace('manual_', '')]
    fc.draw_group_barplot(
        ax, [vasc_animal.loc[vasc_animal.condition == c, col_name].to_numpy()
             for c in COND_ORDER], label,
        roi_values=[vasc_roi.loc[vasc_roi.condition == c, col_name].to_numpy()
                    for c in COND_ORDER],
        err='ci', p=float(srow.p_welch), annot=f'g = {srow.hedges_g:+.2f}')

fc.save(fig, f'{out_dir}/figure_4')
print(f'wrote {out_dir}/figure_4.png and .svg '
      f'({L.fig_w:.1f}x{L.fig_h:.1f}in, {len(ordered_cts)} cells)')

if __name__ == '__main__':
    rank_report()
