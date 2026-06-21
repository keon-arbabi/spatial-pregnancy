"""Figure 4: unified microglia + vascular glial-niche figure."""
import os
import importlib

import polars as pl
import matplotlib.pyplot as plt

fc = importlib.import_module('12_figure_helper')

fc.setup_style()
working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures'
os.makedirs(out_dir, exist_ok=True)

# --- config -----------------------------------------------------------------
PATHWAY_BANDS = [
    ('Immune activation', [
        'GOBP_ADAPTIVE_IMMUNE_RESPONSE',
        'GOBP_ACTIVATION_OF_IMMUNE_RESPONSE',
    ]),
    ('Cytokine production', [
        'GOBP_CYTOKINE_PRODUCTION',
        'GOBP_CYTOKINE_MEDIATED_SIGNALING_PATHWAY',
    ]),
    ('Inflammatory response', [
        'GOBP_INFLAMMATORY_RESPONSE',
        'GOBP_ACUTE_INFLAMMATORY_RESPONSE',
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

BAND_COLORS = {
    'Immune activation':     '#0072B2',
    'Cytokine production':   '#E69F00',
    'Inflammatory response': '#009E73',
    'Myeloid effector':      '#CC79A7',
    'Angiogenic sprouting':  '#56B4E9',
    'Endothelial dynamics':  '#332288',
    'Barrier & ECM':         '#661100',
}

GENE_BANDS = [
    ('Immune activation', ['Cd81', 'Cd38', 'Isg15', 'Cd27', 'Lacc1']),
    ('Cytokine production', ['Lifr', 'Il6st', 'Il4ra', 'Il10ra', 'Stat3']),
    ('Inflammatory response', ['Nlrp3', 'Txnip', 'Grn', 'Ifngr1', 'Lyn']),
    ('Myeloid effector', ['Mertk', 'Csf3r', 'Pik3cd', 'Trpm2', 'Adgre1']),
    ('Angiogenic sprouting',
     ['Vegfa', 'Flt1', 'Notch1', 'Eng', 'Id1', 'Bmpr2']),
    ('Endothelial dynamics', ['Rgcc', 'Kdr', 'Cxcl12', 'Pecam1', 'Cdh5']),
    ('Barrier & ECM', ['Slc2a1', 'Mfsd2a', 'Dag1', 'Col1a1', 'Tnxb']),
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


CARD_GENES = ['Cd81', 'Lifr', 'Lyn', 'Mertk',
              'Vegfa', 'Flt1', 'Cxcl12', 'Slc2a1']
_mg = ['334 Microglia NN']
card_ctx = {'Cd81': ['318 Astro-NT NN', '334 Microglia NN'], 'Lifr': _mg,
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
    'Immune activation':
        r'IMMUNE_RESPONSE|IMMUNE_SYSTEM|IMMUNE_EFFECTOR|ANTIGEN',
    'Cytokine production': r'CYTOKINE|INTERLEUKIN|TUMOR_NECROSIS|INTERFERON',
    'Inflammatory response': r'INFLAMMAT',
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

# --- layout + draw ----------------------------------------------------------
class_spans = fc.spans([ct_class[c] for c in ordered_cts])
band_spans = fc.spans([pathway_band[p] for p in ordered_pathways])
gene_band_spans = fc.spans([gene_band[g] for g in ordered_genes])
norm_nes, norm_lfc, nes_vmax, lfc_vmax = fc.make_norms(
    nes_mat, lfc_mat, d_mat_b)

# split dotplot rows into boxed immune (first 4 bands) / vascular super-sections
IMMUNE_BANDS = 4
ROW_GAP = 0.9
_p_split = sum(len(ps) for _, ps in PATHWAY_BANDS[:IMMUNE_BANDS])
_g_split = sum(len(gs) for _, gs in GENE_BANDS[:IMMUNE_BANDS])


def _row_y(n, split):
    return [i if i < split else i + ROW_GAP for i in range(n)]


row_y_a = _row_y(len(ordered_pathways), _p_split)
row_y_b = _row_y(len(ordered_genes), _g_split)
row_sections_a = [(0, _p_split - 1), (_p_split, len(ordered_pathways) - 1)]
row_sections_b = [(0, _g_split - 1), (_g_split, len(ordered_genes) - 1)]

CHORD_NCOLS = 3
CHORD_ROW_GAP, CHORD_TITLE_H = 0.20, fc.CHORD_TITLE_H_IN
CHORD_PANEL_GAP = fc.CHORD_PANEL_GAP_IN
CHORD_BOT_PAD, CHORD_CLEARANCE = 0.02, 0.55

# cards extend down to the bottom of the Panel B cell-type labels
card_extend = (fc.COL_ANNO_GAP_IN + fc.ANNO_W_IN
               + fc.label_drop_in(ordered_cts))
# forest-plot label column sized for full cell-type names (match dotplot)
label_w = fc.text_width_in([f'{c}  ***' for c in ordered_cts], pad_in=0.10)
cm = fc.card_metrics(len(ordered_pathways), len(ordered_genes),
                     len(ordered_cts), len(CARD_GENES), max_sp_n,
                     label_w_in=label_w,
                     n_path_extra=ROW_GAP, n_gene_extra=ROW_GAP,
                     card_extend_in=card_extend)
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
                   label_w_in=label_w,
                   n_path_extra=ROW_GAP, n_gene_extra=ROW_GAP,
                   card_extend_in=card_extend)
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
tleg_bot = fc.draw_dot_legends(fig, L, norm_nes, norm_lfc, nes_vmax, lfc_vmax,
                               [b for b, _ in GENE_BANDS], BAND_COLORS)
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

fc.save(fig, f'{out_dir}/figure_4')
print(f'wrote {out_dir}/figure_4.png and .svg '
      f'({L.fig_w:.1f}x{L.fig_h:.1f}in, {len(ordered_cts)} cells)')

if __name__ == '__main__':
    rank_report()
