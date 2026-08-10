#region imports + setup #######################################################

import os
import re

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy'
de_suffix = ''            # '' for subclass; '_class' for class-level
EMP_P_THRESH = 0.05

seismic_cmap = plt.get_cmap('seismic')
UP_COLOR = seismic_cmap(0.9)
DN_COLOR = seismic_cmap(0.1)

contrasts = ['PREG_vs_CTRL', 'POSTPART_vs_PREG', 'POSTPART_vs_CTRL']
contrast_titles = {
    'PREG_vs_CTRL': 'Pregnant vs Nulliparous',
    'POSTPART_vs_PREG': 'Postpartum vs\nPregnant',
    'POSTPART_vs_CTRL': 'Postpartum vs\nNulliparous',
}
BAR_H = 0.7

# cell-type universe + subclass colours (shared with figure 1)
MIN_CELLS, MIN_PLATFORMS = 100, 2
dataset_names = ['slidetags', 'merfish', 'xenium']

cells_joined = pd.read_csv(
    '/home/karbabi/single-cell/ABC/metadata/cells_joined.csv',
    usecols=['class', 'class_color', 'subclass', 'subclass_color'])
subclass_color = {k.replace('_', '/'): v for k, v in dict(zip(
    cells_joined['subclass'].str.replace('/', '_'),
    cells_joined['subclass_color'])).items()}
subclass_color['Unlabelled'] = '#d3d3d3'
subclass_prefix = dict(zip(
    cells_joined['subclass'],
    cells_joined['class'].str.split(' ', n=1).str[0]))
del cells_joined

# cellular neighbourhoods -> row blocks (class prefixes; figure-1 order)
class_groups = {
    'Pallium Glut':    ['01', '02', '03', '04'],
    'Pallium GABA':    ['06', '07'],
    'Subpallium GABA': ['05', '08', '09', '10'],
    'HY-EA':           ['11', '12', '13', '14', '15', '18', '19', '20', '24'],
    'Non-neuronal':    ['30', '31', '33', '34'],
}

#endregion

#region cell-type universe ####################################################

def sub_num(s):
    p = s.split(' ', 1)[0]
    return int(p) if p.isdigit() else 9999

def sub_code(s):
    """Leading numeric code of a subclass ('318 Astro-NT NN' -> '318')."""
    p = s.split(' ', 1)[0]
    return p if p.isdigit() else ''

def included_subclasses(min_cells=MIN_CELLS, min_platforms=MIN_PLATFORMS):
    """Subclasses with >= min_cells in >= min_platforms platforms (figure 1)."""
    n_pass = {}
    for name in dataset_names:
        adata = sc.read_h5ad(
            f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad',
            backed='r')
        vc = adata.obs['subclass'].astype(str).value_counts()
        adata.file.close()
        for s in vc.index[vc >= min_cells]:
            n_pass[s] = n_pass.get(s, 0) + 1
    return {s for s, k in n_pass.items() if k >= min_platforms}

def neighbourhood_blocks(keep):
    """[(neighbourhood, [cell_types])]; prefix-numeric within block."""
    blocks = []
    for gname, prefixes in class_groups.items():
        cts = sorted((s for s in keep if subclass_prefix.get(s) in prefixes),
                     key=sub_num)
        if cts:
            blocks.append((gname, cts))
    return blocks

#endregion

#region gsea themes ###########################################################

# Curated theme pathways, consolidated into three super-themes: Neuronal,
# Glial (microglia + vascular), Lipid. The bands and their pathway sets mirror
# PATHWAY_BANDS in 13_figure_3.py, 14_figure_4.py and 15_figure_5.py exactly,
# so the curated core of this landscape is the union of what the deep-dive
# figures show. Keep them in sync when a deep-dive band changes.
# (theme, color, [(band, [(pathway, label)])]).
GSEA_THEMES = [
    ('Neuron', '#5B6BBF', [                          # 13_figure_3.py
        ('Synaptic adhesion', [
            ('GOBP_SYNAPSE_ASSEMBLY', 'synapse assembly'),
            ('GOBP_SYNAPSE_ORGANIZATION', 'synapse organization'),
            ('GOBP_HOMOPHILIC_CELL_CELL_ADHESION', 'homophilic cell adhesion'),
        ]),
        ('Excitability', [
            ('GOBP_REGULATION_OF_MEMBRANE_POTENTIAL', 'membrane potential'),
            ('GOBP_REGULATION_OF_POSTSYNAPTIC_MEMBRANE_POTENTIAL',
             'postsynaptic potential'),
            ('GOBP_POTASSIUM_ION_TRANSPORT', 'potassium transport'),
        ]),
        ('GABA & neuropeptide', [
            ('GOBP_SYNAPTIC_TRANSMISSION_GABAERGIC', 'GABAergic transmission'),
            ('GOBP_NEUROPEPTIDE_SIGNALING_PATHWAY', 'neuropeptide signaling'),
            ('GOBP_NEUROTRANSMITTER_SECRETION', 'neurotransmitter secretion'),
        ]),
        ('Glucocorticoid stress', [
            ('GOBP_RESPONSE_TO_CORTICOSTEROID', 'corticosteroid response'),
            ('GOBP_RESPONSE_TO_STEROID_HORMONE', 'steroid hormone response'),
        ]),
        ('Neurotrophic', [
            ('GOBP_RESPONSE_TO_NERVE_GROWTH_FACTOR', 'NGF response'),
            ('GOBP_NEUROTROPHIN_TRK_RECEPTOR_SIGNALING_PATHWAY',
             'Trk receptor signaling'),
            ('GOBP_RESPONSE_TO_GROWTH_FACTOR', 'growth factor response'),
        ]),
        ('Neuronal development', [
            ('GOBP_REGULATION_OF_NEURON_DIFFERENTIATION',
             'neuron differentiation'),
            ('GOBP_NEURON_FATE_COMMITMENT', 'neuron fate commitment'),
            ('GOBP_AXON_DEVELOPMENT', 'axon development'),
        ]),
        ('Activity-dependent plasticity', [
            ('GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY', 'synaptic plasticity'),
            ('GOBP_REGULATION_OF_TRANS_SYNAPTIC_SIGNALING',
             'trans-synaptic signaling'),
            ('GOBP_REGULATION_OF_LONG_TERM_SYNAPTIC_POTENTIATION',
             'long-term potentiation'),
        ]),
    ]),
    ('Lipid', '#E69F00', [                           # 15_figure_5.py
        ('Membrane lipid', [
            ('GOBP_MEMBRANE_LIPID_METABOLIC_PROCESS',
             'membrane lipid metabolism'),
            ('GOBP_GLYCEROLIPID_METABOLIC_PROCESS', 'glycerolipid metabolism'),
            ('GOBP_LIPID_TRANSLOCATION', 'lipid translocation'),
        ]),
        ('Ceramide & sphingolipid', [
            ('GOBP_SPHINGOLIPID_METABOLIC_PROCESS', 'sphingolipid metabolism'),
            ('GOBP_CERAMIDE_METABOLIC_PROCESS', 'ceramide metabolism'),
        ]),
        ('Fatty acid catabolism', [
            ('GOBP_FATTY_ACID_CATABOLIC_PROCESS', 'fatty acid catabolism'),
            ('GOBP_FATTY_ACID_BETA_OXIDATION', 'fatty acid β-oxidation'),
        ]),
        ('Sterol & lipoprotein supply', [
            ('GOBP_REGULATION_OF_LIPID_LOCALIZATION',
             'lipid localization regulation'),
            ('GOBP_NEGATIVE_REGULATION_OF_LIPID_TRANSPORT', 'lipid transport'),
        ]),
        ('Lipoprotein uptake & efflux', [
            ('GOBP_POSITIVE_REGULATION_OF_LIPID_LOCALIZATION', 'lipid uptake'),
            ('GOBP_POSITIVE_REGULATION_OF_CHOLESTEROL_EFFLUX',
             'cholesterol efflux'),
        ]),
    ]),
    ('Microglia', '#117733', [                       # 14_figure_4.py
        ('Innate immune activation', [
            ('GOBP_ADAPTIVE_IMMUNE_RESPONSE', 'adaptive immune response'),
            ('GOBP_ACTIVATION_OF_IMMUNE_RESPONSE',
             'activation of immune response'),
            ('GOBP_INFLAMMATORY_RESPONSE', 'inflammatory response'),
            ('GOBP_ACUTE_INFLAMMATORY_RESPONSE', 'acute inflammatory response'),
        ]),
        ('Cytokine signalling', [
            ('GOBP_CYTOKINE_PRODUCTION', 'cytokine production'),
            ('GOBP_CYTOKINE_MEDIATED_SIGNALING_PATHWAY',
             'cytokine-mediated signaling'),
        ]),
        ('Myeloid effector', [
            ('GOBP_MYELOID_LEUKOCYTE_MIGRATION', 'myeloid leukocyte migration'),
            ('GOBP_LEUKOCYTE_DIFFERENTIATION', 'leukocyte differentiation'),
        ]),
    ]),
    ('Vascular', '#CC3311', [                        # 14_figure_4.py
        ('Angiogenic sprouting', [
            ('GOBP_VASCULATURE_DEVELOPMENT', 'vasculature development'),
            ('GOBP_SPROUTING_ANGIOGENESIS', 'sprouting angiogenesis'),
        ]),
        ('Endothelial dynamics', [
            ('GOBP_ENDOTHELIAL_CELL_MIGRATION', 'endothelial migration'),
            ('GOBP_ENDOTHELIAL_CELL_PROLIFERATION',
             'endothelial proliferation'),
        ]),
        ('Barrier & ECM', [
            ('GOBP_COLLAGEN_BIOSYNTHETIC_PROCESS', 'collagen biosynthesis'),
            ('GOBP_EXTRACELLULAR_MATRIX_ASSEMBLY', 'ECM assembly'),
        ]),
    ]),
]

THEME3 = ['Neuronal', 'Glial', 'Lipid']
THEME3_OF_FIG = {'Neuron': 'Neuronal', 'Microglia': 'Glial',
                 'Vascular': 'Glial', 'Lipid': 'Lipid'}
curated_group = {p: THEME3_OF_FIG[theme]
                 for theme, _, bands in GSEA_THEMES
                 for _, items in bands for p, _ in items}

# extend each super-theme to keyword-matched GO terms (priority resolves
# overlaps; curated terms are forced to their super-theme)
KW3 = [
    ('Lipid', r'LIPID|STEROL|CHOLESTEROL|FATTY_ACID|FATTY|ACYL|SPHINGO|'
     r'CERAMIDE|LIPOPROTEIN|GLYCEROLIPID|PHOSPHOLIPID'),
    ('Glial', r'IMMUN|CYTOKINE|INFLAMMAT|INTERLEUKIN|LEUKOCYTE|MACROPHAGE|'
     r'MYELOID|CHEMOTAXIS|TUMOR_NECROSIS|PHAGOCYT|COMPLEMENT|MICROGLI|'
     r'INTERFERON|TOLL_LIKE|VASCUL|ANGIO|ENDOTHELI|VEGF|COLLAGEN|'
     r'TIGHT_JUNCTION|BLOOD_VESSEL|EXTRACELLULAR_MATRIX|BASEMENT_MEMBRANE|'
     r'GLIAL|GLIOGENESIS|ASTROCYTE|OLIGODENDROCYTE|MYELIN'),
    ('Neuronal', r'SYNAP|NEURON|AXON|DENDRIT|NEUROTRANSMITTER|GABA|GLUTAMATE|'
     r'NEUROPEPTIDE|MEMBRANE_POTENTIAL|ION_TRANSPORT|POTASSIUM|SODIUM_ION|'
     r'CALCIUM_ION|ACTION_POTENTIAL|NERVE|CORTICOSTEROID'),
]
def assign_theme3(p):
    if p in curated_group:
        return curated_group[p]
    for t, pat in KW3:
        if re.search(pat, p):
            return t
    return None

# Call-outs are derived from GSEA_THEMES rather than written out, so the bands
# named on the figure cannot drift from the bands actually curated above (and
# therefore from figures 3-5).
BANDS_OF_THEME3 = {t: [] for t in THEME3}
for _theme, _, _bands in GSEA_THEMES:
    for _band, _ in _bands:
        BANDS_OF_THEME3[THEME3_OF_FIG[_theme]].append(_band)

THEME3_CALLOUT = {
    t: r'$\mathbf{%s}$' % t + '\n'
       + '\n'.join(f'• {b}' for b in BANDS_OF_THEME3[t])
    for t in THEME3}

GSEA_CAP = 4.0           # cap on |signed -log10 emp_p|
GSEA_CONTRAST = 2.5      # color saturation range (smaller = punchier)
THEME_CAP = 50           # max GO terms per super-theme band
BG_ALPHA = 0.22          # faded background (non-theme GO terms)

#endregion

#region figure 2: panel A barplot + panel B GSEA landscape ####################

# layout geometry (inches)
ROW_H, BLOCK_GAP, BOX_LW = 0.10, 0.12, 0.6
M_L, M_T, M_B = 0.10, 0.20, 0.85
NEI_W, CELL_LAB_W, CBAR_W = 0.20, 1.50, 0.09
TICK_LEN = 2.0            # row tick marks left of each subclass strip, points
LAB_PAD = 0.055           # gap between panel A's cell-type label and its strip
CODE_PAD = 0.055          # gap between panel B's subclass code and its strip
AX_GAP = 0.05
PANEL_GAP, PANEL_W, TOP_H, RIGHT_W = 0.50, 4.4, 0.40, 0.25
U = (PANEL_W - len(contrasts) * AX_GAP) / 4          # contrast unit (PREG = 2U)
HEATW = PANEL_W                                       # both panels share width


def row_ticks(ax, n):
    """One tick per row, protruding left of a subclass colour strip. The strip
    has no spines, but tick marks draw independently of them."""
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([])
    ax.tick_params(axis='y', left=True, right=False, direction='out',
                   length=TICK_LEN, width=0.5, color='black', pad=1)


def build_figure():
    # shared row axis: 84 cells in neighbourhood order (= Panel A / figure 1)
    blocks = neighbourhood_blocks(included_subclasses())
    cells = [c for _, cts in blocks for c in cts]
    ci = {c: i for i, c in enumerate(cells)}

    # Panel A: meta-DEG up/down counts (D>=2)
    sr = pl.read_csv(f'{working_dir}/output/de/sumrank_results{de_suffix}.csv')
    deg = sr.group_by(['cell_type', 'contrast']).agg(
        ((pl.col('emp_p_up') < EMP_P_THRESH) & (pl.col('D') >= 2)).sum()
        .alias('up'),
        ((pl.col('emp_p_down') < EMP_P_THRESH) & (pl.col('D') >= 2)).sum()
        .alias('dn'))
    deg_d = {(r['cell_type'], r['contrast']): r
             for r in deg.iter_rows(named=True)}

    # Panel B: GSEA signed -log10(emp_p), direction from nlp_up/down
    g = (pl.read_csv(f'{working_dir}/output/gsea/sumrank_gsea_results.csv')
         .filter((pl.col('contrast') == 'PREG_vs_CTRL') & (pl.col('D') >= 2))
         .with_columns(
            pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
              .then(pl.col('emp_p_up')).otherwise(pl.col('emp_p_down'))
              .alias('emp_p'),
            pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
              .then(1).otherwise(-1).alias('sgn')))
    sigp = (g.filter(pl.col('emp_p') <= 0.05).group_by('pathway')
            .agg(pl.len().alias('n')).filter(pl.col('n') >= 1))
    paths = sorted(set(sigp['pathway'].to_list()) | set(curated_group))
    mep = g.group_by('pathway').agg(pl.col('emp_p').min().alias('mep'))
    peak = {r['pathway']: -np.log10(max(r['mep'], 1e-12))
            for r in mep.iter_rows(named=True)}

    # super-theme membership (curated forced), capped per theme by significance
    group_of, core3 = {}, {t: [] for t in THEME3}
    assigned = {p: assign_theme3(p) for p in paths}
    for t in THEME3:
        mem = sorted([p for p in paths if assigned[p] == t],
                     key=lambda p: (p not in curated_group, -peak.get(p, 0)))
        for p in mem[:THEME_CAP]:
            group_of[p] = t
            core3[t].append(p)

    pj = {p: j for j, p in enumerate(paths)}
    paths_set = set(paths)
    M = np.full((len(cells), len(paths)), np.nan)    # signed -log10 emp_p
    S = np.zeros((len(cells), len(paths)), bool)      # significant mask
    for r in g.filter(pl.col('pathway').is_in(paths_set)).iter_rows(named=True):
        if r['cell_type'] not in ci:
            continue
        i, j = ci[r['cell_type']], pj[r['pathway']]
        M[i, j] = np.clip(r['sgn'] * -np.log10(max(r['emp_p'], 1e-12)),
                          -GSEA_CAP, GSEA_CAP)
        S[i, j] = r['emp_p'] <= 0.05

    # cluster columns on tested rows, soft must-link within each super-theme
    tested = set(g['cell_type'].unique().to_list())
    Mt = np.nan_to_num(M[[ci[c] for c in cells if c in tested]])
    Dc = squareform(pdist(Mt.T, 'correlation'))
    fin = Dc[np.isfinite(Dc)]
    Dc = np.where(np.isfinite(Dc), Dc, fin.max())
    for t in THEME3:
        idx = [pj[p] for p in core3[t]]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                Dc[idx[a], idx[b]] = Dc[idx[b], idx[a]] = \
                    Dc[idx[a], idx[b]] * 0.02
    Zc = linkage(squareform(Dc, checks=False), 'average')
    co = dendrogram(Zc, no_plot=True)['leaves']
    paths_o = [paths[j] for j in co]
    is_core = np.array([p in group_of for p in paths_o])
    Mo, So = M[:, co], S[:, co]
    ncol = len(paths_o)

    # figure geometry
    contrast_w = [2 * U if c == 'PREG_vs_CTRL' else U for c in contrasts]
    x_cbar = M_L + NEI_W + CELL_LAB_W + 0.04
    x_pa = x_cbar + CBAR_W + AX_GAP
    x_heat = x_pa + sum(contrast_w) + len(contrasts) * AX_GAP + PANEL_GAP
    W = x_heat + HEATW + RIGHT_W
    rows_top, y, spans = M_T + TOP_H, M_T + TOP_H, []
    for name, cts in blocks:
        h = len(cts) * ROW_H
        spans.append((name, cts, y, h))
        y += h + BLOCK_GAP
    rows_bot = y - BLOCK_GAP
    H = rows_bot + M_B
    y_title = rows_top - 0.26          # single-/two-line titles share baseline

    fig = plt.figure(figsize=(W, H))

    def rect(x, yt, w, h):
        return [x / W, 1 - (yt + h) / H, w / W, h / H]

    cmap = plt.get_cmap('seismic').copy()
    cmap.set_bad((1, 1, 1, 0))

    for bk, (name, cts, by, bh) in enumerate(spans):
        n = len(cts)
        ridx = [ci[c] for c in cts]

        # subclass colour bar + cell-type + neighbourhood labels
        cax = fig.add_axes(rect(x_cbar, by, CBAR_W, bh))
        cax.set_xlim(0, 1); cax.set_ylim(n - 0.5, -0.5)
        cax.set_xticks([])
        for s in cax.spines.values():
            s.set_visible(False)
        row_ticks(cax, n)
        for k, c in enumerate(cts):
            cax.add_patch(Rectangle((0, k - 0.5), 1, 1,
                          facecolor=subclass_color.get(c, '#333'), lw=0))
            fig.text((x_cbar - LAB_PAD) / W,
                     1 - (by + (k + 0.5) * ROW_H) / H, c,
                     ha='right', va='center', fontsize=5.5)
        fig.text((M_L + NEI_W / 2) / W, 1 - (by + bh / 2) / H, name,
                 rotation=90, ha='center', va='center', fontsize=8)

        # Panel A: diverging barh per contrast
        x = x_pa
        for j, contrast in enumerate(contrasts):
            ax = fig.add_axes(rect(x, by, contrast_w[j], bh))
            au, ad = [], []
            for k, c in enumerate(cts):
                r = deg_d.get((c, contrast), {})
                u, d = r.get('up', 0), r.get('dn', 0)
                ax.barh(k, u, height=BAR_H, facecolor=UP_COLOR,
                        edgecolor=UP_COLOR, lw=0.4, zorder=5)
                ax.barh(k, -d, height=BAR_H, facecolor=DN_COLOR,
                        edgecolor=DN_COLOR, lw=0.4, zorder=5)
                au.append(u); ad.append(d)
            xl = max(max(au + [1]), max(ad + [1])) * 1.25
            ax.axvline(0, color='grey', lw=0.5, zorder=0)
            ax.grid(True, 'major', 'both', ls='-', lw=0.3, c='lightgray',
                    zorder=0)
            ax.set_xlim(-xl, xl); ax.set_ylim(n - 0.5, -0.5)
            ax.set_yticks([]); ax.tick_params(length=0, labelsize=6)
            for s in ax.spines.values():
                s.set_linewidth(BOX_LW)
            if bk == 0:
                fig.text((x + contrast_w[j] / 2) / W, 1 - y_title / H,
                         contrast_titles[contrast], ha='center', va='center',
                         fontsize=8, linespacing=1.0)
            if bk != len(spans) - 1:
                ax.set_xticklabels([])
            x += contrast_w[j] + AX_GAP

        # Panel B: subclass code + strip + GSEA landscape (faded bg + bands)
        x_sc2 = x_heat - 0.04 - CBAR_W
        sc2 = fig.add_axes(rect(x_sc2, by, CBAR_W, bh))
        sc2.set_xlim(0, 1); sc2.set_ylim(n - 0.5, -0.5)
        sc2.set_xticks([])
        for s in sc2.spines.values():
            s.set_visible(False)
        row_ticks(sc2, n)
        for k, c in enumerate(cts):
            sc2.add_patch(Rectangle((0, k - 0.5), 1, 1,
                          facecolor=subclass_color.get(c, '#333'), lw=0))
            # numeric subclass code, so panel B rows can be read without
            # tracking back across the panel gap to the panel A labels
            fig.text((x_sc2 - CODE_PAD) / W,
                     1 - (by + (k + 0.5) * ROW_H) / H, sub_code(c),
                     ha='right', va='center', fontsize=5.5)

        hb = fig.add_axes(rect(x_heat, by, HEATW, bh))
        bgd = np.where(So[ridx], Mo[ridx], np.nan)
        cored = np.where(So[ridx] & is_core[None, :], Mo[ridx], np.nan)
        hb.imshow(np.ma.masked_invalid(bgd), cmap=cmap,
                  norm=Normalize(-GSEA_CONTRAST, GSEA_CONTRAST), aspect='auto',
                  extent=[0, ncol, n, 0], interpolation='nearest',
                  alpha=BG_ALPHA, zorder=2)
        hb.imshow(np.ma.masked_invalid(cored), cmap=cmap,
                  norm=Normalize(-GSEA_CONTRAST, GSEA_CONTRAST), aspect='auto',
                  extent=[0, ncol, n, 0], interpolation='nearest', zorder=3)
        hb.set_xlim(0, ncol); hb.set_ylim(n, 0)
        hb.set_xticks([]); hb.set_yticks([])
        for s in hb.spines.values():
            s.set_linewidth(BOX_LW)

    fig.text((x_heat + HEATW / 2) / W, 1 - y_title / H,
             'Pregnant vs Nulliparous', ha='center', va='center', fontsize=8)

    # overlay: black dashed theme boxes + call-outs
    ov = fig.add_axes(rect(x_heat, rows_top, HEATW, rows_bot - rows_top))
    ov.set_xlim(0, ncol); ov.set_ylim(1, 0)            # y: 0 top, 1 bottom
    ov.set_xticks([]); ov.set_yticks([]); ov.patch.set_alpha(0)
    for s in ov.spines.values():
        s.set_visible(False)

    def _yfrac(a):
        return (a - rows_top) / (rows_bot - rows_top)

    def _block(name):
        for nm, ct, by, bh in spans:
            if nm == name:
                return _yfrac(by + bh / 2), (_yfrac(by), _yfrac(by + bh))
        return 0.5, (0.0, 1.0)

    band = {}
    for t in THEME3:
        pos = [j for j, p in enumerate(paths_o) if group_of.get(p) == t]
        band[t] = (min(pos), max(pos))
    order = sorted(THEME3, key=lambda t: band[t][0])
    g01 = (band[order[0]][1] + band[order[1]][0]) / 2
    g12 = (band[order[1]][1] + band[order[2]][0]) / 2

    # Glial box spans only the glia rows; each callout has a horizontal leader
    theme_yspan = {'Neuronal': (0.0, 1.0), 'Lipid': (0.0, 1.0),
                   'Glial': _block('Non-neuronal')[1]}
    mid_sub = _block('Subpallium GABA')[0]
    mid_hy = _block('HY-EA')[0]
    mid_nn = _block('Non-neuronal')[0]
    callout = {
        'Neuronal': (g01, mid_sub, band['Neuronal'][1] + 1),
        'Lipid':    (g01, mid_hy,  band['Lipid'][0] - 1),
        'Glial':    (g12, mid_nn,  band['Glial'][0] - 1),
    }
    for t in THEME3:
        c0, c1 = band[t]
        y0, y1 = theme_yspan[t]
        ov.add_patch(Rectangle((c0 - 1, y0), c1 - c0 + 2, y1 - y0,
                     facecolor='none', edgecolor='black', lw=1.3,
                     linestyle=(0, (5, 3)), zorder=7, clip_on=False))
        tx, ty, edge = callout[t]
        ov.annotate(THEME3_CALLOUT[t], xy=(edge, ty), xytext=(tx, ty),
                    textcoords='data', ha='center', va='center', fontsize=5.7,
                    color='black', zorder=8, linespacing=1.2,
                    bbox=dict(boxstyle='square,pad=0.3', fc='white',
                              ec='black', lw=1.3),
                    arrowprops=dict(arrowstyle='-', lw=0.8, color='black',
                                    shrinkA=2, shrinkB=0))

    # single x-axis title per panel + legends (aligned at leg_y)
    pa_cx = x_pa + (sum(contrast_w) + (len(contrasts) - 1) * AX_GAP) / 2
    fig.text(pa_cx / W, 1 - (rows_bot + 0.18) / H,
             f'meta DEGs (emp_p<{EMP_P_THRESH}, D>=2)',
             ha='center', va='top', fontsize=8)
    fig.text((x_heat + HEATW / 2) / W, 1 - (rows_bot + 0.18) / H, 'GO terms',
             ha='center', va='top', fontsize=8)
    leg_y = rows_bot + 0.45
    fig.legend(handles=[Patch(facecolor=UP_COLOR, label='Upregulated'),
                        Patch(facecolor=DN_COLOR, label='Downregulated')],
               loc='upper left', bbox_to_anchor=(x_pa / W, 1 - leg_y / H),
               fontsize=7, frameon=False, ncol=1)
    caxh = fig.add_axes(rect(x_heat + HEATW - 1.3, leg_y, 1.3, 0.07))
    cb = fig.colorbar(plt.cm.ScalarMappable(
        norm=Normalize(-GSEA_CONTRAST, GSEA_CONTRAST), cmap='seismic'),
        cax=caxh, orientation='horizontal')
    cb.set_ticks([-GSEA_CONTRAST, 0, GSEA_CONTRAST])
    cb.set_ticklabels(['↓ down', 'NS', '↑ up'])
    cb.ax.tick_params(labelsize=6.5, length=2, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label('GSEA  signed −log10 emp p', fontsize=7, labelpad=2)

    os.makedirs(f'{working_dir}/figures', exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(f'{working_dir}/figures/figure_2.{ext}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'figure_2 saved: {W:.1f}x{H:.1f}in, {len(cells)} cells')

#endregion

#region run ###################################################################

if __name__ == '__main__':
    build_figure()

#endregion
