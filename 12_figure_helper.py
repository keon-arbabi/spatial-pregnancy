"""Shared building blocks for the spatial-pregnancy theme figures
(lipid / microglia / vascular / neuron).

Each figure supplies its own config -- pathways, genes, mechanistic bands,
cell-selection rule, gene cards, chord themes -- and orchestrates these
helpers. This module owns the shared data loading, selection, matrices,
drawing (Panel A/B dotplots, gene cards, all legends, annotation strips,
LIANA chord) and the core geometry. Figure-specific extras (e.g. the
vascular imaging block) live in the figure script using the layout anchors
returned by ``core_layout``.
"""
import os
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc

import matplotlib as mpl
import matplotlib.pyplot as plt
from pycirclize import Circos

# --- shared style + palettes -------------------------------------------------

def setup_style():
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['figure.dpi'] = 400


CMAP = plt.get_cmap('seismic')
SUBCLASS_CSV = '/home/karbabi/single-cell/ABC/metadata/cells_joined.csv'

PLATFORMS = ['slidetags', 'merfish', 'xenium']
PLATFORM_COLORS = {'slidetags': '#1C6CC6', 'merfish': '#E8A628',
                   'xenium': '#2F7F2E'}
PLATFORM_LABELS = {'xenium': 'Xenium', 'merfish': 'MERFISH',
                   'slidetags': 'Slide-tags'}
D_COLORS = {2: '#888888', 3: '#000000'}

CHORD_COLOR_UP = '#b2182b'
CHORD_COLOR_DOWN = '#2166ac'
CHORD_CLASS_COLORS = {'NN': '#7570b3', 'Glut': '#d95f02', 'GABA': '#1b9e77'}
CHORD_CLASS_ORDER = ['NN', 'Glut', 'GABA']
CHORD_CLASS_LABELS = {'NN': 'Non-neuronal', 'Glut': 'Glutamatergic',
                      'GABA': 'GABAergic'}

# dot-size + colour-norm constants (shared across all dotplots)
NLP_MIN, NLP_MAX = 1.30, 5.0
SIZE_MIN, SIZE_MAX = 16.0, 80.0
SIZE_MAX_A = 140.0
SIG_DOT_SIZE = 4.0
TITLE_FS = 11.0  # axis / neighbourhood / card-gene / chord titles
LABEL_FS = 9.0   # dotplot ticks + forest-plot labels

# --- small helpers -----------------------------------------------------------

def pretty_ceil(v):
    if v == 0: return 1.0
    if v >= 1: return float(np.ceil(v * 10) / 10)
    if v >= 0.01: return float(np.ceil(v * 100) / 100)
    return float(np.ceil(v * 1000) / 1000)


def quant_vmax(arr, q_lo=0.05, q_hi=0.95):
    flat = np.asarray(arr, dtype=float)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0: return 1.0
    q1, q2 = np.quantile(flat, [q_lo, q_hi])
    return pretty_ceil(max(abs(q1), abs(q2)))


def numeric_prefix(ct):
    m = re.match(r'^(\d+)', ct)
    return int(m.group(1)) if m else 9999


def spans(labels):
    out, prev, start = [], labels[0], 0
    for k, lab in enumerate(labels[1:], 1):
        if lab != prev:
            out.append((prev, start, k - 1))
            prev, start = lab, k
    out.append((prev, start, len(labels) - 1))
    return out


def column_positions(class_spans, n_cols, gap):
    """Per-column x positions with ``gap`` (column units) inserted between the
    contiguous sections in ``class_spans``; pass the result as ``L.col_x``."""
    col_x = list(range(n_cols))
    for si, (_, lo, hi) in enumerate(class_spans):
        for j in range(lo, hi + 1):
            col_x[j] = j + si * gap
    return col_x


def stars_for(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return ''
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    return ''


def pctl_range(arr, lo=5, hi=95):
    if len(arr) == 0: return 0.0, 1.0
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def nlp_to_size(x):
    x = np.clip(x, NLP_MIN, NLP_MAX)
    return SIZE_MIN + (x - NLP_MIN) / (NLP_MAX - NLP_MIN) * (SIZE_MAX_A
                                                             - SIZE_MIN)


def pct_to_size(p):
    p = float(np.clip(p, 0, 100))
    return SIZE_MIN + (p / 100.0) * (SIZE_MAX - SIZE_MIN)


def make_norms(nes_mat, lfc_mat, d_mat_b):
    nes_vmax = quant_vmax(nes_mat)
    lfc_vmax = quant_vmax(lfc_mat[d_mat_b >= 2])
    return (mpl.colors.Normalize(-nes_vmax, nes_vmax),
            mpl.colors.Normalize(-lfc_vmax, lfc_vmax), nes_vmax, lfc_vmax)

# --- data loading ------------------------------------------------------------

def load_gsea(working_dir, contrast='PREG_vs_CTRL'):
    gsea_all = (pl.read_csv(
        f'{working_dir}/output/gsea/sumrank_gsea_results.csv')
        .filter((pl.col('contrast') == contrast) & (pl.col('D') >= 2))
        .with_columns(
            pl.max_horizontal('nlp_up', 'nlp_down').alias('nlp'),
            pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
              .then(pl.col('emp_p_up')).otherwise(pl.col('emp_p_down'))
              .alias('emp_p')))
    gsea = gsea_all.filter(pl.col('emp_p') <= 0.05)
    real_nes = (pl.read_parquet(
        f'{working_dir}/output/gsea/perms/real_gsea.parquet')
        .filter(pl.col('contrast') == contrast))
    return gsea_all, gsea, real_nes


def load_de(working_dir, contrast='PREG_vs_CTRL'):
    de_sr_all = (pl.read_csv(f'{working_dir}/output/de/sumrank_results.csv')
        .filter((pl.col('contrast') == contrast) & (pl.col('D') >= 2)
                & (pl.col('ref_pct_detected') >= 5.0))
        .with_columns(
            pl.max_horizontal('nlp_up', 'nlp_down').alias('nlp'),
            pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
              .then(pl.col('emp_p_up')).otherwise(pl.col('emp_p_down'))
              .alias('emp_p')))
    de_sr = de_sr_all.filter(pl.col('emp_p') <= 0.05)
    de_pp = (pl.read_csv(f'{working_dir}/output/de/de_results.csv')
        .filter(pl.col('contrast') == contrast)
        .select(['gene', 'cell_type', 'dataset', 'logFC', 'SE', 'LCI',
                 'UCI', 'AveExpr', 'PValue', 'FDR', 'ref_pct_detected',
                 'n_cells_treat', 'n_cells_base']))
    return de_sr_all, de_sr, de_pp


def load_subclass_colors(path=SUBCLASS_CSV):
    cj = pd.read_csv(path, usecols=['subclass', 'subclass_color']) \
        .drop_duplicates()
    colors = {k.replace('_', '/'): v for k, v in
              dict(zip(cj['subclass'].str.replace('/', '_'),
                       cj['subclass_color'])).items()}
    colors['Unlabelled'] = '#d3d3d3'
    return colors


def _totals(a, ds, working_dir):
    cdir = f'{working_dir}/output/cache'
    os.makedirs(cdir, exist_ok=True)
    fp = f'{cdir}/totals_{ds}_{a.n_obs}.npy'
    if os.path.exists(fp):
        return np.load(fp)
    totals = np.zeros(a.n_obs, dtype=np.float64)
    for start in range(0, a.n_obs, 20000):
        end = min(start + 20000, a.n_obs)
        block = a.X[start:end]
        block = block.toarray() if hasattr(block, 'toarray') \
            else np.asarray(block)
        totals[start:end] = block.sum(axis=1)
    totals = np.where(totals > 0, totals, 1.0)
    np.save(fp, totals)
    return totals


def load_spatial(working_dir, card_genes, ordered_genes, ordered_cts,
                 platforms=PLATFORMS):
    sp_coords, sp_expr, sp_in = {}, {}, {}
    pct = {(g, ct): [] for g in ordered_genes for ct in ordered_cts}
    for ds in platforms:
        a = sc.read_h5ad(
            f'{working_dir}/output/{ds}/03_adata_query_{ds}.h5ad',
            backed='r')
        var = (a.var['gene_symbol'].astype(str).to_list()
               if 'gene_symbol' in a.var.columns else list(a.var_names))
        keep = (a.obs['sample'].values != 'CTRL_3') if ds == 'xenium' \
            else np.ones(a.n_obs, dtype=bool)
        obs = a.obs.loc[keep, ['x_ffd', 'y_ffd', 'condition', 'subclass']]
        xv = obs['x_ffd'].values.astype(float)
        yv = obs['y_ffd'].values.astype(float)
        cts = obs['subclass'].values.astype(str)
        sp_coords[ds] = dict(
            x=xv, y=yv, cond=obs['condition'].values.astype(str),
            cell_type=cts, midline=(xv.min() + xv.max()) / 2,
            fov_cx=(xv.min() + xv.max()) / 2,
            fov_cy=(yv.min() + yv.max()) / 2,
            fov_half=max(np.ptp(xv), np.ptp(yv)) / 2 * 1.05)
        idx = {n: i for i, n in enumerate(var)}
        totals = _totals(a, ds, working_dir)
        sp_expr[ds], sp_in[ds] = {}, {}
        for g in card_genes:
            if g in idx:
                col = a.X[:, idx[g]]
                arr = (col.toarray().flatten() if hasattr(col, 'toarray')
                       else np.asarray(col).flatten())
                sp_expr[ds][g] = np.log2(arr / totals * 1e4 + 1)[keep]
                sp_in[ds][g] = True
            else:
                sp_in[ds][g] = False
        ct_masks = {ct: (cts == ct) for ct in ordered_cts}
        for g in ordered_genes:
            if g not in idx:
                continue
            col = a.X[:, idx[g]]
            arr = (col.toarray().flatten() if hasattr(col, 'toarray')
                   else np.asarray(col).flatten())[keep]
            nz = arr > 0
            for ct, m in ct_masks.items():
                nt = int(m.sum())
                if nt:
                    pct[(g, ct)].append(float(nz[m].sum()) / nt * 100.0)
        a.file.close()
        del a
    return sp_coords, sp_expr, sp_in, pct

# --- cell-type selection -----------------------------------------------------

def select_cell_types(gsea, ordered_pathways, allowlist, assign_class,
                      class_order, min_hits, candidate_expr=None):
    cand = gsea.filter(pl.col('pathway').is_in(ordered_pathways))
    if candidate_expr is not None:
        cand = cand.filter(candidate_expr)
    counts = (cand.group_by('cell_type').agg(pl.len().alias('n'))
              .filter(pl.col('n') >= min_hits))
    keep = set(counts['cell_type'].to_list()) | set(allowlist)
    rank = {c: i for i, c in enumerate(class_order)}
    ordered = sorted(keep, key=lambda c: (rank[assign_class(c)],
                                          numeric_prefix(c)))
    ct_class = {c: assign_class(c) for c in ordered}
    return ordered, ct_class

# --- matrices ----------------------------------------------------------------

def gsea_matrices(gsea_all, real_nes, ordered_pathways, ordered_cts):
    keep = set(ordered_cts)
    nr, nc = len(ordered_pathways), len(ordered_cts)
    nlp = np.full((nr, nc), np.nan)
    nes = np.full((nr, nc), np.nan)
    d = np.zeros((nr, nc), dtype=int)
    sig = np.zeros((nr, nc), dtype=bool)
    ri = {p: i for i, p in enumerate(ordered_pathways)}
    ci = {c: j for j, c in enumerate(ordered_cts)}
    sub = gsea_all.filter(pl.col('pathway').is_in(ordered_pathways)
                          & pl.col('cell_type').is_in(list(keep)))
    for r in sub.iter_rows(named=True):
        i, j = ri[r['pathway']], ci[r['cell_type']]
        if np.isnan(nlp[i, j]) or r['nlp'] > nlp[i, j]:
            nlp[i, j] = r['nlp']
            d[i, j] = r['D']
            sig[i, j] = r['emp_p'] <= 0.05
    agg = (real_nes.filter(pl.col('pathway').is_in(ordered_pathways)
                           & pl.col('cell_type').is_in(list(keep)))
           .group_by(['pathway', 'cell_type'])
           .agg(pl.col('NES').median().alias('m')))
    look = {(r['pathway'], r['cell_type']): r['m']
            for r in agg.iter_rows(named=True)}
    for i, p in enumerate(ordered_pathways):
        for j, ct in enumerate(ordered_cts):
            if not np.isnan(nlp[i, j]):
                v = look.get((p, ct))
                if v is not None:
                    nes[i, j] = v
    return nlp, nes, d, sig


def de_matrices(de_sr, de_sr_all, de_pp, ordered_genes, ordered_cts, pct):
    keep = set(ordered_cts)
    pan = (de_pp.filter(pl.col('cell_type').is_in(list(keep))
                        & pl.col('gene').is_in(ordered_genes)
                        & pl.col('logFC').is_not_null())
           .group_by(['gene', 'cell_type'])
           .agg(pl.col('logFC').median().alias('m')))
    meta = {(r['gene'], r['cell_type']): r['m']
            for r in pan.iter_rows(named=True)}
    d_look = {(r['gene'], r['cell_type']): r['D'] for r in de_sr_all.filter(
        pl.col('cell_type').is_in(list(keep))
        & pl.col('gene').is_in(ordered_genes)).iter_rows(named=True)}
    sig_set = {(r['gene'], r['cell_type']) for r in de_sr.filter(
        pl.col('cell_type').is_in(list(keep))
        & pl.col('gene').is_in(ordered_genes)).iter_rows(named=True)}
    ng, nc = len(ordered_genes), len(ordered_cts)
    lfc = np.full((ng, nc), np.nan)
    pmat = np.full((ng, nc), np.nan)
    sig = np.zeros((ng, nc), dtype=bool)
    d = np.zeros((ng, nc), dtype=int)
    for i, g in enumerate(ordered_genes):
        for j, ct in enumerate(ordered_cts):
            lfc[i, j] = meta.get((g, ct), np.nan)
            sig[i, j] = (g, ct) in sig_set
            d[i, j] = d_look.get((g, ct), 0)
            pcts = pct.get((g, ct), [])
            if pcts:
                pmat[i, j] = float(np.median(pcts))
    return lfc, pmat, sig, d

# --- LIANA chord backend -----------------------------------------------------

def _is_nn(c):
    return ' NN' in c or c.endswith(' NN')


def _is_neuron(c):
    return ('Glut' in c) or ('Gaba' in c) or ('IMN' in c)


def chord_class(ct):
    if 'NN' in ct: return 'NN'
    if 'Glut' in ct: return 'Glut'
    return 'GABA'


def build_chord(liana_path, canonical_pairs, theme_ligands, k_neurons=10,
                contrast='PREG_vs_CTRL', neuron_intrinsic=False,
                include_nn=False, cell_allow=None, mag_floor=True):
    """LIANA-only chord pool + edges. pool = NN-niche backbone (all NN with
    >=1 surviving theme edge) + top-K neurons by |meta_diff| to the backbone.
    With ``neuron_intrinsic=True`` the pool is neuron-centric (top-K neurons by
    total edge |meta_diff|); ``include_nn=True`` additionally keeps the glial
    (NN) partners that signal to those neurons, so a neuron figure can show its
    glial niche without the glia-backbone framing. Returns (cell_set, edges)
    with edges cols [source, target, ligand_complex, receptor_complex,
    meta_diff, theme, mag]."""
    ligs = sorted({p[0] for p in canonical_pairs})
    recs = sorted({p[1] for p in canonical_pairs})
    lig2t = {l: t for t, ls in theme_ligands.items() for l in ls}
    scan = pl.scan_parquet if str(liana_path).endswith('.parquet') \
        else pl.scan_csv
    d = (scan(liana_path).filter(pl.col('contrast') == contrast)
         .filter(pl.col('ligand_complex').is_in(ligs)
                 & pl.col('receptor_complex').is_in(recs))
         .select(['dataset', 'source', 'target', 'ligand_complex',
                  'receptor_complex', 'lr_mean_diff'])
         .collect(engine='streaming'))
    d = d.filter(pl.struct(['ligand_complex', 'receptor_complex'])
                 .map_elements(lambda s: (s['ligand_complex'],
                                          s['receptor_complex'])
                               in canonical_pairs, return_dtype=pl.Boolean))
    keys = ['source', 'target', 'ligand_complex', 'receptor_complex']
    wide = (d.group_by(keys + ['dataset'])
            .agg(pl.col('lr_mean_diff').first())
            .pivot(on='dataset', index=keys, values='lr_mean_diff'))
    for c in ('slidetags', 'xenium'):
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    wide = wide.rename({'slidetags': 'st', 'xenium': 'xn'})
    nz = d.filter(pl.col('lr_mean_diff') != 0)
    if mag_floor:
        m_xn = float(nz.filter(pl.col('dataset') == 'xenium')['lr_mean_diff']
                     .abs().median() or 0.0)
        m_st = float(nz.filter(pl.col('dataset') == 'slidetags')['lr_mean_diff']
                     .abs().median() or 0.0)
    else:
        m_xn = m_st = 0.0
    xp = (wide.filter(pl.col('st').is_not_null() & pl.col('xn').is_not_null()
                      & (pl.col('st') != 0) & (pl.col('xn') != 0)
                      & (pl.col('st').sign() == pl.col('xn').sign())
                      & (pl.col('st').abs() >= m_st)
                      & (pl.col('xn').abs() >= m_xn))
          .with_columns(((pl.col('st') + pl.col('xn')) / 2).alias('meta_diff'))
          .select(keys + ['meta_diff']))
    xp_pairs = {(r['ligand_complex'], r['receptor_complex']) for r in
                wide.filter(pl.col('st').is_not_null()
                            & pl.col('xn').is_not_null())
                .select(['ligand_complex', 'receptor_complex'])
                .unique().iter_rows(named=True)}
    st_only = set(canonical_pairs) - xp_pairs
    st = d.filter((pl.col('dataset') == 'slidetags')
                  & (pl.col('lr_mean_diff') != 0)).filter(
        pl.struct(['ligand_complex', 'receptor_complex']).map_elements(
            lambda s: (s['ligand_complex'], s['receptor_complex'])
            in st_only, return_dtype=pl.Boolean))
    thr = float(st['lr_mean_diff'].abs().quantile(0.90)) if st.height else 0.0
    st = (st.filter(pl.col('lr_mean_diff').abs() >= thr)
          .rename({'lr_mean_diff': 'meta_diff'}).select(keys + ['meta_diff']))
    edges = pl.concat([xp, st]).filter(pl.col('source') != pl.col('target'))
    edges = edges.with_columns(
        pl.col('ligand_complex')
          .map_elements(lambda l: lig2t.get(l), return_dtype=pl.Utf8)
          .alias('theme'),
        pl.col('meta_diff').abs().alias('mag')).filter(
            pl.col('theme').is_not_null())
    if cell_allow is not None:
        edges = edges.filter(pl.col('source').is_in(list(cell_allow))
                             & pl.col('target').is_in(list(cell_allow)))
    if neuron_intrinsic:
        # rank neurons by total incident edge magnitude (over all edges if
        # NN partners are kept, else only neuron<->neuron)
        rows = list(edges.select(['source', 'target', 'mag'])
                    .iter_rows(named=True))
        nscore = {}
        for r in rows:
            for c in (r['source'], r['target']):
                if _is_neuron(c):
                    nscore[c] = nscore.get(c, 0.0) + r['mag']
        neurons = set(c for c, _ in sorted(nscore.items(),
                      key=lambda kv: -kv[1])[:k_neurons])
        if include_nn:
            nn = set()
            for r in rows:
                s, t = r['source'], r['target']
                if _is_nn(s) and t in neurons:
                    nn.add(s)
                if _is_nn(t) and s in neurons:
                    nn.add(t)
            cell_set = sorted(neurons | nn, key=numeric_prefix)
        else:
            cell_set = sorted(neurons, key=numeric_prefix)
    else:
        cells = (set(edges['source'].to_list())
                 | set(edges['target'].to_list()))
        backbone = {c for c in cells if _is_nn(c)}
        score = {}
        for r in edges.select(['source', 'target', 'mag']).iter_rows(
                named=True):
            s, t, m = r['source'], r['target'], r['mag']
            if _is_neuron(s) and t in backbone:
                score[s] = score.get(s, 0.0) + m
            if _is_neuron(t) and s in backbone:
                score[t] = score.get(t, 0.0) + m
        neurons = [c for c, _ in sorted(score.items(), key=lambda kv: -kv[1])
                   [:k_neurons]]
        cell_set = sorted(backbone | set(neurons), key=numeric_prefix)
    edges = edges.filter(pl.col('source').is_in(cell_set)
                         & pl.col('target').is_in(cell_set))
    return cell_set, edges


def theme_chord_edges(edges, ligands):
    return (edges.filter(pl.col('ligand_complex').is_in(ligands))
            .group_by(['source', 'target'])
            .agg(pl.col('meta_diff').sum().alias('signed_sum'),
                 pl.col('meta_diff').abs().sum().alias('mag'),
                 pl.len().alias('n_lr')))

# --- core layout -------------------------------------------------------------

# fixed geometry shared by every figure
LEFT_FIG_PAD_IN = 1.15
LABEL_MARGIN_IN = 1.55
LEG_LEFT_IN = 0.45
LEG_W_IN = 1.00
LEG_TITLE_FS = 8.4       # legend section titles
LEG_ITEM_FS = 7.9        # legend item labels
SWATCH_W_IN = 0.11
SWATCH_H_IN = 0.085
PATH_PITCH = 0.21
GENE_PITCH = 0.155
GAP_IN = 0.50
ANNO_W_IN, ANNO_GAP_IN = 0.07, 0.03
COL_ANNO_GAP_IN = 0.02
TOP_MARGIN_IN = 0.85
CARD_GAP_IN = 0.10
CARD_TITLE_H_IN = 0.30
SP_GAP_IN = 0.05
SP_FOREST_GAP_IN = 0.10
FOREST_W_IN = 1.15
FOREST_LABEL_GAP_IN = 0.05
CARD_FOREST_ROW_H = 0.17
CARD_FOREST_MIN_H = 0.42
FLEG_GAP_IN = 0.12
FLEG_EXTRA_RIGHT_IN = 0.20
FLEG_COL_W_IN = 1.28
CBAR_W_FIG = 0.005


def label_drop_in(labels, fontsize=LABEL_FS, rotation=45.0):
    """Vertical extent (inches) of the longest rotated column label. Used to
    extend the gene cards down to the bottom of the x-axis cell-type labels
    (rather than stopping at the dotplot's bottom edge). Independent of figure
    size, so it can be measured before the layout is built."""
    figt = plt.figure(figsize=(2, 2))
    figt.canvas.draw()
    r = figt.canvas.get_renderer()
    drop = 0.0
    for lab in labels:
        t = figt.text(0.5, 0.5, str(lab), rotation=rotation, ha='right',
                      va='top', rotation_mode='anchor', fontsize=fontsize)
        drop = max(drop, t.get_window_extent(renderer=r).height / figt.dpi)
        t.remove()
    plt.close(figt)
    return drop


def text_width_in(labels, fontsize=LABEL_FS, pad_in=0.0):
    """Max rendered width (inches) of the given strings at ``fontsize``. Used to
    size the forest-plot label column so full cell-type names fit (the cards use
    the same full names as the dotplot x-axis)."""
    figt = plt.figure(figsize=(2, 2))
    figt.canvas.draw()
    r = figt.canvas.get_renderer()
    w = 0.0
    for lab in labels:
        t = figt.text(0.5, 0.5, str(lab), ha='left', va='center',
                      fontsize=fontsize)
        w = max(w, t.get_window_extent(renderer=r).width / figt.dpi)
        t.remove()
    plt.close(figt)
    return w + pad_in


def card_metrics(n_path, n_gene, n_ct, n_cards, max_sp_n, label_w_in=0.95,
                 n_sections=1, section_gap=0.0, n_path_extra=0.0,
                 n_gene_extra=0.0, card_extend_in=0.0, cards_pad_in=0.0):
    """Position-independent geometry (no vertical anchor). Lets a figure size
    a chord row before fixing ``ax_b_bot_in``. ``section_gap`` (column units)
    widens Panel A/B to make room for inter-section gaps; ``n_path_extra`` /
    ``n_gene_extra`` (row units) taller for inter-row-section gaps.
    ``card_extend_in`` (inches) grows the card stack downward by that amount so
    the cards reach the bottom of the column cell-type labels."""
    L = SimpleNamespace()
    L.n_rows, L.n_g, L.n_cols = n_path, n_gene, n_ct
    L.n_sections, L.section_gap = n_sections, section_gap
    L.ax_left_in = LEFT_FIG_PAD_IN + LABEL_MARGIN_IN
    L.ax_w_in = 0.20 * (n_ct + (n_sections - 1) * section_gap)
    L.ax_h_a_in = PATH_PITCH * (n_path + n_path_extra)
    L.ax_h_b_in = GENE_PITCH * (n_gene + n_gene_extra)
    L.N_CARDS = n_cards
    L.MAX_SP_N = max_sp_n
    ab_vspan = L.ax_h_b_in + GAP_IN + L.ax_h_a_in
    L.content_h_in = (ab_vspan + card_extend_in - (n_cards - 1)
                      * (CARD_TITLE_H_IN + CARD_GAP_IN)) / n_cards
    L.SP_W_IN = L.content_h_in
    L.card_total_h_in = L.content_h_in + CARD_TITLE_H_IN
    card_w = (2 * L.SP_W_IN + SP_GAP_IN + SP_FOREST_GAP_IN + FOREST_W_IN
              + FOREST_LABEL_GAP_IN + label_w_in)
    L.card_w_max_in = card_w + max(0, max_sp_n - 2) * (L.SP_W_IN + SP_GAP_IN)
    L.cards_left_in = (L.ax_left_in + L.ax_w_in + ANNO_GAP_IN + ANNO_W_IN
                       + 0.35 + cards_pad_in)
    L.label_w_in = label_w_in
    return L


def core_layout(n_path, n_gene, n_ct, n_cards, max_sp_n, ax_b_bot_in,
                label_w_in=0.95, n_sections=1, section_gap=0.0,
                n_path_extra=0.0, n_gene_extra=0.0, card_extend_in=0.0,
                cards_pad_in=0.0):
    """Full positions for the shared core (Panel A/B + cards + legends),
    stacked above ``ax_b_bot_in`` (Panel B bottom edge in inches). The figure
    reserves whatever it needs below ax_b_bot_in for a chord row / extras.
    ``card_extend_in`` grows the cards down to the column-label bottom."""
    L = card_metrics(n_path, n_gene, n_ct, n_cards, max_sp_n, label_w_in,
                     n_sections, section_gap, n_path_extra, n_gene_extra,
                     card_extend_in, cards_pad_in)
    L.ax_b_bot_in = ax_b_bot_in
    L.ax_b_top_in = ax_b_bot_in + L.ax_h_b_in
    L.ax_a_bot_in = L.ax_b_top_in + GAP_IN
    L.ax_a_top_in = L.ax_a_bot_in + L.ax_h_a_in
    L.cards_top_in = L.ax_a_top_in + CARD_TITLE_H_IN
    L.fig_w = (L.cards_left_in + L.card_w_max_in + FLEG_GAP_IN
               + FLEG_EXTRA_RIGHT_IN + FLEG_COL_W_IN + 0.10)
    L.fig_h = L.ax_a_top_in + TOP_MARGIN_IN
    L.anno_x_in = L.ax_left_in + L.ax_w_in + ANNO_GAP_IN
    L.flegv_left_in = (L.cards_left_in + L.card_w_max_in + FLEG_GAP_IN
                       + FLEG_EXTRA_RIGHT_IN)
    L.cbar_x_fig = ((LEG_LEFT_IN + 0.95 * LEG_W_IN) / L.fig_w
                    - CBAR_W_FIG / 2)
    return L


# chord row geometry (figure supplies the vertical scheme; these size + place
# the panel grid: 1xN row for microglia/vascular, 2x2 for lipid)
CHORD_PANEL_GAP_IN = 0.30
CHORD_TITLE_H_IN = 0.30


def chord_full_span(cm, max_sp_n):
    return (cm.cards_left_in + max_sp_n * cm.SP_W_IN
            + (max_sp_n - 1) * SP_GAP_IN + SP_FOREST_GAP_IN + FOREST_W_IN
            - cm.ax_left_in)


def chord_panel_size(span_in, n_themes, ncols, panel_gap, aspect=0.85):
    cpw = (span_in - (ncols - 1) * panel_gap) / ncols
    return cpw, cpw * aspect


def chord_band_height(cph, n_themes, ncols, title_h, row_gap=0.0, extra=0.0):
    nrows = -(-n_themes // ncols)
    return nrows * (cph + title_h) + (nrows - 1) * row_gap + extra


def chord_grid_specs(themes, ax_left_in, row_bot_in, cpw, cph, ncols,
                     panel_gap, row_gap=0.0, title_h=0.0, bottom_drop=0.0):
    nrows = -(-len(themes) // ncols)
    step = cph + row_gap + title_h
    specs = []
    for i, t in enumerate(themes):
        r, c = divmod(i, ncols)
        left = ax_left_in + c * (cpw + panel_gap)
        bot = (row_bot_in - bottom_drop) + (nrows - 1 - r) * step
        specs.append((t, left, bot, cpw, cph))
    return specs


def add_axes(fig, L, x_in, y_in, w_in, h_in, **kw):
    return fig.add_axes([x_in / L.fig_w, y_in / L.fig_h,
                         w_in / L.fig_w, h_in / L.fig_h], **kw)


def save(fig, stem, pad=0.1):
    """Save png + svg cropped to the figure's tight bbox + ``pad`` inches.
    Uses get_tightbbox explicitly (deterministic; savefig's own
    bbox_inches='tight' over-pads for these figures)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(pad)
    for ext in ('png', 'svg'):
        fig.savefig(f'{stem}.{ext}', bbox_inches=bb, facecolor='white')
    plt.close(fig)

# --- Panel A / B dotplots ----------------------------------------------------

def _dotplot(ax, L, nr, present, size_of, color_of, d_of, sig_of,
             class_spans, band_spans, yticklabels, ylabel, ytick_italic,
             bold_labels=None, row_y=None, row_sections=None):
    # ``row_y`` (optional) maps each row index to a y-position, letting a
    # figure insert vertical gaps between row super-sections; ``row_sections``
    # is a list of (lo, hi) row spans, each boxed (outer spine suppressed).
    nc = L.n_cols
    sectioned = getattr(L, 'section_gap', 0.0) > 0
    col_x = getattr(L, 'col_x', None) if sectioned else None
    if col_x is None:
        col_x = list(range(nc))
    x_hi = col_x[-1]
    ry = row_y if row_y is not None else list(range(nr))
    y_hi = ry[-1]
    xs, ys, ss, cs, ec, lw = [], [], [], [], [], []
    sx, sy = [], []
    for i in range(nr):
        for j in range(nc):
            if not present(i, j):
                continue
            xs.append(col_x[j]); ys.append(ry[i])
            ss.append(size_of(i, j)); cs.append(color_of(i, j))
            d3 = d_of(i, j) == 3
            ec.append('#999999' if d3 else 'none')
            lw.append(0.8 if d3 else 0.0)
            if sig_of(i, j):
                sx.append(col_x[j]); sy.append(ry[i])
    ax.scatter(xs, ys, s=ss, c=cs, edgecolors=ec, linewidths=lw, zorder=3)
    if sx:
        ax.scatter(sx, sy, s=SIG_DOT_SIZE, c='white', edgecolors='none',
                   linewidths=0, zorder=4)
    if sectioned:
        # boxes = (column super-section) x (row super-section); when no
        # row_sections are given this is one box per column group (old behavior)
        rsecs = row_sections if row_sections is not None else [(0, nr - 1)]
        for _, lo, hi in class_spans:
            x0, x1 = col_x[lo] - 0.5, col_x[hi] + 0.5
            for rlo, rhi in rsecs:  # vgridlines + box per section
                y0, y1 = ry[rlo] - 0.5, ry[rhi] + 0.5
                for j in range(lo + 1, hi + 1):
                    ax.plot([col_x[j] - 0.5] * 2, [y0, y1], color='#F2F2F2',
                            lw=0.25, zorder=1)
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                             fill=False, edgecolor='black', lw=0.9, zorder=6,
                             clip_on=False))
            for k in range(1, nr):  # row gridlines (skip gaps)
                if ry[k] - ry[k - 1] < 1.5:
                    ax.plot([x0, x1], [(ry[k] + ry[k - 1]) / 2] * 2,
                            color='#F2F2F2', lw=0.25, zorder=1)
            for _, _, bhi in band_spans[:-1]:  # band sep (skip gaps)
                if bhi + 1 < nr and ry[bhi + 1] - ry[bhi] < 1.5:
                    ax.plot([x0, x1], [(ry[bhi] + ry[bhi + 1]) / 2] * 2,
                            color='#BBBBBB', lw=0.4, zorder=2)
    else:
        for k in range(1, nc):
            ax.axvline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
        for i in range(1, nr):                       # row gridlines (skip gaps)
            if ry[i] - ry[i - 1] < 1.5:
                ax.axhline((ry[i] + ry[i - 1]) / 2, color='#F2F2F2',
                           lw=0.25, zorder=1)
        for _, _, hi in class_spans[:-1]:
            ax.axvline(hi + 0.5, color='#BBBBBB', lw=0.4, zorder=2)
        for _, _, hi in band_spans[:-1]:             # band sep (skip gaps)
            if hi + 1 < nr and ry[hi + 1] - ry[hi] < 1.5:
                ax.axhline((ry[hi] + ry[hi + 1]) / 2, color='#BBBBBB',
                           lw=0.4, zorder=2)
        if row_sections is not None:
            for lo, hi in row_sections:
                ax.add_patch(plt.Rectangle((-0.5, ry[lo] - 0.5), nc,
                             ry[hi] - ry[lo] + 1, fill=False,
                             edgecolor='black', lw=0.9, zorder=6,
                             clip_on=False))
    ax.set_xlim(-0.5, x_hi + 0.5)
    ax.set_ylim(y_hi + 0.5, -0.5)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.set_yticks(ry)
    ax.set_yticklabels(yticklabels, fontsize=LABEL_FS,
                       fontstyle='italic' if ytick_italic else 'normal')
    if bold_labels:
        bs = set(bold_labels)
        for lab in ax.get_yticklabels():
            if lab.get_text() in bs:
                lab.set_fontweight('bold')
    ax.tick_params(axis='y', length=2, pad=1)
    ax.set_ylabel(ylabel, fontsize=TITLE_FS, labelpad=8)
    show_spines = (not sectioned) and (row_sections is None)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax.spines[sp].set_visible(show_spines)
        if show_spines:
            ax.spines[sp].set_color('black')
            ax.spines[sp].set_linewidth(0.9)


def _section_titles(ax, L, class_spans):
    col_x = getattr(L, 'col_x', None)
    trans = ax.get_xaxis_transform()
    for cls, lo, hi in class_spans:
        cx = (col_x[lo] + col_x[hi]) / 2 if col_x else (lo + hi) / 2
        ax.text(cx, 1.04, cls, ha='center', va='bottom', fontsize=TITLE_FS,
                color='black', transform=trans, clip_on=False,
                linespacing=1.05)


def draw_panel_a(ax, L, nlp, nes, d, sig, norm_nes, ordered_pathways,
                 pathway_labels, class_spans, band_spans,
                 row_y=None, row_sections=None):
    def color_of(i, j):
        v = nes[i, j]
        if np.isnan(v):
            return '#999999'
        return CMAP(norm_nes(np.clip(v, norm_nes.vmin, norm_nes.vmax)))
    _dotplot(ax, L, len(ordered_pathways),
             lambda i, j: not np.isnan(nlp[i, j]),
             lambda i, j: nlp_to_size(nlp[i, j]), color_of,
             lambda i, j: d[i, j], lambda i, j: sig[i, j],
             class_spans, band_spans,
             [pathway_labels.get(p, p) for p in ordered_pathways],
             'GSEA pathways (meta)', False,
             row_y=row_y, row_sections=row_sections)
    _section_titles(ax, L, class_spans)


def draw_panel_b(ax, L, lfc, pct, d, sig, norm_lfc, ordered_genes,
                 class_spans, gene_band_spans, card_genes=(),
                 row_y=None, row_sections=None):
    ax.set_facecolor('white')

    def present(i, j):
        return d[i, j] >= 2 and not np.isnan(pct[i, j])

    def color_of(i, j):
        v = lfc[i, j]
        if np.isnan(v):
            return '#CCCCCC'
        return CMAP(norm_lfc(np.clip(v, norm_lfc.vmin, norm_lfc.vmax)))
    _dotplot(ax, L, len(ordered_genes), present,
             lambda i, j: pct_to_size(pct[i, j]), color_of,
             lambda i, j: d[i, j], lambda i, j: sig[i, j],
             class_spans, gene_band_spans, ordered_genes,
             'DE genes (meta)', True, bold_labels=card_genes,
             row_y=row_y, row_sections=row_sections)

# --- annotation strips -------------------------------------------------------

def draw_col_anno(fig, L, panel_bot_in, ordered_cts, ticklabels,
                  subclass_colors):
    bot = panel_bot_in - COL_ANNO_GAP_IN - ANNO_W_IN
    ax = add_axes(fig, L, L.ax_left_in, bot, L.ax_w_in, ANNO_W_IN)
    col_x = getattr(L, 'col_x', None) or list(range(L.n_cols))
    ax.set_xlim(-0.5, col_x[-1] + 0.5); ax.set_ylim(0, 1)
    for j, ct in enumerate(ordered_cts):
        ax.add_patch(plt.Rectangle((col_x[j] - 0.5, 0), 1, 1,
                     facecolor=subclass_colors.get(ct, '#d3d3d3'),
                     edgecolor='none'))
    ax.set_yticks([])
    ax.set_xticks(col_x)
    ax.set_xticklabels(ticklabels, rotation=45, ha='right',
                       rotation_mode='anchor', fontsize=LABEL_FS)
    ax.tick_params(axis='x', length=3, pad=2, direction='out', bottom=True,
                   top=False, labelbottom=True, labeltop=False)
    ax.tick_params(axis='y', left=False, right=False, labelleft=False,
                   labelright=False)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_band_anno(fig, L, bot_in, h_in, n, item_band, band_colors,
                   row_y=None):
    ax = add_axes(fig, L, L.anno_x_in, bot_in, ANNO_W_IN, h_in)
    ry = row_y if row_y is not None else list(range(n))
    ax.set_xlim(0, 1); ax.set_ylim(ry[-1] + 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        ax.add_patch(plt.Rectangle((0, ry[i] - 0.5), 1, 1,
                     facecolor=band_colors[item_band[i]], edgecolor='none'))


def draw_group_labels(fig, L, panel_bot_in, panel_h_in, ry, sections, labels,
                      x_in, fontsize=LABEL_FS, rotation=270):
    """Rotated theme-super-group titles: one per ``sections`` (lo, hi) span,
    centered on the group's vertical extent, at figure x ``x_in`` (inches)."""
    y_top, y_bot = -0.5, ry[-1] + 0.5
    span = y_bot - y_top
    for (lo, hi), lab in zip(sections, labels):
        yc = (ry[lo] + ry[hi]) / 2
        frac = (yc - y_top) / span                       # 0 at panel top
        y_in = (panel_bot_in + panel_h_in) - frac * panel_h_in
        fig.text(x_in / L.fig_w, y_in / L.fig_h, lab, rotation=rotation,
                 ha='center', va='center', fontsize=fontsize, linespacing=0.9)

# --- legends -----------------------------------------------------------------

def _leg_axes(fig, L, bot_in, h_in):
    ax = add_axes(fig, L, LEG_LEFT_IN, bot_in, LEG_W_IN, h_in, zorder=100)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.patch.set_facecolor('white'); ax.patch.set_alpha(1.0)
    ax.patch.set_zorder(99)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def draw_dot_legends(fig, L, norm_nes, norm_lfc, nes_vmax, lfc_vmax,
                     band_names, band_colors):
    """Single top-down flow (inches). Uniform title->element + inter-item
    gaps; an extra category gap separates the GSEA / DE / DE-GSEA groups."""
    TITLE_H = 0.24         # 2-line title block
    TITLE_GAP = 0.06       # title bottom -> first element (tight, consistent)
    DOT_PITCH = 0.20       # row pitch for dot rows
    SW_PITCH = 0.155       # row pitch for swatch rows
    CBAR_H = 0.19          # colorbar bar height
    END_DOT = 0.12         # trailing pad below last dot
    END_SW = 0.08          # trailing pad below last swatch
    SUB_GAP = 0.15         # gap between items within a category
    CAT_GAP = 0.34         # extra gap between categories
    DOT_X, LAB_X = 0.86, 0.70

    # (category, title, kind, payload)
    items = [
        ('GSEA', 'GSEA\n' + r'$-\log_{10}$ emp $p$', 'sizedots',
         [1.5, 2.5, 4.0]),
        ('GSEA', 'GSEA\nNES (median)', 'cbar', (norm_nes, nes_vmax)),
        ('DE', 'DE\nlogFC (median)', 'cbar', (norm_lfc, lfc_vmax)),
        ('DE', 'DE\nPercent expressed', 'pctdots', [10, 50, 90]),
        ('DE/GSEA', 'DE/GSEA\nCross-platform', 'ddots', None),
        ('DE/GSEA', 'DE/GSEA\nSignificance', 'sig', None),
        ('DE/GSEA', 'DE/GSEA\nTheme', 'swatches', band_names),
    ]

    def item_h(kind, payload):
        if kind == 'cbar':
            return TITLE_H + TITLE_GAP + CBAR_H + 0.06
        n = 1 if kind == 'sig' else 2 if kind == 'ddots' else len(payload)
        pitch = SW_PITCH if kind == 'swatches' else DOT_PITCH
        end = END_SW if kind == 'swatches' else END_DOT
        return TITLE_H + TITLE_GAP + (n - 1) * pitch + end

    # flow: title-top (inches) of each item, top-down from Panel B top edge
    tops, y, prev = [], L.ax_b_top_in, None
    for cat, _, kind, payload in items:
        if prev is not None:
            y -= SUB_GAP + (CAT_GAP if cat != prev else 0.0)
        tops.append(y)
        y -= item_h(kind, payload)
        prev = cat
    bottom_in = y

    lax = add_axes(fig, L, LEG_LEFT_IN, bottom_in, LEG_W_IN,
                   L.ax_b_top_in - bottom_in, zorder=100)
    lax.set_xlim(0, 1); lax.set_ylim(bottom_in, L.ax_b_top_in)
    lax.set_xticks([]); lax.set_yticks([]); lax.patch.set_alpha(0)
    for s in lax.spines.values():
        s.set_visible(False)

    for (cat, ttl, kind, payload), yt in zip(items, tops):
        lax.text(1.0, yt, ttl, ha='right', va='top', fontsize=LEG_TITLE_FS,
                 linespacing=1.0)
        e0 = yt - TITLE_H - TITLE_GAP            # first-element baseline
        if kind in ('sizedots', 'pctdots'):
            col = '#555555' if kind == 'sizedots' else '#777777'
            for k, lev in enumerate(payload):
                yc = e0 - k * DOT_PITCH
                s = nlp_to_size(lev) if kind == 'sizedots' else pct_to_size(lev)
                lax.scatter([DOT_X], [yc], s=s, c=[col], edgecolors='none',
                            clip_on=False)
                lab = f'{lev:.1f}' if kind == 'sizedots' else f'{lev}%'
                lax.text(LAB_X, yc, lab, ha='right', va='center',
                         fontsize=LEG_ITEM_FS)
        elif kind == 'ddots':
            for k, (D, lab) in enumerate([(3, 'D = 3'), (2, 'D = 2')]):
                yc = e0 - k * DOT_PITCH
                lax.scatter([DOT_X], [yc], s=64, c=['#bbbbbb'],
                            edgecolors='#999999' if D == 3 else 'none',
                            linewidths=0.8 if D == 3 else 0, clip_on=False)
                lax.text(LAB_X, yc, lab, ha='right', va='center',
                         fontsize=LEG_ITEM_FS)
        elif kind == 'sig':
            yc = e0
            lax.scatter([DOT_X], [yc], s=64, c=[CMAP(norm_lfc(lfc_vmax * 0.7))],
                        edgecolors='none', clip_on=False)
            lax.scatter([DOT_X], [yc], s=SIG_DOT_SIZE, c='white',
                        edgecolors='none', clip_on=False)
            lax.text(LAB_X, yc, r'emp $p \leq 0.05$', ha='right', va='center',
                     fontsize=LEG_ITEM_FS)
        elif kind == 'swatches':
            bw = SWATCH_W_IN / LEG_W_IN
            for k, band in enumerate(payload):
                yc = e0 - k * SW_PITCH
                lax.add_patch(plt.Rectangle((DOT_X - bw / 2, yc - SWATCH_H_IN
                              / 2), bw, SWATCH_H_IN,
                              facecolor=band_colors[band], edgecolor='none'))
                lax.text(DOT_X - bw / 2 - 0.03, yc, band, ha='right',
                         va='center', fontsize=LEG_ITEM_FS, color='black')
        elif kind == 'cbar':
            norm, vmax = payload
            bar_bot = e0 - CBAR_H
            cax = fig.add_axes([L.cbar_x_fig, bar_bot / L.fig_h, CBAR_W_FIG,
                                CBAR_H / L.fig_h], zorder=110)
            cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CMAP),
                              cax=cax, orientation='vertical')
            cb.set_ticks([-vmax, 0, vmax]); cb.ax.yaxis.tick_left()
            cb.ax.tick_params(labelsize=LEG_ITEM_FS, length=2, pad=1)

    return bottom_in   # bottom of legend column, for stacking below


def draw_forest_legend(fig, L):
    left = L.flegv_left_in
    top = L.cards_top_in
    bot = L.ax_b_bot_in
    h = top - bot
    ax = add_axes(fig, L, left, bot, FLEG_COL_W_IN, h)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    sections = [
        ('Platforms', [('line', p, PLATFORM_LABELS[p])
                       for p in ('slidetags', 'merfish', 'xenium')]),
        ('Meta', [('diamond', 3, 'D=3'), ('diamond', 2, 'D=2')]),
        ('Significance', [('star', 1, r'*  emp$\,p\leq 0.05$'),
                          ('star', 2, r'**  emp$\,p\leq 0.01$'),
                          ('star', 3, r'***  emp$\,p\leq 0.001$')]),
    ]
    hh = 0.090 * 1.95 / h
    ih = 0.074 * 1.95 / h
    sg = 0.058 * 1.95 / h
    y = 1.0 - 0.03 * 1.95 / h
    for header, items in sections:
        ax.text(0.05, y, header, ha='left', va='top', fontsize=LEG_TITLE_FS,
                transform=ax.transAxes)
        y -= hh
        for kind, key, label in items:
            if kind == 'line':
                ax.plot([0.05, 0.14], [y, y], color=PLATFORM_COLORS[key],
                        lw=1.8, solid_capstyle='butt',
                        transform=ax.transAxes, clip_on=False)
                ax.text(0.17, y, label, ha='left', va='center',
                        fontsize=LEG_ITEM_FS, transform=ax.transAxes)
            elif kind == 'diamond':
                ax.plot(0.095, y, 'D', mfc=D_COLORS[key], mec='black',
                        mew=0.4, ms=4.6, transform=ax.transAxes,
                        clip_on=False)
                ax.text(0.17, y, label, ha='left', va='center',
                        fontsize=LEG_ITEM_FS, transform=ax.transAxes)
            else:
                ax.text(0.05, y, label, ha='left', va='center',
                        fontsize=LEG_ITEM_FS, transform=ax.transAxes)
            y -= ih
        y -= sg

# --- gene cards --------------------------------------------------------------

def _class_rank_card(ct):
    if 'Glut' in ct: return 0
    if 'NN' in ct: return 2
    return 1


def build_cards(card_genes, card_ctx, max_rows_map, de_sr, de_sr_all, de_pp,
                keep_cts, sp_in, sp_coords, sp_expr):
    keep = set(keep_cts)
    d_look = {(r['gene'], r['cell_type']): r['D']
              for r in de_sr_all.iter_rows(named=True)}

    def build(gene, ctx, max_rows):
        hits = de_sr.filter((pl.col('gene') == gene)
                            & pl.col('cell_type').is_in(list(keep)))
        cts_in = list(hits['cell_type'].to_list())
        for ct in ctx:
            if ct in keep and ct not in cts_in:
                cts_in.append(ct)
        rows = []
        for ct in cts_in:
            plat = {p: dict(lfc=np.nan, lci=np.nan, uci=np.nan)
                    for p in PLATFORMS}
            ppr = de_pp.filter((pl.col('gene') == gene)
                               & (pl.col('cell_type') == ct))
            for r in ppr.iter_rows(named=True):
                if r['dataset'] in plat:
                    plat[r['dataset']] = dict(lfc=r['logFC'], lci=r['LCI'],
                                              uci=r['UCI'])
            valid = [plat[p]['lfc'] for p in PLATFORMS
                     if not np.isnan(plat[p]['lfc'])]
            meta_lfc = float(np.median(valid)) if valid else np.nan
            srr = hits.filter(pl.col('cell_type') == ct)
            if srr.height:
                sr = srr.to_dicts()[0]
                ep, D, st = sr['emp_p'], int(sr['D']), stars_for(sr['emp_p'])
            else:
                ep, D, st = np.nan, int(d_look.get((gene, ct), 0)), ''
            rows.append(dict(cell_type=ct, plat=plat, D=D, meta_lfc=meta_lfc,
                             emp_p=ep, stars=st))
        if max_rows and len(rows) > max_rows:
            forced = set(ctx)
            keep_r = [r for r in rows if r['cell_type'] in forced]
            others = [r for r in rows if r['cell_type'] not in forced]
            others.sort(key=lambda r: r['emp_p']
                        if not np.isnan(r['emp_p']) else 9.0)
            rows = keep_r + others[:max_rows - len(keep_r)]
        rows.sort(key=lambda r: (_class_rank_card(r['cell_type']),
                                 numeric_prefix(r['cell_type'])))
        cts = [r['cell_type'] for r in rows]
        sp_pair = [p for p in ('xenium', 'merfish', 'slidetags')
                   if sp_in[p].get(gene, False)]
        sp_data, sp_vr = [], []
        ct_set = set(cts)
        for ds in sp_pair:
            c = sp_coords[ds]
            expr = sp_expr[ds][gene]
            mask = np.isin(c['cell_type'], list(ct_set)) & (
                ((c['cond'] == 'CTRL') & (c['x'] < c['midline']))
                | ((c['cond'] == 'PREG') & (c['x'] >= c['midline'])))
            sd = dict(x=c['x'][mask], y=c['y'][mask], expr=expr[mask],
                      midline=c['midline'], fov_cx=c['fov_cx'],
                      fov_cy=c['fov_cy'], fov_half=c['fov_half'])
            sp_data.append(sd)
            sp_vr.append(pctl_range(sd['expr'][sd['expr'] > 0]))
        return dict(cell_types=cts, rows=rows, sp_pair=sp_pair,
                    sp_data=sp_data, sp_vranges=sp_vr)

    cards = {g: build(g, card_ctx.get(g, []), max_rows_map.get(g))
             for g in card_genes}
    max_sp_n = max(len(c['sp_pair']) for c in cards.values())
    return cards, max_sp_n


def _draw_forest(axf, gd, show_xlabel):
    n = len(gd['rows'])
    axf.set_ylim(n - 0.5, -0.5)
    axf.axvline(0, color='grey', lw=0.4, zorder=1)
    vals = []
    for r in gd['rows']:
        for p in PLATFORMS:
            for k in ('lci', 'uci', 'lfc'):
                v = r['plat'][p].get(k, np.nan)
                if not np.isnan(v):
                    vals.append(v)
        if not np.isnan(r['meta_lfc']):
            vals.append(r['meta_lfc'])
    lo, hi = (float(np.percentile(vals, 2)),
              float(np.percentile(vals, 98))) if vals else (-1.0, 1.0)
    lo = min(lo, 0.0); hi = max(hi, 0.0)
    if hi - lo < 0.4:
        mid = (hi + lo) / 2
        lo, hi = mid - 0.2, mid + 0.2
    pad = (hi - lo) * 0.10
    axf.set_xlim(lo - pad, hi + pad)
    pos = {'slidetags': -0.18, 'merfish': 0, 'xenium': 0.18}
    for i, r in enumerate(gd['rows']):
        for p in PLATFORMS:
            lci, uci = r['plat'][p].get('lci', np.nan), \
                r['plat'][p].get('uci', np.nan)
            if not (np.isnan(lci) or np.isnan(uci)):
                axf.hlines(i + pos[p], lci, uci, color=PLATFORM_COLORS[p],
                           lw=0.9, zorder=3)
        if not np.isnan(r['meta_lfc']) and r['D'] >= 2:
            axf.plot(r['meta_lfc'], i, 'D', mfc=D_COLORS[r['D']],
                     mec='black', mew=0.3, ms=3.0, zorder=5)
    axf.set_yticks([])
    axf.tick_params(axis='x', labelsize=7.5, length=1.5, pad=1)
    if show_xlabel:
        axf.set_xlabel('logFC', fontsize=LABEL_FS, labelpad=1)
    for sp in axf.spines.values():
        sp.set_linewidth(0.5)


def draw_cards(fig, L, card_genes, cards, sp_coords):
    for i, g in enumerate(card_genes):
        gd = cards[g]
        card_top = L.cards_top_in - i * (L.card_total_h_in + CARD_GAP_IN)
        content_bot = card_top - CARD_TITLE_H_IN - L.content_h_in
        fig.text(L.cards_left_in / L.fig_w, (card_top - 0.02) / L.fig_h, g,
                 ha='left', va='top', fontsize=TITLE_FS, fontstyle='italic')
        n_sp = len(gd['sp_pair'])
        for si in range(n_sp):
            left = L.cards_left_in + si * (L.SP_W_IN + SP_GAP_IN)
            ax = add_axes(fig, L, left, content_bot, L.SP_W_IN, L.SP_W_IN)
            ds = gd['sp_pair'][si]
            ax.set_title(PLATFORM_LABELS[ds], fontsize=7.5, pad=1.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('black')
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            sd = gd['sp_data'][si]
            if sd is None or len(sd['x']) == 0:
                continue
            vmin, vmax = gd['sp_vranges'][si]
            order = np.argsort(sd['expr'])
            size = 250.0 / np.sqrt(len(sp_coords[ds]['x']))
            ax.scatter(sd['x'][order], sd['y'][order], c=sd['expr'][order],
                       cmap='viridis', s=size, vmin=vmin, vmax=vmax,
                       linewidths=0, rasterized=True)
            c = sp_coords[ds]
            ax.plot([sd['midline'], sd['midline']],
                    [c['y'].min(), c['y'].max()], color='white', lw=0.3,
                    zorder=2)
            ax.set_xlim(sd['fov_cx'] - sd['fov_half'],
                        sd['fov_cx'] + sd['fov_half'])
            ax.set_ylim(sd['fov_cy'] - sd['fov_half'],
                        sd['fov_cy'] + sd['fov_half'])
            ax.set_aspect('equal')
            ax.text(0.25, 0.97, 'Null', transform=ax.transAxes, ha='center',
                    va='top', color='white', fontsize=6.0, zorder=4)
            ax.text(0.75, 0.97, 'Preg', transform=ax.transAxes, ha='center',
                    va='top', color='white', fontsize=6.0, zorder=4)
        n = len(gd['rows'])
        fh = min(max(n * CARD_FOREST_ROW_H, CARD_FOREST_MIN_H), L.content_h_in)
        fbot = content_bot + (L.content_h_in - fh) / 2
        fleft = (L.cards_left_in + n_sp * L.SP_W_IN + (n_sp - 1) * SP_GAP_IN
                 + SP_FOREST_GAP_IN)
        axf = add_axes(fig, L, fleft, fbot, FOREST_W_IN, fh)
        _draw_forest(axf, gd, show_xlabel=(i == L.N_CARDS - 1))
        lab_frac = 1.0 + FOREST_LABEL_GAP_IN / FOREST_W_IN
        trans = axf.get_yaxis_transform()
        for ri, r in enumerate(gd['rows']):
            axf.text(lab_frac, ri, f"{r['cell_type']} "
                     f"{r['stars']}".strip(), transform=trans, ha='left',
                     va='center', fontsize=LABEL_FS, clip_on=False)

# --- chord -------------------------------------------------------------------

def _num_label(ct):
    m = re.match(r'^(\d+)', ct)
    return m.group(1) if m else ct


def draw_theme_chord(ax, df, cell_set, cells_ordered, subclass_colors):
    if df.height == 0:
        ax.text(0.5, 0.5, 'no edges', transform=ax.transAxes, ha='center',
                va='center', fontsize=7.0, color='#888888')
        return
    df = df.sort('mag', descending=True).head(40)
    sec = {c: 0.0 for c in cell_set}
    for r in df.iter_rows(named=True):
        sec[r['source']] += r['mag']
        sec[r['target']] += r['mag']
    floor = (max(sec.values()) if any(sec.values()) else 1.0) * 0.05
    sector = {c: max(sec[c], floor) for c in cells_ordered}
    space = [2.0]
    for k in range(1, len(cells_ordered)):
        a, b = cells_ordered[k], cells_ordered[k - 1]
        space.append(7.0 if chord_class(a) != chord_class(b) else 2.0)
    circos = Circos(sector, space=space)
    for s in circos.sectors:
        s.add_track((85, 91), r_pad_ratio=0.0).axis(
            fc=subclass_colors.get(s.name, '#d3d3d3'), ec='black', lw=0.3)
        s.text(_num_label(s.name), r=106, size=8.5, color='black',
               orientation='vertical')
    mx = float(df['mag'].max())
    so = {c: 0.0 for c in cell_set}
    to = {c: 0.0 for c in cell_set}
    for r in df.sort('mag', descending=False).iter_rows(named=True):
        s, t, w = r['source'], r['target'], r['mag']
        col = CHORD_COLOR_UP if r['signed_sum'] > 0 else CHORD_COLOR_DOWN
        s0, s1 = so[s], so[s] + w
        t0, t1 = to[t], to[t] + w
        so[s], to[t] = s1, t1
        circos.link((s, s0, s1), (t, t0, t1), color=col,
                    alpha=0.30 + 0.60 * (w / mx), direction=1,
                    height_ratio=0.50, arrow_length_ratio=0.05,
                    allow_twist=True)
    circos.plotfig(ax=ax)
    groups = {}
    for s in circos.sectors:
        groups.setdefault(chord_class(s.name), []).append(s)
    for cls, secs in groups.items():
        rads = [s.x_to_rad(0) for s in secs] + \
               [s.x_to_rad(s.size) for s in secs]
        ax.bar(x=(min(rads) + max(rads)) / 2, height=6,
               width=max(rads) - min(rads), bottom=94,
               facecolor=CHORD_CLASS_COLORS[cls], edgecolor='black',
               linewidth=0.4, align='center', zorder=0.5)


def draw_chord_legend(fig, L, top_in, cell_set, subclass_colors):
    hdr, row, gapv = 0.31, 0.128, 0.085
    h_in = (3 * hdr + (5 + len(cell_set)) * row + 2 * gapv + 0.06)
    ax = _leg_axes(fig, L, top_in - h_in, h_in)
    sw = SWATCH_W_IN / LEG_W_IN
    sh = SWATCH_H_IN / h_in
    sx = 0.95 - sw

    def d(v):
        return v / h_in

    def swatch(yy, label, color):
        ax.add_patch(plt.Rectangle((sx, yy - sh / 2), sw, sh,
                     facecolor=color, edgecolor='black', lw=0.3))
        ax.text(sx - 0.02, yy, label, ha='right', va='center',
                fontsize=LEG_ITEM_FS, clip_on=False)
    y = 1.0 - d(0.03)
    ax.text(0.95, y, 'CCC\nDirection', ha='right', va='top',
            fontsize=LEG_TITLE_FS, linespacing=1.0)
    y -= d(hdr)
    for lbl, col in [('UP in pregnancy', CHORD_COLOR_UP),
                     ('DOWN in pregnancy', CHORD_COLOR_DOWN)]:
        swatch(y, lbl, col); y -= d(row)
    y -= d(gapv)
    ax.text(0.95, y, 'CCC\nCell class', ha='right', va='top',
            fontsize=LEG_TITLE_FS, linespacing=1.0)
    y -= d(hdr)
    for cls in CHORD_CLASS_ORDER:
        swatch(y, CHORD_CLASS_LABELS[cls], CHORD_CLASS_COLORS[cls])
        y -= d(row)
    y -= d(gapv)
    ax.text(0.95, y, 'CCC\nSubclass', ha='right', va='top',
            fontsize=LEG_TITLE_FS, linespacing=1.0)
    y -= d(hdr)
    for ct in sorted(cell_set, key=numeric_prefix):
        swatch(y, ct, subclass_colors.get(ct, '#d3d3d3'))
        y -= d(row)


def draw_chord_row(fig, L, specs, edges, theme_ligands, theme_titles,
                   cell_set, subclass_colors, title_off_in=0.30,
                   ytitle_x_in=None, ytitle_cy_in=None):
    """specs: list of (theme, left_in, bot_in, w_in, h_in)."""
    cells_ordered = sorted(cell_set, key=lambda c: (
        CHORD_CLASS_ORDER.index(chord_class(c)), numeric_prefix(c)))
    axes = {}
    for theme, left, bot, w, h in specs:
        ax = add_axes(fig, L, left, bot, w, h, projection='polar')
        df = theme_chord_edges(edges, theme_ligands[theme])
        draw_theme_chord(ax, df, cell_set, cells_ordered, subclass_colors)
        axes[theme] = ax
    fig.canvas.draw()
    off = title_off_in / L.fig_h
    for theme, *_ in specs:
        bb = axes[theme].get_position()
        fig.text((bb.x0 + bb.x1) / 2, bb.y1 + off, theme_titles[theme],
                 ha='center', va='bottom', fontsize=TITLE_FS)
    if ytitle_x_in is not None:
        fig.text(ytitle_x_in / L.fig_w, ytitle_cy_in / L.fig_h,
                 'Spatial cell-cell\ncommunication (meta)', rotation=90,
                 ha='center', va='center', fontsize=9.0)


# --- IF-validation bar plots (GraphPad Prism style) --------------------------

CONDITION_COLORS = {'CTRL': '#7209b7', 'PREG': '#b5179e', 'POSTPART': '#f72585'}
IF_BAR_COLORS = (CONDITION_COLORS['CTRL'], CONDITION_COLORS['PREG'])


def _if_norm(x):
    # sheet headers use a non-breaking space ('Non\xa0Pregnant')
    return str(x).replace('\xa0', ' ').strip()


def read_if_validation(xlsx_path, sheet, groups=('Non Pregnant', 'Pregnant')):
    """Return {group: 1D float array} from the per-animal summary columns
    (headed by the group names) on the right of an IF-quantification sheet."""
    raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
    vs = raw.apply(lambda s: s.map(_if_norm)).values
    out = {}
    for g in groups:
        hits = np.argwhere(vs == g)
        if not len(hits):
            raise ValueError(f'{g!r} not found in sheet {sheet!r}')
        r, c = hits[0]
        col = pd.to_numeric(raw.iloc[r + 1:, c], errors='coerce').to_numpy()
        run = []
        for v in col:                    # leading contiguous numeric run
            if np.isnan(v):
                break
            run.append(float(v))
        out[g] = np.asarray(run)
    return out


def _sig_stars(p):
    return ('***' if p < 1e-3 else '**' if p < 1e-2
            else '*' if p < 0.05 else 'ns')


def draw_if_barplot(ax, xlsx_path, sheet, ylabel, *,
                    groups=('Non Pregnant', 'Pregnant'),
                    display_labels=None, colors=IF_BAR_COLORS,
                    bar_w=0.6, jitter=0.11, equal_var=False):
    """Two-group bar + scatter (mean bar, SEM whiskers, per-animal points,
    unpaired t-test significance bracket), styled to match the main figure:
    condition colours, no bold, shared LABEL_FS/TITLE_FS. Returns p."""
    from scipy import stats
    import matplotlib.ticker as mtick
    dl = {'Non Pregnant': 'Nulliparous'}
    if display_labels:
        dl.update(display_labels)
    labels = [dl.get(g, g) for g in groups]
    data = read_if_validation(xlsx_path, sheet, groups)
    xs = [0, 1]
    means = [float(np.mean(data[g])) for g in groups]
    sems = [float(stats.sem(data[g])) for g in groups]
    for x, g, col in zip(xs, groups, colors):
        ax.bar(x, np.mean(data[g]), width=bar_w, facecolor=col,
               edgecolor='black', linewidth=0.9, zorder=2)
    ax.errorbar(xs, means, yerr=sems, fmt='none', ecolor='black',
                elinewidth=0.9, capsize=4, capthick=0.9, zorder=4)
    rng = np.random.default_rng(0)
    for x, g in zip(xs, groups):
        v = data[g]
        ax.scatter(x + rng.uniform(-jitter, jitter, size=len(v)), v, s=22,
                   color='black', edgecolors='white', linewidths=0.5, zorder=5)
    a, b = data[groups[0]], data[groups[1]]
    p = float(stats.ttest_ind(a, b, equal_var=equal_var).pvalue)
    top = max(float(np.concatenate([a, b]).max()),
              max(m + s for m, s in zip(means, sems)))
    by, drop = top * 1.07, top * 0.03
    ax.plot([0, 0, 1, 1], [by - drop, by, by, by - drop], color='black',
            lw=0.9, clip_on=False, zorder=6)
    ax.text(0.5, by, _sig_stars(p), ha='center', va='bottom', fontsize=TITLE_FS)
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(0, by * 1.28)
    ax.yaxis.set_major_locator(
        mtick.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=TITLE_FS)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.9)
    ax.spines['bottom'].set_linewidth(0.9)
    ax.tick_params(axis='both', direction='out', width=0.9, length=4,
                   labelsize=LABEL_FS)
    return p


def if_validation_figure(xlsx_path, panels, out_stem, *, panel_w=2.5,
                         panel_h=3.8):
    """panels: list of (sheet, ylabel). One Prism bar panel each, in a row.
    Saves <out_stem>.png/.svg; returns the per-panel p-values. panel_h is
    generous so long rotated y-axis titles fit within the axis height."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(panel_w * n, panel_h),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    ps = [draw_if_barplot(ax, xlsx_path, sheet, ylabel)
          for ax, (sheet, ylabel) in zip(axes, panels)]
    fig.savefig(f'{out_stem}.png', dpi=400)
    fig.savefig(f'{out_stem}.svg')
    plt.close(fig)
    return ps
