"""Microglia activation figure -- Panel A (GSEA dotplot) + Panel B
(DE dotplot) + gene cards (right column) + Panel D/E (cell-state) +
LR chord row.

Mirrors the vascular figure architecture: rows = curated immune pathways
(A) or genes (B) grouped into the same 4 mechanistic bands (immune
activation / cytokine production / inflammatory response / myeloid
effector); columns = same 11 non-neuronal niche cell types in ABCA
numeric order. Panel D = combined Microglia+BAM UCell density; Panel E =
top-10 neuron subclasses by environment-score; chord row = 3 immune LR
themes.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import scipy.sparse as sps_sparse
from scipy.spatial import cKDTree
from scipy.stats import rankdata, ttest_ind, gaussian_kde

import matplotlib as mpl
import matplotlib.pyplot as plt
from pycirclize import Circos

warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures/microglia'
os.makedirs(out_dir, exist_ok=True)

cmap = plt.get_cmap('seismic')

# --- helpers -----------------------------------------------------------------

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

# =============================================================================
# Panel A: GSEA pathway selection (4 bands, 12 pathways)
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
    'GOBP_ADAPTIVE_IMMUNE_RESPONSE':                          'adaptive immune response',
    'GOBP_ACTIVATION_OF_IMMUNE_RESPONSE':                     'activation of immune response',
    'GOBP_REGULATION_OF_INNATE_IMMUNE_RESPONSE':              'innate immune response',
    'GOBP_CYTOKINE_PRODUCTION':                               'cytokine production',
    'GOBP_INTERLEUKIN_1_PRODUCTION':                          'IL-1 production',
    'GOBP_TUMOR_NECROSIS_FACTOR_SUPERFAMILY_CYTOKINE_PRODUCTION':
                                                              'TNF production',
    'GOBP_INFLAMMATORY_RESPONSE':                             'inflammatory response',
    'GOBP_ACUTE_INFLAMMATORY_RESPONSE':                       'acute inflammatory response',
    'GOBP_NEUROINFLAMMATORY_RESPONSE':                        'neuroinflammatory response',
    'GOBP_MACROPHAGE_ACTIVATION':                             'macrophage activation',
    'GOBP_MYELOID_LEUKOCYTE_MIGRATION':                       'myeloid leukocyte migration',
    'GOBP_LEUKOCYTE_CHEMOTAXIS':                              'leukocyte chemotaxis',
}

# Okabe-Ito (colorblind-safe) -- shared between Panel A and Panel B.
BAND_COLORS = {
    'Immune activation':     '#0072B2',
    'Cytokine production':   '#E69F00',
    'Inflammatory response': '#009E73',
    'Myeloid effector':      '#CC79A7',
}

# Condition colors (Panel D/E)
COND_COLORS = {
    'CTRL': '#7209b7', 'PREG': '#b5179e', 'POSTPART': '#f72585',
}

# =============================================================================
# Panel B: curated gene bands (same 4-band structure as Panel A)
# =============================================================================
GENE_BANDS = [
    # Sensing + inflammasome + complement (innate sensors)
    ('Immune activation', [
        'Trem2', 'Nlrp3', 'Casp1', 'C3', 'Ifngr1', 'Stat1',
    ]),
    # Produced cytokines (ligand side)
    ('Cytokine production', [
        'Il18', 'Il10', 'Il34', 'Tnf', 'Tgfb2', 'Csf1',
    ]),
    # Receptors + downstream signaling (receiver side)
    ('Inflammatory response', [
        'Il6st', 'Il4ra', 'Il10rb', 'Tnfrsf1b', 'Tgfbr1', 'Stat3', 'Jak2',
    ]),
    # Microglia effector + chemotaxis + DAM signaling
    ('Myeloid effector', [
        'Ccl2', 'Cxcl12', 'Cx3cr1', 'P2ry12', 'Tmem119', 'Pik3cd', 'Mef2c',
    ]),
]
ordered_genes = [g for _, gs in GENE_BANDS for g in gs]
gene_band = {g: b for b, gs in GENE_BANDS for g in gs}

# =============================================================================
# Load GSEA + collapse direction
# =============================================================================
gsea_all = (pl.read_csv(f'{working_dir}/output/gsea/sumrank_gsea_results.csv')
    .filter((pl.col('contrast') == 'PREG_vs_CTRL') & (pl.col('D') >= 2))
    .with_columns(
        pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
            .then(pl.lit('up')).otherwise(pl.lit('down')).alias('direction'),
        pl.max_horizontal(pl.col('nlp_up'), pl.col('nlp_down')).alias('nlp'),
        pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
            .then(pl.col('emp_p_up')).otherwise(pl.col('emp_p_down'))
            .alias('emp_p')))
gsea = gsea_all.filter(pl.col('emp_p') <= 0.05)
print(f'GSEA D>=2: {gsea_all.height}; sig (emp_p<=0.05): {gsea.height}')

real_nes = (pl.read_parquet(f'{working_dir}/output/gsea/perms/real_gsea.parquet')
    .filter(pl.col('contrast') == 'PREG_vs_CTRL'))

# =============================================================================
# Cell-type selection: same data-driven rule as the lipid + vascular figures
# (>=2 sig hits + D>=2) but WITHOUT the NN-only filter. Microglia/immune
# pathways are genuinely engaged by neurons in pregnancy: a sweep of these
# 12 pathways shows neurons hit them UP-dominantly (~71% UP), unlike the
# vascular case where neuron hits were uniformly artifactual DOWN.
# =============================================================================
# Niche allowlist (anatomically important even where pathway-hit count
# is modest) -- mirrors the vascular figure's allowlist scheme.
ct_allowlist = {
    '318 Astro-NT NN', '319 Astro-TE NN', '323 Ependymal NN',
    '326 OPC NN', '327 Oligo NN', '334 Microglia NN', '335 BAM NN',
}

vasc = gsea.filter(pl.col('pathway').is_in(ordered_pathways))
ct_counts = (vasc.group_by('cell_type')
                  .agg(pl.len().alias('n'))
                  .filter(pl.col('n') >= 2)
                  .sort('n', descending=True))
keep_cts = set(ct_counts['cell_type'].to_list()) | ct_allowlist
ct_counts = (gsea.filter(pl.col('pathway').is_in(ordered_pathways) &
                         pl.col('cell_type').is_in(keep_cts))
             .group_by('cell_type').agg(pl.len().alias('n')))
missing = keep_cts - set(ct_counts['cell_type'].to_list())
if missing:
    ct_counts = pl.concat([
        ct_counts,
        pl.DataFrame({'cell_type': sorted(missing),
                      'n': [0] * len(missing)},
                     schema={'cell_type': pl.Utf8, 'n': pl.UInt32})])
vasc = vasc.filter(pl.col('cell_type').is_in(keep_cts))
vasc_all = gsea_all.filter(pl.col('pathway').is_in(ordered_pathways) &
                           pl.col('cell_type').is_in(keep_cts))
print(f'Panel A: sig hits={vasc.height} all hits={vasc_all.height} '
      f'cell_types={len(keep_cts)}')

CORTICAL_GABA = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg', 'Pax6']

def assign_class(ct):
    if 'NN' in ct:   return 'Non-neuronal'
    if 'Glut' in ct: return 'Glutamatergic'
    if any(t in ct for t in CORTICAL_GABA):
        return 'GABAergic\nCortex'
    return 'GABAergic\nSubcortex'

CLASS_ORDER = ['Glutamatergic',
               'GABAergic\nCortex', 'GABAergic\nSubcortex',
               'Non-neuronal']

ct_table = (ct_counts
    .with_columns(
        pl.col('cell_type').map_elements(assign_class,
                                          return_dtype=pl.Utf8).alias('class'),
        pl.col('cell_type').map_elements(numeric_prefix,
                                          return_dtype=pl.Int32).alias('num'))
    .with_columns(
        pl.col('class').replace_strict(
            {c: i for i, c in enumerate(CLASS_ORDER)}).alias('class_rank'))
    .sort(['class_rank', 'num']))
ordered_cts = ct_table['cell_type'].to_list()
ct_class = dict(zip(ct_table['cell_type'].to_list(),
                    ct_table['class'].to_list()))

# =============================================================================
# Panel A matrices
# =============================================================================
n_rows, n_cols = len(ordered_pathways), len(ordered_cts)
nlp_mat = np.full((n_rows, n_cols), np.nan)
nes_mat = np.full((n_rows, n_cols), np.nan)
d_mat = np.zeros((n_rows, n_cols), dtype=int)
sig_mat_a = np.zeros((n_rows, n_cols), dtype=bool)

ri = {p: i for i, p in enumerate(ordered_pathways)}
ci = {c: j for j, c in enumerate(ordered_cts)}
for r in vasc_all.iter_rows(named=True):
    i, j = ri[r['pathway']], ci[r['cell_type']]
    if np.isnan(nlp_mat[i, j]) or r['nlp'] > nlp_mat[i, j]:
        nlp_mat[i, j] = r['nlp']
        d_mat[i, j] = r['D']
        sig_mat_a[i, j] = r['emp_p'] <= 0.05

nes_agg = (real_nes
    .filter(pl.col('pathway').is_in(ordered_pathways) &
            pl.col('cell_type').is_in(keep_cts))
    .group_by(['pathway', 'cell_type'])
    .agg(pl.col('NES').median().alias('NES_med')))
nes_lookup = {(r['pathway'], r['cell_type']): r['NES_med']
              for r in nes_agg.iter_rows(named=True)}
for i, p in enumerate(ordered_pathways):
    for j, ct in enumerate(ordered_cts):
        if not np.isnan(nlp_mat[i, j]):
            v = nes_lookup.get((p, ct))
            if v is not None:
                nes_mat[i, j] = v
print(f'Panel A occupied: {int(np.sum(~np.isnan(nlp_mat)))}/{n_rows * n_cols}; '
      f'sig: {int(sig_mat_a.sum())}; D=3: {int((d_mat == 3).sum())}')

class_spans = spans([ct_class[c] for c in ordered_cts])
band_spans = spans([pathway_band[p] for p in ordered_pathways])

# =============================================================================
# Panel B: load DE data (sumrank for sig + D, per-platform for logFC + pct)
# =============================================================================
de_sr_all = (pl.read_csv(f'{working_dir}/output/de/sumrank_results.csv')
    .filter((pl.col('contrast') == 'PREG_vs_CTRL') &
            (pl.col('D') >= 2) &
            (pl.col('ref_pct_detected') >= 5.0))
    .with_columns(
        pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
            .then(pl.lit('up')).otherwise(pl.lit('down')).alias('direction'),
        pl.max_horizontal(pl.col('nlp_up'), pl.col('nlp_down')).alias('nlp'),
        pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
            .then(pl.col('emp_p_up')).otherwise(pl.col('emp_p_down')).alias('emp_p')))
de_sr = de_sr_all.filter(pl.col('emp_p') <= 0.05)

# Card forests bypass the 5% ref_pct filter to capture real cross-platform
# sig hits in low-expression cells (e.g., Pecam1 in mural cells at ~2%).
_de_sr_card_full = (pl.read_csv(f'{working_dir}/output/de/sumrank_results.csv')
    .filter((pl.col('contrast') == 'PREG_vs_CTRL') & (pl.col('D') >= 2))
    .with_columns(
        pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
            .then(pl.lit('up')).otherwise(pl.lit('down')).alias('direction'),
        pl.max_horizontal(pl.col('nlp_up'), pl.col('nlp_down')).alias('nlp'),
        pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
            .then(pl.col('emp_p_up')).otherwise(pl.col('emp_p_down')).alias('emp_p')))
de_sr_card = _de_sr_card_full.filter(pl.col('emp_p') <= 0.05)
# D lookup independent of significance: any (gene, ct) with D>=2 testability
# gets its D so forced context cells still display a meta diamond.
d_lookup_card = {(r['gene'], r['cell_type']): r['D']
                 for r in _de_sr_card_full.iter_rows(named=True)}
de_sr_panel = de_sr.filter(
    pl.col('cell_type').is_in(keep_cts) &
    pl.col('gene').is_in(ordered_genes))
de_sr_all_panel = de_sr_all.filter(
    pl.col('cell_type').is_in(keep_cts) &
    pl.col('gene').is_in(ordered_genes))
d_lookup_b = {(r['gene'], r['cell_type']): r['D']
              for r in de_sr_all_panel.iter_rows(named=True)}

de_pp = (pl.read_csv(f'{working_dir}/output/de/de_results.csv')
    .filter(pl.col('contrast') == 'PREG_vs_CTRL')
    .select(['gene', 'cell_type', 'dataset', 'logFC', 'SE', 'LCI', 'UCI',
             'AveExpr', 'PValue', 'FDR', 'ref_pct_detected',
             'n_cells_treat', 'n_cells_base']))
de_pp_panel = (de_pp
    .filter(pl.col('cell_type').is_in(keep_cts) &
            pl.col('gene').is_in(ordered_genes) &
            pl.col('logFC').is_not_null())
    .group_by(['gene', 'cell_type'])
    .agg(pl.col('logFC').median().alias('meta_lfc'),
         pl.len().alias('n_pf')))
meta_lookup = {(r['gene'], r['cell_type']):
               (r['meta_lfc'], r['n_pf'])
               for r in de_pp_panel.iter_rows(named=True)}
sig_set = {(r['gene'], r['cell_type'])
           for r in de_sr_panel.iter_rows(named=True)}

n_g = len(ordered_genes)
lfc_mat = np.full((n_g, n_cols), np.nan)
ndet_mat = np.zeros((n_g, n_cols), dtype=int)
sig_mat_b = np.zeros((n_g, n_cols), dtype=bool)
d_mat_b = np.zeros((n_g, n_cols), dtype=int)
for i, g in enumerate(ordered_genes):
    for j, ct in enumerate(ordered_cts):
        lfc, n = meta_lookup.get((g, ct), (np.nan, 0))
        lfc_mat[i, j] = lfc
        ndet_mat[i, j] = n
        sig_mat_b[i, j] = (g, ct) in sig_set
        d_mat_b[i, j] = d_lookup_b.get((g, ct), 0)
gene_band_spans = spans([gene_band[g] for g in ordered_genes])
print(f'Panel B genes={n_g} D>=2 cells={int((d_mat_b >= 2).sum())} '
      f'sig_cells={int(sig_mat_b.sum())}')

# ABCA subclass colors
_cj = pd.read_csv('/home/karbabi/single-cell/ABC/metadata/cells_joined.csv',
                  usecols=['subclass', 'subclass_color']).drop_duplicates()
SUBCLASS_COLORS = {k.replace('_', '/'): v for k, v in
                   dict(zip(_cj['subclass'].str.replace('/', '_'),
                            _cj['subclass_color'])).items()}
SUBCLASS_COLORS['Unlabelled'] = '#d3d3d3'

# =============================================================================
# Gene cards: load spatial + per-cell expression for the 5 anchor genes
# =============================================================================
CARD_GENES = ['Trem2', 'Nlrp3', 'Tgfb2', 'Stat3', 'Cx3cr1', 'Tmem119']
PLATFORMS = ['slidetags', 'merfish', 'xenium']
PLATFORM_COLORS = {
    'slidetags': '#1C6CC6',
    'merfish':   '#E8A628',
    'xenium':    '#2F7F2E',
}
D_COLORS = {2: '#888888', 3: '#000000'}
PLATFORM_LABELS = {'xenium': 'Xenium', 'merfish': 'MERFISH',
                   'slidetags': 'Slide-tags'}

sp_coords = {}
sp_expr = {}
sp_in = {}
pct_nonzero_lists = {(g, ct): [] for g in ordered_genes for ct in ordered_cts}
for ds in PLATFORMS:
    a = sc.read_h5ad(
        f'{working_dir}/output/{ds}/03_adata_query_{ds}.h5ad', backed='r')
    var_names = (a.var['gene_symbol'].astype(str).to_list()
                 if 'gene_symbol' in a.var.columns else list(a.var_names))
    keep_mask = (a.obs['sample'].values != 'CTRL_3') \
        if ds == 'xenium' else np.ones(a.n_obs, dtype=bool)

    obs_subset = a.obs.loc[keep_mask, ['x_ffd', 'y_ffd',
                                       'condition', 'subclass']]
    xv = obs_subset['x_ffd'].values.astype(float)
    yv = obs_subset['y_ffd'].values.astype(float)
    fov_half = max(np.ptp(xv), np.ptp(yv)) / 2 * 1.05
    cell_types_kept = obs_subset['subclass'].values.astype(str)
    sp_coords[ds] = dict(
        x=xv, y=yv,
        cond=obs_subset['condition'].values.astype(str),
        cell_type=cell_types_kept,
        midline=(xv.min() + xv.max()) / 2,
        fov_cx=(xv.min() + xv.max()) / 2,
        fov_cy=(yv.min() + yv.max()) / 2,
        fov_half=fov_half)
    ct_masks = {ct: (cell_types_kept == ct) for ct in ordered_cts}

    sp_expr[ds] = {}
    sp_in[ds] = {}
    name_to_idx = {n: i for i, n in enumerate(var_names)}
    print(f'  {ds}: computing per-cell totals (chunked)...')
    n_obs = a.n_obs
    totals = np.zeros(n_obs, dtype=np.float64)
    chunk = 20000
    for start in range(0, n_obs, chunk):
        end = min(start + chunk, n_obs)
        block = a.X[start:end]
        if hasattr(block, 'toarray'):
            block = block.toarray()
        totals[start:end] = np.asarray(block).sum(axis=1)
    totals = np.where(totals > 0, totals, 1.0)
    for g in CARD_GENES:
        if g in name_to_idx:
            col = a.X[:, name_to_idx[g]]
            arr = np.asarray(col.toarray()).flatten() \
                if hasattr(col, 'toarray') else np.asarray(col).flatten()
            norm = arr / totals * 1e4
            sp_expr[ds][g] = np.log2(norm[keep_mask] + 1)
            sp_in[ds][g] = True
        else:
            sp_in[ds][g] = False

    for g in ordered_genes:
        if g not in name_to_idx:
            continue
        col = a.X[:, name_to_idx[g]]
        arr = np.asarray(col.toarray()).flatten() \
            if hasattr(col, 'toarray') else np.asarray(col).flatten()
        arr = arr[keep_mask]
        nonzero = arr > 0
        for ct, ct_mask in ct_masks.items():
            n_total = int(ct_mask.sum())
            if n_total == 0:
                continue
            pct = float(nonzero[ct_mask].sum()) / n_total * 100.0
            pct_nonzero_lists[(g, ct)].append(pct)

    a.file.close()
    del a
    print(f'  {ds}: loaded coords ({sum(keep_mask):,} cells) '
          f'+ genes {[g for g in CARD_GENES if sp_in[ds][g]]}')

# Replace approximate ref_pct_detected with actual % nonzero from h5ad
pct_mat = np.full((n_g, n_cols), np.nan)
for i, g in enumerate(ordered_genes):
    for j, ct in enumerate(ordered_cts):
        pcts = pct_nonzero_lists.get((g, ct), [])
        if pcts:
            pct_mat[i, j] = float(np.median(pcts))

de_pp_full = de_pp

def short_ct(ct):
    parts = ct.split(' ')
    if len(parts) <= 3: return ct
    return f'{parts[0]} {parts[1]} {parts[-1]}'

def class_rank(ct):
    if 'Glut' in ct: return 0
    if 'NN' in ct:   return 2
    return 1

def pctl_range(arr, lo=5, hi=95):
    if len(arr) == 0: return 0.0, 1.0
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

def stars_for(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

def build_card(gene, ctx_cells=()):
    sr_hits = de_sr_card.filter((pl.col('gene') == gene) &
                                pl.col('cell_type').is_in(keep_cts))
    cts_in = list(sr_hits['cell_type'].to_list())
    for ct in ctx_cells:
        if ct in keep_cts and ct not in cts_in:
            cts_in.append(ct)

    rows = []
    for ct in cts_in:
        plat_d = {p: dict(lfc=np.nan, lci=np.nan, uci=np.nan)
                  for p in PLATFORMS}
        ppr = de_pp_full.filter(
            (pl.col('gene') == gene) & (pl.col('cell_type') == ct))
        for r in ppr.iter_rows(named=True):
            if r['dataset'] in plat_d:
                plat_d[r['dataset']] = dict(
                    lfc=r['logFC'], lci=r['LCI'], uci=r['UCI'])
        valid = [plat_d[p]['lfc'] for p in PLATFORMS
                 if not np.isnan(plat_d[p]['lfc'])]
        meta_lfc = float(np.median(valid)) if valid else np.nan

        srr = sr_hits.filter(pl.col('cell_type') == ct)
        if srr.height:
            sr = srr.to_dicts()[0]
            ep, D = sr['emp_p'], int(sr['D'])
            stars = stars_for(ep)
        else:
            # Forced context cell: pull D from unfiltered sumrank so a
            # D>=2 testable but non-sig row still shows the meta diamond.
            ep = np.nan
            D = int(d_lookup_card.get((gene, ct), 0))
            stars = ''
        rows.append(dict(cell_type=ct, plat=plat_d, D=D,
                         meta_lfc=meta_lfc, emp_p=ep, stars=stars))

    rows.sort(key=lambda r: (class_rank(r['cell_type']),
                             numeric_prefix(r['cell_type'])))
    cts = [r['cell_type'] for r in rows]

    sp_pair = [p for p in ('xenium', 'merfish', 'slidetags')
               if sp_in[p].get(gene, False)]

    sp_data, sp_vranges = [], []
    ct_set = set(cts)
    for ds in sp_pair:
        c = sp_coords[ds]
        expr = sp_expr[ds][gene]
        ct_mask = np.isin(c['cell_type'], list(ct_set))
        cond_mask = ((c['cond'] == 'CTRL') & (c['x'] < c['midline'])) | \
                    ((c['cond'] == 'PREG') & (c['x'] >= c['midline']))
        mask = ct_mask & cond_mask
        sd = dict(x=c['x'][mask], y=c['y'][mask], expr=expr[mask],
                  midline=c['midline'], fov_cx=c['fov_cx'],
                  fov_cy=c['fov_cy'], fov_half=c['fov_half'])
        sp_data.append(sd)
        nz = sd['expr'][sd['expr'] > 0]
        sp_vranges.append(pctl_range(nz))

    return dict(cell_types=cts, rows=rows, sp_pair=sp_pair,
                sp_data=sp_data, sp_vranges=sp_vranges)

# Context cells: force-include the microglial cell on every card so the
# microglia signal is always visible. Add BAM where relevant.
card_ctx = {g: [] for g in CARD_GENES}
card_ctx['Trem2']   = ['334 Microglia NN', '335 BAM NN']
card_ctx['Nlrp3']   = ['334 Microglia NN']
card_ctx['Tgfb2']   = ['334 Microglia NN']
card_ctx['Stat3']   = ['334 Microglia NN']
card_ctx['Cx3cr1']  = ['334 Microglia NN']
card_ctx['Tmem119'] = ['334 Microglia NN']
cards = {g: build_card(g, card_ctx[g]) for g in CARD_GENES}
MAX_SP_N = max(len(c['sp_pair']) for c in cards.values())
for g, c in cards.items():
    print(f'card {g}: {len(c["cell_types"])} cells, sp_pair={c["sp_pair"]}')

# =============================================================================
# Cell-state data: UCell per slidetags niche cell + per-neuron env_score
# (slidetags only -- sweep showed xenium effect sizes are 2-3x smaller and
# do not survive per-sample t-tests; slidetags has all 26 vascular genes
# present, full transcriptome (32k), 3v3 CTRL/PREG samples)
# =============================================================================
CS_PLATFORM = 'slidetags'
# Microglia + BAM (perivascular macrophage): the two professional immune
# cells of the niche. Combined focal cells for both Panel D density and
# Panel E neuron-environment kNN.
CS_FOCAL_CELLS = ['334 Microglia NN', '335 BAM NN']
CS_KNN_K = 10
CS_CHUNK_SIZE = 20000
# 26 immune genes = flatten Panel B GENE_BANDS
CS_VASCULAR_GENES = ordered_genes  # defined above (name kept for ports)

cs_dir = f'{working_dir}/output/cell_state'
os.makedirs(cs_dir, exist_ok=True)
CS_UCELL_PQ = f'{cs_dir}/ucell_scores_microglia_slidetags.parquet'
CS_ENV_PQ   = f'{cs_dir}/neuron_env_microglia_slidetags.parquet'


def _ucell_chunk(X_chunk, gene_set_idx, n_total):
    n_set = len(gene_set_idx)
    dense = X_chunk.toarray() if sps_sparse.issparse(X_chunk) \
        else np.asarray(X_chunk)
    # UCell convention: rank 1 = HIGHEST expression -> rank negative values
    ranks = rankdata(-dense, axis=1, method='average').astype(np.float32)
    u_sum = ranks[:, gene_set_idx].sum(axis=1)
    u_stat = u_sum - n_set * (n_set + 1) / 2.0
    denom = float(n_set * (n_total - n_set))
    return (1.0 - u_stat / denom).astype(np.float32)


def load_cell_state():
    """Load or compute slidetags UCell + per-neuron env_score parquets."""
    if os.path.exists(CS_UCELL_PQ) and os.path.exists(CS_ENV_PQ):
        print(f'  [cell-state] cached -> {CS_UCELL_PQ}')
        return pl.read_parquet(CS_UCELL_PQ), pl.read_parquet(CS_ENV_PQ)

    a = sc.read_h5ad(
        f'{working_dir}/output/{CS_PLATFORM}/03_adata_query_{CS_PLATFORM}.h5ad',
        backed='r')
    if 'gene_symbol' in a.var.columns:
        var_names = a.var['gene_symbol'].astype(str).to_list()
    else:
        var_names = list(a.var_names)
    name_to_idx = {n: i for i, n in enumerate(var_names)}
    gene_idx = np.array([name_to_idx[g] for g in CS_VASCULAR_GENES
                         if g in name_to_idx], dtype=np.int64)
    print(f'  [cell-state] {len(gene_idx)}/{len(CS_VASCULAR_GENES)} '
          f'vascular genes present in {CS_PLATFORM} (N={len(var_names)})')

    obs = a.obs
    keep_niche = obs['subclass'].isin(ordered_cts).values \
        & (obs['condition'].values != 'POSTPART')
    cell_idx = np.where(keep_niche)[0]
    n_keep = len(cell_idx)
    print(f'  [cell-state] scoring {n_keep:,} niche cells')

    ucell = np.full(n_keep, np.nan, dtype=np.float32)
    n_total = len(var_names)
    for start in range(0, n_keep, CS_CHUNK_SIZE):
        end = min(start + CS_CHUNK_SIZE, n_keep)
        ucell[start:end] = _ucell_chunk(
            a.X[cell_idx[start:end], :], gene_idx, n_total)

    obs_keep = obs.iloc[cell_idx]
    ucell_df = pl.DataFrame({
        'cell_id':   obs_keep.index.astype(str).to_numpy(),
        'subclass':  obs_keep['subclass'].astype(str).to_numpy(),
        'condition': obs_keep['condition'].astype(str).to_numpy(),
        'sample':    obs_keep['sample'].astype(str).to_numpy(),
        'x_affine':  obs_keep['x_affine'].astype(np.float32).to_numpy(),
        'y_affine':  obs_keep['y_affine'].astype(np.float32).to_numpy(),
        'ucell':     ucell,
    })

    # Neuron env_score: kNN over Endo + Peri only (the focal cells)
    obs_all = a.obs.copy()
    obs_all = obs_all[obs_all['condition'] != 'POSTPART']
    obs_all['is_neuron'] = obs_all['subclass'].apply(
        lambda s: ('Glut' in s) or ('Gaba' in s) or ('IMN' in s))
    ucell_pd = ucell_df.select(['cell_id', 'ucell']).to_pandas()
    ucell_pd['cell_id'] = ucell_pd['cell_id'].astype(str)

    rows_out = []
    for sample in sorted(obs_all['sample'].unique()):
        smp = obs_all[obs_all['sample'] == sample]
        focal_smp = smp[smp['subclass'].isin(CS_FOCAL_CELLS)].copy()
        focal_smp['cell_id_str'] = focal_smp.index.astype(str)
        focal_smp = focal_smp.merge(ucell_pd, left_on='cell_id_str',
                                    right_on='cell_id', how='left')
        focal_smp = focal_smp.dropna(subset=['ucell'])
        if len(focal_smp) < CS_KNN_K:
            print(f'  [cell-state] sample {sample}: '
                  f'only {len(focal_smp)} focal cells -- skip')
            continue
        tree = cKDTree(focal_smp[['x_affine', 'y_affine']].to_numpy(
            dtype=np.float64))
        focal_ucell = focal_smp['ucell'].to_numpy(dtype=np.float32)
        neurons = smp[smp['is_neuron']]
        if len(neurons) == 0:
            continue
        dists, idxs = tree.query(neurons[['x_affine', 'y_affine']].to_numpy(
            dtype=np.float64), k=CS_KNN_K, workers=-1)
        env_score = focal_ucell[idxs].mean(axis=1)
        rows_out.append(pd.DataFrame({
            'cell_id':   neurons.index.astype(str).to_numpy(),
            'subclass':  neurons['subclass'].astype(str).to_numpy(),
            'condition': neurons['condition'].astype(str).to_numpy(),
            'sample':    sample,
            'env_score': env_score.astype(np.float32),
            'dist_kth':  dists[:, -1].astype(np.float32),
        }))
        print(f'  [cell-state] sample {sample}: {len(neurons):,} neurons')

    a.file.close()

    env_pd = pd.concat(rows_out, ignore_index=True)
    env_pd.to_parquet(CS_ENV_PQ, index=False)
    ucell_df.write_parquet(CS_UCELL_PQ)
    env_df = pl.read_parquet(CS_ENV_PQ)
    print(f'  [cell-state] wrote {CS_UCELL_PQ}, {CS_ENV_PQ}')
    return ucell_df, env_df


cs_ucell, cs_env = load_cell_state()
print(f'cell-state: ucell rows={cs_ucell.height:,}, env rows={cs_env.height:,}')


# Top neuron subclasses (used for both Panel E and the chord-row neuron pool)
def _compute_top_neurons(env_df, n_top=10, min_per_sample=20):
    rows = []
    for sub in sorted(set(env_df['subclass'].to_list())):
        sdf = env_df.filter(pl.col('subclass') == sub)
        c_smp = (sdf.filter(pl.col('condition') == 'CTRL')
                 .group_by('sample').agg(
                     pl.col('env_score').mean().alias('m'),
                     pl.len().alias('n')))
        p_smp = (sdf.filter(pl.col('condition') == 'PREG')
                 .group_by('sample').agg(
                     pl.col('env_score').mean().alias('m'),
                     pl.len().alias('n')))
        c_smp = c_smp.filter(pl.col('n') >= min_per_sample)
        p_smp = p_smp.filter(pl.col('n') >= min_per_sample)
        if c_smp.height < 2 or p_smp.height < 2:
            continue
        cv = c_smp['m'].to_numpy()
        pv = p_smp['m'].to_numpy()
        rows.append(dict(subclass=sub,
                          mean_ctrl=float(cv.mean()),
                          mean_preg=float(pv.mean()),
                          mean_diff=float(pv.mean() - cv.mean())))
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    return (df.assign(abs_d=df['mean_diff'].abs())
              .sort_values('abs_d', ascending=False).head(n_top)
              .assign(num=lambda x: x['subclass'].map(numeric_prefix))
              .sort_values('num').reset_index(drop=True))


top_neuron_df = _compute_top_neurons(cs_env, n_top=10)
top_neurons = top_neuron_df['subclass'].tolist() if len(top_neuron_df) else []
print(f'Top neuron subclasses ({len(top_neurons)}): {top_neurons}')

# =============================================================================
# Chord-plot themes (LIANA L-R, niche + top-10 affected neurons)
# =============================================================================
CHORD_THEMES = ['Phagocytic', 'CSF1_axis', 'Antiinflam']
CHORD_THEME_LIGANDS = {
    # Phagocytic clearance (efferocytosis, complement-tagged debris,
    # eat-me + don't-eat-me brake)
    'Phagocytic': ['Apoe', 'Gas6', 'Adam10', 'C1qb', 'C4b'],
    # CSF1 / IL-34 myeloid maintenance + signaling
    'CSF1_axis':  ['Csf1', 'Il34', 'Csf1_Il34'],
    # Neuron->microglia restraint (anti-inflammatory)
    'Antiinflam': ['Cx3cl1', 'Tgfb1', 'Tgfb2', 'Cd47'],
}
CHORD_THEME_TITLES = {
    'Phagocytic': 'Phagocytic clearance',
    'CSF1_axis':  'CSF1 / IL-34',
    'Antiinflam': 'Cx3cl1 / TGF-β / Cd47',
}

# Canonical LR pairs use the EXACT names present in LIANA (which uses
# underscore-concatenated complexes for heterodimers and AnnotationLR
# complex ligands -- e.g. Csf1_Il34, Tgfbr1_Tgfbr2). Pairs were chosen
# from the cross-platform-tested microglia-involving audit.
CANONICAL_LR_PAIRS = set()
# Phagocytic clearance theme
for _r in ['Trem2', 'Lrp1']:
    CANONICAL_LR_PAIRS.add(('Apoe', _r))
CANONICAL_LR_PAIRS.add(('Gas6', 'Axl'))
CANONICAL_LR_PAIRS.add(('Gas6', 'Mertk'))
CANONICAL_LR_PAIRS.add(('Adam10', 'Trem2'))
CANONICAL_LR_PAIRS.add(('Adam10', 'Axl'))
CANONICAL_LR_PAIRS.add(('C1qb', 'Lrp1'))
CANONICAL_LR_PAIRS.add(('C4b', 'Nrp1'))
CANONICAL_LR_PAIRS.add(('C4b', 'C3ar1'))
# CSF1 / IL-34 myeloid axis (canonical Csf1r is recorded as the
# Csf1_Il34 complex ligand in LIANA)
CANONICAL_LR_PAIRS.add(('Csf1_Il34', 'Csf1r'))
CANONICAL_LR_PAIRS.add(('Csf1', 'Sirpa'))
CANONICAL_LR_PAIRS.add(('Il34', 'Ptprz1'))
# Cx3cl1 / TGF-beta / Cd47 anti-inflammatory restraint
CANONICAL_LR_PAIRS.add(('Cx3cl1', 'Cx3cr1'))
CANONICAL_LR_PAIRS.add(('Tgfb1', 'Tgfbr1_Tgfbr2'))
CANONICAL_LR_PAIRS.add(('Tgfb1', 'Eng'))
CANONICAL_LR_PAIRS.add(('Tgfb1', 'Itgb1'))
CANONICAL_LR_PAIRS.add(('Tgfb1', 'Itgb5'))
CANONICAL_LR_PAIRS.add(('Cd47', 'Sirpa'))
CANONICAL_LR_PAIRS.add(('Sirpa', 'Cd47'))

_chord_ligs = list({p[0] for p in CANONICAL_LR_PAIRS})
_chord_recs = list({p[1] for p in CANONICAL_LR_PAIRS})

# Cell pool for the chord: 11 niche cells + top-10 affected neurons
chord_cell_set = list(dict.fromkeys(list(ordered_cts) + list(top_neurons)))
print(f'Chord cell pool: {len(chord_cell_set)} '
      f'({len(ordered_cts)} niche + {len(top_neurons)} neurons)')

_liana = (pl.scan_csv(f'{working_dir}/output/liana/inflow_diff.csv')
    .filter(pl.col('contrast') == 'PREG_vs_CTRL')
    .filter(pl.col('ligand_complex').is_in(_chord_ligs) &
            pl.col('receptor_complex').is_in(_chord_recs))
    .filter(pl.col('source').is_in(chord_cell_set) &
            pl.col('target').is_in(chord_cell_set))
    .collect())
_liana = _liana.filter(
    pl.struct(['ligand_complex', 'receptor_complex']).map_elements(
        lambda s: (s['ligand_complex'], s['receptor_complex'])
                  in CANONICAL_LR_PAIRS,
        return_dtype=pl.Boolean))

_keys = ['source', 'target', 'ligand_complex', 'receptor_complex']
_wide = (_liana.group_by(_keys + ['dataset'])
        .agg(pl.col('lr_mean_diff').first())
        .pivot(on='dataset', index=_keys, values='lr_mean_diff')
        .rename({'slidetags': 'diff_st', 'xenium': 'diff_xn'}))

_nonz = _liana.filter(pl.col('lr_mean_diff') != 0)
_med_xn = float(_nonz.filter(pl.col('dataset') == 'xenium')['lr_mean_diff']
                .abs().median())
_med_st = float(_nonz.filter(pl.col('dataset') == 'slidetags')['lr_mean_diff']
                .abs().median())

# cross-platform: both nonzero, same sign, magnitude >= per-platform median
_xp = (_wide.filter(pl.col('diff_st').is_not_null()
                    & pl.col('diff_xn').is_not_null()
                    & (pl.col('diff_st') != 0)
                    & (pl.col('diff_xn') != 0)
                    & (pl.col('diff_st').sign() == pl.col('diff_xn').sign())
                    & (pl.col('diff_st').abs() >= _med_st)
                    & (pl.col('diff_xn').abs() >= _med_xn))
      .with_columns(((pl.col('diff_st') + pl.col('diff_xn')) / 2)
                    .alias('meta_diff'))
      .select(['source', 'target', 'ligand_complex', 'receptor_complex',
               'meta_diff']))

# fallback for slidetags-only pairs (top-10% |diff|)
_xp_pairs = (_wide.filter(pl.col('diff_st').is_not_null()
                          & pl.col('diff_xn').is_not_null())
             .select(['ligand_complex', 'receptor_complex']).unique())
_xp_pair_set = {(r['ligand_complex'], r['receptor_complex'])
                for r in _xp_pairs.iter_rows(named=True)}
_st_only_pairs = CANONICAL_LR_PAIRS - _xp_pair_set

_st = (_liana.filter((pl.col('dataset') == 'slidetags')
                      & (pl.col('lr_mean_diff') != 0))
       .filter(pl.struct(['ligand_complex', 'receptor_complex']).map_elements(
           lambda s: (s['ligand_complex'], s['receptor_complex'])
                     in _st_only_pairs,
           return_dtype=pl.Boolean)))
_st_thr = (float(_st['lr_mean_diff'].abs().quantile(0.90))
           if _st.height else 0.0)
_st = (_st.filter(pl.col('lr_mean_diff').abs() >= _st_thr)
       .rename({'lr_mean_diff': 'meta_diff'})
       .select(['source', 'target', 'ligand_complex', 'receptor_complex',
                'meta_diff']))

chord_edges = pl.concat([_xp, _st]).with_columns(
    (pl.col('source') == pl.col('target')).alias('is_self'))
print(f'Chord curated edges: {chord_edges.height} '
      f'(UP={chord_edges.filter(pl.col("meta_diff") > 0).height}, '
      f'DOWN={chord_edges.filter(pl.col("meta_diff") < 0).height})')

# Drop self-edges (autocrine; clutters chord arcs); keep cell-cell only
chord_edges_intercell = chord_edges.filter(~pl.col('is_self'))


def theme_chord_edges(theme):
    return (chord_edges_intercell
            .filter(pl.col('ligand_complex')
                    .is_in(CHORD_THEME_LIGANDS[theme]))
            .group_by(['source', 'target'])
            .agg(pl.col('meta_diff').sum().alias('signed_sum'),
                 pl.col('meta_diff').abs().sum().alias('mag'),
                 pl.len().alias('n_lr')))


# Chord colors / class helpers
CHORD_COLOR_UP   = '#b2182b'
CHORD_COLOR_DOWN = '#2166ac'
CHORD_CLASS_COLORS = {
    'NN':   '#7570b3',
    'Glut': '#d95f02',
    'GABA': '#1b9e77',
}
CHORD_CLASS_ORDER = ['NN', 'Glut', 'GABA']
CHORD_CLASS_LABELS = {'NN': 'Non-neuronal',
                      'Glut': 'Glutamatergic',
                      'GABA': 'GABAergic'}


def chord_class(ct):
    if 'NN' in ct:   return 'NN'
    if 'Glut' in ct: return 'Glut'
    return 'GABA'


def chord_num_label(ct):
    m = re.match(r'^(\d+)', ct)
    return m.group(1) if m else ct


# Sectors grouped by class so chord arcs cluster within class
chord_cells_ordered = sorted(
    chord_cell_set,
    key=lambda c: (CHORD_CLASS_ORDER.index(chord_class(c)),
                   numeric_prefix(c)))

# =============================================================================
# Layout
# =============================================================================
NES_VMAX = quant_vmax(nes_mat)
LFC_VMAX = quant_vmax(lfc_mat[d_mat_b >= 2])
print(f'cmap vmax: NES={NES_VMAX:.2f}  logFC={LFC_VMAX:.2f}')

NLP_MIN, NLP_MAX = 1.30, 5.0
SIZE_MIN, SIZE_MAX = 16.0, 80.0
SIZE_MAX_A = 140.0
SIG_DOT_SIZE = 4.0
norm_nes = mpl.colors.Normalize(vmin=-NES_VMAX, vmax=NES_VMAX)
norm_lfc = mpl.colors.Normalize(vmin=-LFC_VMAX, vmax=LFC_VMAX)

def nlp_to_size(x):
    x = np.clip(x, NLP_MIN, NLP_MAX)
    f = (x - NLP_MIN) / (NLP_MAX - NLP_MIN)
    return SIZE_MIN + f * (SIZE_MAX_A - SIZE_MIN)

def pct_to_size(p):
    p = float(np.clip(p, 0, 100))
    return SIZE_MIN + (p / 100.0) * (SIZE_MAX - SIZE_MIN)

# Geometry mirrors lipid figure: legend in LEFT fig-pad region (x=0.45..1.30),
# y-tick labels in LABEL_MARGIN region (x=1.15..2.70), axes start at x=2.70.
LEFT_FIG_PAD_IN = 1.15
LABEL_MARGIN_IN = 1.55
ax_left_in = LEFT_FIG_PAD_IN + LABEL_MARGIN_IN
ax_w_in = 0.20 * n_cols
PATH_PITCH = 0.21
GENE_PITCH = 0.155
ax_h_a_in = PATH_PITCH * n_rows
ax_h_b_in = GENE_PITCH * n_g

leg_left_in = 0.45
leg_w_in = 0.85

ANNO_W_IN, ANNO_GAP_IN = 0.07, 0.03
COL_ANNO_H_IN = ANNO_W_IN
COL_ANNO_GAP_IN = 0.02

top_margin_in = 0.85
# bot_margin_in just needs to fit Panel B's rotated x-tick labels.
# Reduced from 2.15 -> brings the cell-state row up by ~1.2".
bot_margin_in = 0.85
gap_in = 0.50          # Panel A col-anno strip lives here

# Cell-state row (Panel D + E) sits just below Panel B's col-anno strip.
# Compact + close to existing panels. Panel E (dotplot) gets a smaller
# height inside this row and is vertically centered on Panel D (density).
CS_ROW_H_IN   = 0.82
CS_ROW_GAP_IN = 0.45   # large gap so Panel B's rotated labels do not overlap D/E
CS_ROW_BOT_PAD_IN = 0.28  # axis-label margin below the cell-state row
CS_E_H_IN     = 0.50      # Panel E height (< CS_ROW_H_IN)

# Gene-card column (right of Panel A/B)
N_CARDS = len(CARD_GENES)
CARD_GAP_IN = 0.10
CARD_TITLE_H_IN = 0.30
ab_vspan_in = ax_h_b_in + gap_in + ax_h_a_in
content_h_in = (ab_vspan_in
                - (N_CARDS - 1) * (CARD_TITLE_H_IN + CARD_GAP_IN)) / N_CARDS
SP_W_IN = content_h_in
SP_GAP_IN = 0.05
SP_FOREST_GAP_IN = 0.10
FOREST_W_IN = 0.75
FOREST_LABEL_GAP_IN = 0.05
LABEL_W_IN = 0.95
CARD_W_IN = (SP_W_IN + SP_GAP_IN + SP_W_IN + SP_FOREST_GAP_IN
             + FOREST_W_IN + FOREST_LABEL_GAP_IN + LABEL_W_IN)
CARD_W_MAX_IN = CARD_W_IN + max(0, MAX_SP_N - 2) * (SP_W_IN + SP_GAP_IN)
CARD_FOREST_ROW_H = 0.13
CARD_FOREST_MIN_H = 0.42
cards_left_in = ax_left_in + ax_w_in + ANNO_GAP_IN + ANNO_W_IN + 0.35

# Forest-legend column (right of cards)
FLEG_GAP_IN = 0.12
FLEG_EXTRA_RIGHT_IN = 0.20
FLEG_COL_W_IN = 1.10

# Chord row constants (below cell-state row). Span from leftmost dotplot
# (ax_left_in) to the right edge of the widest forest box.
CHORD_PANEL_GAP_IN  = 0.30
CHORD_TITLE_H_IN    = 0.30
CHORD_BOT_PAD_IN    = 0.02   # chord sits very close to figure bottom
CS_TO_CHORD_GAP_IN  = 0.72   # larger gap so chord row sits noticeably below D/E
_chord_span_in = (
    (cards_left_in + MAX_SP_N * SP_W_IN + (MAX_SP_N - 1) * SP_GAP_IN
     + SP_FOREST_GAP_IN + FOREST_W_IN) - ax_left_in)
chord_panel_w_in = (_chord_span_in - 2 * CHORD_PANEL_GAP_IN) / 3
chord_panel_h_in = chord_panel_w_in * 0.85
CHORD_ROW_H_IN   = chord_panel_h_in + CHORD_TITLE_H_IN

# Positions (bottom-up)
chord_row_bot_in = CHORD_BOT_PAD_IN
chord_panel_bot_in = chord_row_bot_in
chord_row_top_in = chord_row_bot_in + CHORD_ROW_H_IN
cs_row_axis_bot_in = chord_row_top_in + CS_TO_CHORD_GAP_IN
ax_b_bot_in = (cs_row_axis_bot_in + CS_ROW_H_IN + CS_ROW_GAP_IN
               + bot_margin_in)
ax_b_top_in = ax_b_bot_in + ax_h_b_in
ax_a_bot_in = ax_b_top_in + gap_in
ax_a_top_in = ax_a_bot_in + ax_h_a_in

fig_h = (top_margin_in + ax_h_a_in + gap_in + ax_h_b_in + bot_margin_in
         + CS_ROW_GAP_IN + CS_ROW_H_IN + CS_TO_CHORD_GAP_IN
         + CHORD_ROW_H_IN + CHORD_BOT_PAD_IN)
fig_w = (cards_left_in + CARD_W_MAX_IN + FLEG_GAP_IN + FLEG_EXTRA_RIGHT_IN
         + FLEG_COL_W_IN + 0.10)

fig = plt.figure(figsize=(fig_w, fig_h))
ax_a = fig.add_axes([ax_left_in / fig_w, ax_a_bot_in / fig_h,
                     ax_w_in / fig_w, ax_h_a_in / fig_h])
ax_b = fig.add_axes([ax_left_in / fig_w, ax_b_bot_in / fig_h,
                     ax_w_in / fig_w, ax_h_b_in / fig_h])

# =============================================================================
# Panel A: GSEA dotplot
# =============================================================================
xs, ys, sizes, colors, edges, lws = [], [], [], [], [], []
sig_xs_a, sig_ys_a = [], []
for i in range(n_rows):
    for j in range(n_cols):
        if np.isnan(nlp_mat[i, j]):
            continue
        xs.append(j); ys.append(i)
        sizes.append(nlp_to_size(nlp_mat[i, j]))
        nes = nes_mat[i, j]
        nes_clip = np.clip(nes, -NES_VMAX, NES_VMAX) \
            if not np.isnan(nes) else np.nan
        colors.append(cmap(norm_nes(nes_clip))
                      if not np.isnan(nes) else '#999999')
        edges.append('#555555' if d_mat[i, j] == 3 else 'none')
        lws.append(0.8 if d_mat[i, j] == 3 else 0.0)
        if sig_mat_a[i, j]:
            sig_xs_a.append(j); sig_ys_a.append(i)
ax_a.scatter(xs, ys, s=sizes, c=colors, edgecolors=edges,
             linewidths=lws, zorder=3)
if sig_xs_a:
    ax_a.scatter(sig_xs_a, sig_ys_a, s=SIG_DOT_SIZE, c='white',
                 edgecolors='none', linewidths=0, zorder=4)

for k in range(1, n_cols):
    ax_a.axvline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
for k in range(1, n_rows):
    ax_a.axhline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
for _, _, hi in class_spans[:-1]:
    ax_a.axvline(hi + 0.5, color='#BBBBBB', lw=0.4, zorder=2)
for _, _, hi in band_spans[:-1]:
    ax_a.axhline(hi + 0.5, color='#BBBBBB', lw=0.4, zorder=2)

ax_a.set_xlim(-0.5, n_cols - 0.5)
ax_a.set_ylim(n_rows - 0.5, -0.5)
prefix_labels = [f'{numeric_prefix(ct):03d}' for ct in ordered_cts]
ax_a.tick_params(axis='x', bottom=False, labelbottom=False)
ax_a.set_yticks(range(n_rows))
ax_a.set_yticklabels([PATHWAY_LABELS.get(p, p) for p in ordered_pathways],
                     fontsize=7.5)
ax_a.tick_params(axis='y', length=2, pad=1)
ax_a.set_ylabel('GSEA pathways (meta)', fontsize=9.0, labelpad=8)
for sp in ('top', 'right', 'left', 'bottom'):
    ax_a.spines[sp].set_visible(True)
    ax_a.spines[sp].set_color('black')
    ax_a.spines[sp].set_linewidth(0.9)

trans_x_a = ax_a.get_xaxis_transform()
for cls, lo, hi in class_spans:
    ax_a.text((lo + hi) / 2, 1.04, cls,
              ha='center', va='bottom', fontsize=7.5,
              color='black', transform=trans_x_a, clip_on=False,
              linespacing=1.05)

# =============================================================================
# Panel B: DE dotplot (color = signed logFC, size = % expressed)
# =============================================================================
ax_b.set_facecolor('white')
xs_b, ys_b, sizes_b, colors_b, edges_b, lws_b = [], [], [], [], [], []
sig_xs, sig_ys = [], []
for i in range(n_g):
    for j in range(n_cols):
        if d_mat_b[i, j] < 2:
            continue
        pct = pct_mat[i, j]
        if np.isnan(pct):
            continue
        lfc = lfc_mat[i, j]
        lfc_clip = np.clip(lfc, -LFC_VMAX, LFC_VMAX)
        xs_b.append(j); ys_b.append(i)
        sizes_b.append(pct_to_size(pct))
        colors_b.append(cmap(norm_lfc(lfc_clip))
                        if not np.isnan(lfc) else '#CCCCCC')
        edges_b.append('#555555' if d_mat_b[i, j] == 3 else 'none')
        lws_b.append(0.8 if d_mat_b[i, j] == 3 else 0.0)
        if sig_mat_b[i, j]:
            sig_xs.append(j); sig_ys.append(i)
ax_b.scatter(xs_b, ys_b, s=sizes_b, c=colors_b,
             edgecolors=edges_b, linewidths=lws_b, zorder=3)
if sig_xs:
    ax_b.scatter(sig_xs, sig_ys, s=SIG_DOT_SIZE, c='white',
                 edgecolors='none', linewidths=0, zorder=4)

for k in range(1, n_cols):
    ax_b.axvline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
for k in range(1, n_g):
    ax_b.axhline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
for _, _, hi in class_spans[:-1]:
    ax_b.axvline(hi + 0.5, color='#BBBBBB', lw=0.4, zorder=2)
for _, _, hi in gene_band_spans[:-1]:
    ax_b.axhline(hi + 0.5, color='#BBBBBB', lw=0.4, zorder=2)

ax_b.set_xlim(-0.5, n_cols - 0.5)
ax_b.set_ylim(n_g - 0.5, -0.5)
ax_b.tick_params(axis='x', bottom=False, labelbottom=False)
ax_b.set_yticks(range(n_g))
ax_b.set_yticklabels(ordered_genes, fontsize=7.5, fontstyle='italic')
ax_b.tick_params(axis='y', length=2, pad=1)
ax_b.set_ylabel('DE genes (meta)', fontsize=9.0, labelpad=8)
for sp in ('top', 'right', 'left', 'bottom'):
    ax_b.spines[sp].set_visible(True)
    ax_b.spines[sp].set_color('black')
    ax_b.spines[sp].set_linewidth(0.9)

# =============================================================================
# Column annotation strips (subclass colors). Panel A: prefix labels only.
# Panel B: full subclass names (rotated).
# =============================================================================
def _draw_col_anno(panel_bot_in, ticklabels, ticklabel_fs=7.5):
    bot = panel_bot_in - COL_ANNO_GAP_IN - COL_ANNO_H_IN
    ax = fig.add_axes([ax_left_in / fig_w, bot / fig_h,
                       ax_w_in / fig_w, COL_ANNO_H_IN / fig_h])
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(0, 1)
    for j, ct in enumerate(ordered_cts):
        ax.add_patch(plt.Rectangle(
            (j - 0.5, 0), 1, 1,
            facecolor=SUBCLASS_COLORS.get(ct, '#d3d3d3'),
            edgecolor='none'))
    ax.set_yticks([])
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(ticklabels, rotation=45, ha='right',
                       fontsize=ticklabel_fs)
    ax.tick_params(axis='x', length=3, pad=2, direction='out',
                   bottom=True, top=False,
                   labelbottom=True, labeltop=False)
    ax.tick_params(axis='y', left=False, right=False,
                   labelleft=False, labelright=False)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax

ax_a_col = _draw_col_anno(ax_a_bot_in, prefix_labels)
ax_b_col = _draw_col_anno(ax_b_bot_in, ordered_cts)

# =============================================================================
# Band-color annotation strips (right of A + B)
# =============================================================================
ANNO_X_IN = ax_left_in + ax_w_in + ANNO_GAP_IN

ax_a_anno = fig.add_axes([ANNO_X_IN / fig_w, ax_a_bot_in / fig_h,
                          ANNO_W_IN / fig_w, ax_h_a_in / fig_h])
ax_a_anno.set_xlim(0, 1); ax_a_anno.set_ylim(n_rows - 0.5, -0.5)
ax_a_anno.set_xticks([]); ax_a_anno.set_yticks([])
for s in ax_a_anno.spines.values():
    s.set_visible(False)
for i, p in enumerate(ordered_pathways):
    ax_a_anno.add_patch(plt.Rectangle(
        (0, i - 0.5), 1, 1,
        facecolor=BAND_COLORS[pathway_band[p]], edgecolor='none'))

ax_b_anno = fig.add_axes([ANNO_X_IN / fig_w, ax_b_bot_in / fig_h,
                          ANNO_W_IN / fig_w, ax_h_b_in / fig_h])
ax_b_anno.set_xlim(0, 1); ax_b_anno.set_ylim(n_g - 0.5, -0.5)
ax_b_anno.set_xticks([]); ax_b_anno.set_yticks([])
for s in ax_b_anno.spines.values():
    s.set_visible(False)
for i, g in enumerate(ordered_genes):
    ax_b_anno.add_patch(plt.Rectangle(
        (0, i - 0.5), 1, 1,
        facecolor=BAND_COLORS[gene_band[g]], edgecolor='none'))

# =============================================================================
# Legends: LEG_A (Panel A) / LEG_B (Panel B) / LEG_S (shared) / TLEG (bands).
# Stacked in LEFT fig-pad region, right-justified, vertical thin NES/logFC
# colorbars hugging the right edge.
# =============================================================================
CBAR_W_FIG = 0.005
CBAR_TARGET_X_AXES = 0.95
cbar_x_fig = ((leg_left_in + CBAR_TARGET_X_AXES * leg_w_in) / fig_w
              - CBAR_W_FIG / 2)

def _make_leg_axes(bot_in, h_in):
    ax = fig.add_axes([leg_left_in / fig_w, bot_in / fig_h,
                       leg_w_in / fig_w, h_in / fig_h], zorder=100)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor('white')
    ax.patch.set_facecolor('white')
    ax.patch.set_alpha(1.0)
    ax.patch.set_zorder(99)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax

# Anchor: LEG_A sits at TOP of Panel B (visually aligned with the gap
# between A and B, like the lipid figure).
LEG_A_TOP_IN = ax_b_top_in
LEG_A_H_IN = 8 * GENE_PITCH
LEG_A_BOT_IN = LEG_A_TOP_IN - LEG_A_H_IN

LEG_B_TOP_IN = LEG_A_BOT_IN - 1 * GENE_PITCH
LEG_B_H_IN = 8 * GENE_PITCH
LEG_B_BOT_IN = LEG_B_TOP_IN - LEG_B_H_IN

LEG_S_TOP_IN = LEG_B_BOT_IN - 1 * GENE_PITCH
LEG_S_H_IN = 8 * GENE_PITCH
LEG_S_BOT_IN = LEG_S_TOP_IN - LEG_S_H_IN

# LEG_A: -log10 emp p sizes + NES colorbar (GSEA)
ax_leg_a = _make_leg_axes(LEG_A_BOT_IN, LEG_A_H_IN)
ax_leg_a.text(1.0, 0.96, 'GSEA\n' + r'$-\log_{10}$ emp $p$',
              ha='right', va='top', fontsize=6.8, linespacing=1.0)
for k, lev in enumerate([1.5, 2.5, 4.0]):
    y = 0.72 - k * 0.13
    ax_leg_a.scatter([0.95], [y], s=nlp_to_size(lev),
                     c=['#555555'], edgecolors='none')
    ax_leg_a.text(0.85, y, f'{lev:.1f}',
                  ha='right', va='center', fontsize=6.5)
ax_leg_a.text(1.0, 0.30, 'GSEA\nNES (median)',
              ha='right', va='top', fontsize=6.8, linespacing=1.0)
cbar_ax_a = fig.add_axes([cbar_x_fig,
                          LEG_A_BOT_IN / fig_h + LEG_A_H_IN / fig_h * 0.02,
                          CBAR_W_FIG,
                          LEG_A_H_IN / fig_h * 0.16],
                         zorder=110)
cb_a = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_nes, cmap=cmap),
                    cax=cbar_ax_a, orientation='vertical')
cb_a.set_ticks([-NES_VMAX, 0, NES_VMAX])
cb_a.ax.yaxis.tick_left()
cb_a.ax.tick_params(labelsize=6.0, length=2, pad=1)
cbar_ax_a.set_zorder(110)

# LEG_B: logFC colorbar + Percent expressed sizes (DE)
ax_leg_b = _make_leg_axes(LEG_B_BOT_IN, LEG_B_H_IN)
ax_leg_b.text(1.0, 0.96, 'DE\nlogFC (median)',
              ha='right', va='top', fontsize=6.8, linespacing=1.0)
cbar_ax_b = fig.add_axes([cbar_x_fig,
                          LEG_B_BOT_IN / fig_h + LEG_B_H_IN / fig_h * 0.60,
                          CBAR_W_FIG,
                          LEG_B_H_IN / fig_h * 0.16],
                         zorder=110)
cb_b = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_lfc, cmap=cmap),
                    cax=cbar_ax_b, orientation='vertical')
cb_b.set_ticks([-LFC_VMAX, 0, LFC_VMAX])
cb_b.ax.yaxis.tick_left()
cb_b.ax.tick_params(labelsize=6.0, length=2, pad=1)
cbar_ax_b.set_zorder(110)

ax_leg_b.text(1.0, 0.45, 'DE\nPercent expressed',
              ha='right', va='top', fontsize=6.8, linespacing=1.0)
for k, lev in enumerate([10, 50, 90]):
    y = 0.24 - k * 0.10
    ax_leg_b.scatter([0.95], [y], s=pct_to_size(lev),
                     c=['#777777'], edgecolors='none')
    ax_leg_b.text(0.85, y, f'{lev}%',
                  ha='right', va='center', fontsize=6.5)

# LEG_S: shared D + Significance
ax_leg_s = _make_leg_axes(LEG_S_BOT_IN, LEG_S_H_IN)
ax_leg_s.text(1.0, 0.95, 'Cross-platform',
              ha='right', va='top', fontsize=6.8)
ax_leg_s.scatter([0.95], [0.74], s=50, c=['#bbbbbb'],
                 edgecolors='#555555', linewidths=0.8)
ax_leg_s.text(0.85, 0.74, 'D = 3', ha='right', va='center', fontsize=6.5)
ax_leg_s.scatter([0.95], [0.60], s=50, c=['#bbbbbb'],
                 edgecolors='none', linewidths=0)
ax_leg_s.text(0.85, 0.60, 'D = 2', ha='right', va='center', fontsize=6.5)
ax_leg_s.text(1.0, 0.40, 'Significance',
              ha='right', va='top', fontsize=6.8)
_sig_demo_color = cmap(norm_lfc(LFC_VMAX * 0.7))
ax_leg_s.scatter([0.95], [0.19], s=50, c=[_sig_demo_color],
                 edgecolors='none', linewidths=0)
ax_leg_s.scatter([0.95], [0.19], s=SIG_DOT_SIZE, c='white',
                 edgecolors='none', linewidths=0)
ax_leg_s.text(0.85, 0.19, r'emp $p \leq 0.05$',
              ha='right', va='center', fontsize=6.5)

# TLEG: band-color theme legend
TLEG_TOP_IN = LEG_S_BOT_IN - 1 * GENE_PITCH
TLEG_H_IN = 6 * GENE_PITCH
TLEG_BOT_IN = TLEG_TOP_IN - TLEG_H_IN
ax_tleg = _make_leg_axes(TLEG_BOT_IN, TLEG_H_IN)
band_names = [b for b, _ in GENE_BANDS]
y_top, y_bot = 0.85, 0.30
y_step = (y_top - y_bot) / (len(band_names) - 1)
for i, band in enumerate(band_names):
    y = y_top - i * y_step
    ax_tleg.add_patch(plt.Rectangle((0.85, y - 0.06), 0.10, 0.12,
                                    facecolor=BAND_COLORS[band],
                                    edgecolor='none'))
    ax_tleg.text(0.82, y, band, ha='right', va='center',
                 fontsize=6.8, color='black')

# =============================================================================
# Gene cards (right column)
# =============================================================================
LABEL_FS = 7.5
AXIS_FS = 7.5
TITLE_FS = 8.5
JIT = 0.18

def draw_forest(axf, gd, show_xlabel=False):
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
    if vals:
        lo = float(np.percentile(vals, 2))
        hi = float(np.percentile(vals, 98))
    else:
        lo, hi = -1.0, 1.0
    lo = min(lo, 0.0); hi = max(hi, 0.0)
    if hi - lo < 0.4:
        mid = (hi + lo) / 2
        lo, hi = mid - 0.2, mid + 0.2
    pad = (hi - lo) * 0.10
    axf.set_xlim(lo - pad, hi + pad)

    pos = {'slidetags': -JIT, 'merfish': 0, 'xenium': +JIT}
    for i, r in enumerate(gd['rows']):
        for p in PLATFORMS:
            d = r['plat'][p]
            lci, uci = d.get('lci', np.nan), d.get('uci', np.nan)
            if np.isnan(lci) or np.isnan(uci):
                continue
            axf.hlines(i + pos[p], lci, uci,
                       color=PLATFORM_COLORS[p], lw=0.9, zorder=3)
        if not np.isnan(r['meta_lfc']) and r['D'] >= 2:
            axf.plot(r['meta_lfc'], i, 'D',
                     mfc=D_COLORS[r['D']], mec='black',
                     mew=0.3, ms=3.0, zorder=5)
    axf.set_yticks([])
    axf.tick_params(axis='x', labelsize=AXIS_FS, length=1.5, pad=1)
    if show_xlabel:
        axf.set_xlabel('logFC', fontsize=AXIS_FS, labelpad=1)
    for sp in axf.spines.values():
        sp.set_linewidth(0.5)

cards_top_in = ax_a_top_in + CARD_TITLE_H_IN
card_total_h_in = content_h_in + CARD_TITLE_H_IN
for i, g in enumerate(CARD_GENES):
    gd = cards[g]
    card_top_in = cards_top_in - i * (card_total_h_in + CARD_GAP_IN)
    title_bot_in = card_top_in - CARD_TITLE_H_IN
    content_bot_in = title_bot_in - content_h_in

    fig.text(cards_left_in / fig_w,
             (card_top_in - 0.02) / fig_h, g,
             ha='left', va='top', fontsize=TITLE_FS,
             fontstyle='italic')

    n_sp = len(gd['sp_pair'])
    sp_bot_in = content_bot_in
    for si in range(n_sp):
        sp_left_in = cards_left_in + si * (SP_W_IN + SP_GAP_IN)
        ax_sp = fig.add_axes([sp_left_in / fig_w, sp_bot_in / fig_h,
                              SP_W_IN / fig_w, SP_W_IN / fig_h])
        ds = gd['sp_pair'][si]
        ax_sp.set_title(PLATFORM_LABELS[ds], fontsize=AXIS_FS, pad=1.5)
        ax_sp.set_xticks([]); ax_sp.set_yticks([])
        ax_sp.set_facecolor('black')
        for sp in ax_sp.spines.values():
            sp.set_linewidth(0.5)
        sd = gd['sp_data'][si]
        if sd is None or len(sd['x']) == 0:
            continue
        vmin_sp, vmax_sp = gd['sp_vranges'][si]
        order = np.argsort(sd['expr'])
        n_total_ds = len(sp_coords[ds]['x'])
        sp_size = 250.0 / np.sqrt(n_total_ds)
        ax_sp.scatter(sd['x'][order], sd['y'][order],
                      c=sd['expr'][order], cmap='viridis',
                      s=sp_size, vmin=vmin_sp, vmax=vmax_sp,
                      linewidths=0, rasterized=True)
        c = sp_coords[ds]
        ax_sp.plot([sd['midline'], sd['midline']],
                   [c['y'].min(), c['y'].max()],
                   color='white', lw=0.3, zorder=2)
        ax_sp.set_xlim(sd['fov_cx'] - sd['fov_half'],
                       sd['fov_cx'] + sd['fov_half'])
        ax_sp.set_ylim(sd['fov_cy'] - sd['fov_half'],
                       sd['fov_cy'] + sd['fov_half'])
        ax_sp.set_aspect('equal')
        ax_sp.text(0.25, 0.97, 'Null', transform=ax_sp.transAxes,
                   ha='center', va='top', color='white',
                   fontsize=AXIS_FS - 1.5, zorder=4)
        ax_sp.text(0.75, 0.97, 'Preg', transform=ax_sp.transAxes,
                   ha='center', va='top', color='white',
                   fontsize=AXIS_FS - 1.5, zorder=4)

    n_cells = len(gd['rows'])
    forest_h_in = max(n_cells * CARD_FOREST_ROW_H, CARD_FOREST_MIN_H)
    forest_h_in = min(forest_h_in, content_h_in)
    forest_bot_in = content_bot_in + (content_h_in - forest_h_in) / 2
    forest_left_in = (cards_left_in
                      + n_sp * SP_W_IN + (n_sp - 1) * SP_GAP_IN
                      + SP_FOREST_GAP_IN)
    ax_f = fig.add_axes([forest_left_in / fig_w, forest_bot_in / fig_h,
                         FOREST_W_IN / fig_w, forest_h_in / fig_h])
    draw_forest(ax_f, gd, show_xlabel=(i == N_CARDS - 1))

    label_left_frac = 1.0 + FOREST_LABEL_GAP_IN / FOREST_W_IN
    trans_y_f = ax_f.get_yaxis_transform()
    for ri, r in enumerate(gd['rows']):
        text = f"{short_ct(r['cell_type'])} {r['stars']}".strip()
        ax_f.text(label_left_frac, ri, text,
                  transform=trans_y_f, ha='left', va='center',
                  fontsize=LABEL_FS, clip_on=False)

# =============================================================================
# Forest legend column (Platforms / Meta / Significance)
# =============================================================================
flegv_left_in = (cards_left_in + CARD_W_MAX_IN + FLEG_GAP_IN
                 + FLEG_EXTRA_RIGHT_IN)
flegv_top_in = cards_top_in
flegv_bot_in = ax_b_bot_in
flegv_h_in = flegv_top_in - flegv_bot_in
ax_fleg = fig.add_axes([flegv_left_in / fig_w, flegv_bot_in / fig_h,
                        FLEG_COL_W_IN / fig_w, flegv_h_in / fig_h])
ax_fleg.set_xlim(0, 1); ax_fleg.set_ylim(0, 1)
ax_fleg.set_xticks([]); ax_fleg.set_yticks([])
for s in ax_fleg.spines.values():
    s.set_visible(False)

FLEG_FS = 6.5
FLEG_HEADER_FS = 6.8
sections = [
    ('Platforms', [
        ('line', 'slidetags', PLATFORM_LABELS['slidetags']),
        ('line', 'merfish',   PLATFORM_LABELS['merfish']),
        ('line', 'xenium',    PLATFORM_LABELS['xenium']),
    ]),
    ('Meta', [
        ('diamond', 3, 'D=3'),
        ('diamond', 2, 'D=2'),
    ]),
    ('Significance', [
        ('star', 1, r'*  emp$\,p\leq 0.05$'),
        ('star', 2, r'**  emp$\,p\leq 0.01$'),
        ('star', 3, r'***  emp$\,p\leq 0.001$'),
    ]),
]
header_h = 0.075 * 1.95 / flegv_h_in
item_h = 0.060 * 1.95 / flegv_h_in
section_gap = 0.050 * 1.95 / flegv_h_in
y = 1.0 - (1.0 - 0.97) * 1.95 / flegv_h_in
for header, items in sections:
    ax_fleg.text(0.05, y, header, ha='left', va='top',
                 fontsize=FLEG_HEADER_FS, transform=ax_fleg.transAxes)
    y -= header_h
    for kind, key, label in items:
        if kind == 'line':
            ax_fleg.plot([0.06, 0.13], [y, y],
                         color=PLATFORM_COLORS[key], lw=1.4,
                         solid_capstyle='butt',
                         transform=ax_fleg.transAxes, clip_on=False)
            ax_fleg.text(0.17, y, label, ha='left', va='center',
                         fontsize=FLEG_FS, transform=ax_fleg.transAxes)
        elif kind == 'diamond':
            ax_fleg.plot(0.095, y, 'D', mfc=D_COLORS[key], mec='black',
                         mew=0.3, ms=3.5,
                         transform=ax_fleg.transAxes, clip_on=False)
            ax_fleg.text(0.17, y, label, ha='left', va='center',
                         fontsize=FLEG_FS, transform=ax_fleg.transAxes)
        elif kind == 'star':
            ax_fleg.text(0.05, y, label, ha='left', va='center',
                         fontsize=FLEG_FS, transform=ax_fleg.transAxes)
        y -= item_h
    y -= section_gap

# =============================================================================
# Cell-state bottom row: Panel D (Endo+Peri UCell density) + Panel E (top-5
# neuron-subclass env_score heatmap). Slidetags only.
# =============================================================================
def _cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1)
                      + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    if pooled == 0:
        return np.nan
    return (b.mean() - a.mean()) / pooled


def _stars_for(p):
    if p is None or np.isnan(p): return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

# Panel D spans the dotplot width (ax_w_in, left-aligned with ax_left_in).
# Panel E spans from the leftmost spatial plot (cards_left_in) to the
# RIGHT EDGE OF THE WIDEST FOREST BOX (D=3 cards), excluding the cell-type
# label column. Panel E gets a smaller height (CS_E_H_IN) vertically
# centered on Panel D.
cs_row_bot_in = cs_row_axis_bot_in + CS_ROW_BOT_PAD_IN
cs_d_left_in = ax_left_in
cs_d_w_in = ax_w_in
cs_e_left_in = cards_left_in
# Right edge of forest box for the widest card (D=MAX_SP_N):
cs_e_right_in = (cards_left_in + MAX_SP_N * SP_W_IN
                 + (MAX_SP_N - 1) * SP_GAP_IN
                 + SP_FOREST_GAP_IN + FOREST_W_IN)
cs_e_w_in = cs_e_right_in - cs_e_left_in
cs_e_bot_in = cs_row_bot_in + (CS_ROW_H_IN - CS_E_H_IN) / 2

ax_d = fig.add_axes([cs_d_left_in / fig_w, cs_row_bot_in / fig_h,
                     cs_d_w_in / fig_w, CS_ROW_H_IN / fig_h])
ax_e = fig.add_axes([cs_e_left_in / fig_w, cs_e_bot_in / fig_h,
                     cs_e_w_in / fig_w, CS_E_H_IN / fig_h])

# ----- Panel D: VLMC+Peri+SMC+Endo combined UCell density -----
focal = cs_ucell.filter(pl.col('subclass').is_in(CS_FOCAL_CELLS))
ctrl_u = focal.filter(pl.col('condition') == 'CTRL')['ucell'].to_numpy()
preg_u = focal.filter(pl.col('condition') == 'PREG')['ucell'].to_numpy()
med_c = float(np.median(ctrl_u))
med_p = float(np.median(preg_u))

# Per-sample t-test for sig stars between medians
cm = (focal.filter(pl.col('condition') == 'CTRL')
      .group_by('sample').agg(pl.col('ucell').mean().alias('m')))['m'].to_numpy()
pm = (focal.filter(pl.col('condition') == 'PREG')
      .group_by('sample').agg(pl.col('ucell').mean().alias('m')))['m'].to_numpy()
t_p = ttest_ind(cm, pm, equal_var=False)[1] \
    if len(cm) >= 2 and len(pm) >= 2 else np.nan
sig_str = _stars_for(t_p)
if not sig_str:
    sig_str = 'ns'

lo = float(np.quantile(np.concatenate([ctrl_u, preg_u]), 0.005))
hi = float(np.quantile(np.concatenate([ctrl_u, preg_u]), 0.995))
grid = np.linspace(lo, hi, 400)
kde_c = gaussian_kde(ctrl_u, bw_method=0.30)(grid)
kde_p = gaussian_kde(preg_u, bw_method=0.30)(grid)

ax_d.fill_between(grid, 0, kde_c, color=COND_COLORS['CTRL'], alpha=0.40,
                  lw=0.8, edgecolor=COND_COLORS['CTRL'])
ax_d.fill_between(grid, 0, kde_p, color=COND_COLORS['PREG'], alpha=0.40,
                  lw=0.8, edgecolor=COND_COLORS['PREG'])
y_top = max(kde_c.max(), kde_p.max()) * 1.18
ax_d.set_ylim(0, y_top)
ax_d.set_xlim(lo, hi)

# Dashed median lines (no value labels)
ax_d.vlines(med_c, 0, np.interp(med_c, grid, kde_c),
            color=COND_COLORS['CTRL'], lw=1.2, linestyles='--')
ax_d.vlines(med_p, 0, np.interp(med_p, grid, kde_p),
            color=COND_COLORS['PREG'], lw=1.2, linestyles='--')

# Significance asterisk centered between medians, above them
x_mid = (med_c + med_p) / 2.0
ax_d.text(x_mid, y_top * 0.92, sig_str,
          ha='center', va='top', fontsize=8.0,
          color='black' if sig_str != 'ns' else '#777777',
          fontweight='bold' if sig_str not in ('ns', '') else 'normal')

ax_d.set_xlabel('Microglia activation score\n(Microglia, BAM)',
                fontsize=9.0, labelpad=2, linespacing=1.05)
ax_d.set_ylabel('Cell state scoring\nDensity\n(Slide-tags)',
                fontsize=9.0, labelpad=8, linespacing=1.05)
ax_d.tick_params(axis='both', labelsize=7.0, length=2, pad=1)
for s in ('top', 'right'):
    ax_d.spines[s].set_visible(False)

# ----- Panel E: 2-row x 10-col dotplot, matching B aesthetic.
# Reuses top_neuron_df computed above (also drives the chord-row neuron pool).
sub_df = top_neuron_df

# Compute size mapping (shared between Panel E and LEG_E)
if len(sub_df):
    _vals = np.concatenate([sub_df['mean_ctrl'].to_numpy(),
                            sub_df['mean_preg'].to_numpy()])
    _size_mn = float(_vals.min())
    _size_mx = float(_vals.max())
else:
    _size_mn, _size_mx = 0.0, 1.0


def env_to_size(v):
    """Map env-score to scatter-marker area; matches Panel B's pct_to_size
    range (SIZE_MIN..SIZE_MAX) so max dot size is consistent across the
    figure's dotplots."""
    if _size_mx <= _size_mn:
        return SIZE_MAX
    f = (v - _size_mn) / (_size_mx - _size_mn)
    return SIZE_MIN + f * (SIZE_MAX - SIZE_MIN)


def _pretty_breaks(lo, hi, target_n=3):
    """R-style pretty breaks: pick a 1/2/5 * 10^k step that produces
    ~target_n round ticks spanning [lo, hi]."""
    if hi <= lo:
        return [round(float(lo), 3)]
    approx_step = (hi - lo) / max(1, target_n - 1)
    magnitude = 10 ** np.floor(np.log10(approx_step))
    norm_step = approx_step / magnitude
    if norm_step < 1.5:    nice = 1
    elif norm_step < 3:    nice = 2
    elif norm_step < 7:    nice = 5
    else:                  nice = 10
    step = nice * magnitude
    b_lo = np.floor(lo / step) * step
    b_hi = np.ceil(hi / step) * step
    n = int(round((b_hi - b_lo) / step)) + 1
    return [round(b_lo + i * step, 3) for i in range(n)]


if len(sub_df):
    n_sub = len(sub_df)
    ax_e.set_facecolor('white')
    ax_e.set_xlim(-0.5, n_sub - 0.5)
    ax_e.set_ylim(1.5, -0.5)  # row 0 (Nulliparous) top, row 1 (Pregnancy) bot

    # Grid + box: match Panel A/B aesthetic
    for k in range(1, n_sub):
        ax_e.axvline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
    ax_e.axhline(0.5, color='#F2F2F2', lw=0.25, zorder=1)

    ax_e.set_xticks(range(n_sub))
    ax_e.set_xticklabels([short_ct(s) for s in sub_df['subclass']],
                         rotation=45, ha='right', fontsize=7.5)
    ax_e.tick_params(axis='x', length=3, pad=2, direction='out',
                     bottom=True, top=False,
                     labelbottom=True, labeltop=False)
    ax_e.set_yticks([0, 1])
    ax_e.set_yticklabels(['Nulliparous', 'Pregnancy'], fontsize=7.5,
                         color='black')
    ax_e.tick_params(axis='y', length=2, pad=2,
                     left=True, right=False,
                     labelleft=True, labelright=False)
    for sp in ('top', 'right', 'left', 'bottom'):
        ax_e.spines[sp].set_visible(True)
        ax_e.spines[sp].set_color('black')
        ax_e.spines[sp].set_linewidth(0.9)

    for i, r in sub_df.iterrows():
        ax_e.scatter([i], [0],
                     s=env_to_size(r['mean_ctrl']),
                     c=COND_COLORS['CTRL'], marker='o',
                     edgecolors='black', linewidths=0.4, zorder=3)
        ax_e.scatter([i], [1],
                     s=env_to_size(r['mean_preg']),
                     c=COND_COLORS['PREG'], marker='o',
                     edgecolors='black', linewidths=0.4, zorder=3)
else:
    ax_e.text(0.5, 0.5, 'No subclasses with sufficient n',
              transform=ax_e.transAxes, ha='center', va='center',
              fontsize=8.0)
    ax_e.set_xticks([]); ax_e.set_yticks([])

# ----- LEG_E: size legend for Panel E, RIGHT side, left-justified.
# Positioned in the same vertical band as Panel D, in the column
# previously used by the forest legend (which ends at ax_b_bot_in).
LEG_E_LEFT_IN = flegv_left_in
LEG_E_W_IN    = FLEG_COL_W_IN
LEG_E_BOT_IN  = cs_row_bot_in
LEG_E_H_IN    = CS_ROW_H_IN
ax_leg_e = fig.add_axes([LEG_E_LEFT_IN / fig_w, LEG_E_BOT_IN / fig_h,
                         LEG_E_W_IN / fig_w, LEG_E_H_IN / fig_h],
                        zorder=100)
ax_leg_e.set_xlim(0, 1); ax_leg_e.set_ylim(0, 1)
ax_leg_e.set_xticks([]); ax_leg_e.set_yticks([])
ax_leg_e.set_facecolor('white')
ax_leg_e.patch.set_alpha(1.0)
for s in ax_leg_e.spines.values():
    s.set_visible(False)
ax_leg_e.text(0.02, 0.96, 'Mean activation score\nk=10 nearest vasc. cells',
              ha='left', va='top', fontsize=6.8, linespacing=1.05)
legend_vals = _pretty_breaks(_size_mn, _size_mx, target_n=3)
y_top, y_step = 0.55, 0.18
for k, lev in enumerate(legend_vals):
    y = y_top - k * y_step
    ax_leg_e.scatter([0.10], [y], s=env_to_size(lev),
                     c=['#888888'], marker='o',
                     edgecolors='black', linewidths=0.4)
    ax_leg_e.text(0.22, y, f'{lev:.3f}',
                  ha='left', va='center', fontsize=6.5)

# =============================================================================
# Chord row: 3 LR themes (Notch / VEGF / Ang2+Cxcl12), polar pycirclize plots
# =============================================================================
chord_panels_left_in = ax_left_in
chord_theme_axes = {}
for i, theme in enumerate(CHORD_THEMES):
    px_left_in = (chord_panels_left_in
                  + i * (chord_panel_w_in + CHORD_PANEL_GAP_IN))
    ax = fig.add_axes([px_left_in / fig_w, chord_panel_bot_in / fig_h,
                       chord_panel_w_in / fig_w,
                       chord_panel_h_in / fig_h],
                      projection='polar')
    chord_theme_axes[theme] = ax


def draw_theme_chord(ax, theme):
    df = theme_chord_edges(theme)
    if df.height == 0:
        ax.text(0.5, 0.5, 'no edges', transform=ax.transAxes,
                ha='center', va='center', fontsize=7.0, color='#888888')
        return df
    df = df.sort('mag', descending=True).head(40)

    sec_size = {c: 0.0 for c in chord_cell_set}
    for r in df.iter_rows(named=True):
        sec_size[r['source']] = sec_size.get(r['source'], 0.0) + r['mag']
        sec_size[r['target']] = sec_size.get(r['target'], 0.0) + r['mag']
    max_sz = max(sec_size.values()) if any(sec_size.values()) else 1.0
    floor = max_sz * 0.05
    sector_dict = {c: max(sec_size.get(c, 0.0), floor)
                   for c in chord_cells_ordered}

    # Gap between sectors: larger when crossing class boundaries
    space_per = []
    for k, c in enumerate(chord_cells_ordered):
        if k == 0:
            space_per.append(2.0); continue
        prev_c = chord_cells_ordered[k - 1]
        space_per.append(7.0 if chord_class(c) != chord_class(prev_c)
                         else 2.0)
    circos = Circos(sector_dict, space=space_per)

    for sector in circos.sectors:
        c = sector.name
        inner = sector.add_track((85, 91), r_pad_ratio=0.0)
        inner.axis(fc=SUBCLASS_COLORS.get(c, '#d3d3d3'),
                   ec='black', lw=0.3)
        sector.text(chord_num_label(c), r=106, size=5.5, color='black',
                    orientation='vertical')

    max_mag = float(df['mag'].max())
    src_off = {c: 0.0 for c in chord_cell_set}
    tgt_off = {c: 0.0 for c in chord_cell_set}
    for r in df.sort('mag', descending=False).iter_rows(named=True):
        s, t, w = r['source'], r['target'], r['mag']
        col = CHORD_COLOR_UP if r['signed_sum'] > 0 else CHORD_COLOR_DOWN
        alpha = 0.30 + 0.60 * (w / max_mag)
        s0, s1 = src_off[s], src_off[s] + w
        t0, t1 = tgt_off[t], tgt_off[t] + w
        src_off[s], tgt_off[t] = s1, t1
        circos.link((s, s0, s1), (t, t0, t1),
                    color=col, alpha=alpha,
                    direction=1, height_ratio=0.50,
                    arrow_length_ratio=0.05, allow_twist=True)
    circos.plotfig(ax=ax)

    # Outer class arc (NN / Glut / GABA)
    class_groups = {}
    for sector in circos.sectors:
        class_groups.setdefault(chord_class(sector.name), []).append(sector)
    for cls, sectors_in_class in class_groups.items():
        rads = [s.x_to_rad(0) for s in sectors_in_class] + \
               [s.x_to_rad(s.size) for s in sectors_in_class]
        start_rad, end_rad = min(rads), max(rads)
        center = (start_rad + end_rad) / 2
        width = end_rad - start_rad
        ax.bar(x=center, height=6, width=width, bottom=94,
               facecolor=CHORD_CLASS_COLORS[cls],
               edgecolor='black', linewidth=0.4,
               align='center', zorder=0.5)
    return df


for theme, ax in chord_theme_axes.items():
    draw_theme_chord(ax, theme)

# Theme titles above each chord
fig.canvas.draw()
_title_offset_frac = 0.30 / fig_h
for theme, ax in chord_theme_axes.items():
    bbox = ax.get_position()
    fig.text((bbox.x0 + bbox.x1) / 2,
             bbox.y1 + _title_offset_frac,
             CHORD_THEME_TITLES[theme], ha='center', va='bottom',
             fontsize=8.5)

# Chord legend (LEFT column, below TLEG): Direction + Cell class.
# Generous spacing so items don't visually crowd each other.
CHORD_LEG_TOP_IN = TLEG_BOT_IN - 1 * GENE_PITCH
CHORD_LEG_H_IN   = 8 * GENE_PITCH
CHORD_LEG_BOT_IN = CHORD_LEG_TOP_IN - CHORD_LEG_H_IN
ax_chord_leg = _make_leg_axes(CHORD_LEG_BOT_IN, CHORD_LEG_H_IN)

SW_W, SW_H = 0.10, 0.055
SW_X = 0.95 - SW_W
LBL_X = SW_X - 0.02

y = 0.95
ax_chord_leg.text(0.95, y, 'Direction', ha='right', va='top', fontsize=6.8)
y -= 0.12
for lbl, col in [('UP in pregnancy',   CHORD_COLOR_UP),
                 ('DOWN in pregnancy', CHORD_COLOR_DOWN)]:
    ax_chord_leg.add_patch(mpl.patches.Rectangle(
        (SW_X, y - SW_H / 2), SW_W, SW_H,
        facecolor=col, edgecolor='black', lw=0.3))
    ax_chord_leg.text(LBL_X, y, lbl,
                      ha='right', va='center', fontsize=6.5)
    y -= 0.11

y -= 0.06
ax_chord_leg.text(0.95, y, 'Cell class', ha='right', va='top', fontsize=6.8)
y -= 0.12
for cls in CHORD_CLASS_ORDER:
    ax_chord_leg.add_patch(mpl.patches.Rectangle(
        (SW_X, y - SW_H / 2), SW_W, SW_H,
        facecolor=CHORD_CLASS_COLORS[cls], edgecolor='black', lw=0.4))
    ax_chord_leg.text(LBL_X, y, CHORD_CLASS_LABELS[cls],
                      ha='right', va='center', fontsize=6.5)
    y -= 0.11

# Subclass legend (LEFT column, below CHORD_LEG): right-justified, matches
# the other left-column legends' aesthetic. Cells listed by pure ABCA
# numeric prefix (no class grouping). Swatch height kept well below the
# inter-row step so each row has visible breathing space.
SCLEG_TOP_IN = CHORD_LEG_BOT_IN - 1 * GENE_PITCH
n_sc_entries = len(chord_cell_set)
SCLEG_H_IN = max(8 * GENE_PITCH, (n_sc_entries + 2) * 0.100)
SCLEG_BOT_IN = SCLEG_TOP_IN - SCLEG_H_IN
ax_scleg = _make_leg_axes(SCLEG_BOT_IN, SCLEG_H_IN)
SC_SW_W, SC_SW_H = 0.08, 0.025
SC_SW_X = 0.95 - SC_SW_W
SC_LBL_X = SC_SW_X - 0.02

scleg_cells = sorted(chord_cell_set, key=numeric_prefix)
y = 0.96
ax_scleg.text(0.95, y, 'Subclass', ha='right', va='top', fontsize=6.8)
y -= 0.07
item_step = (y - 0.04) / max(1, n_sc_entries)
for ct in scleg_cells:
    y -= item_step
    ax_scleg.add_patch(plt.Rectangle(
        (SC_SW_X, y - SC_SW_H / 2), SC_SW_W, SC_SW_H,
        facecolor=SUBCLASS_COLORS.get(ct, '#d3d3d3'),
        edgecolor='black', lw=0.25))
    ax_scleg.text(SC_LBL_X, y, short_ct(ct),
                  ha='right', va='center', fontsize=6.5,
                  clip_on=False)

for ext in ('png', 'svg'):
    fig.savefig(f'{out_dir}/microglia_combined.{ext}',
                bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'wrote {out_dir}/microglia_combined.png and .svg')
