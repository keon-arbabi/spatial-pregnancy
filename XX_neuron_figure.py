import os
import re
import warnings

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc

import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=RuntimeWarning)

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures/neuron'
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
# Panel A: GSEA pathway selection (5 bands, 15 pathways).
# Neuronal remodeling in late pregnancy: synaptic connectivity, excitability,
# neuropeptide and glucocorticoid/stress machinery UP (the stress arm is
# orthogonally validated by FKBP5 IHC); neuronal differentiation program DOWN.
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
    ('Neuronal development', [
        'GOBP_REGULATION_OF_NEURON_DIFFERENTIATION',
        'GOBP_NEURON_FATE_COMMITMENT',
        'GOBP_AXON_DEVELOPMENT',
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
    'GOBP_CELLULAR_RESPONSE_TO_CORTICOSTEROID_STIMULUS': 'corticosteroid signaling',
    'GOBP_REGULATION_OF_NEURON_DIFFERENTIATION': 'neuron differentiation',
    'GOBP_NEURON_FATE_COMMITMENT':               'neuron fate commitment',
    'GOBP_AXON_DEVELOPMENT':                     'axon development',
}

# Okabe-Ito (colorblind-safe) -- shared between Panel A and Panel B.
BAND_COLORS = {
    'Synaptic adhesion':     '#0072B2',
    'Excitability':          '#E69F00',
    'GABA & neuropeptide':   '#009E73',
    'Glucocorticoid stress': '#D55E00',
    'Neuronal development':  '#CC79A7',
}

# =============================================================================
# Panel B: curated gene bands (same 4-band structure as Panel A). Only
# panel-covered genes (testable at D>=2) are eligible, like vascular/lipid.
# =============================================================================
GENE_BANDS = [
    # Synaptic adhesion / organizer molecules.
    ('Synaptic adhesion', [
        'Cntn1', 'Sdk1', 'Nrcam', 'Ncam1', 'Cadm1', 'Robo1', 'Cdh13',
    ]),
    # Glutamate receptors + voltage-gated channels + Ca/excitability kinase.
    ('Excitability', [
        'Gria1', 'Gria2', 'Grin1', 'Grin2a', 'Kcnh1', 'Scn8a', 'Camk4',
    ]),
    # GABA synthesis + GABA-A receptors + neuropeptides + release.
    ('GABA & neuropeptide', [
        'Gad2', 'Gad1', 'Gabrb3', 'Gabra1', 'Tac1', 'Pdyn', 'Syt1',
    ]),
    # Glucocorticoid-response genes (FKBP5 IHC-validated); elevated
    # late-pregnancy corticosterone -> GR/MR target induction.
    ('Glucocorticoid stress', [
        'Fkbp5', 'Gpr83', 'Zbtb16', 'Nr3c2', 'Ddit4', 'Bcl2',
    ]),
    # Developmental TFs + neurofilaments + activity-dependent IEGs (DOWN).
    ('Neuronal development', [
        'Tbr1', 'Foxg1', 'Sox11', 'Zfhx3', 'Nefl', 'Nefm', 'Ptn',
        'Egr1', 'Arc', 'Homer1',
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
# Cell-type selection: neurons (Glut + GABA) significant in >=3 of the 15
# curated pathways -- "consistently engaged across the neuronal themes" --
# plus the MPOA parenting nuclei forced in (sparse but biologically central).
# =============================================================================
# MPOA parenting hub (Wu et al. Nature 2014): too sparse to clear the data-
# driven threshold but force-included as the maternal-behavior anchor.
ct_allowlist = {'085 SI-MPO-LPO Lhx8 Gaba', '086 MPO-ADP Lhx8 Gaba',
                '124 MPN-MPO-PVpo Hmx2 Glut'}

_is_neuron = (pl.col('cell_type').str.contains(' Glut') |
              pl.col('cell_type').str.contains(' Gaba'))

neur = gsea.filter(pl.col('pathway').is_in(ordered_pathways) & _is_neuron)
ct_counts = (neur.group_by('cell_type')
                 .agg(pl.len().alias('n'))
                 .filter(pl.col('n') >= 3)
                 .sort('n', descending=True))
keep_cts = set(ct_counts['cell_type'].to_list()) | ct_allowlist
ct_counts = (gsea.filter(pl.col('pathway').is_in(ordered_pathways) &
                         pl.col('cell_type').is_in(keep_cts))
             .group_by('cell_type').agg(pl.len().alias('n')))
missing = keep_cts - set(ct_counts['cell_type'].to_list())
if missing:
    ct_counts = pl.concat([
        ct_counts,
        pl.DataFrame({'cell_type': sorted(missing), 'n': [0] * len(missing)},
                     schema={'cell_type': pl.Utf8, 'n': pl.UInt32})])
neur = neur.filter(pl.col('cell_type').is_in(keep_cts))
neur_all = gsea_all.filter(pl.col('pathway').is_in(ordered_pathways) &
                           pl.col('cell_type').is_in(keep_cts))
print(f'Panel A: sig hits={neur.height} all hits={neur_all.height} '
      f'cell_types={len(keep_cts)}')

CORTICAL_GABA = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg', 'Pax6']

def assign_class(ct):
    if 'Glut' in ct: return 'Glutamatergic'
    if any(t in ct for t in CORTICAL_GABA):
        return 'GABAergic\nCortex'
    return 'GABAergic\nSubcortex'

CLASS_ORDER = ['Glutamatergic', 'GABAergic\nCortex', 'GABAergic\nSubcortex']

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
for r in neur_all.iter_rows(named=True):
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
            .then(pl.col('emp_p_up')).otherwise(pl.col('emp_p_down'))
            .alias('emp_p')))
de_sr = de_sr_all.filter(pl.col('emp_p') <= 0.05)
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

d_lookup_card = {(r['gene'], r['cell_type']): r['D']
                 for r in de_sr_all.iter_rows(named=True)}

# =============================================================================
# Gene cards: anchor genes (1 per theme; stress card = FKBP5, IHC-validated).
# Load spatial coords + per-cell expression (cards) and % nonzero (dot size).
# =============================================================================
CARD_GENES = ['Cntn1', 'Gria1', 'Gad2', 'Tac1', 'Fkbp5', 'Tbr1']
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
    print(f'  {ds}: loaded coords ({int(keep_mask.sum()):,} cells) '
          f'+ genes {[g for g in CARD_GENES if sp_in[ds][g]]}')

pct_mat = np.full((len(ordered_genes), n_cols), np.nan)
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
        pcts = pct_nonzero_lists.get((g, ct), [])
        if pcts:
            pct_mat[i, j] = float(np.median(pcts))
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
# Gene-card builders (forest across platforms + spatial maps)
# =============================================================================
de_pp_full = de_pp

def short_ct(ct):
    parts = ct.split(' ')
    if len(parts) <= 3: return ct
    return f'{parts[0]} {parts[1]} {parts[-1]}'

def class_rank(ct):
    if 'Glut' in ct: return 0
    return 1

def pctl_range(arr, lo=5, hi=95):
    if len(arr) == 0: return 0.0, 1.0
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

def stars_for(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

MAX_CARD_ROWS = 6

def build_card(gene, ctx_cells=(), max_rows=MAX_CARD_ROWS):
    sr_hits = de_sr.filter((pl.col('gene') == gene) &
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
            # Forced context cell: pull D from the unfiltered sumrank so a
            # D>=2 testable but non-sig row still shows the meta diamond.
            ep = np.nan
            D = int(d_lookup_card.get((gene, ct), 0))
            stars = ''
        rows.append(dict(cell_type=ct, plat=plat_d, D=D,
                         meta_lfc=meta_lfc, emp_p=ep, stars=stars))

    # Cap to the most-significant rows (keep forced context), then order by
    # class + ABCA prefix for display.
    if len(rows) > max_rows:
        forced = set(ctx_cells)
        keep = [r for r in rows if r['cell_type'] in forced]
        others = [r for r in rows if r['cell_type'] not in forced]
        others.sort(key=lambda r: r['emp_p']
                    if not np.isnan(r['emp_p']) else 9.0)
        keep += others[:max_rows - len(keep)]
        rows = keep
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

card_ctx = {g: [] for g in CARD_GENES}
cards = {g: build_card(g, card_ctx[g]) for g in CARD_GENES}
MAX_SP_N = max(len(c['sp_pair']) for c in cards.values())
for g, c in cards.items():
    print(f'card {g}: {len(c["cell_types"])} cells, sp_pair={c["sp_pair"]}')

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

LEFT_FIG_PAD_IN = 1.15
LABEL_MARGIN_IN = 1.55
ax_left_in = LEFT_FIG_PAD_IN + LABEL_MARGIN_IN
ax_w_in = 0.20 * n_cols

leg_left_in = 0.45
leg_w_in = 0.85

# Uniform colored-box (swatch) size shared by all legends, in inches.
SWATCH_W_IN = 0.09
SWATCH_H_IN = 0.07

PATH_PITCH = 0.21
GENE_PITCH = 0.155
ax_h_a_in = PATH_PITCH * n_rows
ax_h_b_in = GENE_PITCH * n_g
gap_in = 0.50          # holds Panel A's class labels + spacing

ANNO_W_IN, ANNO_GAP_IN = 0.07, 0.03
COL_ANNO_H_IN = ANNO_W_IN
COL_ANNO_GAP_IN = 0.02

top_margin_in = 0.85
bot_margin_in = 1.80   # fits Panel B's rotated subclass x-tick labels

# Gene-card column (right of Panel A/B): spatial maps + forest + labels.
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
LABEL_W_IN = 1.05
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

fig_w = (cards_left_in + CARD_W_MAX_IN + FLEG_GAP_IN + FLEG_EXTRA_RIGHT_IN
         + FLEG_COL_W_IN + 0.10)
fig_h = (top_margin_in + ax_h_a_in + gap_in + ax_h_b_in + bot_margin_in)

ax_b_bot_in = bot_margin_in
ax_b_top_in = ax_b_bot_in + ax_h_b_in
ax_a_bot_in = ax_b_top_in + gap_in
ax_a_top_in = ax_a_bot_in + ax_h_a_in

fig = plt.figure(figsize=(fig_w, fig_h))
ax_a = fig.add_axes([ax_left_in / fig_w, ax_a_bot_in / fig_h,
                     ax_w_in / fig_w, ax_h_a_in / fig_h])
ax_b = fig.add_axes([ax_left_in / fig_w, ax_b_bot_in / fig_h,
                     ax_w_in / fig_w, ax_h_b_in / fig_h])

# =============================================================================
# Panel A: GSEA dotplot (color = NES, size = -log10 emp p)
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
# Stacked in LEFT fig-pad region, right-justified, vertical thin colorbars.
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
cb_a.ax.tick_params(labelsize=6.5, length=2, pad=1)
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
cb_b.ax.tick_params(labelsize=6.5, length=2, pad=1)
cbar_ax_b.set_zorder(110)

ax_leg_b.text(1.0, 0.45, 'DE\nPercent expressed',
              ha='right', va='top', fontsize=6.8, linespacing=1.0)
for k, lev in enumerate([10, 50, 90]):
    y = 0.24 - k * 0.10
    ax_leg_b.scatter([0.95], [y], s=pct_to_size(lev),
                     c=['#777777'], edgecolors='none')
    ax_leg_b.text(0.85, y, f'{lev}%',
                  ha='right', va='center', fontsize=6.5)

# LEG_S: shared D + Significance (applies to both panels)
ax_leg_s = _make_leg_axes(LEG_S_BOT_IN, LEG_S_H_IN)
ax_leg_s.text(1.0, 0.95, 'DE/GSEA\nCross-platform',
              ha='right', va='top', fontsize=6.8, linespacing=1.0)
ax_leg_s.scatter([0.95], [0.74], s=50, c=['#bbbbbb'],
                 edgecolors='#555555', linewidths=0.8)
ax_leg_s.text(0.85, 0.74, 'D = 3', ha='right', va='center', fontsize=6.5)
ax_leg_s.scatter([0.95], [0.60], s=50, c=['#bbbbbb'],
                 edgecolors='none', linewidths=0)
ax_leg_s.text(0.85, 0.60, 'D = 2', ha='right', va='center', fontsize=6.5)
ax_leg_s.text(1.0, 0.40, 'DE/GSEA\nSignificance',
              ha='right', va='top', fontsize=6.8, linespacing=1.0)
_sig_demo_color = cmap(norm_lfc(LFC_VMAX * 0.7))
ax_leg_s.scatter([0.95], [0.19], s=50, c=[_sig_demo_color],
                 edgecolors='none', linewidths=0)
ax_leg_s.scatter([0.95], [0.19], s=SIG_DOT_SIZE, c='white',
                 edgecolors='none', linewidths=0)
ax_leg_s.text(0.85, 0.19, r'emp $p \leq 0.05$',
              ha='right', va='center', fontsize=6.5)

# TLEG: band-color theme legend (with title)
TLEG_TOP_IN = LEG_S_BOT_IN - 1 * GENE_PITCH
TLEG_H_IN = 8 * GENE_PITCH
TLEG_BOT_IN = TLEG_TOP_IN - TLEG_H_IN
ax_tleg = _make_leg_axes(TLEG_BOT_IN, TLEG_H_IN)
ax_tleg.text(0.95, 0.97, 'DE/GSEA\nTheme', ha='right', va='top',
             fontsize=6.8, linespacing=1.0)
band_names = [b for b, _ in GENE_BANDS]
y_top, y_bot = 0.60, 0.10
y_step = (y_top - y_bot) / (len(band_names) - 1)
bw, bh = SWATCH_W_IN / leg_w_in, SWATCH_H_IN / TLEG_H_IN
for i, band in enumerate(band_names):
    y = y_top - i * y_step
    ax_tleg.add_patch(plt.Rectangle((0.95 - bw, y - bh / 2), bw, bh,
                                    facecolor=BAND_COLORS[band],
                                    edgecolor='none'))
    ax_tleg.text(0.95 - bw - 0.03, y, band, ha='right', va='center',
                 fontsize=6.5, color='black')

# =============================================================================
# Gene cards (right column): spatial maps (Null | Preg) + forest across
# platforms + cell labels with significance stars.
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

for ext in ('png', 'svg'):
    fig.savefig(f'{out_dir}/neuron_combined.{ext}',
                bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'wrote {out_dir}/neuron_combined.png and .svg')
