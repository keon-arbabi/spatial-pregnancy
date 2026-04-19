"""
Definitive count-level comparison: corrected counts vs raw counts.

No DE pipeline intermediates — we ask directly:
    "Does the correction remove off-target (contamination-like) expression
     while preserving on-target (cell-type-specific) expression?"

Test design:
  - For each cell type c, partition genes into:
      OWN markers:  ref_pct_in(c) >= 50%  AND  ref_pct_in(other) <= 20% (for
                    at least the max other-class)
      OFF markers:  ref_pct_in(c) <= 10%  AND  max_ref_pct_in(others) >= 50%
  - For each cell in c, compute mean expression of OWN and OFF markers,
    using (raw) `layers['counts']` and (corrected) `adata.X`.
  - Ideal correction:
      mean(OWN, corrected) / mean(OWN, raw)   ≈ 1      (preserved)
      mean(OFF, corrected) / mean(OFF, raw)   << 1     (removed)

  We then aggregate per cell class (Glut / Gaba / NN) so each platform
  yields three numbers: retention ratio, removal ratio, and their gap.
  A correction is "superior" iff removal is substantial AND retention is
  near 1 — that is, the gap is large.

Also reports:
  - library-size change (corrected should be <= raw; drop reflects total
    contamination)
  - mutually-exclusive marker co-expression rate (contamination double-
    positives), before and after — the classic RESOLVI paper metric.

Run with: POLARS_MAX_THREADS=4 python XX_clipboard.py
"""

import os
import gc
import pickle
import time
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

WORK = '/home/karbabi/spatial-pregnancy'
OWN_HI = 50      # own cell-type ref_pct threshold (%)
OWN_LO = 20      # max other-type ref_pct for OWN (%)
OFF_HI = 50      # other-type ref_pct threshold for OFF (%)
OFF_LO = 10      # own-type ref_pct for OFF (%)
MIN_CELLS_PER_TYPE = 20
MIN_MARKERS = 5

DATASETS = ['merfish', 'xenium', 'slidetags']


def cell_class(ct):
    s = str(ct)
    if any(k in s for k in ['NN', 'Astro', 'Oligo', 'OPC', 'Endo',
                             'Microglia', 'VLMC', 'Epen', 'Peri', 'SMC',
                             'Macrophage']):
        return 'NN'
    if 'Glut' in s:
        return 'Glut'
    if any(k in s for k in ['Gaba', 'GABA', 'IMN', 'Chol']):
        return 'Gaba'
    return 'Other'


def hr(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


# ----- load ref_pct lookup --------------------------------------------------

pct_file = f'{WORK}/output/ref_pct_detected.pkl'
with open(pct_file, 'rb') as f:
    pct_detected = pickle.load(f)  # dict[cell_type] -> pd.Series (gene -> frac)
print(f'loaded ref_pct for {len(pct_detected)} cell types')


def build_marker_sets(adata, cell_types, var_names):
    """For each cell type, find OWN and OFF marker gene indices (in var_names).

    OWN: highly detected in this type (>= OWN_HI%), low in the *max*
         of other types (<= OWN_LO%).
    OFF: highly detected in some OTHER type (>= OFF_HI%) AND low in
         this type (<= OFF_LO%).
    """
    g2i = {g: i for i, g in enumerate(var_names)}
    shared_types = [ct for ct in cell_types if ct in pct_detected]
    # precompute per-type ref_pct vectors aligned to var_names, in %
    ref_mat = np.zeros((len(shared_types), len(var_names)), dtype=np.float32)
    for i, ct in enumerate(shared_types):
        s = pct_detected[ct]
        for g, v in s.items():
            j = g2i.get(g)
            if j is not None:
                ref_mat[i, j] = v * 100.0
    type2idx = {ct: i for i, ct in enumerate(shared_types)}
    results = {}
    for ct in shared_types:
        i = type2idx[ct]
        own_in = ref_mat[i]
        # "others": max ref_pct across all other types
        other_mask = np.ones(len(shared_types), dtype=bool)
        other_mask[i] = False
        others_max = ref_mat[other_mask].max(axis=0)
        own = np.where((own_in >= OWN_HI) & (others_max <= OWN_LO))[0]
        off = np.where((own_in <= OFF_LO) & (others_max >= OFF_HI))[0]
        results[ct] = {'own': own, 'off': off}
    return results


def mean_per_cell(mat, cell_idx, gene_idx):
    """Mean expression per cell across `gene_idx` columns.  Returns (n_cells,)."""
    if len(gene_idx) == 0 or len(cell_idx) == 0:
        return np.full(len(cell_idx), np.nan, dtype=np.float32)
    sub = mat[cell_idx][:, gene_idx]
    if sp.issparse(sub):
        s = np.asarray(sub.sum(axis=1)).ravel()
    else:
        s = sub.sum(axis=1)
    return (s / max(1, len(gene_idx))).astype(np.float32)


# ----- mutually-exclusive marker gene pairs (coarse classes) ----------------
# Use canonical pan-class markers; look up actual gene presence below.
CLASS_MARKERS = {
    'Glut':  ['Slc17a7', 'Slc17a6', 'Slc17a8'],
    'Gaba':  ['Gad1', 'Gad2', 'Slc32a1'],
    'Astro': ['Gfap', 'Aqp4', 'Slc1a3'],
    'Oligo': ['Mbp', 'Mog', 'Mag', 'Plp1'],
    'OPC':   ['Pdgfra', 'Cspg4'],
    'Endo':  ['Pecam1', 'Cldn5', 'Flt1', 'Mfsd2a'],
    'Micro': ['Cx3cr1', 'Tmem119', 'P2ry12', 'Csf1r'],
}
# pairs that should be mutually exclusive
MX_PAIRS = [
    ('Slc17a7', 'Gad1'),     # glut vs gaba
    ('Slc17a7', 'Mbp'),      # neuron vs oligo
    ('Slc17a7', 'Aqp4'),     # neuron vs astro
    ('Slc17a7', 'Cldn5'),    # neuron vs endo
    ('Gad1',    'Mbp'),      # gaba vs oligo
    ('Gad1',    'Aqp4'),     # gaba vs astro
    ('Mbp',     'Aqp4'),     # oligo vs astro
    ('Mbp',     'Pdgfra'),   # mature oligo vs OPC
    ('Aqp4',    'Cldn5'),    # astro vs endo
    ('Cx3cr1',  'Mbp'),      # microglia vs oligo
]


def mx_double_pos(mat, var_names, g_a, g_b):
    """Fraction of cells with observed count > 0 in both genes."""
    g2i = {g: i for i, g in enumerate(var_names)}
    ia, ib = g2i.get(g_a), g2i.get(g_b)
    if ia is None or ib is None:
        return None, None, None, None
    if sp.issparse(mat):
        a = mat[:, [ia]].toarray().ravel() > 0
        b = mat[:, [ib]].toarray().ravel() > 0
    else:
        a = mat[:, ia] > 0
        b = mat[:, ib] > 0
    pa = a.mean()
    pb = b.mean()
    p_both = (a & b).mean()
    # expected under independence
    p_exp = pa * pb
    return pa, pb, p_both, p_exp


# ----- per-dataset analysis -------------------------------------------------

summary_rows = []

for name in DATASETS:
    path = f'{WORK}/output/{name}/03_adata_query_{name}.h5ad'
    if not os.path.exists(path):
        print(f'[{name}] missing: {path}')
        continue
    hr(f'DATASET: {name}')
    t0 = time.time()
    adata = sc.read_h5ad(path)
    print(f'loaded {adata.shape[0]:,} cells × {adata.shape[1]:,} genes '
          f'in {time.time()-t0:.1f}s')
    # Confirm which layer is raw
    assert 'counts' in adata.layers, 'expected layers["counts"] = raw'
    X_corr = adata.X
    X_raw = adata.layers['counts']
    # gene-symbol normalization (slidetags may use gene_symbol col)
    if 'gene_symbol' in adata.var.columns:
        var_names = adata.var['gene_symbol'].astype(str).values
    else:
        var_names = adata.var_names.astype(str).values

    # Overall library sizes
    raw_libs = np.asarray(X_raw.sum(1)).ravel() if sp.issparse(X_raw) \
        else X_raw.sum(1)
    corr_libs = np.asarray(X_corr.sum(1)).ravel() if sp.issparse(X_corr) \
        else X_corr.sum(1)
    kept = corr_libs / np.maximum(raw_libs, 1)
    print(f'\nLibrary size: raw mean={raw_libs.mean():,.1f} '
          f'corr mean={corr_libs.mean():,.1f} '
          f'kept frac={kept.mean():.3f} (median={np.median(kept):.3f})')
    if 'class' in adata.obs.columns:
        for cls in sorted(adata.obs['class'].unique()):
            m = adata.obs['class'].values == cls
            if m.sum() < 20:
                continue
            print(f'  {cls[:45]:<45} kept={kept[m].mean():.3f}  '
                  f'(n={m.sum():,})')

    # --- marker-set specificity test ---
    cell_types = adata.obs['subclass'].astype(str).values
    unique_types = [t for t in pd.Series(cell_types).unique()
                    if t in pct_detected]
    marker_sets = build_marker_sets(adata, unique_types, var_names)

    # Aggregate per-class
    print(f'\nMarker specificity (OWN retained vs OFF removed):')
    print(f'  {"class":<6} {"n_types":>8} {"OWN ret":>9} {"OFF ret":>9}'
          f' {"gap":>7} {"OWN med/cell_raw":>18}')
    class_rows = {'Glut': [], 'Gaba': [], 'NN': []}
    for ct in unique_types:
        cls = cell_class(ct)
        if cls not in class_rows:
            continue
        mask = (cell_types == ct)
        if mask.sum() < MIN_CELLS_PER_TYPE:
            continue
        own_idx = marker_sets[ct]['own']
        off_idx = marker_sets[ct]['off']
        if len(own_idx) < MIN_MARKERS or len(off_idx) < MIN_MARKERS:
            continue
        idx = np.where(mask)[0]
        own_raw = mean_per_cell(X_raw, idx, own_idx)
        off_raw = mean_per_cell(X_raw, idx, off_idx)
        own_corr = mean_per_cell(X_corr, idx, own_idx)
        off_corr = mean_per_cell(X_corr, idx, off_idx)
        # ratio per cell, skipping zero-denominators, then mean
        def ratio(a, b):
            m = b > 0
            return (a[m] / b[m]).mean() if m.any() else np.nan
        own_ret = ratio(own_corr, own_raw)
        off_ret = ratio(off_corr, off_raw)
        class_rows[cls].append((ct, len(own_idx), len(off_idx),
                                own_ret, off_ret,
                                float(np.median(own_raw))))
    for cls in ['Glut', 'Gaba', 'NN']:
        rows = class_rows[cls]
        if not rows:
            continue
        own_rets = np.array([r[3] for r in rows])
        off_rets = np.array([r[4] for r in rows])
        gap = np.nanmedian(own_rets - off_rets)
        print(f'  {cls:<6} {len(rows):>8} {np.nanmedian(own_rets):>9.3f}'
              f' {np.nanmedian(off_rets):>9.3f} {gap:>+7.3f}'
              f' {np.nanmedian([r[5] for r in rows]):>18.3f}')
        summary_rows.append(dict(dataset=name, cls=cls,
                                 n_types=len(rows),
                                 own_retained=float(np.nanmedian(own_rets)),
                                 off_retained=float(np.nanmedian(off_rets)),
                                 gap=float(gap)))

    # --- mutually exclusive double-positive rates ---
    print(f'\nMutually-exclusive marker double-positive rates:')
    print(f'  {"pair":<28} {"raw p(both)":>12} {"corr p(both)":>13}'
          f' {"Δ":>8} {"expected":>10}')
    dp_rows = []
    for g_a, g_b in MX_PAIRS:
        pa_r, pb_r, p_r, p_exp_r = mx_double_pos(X_raw, var_names, g_a, g_b)
        if p_r is None:
            continue
        pa_c, pb_c, p_c, p_exp_c = mx_double_pos(
            X_corr, var_names, g_a, g_b)
        if p_c is None:
            continue
        delta = p_c - p_r
        print(f'  {g_a+"+"+g_b:<28} {p_r:>12.4%} {p_c:>13.4%}'
              f' {delta:>+8.4%} {p_exp_r:>10.4%}')
        dp_rows.append(dict(dataset=name, pair=f'{g_a}+{g_b}',
                            raw=float(p_r), corr=float(p_c),
                            delta=float(delta),
                            expected_ind=float(p_exp_r)))

    del adata, X_corr, X_raw
    gc.collect()


# ----- summary + verdict ----------------------------------------------------

hr('FINAL SYNTHESIS: per-dataset, per-class marker specificity')
df = pd.DataFrame(summary_rows)
if len(df) > 0:
    print(df.to_string(index=False))

    print('\nReading:')
    print('  own_retained ≈ 1.0 means OWN-type markers preserved')
    print('  off_retained << 1.0 means OFF-type markers suppressed')
    print('  gap = (own - off); larger gap = cleaner contamination removal')
    print('  gap > 0.3 is strong; gap > 0.1 is detectable; gap ≤ 0 is bad')

    print('\nPer-dataset scores (median gap across classes):')
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        if len(sub) == 0:
            continue
        med_gap = sub.gap.median()
        verdict = ('SUPERIOR (targeted)' if med_gap > 0.3
                   else 'MIXED (modest)' if med_gap > 0.1
                   else 'WEAK/NEUTRAL' if med_gap > 0
                   else 'INFERIOR (broken)')
        print(f'  {ds:<10} median gap={med_gap:+.3f} → {verdict}')

print('\nDouble-positive change across all pairs/datasets: negative Δ (less')
print('co-expression of mutually exclusive markers) is the ideal signal that')
print('correction is removing contamination rather than shrinking uniformly.')
