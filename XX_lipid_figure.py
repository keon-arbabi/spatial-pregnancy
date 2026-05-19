#region setup #################################################################

import os
import re

import numpy as np
import polars as pl
import scanpy as sc

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures/lipid'
os.makedirs(out_dir, exist_ok=True)

cmap = plt.get_cmap('seismic')

def quant_vmax(arr, q_lo=0.05, q_hi=0.95):
    """Symmetric vmax from 10/90 (or chosen) quantiles of the array."""
    flat = np.asarray(arr, dtype=float)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 1.0
    q1, q2 = np.quantile(flat, [q_lo, q_hi])
    return float(max(abs(q1), abs(q2)))

gsea = pl.read_csv(
    f'{working_dir}/output/themes/hits/gsea_sumrank.tsv', separator='\t')
real_nes = pl.read_parquet(
    f'{working_dir}/output/gsea/perms/real_gsea.parquet')\
    .filter(pl.col('contrast') == 'PREG_vs_CTRL')

#endregion

#region pathway selection #####################################################

# within each synonym cluster, pick the term with the greatest sum of nlp
synonym_groups = {
    'membrane_lipid':     ['GOBP_MEMBRANE_LIPID_METABOLIC_PROCESS',
                           'GOBP_MEMBRANE_LIPID_BIOSYNTHETIC_PROCESS'],
    'membrane_transport': ['GOBP_LIPID_TRANSLOCATION',
                           'GOBP_REGULATION_OF_MEMBRANE_LIPID_DISTRIBUTION'],
    'fa_oxidation':       ['GOBP_FATTY_ACID_BETA_OXIDATION',
                           'GOBP_FATTY_ACID_CATABOLIC_PROCESS',
                           'GOBP_LIPID_OXIDATION',
                           'GOBP_FATTY_ACID_METABOLIC_PROCESS'],
    'sphingolipid':       ['GOBP_SPHINGOLIPID_METABOLIC_PROCESS',
                           'GOBP_SPHINGOLIPID_BIOSYNTHETIC_PROCESS'],
    'lipid_broad':        ['GOBP_LIPID_METABOLIC_PROCESS',
                           'GOBP_LIPID_BIOSYNTHETIC_PROCESS',
                           'GOBP_LIPID_CATABOLIC_PROCESS'],
}


def pick_winner(candidates):
    sig = (gsea.filter(pl.col('pathway').is_in(candidates))
                .group_by('pathway')
                .agg(pl.col('nlp').sum().round(2).alias('sum_nlp'),
                     pl.len().alias('n'))
                .sort('sum_nlp', descending=True))
    return sig.row(0)[0], sig


winners = {}
for group, cands in synonym_groups.items():
    w, sig = pick_winner(cands)
    winners[group] = w
    print(f'{group}:')
    for r in sig.iter_rows(named=True):
        mark = '  <-' if r["pathway"] == w else ''
        print(f'  {r["sum_nlp"]:>5.2f}  n={r["n"]:>2}  {r["pathway"]}{mark}')

# Theme palette - Okabe-Ito (colorblind-safe, Nature-style)
BAND_COLORS = {
    'Chol & transport': '#0072B2',     # blue
    'Membrane':         '#E69F00',     # orange
    'FA oxidation':     '#009E73',     # bluish-green
    'Sphingolipid':     '#CC79A7',     # reddish-purple
    'Lipid metab.':     '#D55E00',     # vermillion
}

# 10 pathways, grouped into 5 functional bands. The carrier+chol band sits
# first to anchor the down side of the lipid economy.
pathway_bands = [
    ('Chol & transport', ['GOBP_STEROID_BIOSYNTHETIC_PROCESS',
                          'GOBP_REGULATION_OF_LIPID_LOCALIZATION']),
    ('Membrane',     [winners['membrane_lipid'],
                      winners['membrane_transport']]),
    ('FA oxidation', [winners['fa_oxidation']]),
    ('Sphingolipid', [winners['sphingolipid'],
                      'GOBP_CERAMIDE_METABOLIC_PROCESS']),
    ('Lipid metab.', [winners['lipid_broad'],
                      'GOBP_LIPID_MODIFICATION',
                      'GOBP_RESPONSE_TO_LIPID']),
]
ordered_pathways = [p for _, ps in pathway_bands for p in ps]
pathway_band = {p: b for b, ps in pathway_bands for p in ps}

pretty = {
    'GOBP_STEROID_BIOSYNTHETIC_PROCESS':               'steroid biosynthesis',
    'GOBP_REGULATION_OF_LIPID_LOCALIZATION':           'lipid localization',
    'GOBP_MEMBRANE_LIPID_METABOLIC_PROCESS':           'membrane lipid metabolism',
    'GOBP_MEMBRANE_LIPID_BIOSYNTHETIC_PROCESS':        'membrane lipid biosynthesis',
    'GOBP_LIPID_TRANSLOCATION':                        'lipid translocation',
    'GOBP_REGULATION_OF_MEMBRANE_LIPID_DISTRIBUTION':  'membrane lipid distribution',
    'GOBP_FATTY_ACID_BETA_OXIDATION':                  'fatty acid β-oxidation',
    'GOBP_FATTY_ACID_CATABOLIC_PROCESS':               'fatty acid catabolism',
    'GOBP_LIPID_OXIDATION':                            'lipid oxidation',
    'GOBP_FATTY_ACID_METABOLIC_PROCESS':               'fatty acid metabolism',
    'GOBP_SPHINGOLIPID_METABOLIC_PROCESS':             'sphingolipid metabolism',
    'GOBP_SPHINGOLIPID_BIOSYNTHETIC_PROCESS':          'sphingolipid biosynthesis',
    'GOBP_CERAMIDE_METABOLIC_PROCESS':                 'ceramide metabolism',
    'GOBP_LIPID_METABOLIC_PROCESS':                    'lipid metabolism',
    'GOBP_LIPID_BIOSYNTHETIC_PROCESS':                 'lipid biosynthesis',
    'GOBP_LIPID_CATABOLIC_PROCESS':                    'lipid catabolism',
    'GOBP_LIPID_MODIFICATION':                         'lipid modification',
    'GOBP_RESPONSE_TO_LIPID':                          'response to lipid',
}

#endregion

#region cell-type selection ###################################################

# always-included: 334 Microglia NN + 327 Oligo NN anchor the IHC validation
# story even when their lipid-pathway hit count is below the >=2 threshold
# (regional dilution masks SVZ/corpus-callosum signal). 323 Ependymal NN
# (high baseline ApoE expression in slide-tags, ~85% positive cells, n=150)
# is added to extend the lipid story to the CSF-interface arm via Panel C.
# CHOR was previously included but dropped: slide-tags samples it sparsely
# (only ~93 cells; 40% Apoe-positive; mean log2 Apoe = 1.5), so its LR
# signal in Panel C is not supported by the underlying expression data.
ct_allowlist = {'334 Microglia NN', '327 Oligo NN',
                '323 Ependymal NN'}

lipid = gsea.filter(pl.col('pathway').is_in(ordered_pathways))
ct_counts = (lipid.group_by('cell_type')
                  .agg(pl.len().alias('n'))
                  .filter(pl.col('n') >= 2)
                  .sort('n', descending=True))
keep_cts = set(ct_counts['cell_type'].to_list()) | ct_allowlist
# rebuild ct_counts so allow-listed cells get an entry (n possibly 0/1)
ct_counts = (gsea.filter(pl.col('pathway').is_in(ordered_pathways) &
                         pl.col('cell_type').is_in(keep_cts))
             .group_by('cell_type').agg(pl.len().alias('n')))
# allow-listed cells with zero hits in the curated set still need a row
missing = keep_cts - set(ct_counts['cell_type'].to_list())
if missing:
    ct_counts = pl.concat([
        ct_counts,
        pl.DataFrame({'cell_type': sorted(missing),
                      'n': [0] * len(missing)},
                     schema={'cell_type': pl.Utf8, 'n': pl.UInt32})])
lipid = lipid.filter(pl.col('cell_type').is_in(keep_cts))
print(f'hits={lipid.height} cell_types={len(keep_cts)}')

# split Gaba into cortical (Pvalb/Sst/Vip/Lamp5/...) vs subcortical (STR/PAL/OT/etc)
cortical_gaba_tokens = ['Pvalb', 'Sst', 'Vip', 'Lamp5', 'Sncg', 'Pax6']

def assign_class(ct):
    if 'NN' in ct:    return 'Non-neuronal'
    if 'Glut' in ct:  return 'Glutamatergic'
    if any(t in ct for t in cortical_gaba_tokens):
        return 'GABAergic\nCortex'
    return 'GABAergic\nSubcortex'

class_order = ['Non-neuronal', 'Glutamatergic',
               'GABAergic\nCortex', 'GABAergic\nSubcortex']

def numeric_prefix(ct):
    m = re.match(r'^(\d+)', ct)
    return int(m.group(1)) if m else 9999

ct_table = (ct_counts
    .with_columns(
        pl.col('cell_type').map_elements(
            assign_class, return_dtype=pl.Utf8).alias('class'),
        pl.col('cell_type').map_elements(
            numeric_prefix, return_dtype=pl.Int32).alias('num'))
    .with_columns(
        pl.col('class').replace_strict(
            {c: i for i, c in enumerate(class_order)}).alias('class_rank'))
    .sort(['class_rank', 'num']))

ordered_cts = ct_table['cell_type'].to_list()
ct_class = dict(zip(ct_table['cell_type'].to_list(),
                    ct_table['class'].to_list()))

#endregion

#region matrices ##############################################################

n_rows, n_cols = len(ordered_pathways), len(ordered_cts)
nlp_mat = np.full((n_rows, n_cols), np.nan)
nes_mat = np.full((n_rows, n_cols), np.nan)
d_mat = np.zeros((n_rows, n_cols), dtype=int)

ri = {p: i for i, p in enumerate(ordered_pathways)}
ci = {c: j for j, c in enumerate(ordered_cts)}
for r in lipid.iter_rows(named=True):
    i, j = ri[r['pathway']], ci[r['cell_type']]
    if np.isnan(nlp_mat[i, j]) or r['nlp'] > nlp_mat[i, j]:
        nlp_mat[i, j] = r['nlp']
        d_mat[i, j] = r['D']

# median NES across platforms (drops platforms with no measurement)
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
print(f'occupied: {int(np.sum(~np.isnan(nlp_mat)))}/{n_rows * n_cols}; '
      f'NES populated: {int(np.sum(~np.isnan(nes_mat)))}')


# contiguous runs of identical labels (used for class/band separators)
def spans(labels):
    out = []
    prev, start = labels[0], 0
    for k, lab in enumerate(labels[1:], 1):
        if lab != prev:
            out.append((prev, start, k - 1))
            prev, start = lab, k
    out.append((prev, start, len(labels) - 1))
    return out

class_spans = spans([ct_class[c] for c in ordered_cts])
band_spans = spans([pathway_band[p] for p in ordered_pathways])

#endregion

#region panel 2 gene selection ################################################

# leading-edge consensus genes from the 8 panel-1 pathways x 18 cell types
le_rows = []
for r in lipid.iter_rows(named=True):
    s = r.get('leading_edge_consensus')
    if s is None or not s.strip():
        continue
    for g in s.split(','):
        g = g.strip()
        if g:
            le_rows.append((g, r['pathway'], r['cell_type']))

gene_le = pl.DataFrame(le_rows, schema=['gene', 'pathway', 'cell_type'],
                       orient='row')

# bottom-up rule: gene appears in LE for >=3 panel-1 pathway combos AND
# has >=2 sumrank DE hits at emp_p <= 0.05 in panel-1 cell types
le_counts = gene_le.group_by('gene').agg(pl.len().alias('n_le'))
de_sr = pl.read_csv(
    f'{working_dir}/output/themes/hits/de_sumrank.tsv', separator='\t')
sr_counts = (de_sr.filter(pl.col('cell_type').is_in(keep_cts))
                  .group_by('gene').agg(pl.len().alias('n_sr')))
le_pool_df = (le_counts.join(sr_counts, on='gene', how='inner')
    .filter((pl.col('n_le') >= 3) & (pl.col('n_sr') >= 2)))
le_pool = set(le_pool_df['gene'].to_list())

# deny-list: genes that pass the LE filter but are not lipid biology - they
# enter via the broad GOBP_RESPONSE_TO_LIPID gene set (any gene whose
# expression changes upon lipid exposure: cytoskeleton, chaperones, cytokines)
gene_denylist = {'Nefl', 'Cryab', 'Mif', 'Cd68', 'Gsk3a', 'Adcy1'}
le_pool -= gene_denylist

# allow-list: canonical lipid carriers + cholesterol-biosynthesis genes that
# have >=2 sumrank DE hits in panel-1 cells. These genes carry the down-side
# of the lipid economy but were excluded by the LE-based filter because the
# 8 panel-1 pathways don't cover lipid transport / mevalonate biology.
carrier_allowlist = ['Apoe', 'Fabp7', 'Hmgcr', 'Idi1', 'Sqle', 'Msmo1', 'Srebf1']
carrier_pool = set(
    sr_counts.filter((pl.col('gene').is_in(carrier_allowlist)) &
                     (pl.col('n_sr') >= 2))['gene'].to_list())

gene_pool = sorted(le_pool | carrier_pool)
de_sr_panel = de_sr.filter(
    pl.col('cell_type').is_in(keep_cts) &
    pl.col('gene').is_in(gene_pool))
print(f'le_pool={len(le_pool)} carrier_pool={len(carrier_pool)} '
      f'panel_gene_pool={len(gene_pool)}')

# manual band override for genes where canonical biology disagrees with the
# 'majority LE pathway' rule (ceramide synthases nominated by membrane-lipid
# pathways but biologically sphingolipid; hexosaminidase/gangliosides similar)
band_override = {
    'Cers4': 'Sphingolipid', 'Cers5': 'Sphingolipid', 'Cers6': 'Sphingolipid',
    'Hexa':  'Sphingolipid', 'St6galnac5': 'Sphingolipid',
    'Lpar1': 'Lipid metab.', 'Irs2': 'Lipid metab.',
}

# assign each gene to a pathway band. Allow-list carriers -> 'Chol & transport';
# LE genes -> majority panel-1 band, with manual override where applicable.
band_priority = ['Chol & transport', 'Membrane', 'FA oxidation',
                 'Sphingolipid', 'Lipid metab.']
def gene_to_band(g):
    if g in carrier_pool:
        return 'Chol & transport'
    if g in band_override:
        return band_override[g]
    counts = (gene_le.filter(pl.col('gene') == g)
                     .with_columns(pl.col('pathway').replace_strict(
                         pathway_band).alias('band'))
                     .group_by('band').agg(pl.len().alias('n')))
    pairs = sorted(counts.iter_rows(),
                   key=lambda x: (-x[1], band_priority.index(x[0])))
    return pairs[0][0]

gene_band = {g: gene_to_band(g) for g in gene_pool}

#endregion

#region panel 2 matrices ######################################################

de_pp = pl.read_csv(
    f'{working_dir}/output/themes/hits/de_per_platform.tsv', separator='\t')
de_pp_panel = (de_pp
    .filter(pl.col('cell_type').is_in(keep_cts) &
            pl.col('gene').is_in(gene_pool) &
            pl.col('logFC').is_not_null())
    .group_by(['gene', 'cell_type'])
    .agg(pl.col('logFC').median().alias('meta_lfc'),
         pl.len().alias('n_detected')))
meta_lookup = {(r['gene'], r['cell_type']):
               (r['meta_lfc'], r['n_detected'])
               for r in de_pp_panel.iter_rows(named=True)}
sig_set = {(r['gene'], r['cell_type'])
           for r in de_sr_panel.iter_rows(named=True)}

# sort genes: by band order, then ascending mean meta_lfc (down at top)
gene_lfc_mean = {
    g: np.nanmean([meta_lookup.get((g, ct), (np.nan, 0))[0]
                   for ct in ordered_cts])
    for g in gene_pool}
ordered_genes = sorted(
    gene_pool,
    key=lambda g: (band_priority.index(gene_band[g]),
                   gene_lfc_mean[g]))

n_g, n_c = len(ordered_genes), len(ordered_cts)
lfc_mat = np.full((n_g, n_c), np.nan)
ndet_mat = np.zeros((n_g, n_c), dtype=int)
sig_mat = np.zeros((n_g, n_c), dtype=bool)
for i, g in enumerate(ordered_genes):
    for j, ct in enumerate(ordered_cts):
        lfc, n = meta_lookup.get((g, ct), (np.nan, 0))
        lfc_mat[i, j] = lfc
        ndet_mat[i, j] = n
        sig_mat[i, j] = (g, ct) in sig_set
print(f'genes={n_g} sig_cells={int(sig_mat.sum())} '
      f'missing_cells={int((ndet_mat < 2).sum())}')

gene_band_spans = spans([gene_band[g] for g in ordered_genes])

#endregion

#region panel C: Apoe outgoing signaling (LIANA) ##############################

# slide-tags only (Apoe absent from Xenium panel). D=1 above magnitude floor.
# For each (sender, receptor) pair, mean signed LR-diff across all receivers
# above the floor + scope (n_targets above floor). Cell-side senders are the
# same 20 panel-A/B subclasses; receptors are canonical Apoe receptors.
APOE_RECEPTORS = ['Lrp1', 'Lrp8', 'Vldlr', 'Ldlr',
                  'Sorl1', 'Lrp6', 'Abca1', 'Trem2']
LIANA_MAG_FLOOR = 0.01

apoe_lr = (pl.scan_csv(f'{working_dir}/output/liana/inflow_diff.csv')
    .filter((pl.col('contrast') == 'PREG_vs_CTRL') &
            (pl.col('dataset') == 'slidetags') &
            (pl.col('ligand_complex') == 'Apoe') &
            (pl.col('source').is_in(ordered_cts)) &
            (pl.col('receptor_complex').is_in(APOE_RECEPTORS)) &
            (pl.col('lr_mean_diff').abs() >= LIANA_MAG_FLOOR))
    .collect())

apoe_agg = (apoe_lr
    .group_by(['source', 'receptor_complex'])
    .agg(pl.col('lr_mean_diff').mean().alias('mean_eff'),
         pl.len().alias('n_targets')))

n_rec = len(APOE_RECEPTORS)
apoe_eff_mat = np.full((n_rec, n_c), np.nan)
apoe_n_mat = np.zeros((n_rec, n_c), dtype=int)
ri_c = {r: i for i, r in enumerate(APOE_RECEPTORS)}
for r in apoe_agg.iter_rows(named=True):
    if r['source'] not in ci:
        continue
    i, j = ri_c[r['receptor_complex']], ci[r['source']]
    apoe_eff_mat[i, j] = r['mean_eff']
    apoe_n_mat[i, j] = r['n_targets']
print(f'Apoe outgoing matrix: {int(np.sum(~np.isnan(apoe_eff_mat)))}/'
      f'{n_rec * n_c} cells; n_targets range '
      f'{apoe_n_mat[apoe_n_mat > 0].min() if (apoe_n_mat > 0).any() else 0}-'
      f'{apoe_n_mat.max()}')

#endregion

#region gene-card data #######################################################

# 6 exemplar genes. 5 anchors of the DOWN supply arm (carriers + biosynth)
# plus Srebf1 - the master sterol regulatory element binding factor - which
# is paradoxically UP across the same glia despite its downstream targets
# (Hmgcr, Idi1, Sqle) being DOWN. Srebf1 captures the transcriptional
# compensation arm: cells sense low intracellular cholesterol and try to
# rev biosynthesis back up, but the executors stay throttled.
CARD_GENES = ['Apoe', 'Fabp7', 'Hmgcr', 'Idi1', 'Srebf1', 'Mfsd2a']
PLATFORMS = ['slidetags', 'merfish', 'xenium']
# match 08_de_gsea_plots.py: cobalt / saffron / forest
PLATFORM_COLORS = {
    'slidetags': '#1C6CC6',
    'merfish':   '#E8A628',
    'xenium':    '#2F7F2E',
}
D_COLORS = {2: '#888888', 3: '#000000'}
sp_ds_labels = {'xenium': 'Xenium', 'merfish': 'MERFISH',
                'slidetags': 'Slide-tags'}

# load spatial data in backed mode, materialize only the obs we need + the
# gene columns. The xenium adata is 13GB so full load exhausts thread limits.
sp_coords = {}
sp_expr = {}              # {ds: {gene: log2(norm+1) array of length n_cells}}
sp_in = {}                # {ds: {gene: bool}}
# pct_nonzero_lists[(gene, cell_type)] = list of %nonzero per platform that
# has the gene. Used to build the Panel B dotplot size matrix (median across
# detecting platforms, across CTRL+PREG combined).
pct_nonzero_lists = {(g, ct): [] for g in ordered_genes for ct in ordered_cts}
for ds in PLATFORMS:
    a = sc.read_h5ad(
        f'{working_dir}/output/{ds}/03_adata_query_{ds}.h5ad',
        backed='r')
    var_names = a.var['gene_symbol'].astype(str).to_list() \
        if 'gene_symbol' in a.var.columns else list(a.var_names)
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

    # pre-compute cell-type masks (over kept cells) for the %nonzero pass
    ct_masks = {ct: (cell_types_kept == ct) for ct in ordered_cts}

    # cell-wise normalization scale (target_sum=1e4)
    # backed mode: read X by gene column for the gene cards. Per-cell totals
    # computed in chunks to keep memory bounded.
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

    # %nonzero for ALL Panel B genes (CTRL + PREG combined, this platform)
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

# build %nonzero matrix (n_g x n_c): median across platforms detecting the gene
pct_mat = np.full((n_g, n_c), np.nan)
for i, g in enumerate(ordered_genes):
    for j, ct in enumerate(ordered_cts):
        pcts = pct_nonzero_lists.get((g, ct), [])
        if pcts:
            pct_mat[i, j] = float(np.median(pcts))

# baseline ApoE expression per panel sender (slide-tags only; used for the
# annotation row above Panel C). mean log2(norm+1) Apoe across CTRL+PREG.
baseline_apoe = {}
if 'Apoe' in sp_expr['slidetags']:
    st_expr = sp_expr['slidetags']['Apoe']
    st_cts = sp_coords['slidetags']['cell_type']
    for ct in ordered_cts:
        m = (st_cts == ct)
        baseline_apoe[ct] = float(st_expr[m].mean()) if m.any() else np.nan
else:
    for ct in ordered_cts:
        baseline_apoe[ct] = np.nan
baseline_vec = np.array([baseline_apoe[ct] for ct in ordered_cts])
print(f'baseline ApoE per sender (slide-tags log2): '
      f'min={np.nanmin(baseline_vec):.2f} '
      f'max={np.nanmax(baseline_vec):.2f}')

de_pp_full = pl.read_csv(
    f'{working_dir}/output/themes/hits/de_per_platform.tsv', separator='\t')

def short_ct(ct):
    parts = ct.split(' ')
    if len(parts) <= 3:
        return ct
    return f'{parts[0]} {parts[1]} {parts[-1]}'

def class_rank(ct):
    if 'Glut' in ct:    return 0
    if 'NN' in ct:      return 2
    return 1

def pctl_range(arr, lo=5, hi=95):
    if len(arr) == 0:
        return 0.0, 1.0
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

def stars_for(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

def build_card(gene, ctx_cells=()):
    # cells: all panel-1 cells with a sumrank hit + optional validation context
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
            ep, D, stars = np.nan, 0, ''
        rows.append(dict(cell_type=ct, plat=plat_d, D=D,
                         meta_lfc=meta_lfc, emp_p=ep, stars=stars))

    # group by class (Glut, Gaba, NN); within class sort by |meta_lfc| desc
    rows.sort(key=lambda r: (class_rank(r['cell_type']),
                             -abs(r['meta_lfc'])
                             if not np.isnan(r['meta_lfc']) else 0))
    cts = [r['cell_type'] for r in rows]

    # spatial pair: prefer xenium+merfish; fall back to slidetags
    pref = [p for p in ('xenium', 'merfish', 'slidetags')
            if sp_in[p].get(gene, False)]
    sp_pair = pref[:2]

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

# Apoe gets 326 OPC + 334 Microglia as validation context cells
card_ctx = {
    'Apoe':   ['326 OPC NN', '334 Microglia NN'],
    'Fabp7':  [],
    'Hmgcr':  [],
    'Idi1':   [],
    'Srebf1': [],
    'Mfsd2a': [],
}
cards = {g: build_card(g, card_ctx[g]) for g in CARD_GENES}

# cap large forests so the per-cell label fontsize matches panel A/B (=7).
# row pitch CARD_FOREST_ROW_H=0.13" needs fontsize-7 line height 0.117"; with
# content_h_in=1.05" the max rows that fit is 8. Apoe has 13 hits - keep the
# 2 validation-context cells + top 6 by emp_p.
APOE_MAX_ROWS = 8
if len(cards['Apoe']['rows']) > APOE_MAX_ROWS:
    rows = cards['Apoe']['rows']
    forced = set(card_ctx['Apoe'])
    keep = [r for r in rows if r['cell_type'] in forced]
    others = [r for r in rows if r['cell_type'] not in forced]
    others.sort(key=lambda r: r['emp_p'] if not np.isnan(r['emp_p']) else 9)
    keep += others[:APOE_MAX_ROWS - len(keep)]
    # re-sort with the original (class_rank, |meta_lfc| desc) order
    keep.sort(key=lambda r: (class_rank(r['cell_type']),
                             -abs(r['meta_lfc'])
                             if not np.isnan(r['meta_lfc']) else 0))
    cards['Apoe']['rows'] = keep
    cards['Apoe']['cell_types'] = [r['cell_type'] for r in keep]

for g, c in cards.items():
    print(f'card {g}: {len(c["cell_types"])} cells, sp_pair={c["sp_pair"]}')

#endregion

#region combined figure ######################################################

# panel A (dotplot) + panel B (heatmap) share x columns - boxes aligned L/R
# 4 gene cards on the right span the combined A+B height
# cmap vmax for each panel = 10/90 quantile envelope of the panel's data
NES_VMAX = quant_vmax(nes_mat)
LFC_VMAX = quant_vmax(lfc_mat[ndet_mat >= 2])
LR_VMAX = quant_vmax(apoe_eff_mat)
print(f'cmap vmax (10/90 quantile): NES={NES_VMAX:.2f}  '
      f'logFC={LFC_VMAX:.2f}  LR={LR_VMAX:.3f}')

NLP_MIN, NLP_MAX = 1.30, 5.0
# unified dot size range across Panels A, B, C
SIZE_MIN, SIZE_MAX = 16.0, 80.0
N_TARG_MAX = 50.0
MISSING_FILL = '#ECECEC'
norm_nes = mpl.colors.Normalize(vmin=-NES_VMAX, vmax=NES_VMAX)
norm_lfc = mpl.colors.Normalize(vmin=-LFC_VMAX, vmax=LFC_VMAX)
norm_lr = mpl.colors.Normalize(vmin=-LR_VMAX, vmax=LR_VMAX)

def nlp_to_size(x):
    x = np.clip(x, NLP_MIN, NLP_MAX)
    f = (x - NLP_MIN) / (NLP_MAX - NLP_MIN)
    return SIZE_MIN + f * (SIZE_MAX - SIZE_MIN)

def n_to_size(n):
    n = max(1, min(int(n), int(N_TARG_MAX)))
    f = np.log2(n + 1) / np.log2(N_TARG_MAX + 1)
    return SIZE_MIN + f * (SIZE_MAX - SIZE_MIN)

def pct_to_size(p):
    p = float(np.clip(p, 0, 100))
    return SIZE_MIN + (p / 100.0) * (SIZE_MAX - SIZE_MIN)

# main-panel dimensions (inches)
LABEL_MARGIN_IN = 1.55                  # y-axis labels + axis title
LEFT_FIG_PAD_IN = 1.15                  # space to the left of y-axis margin
ax_left_in = LEFT_FIG_PAD_IN + LABEL_MARGIN_IN
ax_w_in = 0.20 * n_c                    # slightly narrower than before

# panel-A/B legend column - to the LEFT of the y-axis labels,
# vertically aligned with specific row ranges of Panel B
leg_left_in = 0.45                      # legends slightly more left
leg_w_in = 0.85                         # leaves clear gap before tick labels
PATH_PITCH = 0.21
GENE_PITCH = 0.155
REC_PITCH = 0.21                            # Panel C row pitch (dotplot)
ax_h_a_in = PATH_PITCH * n_rows
ax_h_b_in = GENE_PITCH * n_g
ax_h_c_in = REC_PITCH * n_rec
# gap_in holds Panel A's rotated numeric x-tick labels + "Cell type" title.
# gap_bc_in holds Panel B's labels + "Cell type" title + the baseline-ApoE
# annotation strip that sits just above Panel C.
gap_in = 0.50
gap_bc_in = 0.80

# gene-card column to the right of A+B; cards stacked vertically.
# Layout per card (left-to-right): SP1 | SP2 | FOREST | LABELS (left-just)
ANNO_W_IN = 0.07
ANNO_GAP_IN = 0.03

N_CARDS = len(CARD_GENES)
CARD_GAP_IN = 0.10
CARD_TITLE_H_IN = 0.30
# content_h_in is derived from the A-B vertical span so cards align EXACTLY:
# top of card-1 SP box at ax_a_top_in, bottom of card-N SP box at ax_b_bot_in
ab_vspan_in = ax_h_b_in + gap_in + ax_h_a_in
content_h_in = (ab_vspan_in
                - (N_CARDS - 1) * (CARD_TITLE_H_IN + CARD_GAP_IN)) / N_CARDS
SP_W_IN = content_h_in              # square spatial maps (W = H = content_h)
SP_GAP_IN = 0.05
SP_FOREST_GAP_IN = 0.10
FOREST_W_IN = 0.75
FOREST_LABEL_GAP_IN = 0.05
LABEL_W_IN = 0.95
CARD_W_IN = (SP_W_IN + SP_GAP_IN + SP_W_IN + SP_FOREST_GAP_IN
             + FOREST_W_IN + FOREST_LABEL_GAP_IN + LABEL_W_IN)
CARD_FOREST_ROW_H = 0.13
CARD_FOREST_MIN_H = 0.42
cards_left_in = ax_left_in + ax_w_in + ANNO_GAP_IN + ANNO_W_IN + 0.35

top_margin_in = 0.85         # extra space for gene-card titles above ax_a
bot_margin_in = 2.15         # x labels + Sender cell type + theme legend row
card_total_h_in = content_h_in + CARD_TITLE_H_IN
cards_h_in = (N_CARDS * card_total_h_in
              + (N_CARDS - 1) * CARD_GAP_IN)

# vertical forest legend column to the right of gene cards
FLEG_GAP_IN = 0.12           # gap between cards and forest legend column
FLEG_COL_W_IN = 1.10         # width of forest legend column

# mock IHC validation block - below the gene cards, same horizontal range
MOCK_IHC_GAP_IN = 0.30                # gap from bottom of cards to top of IHC
MOCK_IHC_SECTION_TITLE_H_IN = 0.26    # row for "IHC validation (mock)" title
MOCK_IHC_PANEL_TITLE_H_IN = 0.48      # per-panel title + Δ subtitle
MOCK_IHC_TITLE_H_IN = (MOCK_IHC_SECTION_TITLE_H_IN
                       + MOCK_IHC_PANEL_TITLE_H_IN)
MOCK_IHC_H_IN = 1.75                  # height of mock IHC panel image

fig_h = (top_margin_in + ax_h_a_in + gap_in + ax_h_b_in + gap_bc_in
         + ax_h_c_in + bot_margin_in)
fig_w = cards_left_in + CARD_W_IN + FLEG_GAP_IN + FLEG_COL_W_IN + 0.10

# axes bottoms
ax_c_bot_in = bot_margin_in
ax_c_top_in = ax_c_bot_in + ax_h_c_in
ax_b_bot_in = ax_c_top_in + gap_bc_in
ax_a_bot_in = ax_b_bot_in + ax_h_b_in + gap_in
ax_a_top_in = ax_a_bot_in + ax_h_a_in

fig = plt.figure(figsize=(fig_w, fig_h))
ax_a = fig.add_axes([ax_left_in / fig_w, ax_a_bot_in / fig_h,
                     ax_w_in / fig_w, ax_h_a_in / fig_h])
ax_b = fig.add_axes([ax_left_in / fig_w, ax_b_bot_in / fig_h,
                     ax_w_in / fig_w, ax_h_b_in / fig_h])
ax_c = fig.add_axes([ax_left_in / fig_w, ax_c_bot_in / fig_h,
                     ax_w_in / fig_w, ax_h_c_in / fig_h])

# ----- panel A: dotplot -----
xs, ys, sizes, colors, edges, lws = [], [], [], [], [], []
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
        edges.append('black' if d_mat[i, j] == 3 else 'none')
        lws.append(0.5 if d_mat[i, j] == 3 else 0.0)
ax_a.scatter(xs, ys, s=sizes, c=colors, edgecolors=edges,
             linewidths=lws, zorder=3)

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
# Panel A x-tick labels: subclass numeric prefix at 45 degrees
prefix_labels = [f'{numeric_prefix(ct):03d}' for ct in ordered_cts]
ax_a.set_xticks(range(n_cols))
ax_a.set_xticklabels(prefix_labels, rotation=45, ha='right', fontsize=7.5)
ax_a.tick_params(axis='x', length=2, pad=1)
ax_a.set_yticks(range(n_rows))
ax_a.set_yticklabels([pretty.get(p, p) for p in ordered_pathways],
                     fontsize=7.5)
ax_a.tick_params(axis='y', length=2, pad=1)
ax_a.set_ylabel('GSEA pathways', fontsize=9.0, labelpad=8)
for sp in ('top', 'right', 'left', 'bottom'):
    ax_a.spines[sp].set_visible(True)
    ax_a.spines[sp].set_color('black')
    ax_a.spines[sp].set_linewidth(0.9)

# class-group titles above panel A
trans_x_a = ax_a.get_xaxis_transform()
for cls, lo, hi in class_spans:
    ax_a.text((lo + hi) / 2, 1.04, cls,
              ha='center', va='bottom', fontsize=7.5,
              color='black', transform=trans_x_a, clip_on=False,
              linespacing=1.05)

# ----- panel B: dotplot (color = signed logFC, size = % nonzero) -----
# Size encodes % nonzero across CTRL+PREG cells (median across platforms
# detecting the gene). Significant cells get a black edge; cells with
# insufficient cross-platform detection (n_det < 2) get no dot.
ax_b.set_facecolor('white')
xs_b, ys_b, sizes_b, colors_b, edges_b, lws_b = [], [], [], [], [], []
for i in range(n_g):
    for j in range(n_c):
        if ndet_mat[i, j] < 2:
            continue
        pct = pct_mat[i, j]
        if np.isnan(pct):
            continue
        lfc = lfc_mat[i, j]
        # clip lfc to the new tighter vmax for proper colormap saturation
        lfc_clip = np.clip(lfc, -LFC_VMAX, LFC_VMAX)
        xs_b.append(j); ys_b.append(i)
        sizes_b.append(pct_to_size(pct))
        colors_b.append(cmap(norm_lfc(lfc_clip))
                        if not np.isnan(lfc) else '#CCCCCC')
        if sig_mat[i, j]:
            edges_b.append('black'); lws_b.append(0.5)
        else:
            edges_b.append('none'); lws_b.append(0.0)
ax_b.scatter(xs_b, ys_b, s=sizes_b, c=colors_b,
             edgecolors=edges_b, linewidths=lws_b, zorder=3)

for k in range(1, n_c):
    ax_b.axvline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
for k in range(1, n_g):
    ax_b.axhline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
for _, _, hi in class_spans[:-1]:
    ax_b.axvline(hi + 0.5, color='#BBBBBB', lw=0.4, zorder=2)
for _, _, hi in gene_band_spans[:-1]:
    ax_b.axhline(hi + 0.5, color='#BBBBBB', lw=0.4, zorder=2)

ax_b.set_xlim(-0.5, n_c - 0.5)
ax_b.set_ylim(n_g - 0.5, -0.5)
# Panel B x-tick labels: same numeric-prefix scheme as Panel A
ax_b.set_xticks(range(n_c))
ax_b.set_xticklabels(prefix_labels, rotation=45, ha='right', fontsize=7.5)
ax_b.tick_params(axis='x', length=2, pad=1)
ax_b.set_yticks(range(n_g))
ax_b.set_yticklabels(ordered_genes, fontsize=7.5, fontstyle='italic')
ax_b.tick_params(axis='y', length=2, pad=1)
ax_b.set_ylabel('Leading edge genes', fontsize=9.0, labelpad=8)
for sp in ('top', 'right', 'left', 'bottom'):
    ax_b.spines[sp].set_visible(True)
    ax_b.spines[sp].set_color('black')
    ax_b.spines[sp].set_linewidth(0.9)

# ----- panel C: Apoe outgoing signaling (LIANA, slide-tags only) -----
xs_c, ys_c, sizes_c, colors_c = [], [], [], []
for i in range(n_rec):
    for j in range(n_c):
        if np.isnan(apoe_eff_mat[i, j]):
            continue
        xs_c.append(j); ys_c.append(i)
        sizes_c.append(n_to_size(apoe_n_mat[i, j]))
        eff_clip = np.clip(apoe_eff_mat[i, j], -LR_VMAX, LR_VMAX)
        colors_c.append(cmap(norm_lr(eff_clip)))
ax_c.scatter(xs_c, ys_c, s=sizes_c, c=colors_c,
             edgecolors='none', zorder=3)

for k in range(1, n_c):
    ax_c.axvline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
for k in range(1, n_rec):
    ax_c.axhline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
for _, _, hi in class_spans[:-1]:
    ax_c.axvline(hi + 0.5, color='#BBBBBB', lw=0.4, zorder=2)

ax_c.set_xlim(-0.5, n_c - 0.5)
ax_c.set_ylim(n_rec - 0.5, -0.5)
ax_c.set_xticks(range(n_c))
ax_c.set_xticklabels(ordered_cts, rotation=45, ha='right', fontsize=7.5)
ax_c.tick_params(axis='x', length=2, pad=1)
ax_c.set_yticks(range(n_rec))
ax_c.set_yticklabels([f'ApoE → {r}' for r in APOE_RECEPTORS],
                     fontsize=7.5)
ax_c.tick_params(axis='y', length=2, pad=1)
ax_c.set_ylabel('ApoE → Receptor\nsignaling (Slide-tag)',
                fontsize=9.0, labelpad=8)
for sp in ('top', 'right', 'left', 'bottom'):
    ax_c.spines[sp].set_visible(True)
    ax_c.spines[sp].set_color('black')
    ax_c.spines[sp].set_linewidth(0.9)

# Three x-axis titles, one per panel:
# A: "Cell type" below A's numeric-prefix tick labels
# B: "Cell type" below B's numeric-prefix tick labels
# C: "Sender cell type" below C's rotated full tick labels
AX_LEFT, AX_RIGHT = ax_left_in / fig_w, (ax_left_in + ax_w_in) / fig_w
ax_x_mid_fig = 0.5 * (AX_LEFT + AX_RIGHT)

# A title - directly under A's horizontal numeric labels
a_title_y_in = ax_a_bot_in - 0.22
fig.text(ax_x_mid_fig, a_title_y_in / fig_h,
         'Cell type', ha='center', va='top',
         fontsize=9.0, color='black')

# B title - directly under B's horizontal numeric labels
b_title_y_in = ax_b_bot_in - 0.22
fig.text(ax_x_mid_fig, b_title_y_in / fig_h,
         'Cell type', ha='center', va='top',
         fontsize=9.0, color='black')

# C title - under C's 45 degree rotated full-label tick labels
# (longest label ~22 chars at fontsize 7.5 extends ~0.85 in below tick)
c_title_y_in = ax_c_bot_in - 1.00
fig.text(ax_x_mid_fig, c_title_y_in / fig_h,
         'Sender cell type', ha='center', va='top',
         fontsize=9.0, color='black')

# ----- baseline ApoE annotation strip (above Panel C) -----
# 1-row strip per sender cell type colored by slide-tags mean log2 ApoE
# expression (CTRL+PREG combined). Gives the reader a per-column trust gauge
# for Panel C: senders with low baseline are the ones where a uniform-down
# LR signal is a ligand-side echo rather than receptor-specific biology.
BASELINE_H_IN = 0.20
baseline_bot_in = ax_c_top_in + 0.08
baseline_top_in = baseline_bot_in + BASELINE_H_IN
baseline_finite = baseline_vec[~np.isnan(baseline_vec)]
baseline_norm = mpl.colors.Normalize(
    vmin=float(baseline_finite.min()) if baseline_finite.size else 0.0,
    vmax=float(baseline_finite.max()) if baseline_finite.size else 1.0)
baseline_cmap = plt.get_cmap('viridis')

ax_baseline = fig.add_axes([ax_left_in / fig_w, baseline_bot_in / fig_h,
                            ax_w_in / fig_w, BASELINE_H_IN / fig_h])
ax_baseline.set_xlim(-0.5, n_c - 0.5)
ax_baseline.set_ylim(-0.5, 0.5)
ax_baseline.set_xticks([]); ax_baseline.set_yticks([])
for sp in ax_baseline.spines.values():
    sp.set_linewidth(0.5)
for j, ct in enumerate(ordered_cts):
    val = baseline_apoe[ct]
    if np.isnan(val):
        color = '#DDDDDD'
    else:
        color = baseline_cmap(baseline_norm(val))
    ax_baseline.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, 1,
                                         facecolor=color, edgecolor='none'))
# class-group separators on the baseline strip
for _, _, hi in class_spans[:-1]:
    ax_baseline.axvline(hi + 0.5, color='white', lw=0.5, zorder=4)
# y-axis label for the strip
fig.text((ax_left_in - 0.05) / fig_w,
         (baseline_bot_in + BASELINE_H_IN / 2) / fig_h,
         'ApoE\nbaseline', ha='right', va='center', fontsize=7.5,
         color='black', linespacing=1.05)

# tiny vertical colorbar to the right of the baseline strip
baseline_cbar_w_in = 0.05
baseline_cbar_x_in = ax_left_in + ax_w_in + ANNO_GAP_IN
ax_baseline_cbar = fig.add_axes([baseline_cbar_x_in / fig_w,
                                  baseline_bot_in / fig_h,
                                  baseline_cbar_w_in / fig_w,
                                  BASELINE_H_IN / fig_h])
cb_baseline = fig.colorbar(
    mpl.cm.ScalarMappable(norm=baseline_norm, cmap=baseline_cmap),
    cax=ax_baseline_cbar, orientation='vertical')
cb_baseline.set_ticks([baseline_norm.vmin, baseline_norm.vmax])
cb_baseline.set_ticklabels(
    [f'{baseline_norm.vmin:.1f}', f'{baseline_norm.vmax:.1f}'])
cb_baseline.ax.tick_params(labelsize=5.5, length=1.5, pad=1)
fig.text((baseline_cbar_x_in + baseline_cbar_w_in + 0.18) / fig_w,
         (baseline_bot_in + BASELINE_H_IN + 0.02) / fig_h,
         r'log$_2$ ApoE', ha='left', va='bottom', fontsize=6.5,
         color='#555555')

# ----- band-color annotation strips (right of A + B) -----
ANNO_X_IN = ax_left_in + ax_w_in + ANNO_GAP_IN

# Panel A annotation
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

# Panel B annotation
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

# ----- legend for panel A (vertically aligned with TOP rows of Panel B,
# rows 0-12 = Fabp7 through Dpm1; extends 2 rows past Abca8b for breathing
# room between the size scale, NES colorbar, and Platforms sections) -----
ax_b_top_in = ax_b_bot_in + ax_h_b_in
LEG_A_TOP_IN = ax_b_top_in                           # = row 0 top edge
LEG_A_BOT_IN = ax_b_top_in - 12 * GENE_PITCH         # = row 12 bottom edge
LEG_A_H_IN = LEG_A_TOP_IN - LEG_A_BOT_IN

ax_leg_a = fig.add_axes([leg_left_in / fig_w, LEG_A_BOT_IN / fig_h,
                         leg_w_in / fig_w, LEG_A_H_IN / fig_h],
                        zorder=100)
ax_leg_a.set_xlim(0, 1); ax_leg_a.set_ylim(0, 1)
ax_leg_a.set_xticks([]); ax_leg_a.set_yticks([])
ax_leg_a.set_facecolor('white')
ax_leg_a.patch.set_facecolor('white')
ax_leg_a.patch.set_alpha(1.0)
ax_leg_a.patch.set_zorder(99)
for s in ax_leg_a.spines.values():
    s.set_visible(False)

# size scale (right-justified, dots at axes-x 0.95 to align with cbar below)
ax_leg_a.text(1.0, 0.97, r'$-\log_{10}\,p_\mathrm{emp}$',
              ha='right', va='top', fontsize=6.8)
for k, lev in enumerate([1.5, 2.5, 4.0]):
    y = 0.88 - k * 0.08
    ax_leg_a.scatter([0.95], [y], s=nlp_to_size(lev),
                     c=['#555555'], edgecolors='none')
    ax_leg_a.text(0.85, y, f'{lev:.1f}', ha='right', va='center', fontsize=6.5)

# NES colorbar - thin vertical, centered at axes-x 0.95
ax_leg_a.text(1.0, 0.62, 'NES (median)', ha='right', va='top', fontsize=6.8)
CBAR_W_FIG = 0.005                    # thin colorbar
# center colorbar at the same axes-x (~0.95) as the marker dots
CBAR_TARGET_X_AXES = 0.95
cbar_center_a_fig = (leg_left_in + CBAR_TARGET_X_AXES * leg_w_in) / fig_w
cbar_a_x_fig = cbar_center_a_fig - CBAR_W_FIG / 2
cbar_ax_a = fig.add_axes([cbar_a_x_fig,
                          LEG_A_BOT_IN / fig_h + LEG_A_H_IN / fig_h * 0.33,
                          CBAR_W_FIG,
                          LEG_A_H_IN / fig_h * 0.22],
                         zorder=110)
cb_a = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_nes, cmap=cmap),
                    cax=cbar_ax_a, orientation='vertical')
cb_a.set_ticks([-NES_VMAX, 0, NES_VMAX])
cb_a.ax.yaxis.tick_left()           # ticks/labels on the left
cb_a.ax.tick_params(labelsize=6.0, length=2, pad=1)
cbar_ax_a.set_zorder(110)

# platforms (dots at axes-x 0.95 to align with cbar + size dots)
ax_leg_a.text(1.0, 0.24, 'Platforms', ha='right', va='top', fontsize=6.8)
ax_leg_a.scatter([0.95], [0.14], s=40, c=['lightgray'],
                 edgecolors='black', linewidths=0.5)
ax_leg_a.text(0.85, 0.14, 'D=3', ha='right', va='center', fontsize=6.5)
ax_leg_a.scatter([0.95], [0.04], s=40, c=['lightgray'], edgecolors='none')
ax_leg_a.text(0.85, 0.04, 'D=2', ha='right', va='center', fontsize=6.5)

# ----- legend for panel B (vertically aligned with BOTTOM rows of Panel B,
# rows 21-29 = Acaa2 to Mfsd2a; same x column as Panel A legend) -----
# row 21 top edge ... row 30 bottom edge = ax_b_bot_in
LEG_B_TOP_IN = ax_b_top_in - 21 * GENE_PITCH
LEG_B_BOT_IN = ax_b_bot_in
LEG_B_H_IN = LEG_B_TOP_IN - LEG_B_BOT_IN
ax_leg_b = fig.add_axes([leg_left_in / fig_w, LEG_B_BOT_IN / fig_h,
                         leg_w_in / fig_w, LEG_B_H_IN / fig_h],
                        zorder=100)
ax_leg_b.set_xlim(0, 1); ax_leg_b.set_ylim(0, 1)
ax_leg_b.set_xticks([]); ax_leg_b.set_yticks([])
ax_leg_b.set_facecolor('white')
ax_leg_b.patch.set_facecolor('white')
ax_leg_b.patch.set_alpha(1.0)
ax_leg_b.patch.set_zorder(99)
for s in ax_leg_b.spines.values():
    s.set_visible(False)

# median logFC colorbar - thin vertical, centered at axes-x 0.95
ax_leg_b.text(1.0, 0.97, 'logFC (median)', ha='right', va='top', fontsize=6.8)
cbar_center_b_fig = (leg_left_in + CBAR_TARGET_X_AXES * leg_w_in) / fig_w
cbar_b_x_fig = cbar_center_b_fig - CBAR_W_FIG / 2
cbar_ax_b = fig.add_axes([cbar_b_x_fig,
                          LEG_B_BOT_IN / fig_h + LEG_B_H_IN / fig_h * 0.72,
                          CBAR_W_FIG,
                          LEG_B_H_IN / fig_h * 0.21],
                         zorder=110)
cb_b = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_lfc, cmap=cmap),
                    cax=cbar_ax_b, orientation='vertical')
cb_b.set_ticks([-LFC_VMAX, 0, LFC_VMAX])
cb_b.ax.yaxis.tick_left()
cb_b.ax.tick_params(labelsize=6.0, length=2, pad=1)
cbar_ax_b.set_zorder(110)

# % nonzero size scale - dots at axes-x 0.95 (aligned with cbar above)
ax_leg_b.text(1.0, 0.62, '% nonzero', ha='right', va='top', fontsize=6.8)
for k, lev in enumerate([10, 50, 90]):
    y = 0.52 - k * 0.08
    ax_leg_b.scatter([0.95], [y], s=pct_to_size(lev),
                     c=['#777777'], edgecolors='none')
    ax_leg_b.text(0.85, y, f'{lev}%',
                  ha='right', va='center', fontsize=6.5)

# significance: black edge
ax_leg_b.scatter([0.95], [0.22], s=pct_to_size(50),
                 c=['lightgray'], edgecolors='black', linewidths=0.5)
ax_leg_b.text(0.85, 0.22, r'emp$\,p\leq 0.05$',
              ha='right', va='center', fontsize=6.5)

# ===== gene cards (right column) =====
LABEL_FS = 7.5                        # cell-type labels (match panel A/B)
AXIS_FS = 7.5                         # match panel A/B x-axis label fontsize
TITLE_FS = 8.5                        # gene title (left-justified)
JIT = 0.18                            # platform jitter for forest

def draw_forest(axf, gd, show_xlabel=False):
    n = len(gd['rows'])
    axf.set_ylim(n - 0.5, -0.5)
    axf.axvline(0, color='grey', lw=0.4, zorder=1)
    # x-range from data
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
    # ticks: x-axis logFC labels on every subpanel; y-axis labels rendered
    # manually outside the forest. Only the bottom card carries the "logFC"
    # axis title to avoid repetition.
    axf.set_yticks([])
    axf.tick_params(axis='x', labelsize=AXIS_FS, length=1.5, pad=1)
    if show_xlabel:
        axf.set_xlabel('logFC', fontsize=AXIS_FS, labelpad=1)
    for sp in axf.spines.values():
        sp.set_linewidth(0.5)

# 5 cards stacked top-down; SHIFT UP so top of the first spatial box aligns
# with the top of the dotplot (ax_a_top_in). The gene title sits above that.
cards_top_in = ax_a_top_in + CARD_TITLE_H_IN
for i, g in enumerate(CARD_GENES):
    gd = cards[g]
    card_top_in = cards_top_in - i * (card_total_h_in + CARD_GAP_IN)
    title_bot_in = card_top_in - CARD_TITLE_H_IN
    content_bot_in = title_bot_in - content_h_in

    # gene title - LEFT-justified at card left edge
    fig.text(cards_left_in / fig_w,
             (card_top_in - 0.02) / fig_h, g,
             ha='left', va='top', fontsize=TITLE_FS,
             fontstyle='italic')

    # 2 spatial maps (LEFT side of card), fill content area
    sp_bot_in = content_bot_in
    for si in range(2):
        if si >= len(gd['sp_pair']):
            continue
        sp_left_in = cards_left_in + si * (SP_W_IN + SP_GAP_IN)
        ax_sp = fig.add_axes([sp_left_in / fig_w, sp_bot_in / fig_h,
                              SP_W_IN / fig_w, SP_W_IN / fig_h])
        ds = gd['sp_pair'][si]
        ax_sp.set_title(sp_ds_labels[ds], fontsize=AXIS_FS, pad=1.5)
        ax_sp.set_xticks([]); ax_sp.set_yticks([])
        ax_sp.set_facecolor('black')
        for sp in ax_sp.spines.values():
            sp.set_linewidth(0.5)
        sd = gd['sp_data'][si]
        if sd is None or len(sd['x']) == 0:
            continue
        vmin_sp, vmax_sp = gd['sp_vranges'][si]
        order = np.argsort(sd['expr'])
        # scale point size inversely with sqrt of total cells on this platform
        # so total visual coverage is comparable across the 3 datasets
        # (slidetags is much sparser than xenium/merfish)
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

    # forest plot (after the 2 spatial maps), centered vertically
    n_cells = len(gd['rows'])
    forest_h_in = max(n_cells * CARD_FOREST_ROW_H, CARD_FOREST_MIN_H)
    forest_h_in = min(forest_h_in, content_h_in)
    forest_bot_in = content_bot_in + (content_h_in - forest_h_in) / 2
    forest_left_in = (cards_left_in + 2 * SP_W_IN + SP_GAP_IN
                      + SP_FOREST_GAP_IN)
    ax_f = fig.add_axes([forest_left_in / fig_w, forest_bot_in / fig_h,
                         FOREST_W_IN / fig_w, forest_h_in / fig_h])
    draw_forest(ax_f, gd, show_xlabel=(i == N_CARDS - 1))

    # cell-type labels: LEFT-justified, outside forest right edge
    label_left_frac = 1.0 + FOREST_LABEL_GAP_IN / FOREST_W_IN
    trans_y_f = ax_f.get_yaxis_transform()
    for ri, r in enumerate(gd['rows']):
        text = f"{short_ct(r['cell_type'])} {r['stars']}".strip()
        ax_f.text(label_left_frac, ri, text,
                  transform=trans_y_f, ha='left', va='center',
                  fontsize=LABEL_FS, clip_on=False)

# ===== vertical forest legend column (to the RIGHT of gene cards) =====
# compact: anchored near top of cards, height fits content only
cards_bottom_in = ax_a_top_in - cards_h_in
flegv_left_in = cards_left_in + CARD_W_IN + FLEG_GAP_IN
flegv_top_in = cards_top_in - 0.10
flegv_h_in = 1.95                            # fits content with breathing room
flegv_bot_in = flegv_top_in - flegv_h_in
ax_fleg = fig.add_axes([flegv_left_in / fig_w, flegv_bot_in / fig_h,
                        FLEG_COL_W_IN / fig_w, flegv_h_in / fig_h])
ax_fleg.set_xlim(0, 1); ax_fleg.set_ylim(0, 1)
ax_fleg.set_xticks([]); ax_fleg.set_yticks([])
for s in ax_fleg.spines.values():
    s.set_visible(False)

FLEG_FS = 6.5
FLEG_HEADER_FS = 6.8
# single vertical stack: 3 sections (Platforms, Meta, Significance)
sections = [
    ('Platforms', [
        ('line', 'slidetags', sp_ds_labels['slidetags']),
        ('line', 'merfish',   sp_ds_labels['merfish']),
        ('line', 'xenium',    sp_ds_labels['xenium']),
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
# compact axes-y spacing (slightly looser than minimum to avoid crush)
header_h, item_h, section_gap = 0.075, 0.060, 0.050
y = 0.97
for s_i, (header, items) in enumerate(sections):
    ax_fleg.text(0.05, y, header, ha='left', va='top',
                 fontsize=FLEG_HEADER_FS,
                 transform=ax_fleg.transAxes)
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

# ===== Mock IHC validation panels (below gene cards) =====
# Targets and effect sizes from cholesterol_validation_targets_final.pdf:
# - ApoE in Pdgfra+ OPC: ~-43% (KEEP, pseudobulk+per-cell concordant)
# - ApoE in Iba1+ microglia: ~-43% (KEEP, same as OPC)
# - HMGCR in Aldh1l1+ astrocytes: ~-20% (KEEP, pilot first)
def draw_mock_ihc(ax, marker_color, drop_frac, seed=0,
                  n_cells_side=85, marker_frac=0.22):
    """Render a CTRL | PREG mock IHC panel. Left half = CTRL, right = PREG;
    target signal (red) is multiplied by (1 - drop_frac) in PREG."""
    rng = np.random.default_rng(seed)
    ax.set_facecolor('black')
    ax.set_xlim(0, 2.0); ax.set_ylim(0, 1.0)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_edgecolor('white')

    for x0, intensity_mult in [(0.0, 1.0), (1.0, 1 - drop_frac)]:
        # DAPI: all cells (blue)
        xs = rng.uniform(x0 + 0.04, x0 + 0.96, n_cells_side)
        ys = rng.uniform(0.04, 0.96, n_cells_side)
        ax.scatter(xs, ys, c='#1F4FBC', s=10, alpha=0.55,
                   edgecolors='none', zorder=1)

        # marker+ cells (subset) - colored outline
        is_mark = rng.random(n_cells_side) < marker_frac
        mx, my = xs[is_mark], ys[is_mark]
        ax.scatter(mx, my, facecolors='none', edgecolors=marker_color,
                   s=30, linewidths=0.7, zorder=2)

        # target gene signal (red) co-localized with marker+ cells;
        # alpha encodes intensity
        base_int = rng.uniform(0.55, 1.0, len(mx))
        intensity = np.clip(base_int * intensity_mult, 0, 1)
        for cx, cy, ai in zip(mx, my, intensity):
            ax.scatter([cx], [cy], c=[(1.0, 0.25, 0.25, float(ai))],
                       s=46, edgecolors='none', zorder=3)

    # CTRL/PREG midline
    ax.plot([1.0, 1.0], [0, 1], color='white', lw=1.4, alpha=0.9, zorder=4)
    ax.text(0.25, 0.97, 'CTRL', transform=ax.transAxes,
            ha='center', va='top', color='white',
            fontsize=AXIS_FS - 1.5, zorder=5)
    ax.text(0.75, 0.97, 'PREG', transform=ax.transAxes,
            ha='center', va='top', color='white',
            fontsize=AXIS_FS - 1.5, zorder=5)


ihc_targets = [
    {'gene': 'ApoE',  'marker': 'Pdgfra$^+$ OPC',
     'drop': 0.43, 'marker_color': '#3FFF3F', 'seed': 1},
    {'gene': 'ApoE',  'marker': 'Iba1$^+$ microglia',
     'drop': 0.43, 'marker_color': '#FF66E0', 'seed': 2},
    {'gene': 'HMGCR', 'marker': 'Aldh1l1$^+$ astrocytes',
     'drop': 0.20, 'marker_color': '#FFD040', 'seed': 3},
]
ihc_top_in = cards_bottom_in - MOCK_IHC_GAP_IN
# section title row at the very top (own row, well above panel titles)
ihc_section_title_y_in = ihc_top_in - 0.06    # text top
# per-panel title row below the section title
ihc_panel_title_y_in = ihc_top_in - MOCK_IHC_SECTION_TITLE_H_IN - 0.06
# panel image
ihc_panel_top_in = ihc_top_in - MOCK_IHC_TITLE_H_IN
ihc_panel_bot_in = ihc_panel_top_in - MOCK_IHC_H_IN

n_ihc = len(ihc_targets)
IHC_GAP_IN = 0.10
ihc_panel_w_in = (CARD_W_IN - (n_ihc - 1) * IHC_GAP_IN) / n_ihc

# overall section title (left-justified, not bold)
fig.text(cards_left_in / fig_w,
         ihc_section_title_y_in / fig_h,
         'IHC validation (mock)', ha='left', va='top',
         fontsize=TITLE_FS)

for k, tg in enumerate(ihc_targets):
    px_in = cards_left_in + k * (ihc_panel_w_in + IHC_GAP_IN)
    # per-panel title (gene + marker), centered above panel
    fig.text((px_in + ihc_panel_w_in / 2) / fig_w,
             ihc_panel_title_y_in / fig_h,
             tg['gene'] + r'$\,$in$\,$' + tg['marker'],
             ha='center', va='top', fontsize=AXIS_FS, fontstyle='italic')
    # Δ subtitle just above panel
    fig.text((px_in + ihc_panel_w_in / 2) / fig_w,
             (ihc_panel_top_in + 0.04) / fig_h,
             rf'$\Delta\sim {int(-tg["drop"] * 100)}\%$',
             ha='center', va='bottom', fontsize=AXIS_FS - 1,
             color='#444444')
    ax_ihc = fig.add_axes([px_in / fig_w, ihc_panel_bot_in / fig_h,
                           ihc_panel_w_in / fig_w,
                           MOCK_IHC_H_IN / fig_h])
    draw_mock_ihc(ax_ihc, tg['marker_color'], tg['drop'], seed=tg['seed'])

# IHC channel legend (right of the 3 mock panels, vertical stack)
ihc_leg_left_in = cards_left_in + CARD_W_IN + FLEG_GAP_IN
ihc_leg_top_in = ihc_panel_top_in
ihc_leg_bot_in = ihc_panel_bot_in
ihc_leg_h_in = ihc_leg_top_in - ihc_leg_bot_in
ax_ihc_leg = fig.add_axes([ihc_leg_left_in / fig_w,
                           ihc_leg_bot_in / fig_h,
                           FLEG_COL_W_IN / fig_w,
                           ihc_leg_h_in / fig_h])
ax_ihc_leg.set_xlim(0, 1); ax_ihc_leg.set_ylim(0, 1)
ax_ihc_leg.set_xticks([]); ax_ihc_leg.set_yticks([])
for s in ax_ihc_leg.spines.values():
    s.set_visible(False)

# non-bold channel header
ax_ihc_leg.text(0.05, 0.95, 'Channels', ha='left', va='top',
                fontsize=FLEG_HEADER_FS,
                transform=ax_ihc_leg.transAxes)
ihc_channels = [
    ('#1F4FBC', 'DAPI'),
    ('#3FFF3F', 'Pdgfra (OPC)'),
    ('#FF66E0', 'Iba1 (microglia)'),
    ('#FFD040', 'Aldh1l1 (astrocyte)'),
    ((1.0, 0.25, 0.25, 1.0), 'ApoE / HMGCR'),
]
ch_y = 0.83
for color, label in ihc_channels:
    ax_ihc_leg.scatter(0.09, ch_y, c=[color], s=30, edgecolors='none',
                       transform=ax_ihc_leg.transAxes)
    ax_ihc_leg.text(0.17, ch_y, label, ha='left', va='center',
                    fontsize=FLEG_FS, transform=ax_ihc_leg.transAxes)
    ch_y -= 0.09

# ----- legend for panel C (aligned with Panel C rows, same column) -----
LEG_C_TOP_IN = ax_c_top_in
LEG_C_BOT_IN = ax_c_bot_in
LEG_C_H_IN = LEG_C_TOP_IN - LEG_C_BOT_IN
ax_leg_c = fig.add_axes([leg_left_in / fig_w, LEG_C_BOT_IN / fig_h,
                         leg_w_in / fig_w, LEG_C_H_IN / fig_h],
                        zorder=100)
ax_leg_c.set_xlim(0, 1); ax_leg_c.set_ylim(0, 1)
ax_leg_c.set_xticks([]); ax_leg_c.set_yticks([])
ax_leg_c.set_facecolor('white')
ax_leg_c.patch.set_facecolor('white')
ax_leg_c.patch.set_alpha(1.0)
ax_leg_c.patch.set_zorder(99)
for s in ax_leg_c.spines.values():
    s.set_visible(False)

# size scale: number of distinct receiver subclasses with above-floor
# (sender's ApoE × receiver's R) coupling shift
ax_leg_c.text(1.0, 0.97, 'N receiver cell types',
              ha='right', va='top', fontsize=6.8)
for k, lev in enumerate([2, 10, 40]):
    y = 0.85 - k * 0.10
    ax_leg_c.scatter([0.95], [y], s=n_to_size(lev),
                     c=['#555555'], edgecolors='none')
    ax_leg_c.text(0.85, y, f'{lev}', ha='right', va='center', fontsize=6.5)

# colorbar: change in (ligand × receptor) product, preg − ctrl, averaged
# across the above-floor receiver subclasses
ax_leg_c.text(1.0, 0.45, r'$\Delta$LR',
              ha='right', va='top', fontsize=6.8)
cbar_center_c_fig = (leg_left_in + CBAR_TARGET_X_AXES * leg_w_in) / fig_w
cbar_c_x_fig = cbar_center_c_fig - CBAR_W_FIG / 2
cbar_ax_c = fig.add_axes([cbar_c_x_fig,
                          LEG_C_BOT_IN / fig_h + LEG_C_H_IN / fig_h * 0.08,
                          CBAR_W_FIG,
                          LEG_C_H_IN / fig_h * 0.28],
                         zorder=110)
cb_c = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_lr, cmap=cmap),
                    cax=cbar_ax_c, orientation='vertical')
cb_c.set_ticks([-LR_VMAX, 0, LR_VMAX])
cb_c.set_ticklabels([f'{-LR_VMAX:+.2f}', '0', f'{LR_VMAX:+.2f}'])
cb_c.ax.yaxis.tick_left()
cb_c.ax.tick_params(labelsize=6.0, length=2, pad=1)
cbar_ax_c.set_zorder(110)

# ===== vertical THEME legend (5 stacked chips + names) =====
# placed below LEG_C in the left column (Panel C now occupies the space
# previously used for theme legend below LEG_B)
TLEG_TOP_IN = LEG_C_BOT_IN - 0.18            # gap below Panel C legend
TLEG_H_IN = 0.75                             # tighter
TLEG_BOT_IN = TLEG_TOP_IN - TLEG_H_IN
ax_tleg = fig.add_axes([leg_left_in / fig_w, TLEG_BOT_IN / fig_h,
                        leg_w_in / fig_w, TLEG_H_IN / fig_h],
                       zorder=100)
ax_tleg.set_xlim(0, 1); ax_tleg.set_ylim(0, 1)
ax_tleg.set_xticks([]); ax_tleg.set_yticks([])
ax_tleg.set_facecolor('white')
ax_tleg.patch.set_facecolor('white')
ax_tleg.patch.set_alpha(1.0)
for s in ax_tleg.spines.values():
    s.set_visible(False)
band_names = list(BAND_COLORS.keys())
n_themes = len(band_names)
y_top, y_bot = 0.92, 0.08
y_step = (y_top - y_bot) / (n_themes - 1)
for i, band in enumerate(band_names):
    y = y_top - i * y_step
    # chips on the right (right edge at axes-x 0.95, matching dots in
    # LEG_A/B/C); names right-justified to their left
    ax_tleg.add_patch(plt.Rectangle((0.85, y - 0.07), 0.10, 0.14,
                                     facecolor=BAND_COLORS[band],
                                     edgecolor='none'))
    ax_tleg.text(0.82, y, band, ha='right', va='center',
                 fontsize=6.8, color='black')

for ext in ('png', 'pdf', 'svg'):
    fig.savefig(f'{out_dir}/lipid_combined.{ext}',
                bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f'wrote {out_dir}/lipid_combined.png/pdf/svg')

#endregion
