"""Correspondence between the MPOA cell types of Moffitt et al. (2018) and
the Allen subclasses used here (Supplementary Fig. 6D).

Their dissociated POA cells are labelled with Allen subclasses by the same
expression-based transfer as the main pipeline, minus the spatial constraint
(dissociated data has no coordinates). Each Moffitt cluster is then scored by
the subclasses its cells receive, and each subclass is marked by whether we
detect it and whether it is one of the nineteen maternal-circuit subclasses.

The reference is restricted to the divisions a POA dissection can contain, not
to the subclasses we detect: a cluster has to be free to land on a subclass we
miss, or the comparison answers nothing.
"""
import os
import faiss
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import scanorama
import scipy.sparse as sp
import seaborn as sns  # registers the 'rocket' colormaps
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from brisc import SingleCell

working_dir = '/home/karbabi/spatial-pregnancy'
moffitt_dir = '/home/karbabi/single-cell/Kalish/GSE113576'
ref_path = '/home/karbabi/single-cell/ABC/zeng_combined_10Xv3.h5ad'
out_dir = f'{working_dir}/output/moffitt'
fig_dir = f'{working_dir}/figures'
os.makedirs(out_dir, exist_ok=True)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

# HY covers the preoptic hypothalamus, PAL the BST/NDB/SI margins of the
# dissection, and the rest the neighbouring telencephalic structures
REF_DIVISIONS = ['HY', 'PAL', 'STR', 'sAMY', 'OLF', 'CTXsp']  # None = all
# plain PCA leaves the two datasets disjoint (their clouds sat 40 units apart
# with 0.4% of a query cell's neighbours in the reference), so Harmony has no
# shared clusters to correct within. Scanorama on the HVGs plus a strong theta
# is what every integration in 04_project_cell_types.py uses.
N_PCS = 30
HARMONY_KWARGS = dict(theta=12, alpha=0.05, tolerance=0.001,
                      max_iterations=20)
MAX_PER_SUBCLASS = 2000           # caps the reference; None to disable
K_NEIGHBORS = 20                  # as in the main annotation
CONF_MIN, MARGIN_MIN = 0.6, 0.2   # thresholds of the main annotation
MIN_FRAC = 0.15                   # a subclass is shown if it takes this
                                  # fraction of any cluster
MAX_COLS = 40                     # remaining subclasses pool into 'Other'
FIG_NEURONAL_ONLY = True          # glia stay in the summary table
MIN_CELLS, MIN_PLATFORMS = 100, 2  # detection rule, as in 11_figure_2.py

dataset_names = ['slidetags', 'merfish', 'xenium']

# the nineteen maternal-circuit subclasses of Supplementary Fig. 6
SUB19 = [
    '085 SI-MPO-LPO Lhx8 Gaba', '086 MPO-ADP Lhx8 Gaba',
    '106 PVpo-VMPO-MPN Hmx2 Gaba', '118 ADP-MPO Trp73 Glut',
    '124 MPN-MPO-PVpo Hmx2 Glut', '073 MEA-BST Sox6 Gaba',
    '075 MEA-BST Lhx6 Nr2e1 Gaba', '076 MEA-BST Lhx6 Nfib Gaba',
    '080 CEA-AAA-BST Six3 Sp9 Gaba', '082 CEA-BST Ebf1 Pdyn Gaba',
    '067 LSX Sall3 Pax6 Gaba', '068 LSX Otx2 Gaba', '069 LSX Nkx2-1 Gaba',
    '071 LSX Prdm12 Zeb2 Gaba', '057 NDB-SI-MA-STRv Lhx8 Gaba',
    '066 NDB-SI-ant Prdm12 Gaba', '060 OT D3 Folh1 Gaba',
    '114 COAa-PAA-MEA Barhl2 Glut', '119 SI-MA-LPO-LHA Skor1 Glut',
]

# =============================================================================
# Integration (cached: this is the expensive step)
# =============================================================================
# neurons and glia are integrated separately. The reference subset is mostly
# glia (125k OPC-Oligo, 100k Astro-Epen) while the query is 61% neurons, and
# Harmony corrects per cluster, so the composition mismatch drags whole
# populations across compartments: run jointly, the neurons collapse onto a
# rare immune subclass while the glia map correctly.
PARTS = {
    'neuronal': (
        ~pl.col('subclass').cast(pl.String).str.ends_with(' NN'),
        pl.col('cell_class').cast(pl.String).is_in(['Excitatory',
                                                    'Inhibitory'])),
    'nonneuronal': (
        pl.col('subclass').cast(pl.String).str.ends_with(' NN'),
        ~pl.col('cell_class').cast(pl.String).is_in(['Excitatory',
                                                     'Inhibitory'])),
}
transfer_path = f'{out_dir}/moffitt_transfer.parquet'
todo = [p for p in PARTS
        if not os.path.exists(f'{out_dir}/harmony_{p}.npz')]

if todo:
    print('[ref] loading Allen 10Xv3 reference...')
    sc_ref = SingleCell(ref_path).skip_qc()
    if REF_DIVISIONS is not None:
        sc_ref = sc_ref.filter_obs(
            pl.col('anatomical_division_label').cast(pl.String)
            .is_in(REF_DIVISIONS))
    sc_ref = sc_ref.filter_obs(pl.col('subclass').is_not_null())
    # 40 gene symbols are duplicated in the reference; the gene-name join in
    # hvg() expands on them, so keep the first occurrence of each
    sc_ref = sc_ref.filter_var(pl.col('gene_symbol').is_first_distinct())
    print(f'[ref] {sc_ref.X.shape[0]:,} cells, '
          f'{sc_ref.obs["subclass"].cast(pl.String).n_unique()} subclasses')

    print('[query] loading Moffitt GSE113576...')
    meta = pl.read_csv(f'{moffitt_dir}/aau5324_moffitt_table-s1.csv')
    sc_query = SingleCell(
        f'{moffitt_dir}/GSE113576_matrix.mtx.gz',
        obs=f'{moffitt_dir}/GSE113576_barcodes.tsv.gz',
        var=f'{moffitt_dir}/GSE113576_genes.tsv.gz')

    # the features file is two tab-separated columns (Ensembl ID, symbol) but
    # brisc reads it as one field; split off the symbol, which is what the
    # Allen reference is indexed by, and drop the 65 duplicated ones
    sc_query = sc_query\
        .with_columns_var(
            pl.col('gene').cast(pl.String).str.split('\t').list.last()
            .alias('gene_symbol'))\
        .filter_var(pl.col('gene_symbol').is_first_distinct())\
        .set_var_names('gene_symbol')

    # 'Ambiguous' and 'Unstable' are Moffitt's own QC rejects
    sc_query = sc_query\
        .join_obs(meta, on='cell', validate='1:1')\
        .filter_obs(
            pl.col('cell_class').is_not_null() &
            ~pl.col('cell_class').cast(pl.String)
            .is_in(['Ambiguous', 'Unstable']))\
        .with_columns_obs(
            pl.coalesce(pl.col('neuronal_cluster').cast(pl.String),
                        pl.col('non_neuronal_cluster').cast(pl.String))
            .alias('moffitt_cluster'))\
        .filter_obs(pl.col('moffitt_cluster').is_not_null())\
        .skip_qc()
    print(f'[query] {sc_query.X.shape[0]:,} cells, '
          f'{sc_query.obs["moffitt_cluster"].n_unique()} clusters')

    for part in todo:
        ref_filter, query_filter = PARTS[part]
        r = sc_ref.filter_obs(ref_filter)
        q = sc_query.filter_obs(query_filter)
        if MAX_PER_SUBCLASS is not None:
            # cap each subclass so the reference does not swamp the query, and
            # so its composition is closer to a POA dissection
            rng = np.random.default_rng(0)
            idx = r.obs.with_row_index('_i').select('_i', 'subclass')
            keep = np.concatenate([
                v if len(v := g['_i'].to_numpy()) <= MAX_PER_SUBCLASS
                else rng.choice(v, MAX_PER_SUBCLASS, replace=False)
                for _, g in idx.group_by('subclass')])
            mask = np.zeros(idx.height, dtype=bool)
            mask[keep] = True
            r = r.filter_obs(pl.Series(mask))
        print(f'[{part}] {r.X.shape[0]:,} reference and {q.X.shape[0]:,} '
              f'query cells')

        # hvg -> normalize -> scanorama -> pca -> harmony, as in
        # 04_project_cell_types.py
        if r.uns.get('normalized'):
            r = r.with_uns(normalized=False)
        if q.uns.get('normalized'):
            q = q.with_uns(normalized=False)
        r, q = r.hvg(q)
        r = r.normalize()
        q = q.normalize()

        r = r.filter_var(pl.col('highly_variable'))
        q = q.filter_var(pl.col('highly_variable'))
        a1, a2 = r.to_scanpy(), q.to_scanpy()
        print(f'[{part}] running scanorama ({a1.shape[1]} genes)...')
        corrected, genes = scanorama.correct(
            [a1.X, a2.X], [list(a1.var_names), list(a2.var_names)])
        var = pl.DataFrame({r.var_names.name: genes})
        r = SingleCell(X=sp.csr_matrix(corrected[0]).astype(np.float32),
                       obs=r.obs, var=var)\
            .with_uns(QCed=True, normalized=True)
        q = SingleCell(X=sp.csr_matrix(corrected[1]).astype(np.float32),
                       obs=q.obs, var=var)\
            .with_uns(QCed=True, normalized=True)
        del a1, a2, corrected

        r, q = r.pca(q, num_PCs=N_PCS, hvg_column=None)
        print(f'[{part}] running Harmony...')
        r, q = r.harmonize(q, **HARMONY_KWARGS)

        # did the datasets actually mix? if the query keeps its own corner of
        # the embedding, every query cell retrieves the same reference cells
        R = np.ascontiguousarray(r.obsm['harmony'], np.float32)
        Q = np.ascontiguousarray(q.obsm['harmony'], np.float32)
        C = np.vstack([R, Q]).copy()
        faiss.normalize_L2(C)
        ix = faiss.IndexFlatIP(C.shape[1])
        ix.add(C)
        rng = np.random.default_rng(0)
        s = rng.choice(len(Q), min(2000, len(Q)), replace=False) + len(R)
        _, nn_mix = ix.search(C[s], K_NEIGHBORS + 1)
        print(f'[{part}] mixing: {(nn_mix[:, 1:] < len(R)).mean():.1%} of '
              f'query neighbours are reference '
              f'(expect ~{len(R) / (len(R) + len(Q)):.0%})')
        del C, ix

        np.savez(
            f'{out_dir}/harmony_{part}.npz',
            ref_harmony=r.obsm['harmony'].astype(np.float32),
            query_harmony=q.obsm['harmony'].astype(np.float32),
            ref_subclass=r.obs['subclass'].cast(pl.String).to_numpy())
        q.obs\
            .select('cell', 'sex', 'cell_class', 'moffitt_cluster')\
            .write_parquet(f'{out_dir}/query_obs_{part}.parquet')
        print(f'[{part}] saved harmony_{part}.npz')
        del r, q
    del sc_ref, sc_query

# =============================================================================
# Label transfer: exact kNN on the Harmony embeddings
# =============================================================================
# exact cosine kNN on plain string labels, as in the faiss step of
# 04_project_cell_types.py
def transfer(part):
    z = np.load(f'{out_dir}/harmony_{part}.npz', allow_pickle=True)
    ref_h = np.ascontiguousarray(z['ref_harmony'], dtype=np.float32)
    qry_h = np.ascontiguousarray(z['query_harmony'], dtype=np.float32)
    ref_labels = z['ref_subclass'].astype(str)

    ok = np.isfinite(ref_h).all(axis=1)
    if not ok.all():
        print(f'  [{part}] dropping {(~ok).sum():,} reference cells with '
              f'NaN embeddings')
        ref_h, ref_labels = ref_h[ok], ref_labels[ok]
    assert np.isfinite(qry_h).all(), f'{part}: query embeddings contain NaN'

    names, codes = np.unique(ref_labels, return_inverse=True)
    faiss.normalize_L2(ref_h)
    faiss.normalize_L2(qry_h)
    index = faiss.IndexFlatIP(ref_h.shape[1])
    index.add(ref_h)
    _, nn = index.search(qry_h, K_NEIGHBORS)

    votes = codes[nn]
    counts = np.zeros((len(votes), len(names)), dtype=np.int16)
    np.add.at(counts, (np.repeat(np.arange(len(votes)), K_NEIGHBORS),
                       votes.ravel()), 1)
    order = np.argsort(-counts, axis=1)
    best, second = order[:, 0], order[:, 1]
    rows = np.arange(len(votes))
    conf = counts[rows, best] / K_NEIGHBORS
    margin = conf - counts[rows, second] / K_NEIGHBORS

    return pl.read_parquet(f'{out_dir}/query_obs_{part}.parquet')\
        .with_columns(
            pl.Series('subclass_transferred', names[best]),
            pl.Series('subclass_confidence', conf.astype(np.float32)),
            pl.Series('margin', margin.astype(np.float32)))\
        .with_columns(
            # a weak or contested call is left unassigned rather than forced:
            # a cluster that fails to map is itself part of the answer
            pl.when((pl.col('subclass_confidence') >= CONF_MIN) &
                    (pl.col('margin') >= MARGIN_MIN))
            .then(pl.col('subclass_transferred'))
            .otherwise(pl.lit('Unassigned')).alias('assigned'))

print('[transfer] assigning Allen subclasses (exact kNN)...')
obs = pl.concat([transfer(part) for part in PARTS])
obs.write_parquet(transfer_path)
obs = obs.to_pandas()
for part, sub in (('neuronal', obs.cell_class.isin(['Excitatory',
                                                    'Inhibitory'])),
                  ('nonneuronal', ~obs.cell_class.isin(['Excitatory',
                                                        'Inhibitory']))):
    print(f'  {part}: {sub.sum():,} cells, '
          f'{(obs.assigned[sub] == "Unassigned").mean():.1%} unassigned, '
          f'{obs.assigned[sub].nunique()} distinct subclasses')

# =============================================================================
# Which subclasses do we detect?
# =============================================================================
def included_subclasses(min_cells=MIN_CELLS, min_platforms=MIN_PLATFORMS):
    """Subclasses with >= min_cells on >= min_platforms platforms."""
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

detected = included_subclasses()
maternal = set(SUB19)
print(f'[detected] {len(detected)} subclasses, {len(maternal)} maternal')

# =============================================================================
# Cluster x subclass matrix
# =============================================================================
cm = pd.crosstab(obs['moffitt_cluster'], obs['assigned'])
n_all = cm.sum(axis=1)
unassigned = (cm['Unassigned'] / n_all if 'Unassigned' in cm
              else pd.Series(0.0, index=cm.index))
# the heatmap is about correspondence, so it is normalized over the cells that
# were assigned; the abstention rate is carried alongside it instead
assigned_cm = cm.drop(columns=['Unassigned'], errors='ignore')
frac = assigned_cm.div(assigned_cm.sum(axis=1).replace(0, np.nan), axis=0)\
    .fillna(0)

# rows: inhibitory, excitatory, then non-neuronal, each ordered so the
# strongest assignments run down the diagonal
def cluster_group(c):
    # i, e and h (hybrid) are Moffitt's neuronal clusters
    return 0 if c.startswith('i') else 1 if c.startswith('e') \
        else 2 if c.startswith('h') else 3

def cluster_num(c):
    head = c.split(':')[0]
    digits = ''.join(ch for ch in head if ch.isdigit())
    return int(digits) if digits else 0

top_call = frac.idxmax(axis=1)
top_frac = frac.max(axis=1)

row_order = sorted(frac.index,
                   key=lambda c: (cluster_group(c), cluster_num(c)))
if FIG_NEURONAL_ONLY:
    row_order = [c for c in row_order if cluster_group(c) < 3]
frac = frac.loc[row_order]

# columns: every cluster's top call, then any other subclass taking >=
# MIN_FRAC, capped at MAX_COLS by total mass
col_order = list(dict.fromkeys(top_call.loc[row_order].tolist()))
col_order += [c for c in frac.columns
              if c not in col_order and frac[c].max() >= MIN_FRAC]
if len(col_order) > MAX_COLS:
    ranked = frac[col_order].sum().sort_values(ascending=False)
    keep = set(ranked.index[:MAX_COLS])
    col_order = [c for c in col_order if c in keep]

# everything else a cluster's cells were assigned to, so each row sums to 1
other = (1 - frac[col_order].sum(axis=1)).clip(lower=0)
n_pooled = frac.shape[1] - len(col_order)
print(f'[figure] {len(col_order)} subclasses shown individually; the other '
      f'{n_pooled} hold {other.mean():.0%} of assigned cells on average')
frac = frac[col_order]
frac['Other subclasses'] = other
col_order.append('Other subclasses')

n_cells = n_all.loc[row_order]
unassigned_row = unassigned.loc[row_order]

# =============================================================================
# Summary: how much of their MPOA do we detect?
# =============================================================================
# a cluster counts as detected by the share of its cells landing on subclasses
# we report, not by its single top call: the largest inhibitory cluster tops
# out at 4%, so its top call describes almost none of it
n_assigned = assigned_cm.sum(axis=1).replace(0, np.nan)
det_frac = (assigned_cm[[c for c in assigned_cm.columns if c in detected]]
            .sum(axis=1) / n_assigned).fillna(0)
mat_frac = (assigned_cm[[c for c in assigned_cm.columns if c in maternal]]
            .sum(axis=1) / n_assigned).fillna(0)

summary_rows = sorted(cm.index, key=lambda c: (cluster_group(c),
                                               cluster_num(c)))
summary = pd.DataFrame({
    'moffitt_cluster': summary_rows,
    'n_cells': n_all.loc[summary_rows].values,
    'class': [['Inhibitory', 'Excitatory', 'Hybrid', 'Non-neuronal']
              [cluster_group(c)] for c in summary_rows],
    'top_subclass': top_call.loc[summary_rows].values,
    'top_fraction': top_frac.loc[summary_rows].values,
    'detected_fraction': det_frac.loc[summary_rows].values,
    'maternal_fraction': mat_frac.loc[summary_rows].values,
    'unassigned_fraction': unassigned.loc[summary_rows].values,
})
# top_fraction and detected_fraction are shares of the assigned cells, so the
# abstention rate is a separate column rather than folded into both
summary['detected'] = summary.detected_fraction >= 0.5
summary['maternal_circuit'] = summary.maternal_fraction >= 0.25
summary['dispersed'] = summary.top_fraction < 0.25
summary.to_csv(f'{out_dir}/moffitt_cluster_summary.csv', index=False)
cm.to_csv(f'{out_dir}/moffitt_allen_confusion.csv')

# sanity check: marker-defined clusters should land on the matching anatomy
print('\n[check] top call for marker-defined clusters')
for c in ['i8:Gal/Amigo2', 'i16:Gal/Th', 'i6:Avp/Nms', 'i38: Kiss1/Th',
          'e15:Ucn3/Brs3', 'Astrocytes 1', 'Microglia 1']:
    if c in top_call.index:
        print(f'  {c:22s} -> {top_call[c]} ({top_frac[c]:.0%})')

neu = summary[summary['class'] != 'Non-neuronal']
w = neu.n_cells * (1 - neu.unassigned_fraction)
cell_cov = (neu.detected_fraction * w).sum() / w.sum()
unass = ((neu.unassigned_fraction * neu.n_cells).sum() / neu.n_cells.sum())
print(f'\n[summary] {neu.detected.sum()}/{len(neu)} neuronal clusters place '
      f'most of their assigned cells in subclasses we detect; {cell_cov:.0%} '
      f'of their assigned neurons land in one ({unass:.0%} of neurons were '
      f'left unassigned)')
for grp in ['Excitatory', 'Inhibitory', 'Hybrid', 'Non-neuronal']:
    g = summary[summary['class'] == grp]
    print(f'  {grp}: {g.detected.sum()}/{len(g)} detected, '
          f'{g.maternal_circuit.sum()} in the maternal-circuit set, '
          f'{g.dispersed.sum()} dispersed')
missed = summary[~summary.detected].sort_values('n_cells', ascending=False)
if len(missed):
    print('\n[not detected] cluster -> top Allen subclass '
          '(detected share of cells)')
    for _, r in missed.iterrows():
        tag = ' [dispersed]' if r.dispersed else ''
        print(f'  {r.moffitt_cluster:24s} -> {r.top_subclass} '
              f'({r.top_fraction:.0%}, n={r.n_cells}, '
              f'detected {r.detected_fraction:.0%}){tag}')

# =============================================================================
# Figure
# =============================================================================
# column annotation. The 19 maternal-circuit subclasses are a subset of the 84
# we detect, so the two shades are nested rather than exclusive
ann = np.array([
    3 if c == 'Other subclasses' else 2 if c in maternal else
    1 if c in detected else 0
    for c in frac.columns])
ann_colors = ListedColormap(['#e8e8e8', '#a8c6e8', '#1f4e79', '#ffffff'])

nrow, ncol = frac.shape
h = 0.9 + 0.105 * nrow
w = 2.2 + 0.105 * ncol
fig, axes = plt.subplots(
    2, 2, figsize=(w, h),
    gridspec_kw={'height_ratios': [0.28, h - 0.28],
                 'width_ratios': [w - 0.55, 0.28],
                 'hspace': 0.015, 'wspace': 0.02})
(cax, blank), (ax, uax) = axes
blank.axis('off')

cax.pcolormesh(np.arange(ncol + 1), [0, 1], ann[None, :],
               cmap=ann_colors, norm=BoundaryNorm([0, 1, 2, 3, 4], 4),
               edgecolors='white', linewidth=0.4)
cax.set_xlim(0, ncol)
cax.set_xticks([])
cax.set_yticks([])
for s in cax.spines.values():
    s.set_visible(False)

mesh = ax.pcolormesh(np.arange(ncol + 1), np.arange(nrow + 1), frac.values,
                     cmap=sns.color_palette('rocket_r', as_cmap=True),
                     vmin=0, vmax=1, edgecolors='white', linewidth=0.4)
ax.invert_yaxis()
ax.set_xticks(np.arange(ncol) + 0.5)
ax.set_xticklabels(frac.columns, rotation=90, fontsize=5)
ax.set_yticks(np.arange(nrow) + 0.5)
ax.set_yticklabels([f'{c}  ({n:,})' for c, n in zip(frac.index, n_cells)],
                   fontsize=5)
ax.set_xlabel('Allen subclass assigned', fontsize=7)
ax.set_ylabel('Moffitt et al. (2018) cluster (n cells)', fontsize=7)
for s in ax.spines.values():
    s.set_visible(False)

# abstention rate, kept out of the matrix so it cannot be read as a mapping
uax.pcolormesh(np.arange(2), np.arange(nrow + 1),
               unassigned_row.values[:, None], cmap='Greys', vmin=0, vmax=1,
               edgecolors='white', linewidth=0.4)
uax.invert_yaxis()
uax.set_yticks([])
uax.set_xticks([0.5])
uax.set_xticklabels(['unassigned'], rotation=90, fontsize=5)
for s in uax.spines.values():
    s.set_visible(False)

# separators between the inhibitory, excitatory and hybrid blocks
groups = [cluster_group(c) for c in frac.index]
for i in range(1, len(groups)):
    if groups[i] != groups[i - 1]:
        for a in (ax, uax):
            a.axhline(i, color='black', linewidth=0.7)

cb = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.3, pad=0.02)
cb.set_label('Fraction of assigned cells', fontsize=6)
cb.ax.tick_params(labelsize=5)
handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in
           ['#1f4e79', '#a8c6e8', '#e8e8e8']]
cax.legend(handles, ['Detected: maternal circuit', 'Detected: other',
                     'Not detected'],
           loc='lower left', bbox_to_anchor=(0, 1.3), ncol=3,
           frameon=False, fontsize=5)

fig.savefig(f'{fig_dir}/supp_figure_6D.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{fig_dir}/supp_figure_6D.svg', bbox_inches='tight')
print(f'\nwrote {fig_dir}/supp_figure_6D.png/.svg')
