#region imports and setup ######################################################

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hc

sys.path.append(str(Path.home()))
from single_cell import SingleCell

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy'

cells_joined = pd.read_csv(
    '/home/karbabi/single-cell/ABC/metadata/cells_joined.csv',
    usecols=['class', 'class_color', 'subclass', 'subclass_color'])
color_mappings = {
    'class': dict(zip(
        cells_joined['class'].str.replace('/', '_'),
        cells_joined['class_color'])),
    'subclass': {k.replace('_', '/'): v for k, v in dict(zip(
        cells_joined['subclass'].str.replace('/', '_'),
        cells_joined['subclass_color'])).items()},
}
for level in color_mappings:
    color_mappings[level]['Unlabelled'] = '#d3d3d3'
# subclass -> class numeric prefix, for grouping subclasses into card rows
subclass_prefix = dict(zip(
    cells_joined['subclass'],
    cells_joined['class'].str.split(' ', n=1).str[0]))
del cells_joined

# datasets ordered top-to-bottom; exemplar = one physical section per dataset.
# only xenium carries the subclass legends (the shared set is the same for all)
datasets = {
    'slidetags': dict(
        display='Slide-tags', exemplar='CTRL_2', exemplar_col='sample',
        sp_size=4, umap_size=0.50, card_size=1.5),
    'merfish': dict(
        display='MERFISH', exemplar='PREG_1', exemplar_col='sample',
        sp_size=1, umap_size=0.05, card_size=0.3),
    'xenium': dict(
        display='Xenium', exemplar='CTRL_2_1', exemplar_col='sample_rep',
        sp_size=1, umap_size=0.25, card_size=0.3, legend=True),
}

# condition subrows (xenium has no POSTPART -> empty card)
conditions = ['CTRL', 'PREG', 'POSTPART']
condition_labels = {'CTRL': 'Nulliparous', 'PREG': 'Pregnant',
                    'POSTPART': 'Postpartum'}
condition_colors = {'CTRL': '#7209b7', 'PREG': '#b5179e', 'POSTPART': '#f72585'}
platform_colors = {'slidetags': '#0b6fa8', 'merfish': '#e8a628',
                   'xenium': '#1faa6b'}

# subclass inclusion filter: a subclass is shown if it reaches >= MIN_CELLS
# cells in at least MIN_PLATFORMS of the three platforms (reproducible +
# abundant; not gated by the shallow slide-tags platform)
MIN_CELLS = 100
MIN_PLATFORMS = 2

# class-label groups -> the 5 card rows: class prefixes, wrapped row label,
# and a per-group point-size multiplier (x dataset card_size)
class_groups = {
    'Pallium Glut':    dict(prefixes=['01', '02', '03', '04'],
                            label='Pallium\nGlut', size=1.0),
    'Pallium GABA':    dict(prefixes=['06', '07'],
                            label='Pallium\nGABA', size=1.5),
    'Subpallium GABA': dict(prefixes=['05', '08', '09', '10'],
                            label='Subpallium\nGABA', size=1.0),
    'HY-EA':           dict(prefixes=['11', '12', '13', '14', '15',
                                      '18', '19', '20', '24'],
                            label='HY-EA', size=1.2),
    'Non-neuronal':    dict(prefixes=['30', '31', '33', '34'],
                            label='Non-neuronal', size=0.8),
}

# combined-dotplot markers: genes in >=2 of the 3 panels (a missing panel just
# drops that wedge), grouped by neighbourhood and ordered within each group by
# their restricted within-group peak -> a clean block diagonal.
shared_markers = [
    'Slc17a7', 'Cux2', 'Rorb', 'Tle4', 'Foxp2', 'Fezf2',          # Pallium Glut
    'Gad2', 'Vip', 'Sncg', 'Lamp5', 'Pvalb', 'Lhx6', 'Sst',       # Pallium GABA
    'Dlx2', 'Cyp26b1', 'Tac1', 'Ppp1r1b', 'Drd1', 'Drd2', 'Pax6', 'Nr3c2',
    'Bnc2', 'Pdyn', 'Lhx8', 'Tac2', 'Six3', 'Gal', 'Ebf3', 'Barhl2',  # HY-EA
    'Bsx', 'Esr1', 'Otp', 'Trh',
    'Lrig1', 'Aqp4', 'Foxj1', 'Prlr', 'Pdgfra', 'Mog', 'Col1a1',  # Non-neuronal
    'Pdgfrb', 'Myh11', 'Flt1', 'Cx3cr1', 'Mrc1']
marker_group = (['Pallium Glut'] * 6 + ['Pallium GABA'] * 7
                + ['Subpallium GABA'] * 8 + ['HY-EA'] * 12
                + ['Non-neuronal'] * 12)

#endregion

#region brisc hogwild umap embedding ###########################################

def compute_umap(name):
    """Hogwild UMAP (brisc SingleCell) over all cells; cached to disk."""
    cache = f'{working_dir}/output/{name}/umap_hogwild.npz'
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d['ids'], d['umap']
    path = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    data = SingleCell(path, num_threads=-1).skip_qc()
    if data.shape[1] > 2000:
        data = data.hvg()
        hvg_column = 'highly_variable'
    else:
        hvg_column = None
    data = data.normalize()
    data = data.pca(hvg_column=hvg_column)
    data = data.neighbors()
    data = data.umap(hogwild=True)
    ids = np.asarray(data.obs['_index'], dtype=str)
    umap = np.asarray(data.obsm['umap'])
    np.savez(cache, ids=ids, umap=umap)
    return ids, umap

#endregion

#region figure: spatial exemplar + umap per dataset ###########################

def load_obs(name):
    adata = sc.read_h5ad(
        f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad', backed='r')
    obs = adata.obs.copy()
    adata.file.close()
    obs['_prefix'] = obs['class'].astype(str).str.split(' ', n=1).str[0]
    obs['_color'] = obs['subclass'].astype(str)\
        .map(color_mappings['subclass']).fillna('#333333')
    return obs

def sub_num(s):
    p = s.split(' ', 1)[0]
    return int(p) if p.isdigit() else 9999

def included_subclasses(min_cells=MIN_CELLS, min_platforms=MIN_PLATFORMS):
    """One shared subclass set across datasets: >= min_cells cells in at least
    min_platforms of the three platforms."""
    n_pass = {}
    for name in datasets:
        adata = sc.read_h5ad(
            f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad',
            backed='r')
        vc = adata.obs['subclass'].astype(str).value_counts()
        adata.file.close()
        for s in vc.index[vc >= min_cells]:
            n_pass[s] = n_pass.get(s, 0) + 1
    return {s for s, k in n_pass.items() if k >= min_platforms}

def grouped_order(keep):
    """Kept subclasses ordered by class group, then numerically within group."""
    order = []
    for grp_cfg in class_groups.values():
        order += sorted(
            (s for s in keep if subclass_prefix.get(s) in grp_cfg['prefixes']),
            key=sub_num)
    return order

def ordered_linkage(D):
    """Contiguity-constrained average linkage (only adjacent clusters merge),
    so the leaf order stays 0..n-1: a valid dendrogram over the forced order."""
    n = D.shape[0]
    active = [dict(id=i, members=[i], height=0.0) for i in range(n)]
    Z = []
    nid = n
    while len(active) > 1:
        best, bk = np.inf, 0
        for k in range(len(active) - 1):
            d = D[np.ix_(active[k]['members'], active[k + 1]['members'])].mean()
            if d < best:
                best, bk = d, k
        a, b = active[bk], active[bk + 1]
        h = max(best, a['height'], b['height']) + 1e-9
        Z.append([a['id'], b['id'], h, len(a['members']) + len(b['members'])])
        active[bk:bk + 2] = [dict(
            id=nid, members=a['members'] + b['members'], height=h)]
        nid += 1
    return np.array(Z, dtype=float)

def cluster_order(keep):
    """Forced grouped/numeric subclass order + a contiguity-constrained
    dendrogram of that order, from xenium mean expression."""
    base = grouped_order(keep)
    adata = sc.read_h5ad(
        f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad', backed='r')
    subc = adata.obs['subclass'].astype(str).values
    rng = np.random.default_rng(0)
    idx = []
    for s in base:                                     # <=500 cells/subclass
        ii = np.where(subc == s)[0]
        idx.append(rng.choice(ii, 500, replace=False) if len(ii) > 500 else ii)
    idx = np.sort(np.concatenate(idx))
    am = adata[idx].to_memory(); adata.file.close()
    X = am.X; X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    libs = X.sum(1, keepdims=True); libs[libs == 0] = 1
    Xn = np.log1p(X / libs * np.median(libs))
    mean = pd.DataFrame(Xn).assign(s=subc[idx]).groupby('s').mean()\
        .reindex(base)
    from scipy.spatial.distance import squareform
    D = squareform(pdist(mean.to_numpy(), 'correlation'))
    return base, ordered_linkage(D)

def compute_dotplot(name, markers, sub_order):
    """Per-subclass mean expression (z-scored per marker across subclasses) and
    fraction expressing, for markers present in this dataset's panel, with
    subclass columns in `sub_order`."""
    adata = sc.read_h5ad(
        f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad', backed='r')
    present = [m for m in markers if m in adata.var.index]
    subc = adata.obs['subclass'].astype(str).values
    X = adata[:, present].to_memory().X
    adata.file.close()
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    df = pd.DataFrame(X, columns=present)
    mean = df.groupby(subc)[present].mean().reindex(sub_order)
    frac = (df > 0).astype(float).groupby(subc)[present].mean()\
        .reindex(sub_order)
    M, F = mean.to_numpy().T, frac.to_numpy().T          # markers x subclasses
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True); sd[sd == 0] = 1
    return present, (M - mu) / sd, F

def compute_dotplot_aligned(name, markers, sub_order):
    """compute_dotplot, but Z/F aligned to the full `markers` list (rows for
    genes absent from this panel are NaN)."""
    present, Z, F = compute_dotplot(name, markers, sub_order)
    idx = {g: k for k, g in enumerate(present)}
    nz = np.full((len(markers), len(sub_order)), np.nan)
    nf = np.zeros((len(markers), len(sub_order)))
    for k, g in enumerate(markers):
        if g in idx:
            nz[k], nf[k] = Z[idx[g]], F[idx[g]]
    return nz, nf

# layout geometry, in inches
LC_W = 3.2          # first column (exemplar + umap) width
GAP_LC = 0.55       # gutter holding the group row labels
CARD_H = 1.15       # card height; card width = CARD_ASPECT * CARD_H
CARD_ASPECT = 1.38
CARD_W = CARD_ASPECT * CARD_H
GAP = 0.07          # uniform white space between cards (both directions)
LEG_W = 1.15        # legend width per column (doubles for 2-column legends)
TITLE_H = 0.46      # condition-title strip (two lines: label + N)
BLOCK_GAP = 0.28    # space between main rows (datasets)
PAD = 0.12
FS_DATASET = 18     # dataset title (Xenium / Slide-tags / MERFISH)
FS_GROUP = 13       # group row label (Pallium Glut, ...)
FS_COND = 15        # condition column title

# marker dotplot composite (rows=cell types, cols=genes; each dot is a
# 3-wedge glyph, one wedge per dataset)
DOT_GAP = 0.50      # gap between the figure body and the composite
DOT_CMAP = 'seismic'
T_VDEND_W = 0.75    # vertical dendrogram width (right)
T_VCBAR_W = 0.15    # vertical subclass color-bar width
T_ROWLAB_W = 2.05   # cell-type row-label gutter
T_GRP_W = 0.34      # vertical group-title column
GENE_LAB_H = 0.85   # gene-name strip at the bottom (45 deg)
BAR_LAB_H = 1.00    # strip for the vertical bar category labels
T_BAR_W = 0.85      # horizontal stacked-bar width, per bar
T_WEDGE_R = 0.47    # max wedge radius in cell units (cell = 1)
WEDGE_MIN_FRAC = 0.15   # floor on the fraction->radius map
WEDGE_ANGLE = {'slidetags': 90, 'merfish': 210, 'xenium': 330}
FS_GENE = 11        # gene column labels
FS_CTLAB = 10.5     # cell-type row labels
FS_GRP = FS_GROUP   # neighbourhood titles (match the cards)
FS_LEG = 8          # composite legends

def wedge_rfrac(f):
    return np.sqrt(WEDGE_MIN_FRAC + (1.0 - WEDGE_MIN_FRAC) * min(f, 1.0))

def build_figure():
    keep = included_subclasses()
    sub_order, Zlink = cluster_order(keep)             # dendrogram leaf order
    n_sub = len(sub_order)
    nrow = len(class_groups)

    # preload per dataset
    data = {}
    for name in datasets:
        obs = load_obs(name)
        ids, umap = compute_umap(name)
        umap = pd.DataFrame(umap, index=ids).reindex(obs.index).to_numpy()
        obs['_keep'] = obs['subclass'].astype(str).isin(keep)
        ds_conds = [c for c in conditions if (obs['condition'] == c).any()]
        dot = compute_dotplot_aligned(name, shared_markers, sub_order)
        counts = obs['subclass'].astype(str).value_counts().reindex(
            sub_order, fill_value=0)
        data[name] = (obs, umap, ds_conds, dot, counts)

    # condition-enrichment proportions (merfish; all conditions present)
    mobs = data['merfish'][0]
    ctot = {c: (mobs['condition'] == c).sum() for c in conditions}
    cond_prop = np.zeros((n_sub, len(conditions)))
    for i, s in enumerate(sub_order):
        cc = mobs[mobs['subclass'].astype(str) == s]['condition']
        p = np.array([(cc == c).sum() / ctot[c] for c in conditions])
        cond_prop[i] = p / p.sum() if p.sum() > 0 else 0
    # platform proportions (share of each platform's cells per subclass)
    ptot = {n: data[n][4].sum() for n in datasets}
    plat = list(datasets)
    plat_prop = np.zeros((n_sub, len(plat)))
    for i, s in enumerate(sub_order):
        p = np.array([data[n][4][s] / ptot[n] for n in plat])
        plat_prop[i] = p / p.sum() if p.sum() > 0 else 0

    block_h = TITLE_H + nrow * CARD_H + (nrow - 1) * GAP
    base_x = PAD + LC_W + GAP_LC                       # left edge of the cards
    body_w = max((len(data[n][2]) * CARD_W + (len(data[n][2]) - 1) * GAP)
                 + (2 * LEG_W if datasets[n].get('legend') else 0)
                 for n in datasets)
    # transposed combined dotplot geometry (rows = cell types, cols = genes);
    # dotplot vertical extent aligned to the cell-type card stack
    n_gene = len(shared_markers)
    cards_bottom = (PAD + len(datasets) * block_h
                    + (len(datasets) - 1) * BLOCK_GAP)
    dot_top = PAD + TITLE_H                             # = first card top
    dot_h = cards_bottom - dot_top                      # = last card bottom
    T_CELL = dot_h / n_sub
    dot_w = n_gene * T_CELL
    x_lab = base_x + body_w + DOT_GAP                  # cell-type name column
    x_cbar = x_lab + T_ROWLAB_W                         # color bar (by dotplot)
    x_dot = x_cbar + T_VCBAR_W                          # dotplot left edge
    x_bar1 = x_dot + dot_w + 0.14                       # condition bar
    x_bar2 = x_bar1 + T_BAR_W + 0.06                    # platform bar
    x_cbar2 = x_bar2 + T_BAR_W + 0.08                   # 2nd subclass color bar
    x_grp = x_cbar2 + T_VCBAR_W + 0.14                  # group titles
    x_dend = x_grp + T_GRP_W + 0.14                    # dendrogram (right)
    lx_in = x_dend + T_VDEND_W + 0.30                   # legends left edge
    W = lx_in + 1.85
    H = cards_bottom + max(GENE_LAB_H, BAR_LAB_H) + PAD  # bottom-label room

    fig = plt.figure(figsize=(W, H))

    def rect(x, top, w, h):  # inches, top = distance from the figure top
        return [x / W, 1 - (top + h) / H, w / W, h / H]

    def bare(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    def panel(x, top, w, h, equal=True):
        ax = bare(fig.add_axes(rect(x, top, w, h)))
        if equal:
            ax.set_aspect('equal')
        return ax

    #--- left side: per dataset row (exemplar, umap, cards, legend) ----------
    for row, (name, cfg) in enumerate(datasets.items()):
        obs, umap, ds_conditions = data[name][:3]
        xlim = (obs['x_ffd'].min(), obs['x_ffd'].max())
        ylim = (obs['y_ffd'].min(), obs['y_ffd'].max())
        card_top0 = PAD + row * (block_h + BLOCK_GAP) + TITLE_H
        area_h = nrow * CARD_H + (nrow - 1) * GAP

        ex = obs[obs[cfg['exemplar_col']] == cfg['exemplar']]
        ax = panel(PAD, card_top0, LC_W, area_h / 2)
        ax.scatter(ex['x_ffd'], ex['y_ffd'], c=ex['_color'],
                   s=cfg['sp_size'], linewidths=0, rasterized=True)
        fig.text((PAD + LC_W / 2) / W, 1 - (card_top0 - 0.04) / H,
                 cfg['display'], ha='center', va='bottom', fontsize=FS_DATASET)
        ax = panel(PAD, card_top0 + area_h / 2, LC_W, area_h / 2)
        ax.scatter(umap[:, 0], umap[:, 1], c=obs['_color'],
                   s=cfg['umap_size'], linewidths=0, rasterized=True)

        for cj, cond in enumerate(ds_conditions):
            cx = base_x + cj * (CARD_W + GAP) + CARD_W / 2
            nsamp = obs[obs['condition'] == cond]['sample'].nunique()
            fig.text(cx / W, 1 - (card_top0 - 0.04) / H,
                     f'{condition_labels[cond]}\n(N={nsamp})', ha='center',
                     va='bottom', fontsize=FS_COND, linespacing=1.0)

        for gi, (gname, grp_cfg) in enumerate(class_groups.items()):
            prefixes = grp_cfg['prefixes']
            grp = obs[obs['_prefix'].isin(prefixes) & obs['_keep']]
            psize = cfg['card_size'] * grp_cfg['size']
            ctop = card_top0 + gi * (CARD_H + GAP)
            for cj, cond in enumerate(ds_conditions):
                ax = panel(base_x + cj * (CARD_W + GAP), ctop, CARD_W, CARD_H,
                           equal=False)
                ax.set_facecolor('black')
                sub = grp[grp['condition'] == cond]
                if len(sub):
                    ax.scatter(sub['x_ffd'], sub['y_ffd'], c=sub['_color'],
                               s=psize, linewidths=0, rasterized=True)
                ax.set_xlim(xlim); ax.set_ylim(ylim)
                ax.set_aspect('equal', adjustable='datalim')

            fig.text((PAD + LC_W + GAP_LC / 2) / W, 1 - (ctop + CARD_H / 2) / H,
                     grp_cfg['label'], rotation=90, ha='center', va='center',
                     fontsize=FS_GROUP)

            if not cfg.get('legend'):
                continue
            subs = sorted(
                [s for s in keep if subclass_prefix.get(s) in prefixes],
                key=sub_num)
            ncol = 2 if len(subs) > 9 else 1
            leg_x = base_x + (len(ds_conditions) - 1) * (CARD_W + GAP) + CARD_W
            ax = panel(leg_x, ctop, ncol * LEG_W, CARD_H, equal=False)
            ax.set_facecolor('black')
            handles = [Line2D(
                [0], [0], marker='o', linestyle='', markersize=2.5,
                markeredgewidth=0,
                markerfacecolor=color_mappings['subclass'].get(
                    s, '#333333')) for s in subs]
            ax.legend(handles, subs, loc='center left', frameon=False,
                      fontsize=4.0, labelcolor='white', handletextpad=0.3,
                      labelspacing=0.18, columnspacing=0.6, ncol=ncol,
                      borderaxespad=0.2)

    #--- right side: combined 3-dataset dotplot (rows=cell types, cols=genes)-
    sub_colors = [color_mappings['subclass'].get(s, '#333333')
                  for s in sub_order]
    gspans, st = [], 0                                  # cell-type group spans
    for gname, grp_cfg in class_groups.items():
        cnt = sum(subclass_prefix.get(s) in grp_cfg['prefixes']
                  for s in sub_order)
        gspans.append((gname, st, cnt)); st += cnt
    row_bounds = [s + c for _, s, c in gspans][:-1]
    col_bounds = [k for k in range(1, n_gene)
                  if marker_group[k] != marker_group[k - 1]]
    norm = Normalize(-2, 2); cmap = plt.get_cmap(DOT_CMAP)

    # main dotplot: each cell is a 3-wedge glyph (one wedge per dataset)
    ax = bare(fig.add_axes(rect(x_dot, dot_top, dot_w, dot_h)))
    wedges, absent = [], []
    for name in datasets:
        Z, F = data[name][3]
        a = WEDGE_ANGLE[name]
        for gi in range(n_gene):
            for ci in range(n_sub):
                z, f = Z[gi, ci], F[gi, ci]
                if np.isnan(z):                    # gene not in this panel
                    absent.append(Wedge((gi, ci), T_WEDGE_R * 0.9, a - 60,
                                        a + 60, facecolor='#f0f0f0', lw=0))
                    continue
                if f <= 0:
                    continue
                wedges.append(Wedge(
                    (gi, ci), T_WEDGE_R * wedge_rfrac(f), a - 60, a + 60,
                    facecolor=cmap(norm(z)), edgecolor='#777777', lw=0.1))
    ax.add_collection(PatchCollection(absent, match_original=True,
                                      rasterized=True))
    ax.add_collection(PatchCollection(wedges, match_original=True,
                                      rasterized=True))
    ax.set_xlim(-0.5, n_gene - 0.5); ax.set_ylim(n_sub - 0.5, -0.5)
    ax.set_aspect('equal')
    for b in row_bounds:
        ax.axhline(b - 0.5, color='black', lw=0.9)
    for b in col_bounds:
        ax.axvline(b - 0.5, color='black', lw=0.9)
    ax.set_xticks(range(n_gene))
    ax.set_xticklabels(shared_markers, rotation=45, ha='right',
                       rotation_mode='anchor', fontsize=FS_GENE)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', length=2, pad=2, top=False, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.set_yticks([])
    sax = ax.secondary_xaxis('top')                    # repeat gene labels on top
    sax.set_xticks(range(n_gene))
    sax.set_xticklabels(shared_markers, rotation=45, ha='left',
                        rotation_mode='anchor', fontsize=FS_GENE)
    sax.tick_params(length=2, pad=2)
    sax.spines['top'].set_visible(False)
    for i, s in enumerate(sub_order):                  # row labels left of cbar
        fig.text((x_cbar - 0.04) / W, 1 - (dot_top + (i + 0.5) * T_CELL) / H,
                 s, ha='right', va='center', fontsize=FS_CTLAB)

    # vertical dendrogram (right), leaf 0 at top to match the rows
    axd = bare(fig.add_axes(rect(x_dend, dot_top, T_VDEND_W, dot_h)))
    hc.dendrogram(Zlink, ax=axd, orientation='right', no_labels=True,
                  color_threshold=0, above_threshold_color='#555555')
    axd.set_ylim(10 * n_sub, 0); axd.set_xticks([]); axd.set_yticks([])

    # vertical subclass color bars (left of dotplot and right of platform bar)
    def cbar(x):
        axc = bare(fig.add_axes(rect(x, dot_top, T_VCBAR_W, dot_h)))
        axc.set_xlim(0, 1); axc.set_ylim(n_sub - 0.5, -0.5)
        for i, c in enumerate(sub_colors):
            axc.add_patch(Rectangle((0, i - 0.5), 1, 1, facecolor=c, lw=0))
    cbar(x_cbar); cbar(x_cbar2)

    # horizontal stacked bars (right), per cell-type row; categories are
    # labelled directly under each bar in their theme colours (no legend)
    def hbar(x, prop, cols, labels):
        bx = bare(fig.add_axes(rect(x, dot_top, T_BAR_W, dot_h)))
        bx.set_xlim(0, 1); bx.set_ylim(n_sub - 0.5, -0.5)
        for i in range(n_sub):
            base = 0.0
            for k, c in enumerate(cols):
                bx.add_patch(Rectangle((base, i - 0.5), prop[i, k], 1,
                                       facecolor=c, lw=0))
                base += prop[i, k]
        for b in row_bounds:
            bx.axhline(b - 0.5, color='black', lw=0.9)
        for k, (lab, c) in enumerate(zip(labels, cols)):
            cxk = x + T_BAR_W * (k + 0.5) / len(labels)
            fig.text(cxk / W, 1 - (dot_top + dot_h + 0.05) / H, lab,
                     rotation=45, ha='right', va='top', rotation_mode='anchor',
                     fontsize=FS_GENE, color=c)
    hbar(x_bar1, cond_prop, [condition_colors[c] for c in conditions],
         [condition_labels[c] for c in conditions])
    hbar(x_bar2, plat_prop, [platform_colors[n] for n in plat],
         [datasets[n]['display'] for n in plat])

    # cell-type neighbourhood titles (right of dendrogram, vertical, per span)
    for gname, s0, cnt in gspans:
        fig.text((x_grp + T_GRP_W / 2) / W,
                 1 - (dot_top + (s0 + cnt / 2) * T_CELL) / H, gname,
                 rotation=270, ha='center', va='center', fontsize=FS_GRP)

    #--- legends (right margin), clustered ----------------------------------
    lx = lx_in / W
    # glyph schematic with platform names labelling each wedge (inverted y to
    # match the dots); no title
    axk = bare(fig.add_axes([lx, 0.72, 1.7 / W, 1.7 / H]))
    axk.set_xlim(-2.7, 2.7); axk.set_ylim(2.4, -2.7); axk.set_aspect('equal')
    for name in datasets:
        a = WEDGE_ANGLE[name]
        axk.add_patch(Wedge((0, 0), 1, a - 60, a + 60,
                            facecolor=platform_colors[name],
                            edgecolor='white', lw=1.0))
        ar = np.deg2rad(a)
        ha = ('center' if abs(np.cos(ar)) < 0.3
              else ('left' if np.cos(ar) > 0 else 'right'))
        axk.text(1.55 * np.cos(ar), 1.55 * np.sin(ar),
                 datasets[name]['display'], ha=ha, va='center', fontsize=FS_LEG)
    # z-score colorbar (wedge fill)
    sm = plt.cm.ScalarMappable(cmap=DOT_CMAP, norm=norm)
    cb = fig.colorbar(sm, cax=fig.add_axes([lx, 0.61, 0.008, 0.06]))
    cb.outline.set_visible(False)
    cb.ax.set_title('z-scored\nexpression', fontsize=FS_LEG, pad=4)
    cb.ax.tick_params(labelsize=FS_LEG - 2)
    # fraction-expressing size key (wedge radius)
    axf = bare(fig.add_axes([lx, 0.52, 1.2 / W, 0.85 / H]))
    axf.set_xlim(0, 3.4); axf.set_ylim(-1.0, 1.2); axf.set_aspect('equal')
    for xi, fr in zip([0.5, 1.6, 2.8], [0.1, 0.5, 1.0]):
        axf.add_patch(plt.Circle((xi, 0.35), 0.6 * wedge_rfrac(fr),
                                 facecolor='#bbbbbb', lw=0))
        axf.text(xi, -0.75, f'{int(fr * 100)}%', ha='center', va='center',
                 fontsize=FS_LEG - 1)
    axf.set_title('% expressing', fontsize=FS_LEG, pad=2, loc='left')
    for ext in ['png', 'svg']:
        fig.savefig(f'{working_dir}/figures/figure_1.{ext}',
                    dpi=400, facecolor='white')
    plt.close(fig)
    print('figure saved: figures/figure_1.{png,svg}')

#endregion

#region run ####################################################################

if __name__ == '__main__':
    build_figure()

#endregion
