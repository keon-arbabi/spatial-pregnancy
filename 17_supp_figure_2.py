#region imports and setup ######################################################

# Supplementary Figure 2: UMAPs of the shared Harmony space used for cell-type
# projection (04_project_cell_types.py). For each platform we embed the query
# cells together with the scRNA-seq reference (the exact Harmony embeddings
# cached in output/<name>/query_scrna_harmony.npz), compute one combined UMAP,
# and show 5 panels: query coloured by condition / class / subclass and the
# reference coloured by class / subclass. Colours follow the Allen palette used
# in 10_figure_1.py.

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.append(str(Path.home()))
from single_cell import SingleCell

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

working_dir = '/home/karbabi/spatial-pregnancy'
bridge_path = f'{working_dir}/input/adata_ref_zeng_bridge.h5ad'

datasets = {
    'slidetags': dict(display='Slide-tags', q_size=1.5, r_size=0.06),
    'merfish':   dict(display='MERFISH',    q_size=0.10, r_size=0.06),
    'xenium':    dict(display='Xenium',     q_size=0.20, r_size=0.06),
}

# Allen class/subclass colour maps (same construction as 10_figure_1.py)
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
del cells_joined

condition_colors = {'CTRL': '#7209b7', 'PREG': '#b5179e', 'POSTPART': '#f72585'}
condition_labels = {'CTRL': 'Nulliparous', 'PREG': 'Pregnant',
                    'POSTPART': 'Postpartum'}
DEFAULT_GREY = '#cccccc'

#endregion
#region combined UMAP in shared Harmony space ##################################

# Build a combined UMAP (query + scRNA-seq reference) from the cached Harmony
# embeddings using the brisc hogwild UMAP, matching plot_harmony_umap in
# 04_project_cell_types.py. Cached to disk so re-plotting is cheap.
def compute_harmony_umap(name):
    out_dir = f'{working_dir}/output/{name}'
    cache = f'{out_dir}/supp_harmony_umap.npz'
    if os.path.exists(cache):
        print(f'[{name}] loading cached combined UMAP')
        d = np.load(cache)
        return d['q_umap'], d['r_umap']

    print(f'[{name}] loading cached Harmony embeddings')
    data = np.load(f'{out_dir}/query_scrna_harmony.npz', allow_pickle=True)
    qh = np.asarray(data['query_harmony'], dtype=np.float32)
    rh = np.asarray(data['scrna_harmony'], dtype=np.float32)
    n_q, n_r = qh.shape[0], rh.shape[0]
    H = np.vstack([qh, rh])
    del qh, rh

    print(f'[{name}] combined UMAP over {H.shape[0]:,} cells '
          f'({n_q:,} query + {n_r:,} reference)...')
    obs = pl.DataFrame({'cell_id': np.arange(H.shape[0]).astype(str)})
    var = pl.DataFrame({'gene': ['placeholder']})
    X = sp.csr_matrix((H.shape[0], 1), dtype=np.float32)
    s = SingleCell(X=X, obs=obs, var=var, obsm={'harmony': H})
    s = s.neighbors(PC_key='harmony', QC_column=None)
    s = s.umap(PC_key='harmony', QC_column=None, hogwild=True)
    umap = np.asarray(s.obsm['umap'])
    q_umap, r_umap = umap[:n_q], umap[n_q:]
    np.savez(cache, q_umap=q_umap, r_umap=r_umap)
    print(f'[{name}] saved {cache}')
    return q_umap, r_umap

# query labels (row-aligned to query_harmony, i.e. the 02 adata) and reference
# labels (row-aligned to scrna_harmony, from the Phase A bridge uns).
def load_labels(name, ref_labels):
    adata = sc.read_h5ad(
        f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad', backed='r')
    obs = adata.obs
    q = {
        'condition': obs['condition'].astype(str).to_numpy(),
        'class': obs['class'].astype(str).to_numpy(),
        'subclass': obs['subclass'].astype(str).to_numpy(),
    }
    adata.file.close()
    return q, ref_labels

#endregion
#region colour helpers + figure ################################################

# map label strings to colours, tolerating '/' vs '_' differences between the
# label namespace and the colour-map keys.
def to_colors(labels, mapping, default=DEFAULT_GREY):
    labels = np.asarray(labels).astype(str)
    lut = {}
    for l in pd.unique(labels):
        c = (mapping.get(l) or mapping.get(l.replace('/', '_'))
             or mapping.get(l.replace('_', '/')))
        lut[l] = c if c else default
    return mcolors.to_rgba_array(pd.Series(labels).map(lut).to_numpy())

# Columns: first three show the query (condition / class / subclass), last two
# show the reference (class / subclass). Group headers span each set.
COLS = [
    ('q', 'condition', 'Condition'),
    ('q', 'class',     'Class'),
    ('q', 'subclass',  'Subclass'),
    ('r', 'class',     'Class'),
    ('r', 'subclass',  'Subclass'),
]
GROUPS = [(0, 2, 'Query'), (3, 4, 'Reference')]
# the condition column is split into a 2x2 quadrant, one condition per cell
# (top-left -> right -> down); Xenium has no postpartum, so it fills 2 of 4
COND_QUAD = ['CTRL', 'PREG', 'POSTPART']
BG_GREY = '#dcdcdc'

# panel geometry (inches): perfectly square panels in a tight, fused grid
P = 2.6                  # panel side
G = 0.04                 # gap between panels (both directions)
GQ = 0.04                # gap between the condition sub-quadrants
ML, MR = 0.55, 0.12      # left (row labels) / right margins
MT, MB = 0.72, 0.95      # top (titles + headers) / bottom (legend) margins

# ---- discordance element: confusion matrices + featured-subclass spatial ----
CM, GC, CONF_LX = 3.4, 0.5, 1.0     # confusion side, gap, left class-label gutter
SP_LX, SP_QH = 0.5, 2.9             # spatial left margin / query panel height
CONF_TARGET = '060 OT D3 Folh1 Gaba'   # featured discordant subclass
# the two subclasses the expression-only OT D3 calls are reassigned to
TARGET2, TARGET3 = '063 STR D1 Sema5a Gaba', '061 STR D1 Gaba'
OT_COLOR = '#d1006c'                    # 060 OT D3
C2_COLOR = '#2166ac'                    # 063 STR D1 Sema5a
C3_COLOR = '#1b7837'                    # 061 STR D1 Gaba
CELLS_JOINED = '/home/karbabi/single-cell/ABC/metadata/cells_joined.csv'

def _num(s):
    p = str(s).split(' ', 1)[0]
    return int(p) if p.isdigit() else 9999

# subclasses ordered by class (block-diagonal confusion) + per-platform
# constrained/unconstrained label arrays from the 03 (post-resolVI) adata
def _subclass_order():
    cj = pd.read_csv(CELLS_JOINED, usecols=['class', 'subclass']).drop_duplicates()
    sub2class = {str(s): str(c) for s, c in zip(cj['subclass'], cj['class'])}
    subs, obsd = set(), {}
    for name in datasets:
        a = sc.read_h5ad(
            f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad', backed='r')
        con = a.obs['subclass'].astype(str).to_numpy()
        unc = a.obs['subclass_unconstrained'].astype(str).to_numpy()
        a.file.close()
        obsd[name] = (con, unc)
        subs.update(con.tolist()); subs.update(unc.tolist())
    order = sorted(subs, key=lambda s: (_num(sub2class.get(s, '999')), _num(s)))
    cls_seq = [sub2class.get(s, '?') for s in order]
    blocks, st = [], 0
    for i in range(len(order)):
        if i == len(order) - 1 or cls_seq[i + 1] != cls_seq[i]:
            blocks.append((cls_seq[st], st, i - st + 1)); st = i + 1
    return order, blocks, obsd

# row-normalised confusion: rows = expression-only, cols = spatial-constrained
def _confusion(con, unc, order):
    idx = {s: i for i, s in enumerate(order)}; n = len(order)
    ui = np.array([idx.get(u, -1) for u in unc])
    ci = np.array([idx.get(c, -1) for c in con])
    ok = (ui >= 0) & (ci >= 0)
    M = np.zeros((n, n))
    np.add.at(M, (ui[ok], ci[ok]), 1)
    rs = M.sum(1, keepdims=True); rs[rs == 0] = 1
    return M / rs

# pooled xenium query (CAST-aligned x_ffd/y_ffd) with full constrained /
# unconstrained subclass labels, plus reference per coronal section with full
# subclass labels (same coordinate frame)
def _ot_spatial():
    a = sc.read_h5ad(
        f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad', backed='r')
    qxy = a.obs[['x_ffd', 'y_ffd']].values.astype(float)
    q_con = a.obs['subclass'].astype(str).to_numpy()
    q_unc = a.obs['subclass_unconstrained'].astype(str).to_numpy()
    a.file.close()
    b = sc.read_h5ad(bridge_path, backed='r')
    rb = b.obs
    secs = sorted(s for s in rb['sample'].astype(str).unique()
                  if s.startswith('C57BL6J-638850'))
    ref = []
    for s in secs:
        g = rb[rb['sample'].astype(str) == s]
        ref.append((s.split('.')[-1], g[['x_raw', 'y_raw']].values.astype(float),
                    g['subclass'].astype(str).to_numpy()))
    b.file.close()
    return qxy, q_unc, q_con, ref

# per-platform robustness metrics for the confusion row: spatial-vs-expression
# concordance (03) + query<->reference integration (iLISI) in the Harmony space
def _platform_metrics(name):
    a = sc.read_h5ad(
        f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad', backed='r')
    cl = a.obs['class'].astype(str).to_numpy()
    clu = a.obs['class_unconstrained'].astype(str).to_numpy()
    sb = a.obs['subclass'].astype(str).to_numpy()
    sbu = a.obs['subclass_unconstrained'].astype(str).to_numpy()
    a.file.close()
    d = np.load(f'{working_dir}/output/{name}/query_scrna_harmony.npz',
                allow_pickle=True)
    qh = d['query_harmony'].astype(np.float32)
    sh = d['scrna_harmony'].astype(np.float32)
    rng = np.random.default_rng(0); nsub, k = 20000, 30
    iq = rng.choice(qh.shape[0], min(nsub, qh.shape[0]), replace=False)
    isr = rng.choice(sh.shape[0], min(nsub, sh.shape[0]), replace=False)
    Hn = np.vstack([sh[isr], qh[iq]])
    Hn /= np.linalg.norm(Hn, axis=1, keepdims=True).clip(min=1e-8)
    batch = np.array([0] * len(isr) + [1] * len(iq))
    _, nn = cKDTree(Hn).query(Hn, k=k + 1)
    p1 = batch[nn[:, 1:]].mean(1)
    ilisi = float((1.0 / (p1 ** 2 + (1 - p1) ** 2)).mean())
    return dict(class_concord=float((cl == clu).mean()),
                sub_concord=float((sb == sbu).mean()), ilisi=ilisi)

def build_figure():
    print('loading reference labels (bridge)...')
    b = sc.read_h5ad(bridge_path, backed='r')
    ref_labels = {
        'class': np.asarray(b.uns['scrna_class_labels']).astype(str),
        'subclass': np.asarray(b.uns['scrna_subclass_labels']).astype(str),
    }
    b.file.close()

    # preload per dataset: umap coords + label arrays (caches reused if present)
    store = {}
    for name in datasets:
        q_umap, r_umap = compute_harmony_umap(name)
        q, r = load_labels(name, ref_labels)
        assert len(q['class']) == len(q_umap), \
            f'{name}: query labels ({len(q["class"])}) != umap ({len(q_umap)})'
        assert len(r['class']) == len(r_umap), \
            f'{name}: ref labels ({len(r["class"])}) != umap ({len(r_umap)})'
        store[name] = dict(q_umap=q_umap, r_umap=r_umap, q=q, r=r)

    # discordance-element data + per-platform robustness metrics
    print('computing subclass confusion + featured-subclass spatial...')
    order, blocks, obsd = _subclass_order()
    qxy, q_unc, q_con, ref = _ot_spatial()
    metrics = {name: _platform_metrics(name) for name in datasets}
    sx0, sx1 = qxy[:, 0].min(), qxy[:, 0].max()
    sy0, sy1 = qxy[:, 1].min(), qxy[:, 1].max()
    asp = (sy1 - sy0) / (sx1 - sx0)               # data aspect (height / width)

    # spatial panels: query is 1x2 (large); reference is a 2x2 grid whose total
    # height matches the query panel height. Two spatial rows.
    QH = SP_QH; QW = QH / asp; QG = 0.12
    RG = 0.08; RH = (QH - RG) / 2; RW = RH / asp; GROUPG = 0.5

    # vertical layout (inches from figure top): UMAP grid, confusion matrices,
    # two spatial rows. The condition legend lives in Xenium's empty quadrant.
    nrow, ncol = len(datasets), len(COLS)
    W = ML + ncol * P + (ncol - 1) * G + MR
    grid_bottom = MT + nrow * P + (nrow - 1) * G
    GAP0, GAP2 = 0.35, 0.55
    CONF_XLAB_H = 0.95                          # xlabel + per-matrix metrics
    conf_mat_top = grid_bottom + GAP0 + 0.30    # platform titles
    conf_mat_bottom = conf_mat_top + CM
    sp_top = conf_mat_bottom + CONF_XLAB_H + GAP2 + 0.50     # group + titles
    H = sp_top + QH + 0.55
    fig = plt.figure(figsize=(W, H))

    def rect(x, top, w, h):              # inches -> figure-fraction axes box
        return [x / W, 1 - (top + h) / H, w / W, h / H]
    def col_x(ci):                       # left edge of column ci (inches)
        return ML + ci * (P + G)
    def row_top(ri):                     # top edge of row ri (inches from top)
        return MT + ri * (P + G)
    def add_ax(ri, ci):                  # fixed square axes -> square boxes
        return fig.add_axes(rect(col_x(ci), row_top(ri), P, P))
    # 2x2 sub-quadrants inside the column-0 footprint (TL, TR, BL, BR)
    qs = (P - GQ) / 2
    quad_off = [(0, 0), (qs + GQ, 0), (0, qs + GQ), (qs + GQ, qs + GQ)]
    def add_quad(ri, qi):
        dx, dtop = quad_off[qi]
        return fig.add_axes(rect(col_x(0) + dx, row_top(ri) + dtop, qs, qs))

    # ---- top: UMAP grid (rows = platforms) ----
    for ri, (name, cfg) in enumerate(datasets.items()):
        d = store[name]
        # square, undistorted data limits shared across the row's panels
        allxy = np.vstack([d['q_umap'], d['r_umap']])
        (x0, y0), (x1, y1) = allxy.min(0), allxy.max(0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half = max(x1 - x0, y1 - y0) / 2 * 1.03

        def setlim(ax):
            ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)
            ax.set_xticks([]); ax.set_yticks([])

        # column 0: condition quadrant, one filled cell per condition present
        lab = d['q']['condition']
        present = [c for c in COND_QUAD if (lab == c).any()]
        for qi, cond in enumerate(present):
            ax = add_quad(ri, qi)
            ax.scatter(d['q_umap'][:, 0], d['q_umap'][:, 1], c=BG_GREY,
                       s=cfg['q_size'] * 0.5, alpha=0.3, linewidths=0,
                       rasterized=True)
            m = lab == cond
            ax.scatter(d['q_umap'][m, 0], d['q_umap'][m, 1],
                       c=condition_colors[cond], s=cfg['q_size'], alpha=0.8,
                       linewidths=0, rasterized=True)
            setlim(ax)

        # columns 1-4: full square panels
        for ci in range(1, len(COLS)):
            pop, level, _ = COLS[ci]
            ax = add_ax(ri, ci)
            umap = d['q_umap'] if pop == 'q' else d['r_umap']
            size = cfg['q_size'] if pop == 'q' else cfg['r_size']
            ax.scatter(umap[:, 0], umap[:, 1],
                       c=to_colors(d[pop][level], color_mappings[level]),
                       s=size, linewidths=0, alpha=0.6, rasterized=True)
            setlim(ax)

        # row label
        fig.text((ML * 0.4) / W, 1 - (row_top(ri) + P / 2) / H, cfg['display'],
                 rotation=90, ha='center', va='center', fontsize=14)

    # column headers (top row) and group headers (not bold)
    for ci, (_, _, sub) in enumerate(COLS):
        fig.text((col_x(ci) + P / 2) / W, 1 - (MT - 0.12) / H, sub,
                 ha='center', va='bottom', fontsize=12)
    for cl, cr, txt in GROUPS:
        xc = (col_x(cl) + col_x(cr) + P) / 2
        fig.text(xc / W, 1 - (MT - 0.42) / H, txt, ha='center', va='bottom',
                 fontsize=15)

    # condition legend: vertical (1 col) inside Xenium's empty condition
    # quadrants (Xenium has only 2 of 4 conditions, leaving the bottom half)
    cond_handles = [
        Line2D([0], [0], marker='o', linestyle='', markersize=7,
               markerfacecolor=condition_colors[c], markeredgewidth=0,
               label=condition_labels[c])
        for c in COND_QUAD]
    xen_ri = nrow - 1
    fig.legend(handles=cond_handles, loc='center',
               bbox_to_anchor=((col_x(0) + P / 2) / W,
                               1 - (row_top(xen_ri) + 1.5 * qs + GQ) / H),
               frameon=False, title='Condition', ncol=1, fontsize=9,
               title_fontsize=10, handletextpad=0.4, labelspacing=0.5)

    # ---- middle: subclass confusion matrices (one per platform) ----
    # inverted scale: dark = agreement (diagonal); zero / sub-1% entries floor
    # to the light bottom of the colormap. The featured subclass (060) is boxed
    # in red and marked on both axes.
    i0 = order.index(CONF_TARGET)
    im = None
    for j, name in enumerate(datasets):
        ax = fig.add_axes(rect(CONF_LX + j * (CM + GC), conf_mat_top, CM, CM))
        M = np.clip(_confusion(*obsd[name], order), 0.01, 1)
        im = ax.imshow(M, cmap='magma_r', norm=mcolors.LogNorm(0.01, 1),
                       aspect='equal', interpolation='nearest')
        ax.add_patch(Rectangle((i0 - 0.5, i0 - 0.5), 1, 1, fill=False,
                               edgecolor='red', lw=1.2))
        ax.set_xticks([i0]); ax.set_yticks([i0])
        ax.set_xticklabels(['060'], fontsize=7, color='red')
        ax.set_yticklabels(['060'], fontsize=7, color='red')
        ax.tick_params(colors='red', length=4, width=1.0)
        ax.set_title(datasets[name]['display'], fontsize=12)
        ax.set_xlabel('spatial-constrained', fontsize=9, color='black')
        if j == 0:
            ax.set_ylabel('expression-only', fontsize=9, color='black')
        # per-platform metrics beneath each matrix
        m = metrics[name]
        fig.text((CONF_LX + j * (CM + GC) + CM / 2) / W,
                 1 - (conf_mat_bottom + 0.48) / H,
                 f'concordance: {m["class_concord"]:.0%} class · '
                 f'{m["sub_concord"]:.0%} subclass\n'
                 f'query–reference iLISI {m["ilisi"]:.2f} / 2',
                 ha='center', va='top', fontsize=8.5, linespacing=1.4)
    cax = fig.add_axes(rect(CONF_LX + 3 * (CM + GC) - GC + 0.12,
                            conf_mat_top + CM * 0.2, 0.12, CM * 0.55))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('fraction of\nexpression-only cells', fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # ---- bottom: featured-subclass spatial localization (single row) ----
    # query = CAST-aligned x_ffd/y_ffd; reference x_raw/y_raw, same frame.
    # expression-only over-calls OT D3 broadly; the spatial constraint confines
    # it to the olfactory tubercle, matching the atlas (Allen subclass colour).
    def allen(sub):
        m = color_mappings['subclass']
        return (m.get(sub) or m.get(sub.replace('/', '_'))
                or m.get(sub.replace('_', '/')) or DEFAULT_GREY)
    c060 = allen(CONF_TARGET)
    unc060 = q_unc == CONF_TARGET
    con060 = q_con == CONF_TARGET
    ref_left = SP_LX + 2 * QW + QG + GROUPG

    def spat(ax, xy, mask, color, bg_s, hi_s):
        ax.scatter(xy[:, 0], xy[:, 1], s=bg_s, c=BG_GREY, alpha=0.35,
                   linewidths=0, rasterized=True)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=hi_s, c=color,
                   linewidths=0, rasterized=True)
        ax.set_xlim(sx0, sx1); ax.set_ylim(sy0, sy1)
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    # query: expression-only vs spatial-constrained OT D3 call
    a0 = fig.add_axes(rect(SP_LX, sp_top, QW, QH))
    spat(a0, qxy, unc060, c060, 0.1, 0.6)
    a0.set_title('expression-only', fontsize=10)
    a1 = fig.add_axes(rect(SP_LX + QW + QG, sp_top, QW, QH))
    spat(a1, qxy, con060, c060, 0.1, 0.6)
    a1.set_title('spatial-constrained', fontsize=10)
    # reference 2x2: OT D3 in the atlas
    for k, (sid, xy, sub) in enumerate(ref):
        rr, cc = divmod(k, 2)
        ax = fig.add_axes(rect(ref_left + cc * (RW + RG),
                               sp_top + rr * (RH + RG), RW, RH))
        spat(ax, xy, sub == CONF_TARGET, c060, 0.3, 1.4)
        ax.text(0.03, 0.97, f'sec {sid}', transform=ax.transAxes, va='top',
                ha='left', fontsize=8)
    gy = 1 - (sp_top - 0.32) / H
    fig.text((SP_LX + QW + QG / 2) / W, gy, 'Xenium query',
             ha='center', va='bottom', fontsize=12)
    fig.text((ref_left + RW + RG / 2) / W, gy, 'Allen reference',
             ha='center', va='bottom', fontsize=12)
    fig.legend(handles=[Line2D([0], [0], marker='o', linestyle='', markersize=6,
               markerfacecolor=c060, markeredgewidth=0, label=CONF_TARGET)],
               loc='center',
               bbox_to_anchor=(0.5, 1 - (sp_top + QH + 0.30) / H),
               frameon=False, fontsize=9)

    for ext in ['png', 'svg']:
        fig.savefig(f'{working_dir}/figures/figure_supp_2.{ext}',
                    dpi=300, facecolor='white')
    plt.close(fig)
    print('figure saved: figures/figure_supp_2.{png,svg}')

#endregion
#region run ####################################################################

# Combined UMAPs are cached at output/<name>/supp_harmony_umap.npz; the first
# run computes them, later runs reuse the cache. Delete a cache to recompute.
if __name__ == '__main__':
    build_figure()

#endregion
