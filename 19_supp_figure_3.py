"""Supplementary Figure 3: cell-type spatial co-occurrence shifts.
Local-proximity logFC matrices (A center x B surround) for the 84 shared
subclasses of Figure 1, beside a single global-proportion column. 2x2 layout:
Xenium and MERFISH pregnant-vs-nulliparous (top row) and MERFISH
postpartum-vs-pregnant (bottom left). Glyphs: * FDR<0.10, . nominal p<0.05
(per platform).
"""
import os
import warnings
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig1 = importlib.import_module('10_figure_1')
warnings.filterwarnings('ignore')

working_dir = '/home/karbabi/spatial-pregnancy'
FDR_THRESHOLD = 0.10
NOMINAL_THRESHOLD = 0.05

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300
os.makedirs(f'{working_dir}/figures', exist_ok=True)

local_tt = pd.read_csv(f'{working_dir}/output/proximity/local_diff.csv')
global_tt = pd.read_csv(f'{working_dir}/output/proximity/global_props.csv')

# 84 shared subclasses from Figure 1 (>=100 cells in >=2 platforms)
keep = fig1.included_subclasses()


def get_type(ct):
    if 'Glut' in ct:
        return 'Glut'
    if any(x in ct for x in ['Gaba', 'IMN', 'Chol']):
        return 'Gaba'
    return 'NN'


type_order = ['Glut', 'Gaba', 'NN']
display = sorted(keep, key=lambda c: (type_order.index(get_type(c)), c))


def _insert_gaps(items):
    gapped, labels, prev = [], [], None
    for item in items:
        t = get_type(item)
        if prev is not None and t != prev:
            gapped.append(None)
            labels.append('')
        gapped.append(item)
        labels.append(item)
        prev = t
    return gapped, labels


ab_gapped, ab_labels = _insert_gaps(display)
ab_map = {ct: i for i, ct in enumerate(ab_gapped) if ct is not None}
n_ab = len(ab_gapped)
LOCAL_OFF = 2
n_cols = 1 + 1 + n_ab
valid_cols = [0] + [LOCAL_OFF + j for j in range(n_ab)
                    if ab_gapped[j] is not None]
gap_idx = [i for i, x in enumerate(ab_gapped) if x is None]
segs, start = [], 0
for g in gap_idx:
    segs.append((start, g))
    start = g + 1
segs.append((start, n_ab))


def _sig_glyph(fdr, p):
    if pd.notna(fdr) and fdr < FDR_THRESHOLD:
        return '*'
    if pd.notna(p) and p < NOMINAL_THRESHOLD:
        return '•'
    return ''


def _draw_blocks(ax, rs, cs, lw=0.7):
    for r0, r1 in rs:
        for c0, c1 in cs:
            ax.add_patch(plt.Rectangle(
                (c0, r0), c1 - c0, r1 - r0, fill=False,
                edgecolor='black', linewidth=lw, zorder=5))


def _build_matrix(local_mat, local_sig, global_vec, global_sig):
    mat = np.full((n_ab, n_cols), np.nan)
    sig = np.full((n_ab, n_cols), '', dtype=object)
    for a, i in ab_map.items():
        if a in global_vec:
            mat[i, 0] = global_vec[a]
            sig[i, 0] = global_sig.get(a, '')
        for b, j in ab_map.items():
            if (a, b) in local_mat:
                mat[i, LOCAL_OFF + j] = local_mat[(a, b)]
                sig[i, LOCAL_OFF + j] = local_sig.get((a, b), '')
    return mat, sig


def _panel(contrast, dataset):
    loc = local_tt[(local_tt['contrast'] == contrast) &
                   (local_tt['dataset'] == dataset)]
    lmat = {(r['cell_type_a'], r['cell_type_b']): r['logFC']
            for _, r in loc.iterrows()}
    lsig = {(r['cell_type_a'], r['cell_type_b']):
            _sig_glyph(r['adj.P.Val'], r['P.Value']) for _, r in loc.iterrows()}
    glb = global_tt[(global_tt['contrast'] == contrast) &
                    (global_tt['dataset'] == dataset)]
    gvec = {r['cell_type']: r['logFC'] for _, r in glb.iterrows()}
    gsig = {r['cell_type']: _sig_glyph(r['adj.P.Val'], r['P.Value'])
            for _, r in glb.iterrows()}
    return _build_matrix(lmat, lsig, gvec, gsig)


xen_preg = _panel('PREG_vs_CTRL', 'xenium')
mer_preg = _panel('PREG_vs_CTRL', 'merfish')
mer_pp = _panel('POSTPART_vs_PREG', 'merfish')

# robust colour scale: clip to the 5th / 95th percentiles of shown logFC
# (rare pairs reach |logFC|~50; extremes saturate)
allv = np.concatenate([m[~np.isnan(m)]
                       for m, _ in (xen_preg, mer_preg, mer_pp)])
vmin = float(np.percentile(allv, 5))
vmax = float(np.percentile(allv, 95))

# =============================================================================
# 2x2 layout (inches)
# =============================================================================
CELL = 0.085
PAD = 0.25
TITLE_H = 0.62
YLW_FULL = 1.95        # full cell-type y labels
XLH = 1.7              # full cell-type x labels (rotated)
GAP_MID = 1.05         # gap between columns (numeric y labels + axis title)
ROWGAP = 0.45
HM_W = CELL * n_cols
HM_H = CELL * n_ab
GLYPH_FS, LAB_FS = 3.2, 4.0

XL = PAD + YLW_FULL                     # left-column heatmap left
XR = XL + HM_W + GAP_MID                # right-column heatmap left
top1 = PAD + TITLE_H                    # top-row heatmap top
top2 = top1 + HM_H + XLH + ROWGAP       # bottom-row heatmap top
W = XR + HM_W + PAD
H = top2 + HM_H + XLH + PAD

fig = plt.figure(figsize=(W, H))


def rect(x, top, w, h):
    return [x / W, 1 - (top + h) / H, w / W, h / H]


def draw_hm(x, top, panel, title, xlabels, ylabels, letter, ynum=False):
    mat, sig = panel
    ax = fig.add_axes(rect(x, top, HM_W, HM_H))
    ax.pcolormesh(mat, cmap='PRGn', vmin=vmin, vmax=vmax)
    for i in range(n_ab):
        if ab_gapped[i] is None:
            continue
        for jj in valid_cols:
            if not np.isnan(mat[i, jj]):
                ax.add_patch(plt.Rectangle((jj, i), 1, 1, fill=False,
                             edgecolor='black', linewidth=0.15))
            if sig[i, jj]:
                ax.text(jj + 0.5, i + 0.5, sig[i, jj], ha='center',
                        va='center', color='white', fontsize=GLYPH_FS,
                        fontweight='bold')
    _draw_blocks(ax, segs, [(0, 1)])
    _draw_blocks(ax, segs, [(LOCAL_OFF + c0, LOCAL_OFF + c1)
                            for c0, c1 in segs])
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_ab, 0)
    ax.set_aspect('equal')
    if xlabels is not None:
        ax.set_xticks([0.5] + [LOCAL_OFF + j + 0.5 for j in range(n_ab)])
        ax.set_xticklabels(['Global\nproportions'] + xlabels, rotation=45,
                           ha='right', fontsize=LAB_FS)
    else:
        ax.set_xticks([])
    if ylabels is not None:
        ax.set_yticks(np.arange(n_ab) + 0.5)
        ax.set_yticklabels(ylabels, fontsize=LAB_FS)
    else:
        ax.set_yticks([])
    ax.tick_params(length=0, pad=1)
    ax.set_ylabel('Center cell type', fontsize=8,
                  labelpad=16 if ynum else 66)
    ax.set_xlabel('Surround cell type', fontsize=8, labelpad=54)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.text((x + HM_W / 2) / W, 1 - (top - 0.08) / H, title,
             ha='center', va='bottom', fontsize=8)
    fig.text(x / W, 1 - (top - 0.42) / H, letter, ha='left', va='bottom',
             fontsize=13, fontweight='bold')


num_labels = [l.split()[0] if l else '' for l in ab_labels]
draw_hm(XL, top1, xen_preg, 'Xenium\nPregnant vs Nulliparous',
        ab_labels, ab_labels, 'A')
draw_hm(XR, top1, mer_preg, 'MERFISH\nPregnant vs Nulliparous',
        ab_labels, num_labels, 'B', ynum=True)
draw_hm(XL, top2, mer_pp, 'MERFISH\nPostpartum vs Pregnant',
        ab_labels, ab_labels, 'C')

# colorbar in the empty bottom-right quadrant
sm = plt.cm.ScalarMappable(cmap='PRGn', norm=plt.Normalize(vmin, vmax))
cax = fig.add_axes(rect(XR + HM_W * 0.32, top2 + HM_H * 0.28, 0.18,
                        HM_H * 0.44))
cb = fig.colorbar(sm, cax=cax)
cb.set_label('logFC', fontsize=8)
cb.ax.tick_params(labelsize=6)

fig.savefig(f'{working_dir}/figures/figure_supp_3.png', dpi=300,
            bbox_inches='tight')
fig.savefig(f'{working_dir}/figures/figure_supp_3.svg', bbox_inches='tight')
plt.close()
print(f'wrote figure_supp_3 ({len(keep)} subclasses, {W:.1f}x{H:.1f}in)')
