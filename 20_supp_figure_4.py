"""Supplementary Figure 4: maternal circuits captured but stable.
A) spatial distribution of 19 maternal subclasses per platform;
B) identity-marker dotplot (3-wedge glyphs);
C) gene cards - stable effectors vs pregnancy-changed neuroendocrine genes.
Reuses 10_figure_1.py and 12_figure_helper.py.
"""
import os
import pickle
import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

fig1 = importlib.import_module('10_figure_1')
fc = importlib.import_module('12_figure_helper')

working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/figures'
cache_dir = f'{working_dir}/output/cache'
os.makedirs(cache_dir, exist_ok=True)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'

# =============================================================================
# Config
# =============================================================================
BLOCKS = [
    ('Preoptic / MPOA', [
        '085 SI-MPO-LPO Lhx8 Gaba', '086 MPO-ADP Lhx8 Gaba',
        '106 PVpo-VMPO-MPN Hmx2 Gaba', '118 ADP-MPO Trp73 Glut',
        '124 MPN-MPO-PVpo Hmx2 Glut']),
    ('BNST / CeA-MeA', [
        '073 MEA-BST Sox6 Gaba', '075 MEA-BST Lhx6 Nr2e1 Gaba',
        '076 MEA-BST Lhx6 Nfib Gaba', '080 CEA-AAA-BST Six3 Sp9 Gaba',
        '082 CEA-BST Ebf1 Pdyn Gaba']),
    ('Lateral septum', [
        '067 LSX Sall3 Pax6 Gaba', '068 LSX Otx2 Gaba',
        '069 LSX Nkx2-1 Gaba', '071 LSX Prdm12 Zeb2 Gaba']),
    ('Basal forebrain', [
        '057 NDB-SI-MA-STRv Lhx8 Gaba', '066 NDB-SI-ant Prdm12 Gaba']),
    ('Olfactory tubercle', ['060 OT D3 Folh1 Gaba']),
    ('Cortical amygdala / LPO Glut', [
        '114 COAa-PAA-MEA Barhl2 Glut', '119 SI-MA-LPO-LHA Skor1 Glut']),
]
SUB19 = [s for _, ss in BLOCKS for s in ss]

MARKER_BLOCKS = [
    ('class', ['Gad2', 'Slc17a6']),
    ('MPOA', ['Lhx8', 'Hmx2', 'Gal', 'Calcr', 'Adcyap1', 'Trh']),
    ('BNST/CeA-MeA', ['Cyp26b1', 'Tac2', 'Nr2e1', 'Sst', 'Lhx6', 'Moxd1',
                      'Esr1', 'Adora2a', 'Pdyn']),
    ('LSX', ['Trpc4', 'Nr3c2', 'Pax6', 'Otx2', 'Sp8', 'Zeb2']),
    ('basal', ['Nefh', 'Cntn4']),
    ('OT', ['Drd3', 'Tac1', 'Foxp2']),
    ('cortAmy/Glut', ['Slc17a7', 'Barhl2', 'Tcf7l2', 'Nr2f2']),
]
MARKERS = [g for _, gs in MARKER_BLOCKS for g in gs]

# stable effectors, then changed genes (receptors, then neuropeptides)
CARD_ORDER = [
    'Gal', 'Calcr', 'Pgr', 'Tac2', 'Htr2a',
    'Ar', 'Esr1', 'Oxtr', 'Nr3c1', 'Crh', 'Thrb',
    'Nts', 'Tac1', 'Pdyn', 'Sst', 'Peg3',
]
CARD_GENES = CARD_ORDER
_half = (len(CARD_ORDER) + 1) // 2
CARD_COLS = [CARD_ORDER[:_half], CARD_ORDER[_half:]]
# forced context rows (stable genes have no significant hits)
card_ctx = {
    'Gal':   ['106 PVpo-VMPO-MPN Hmx2 Gaba', '124 MPN-MPO-PVpo Hmx2 Glut'],
    'Calcr': ['068 LSX Otx2 Gaba', '076 MEA-BST Lhx6 Nfib Gaba',
              '085 SI-MPO-LPO Lhx8 Gaba', '106 PVpo-VMPO-MPN Hmx2 Gaba',
              '124 MPN-MPO-PVpo Hmx2 Glut'],
    'Pgr':   ['057 NDB-SI-MA-STRv Lhx8 Gaba', '080 CEA-AAA-BST Six3 Sp9 Gaba',
              '082 CEA-BST Ebf1 Pdyn Gaba', '106 PVpo-VMPO-MPN Hmx2 Gaba',
              '119 SI-MA-LPO-LHA Skor1 Glut', '124 MPN-MPO-PVpo Hmx2 Glut'],
    'Tac2':  ['082 CEA-BST Ebf1 Pdyn Gaba', '106 PVpo-VMPO-MPN Hmx2 Gaba',
              '124 MPN-MPO-PVpo Hmx2 Glut'],
    'Ar':    ['080 CEA-AAA-BST Six3 Sp9 Gaba', '060 OT D3 Folh1 Gaba'],
    'Nts':   ['080 CEA-AAA-BST Six3 Sp9 Gaba', '106 PVpo-VMPO-MPN Hmx2 Gaba'],
    'Crh':   ['085 SI-MPO-LPO Lhx8 Gaba'],
    'Peg3':  ['119 SI-MA-LPO-LHA Skor1 Glut'],
    'Nr3c1': ['119 SI-MA-LPO-LHA Skor1 Glut', '114 COAa-PAA-MEA Barhl2 Glut'],
    'Htr2a': ['114 COAa-PAA-MEA Barhl2 Glut', '124 MPN-MPO-PVpo Hmx2 Glut',
              '076 MEA-BST Lhx6 Nfib Gaba', '119 SI-MA-LPO-LHA Skor1 Glut',
              '106 PVpo-VMPO-MPN Hmx2 Gaba'],
    'Oxtr':  ['082 CEA-BST Ebf1 Pdyn Gaba', '057 NDB-SI-MA-STRv Lhx8 Gaba',
              '071 LSX Prdm12 Zeb2 Gaba'],
    'Esr1':  ['076 MEA-BST Lhx6 Nfib Gaba', '106 PVpo-VMPO-MPN Hmx2 Gaba',
              '124 MPN-MPO-PVpo Hmx2 Glut'],
    'Thrb':  ['071 LSX Prdm12 Zeb2 Gaba', '119 SI-MA-LPO-LHA Skor1 Glut',
              '082 CEA-BST Ebf1 Pdyn Gaba', '066 NDB-SI-ant Prdm12 Gaba'],
    'Tac1':  ['060 OT D3 Folh1 Gaba', '082 CEA-BST Ebf1 Pdyn Gaba',
              '114 COAa-PAA-MEA Barhl2 Glut'],
    'Pdyn':  ['082 CEA-BST Ebf1 Pdyn Gaba', '060 OT D3 Folh1 Gaba'],
    'Sst':   ['075 MEA-BST Lhx6 Nr2e1 Gaba', '082 CEA-BST Ebf1 Pdyn Gaba'],
}
max_rows_map = {g: 6 for g in CARD_GENES}
SUBCOL = fig1.color_mappings['subclass']

# =============================================================================
# Data (cached)
# =============================================================================
def load_data():
    # Panel A: superimpose all pregnancy sections per platform
    cA = f'{cache_dir}/supp_maternal_A2.pkl'
    if os.path.exists(cA):
        with open(cA, 'rb') as f:
            panelA = pickle.load(f)
    else:
        print('panel A: superimposed pregnancy sections')
        panelA = {}
        for name, cfg in fig1.datasets.items():
            obs = fig1.load_obs(name)
            ex = obs[obs['condition'] == 'PREG']
            sub = ex['subclass'].astype(str).values
            x = ex['x_ffd'].values.astype(float)
            y = ex['y_ffd'].values.astype(float)
            m = np.isin(sub, SUB19)
            col = np.array(['#e6e6e6'] * len(ex), dtype=object)
            col[m] = [SUBCOL.get(s, '#333333') for s in sub[m]]
            panelA[name] = dict(x=x, y=y, col=col, is_mat=m,
                                display=cfg['display'],
                                xr=(float(x.min()), float(x.max())),
                                yr=(float(y.min()), float(y.max())))
        with open(cA, 'wb') as f:
            pickle.dump(panelA, f)
    # Panel B: marker dotplot, cached on the marker set
    cB = f'{cache_dir}/supp_maternal_B.pkl'
    panelB = None
    if os.path.exists(cB):
        with open(cB, 'rb') as f:
            bd = pickle.load(f)
        if bd.get('markers') == MARKERS:
            panelB = bd['panelB']
    if panelB is None:
        print('panel B: marker dotplot matrices')
        panelB = {n: fig1.compute_dotplot_aligned(n, MARKERS, SUB19)
                  for n in fig1.datasets}
        with open(cB, 'wb') as f:
            pickle.dump(dict(markers=MARKERS, panelB=panelB), f)
    # Panel C: gene cards
    cC = f'{cache_dir}/supp_maternal_C.pkl'
    if os.path.exists(cC):
        with open(cC, 'rb') as f:
            cd = pickle.load(f)
    else:
        print('panel C: cards')
        de_sr_all, de_sr, de_pp = fc.load_de(working_dir)
        sp_coords, sp_expr, sp_in, _ = fc.load_spatial(
            working_dir, CARD_GENES, CARD_GENES, SUB19)
        cards, _ = fc.build_cards(
            CARD_GENES, card_ctx, max_rows_map, de_sr, de_sr_all, de_pp,
            SUB19, sp_in, sp_coords, sp_expr)
        cd = dict(cards=cards)
        with open(cC, 'wb') as f:
            pickle.dump(cd, f)
    return dict(panelA=panelA, panelB=panelB, cards=cd['cards'])

# =============================================================================
# Layout geometry (inches, top-down)
# =============================================================================
PAD = 0.3
# panel A
A_GAP = 0.12
A_LEGW = 3.0
A_TITLE = 0.28
# panel B
B_CELL = 0.26
B_ROWLAB = 2.35
B_CBAR = 0.12
B_GENELAB = 0.6
WEDGE_R = 0.47
WMIN = 0.15
WANG = {'slidetags': 90, 'merfish': 210, 'xenium': 330}
# panel C
C_SP = 1.02
C_SPGAP = 0.05
C_FOREST = fc.FOREST_W_IN
C_ROWLAB = 2.05
C_TITLE = 0.30
C_GAP = 0.16
C_COLGAP = 0.55
PLAT_COL = fc.PLATFORM_COLORS
PLAT_LAB = fc.PLATFORM_LABELS
SECTION_GAP = 0.55


def wr(f):
    return np.sqrt(WMIN + (1.0 - WMIN) * min(f, 1.0))


def build():
    D = load_data()
    cards = D['cards']

    # show a spatial map only for platforms that also carry a forest bar
    def sp_shown(gd):
        out = []
        for i, ds in enumerate(gd['sp_pair']):
            if any(not np.isnan(r['plat'][ds].get('lfc', np.nan))
                   for r in gd['rows']):
                out.append((ds, gd['sp_data'][i], gd['sp_vranges'][i]))
        return out
    max_sp_n = max((len(sp_shown(cards[g])) for g in CARD_GENES), default=1)

    # horizontal geometry: Panel B defines the reference span
    dot_w = len(MARKERS) * B_CELL
    x_cbar = PAD + B_ROWLAB
    x_dot = x_cbar + B_CBAR
    dot_right = x_dot + dot_w
    lgx = dot_right + 0.35                          # shared A/B legend x
    A_x0 = PAD
    A_secw = (dot_right - A_x0 - 2 * A_GAP) / 3.0
    asp = float(np.median([
        (D['panelA'][n]['yr'][1] - D['panelA'][n]['yr'][0]) /
        (D['panelA'][n]['xr'][1] - D['panelA'][n]['xr'][0])
        for n in fig1.datasets]))
    A_sech = A_secw * asp

    # vertical extents
    A_top = PAD + A_TITLE
    A_LEGH = 3.0
    B_top = A_top + A_sech + SECTION_GAP
    dot_h = len(SUB19) * B_CELL
    B_dot_top = B_top
    B_bot = B_dot_top + dot_h

    def col_h(genes):
        return len(genes) * (C_TITLE + C_SP + C_GAP) - C_GAP
    C_top = B_bot + B_GENELAB + 0.25
    C_h = max(col_h(c) for c in CARD_COLS)
    C_bot = C_top + C_h

    card_w = (max_sp_n * C_SP + (max_sp_n - 1) * C_SPGAP + fc.SP_FOREST_GAP_IN
              + C_FOREST + fc.FOREST_LABEL_GAP_IN + C_ROWLAB)
    C_w = PAD + 2 * card_w + C_COLGAP
    W = max(lgx + A_LEGW, C_w) + PAD
    H = C_bot + 0.3

    fig = plt.figure(figsize=(W, H))

    def rect(x, top, w, h):
        return [x / W, 1 - (top + h) / H, w / W, h / H]

    def bare(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        return ax

    # ======================================================================
    # Panel A: spatial distribution
    # ======================================================================
    for i, name in enumerate(fig1.datasets):
        pa = D['panelA'][name]
        x0 = A_x0 + i * (A_secw + A_GAP)
        ax = bare(fig.add_axes(rect(x0, A_top, A_secw, A_sech)))
        bg_s = 0.5 if name == 'slidetags' else 0.3
        mat_s = 6.0 if name == 'slidetags' else 2.0
        ax.scatter(pa['x'][~pa['is_mat']], pa['y'][~pa['is_mat']], s=bg_s,
                   c='#eaeaea', linewidths=0, rasterized=True)
        ax.scatter(pa['x'][pa['is_mat']], pa['y'][pa['is_mat']], s=mat_s,
                   c=list(pa['col'][pa['is_mat']]), linewidths=0,
                   rasterized=True)
        ax.set_aspect('equal')
        fig.text((x0 + A_secw / 2) / W, 1 - (A_top - 0.05) / H, pa['display'],
                 ha='center', va='bottom', fontsize=12)
    axl = bare(fig.add_axes(rect(lgx, A_top - 0.05, A_LEGW, A_LEGH)))
    axl.set_xlim(0, 1); axl.set_ylim(0, 1)
    yy = 0.99
    dy = 1.0 / 26.0
    for bname, subs in BLOCKS:
        axl.text(0.0, yy, bname, ha='left', va='top', fontsize=8)
        yy -= dy
        for s in subs:
            axl.add_patch(Rectangle((0.02, yy - dy * 0.7), 0.03, dy * 0.6,
                          facecolor=SUBCOL.get(s, '#333'), lw=0))
            axl.text(0.08, yy - dy * 0.4, s, ha='left', va='center',
                     fontsize=6.6)
            yy -= dy

    # ======================================================================
    # Panel B: marker dotplot (rows = SUB19, cols = MARKERS)
    # ======================================================================
    n_g, n_c = len(MARKERS), len(SUB19)
    ax = bare(fig.add_axes(rect(x_dot, B_dot_top, dot_w, dot_h)))
    norm = Normalize(-2, 2); cmap = plt.get_cmap('seismic')
    wedges, absent = [], []
    for name in fig1.datasets:
        Z, F = D['panelB'][name]
        a = WANG[name]
        for gi in range(n_g):
            for ci in range(n_c):
                z, f = Z[gi, ci], F[gi, ci]
                if np.isnan(z):
                    absent.append(Wedge((gi, ci), WEDGE_R * 0.9, a - 60,
                                        a + 60, facecolor='#f0f0f0', lw=0))
                    continue
                if f <= 0:
                    continue
                wedges.append(Wedge((gi, ci), WEDGE_R * wr(f), a - 60, a + 60,
                              facecolor=cmap(norm(z)), edgecolor='#777',
                              lw=0.1))
    ax.add_collection(PatchCollection(absent, match_original=True,
                                      rasterized=True))
    ax.add_collection(PatchCollection(wedges, match_original=True,
                                      rasterized=True))
    ax.set_xlim(-0.5, n_g - 0.5); ax.set_ylim(n_c - 0.5, -0.5)
    ax.set_aspect('equal')
    # block dividers
    rb, st = [], 0
    for _, ss in BLOCKS:
        st += len(ss); rb.append(st)
    for b in rb[:-1]:
        ax.axhline(b - 0.5, color='black', lw=0.8)
    cb, st = [], 0
    for _, gs in MARKER_BLOCKS:
        st += len(gs); cb.append(st)
    for b in cb[:-1]:
        ax.axvline(b - 0.5, color='black', lw=0.8)
    ax.set_xticks(range(n_g))
    ax.set_xticklabels(MARKERS, rotation=45, ha='right',
                       rotation_mode='anchor', fontsize=8)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', length=2, pad=2)
    ax.set_yticks([])
    # subclass colour bar + row labels
    axc = bare(fig.add_axes(rect(x_cbar, B_dot_top, B_CBAR, dot_h)))
    axc.set_xlim(0, 1); axc.set_ylim(n_c - 0.5, -0.5)
    for i, s in enumerate(SUB19):
        axc.add_patch(Rectangle((0, i - 0.5), 1, 1,
                      facecolor=SUBCOL.get(s, '#333'), lw=0))
    for i, s in enumerate(SUB19):
        fig.text((x_cbar - 0.05) / W,
                 1 - (B_dot_top + (i + 0.5) * B_CELL) / H,
                 s, ha='right', va='center', fontsize=8)
    # right legend (below the panel-A legend at the shared x)
    blegy = max(B_dot_top, A_top + A_LEGH + 0.2)
    axk = bare(fig.add_axes(rect(lgx, blegy, 1.3, 1.3)))
    axk.set_xlim(-2.6, 2.6); axk.set_ylim(2.3, -2.6); axk.set_aspect('equal')
    for name in fig1.datasets:
        a = WANG[name]
        axk.add_patch(Wedge((0, 0), 1, a - 60, a + 60,
                      facecolor=PLAT_COL[name], edgecolor='white', lw=1.0))
        ar = np.deg2rad(a)
        ha = ('center' if abs(np.cos(ar)) < 0.3
              else ('left' if np.cos(ar) > 0 else 'right'))
        axk.text(1.5 * np.cos(ar), 1.5 * np.sin(ar), PLAT_LAB[name], ha=ha,
                 va='center', fontsize=8)
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=norm)
    cbax = fig.add_axes(rect(lgx + 0.05, blegy + 1.6, 0.14, 0.7))
    cbar = fig.colorbar(sm, cax=cbax)
    cbar.outline.set_visible(False)
    cbar.set_label('z-scored\nexpression', fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    axf = bare(fig.add_axes(rect(lgx, blegy + 2.55, 1.3, 0.85)))
    axf.set_xlim(0, 3.4); axf.set_ylim(-1, 1.2); axf.set_aspect('equal')
    for xi, fr in zip([0.5, 1.6, 2.8], [0.1, 0.5, 1.0]):
        axf.add_patch(plt.Circle((xi, 0.35), 0.55 * wr(fr),
                      facecolor='#bbb', lw=0))
        axf.text(xi, -0.7, f'{int(fr * 100)}%', ha='center', va='center',
                 fontsize=7)
    axf.text(0.0, 1.05, '% expressing', ha='left', va='bottom', fontsize=8)

    # ======================================================================
    # Panel C: gene cards, two evenly split columns
    # ======================================================================
    def draw_card(x0, top, gene, show_xlab):
        gd = cards[gene]
        fig.text(x0 / W, 1 - top / H, gene, ha='left', va='top',
                 fontsize=fc.TITLE_FS, fontstyle='italic')
        shown = sp_shown(gd)
        for si, (ds, sd, vr) in enumerate(shown):
            sx = x0 + si * (C_SP + C_SPGAP)
            ax = fig.add_axes(rect(sx, top + C_TITLE, C_SP, C_SP))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor('black')
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            ax.set_title(PLAT_LAB[ds], fontsize=7.5, pad=1.5)
            if sd is None or len(sd['x']) == 0:
                continue
            vmin, vmax = vr
            order = np.argsort(sd['expr'])
            s = np.clip(150.0 / np.sqrt(max(len(sd['x']), 1)), 0.3, 5)
            ax.scatter(sd['x'][order], sd['y'][order], c=sd['expr'][order],
                       cmap='viridis', s=s, vmin=vmin, vmax=vmax,
                       linewidths=0, rasterized=True)
            ax.plot([sd['midline'], sd['midline']],
                    [sd['fov_cy'] - sd['fov_half'],
                     sd['fov_cy'] + sd['fov_half']], color='white', lw=0.3)
            ax.set_xlim(sd['fov_cx'] - sd['fov_half'],
                        sd['fov_cx'] + sd['fov_half'])
            ax.set_ylim(sd['fov_cy'] - sd['fov_half'],
                        sd['fov_cy'] + sd['fov_half'])
            ax.set_aspect('equal')
            ax.text(0.25, 0.97, 'Null', transform=ax.transAxes,
                    ha='center', va='top', color='white', fontsize=6)
            ax.text(0.75, 0.97, 'Preg', transform=ax.transAxes,
                    ha='center', va='top', color='white', fontsize=6)
        # forest right of the shown maps
        n_sp = len(shown)
        nrow = len(gd['rows'])
        fh = min(max(nrow * fc.CARD_FOREST_ROW_H, fc.CARD_FOREST_MIN_H), C_SP)
        fbot_top = top + C_TITLE + (C_SP - fh) / 2
        fx = x0 + n_sp * C_SP + (n_sp - 1) * C_SPGAP + fc.SP_FOREST_GAP_IN
        axff = fig.add_axes(rect(fx, fbot_top, C_FOREST, fh))
        fc._draw_forest(axff, gd, show_xlabel=show_xlab)
        trans = axff.get_yaxis_transform()
        labf = 1.0 + fc.FOREST_LABEL_GAP_IN / C_FOREST
        for ri, r in enumerate(gd['rows']):
            axff.text(labf, ri, f"{r['cell_type']} {r['stars']}".strip(),
                      transform=trans, ha='left', va='center',
                      fontsize=fc.LABEL_FS, clip_on=False)

    for ci, genes in enumerate(CARD_COLS):
        cx = PAD + ci * (card_w + C_COLGAP)
        y = C_top
        for j, g in enumerate(genes):
            draw_card(cx, y, g, show_xlab=(j == len(genes) - 1))
            y += C_TITLE + C_SP + C_GAP

    for ext in ['png', 'svg']:
        fig.savefig(f'{out_dir}/figure_supp_4.{ext}', dpi=300,
                    facecolor='white')
    plt.close(fig)
    print(f'wrote {out_dir}/figure_supp_4.png/.svg ({W:.1f}x{H:.1f}in)')


if __name__ == '__main__':
    build()
