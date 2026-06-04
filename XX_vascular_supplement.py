"""Supplement: POSTPART-vs-nulliparous vascular dotplots (A, B) + brain-wide
cell-state non-enrichment (densities + heatmap)."""

import os
import re
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr, gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

warnings.filterwarnings('ignore')
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

wd = '/home/karbabi/spatial-pregnancy'
out_dir = f'{wd}/output/cell_state'
fig_dir = f'{wd}/figures/vascular'
cmap = plt.get_cmap('seismic')

VASC = ['330 VLMC NN', '331 Peri NN', '332 SMC NN', '333 Endo NN']
GENE_SET = ['Vcam1', 'Ccn2', 'Cd9', 'Lpl', 'Cd47', 'Pecam1', 'Cdh5',
            'Flt1', 'Pdgfrb', 'Notch1', 'Vegfa']
K = 10
MIN_N = 30
PLATS = [('xenium', 'Xenium'), ('merfish', 'MERFISH'),
         ('slidetags', 'Slide-tags')]
PLAT_KEYS = [p[0] for p in PLATS]
PLAT_LABS = [p[1] for p in PLATS]
COND_COLORS = {'CTRL': '#7209b7', 'PREG': '#b5179e'}
CLASS_ORDER = ['Glutamatergic', 'GABAergic', 'IMN']
LABEL_FS = 7

CONTRAST = 'POSTPART_vs_CTRL'
KEEP_CTS = ['318 Astro-NT NN', '319 Astro-TE NN', '323 Ependymal NN',
            '326 OPC NN', '327 Oligo NN', '330 VLMC NN', '331 Peri NN',
            '332 SMC NN', '333 Endo NN', '334 Microglia NN', '335 BAM NN']
PATHWAY_BANDS = [
    ('Angiogenic sprouting', ['GOBP_VASCULATURE_DEVELOPMENT',
        'GOBP_SPROUTING_ANGIOGENESIS', 'GOBP_VASCULOGENESIS']),
    ('VEGF axis', [
        'GOBP_CELLULAR_RESPONSE_TO_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_STIMULUS',
        'GOBP_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_SIGNALING_PATHWAY',
        'GOBP_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_PRODUCTION']),
    ('Endothelial dynamics', ['GOBP_ENDOTHELIAL_CELL_PROLIFERATION',
        'GOBP_ENDOTHELIAL_CELL_MIGRATION',
        'GOBP_BLOOD_VESSEL_ENDOTHELIAL_CELL_MIGRATION']),
    ('Barrier & ECM', ['GOBP_ESTABLISHMENT_OF_ENDOTHELIAL_BARRIER',
        'GOBP_TIGHT_JUNCTION_ORGANIZATION',
        'GOBP_COLLAGEN_BIOSYNTHETIC_PROCESS']),
]
PATHWAY_LABELS = {
    'GOBP_VASCULATURE_DEVELOPMENT': 'vasculature development',
    'GOBP_SPROUTING_ANGIOGENESIS': 'sprouting angiogenesis',
    'GOBP_VASCULOGENESIS': 'vasculogenesis',
    'GOBP_CELLULAR_RESPONSE_TO_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_STIMULUS':
        'response to VEGF',
    'GOBP_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_SIGNALING_PATHWAY':
        'VEGF signaling',
    'GOBP_VASCULAR_ENDOTHELIAL_GROWTH_FACTOR_PRODUCTION': 'VEGF production',
    'GOBP_ENDOTHELIAL_CELL_PROLIFERATION': 'endothelial proliferation',
    'GOBP_ENDOTHELIAL_CELL_MIGRATION': 'endothelial migration',
    'GOBP_BLOOD_VESSEL_ENDOTHELIAL_CELL_MIGRATION': 'blood-vessel EC migration',
    'GOBP_ESTABLISHMENT_OF_ENDOTHELIAL_BARRIER': 'endothelial barrier',
    'GOBP_TIGHT_JUNCTION_ORGANIZATION': 'tight junction organization',
    'GOBP_COLLAGEN_BIOSYNTHETIC_PROCESS': 'collagen biosynthesis',
}
GENE_BANDS = [
    ('Angiogenic sprouting',
     ['Notch1', 'Notch3', 'Dll4', 'Hey1', 'Angpt2', 'Cspg4']),
    ('VEGF axis', ['Vegfa', 'Vegfc', 'Kdr', 'Flt1', 'Nrp1']),
    ('Endothelial dynamics',
     ['Rgcc', 'Id1', 'Klf4', 'Eng', 'Acvrl1', 'Cdh5', 'Pecam1', 'Pdgfrb']),
    ('Barrier & ECM',
     ['Mfsd2a', 'Slc2a1', 'Tjp1', 'Itgb1', 'Itgav', 'Col4a1', 'Icam1']),
]
BAND_COLORS = {'Angiogenic sprouting': '#0072B2', 'VEGF axis': '#E69F00',
               'Endothelial dynamics': '#009E73', 'Barrier & ECM': '#CC79A7'}
ordered_pathways = [p for _, ps in PATHWAY_BANDS for p in ps]
pathway_band = {p: b for b, ps in PATHWAY_BANDS for p in ps}
ordered_genes = [g for _, gs in GENE_BANDS for g in gs]
gene_band = {g: b for b, gs in GENE_BANDS for g in gs}


def pretty_ceil(v):
    if v == 0:
        return 1.0
    if v >= 1:
        return float(np.ceil(v * 10) / 10)
    if v >= 0.01:
        return float(np.ceil(v * 100) / 100)
    return float(np.ceil(v * 1000) / 1000)


def quant_vmax(arr, ql=0.05, qh=0.95):
    f = np.asarray(arr, float)
    f = f[np.isfinite(f)]
    if f.size == 0:
        return 1.0
    a, b = np.quantile(f, [ql, qh])
    return pretty_ceil(max(abs(a), abs(b)))


def numeric_prefix(ct):
    m = re.match(r'^(\d+)', ct)
    return int(m.group(1)) if m else 9999


def spans(labels):
    out, prev, start = [], labels[0], 0
    for k, l in enumerate(labels[1:], 1):
        if l != prev:
            out.append((prev, start, k - 1))
            prev, start = l, k
    out.append((prev, start, len(labels) - 1))
    return out


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sd = np.sqrt(((len(a) - 1) * a.var(ddof=1)
                  + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (b.mean() - a.mean()) / sd if sd > 0 else np.nan


def nclass(s):
    if 'Glut' in s:
        return 'Glutamatergic'
    if 'Gaba' in s:
        return 'GABAergic'
    return 'IMN'


def run(ds):
    a = sc.read_h5ad(f'{wd}/output/{ds}/03_adata_query_{ds}.h5ad', backed='r')
    vn = (a.var['gene_symbol'].astype(str).tolist()
          if 'gene_symbol' in a.var.columns else list(a.var_names))
    n2i = {n: i for i, n in enumerate(vn)}
    gidx = np.array([n2i[g] for g in GENE_SET if g in n2i])
    nt, ns = len(vn), len(gidx)
    o = a.obs
    base = o['condition'].values != 'POSTPART'
    if ds == 'xenium':
        base = base & (o['sample'].values != 'CTRL_3')
    is_v = base & o['subclass'].isin(VASC).values
    is_n = base & o['subclass'].apply(
        lambda s: ('Glut' in s) or ('Gaba' in s) or ('IMN' in s)).values
    vidx = np.where(is_v)[0]
    uc = np.empty(len(vidx), np.float32)
    for s in range(0, len(vidx), 20000):
        e = min(s + 20000, len(vidx))
        x = a.X[vidx[s:e], :]
        x = x.toarray() if sps.issparse(x) else np.asarray(x)
        x = np.asarray(x, np.float32)
        t = x.sum(1, keepdims=True)
        t[t == 0] = 1
        xn = np.log2(x / t * 1e4 + 1)
        r = rankdata(-xn, axis=1, method='average').astype(np.float32)
        u = r[:, gidx].sum(1) - ns * (ns + 1) / 2
        uc[s:e] = 1 - u / (ns * (nt - ns))
    vo = a.obs.iloc[vidx]
    vasc = pd.DataFrame({'sample': vo['sample'].astype(str).values,
                         'xa': vo['x_affine'].astype(float).values,
                         'ya': vo['y_affine'].astype(float).values, 'uc': uc})
    no = a.obs.iloc[np.where(is_n)[0]]
    neur = pd.DataFrame({'sample': no['sample'].astype(str).values,
                         'cond': no['condition'].astype(str).values,
                         'sub': no['subclass'].astype(str).values,
                         'xa': no['x_affine'].astype(float).values,
                         'ya': no['y_affine'].astype(float).values})
    a.file.close()
    env = np.full(len(neur), np.nan, np.float32)
    for smp in sorted(neur['sample'].unique()):
        vm = (vasc['sample'] == smp).values
        nm = (neur['sample'] == smp).values
        if vm.sum() < K:
            continue
        tree = cKDTree(vasc.loc[vm, ['xa', 'ya']].to_numpy())
        vu = vasc.loc[vm, 'uc'].to_numpy()
        _, ii = tree.query(neur.loc[nm, ['xa', 'ya']].to_numpy(),
                           k=K, workers=-1)
        env[nm] = vu[ii].mean(1)
    neur['env'] = env
    neur = neur.dropna(subset=['env'])
    rows = []
    for sub in sorted(neur['sub'].unique()):
        s = neur[neur['sub'] == sub]
        c = s[s.cond == 'CTRL']['env'].to_numpy()
        p = s[s.cond == 'PREG']['env'].to_numpy()
        if len(c) < MIN_N or len(p) < MIN_N:
            continue
        rows.append(dict(subclass=sub, d=cohens_d(c, p)))
    return neur[['cond', 'env']], pd.DataFrame(rows)


env_cache = {ds: f'{out_dir}/supp_env_{ds}.parquet' for ds in PLAT_KEYS}
m_cache = f'{out_dir}/supplement_subclass_d.csv'
if all(os.path.exists(p) for p in env_cache.values()) \
        and os.path.exists(m_cache):
    data = {ds: pd.read_parquet(env_cache[ds]) for ds in PLAT_KEYS}
    M = pd.read_csv(m_cache)
else:
    data, rank = {}, {}
    for ds in PLAT_KEYS:
        neur, rk = run(ds)
        data[ds] = neur
        neur.to_parquet(env_cache[ds])
        rank[ds] = rk
    M = rank['xenium'][['subclass', 'd']].rename(columns={'d': 'xenium'})
    for ds in ['merfish', 'slidetags']:
        M = M.merge(rank[ds][['subclass', 'd']].rename(columns={'d': ds}),
                    on='subclass', how='outer')
    M.to_csv(m_cache, index=False)

allp = M.dropna(subset=PLAT_KEYS).copy()
for p in PLAT_KEYS:
    allp[f'z_{p}'] = (allp[p] - allp[p].mean()) / allp[p].std()
allp['nclass'] = allp['subclass'].map(nclass)
allp['num'] = allp['subclass'].map(
    lambda s: int(s.split()[0]) if s.split()[0].isdigit() else 9999)
allp['crank'] = allp['nclass'].map({c: i for i, c in enumerate(CLASS_ORDER)})
allp = allp.sort_values(['crank', 'num']).reset_index(drop=True)
Z = allp[[f'z_{p}' for p in PLAT_KEYS]].to_numpy().T
rhos = {f'{a}-{b}': spearmanr(allp[a], allp[b])[0] for a, b in
        [('xenium', 'merfish'), ('xenium', 'slidetags'),
         ('merfish', 'slidetags')]}
mrho = {'Xenium': (rhos['xenium-merfish'] + rhos['xenium-slidetags']) / 2,
        'MERFISH': (rhos['xenium-merfish'] + rhos['merfish-slidetags']) / 2,
        'Slide-tags': (rhos['xenium-slidetags']
                       + rhos['merfish-slidetags']) / 2}
ylabels = [f'{lab}\n' + r'$\bar{\rho}$ = ' + f'{mrho[lab]:+.2f}'
           for lab in PLAT_LABS]

nP, nC = len(ordered_pathways), len(KEEP_CTS)
nG = len(ordered_genes)
ci = {c: j for j, c in enumerate(KEEP_CTS)}
ri = {p: i for i, p in enumerate(ordered_pathways)}
gi = {g: i for i, g in enumerate(ordered_genes)}
SUBSET = ['slidetags', 'merfish']

rg = pd.read_parquet(f'{wd}/output/gsea/perms/real_gsea.parquet',
                     columns=['pathway', 'cell_type', 'NES', 'pvalue',
                              'dataset', 'contrast'])
rg = rg[(rg.contrast == CONTRAST) & rg.dataset.isin(SUBSET)
        & rg.pathway.isin(ordered_pathways) & rg.cell_type.isin(KEEP_CTS)]
rg['nlp'] = -np.log10(np.clip(rg['pvalue'].values, 1e-4, 1))
gA = rg.groupby(['pathway', 'cell_type']).agg(
    nes=('NES', 'median'), nlp=('nlp', 'median'),
    nsig=('pvalue', lambda s: int((s < 0.05).sum()))).reset_index()
nes_mat = np.full((nP, nC), np.nan)
nlpA_mat = np.full((nP, nC), np.nan)
dA_mat = np.zeros((nP, nC), int)
for _, r in gA.iterrows():
    i, j = ri[r['pathway']], ci[r['cell_type']]
    nes_mat[i, j] = r['nes']
    nlpA_mat[i, j] = r['nlp']
    dA_mat[i, j] = r['nsig']

de = pd.read_csv(f'{wd}/output/de/de_results.csv',
                 usecols=['gene', 'cell_type', 'logFC', 'PValue',
                          'ref_pct_detected', 'dataset', 'contrast'])
de = de[(de.contrast == CONTRAST) & de.dataset.isin(SUBSET)
        & de.gene.isin(ordered_genes) & de.cell_type.isin(KEEP_CTS)
        & de.logFC.notna()]
gB = de.groupby(['gene', 'cell_type']).agg(
    lfc=('logFC', 'median'), pct=('ref_pct_detected', 'median'),
    nsig=('PValue', lambda s: int((s < 0.05).sum()))).reset_index()
lfc_mat = np.full((nG, nC), np.nan)
pct_mat = np.full((nG, nC), np.nan)
dB_mat = np.zeros((nG, nC), int)
for _, r in gB.iterrows():
    i, j = gi[r['gene']], ci[r['cell_type']]
    lfc_mat[i, j] = r['lfc']
    pct_mat[i, j] = r['pct']
    dB_mat[i, j] = r['nsig']

NES_VMAX = quant_vmax(nes_mat)
LFC_VMAX = quant_vmax(lfc_mat)
norm_nes = mcolors.Normalize(-NES_VMAX, NES_VMAX)
norm_lfc = mcolors.Normalize(-LFC_VMAX, LFC_VMAX)
NLP_MIN, NLP_MAX = 1.30, 5.0
SIZE_MIN, SIZE_MAX, SIZE_MAX_A, SIG_DOT = 16.0, 80.0, 140.0, 4.0


def nlp_to_size(x):
    f = (np.clip(x, NLP_MIN, NLP_MAX) - NLP_MIN) / (NLP_MAX - NLP_MIN)
    return SIZE_MIN + f * (SIZE_MAX_A - SIZE_MIN)


def pct_to_size(p):
    return SIZE_MIN + np.clip(p, 0, 100) / 100 * (SIZE_MAX - SIZE_MIN)


SCC = pd.read_csv('/home/karbabi/single-cell/ABC/metadata/cells_joined.csv',
                  usecols=['subclass', 'subclass_color']).drop_duplicates()
SCC = dict(zip(SCC['subclass'].str.replace('/', '_'), SCC['subclass_color']))
SUBCOL = {k.replace('_', '/'): v for k, v in SCC.items()}

CW, BSW, GP = 0.185, 0.06, 0.135
AX_W = CW * nC
PATH_PITCH = 0.17
A_H = PATH_PITCH * nP
B_H = GP * nG

# class-grouped heatmap ordering (rows = subclasses; sets the column height)
GAP = 1
order = []
for c in CLASS_ORDER:
    idx = list(np.where(allp['nclass'].values == c)[0])
    if not idx:
        continue
    if order:
        order += [None] * GAP
    order += idx
Zp = np.full((3, len(order)), np.nan)
for i, k in enumerate(order):
    if k is not None:
        Zp[:, i] = Z[:, k]
real = [(i, k) for i, k in enumerate(order) if k is not None]
nOrd = len(order)

# === column layout: dotplots | densities | heatmap (left to right) ===
MARG, TOPM, BOTM, COL_GAP = 0.12, 0.28, 0.12, 0.45
# col 1: legends (left, right-justified, near dotplot) | y-labels | A over B
leg_w_in, GAP_LEG_DOT, LAB1 = 1.0, 0.08, 1.20
GAP_AB, B_XLAB = 0.45, 1.00
leg_left_in = MARG + 0.22
c1_dot_l = leg_left_in + leg_w_in + GAP_LEG_DOT + LAB1
c1_band_l = c1_dot_l + AX_W + 0.02
c1_right = c1_band_l + BSW
col1_h = A_H + GAP_AB + B_H + 0.02 + BSW + B_XLAB
plot_h = A_H + GAP_AB + B_H                       # dotplot plot area
# col 2: dynamics scatters (PREG-vs-Null x vs POSTPART-vs-PREG y), stacked
SC_YLAB, SC_TITLE, SC_XLAB, SC_GAP = 0.45, 0.22, 0.42, 0.45
SC_H = (plot_h - 2 * SC_TITLE - SC_XLAB - SC_GAP) / 2
SC_W = SC_H
sc_l = c1_right + COL_GAP + SC_YLAB
sc_r = sc_l + SC_W
# col 3: density plots spanning the dotplot plot area
DENS_YLAB, DENS_W = 0.40, 1.75
DTITLE, DENS_XLAB, DGAP = 0.24, 0.45, 0.45
DENS_H = (plot_h - 3 * DTITLE - 2 * DENS_XLAB - 2 * DGAP) / 3
c2_dens_l = sc_r + COL_GAP + DENS_YLAB
c2_right = c2_dens_l + DENS_W
# col 3: swapped heatmap (subclass rows, platform cols), labels at bottom
HM_CLASSLAB, HM_CELL_W, HM_SUBLAB = 0.30, 0.34, 1.95
hm_w = HM_CELL_W * 3
c3_class_l = c2_right + COL_GAP
c3_hm_l = c3_class_l + HM_CLASSLAB
c3_sublab_l = c3_hm_l + hm_w + 0.04
c3_sublab_r = c3_sublab_l + HM_SUBLAB
HM_CBAR_W, HM_CBAR_GAP = 0.10, 0.35
c3_cbar_l = c3_sublab_r + HM_CBAR_GAP
c3_right = c3_cbar_l + 1.0

content_h = col1_h
FW = c3_right + MARG
FH = TOPM + content_h + BOTM
TOP_Y = FH - TOPM
CBAR_W_FIG = 0.005
cbar_x_fig = (leg_left_in + 0.95 * leg_w_in) / FW - CBAR_W_FIG / 2

a_top = TOP_Y
a_b = a_top - A_H
b_top = a_b - GAP_AB
b_b = b_top - B_H
hm_b, hm_h = b_b, a_top - b_b


def ax_in(x, y, w, h):
    return fig.add_axes([x / FW, y / FH, w / FW, h / FH])


fig = plt.figure(figsize=(FW, FH))


def draw_dot(ax, nr, smat, vmat, norm, dmat, ylabs, bands, italic):
    xs, ys, ss, cc, ec, lw, sx, sy = [], [], [], [], [], [], [], []
    for i in range(nr):
        for j in range(nC):
            if np.isnan(smat[i, j]):
                continue
            xs.append(j)
            ys.append(i)
            ss.append(smat[i, j])
            v = vmat[i, j]
            col = (cmap(norm(np.clip(v, norm.vmin, norm.vmax)))
                   if not np.isnan(v) else (0.6, 0.6, 0.6, 1.0))
            a = 1.0 if dmat[i, j] >= 1 else 0.5
            cc.append((col[0], col[1], col[2], a))
            ec.append('#222222' if dmat[i, j] == 2 else 'none')
            lw.append(0.9 if dmat[i, j] == 2 else 0.0)
            if dmat[i, j] >= 1:
                sx.append(j)
                sy.append(i)
    ax.scatter(xs, ys, s=ss, c=cc, edgecolors=ec, linewidths=lw, zorder=3)
    if sx:
        ax.scatter(sx, sy, s=SIG_DOT, c='white', edgecolors='none', zorder=4)
    for k in range(1, nC):
        ax.axvline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
    for k in range(1, nr):
        ax.axhline(k - 0.5, color='#F2F2F2', lw=0.25, zorder=1)
    for _, _, hi in spans(bands)[:-1]:
        ax.axhline(hi + 0.5, color='#BBBBBB', lw=0.4, zorder=2)
    ax.set_xlim(-0.5, nC - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks(range(nr))
    ax.set_yticklabels(ylabs, fontsize=6.5,
                       fontstyle='italic' if italic else 'normal')
    ax.tick_params(axis='y', length=2, pad=1)
    for s in ax.spines.values():
        s.set_linewidth(0.9)


def band_strip(left, bot, h, bands):
    an = ax_in(left, bot, BSW, h)
    an.set_xlim(0, 1)
    an.set_ylim(len(bands) - 0.5, -0.5)
    an.axis('off')
    for i, bn in enumerate(bands):
        an.add_patch(plt.Rectangle((0, i - 0.5), 1, 1,
                     facecolor=BAND_COLORS[bn], edgecolor='none'))


def col_strip(left, bot, names=True):
    cs = ax_in(left, bot, AX_W, BSW)
    cs.set_xlim(-0.5, nC - 0.5)
    cs.set_ylim(0, 1)
    for j, c in enumerate(KEEP_CTS):
        cs.add_patch(plt.Rectangle((j - 0.5, 0), 1, 1,
                     facecolor=SUBCOL.get(c, '#ccc'), edgecolor='none'))
    cs.set_yticks([])
    if names:
        cs.set_xticks(range(nC))
        cs.set_xticklabels(KEEP_CTS, rotation=45, ha='right',
                           rotation_mode='anchor', fontsize=6.5)
        cs.tick_params(axis='x', length=2, pad=1)
    else:
        cs.set_xticks([])
    for s in cs.spines.values():
        s.set_visible(False)


ax_a = ax_in(c1_dot_l, a_b, AX_W, A_H)
draw_dot(ax_a, nP, nlp_to_size(nlpA_mat), nes_mat, norm_nes, dA_mat,
         [PATHWAY_LABELS[p] for p in ordered_pathways],
         [pathway_band[p] for p in ordered_pathways], False)
ax_a.set_ylabel('GSEA pathway', fontsize=8, labelpad=2)
band_strip(c1_band_l, a_b, A_H, [pathway_band[p] for p in ordered_pathways])
col_strip(c1_dot_l, a_b - 0.02 - BSW, names=False)

ax_b = ax_in(c1_dot_l, b_b, AX_W, B_H)
draw_dot(ax_b, nG, pct_to_size(pct_mat), lfc_mat, norm_lfc, dB_mat,
         ordered_genes, [gene_band[g] for g in ordered_genes], True)
ax_b.set_ylabel('DE gene', fontsize=8, labelpad=2)
band_strip(c1_band_l, b_b, B_H, [gene_band[g] for g in ordered_genes])
col_strip(c1_dot_l, b_b - 0.02 - BSW, names=True)

def leg_axes(bot, h):
    ax = fig.add_axes([leg_left_in / FW, bot / FH, leg_w_in / FW, h / FH],
                      zorder=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


LA_T = b_top
LA_B = LA_T - 8 * GP
LB_T = LA_B - GP
LB_B = LB_T - 8 * GP
LS_T = LB_B - GP
LS_B = LS_T - 8 * GP
TL_T = LS_B - GP
TL_B = TL_T - 6 * GP

la = leg_axes(LA_B, 8 * GP)
la.text(1.0, 0.96, 'GSEA\n' + r'$-\log_{10}$ $p$', ha='right', va='top',
        fontsize=6.8, linespacing=1.0)
for k, lv in enumerate([1.5, 2.5, 4.0]):
    y = 0.72 - k * 0.13
    la.scatter([0.95], [y], s=nlp_to_size(lv), c=['#555555'], edgecolors='none')
    la.text(0.85, y, f'{lv:.1f}', ha='right', va='center', fontsize=6.5)
la.text(1.0, 0.30, 'GSEA\nNES (median)', ha='right', va='top', fontsize=6.8,
        linespacing=1.0)
ca = fig.add_axes([cbar_x_fig, LA_B / FH + 8 * GP / FH * 0.02, CBAR_W_FIG,
                   8 * GP / FH * 0.16], zorder=110)
cba = fig.colorbar(plt.cm.ScalarMappable(norm_nes, cmap), cax=ca,
                   orientation='vertical')
cba.set_ticks([-NES_VMAX, 0, NES_VMAX])
cba.ax.yaxis.tick_left()
cba.ax.tick_params(labelsize=6.0, length=2, pad=1)

lb = leg_axes(LB_B, 8 * GP)
lb.text(1.0, 0.96, 'DE\nlogFC (median)', ha='right', va='top', fontsize=6.8,
        linespacing=1.0)
cbx = fig.add_axes([cbar_x_fig, LB_B / FH + 8 * GP / FH * 0.60, CBAR_W_FIG,
                    8 * GP / FH * 0.16], zorder=110)
cbb = fig.colorbar(plt.cm.ScalarMappable(norm_lfc, cmap), cax=cbx,
                   orientation='vertical')
cbb.set_ticks([-LFC_VMAX, 0, LFC_VMAX])
cbb.ax.yaxis.tick_left()
cbb.ax.tick_params(labelsize=6.0, length=2, pad=1)
lb.text(1.0, 0.45, 'DE\nPercent expressed', ha='right', va='top',
        fontsize=6.8, linespacing=1.0)
for k, lv in enumerate([10, 50, 90]):
    y = 0.24 - k * 0.10
    lb.scatter([0.95], [y], s=pct_to_size(lv), c=['#777777'], edgecolors='none')
    lb.text(0.85, y, f'{lv}%', ha='right', va='center', fontsize=6.5)

ls = leg_axes(LS_B, 8 * GP)
ls.text(1.0, 0.95, 'Significance', ha='right', va='top', fontsize=6.8)
ls.scatter([0.95], [0.74], s=55, c=['#bbbbbb'], edgecolors='none')
ls.scatter([0.95], [0.74], s=SIG_DOT, c='white', edgecolors='none')
ls.text(0.85, 0.74, r'$p<0.05$ (one)', ha='right', va='center', fontsize=6.5)
ls.scatter([0.95], [0.55], s=55, c=['#bbbbbb'], edgecolors='#222222',
           linewidths=0.9)
ls.scatter([0.95], [0.55], s=SIG_DOT, c='white', edgecolors='none')
ls.text(0.85, 0.55, r'$p<0.05$ (both)', ha='right', va='center', fontsize=6.5)
ls.scatter([0.95], [0.36], s=55, c=['#bbbbbb'], edgecolors='none')
ls.text(0.85, 0.36, 'tested, n.s.', ha='right', va='center', fontsize=6.5)

tl = leg_axes(TL_B, 6 * GP)
bn = [b for b, _ in GENE_BANDS]
yt, yb = 0.85, 0.30
yst = (yt - yb) / (len(bn) - 1)
for i, b in enumerate(bn):
    y = yt - i * yst
    tl.add_patch(plt.Rectangle((0.85, y - 0.06), 0.10, 0.12,
                 facecolor=BAND_COLORS[b], edgecolor='none'))
    tl.text(0.82, y, b, ha='right', va='center', fontsize=6.8)

# col 2: gene-expression dynamics scatters (per dataset, stacked)
de_dyn = pd.read_csv(f'{wd}/output/de/de_results.csv',
                     usecols=['gene', 'cell_type', 'logFC', 'PValue',
                              'dataset', 'contrast'])
de_dyn = de_dyn[de_dyn.gene.isin(ordered_genes)
                & de_dyn.cell_type.isin(KEEP_CTS)
                & de_dyn.dataset.isin(SUBSET) & de_dyn.logFC.notna()]
bands = [b for b, _ in GENE_BANDS]
sblk = SC_TITLE + SC_H + SC_XLAB + SC_GAP
for j, (ds, lab) in enumerate([('slidetags', 'Slide-tags'),
                               ('merfish', 'MERFISH')]):
    s = de_dyn[de_dyn.dataset == ds]
    px = s[s.contrast == 'PREG_vs_CTRL'][['gene', 'cell_type', 'logFC',
                                          'PValue']]
    py = s[s.contrast == 'POSTPART_vs_PREG'][['gene', 'cell_type', 'logFC']]
    m = px.merge(py, on=['gene', 'cell_type'], suffixes=('_preg', '_post'))
    m['band'] = m['gene'].map(gene_band)
    xa, ya = m['logFC_preg'].to_numpy(), m['logFC_post'].to_numpy()
    sig = (m['PValue'].to_numpy() < 0.05)
    sc_b = a_top - j * sblk - SC_TITLE - SC_H
    ax = ax_in(sc_l, sc_b, SC_W, SC_H)
    lim = max(0.3, float(np.nanpercentile(np.abs(np.r_[xa, ya]), 98)))
    ax.axhline(0, color='#cfcfcf', lw=0.5, zorder=1)
    ax.axvline(0, color='#cfcfcf', lw=0.5, zorder=1)
    ax.plot([-lim, lim], [lim, -lim], color='#9a9a9a', lw=0.6, ls='--',
            zorder=1)
    for bn in bands:
        mb = (m['band'] == bn).to_numpy()
        ax.scatter(xa[mb & ~sig], ya[mb & ~sig], s=7, c=BAND_COLORS[bn],
                   alpha=0.30, edgecolors='none', zorder=2)
        ax.scatter(xa[mb & sig], ya[mb & sig], s=22, c=BAND_COLORS[bn],
                   alpha=0.95, edgecolors='white', linewidths=0.3, zorder=3)
    rmask = sig if sig.sum() >= 5 else np.isfinite(xa) & np.isfinite(ya)
    r = (np.corrcoef(xa[rmask], ya[rmask])[0, 1]
         if rmask.sum() > 2 else np.nan)
    ax.set_title(f'{lab}   r = {r:+.2f}', fontsize=8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('logFC  PREG vs Null', fontsize=7.5)
    ax.set_ylabel('logFC  POSTPART vs PREG', fontsize=7.5)
    ax.tick_params(labelsize=6.5, length=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

dblk = DTITLE + DENS_H + DENS_XLAB + DGAP
for j, (ds, lab) in enumerate(PLATS):
    d_b = TOP_Y - j * dblk - DTITLE - DENS_H
    ax = ax_in(c2_dens_l, d_b, DENS_W, DENS_H)
    n = data[ds]
    cv = n[n.cond == 'CTRL']['env'].to_numpy()
    pv = n[n.cond == 'PREG']['env'].to_numpy()
    lo, hi = np.quantile(np.concatenate([cv, pv]), [0.01, 0.99])
    grid = np.linspace(lo, hi, 300)
    for vv, cd, nm in [(cv, 'CTRL', 'Nulliparous'), (pv, 'PREG', 'Pregnancy')]:
        kk = gaussian_kde(vv, bw_method=0.35)(grid)
        ax.fill_between(grid, 0, kk, color=COND_COLORS[cd], alpha=0.40,
                        lw=0.9, edgecolor=COND_COLORS[cd], label=nm)
        ax.vlines(np.median(vv), 0, np.interp(np.median(vv), grid, kk),
                  color=COND_COLORS[cd], lw=1.2, ls='--')
    ax.set_ylim(bottom=0)
    ax.set_title(f'{lab}   d = {cohens_d(cv, pv):+.2f}', fontsize=8)
    ax.set_xlabel('Vascular-environment score', fontsize=7.5)
    ax.set_ylabel('Density', fontsize=8)
    ax.set_yticks([])
    ax.tick_params(labelsize=6.5, length=2)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    if j == 0:
        ax.legend(fontsize=6.3, frameon=False, loc='upper right')

cmap2 = cmap.copy()
cmap2.set_bad('white')
xlab_plat = [f'{lab}  ' + r'$\bar{\rho}$=' + f'{mrho[lab]:+.2f}'
             for lab in PLAT_LABS]
axh = ax_in(c3_hm_l, hm_b, hm_w, hm_h)
im = axh.imshow(np.ma.masked_invalid(Zp.T), aspect='auto', cmap=cmap2,
                vmin=-2.5, vmax=2.5, interpolation='nearest')
axh.set_xticks(range(3))
axh.set_xticklabels(xlab_plat, rotation=45, ha='right',
                    rotation_mode='anchor', fontsize=LABEL_FS)
axh.tick_params(axis='x', length=2, pad=1)
axh.set_yticks([i for i, _ in real])
axh.yaxis.set_ticks_position('right')
axh.set_yticklabels([allp['subclass'].iloc[k] for _, k in real],
                    fontsize=LABEL_FS)
for lbl in axh.get_yticklabels():
    lbl.set_ha('left')
axh.tick_params(axis='y', length=1.5, pad=2)
for sp in axh.spines.values():
    sp.set_visible(False)
for c in CLASS_ORDER:
    pos = [i for i, k in real if allp['nclass'].iloc[k] == c]
    if pos:
        axh.text(-0.95, np.mean(pos), c, ha='center', va='center',
                 rotation=90, fontsize=8, clip_on=False)
        axh.add_patch(plt.Rectangle((-0.5, min(pos) - 0.5), 3.0,
                      max(pos) - min(pos) + 1, fill=False,
                      edgecolor='black', lw=0.8, clip_on=False))
cbar_h = 1.5
cax = ax_in(c3_cbar_l, a_top - cbar_h, HM_CBAR_W, cbar_h)
cb = fig.colorbar(im, cax=cax)
cb.ax.yaxis.set_ticks_position('right')
cb.ax.tick_params(labelsize=6.5, length=2)
fig.text(c3_cbar_l / FW, (a_top + 0.05) / FH,
         'env-score effect\n(z per platform)', ha='left', va='bottom',
         fontsize=7, linespacing=1.0)

for ext in ('png', 'svg'):
    fig.savefig(f'{fig_dir}/vascular_supplement.{ext}',
                bbox_inches='tight', facecolor='white')
print(f'wrote {fig_dir}/vascular_supplement.png/.svg')
