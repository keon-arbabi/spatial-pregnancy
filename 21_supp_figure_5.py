"""Supplementary Figure 5: the CD31 vascular null, and why it holds.

Figure 4F already carries the result (representative fields for all nine
animals, four metrics with effect sizes). This supplement exists only to
defend the null against the ways it can be attacked:

A) the segmentation is wrong        -> manual vs operator-independent
                                       agreement, every analysed field shown
B) it is an underpowered null       -> all 18 measures with confidence bounds
                                       and the effect each could have detected
C) fields are pseudoreplicates      -> the spread within and between animals
D) the analysis was chosen to fail  -> the same comparison under every
                                       denominator, despeckling threshold,
                                       segmentation and domain radius
E) the groups differ technically    -> covariate balance

Layout: A spans the full width; B/C and D/E are two columns beneath it, sharing
one body height so the two columns start and finish on the same lines.

The figure carries no titles, captions or footnotes. Everything below belongs
in the legend instead:
  A  the large tiles are the typical nulliparous field, the typical pregnant
     field and the worst manual-vs-automatic agreement of the 28. CD31 grey,
     manual mask green, pixels found only by the automatic threshold red,
     tissue magenta, analysed domain yellow, guarded core cyan. Agreement
     strip: black line the mean Dice, red line the 0.80 QC threshold.
  B  grey caps mark the smallest change detectable at 80% power. The open
     marker is the one measure quantised on the pixel grid (5 distinct values
     across 28 fields), whose p-values are tie-dominated.
  C  bars are animal means; fields within an animal are technical replicates.
  D  filled markers are the settings used in the reported analysis, dashed
     line the reported effect, shading |g| < 0.2.
  E  red line p = 0.05.

The exploratory size-class and heterogeneity measures are deliberately not
shown: they were not part of the pre-specified battery. They are reported in
full in the supplementary workbook.

Reads  output/vascular_imaging_20260720/{roi_metrics,animal_means,
       stats_primary,stats_sensitivity,confounders,robustness_*}.csv
       and qc/*.png
Writes figures/figure_supp_5.{png,svg}
"""

#region imports and config #####################################################

import os
import importlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from PIL import Image

fc = importlib.import_module('12_figure_helper')
fc.setup_style()
Image.MAX_IMAGE_PIXELS = None

working_dir = '/home/karbabi/spatial-pregnancy'
OUT = f'{working_dir}/output/vascular_imaging_20260720'
FIG = f'{working_dir}/figures'
os.makedirs(FIG, exist_ok=True)

COND = ['Nulliparous', 'Pregnant']
CCOL = dict(zip(COND, fc.IF_BAR_COLORS))
FAIL = '#d1495b'
GREY = '#8a8a8a'
AUTO_DICE_MIN = 0.80
FS = 6.6                       # ticks
FS_LAB = 7.4                   # axis labels
FAMILIES = ['Density', 'Architecture', 'Caliber', 'Supply', 'Cellularity']

# a few battery labels are too long for the column; shortened here only
SHORT = {'Perivascular nuclei per mm vessel': 'Perivascular nuclei / mm vessel',
         '95th pct distance to vessel (µm)': '95th pct dist. to vessel (µm)'}

rng = np.random.default_rng(0)

#endregion

#region geometry ###############################################################
# Every axes is placed by its top-left corner in inches. Panel A spans the full
# text width; B/C and D/E are two columns beneath it. Each column is a fixed
# (label strip + plot strip) budget, and both columns share one body height, so
# plot edges line up horizontally and the columns end on the same line.

FIG_W = 7.2
LEFT, RIGHT = 0.60, 0.10
USABLE = FIG_W - LEFT - RIGHT
PANEL_X = 0.08
TOP, BOT = 0.20, 0.12

# --- A: two rows, left block (tiles) and right block (agreement) in both
A_BLOCK_W = 4.40                            # left block, all of panel A
A_BIG = (A_BLOCK_W - 2 * 0.10) / 3          # three square exemplar tiles
A_COLS, A_ROWS, A_TGAP = 10, 3, 0.025
A_TILE = (A_BLOCK_W - (A_COLS - 1) * A_TGAP) / A_COLS
A_R1 = A_BIG
A_R2 = A_ROWS * A_TILE + (A_ROWS - 1) * A_TGAP
A_ROWGAP = 0.42                             # clears the scatter's x label
A_RIGHT_X = LEFT + A_BLOCK_W + 0.62         # room for the scatter's y label
A_RIGHT_W = FIG_W - RIGHT - A_RIGHT_X
A_H = A_R1 + A_ROWGAP + A_R2

# --- two columns
COL_GAP = 0.60
COL_L_W = 3.40
COL_R_W = USABLE - COL_L_W - COL_GAP
COL_L_X = LEFT
COL_R_X = COL_L_X + COL_L_W + COL_GAP
L_LAB_W = 1.68
R_LAB_W = 1.25
L_PLOT_X, L_PLOT_W = COL_L_X + L_LAB_W, COL_L_W - L_LAB_W
R_PLOT_X, R_PLOT_W = COL_R_X + R_LAB_W, COL_R_W - R_LAB_W
R_LETTER_X = COL_R_X - 0.40

# left column: B over C; right column: D over E; one shared body height
B_ROW = 0.150
B_H = (18 + len(FAMILIES)) * B_ROW
C_H = 1.05
GAP_L = 0.55                                # B has a two-line x label
BODY_H = B_H + GAP_L + C_H

E_H = 0.80
GAP_R = 0.62                                # D has a two-line x label
D_H = BODY_H - GAP_R - E_H
D_UNITS = 15 + 4 * 0.6 + 1                  # 15 rows, 4 group gaps, 1 LOO row

GAP_AB = 0.55
BODY_XLAB = 0.46                            # C's rotated ticks, E's x label
FIG_H = TOP + A_H + GAP_AB + BODY_H + BODY_XLAB + BOT

fig = plt.figure(figsize=(FIG_W, FIG_H))


def ax_in(x, y, w, h):
    """Axes placed by top-left corner, inches from the top-left of the figure."""
    return fig.add_axes([x / FIG_W, (FIG_H - y - h) / FIG_H,
                         w / FIG_W, h / FIG_H])


def letter(ch, y, x=PANEL_X):
    fig.text(x / FIG_W, (FIG_H - y + 0.07) / FIG_H, ch, fontsize=10.5,
             va='top', ha='left')

#endregion

#region helpers ################################################################


def despine(ax, keep=('left', 'bottom')):
    for s in ('top', 'right', 'left', 'bottom'):
        ax.spines[s].set_visible(s in keep)
    ax.tick_params(labelsize=FS, length=2, width=0.6)


def square_thumb(path, px):
    """Thumbnail letterboxed onto a square canvas, so a grid of fields of
    different shapes still tiles cleanly without distorting any of them."""
    img = Image.open(path).convert('RGB')
    img.thumbnail((px, px), Image.LANCZOS)
    canvas = Image.new('RGB', (px, px), (0, 0, 0))
    canvas.paste(img, ((px - img.width) // 2, (px - img.height) // 2))
    return np.asarray(canvas)


def show_overlay(ax, name, edge, lw=0.8, px=340):
    ax.imshow(square_thumb(f'{OUT}/qc/{name}.png', px), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(lw)
        s.set_color(edge)


def row_labels(ax, ticks, size):
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([l for _, l in ticks], fontsize=size)
    ax.tick_params(axis='y', length=0)


def group_labels(ax, heads, size):
    for yy, name in heads:
        ax.text(-0.012, yy, name, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=size, style='italic',
                color='#555555')


def typical_field(roi, animal_means, condition,
                  metric='manual_vessel_area_fraction'):
    """The animal closest to the condition median, then its most typical field.
    Same rule as fc.vascular_representative, so these exemplars correspond to
    the fields drawn in Figure 4F."""
    grp = animal_means[animal_means.condition == condition]
    an = grp.iloc[(grp[metric] - grp[metric].median()).abs().argmin()]['animal']
    sub = roi[roi.animal == an]
    r = sub.iloc[(sub[metric] - sub[metric].median()).abs().argmin()]
    return f'{an}_ROI{int(r.roi)}'

#endregion

#region load ###################################################################

roi = pd.read_csv(f'{OUT}/roi_metrics.csv')
animal = pd.read_csv(f'{OUT}/animal_means.csv')
primary = pd.read_csv(f'{OUT}/stats_primary.csv')
sens = pd.read_csv(f'{OUT}/stats_sensitivity.csv')
alt = pd.read_csv(f'{OUT}/robustness_alternatives.csv').set_index('variant')
loo = pd.read_csv(f'{OUT}/robustness_leave_one_out.csv')
conf = pd.read_csv(f'{OUT}/confounders.csv')

roi = roi.sort_values(['condition', 'animal', 'roi']).reset_index(drop=True)
VA = 'manual_vessel_area_fraction'
prim = primary.set_index('metric')

#endregion

#region A - segmentation validity ##############################################

yA = TOP
letter('A', yA)

# --- row 1 left: typical nulliparous, typical pregnant, worst agreement
w_row = roi.loc[roi.manual_auto_dice.idxmin()]
for i, (name, edge) in enumerate([
        (typical_field(roi, animal, COND[0]), CCOL[COND[0]]),
        (typical_field(roi, animal, COND[1]), CCOL[COND[1]]),
        (f'{w_row.animal}_ROI{int(w_row.roi)}', FAIL)]):
    ax = ax_in(LEFT + i * (A_BIG + 0.10), yA, A_BIG, A_BIG)
    show_overlay(ax, name, edge)

# --- row 1 right: manual vs automatic vessel area
ax = ax_in(A_RIGHT_X, yA, A_RIGHT_W, A_R1)
for c in COND:
    s = roi[roi.condition == c]
    ax.scatter(s[VA], s.auto_vessel_area_fraction, s=11, color=CCOL[c],
               edgecolors='black', linewidths=0.3, label=c, zorder=3)
lim = [3.5, 13]
ax.plot(lim, lim, ls='--', lw=0.7, color=GREY, zorder=1)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.xaxis.set_major_locator(FixedLocator([4, 6, 8, 10, 12]))
ax.yaxis.set_major_locator(FixedLocator([4, 6, 8, 10, 12]))
ax.set_xlabel('Manual mask, vessel area (%)', fontsize=FS_LAB, labelpad=1.2)
ax.set_ylabel('Automatic (Li)\nvessel area (%)', fontsize=FS_LAB, labelpad=1.2)
ax.legend(fontsize=FS - 1.0, frameon=False, loc='upper left',
          handletextpad=0.15, borderpad=0.05, labelspacing=0.12)
despine(ax)

# --- row 2 left: every analysed field
ySheet = yA + A_R1 + A_ROWGAP
for i, (_, r) in enumerate(roi.iterrows()):
    rr, cc = divmod(i, A_COLS)
    ax = ax_in(LEFT + cc * (A_TILE + A_TGAP), ySheet + rr * (A_TILE + A_TGAP),
               A_TILE, A_TILE)
    fail = r.manual_auto_dice < AUTO_DICE_MIN
    show_overlay(ax, f'{r.animal}_ROI{int(r.roi)}',
                 FAIL if fail else CCOL[r.condition],
                 lw=1.0 if fail else 0.55, px=150)

# --- row 2 right: per-field agreement, bottom edge flush with the sheet
ax = ax_in(A_RIGHT_X, ySheet, A_RIGHT_W, A_R2 - 0.36)
for k, c in enumerate(COND):
    v = roi.loc[roi.condition == c, 'manual_auto_dice'].to_numpy()
    ax.scatter(v, np.full(len(v), k) + rng.normal(0, 0.09, len(v)), s=8,
               color=CCOL[c], edgecolors='black', linewidths=0.22, zorder=3)
ax.axvline(roi.manual_auto_dice.mean(), color='black', lw=0.9, zorder=2)
ax.axvline(AUTO_DICE_MIN, color=FAIL, ls='--', lw=0.7, zorder=2)
ax.set_ylim(-0.8, 1.8)
ax.set_yticks([])
ax.set_xlim(0.55, 1.01)
ax.xaxis.set_major_locator(FixedLocator([0.6, 0.7, 0.8, 0.9, 1.0]))
ax.set_xlabel('Manual vs automatic Dice', fontsize=FS_LAB, labelpad=1.2)
despine(ax, keep=('bottom',))

#endregion

#region B - the battery, with bounds and detectable effect #####################

yBody = TOP + A_H + GAP_AB
letter('B', yBody)

ax = ax_in(L_PLOT_X, yBody, L_PLOT_W, B_H)
rows, ticks, heads = [], [], []
y = 0
for fam in FAMILIES:
    sub = primary[primary.family == fam]
    if not len(sub):
        continue
    heads.append((y, fam))
    y += 1
    for _, r in sub.iterrows():
        rows.append((y, r))
        ticks.append((y, SHORT.get(r.label, r.label)))
        y += 1

for yy, r in rows:
    lo, hi = r.diff_lo / r.mean_null * 100, r.diff_hi / r.mean_null * 100
    ax.plot([lo, hi], [yy, yy], color=GREY, lw=0.9, zorder=2,
            solid_capstyle='butt')
    for x in (-r.mde_pct, r.mde_pct):        # detectable at 80% power
        ax.plot([x, x], [yy - 0.34, yy + 0.34], color='#c6c6c6', lw=1.3,
                zorder=1)
    ax.scatter([r['pct_change']], [yy], s=13, zorder=4,
               color='white' if r.quantized else 'black',
               edgecolors='black', linewidths=0.6)

ax.axvline(0, color='black', lw=0.7, zorder=1)
ax.set_ylim(y - 0.5, -0.5)
ax.set_xlim(-68, 62)
ax.xaxis.set_major_locator(FixedLocator([-50, -25, 0, 25, 50]))
row_labels(ax, ticks, FS)
ax.set_xlabel('Change in pregnancy\n(% of nulliparous mean), 95% CI',
              fontsize=FS_LAB, labelpad=1.2)
despine(ax, keep=('bottom',))
group_labels(ax, heads, FS)

#endregion

#region C - fields within an animal are technical replicates ###################

yC = yBody + B_H + GAP_L
letter('C', yC)

ax = ax_in(L_PLOT_X, yC, L_PLOT_W, C_H)
order = animal.sort_values(['condition', 'animal']).animal.tolist()
for i, an in enumerate(order):
    s = roi[roi.animal == an]
    ax.scatter(np.full(len(s), i) + rng.normal(0, 0.07, len(s)), s[VA], s=9,
               color=CCOL[s.condition.iloc[0]], edgecolors='black',
               linewidths=0.22, zorder=3)
    ax.plot([i - 0.3, i + 0.3], [s[VA].mean()] * 2, color='black', lw=1.0,
            zorder=4)
ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=90, fontsize=FS - 1.2)
ax.set_ylabel('Vessel area (%)', fontsize=FS_LAB, labelpad=1.2)
despine(ax)

#endregion

#region D - the null under every analytic choice ###############################

letter('D', yBody, x=R_LETTER_X)


def sens_g(variant, metric):
    s = sens[(sens.variant == variant) & (sens.metric == metric)]
    return float(s.hedges_g.iloc[0]) if len(s) else np.nan


g_primary = float(prim.loc['vessel_area_fraction', 'hedges_g'])
BLOCKS = [
    ('Denominator', [
        ('Whole imaged field', float(alt.loc['vaf_field', 'hedges_g']), 0),
        ('Tissue mask', float(alt.loc['vaf_tissue', 'hedges_g']), 0),
        ('Analysed domain', float(alt.loc['vaf_domain', 'hedges_g']), 0),
        ('Guarded core', float(alt.loc['vaf_core', 'hedges_g']), 1)]),
    ('Despeckling', [
        ('None', float(alt.loc['vaf_speck0', 'hedges_g']), 0),
        ('< 5 µm²', float(alt.loc['vaf_speck5', 'hedges_g']), 0),
        ('< 20 µm²', float(alt.loc['vaf_speck20', 'hedges_g']), 1),
        ('< 50 µm²', float(alt.loc['vaf_speck50', 'hedges_g']), 0),
        ('< 100 µm²', float(alt.loc['vaf_speck100', 'hedges_g']), 0)]),
    ('Segmentation', [
        ('Manual mask', sens_g('manual', 'vessel_area_fraction'), 1),
        ('Automatic (Li)', sens_g('auto', 'vessel_area_fraction'), 0),
        ('Automatic, QC-pass',
         sens_g('auto_qc_pass', 'vessel_area_fraction'), 0)]),
    ('Domain radius', [
        ('100 µm', g_primary, 1),
        ('150 µm', sens_g('envelope_r150', 'r150_vessel_area_fraction'), 0),
        ('200 µm', sens_g('envelope_r200', 'r200_vessel_area_fraction'), 0)]),
]

ax = ax_in(R_PLOT_X, yBody, R_PLOT_W, D_H)
ax.axvspan(-0.2, 0.2, color='#eeeeee', zorder=0)
ax.axvline(0, color='black', lw=0.7, zorder=1)
ax.axvline(g_primary, color=GREY, ls='--', lw=0.7, zorder=1)

y, ticks, heads = 0, [], []
for name, items in BLOCKS:
    heads.append((y - 0.55, name))
    for lab, g, used in items:
        ax.scatter([g], [y], s=15, color='black' if used else 'white',
                   edgecolors='black', linewidths=0.6, zorder=3)
        ticks.append((y, lab))
        y += 1
    y += 0.6
heads.append((y - 0.55, 'Leave one animal out'))
loo_va = loo[loo.metric == 'vessel_area_fraction']
ax.scatter(loo_va.g_loo, np.full(len(loo_va), y) + rng.normal(
    0, 0.09, len(loo_va)), s=9, color=GREY, edgecolors='black',
    linewidths=0.22, zorder=3)
ticks.append((y, f'each of {len(loo_va)} animals dropped'))

ax.set_ylim(y + 0.6, -0.9)
ax.set_xlim(-0.92, 0.28)
ax.xaxis.set_major_locator(FixedLocator([-0.8, -0.4, 0.0]))
row_labels(ax, ticks, FS - 0.4)
ax.set_xlabel("Hedges' g, vessel area\n(pregnant − nulliparous)",
              fontsize=FS_LAB, labelpad=1.2)
despine(ax, keep=('bottom',))
group_labels(ax, heads, FS - 0.4)

#endregion

#region E - covariate balance ##################################################

yE = yBody + D_H + GAP_R
letter('E', yE, x=R_LETTER_X)

CBAL = [('domain_area_mm2', 'Analysed domain (mm²)'),
        ('field_area_mm2', 'Imaged field (mm²)'),
        ('tissue_fraction_of_field', 'Tissue fraction of field'),
        ('thr_li', 'Automatic threshold'),
        ('session_ord', 'Imaging session')]
cb = conf[conf.level == 'animal'].set_index('covariate')

ax = ax_in(R_PLOT_X, yE, R_PLOT_W, E_H)
for i, (k, lab) in enumerate(CBAL):
    p = float(cb.loc[k, 'p'])
    ax.plot([0, p], [i, i], color=GREY, lw=0.8, zorder=2)
    ax.scatter([p], [i], s=13, color='black', zorder=3)
ax.axvline(0.05, color=FAIL, ls='--', lw=0.7, zorder=1)
ax.set_ylim(len(CBAL) - 0.4, -0.6)
ax.set_xlim(0, 1)
ax.xaxis.set_major_locator(FixedLocator([0, 0.5, 1.0]))
ax.set_xticklabels(['0', '0.5', '1'], fontsize=FS)
row_labels(ax, list(enumerate([lab for _, lab in CBAL])), FS - 0.4)
ax.set_xlabel('p (Welch t-test,\nbetween groups)', fontsize=FS_LAB,
              labelpad=1.2)
despine(ax, keep=('bottom',))

#endregion

#region save ###################################################################

fc.save(fig, f'{FIG}/figure_supp_5')
print(f'wrote {FIG}/figure_supp_5.png/.svg ({FIG_W:.1f}x{FIG_H:.1f} in)')

#endregion
