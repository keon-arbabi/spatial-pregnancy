#region imports and setup ######################################################

import os

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

working_dir = '/home/karbabi/spatial-pregnancy'
de_suffix = ''            # '' for subclass; '_class' for class-level
cell_type_col = 'subclass'

# DEG-count thresholds: a gene is a meta-DEG for a cell type/contrast if its
# empirical p < EMP_P_THRESH; cell types are shown only if they reach
# >= MIN_DEGS_TO_SHOW total meta-DEGs across all contrasts.
EMP_P_THRESH = 0.05
MIN_DEGS_TO_SHOW = 5

seismic_cmap = plt.get_cmap('seismic')
UP_COLOR = seismic_cmap(0.9)
DN_COLOR = seismic_cmap(0.1)

contrasts = ['PREG_vs_CTRL', 'POSTPART_vs_PREG', 'POSTPART_vs_CTRL']
contrast_titles = {
    'PREG_vs_CTRL': 'Pregnant vs\nNulliparous',
    'POSTPART_vs_PREG': 'Postpartum vs\nPregnant',
    'POSTPART_vs_CTRL': 'Postpartum vs\nNulliparous',
}

BAR_H = 0.55

#endregion

#region meta de barplot ########################################################

# major class label for cell-type ordering (Glut -> Gaba -> NN)
def get_type(ct):
    if 'Glut' in ct:
        return 'Glut'
    elif any(x in ct for x in ['Gaba', 'IMN', 'Chol']):
        return 'Gaba'
    return 'NN'

def build_barplot():
    sr_meta = pl.read_csv(
        f'{working_dir}/output/de/sumrank_results{de_suffix}.csv')

    # per cell-type/contrast: count up/down meta-DEGs, split by D (number of
    # platforms agreeing: D=3 full alpha, D=2 lighter)
    deg_counts = sr_meta\
        .group_by(['cell_type', 'contrast'])\
        .agg(
            ((pl.col('emp_p_up') < EMP_P_THRESH) &
             (pl.col('D') == 3)).sum().alias('up_d3'),
            ((pl.col('emp_p_up') < EMP_P_THRESH) &
             (pl.col('D') == 2)).sum().alias('up_d2'),
            ((pl.col('emp_p_down') < EMP_P_THRESH) &
             (pl.col('D') == 3)).sum().alias('dn_d3'),
            ((pl.col('emp_p_down') < EMP_P_THRESH) &
             (pl.col('D') == 2)).sum().alias('dn_d2'))

    # total DEGs per cell type -> inclusion filter + grouped/sorted ordering
    ct_totals = deg_counts\
        .group_by('cell_type')\
        .agg((pl.sum('up_d3') + pl.sum('up_d2') +
              pl.sum('dn_d3') + pl.sum('dn_d2')).alias('total'))\
        .filter(pl.col('total') >= MIN_DEGS_TO_SHOW)\
        .with_columns(
            pl.col('cell_type').map_elements(
                get_type, return_dtype=pl.Utf8).alias('type'))\
        .with_columns(
            pl.col('type').replace_strict(
                {'Glut': 0, 'Gaba': 1, 'NN': 2}).alias('type_order'))\
        .sort(['type_order', 'total'], descending=[False, True])

    groups = ct_totals.group_by('type', maintain_order=True).all()
    major_types = groups['type'].to_list()
    height_ratios = groups['cell_type'].list.len().to_list()

    fig = plt.figure(figsize=(7, 12))
    outer_gs = gridspec.GridSpec(
        len(major_types), len(contrasts), figure=fig,
        height_ratios=height_ratios, hspace=0.06, wspace=0.04)

    for i, group_type in enumerate(major_types):
        group_cts = groups.filter(
            pl.col('type') == group_type)['cell_type'].explode().to_list()

        for j, contrast in enumerate(contrasts):
            ax = fig.add_subplot(outer_gs[i, j])

            sub = deg_counts.filter(
                (pl.col('contrast') == contrast) &
                (pl.col('cell_type').is_in(group_cts)))
            ct_d = {r['cell_type']: r for r in sub.to_dicts()}

            all_up, all_dn = [], []
            lw = 0.4
            for idx, ct in enumerate(group_cts):
                r = ct_d.get(ct, {})
                u3 = r.get('up_d3', 0)
                u2 = r.get('up_d2', 0)
                n3 = r.get('dn_d3', 0)
                n2 = r.get('dn_d2', 0)

                # UP: D=3 (full alpha) then D=2 (lower alpha) rightward
                ax.barh(idx, u3, height=BAR_H, align='center',
                        facecolor=UP_COLOR, edgecolor=UP_COLOR,
                        alpha=1.0, linewidth=lw, zorder=5)
                ax.barh(idx, u2, left=u3, height=BAR_H, align='center',
                        facecolor=UP_COLOR, edgecolor=UP_COLOR,
                        alpha=0.45, linewidth=lw, zorder=5)

                # DOWN: D=3 then D=2 leftward
                ax.barh(idx, -n3, height=BAR_H, align='center',
                        facecolor=DN_COLOR, edgecolor=DN_COLOR,
                        alpha=1.0, linewidth=lw, zorder=5)
                ax.barh(idx, -n2, left=-n3, height=BAR_H, align='center',
                        facecolor=DN_COLOR, edgecolor=DN_COLOR,
                        alpha=0.45, linewidth=lw, zorder=5)

                all_up.append(u3 + u2)
                all_dn.append(n3 + n2)

            xlim = max(max(all_up + [1]), max(all_dn + [1])) * 1.25

            ax.axvline(0, color='grey', linewidth=0.5, zorder=0)
            ax.grid(True, 'major', 'y', ls='-', lw=0.3,
                    c='lightgray', zorder=0)
            ax.set_xlim(-xlim, xlim)
            ax.set_yticks(range(len(group_cts)))
            ax.set_ylim(len(group_cts) - 0.5, -0.5)
            ax.tick_params(length=0, labelsize=7)

            if j == 0:
                ax.set_yticklabels(group_cts, fontsize=7.5)
                ax.tick_params(axis='y', pad=8)
            else:
                ax.set_yticklabels([])

            if i == 0:
                ax.set_title(contrast_titles[contrast], fontsize=9, pad=6)

            if i == len(major_types) - 1:
                ax.set_xlabel(f'meta DEGs (emp_p<{EMP_P_THRESH})',
                              fontsize=8.5)
            else:
                ax.set_xticklabels([])

    legend_elements = [
        Patch(facecolor=UP_COLOR, edgecolor=UP_COLOR,
              alpha=1.0, linewidth=0.4, label='Upregulated'),
        Patch(facecolor=DN_COLOR, edgecolor=DN_COLOR,
              alpha=1.0, linewidth=0.4, label='Downregulated'),
        Patch(facecolor='lightgray', edgecolor='black',
              alpha=1.0, linewidth=0.4, label='D=3 (3 platforms)'),
        Patch(facecolor='lightgray', edgecolor='black',
              alpha=0.45, linewidth=0.4, label='D=2 (2 platforms)'),
    ]
    fig.legend(handles=legend_elements, loc='lower right',
               bbox_to_anchor=(0.98, 0.02), fontsize=7,
               frameon=False, ncol=2)

    os.makedirs(f'{working_dir}/figures', exist_ok=True)
    for ext in ['png', 'svg']:
        fig.savefig(f'{working_dir}/figures/figure_2.{ext}',
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('figure saved: figures/figure_2.{png,svg}')

#endregion

#region run ####################################################################

if __name__ == '__main__':
    build_barplot()

#endregion
