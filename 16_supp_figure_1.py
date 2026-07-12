#region imports and setup #####################################################

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

working_dir = '/home/karbabi/spatial-pregnancy'

#endregion
#region Supplementary Figure 1 ################################################

# QC metrics (saved by save_qc in 01_preprocessing.py) and CAST metrics (saved
# by save_cast in 05_resolvi.py) for all three platforms. Each panel is a
# per-sample boxplot with the filter threshold drawn as a dashed line. CAST
# distance cutoffs are per-sample upper percentiles, stored as {sample: value}
# dicts and drawn as per-sample segments; all other thresholds are global
# scalars drawn as full-width lines.
#
# Figure caption:
# Supplementary Fig. 1. Quality control and cell-type annotation performance
# for the Slide-tags, MERFISH and Xenium spatial datasets. Each box shows the
# median and interquartile range (IQR) of a per-cell metric for one biological
# replicate, with whiskers extending to 1.5x the IQR; outliers are omitted. The
# three left-hand columns report pre-filtering quality-control metrics for each
# platform: detected genes per cell, total UMI counts, per-cell doublet score
# (scDblFinder) and a cell-size metric - mitochondrial read percentage (MT %)
# for Slide-tags, cell volume (um^3) for MERFISH and cell area (um^2) for
# Xenium. Pink dashed lines mark the exclusion thresholds applied during
# preprocessing: Slide-tags, >500 genes, >750 UMIs, doublet score <0.2 and
# MT <10%; MERFISH, >10 genes, doublet score <0.2 and cell volume >100 um^3;
# Xenium, >100 genes, >100 UMIs, doublet score <0.2 and cell area >10 um^2. For
# MERFISH and Xenium, cells were additionally capped at 3x the per-sample median
# volume/area. The three right-hand columns report cell-type annotation quality
# after CAST integration with the Allen Mouse Brain Atlas: the cosine distance
# between query and reference expression profiles (Expression distance), the
# confidence of broad cell-class assignment (Class confidence), the physical
# distance to the mapped reference location (Spatial distance) and the
# confidence of fine-grained subclass assignment (Subclass confidence). Pink
# dashed lines indicate the criteria used to retain high-confidence annotations:
# a global subclass-confidence cutoff (>0.6; horizontal line) together with
# per-sample 95th-percentile limits on expression and spatial distance
# (per-sample segments), above which cells were discarded; an additional
# subclass-margin criterion (>0.2) was applied but is not shown. Class
# confidence is displayed for reference and was not used for filtering. Samples
# comprise Nulliparous, Pregnant and Postpartum replicates; the Xenium cohort
# includes Nulliparous and Pregnant replicates only.

null_labels = ['Nulliparous 1', 'Nulliparous 2', 'Nulliparous 3']
preg_labels = ['Pregnant 1', 'Pregnant 2', 'Pregnant 3']
post_labels = ['Postpartum 1', 'Postpartum 2', 'Postpartum 3']

qc_datasets = {
    'slidetags': {
        'data': pd.read_csv(
            f'{working_dir}/output/slidetags/qc_metrics_data.csv'),
        'thr': json.load(open(
            f'{working_dir}/output/slidetags/qc_filter_thresholds.json')),
        'metrics': ['n_genes_by_counts', 'total_counts',
                    'scDblFinder.score', 'pct_counts_mt'],
        'titles': ['Genes per cell', 'Total UMI counts',
                   'Doublet score', 'Mitochondrial %'],
        'y_labels': ['Number of genes', 'UMI counts',
                     'scDblFinder score', 'MT %'],
        'samples': ['CTRL_1', 'CTRL_2', 'CTRL_3', 'PREG_1', 'PREG_2',
                    'PREG_3', 'POSTPART_1', 'POSTPART_2'],
        'labels': null_labels + preg_labels + post_labels[:2],
        'configs': {
            'n_genes_by_counts': dict(
                log=True, ticks=[500, 1000, 2000, 5000],
                ylim=(330, 5500)),
            'total_counts': dict(
                log=True, ticks=[500, 1000, 2000, 5000, 10000],
                ylim=(400, 11500)),
            'scDblFinder.score': dict(
                log=False, ylim=(-0.03, 0.78)),
            'pct_counts_mt': dict(
                log=False, ylim=(-0.5, 11.5))
        },
        'thr_map': {'n_genes_by_counts': 'n_genes_by_counts',
                    'total_counts': 'total_counts',
                    'scDblFinder.score': 'scDblFinder_score',
                    'pct_counts_mt': 'pct_counts_mt'}
    },
    'merfish': {
        'data': pd.read_csv(
            f'{working_dir}/output/merfish/qc_metrics_data.csv'),
        'thr': json.load(open(
            f'{working_dir}/output/merfish/qc_filter_thresholds.json')),
        'metrics': ['n_genes_by_counts', 'total_counts',
                    'scDblFinder.score', 'volume'],
        'titles': ['Genes per cell', 'Total UMI counts',
                   'Doublet score', 'Cell volume'],
        'y_labels': ['Number of genes', 'UMI counts',
                     'scDblFinder score', 'Volume (μm³)'],
        'samples': ['CTRL_1', 'CTRL_2', 'CTRL_3', 'PREG_1', 'PREG_2', 'PREG_3',
                    'POSTPART_1', 'POSTPART_2', 'POSTPART_3'],
        'labels': null_labels + preg_labels + post_labels,
        'configs': {
            'n_genes_by_counts': dict(
                log=True, ticks=[1, 10, 100], ylim=(0.8, 200)),
            'total_counts': dict(
                log=True, ticks=[50, 100, 200], ylim=(42, 320)),
            'scDblFinder.score': dict(
                log=False, ylim=(-0.012, 0.26)),
            'volume': dict(
                log=True, ticks=[100, 300, 1000, 3000], ylim=(55, 3700))
        },
        'thr_map': {'n_genes_by_counts': 'n_genes_by_counts',
                    'scDblFinder.score': 'scDblFinder_score',
                    'volume': 'volume_min'}
    },
    'xenium': {
        'data': pd.read_csv(
            f'{working_dir}/output/xenium/qc_metrics_data.csv'),
        'thr': json.load(open(
            f'{working_dir}/output/xenium/qc_filter_thresholds.json')),
        'metrics': ['n_genes_by_counts', 'total_counts',
                    'scDblFinder.score', 'cell_area'],
        'titles': ['Genes per cell', 'Total UMI counts',
                   'Doublet score', 'Cell area'],
        'y_labels': ['Number of genes', 'UMI counts',
                     'scDblFinder score', 'Area (μm²)'],
        'samples': ['CTRL_1', 'CTRL_2', 'CTRL_3', 'PREG_2', 'PREG_3'],
        'labels': null_labels + ['Pregnant 2', 'Pregnant 3'],
        'configs': {
            'n_genes_by_counts': dict(
                log=True, ticks=[1, 10, 100, 1000], ylim=(0.8, 2400)),
            'total_counts': dict(
                log=True, ticks=[1, 10, 100, 1000],
                ylim=(0.8, 5500)),
            'scDblFinder.score': dict(
                log=False, ylim=(-0.04, 1.05)),
            'cell_area': dict(
                log=True, ticks=[3, 10, 30, 100, 300], ylim=(1.4, 320))
        },
        'thr_map': {'n_genes_by_counts': 'n_genes_by_counts',
                    'total_counts': 'total_counts',
                    'scDblFinder.score': 'scDblFinder_score',
                    'cell_area': 'cell_area_min'}
    }
}

# confidence panels share a scale; spatial distance shares a scale across
# platforms (comparable ranges); expression distance is scaled per platform
# because Xenium's cosine distances are ~10x smaller than the others. All
# ranges contain every whisker (1.5*IQR) and threshold line, with margin.
conf_cfg = dict(ylim=(0.38, 1.05))
pdist_cfg = dict(ylim=(0.21, 0.42))
cast_configs = {
    'slidetags': {
        'min_cos_dist': dict(ylim=(-0.02, 0.7)),
        'class_confidence': conf_cfg, 'avg_pdist': pdist_cfg,
        'subclass_confidence': conf_cfg},
    'merfish': {
        'min_cos_dist': dict(ylim=(-0.015, 0.52)),
        'class_confidence': conf_cfg, 'avg_pdist': pdist_cfg,
        'subclass_confidence': conf_cfg},
    'xenium': {
        'min_cos_dist': dict(ylim=(-0.008, 0.18)),
        'class_confidence': conf_cfg, 'avg_pdist': pdist_cfg,
        'subclass_confidence': conf_cfg},
}
cast_thr_map = {'min_cos_dist': 'min_cos_dist', 'avg_pdist': 'avg_pdist',
                'subclass_confidence': 'subclass_confidence'}

def make_cast(name):
    return {
        'data': pd.read_csv(
            f'{working_dir}/output/{name}/cast_metrics_data.csv'),
        'thr': json.load(open(
            f'{working_dir}/output/{name}/cast_filter_thresholds.json')),
        'metrics': ['min_cos_dist', 'class_confidence',
                    'avg_pdist', 'subclass_confidence'],
        'titles': ['Expression distance', 'Class confidence',
                   'Spatial distance', 'Subclass confidence'],
        'y_labels': ['Cosine distance', 'Confidence',
                     'Physical distance (AU)', 'Confidence'],
        'samples': qc_datasets[name]['samples'],
        'labels': qc_datasets[name]['labels'],
        'configs': cast_configs[name],
        'thr_map': cast_thr_map,
    }

cast_datasets = {name: make_cast(name)
                 for name in ['slidetags', 'merfish', 'xenium']}

display_names = {'slidetags': 'Slide-tags', 'merfish': 'MERFISH',
                 'xenium': 'Xenium'}

n_ds = len(qc_datasets)
fig, axes = plt.subplots(4, 2 * n_ds, figsize=(4 * 2 * n_ds, 12))
pink = sns.color_palette('PiYG')[0]

for col_offset, datasets in enumerate([qc_datasets, cast_datasets]):
    for col, (name, d) in enumerate(datasets.items()):
        actual_col = col_offset * n_ds + col
        for row, (m, title, ylabel) in enumerate(zip(d['metrics'],
                                                       d['titles'],
                                                       d['y_labels'])):
            ax = axes[row, actual_col]
            cfg = d['configs'][m]

            plot_data = d['data']
            if m in ('avg_pdist', 'min_cos_dist'):
                plot_data = plot_data[~np.isinf(plot_data[m])]

            sns.boxplot(data=plot_data, x='sample', y=m, ax=ax,
                       color=pink, linewidth=1, width=0.4,
                       showfliers=False, order=d['samples'])

            if cfg.get('log'):
                ax.set_yscale('log')
                ax.set_yticks(cfg['ticks'])
                ax.set_yticklabels([str(int(x)) for x in cfg['ticks']])
                ax.minorticks_off()

            thr_val = d['thr'].get(d['thr_map'].get(m))
            if isinstance(thr_val, dict):
                for i, s in enumerate(d['samples']):
                    v = thr_val.get(s)
                    if v is not None:
                        ax.plot([i - 0.3, i + 0.3], [v, v], ls='--',
                                color=pink, alpha=0.7, lw=1)
            elif thr_val is not None:
                ax.axhline(y=thr_val, ls='--', color=pink, alpha=0.5)
                ax.text(1.02, thr_val, f'{thr_val:.1f}', va='center',
                       transform=ax.get_yaxis_transform())

            if row < 3:
                ax.set_xticklabels([])
                ax.set_xlabel('')
                ax.set_xticks([])
            else:
                ax.set_xticks(range(len(d['labels'])))
                ax.set_xticklabels(d['labels'], rotation=45, ha='right',
                                  va='top')
                ax.set_xlabel('Sample', fontsize=11)

            if cfg.get('ylim'):
                ax.set_ylim(*cfg['ylim'])

            ax.set_title(f"{display_names[name]}: {title}", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=11)

plt.tight_layout()
plt.savefig(f'{working_dir}/figures/figure_supp_1.svg', bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/figure_supp_1.png', bbox_inches='tight')
plt.close()

#endregion
