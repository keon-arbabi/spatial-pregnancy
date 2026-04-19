#region imports and setup ######################################################

import os
import subprocess
import warnings
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

working_dir = 'spatial-pregnancy'
cell_type_col = 'subclass'
FDR_THRESHOLD = 0.10

datasets = {
    'slidetags': {
        'path': f'{working_dir}/output/slidetags'
                '/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium'
                '/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
    },
}

gsmap_input = f'{working_dir}/input/gsmap'
gwas_dir = f'{gsmap_input}/GWAS'
gwas_formatted_dir = f'{gsmap_input}/GWAS_formatted'
resource_dir = f'{gsmap_input}/gsMap_resource'
homolog_file = f'{resource_dir}/homologs/mouse_human_homologs.txt'
gwas_config_path = f'{gwas_formatted_dir}/gwas_config.yaml'

def load_cauchy(output_dir, conditions, cell_types=None):
    files = []
    for cond in conditions:
        d = Path(output_dir) / cond / 'cauchy_combination'
        if not d.is_dir():
            continue
        for f in d.glob('*.Cauchy.csv.gz'):
            files.append((f, cond))
    if not files:
        return None
    df = pl.concat([
        pl.scan_csv(str(f)) \
            .with_columns(
                condition=pl.lit(cond),
                trait=pl.lit(f.name
                    .replace(f'{cond}_', '')
                    .replace('.Cauchy.csv.gz', '')))
        for f, cond in files
    ])
    if cell_types is not None:
        df = df.filter(pl.col('annotation').is_in(list(cell_types)))
    return df

def cauchy_table(output_dir, conditions, cell_types=None, traits=None):
    df = load_cauchy(output_dir, conditions, cell_types)
    if df is None:
        return pd.DataFrame()
    agg = df \
        .with_columns(
            p_log=(-pl.col('p_cauchy').log10()).fill_null(0.0)) \
        .group_by(['condition', 'trait', 'annotation']) \
        .agg(pl.col('p_log').median()) \
        .collect() \
        .to_pandas()
    avg = agg \
        .groupby(['trait', 'annotation'])['p_log'] \
        .mean().reset_index() \
        .rename(columns={'p_log': 'avg_score'})
    wide = agg.pivot(
        index=['trait', 'annotation'],
        columns='condition', values='p_log'
    ).reset_index()
    result = pd.merge(wide, avg, on=['trait', 'annotation'])
    if traits:
        result = result[result['trait'].isin(traits)]
    return result.sort_values('avg_score', ascending=False)

def load_spatial_ldsc(output_dir, conditions, adata, traits):
    frames = []
    obs = adata.obs
    for cond in conditions:
        ldsc_dir = Path(output_dir) / cond / 'spatial_ldsc'
        if not ldsc_dir.is_dir():
            continue
        for trait in traits:
            f = ldsc_dir / f'{cond}_{trait}.csv.gz'
            if not f.exists():
                continue
            df = pl.scan_csv(str(f)) \
                .with_columns(
                    condition=pl.lit(cond),
                    trait=pl.lit(trait),
                    gsmap_score=-pl.col('p').log10())
            frames.append(df)
    if not frames:
        return None
    cell_ann = pl.DataFrame({
        'spot': obs.index.tolist(),
        'annotation': obs[cell_type_col].astype(str).tolist(),
    })
    return pl.concat(frames) \
        .join(cell_ann.lazy(), on='spot', how='inner') \
        .select(['spot', 'gsmap_score', 'annotation',
                 'trait', 'condition'])

def trait_associations(output_dir, conditions, contrasts,
                       adata, traits):
    df = load_spatial_ldsc(output_dir, conditions, adata, traits)
    if df is None:
        return pd.DataFrame()
    df = df.collect().to_pandas()
    results = []
    for (trait, annotation), group in df.groupby(['trait', 'annotation']):
        for treat, ctrl in contrasts:
            t_vals = group.loc[
                group['condition'] == treat, 'gsmap_score'].values
            c_vals = group.loc[
                group['condition'] == ctrl, 'gsmap_score'].values
            if len(t_vals) < 3 or len(c_vals) < 3:
                continue
            _, p = mannwhitneyu(
                t_vals, c_vals, alternative='two-sided')
            results.append({
                'trait': trait,
                'cell_type': annotation,
                'comparison': f'{treat}_vs_{ctrl}',
                'median_diff': np.median(t_vals) - np.median(c_vals),
                'p_value': p,
                'n_treat': len(t_vals),
                'n_ctrl': len(c_vals),
            })
    result = pd.DataFrame(results)
    if not result.empty:
        valid = ~result['p_value'].isna()
        result['p_adj'] = np.nan
        if valid.sum() > 0:
            _, p_adj, _, _ = multipletests(
                result.loc[valid, 'p_value'], method='fdr_bh')
            result.loc[valid, 'p_adj'] = p_adj
    return result

#endregion

#region prep data ##############################################################

adatas = {}
for name, cfg in datasets.items():
    adata = sc.read_h5ad(cfg['path'])
    drop = cfg.get('drop_samples', [])
    if drop:
        adata = adata[~adata.obs['sample'].isin(drop)].copy()
        print(f'[{name}] dropped samples: {drop}')
    adatas[name] = adata
    print(f'[{name}] {adata.shape[0]:,} cells, '
          f'{adata.obs[cell_type_col].nunique()} subclasses, '
          f'{adata.obs["sample"].nunique()} samples')

for name in datasets:
    adata = adatas[name]
    st_dir = f'{gsmap_input}/{name}/ST'
    os.makedirs(st_dir, exist_ok=True)
    for cond in sorted(adata.obs['condition'].unique()):
        target = f'{st_dir}/{cond}.h5ad'
        if os.path.exists(target):
            continue
        adata_cond = adata[adata.obs['condition'] == cond].copy()
        adata_cond.obsm['spatial'] = \
            adata_cond.obs[['x_affine', 'y_affine']].to_numpy()
        adata_cond.obs_names_make_unique()
        adata_cond.write_h5ad(target)
        print(f'[{name}] {cond}: '
              f'{adata_cond.shape[0]:,} cells -> {target}')

if not os.path.exists(resource_dir):
    os.makedirs(gsmap_input, exist_ok=True)
    url = 'https://yanglab.westlake.edu.cn/data/gsMap' \
          '/gsMap_resource.tar.gz'
    subprocess.run(
        f'wget -q {url} -P {gsmap_input}',
        shell=True, check=True)
    subprocess.run(
        f'tar -xzf {gsmap_input}/gsMap_resource.tar.gz'
        f' -C {gsmap_input}',
        shell=True, check=True)
    os.remove(f'{gsmap_input}/gsMap_resource.tar.gz')
    print('downloaded gsMap resources')

os.makedirs(gwas_formatted_dir, exist_ok=True)
for f in sorted(os.listdir(gwas_dir)):
    if not f.endswith('.sumstats.gz'):
        continue
    basename = f.replace('.sumstats.gz', '')
    if os.path.exists(
            f'{gwas_formatted_dir}/{basename}.sumstats.gz'):
        continue
    subprocess.run(
        f"python -m gsMap format_sumstats "
        f"--sumstats '{gwas_dir}/{f}' "
        f"--out '{gwas_formatted_dir}/{basename}'",
        shell=True, check=True)
    print(f'formatted {basename}')

with open(gwas_config_path, 'w') as fh:
    for gwas_file in sorted(os.listdir(gwas_formatted_dir)):
        if gwas_file.endswith('.sumstats.gz'):
            trait = gwas_file.replace('.sumstats.gz', '')
            path = os.path.abspath(
                f'{gwas_formatted_dir}/{gwas_file}')
            fh.write(f'{trait}: {path}\n')

with open(gwas_config_path) as fh:
    config = yaml.safe_load(fh)
all_traits = list(config.keys()) if config else []
print(f'{len(all_traits)} GWAS traits')

#endregion

#region run gsmap ##############################################################

def submit_slurm(cmd, *, job_name, log_file, depends=None, hours=24):
    from tempfile import NamedTemporaryFile
    cluster = os.environ.get('CLUSTER', '')
    sbatch = '.sbatch' if cluster.startswith('trillium') else 'sbatch'
    partition = 'compute' if cluster.startswith('trillium') else None
    lines = ['#!/bin/bash']
    if partition is not None:
        lines.append(f'#SBATCH -p {partition}')
    lines.append('#SBATCH --account=rrg-shreejoy')
    if not cluster.startswith('trillium'):
        lines.append('#SBATCH -c 1')
    lines += [
        '#SBATCH -N 1',
        '#SBATCH -n 1',
        f'#SBATCH -t {hours}:00:00',
        f'#SBATCH -J {job_name}',
        f'#SBATCH -o {log_file}',
    ]
    dep_ids = [d for d in ([depends] if isinstance(depends, str)
                           else (depends or [])) if d]
    if dep_ids:
        lines.append(f'#SBATCH --dependency=afterok:{":".join(dep_ids)}')
    lines += [
        'export OMP_PLACES=cores',
        'export OMP_PROC_BIND=spread',
        f'set -euo pipefail; {cmd}',
    ]
    scratch = os.environ.get('SCRATCH', '.')
    with NamedTemporaryFile('w', dir=scratch, suffix='.sh',
                            delete=False) as fh:
        fh.write('\n'.join(lines) + '\n')
        script_path = fh.name
    try:
        out = subprocess.check_output(
            f'{sbatch} --parsable {script_path}',
            shell=True, text=True).strip()
    finally:
        os.unlink(script_path)
    return out.split(';')[0]

log_dir = f'{working_dir}/output/gsmap_logs'
os.makedirs(log_dir, exist_ok=True)
ds_abbr = {'slidetags': 'sl', 'xenium': 'xe'}
cond_abbr = {'CTRL': 'C', 'PREG': 'P', 'POSTPART': 'PP'}

for name in datasets:
    conditions = sorted(adatas[name].obs['condition'].unique())
    output = f'{working_dir}/output/{name}/gsmap'
    os.makedirs(output, exist_ok=True)

    for cond in conditions:
        hdf5 = f'{gsmap_input}/{name}/ST/{cond}.h5ad'
        ldsc_done = (
            Path(output) / cond / 'generate_ldscore' /
            f'{cond}_generate_ldscore.done').exists()
        pending = [
            t for t in all_traits
            if not os.path.exists(
                f'{output}/{cond}/cauchy_combination/'
                f'{cond}_{t}.Cauchy.csv.gz')]
        if not pending:
            print(f'[{name}] {cond}: all {len(all_traits)} traits done')
            continue

        def submit(trait, depends=None):
            cmd = (
                f"python -m gsMap quick_mode "
                f"--workdir '{output}' "
                f"--homolog_file '{homolog_file}' "
                f"--sample_name '{cond}' "
                f"--gsMap_resource_dir '{resource_dir}' "
                f"--hdf5_path '{hdf5}' "
                f"--annotation '{cell_type_col}' "
                f"--data_layer 'X' "
                f"--trait_name '{trait}' "
                f"--sumstats_file '{gwas_formatted_dir}/{trait}.sumstats.gz' "
                f"--max_processes $(nproc)")
            tag = (f'{ds_abbr.get(name, name[:2])}_'
                   f'{cond_abbr.get(cond, cond)}_{trait}')
            return submit_slurm(
                cmd, job_name=tag,
                log_file=f'{log_dir}/{name}_{cond}_{trait}.log',
                depends=depends)

        anchor_jid = None
        if not ldsc_done:
            anchor_trait = pending.pop(0)
            anchor_jid = submit(anchor_trait)
            print(f'[{name}] {cond} {anchor_trait} (prep+ldsc) '
                  f'-> job {anchor_jid}')
        for trait in pending:
            jid = submit(trait, depends=anchor_jid)
            dep = f' (waits on {anchor_jid})' if anchor_jid else ''
            print(f'[{name}] {cond} {trait} -> job {jid}{dep}')

#endregion

# #region analysis ###############################################################

# for name, cfg in datasets.items():
#     conditions = sorted(adatas[name].obs['condition'].unique())
#     output = f'{working_dir}/output/{name}/gsmap'
#     out_dir = f'{working_dir}/output/{name}'
#     os.makedirs(out_dir, exist_ok=True)

#     cell_types = set(adatas[name].obs[cell_type_col].unique())

#     ct = cauchy_table(output, conditions, cell_types)
#     if not ct.empty:
#         ct.to_csv(
#             f'{out_dir}/gsmap_cauchy_table.csv', index=False)
#         print(f'[{name}] cauchy table: {ct.shape[0]} rows')

#     ta = trait_associations(
#         output, conditions, cfg['contrasts'],
#         adatas[name], all_traits)
#     if not ta.empty:
#         ta.to_csv(
#             f'{out_dir}/gsmap_associations.csv', index=False)
#         n_sig = int((ta['p_adj'] < FDR_THRESHOLD).sum())
#         print(f'[{name}] associations: {ta.shape[0]} tests, '
#               f'{n_sig} sig (FDR<{FDR_THRESHOLD})')
#         ta_sig = ta[ta['p_adj'] < FDR_THRESHOLD] \
#             .sort_values('p_adj')
#         if not ta_sig.empty:
#             ta_sig.to_csv(
#                 f'{out_dir}/gsmap_associations_sig.csv',
#                 index=False)

# #endregion

# #region plot — trait ranking ###################################################

# plt.rcParams['svg.fonttype'] = 'none'
# plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['figure.dpi'] = 300

# BONFERRONI_ALPHA = 0.05

# rankings = {}
# thresholds = {}
# for name in datasets:
#     conditions = sorted(adatas[name].obs['condition'].unique())
#     cell_types = set(adatas[name].obs[cell_type_col].unique())
#     output = f'{working_dir}/output/{name}/gsmap'
#     df = load_cauchy(output, conditions, cell_types)
#     if df is None:
#         continue
#     stats = df.select(['trait', 'annotation']).collect()
#     n_traits = stats['trait'].n_unique()
#     n_ann = stats['annotation'].n_unique()
#     thresholds[name] = \
#         -np.log10(BONFERRONI_ALPHA / (n_traits * n_ann))
#     rankings[name] = df \
#         .with_columns(p_log=(-pl.col('p_cauchy').log10())) \
#         .group_by('trait', 'annotation', 'condition') \
#         .agg(pl.median('p_log').alias('median_log_p')) \
#         .group_by('trait') \
#         .agg(pl.max('median_log_p').alias('peak_score')) \
#         .sort('peak_score', descending=True) \
#         .collect() \
#         .to_pandas()

# trait_scores = {}
# for ranking in rankings.values():
#     for _, row in ranking.iterrows():
#         trait_scores \
#             .setdefault(row['trait'], []) \
#             .append(row['peak_score'])
# trait_order = sorted(
#     trait_scores, key=lambda t: np.mean(trait_scores[t]))

# ds_list = [n for n in datasets if n in rankings]
# fig, axes = plt.subplots(
#     1, len(ds_list),
#     figsize=(2.5 * len(ds_list), 3.5),
#     sharey=True, squeeze=False)

# for i, name in enumerate(ds_list):
#     ax = axes[0, i]
#     ranking = rankings[name] \
#         .set_index('trait') \
#         .reindex(trait_order) \
#         .reset_index()
#     ranking['peak_score'] = ranking['peak_score'].fillna(0)
#     scores = ranking['peak_score'].values
#     norm = mcolors.Normalize(
#         vmin=scores.min(), vmax=scores.max())
#     cmap = plt.get_cmap('GnBu')
#     sns.barplot(
#         x='peak_score', y='trait', data=ranking,
#         hue='trait',
#         palette=[cmap(norm(s)) for s in scores],
#         ax=ax, orient='h', legend=False)
#     ax.axvline(
#         x=thresholds[name], color='black',
#         linestyle='--', linewidth=1)
#     ax.set_xlabel('Peak Score')
#     ax.set_ylabel('')
#     ax.set_title(name, fontsize=9)
#     sns.despine(ax=ax)
#     ax.tick_params(axis='y', length=0)

# fig.tight_layout()
# os.makedirs(f'{working_dir}/figures', exist_ok=True)
# fig.savefig(
#     f'{working_dir}/figures/gsmap_trait_ranking.png',
#     dpi=300, bbox_inches='tight')
# fig.savefig(
#     f'{working_dir}/figures/gsmap_trait_ranking.svg',
#     bbox_inches='tight')
# plt.close(fig)

# #endregion
