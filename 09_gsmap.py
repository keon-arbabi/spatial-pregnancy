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
        'path': f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
    },
}

traits_selected = ['MDD', 'Neuroticism', 'ADHD', 'Autism', 'PTSD']

gsmap_input = f'{working_dir}/input/gsmap'
gwas_dir = f'{gsmap_input}/GWAS'
gwas_formatted_dir = f'{gsmap_input}/GWAS_formatted'
resource_dir = f'{gsmap_input}/gsMap_resource'
homolog_file = f'{resource_dir}/homologs/mouse_human_homologs.txt'
gwas_config_path = f'{gwas_formatted_dir}/gwas_config.yaml'

def load_cauchy(output_dir, conditions, cell_types=None,
                subdir='cauchy_combination'):
    files = []
    for cond in conditions:
        d = Path(output_dir) / cond / subdir
        if not d.is_dir():
            continue
        for f in d.glob('*.Cauchy.csv.gz'):
            files.append((f, cond))
    if not files:
        return None
    df = pl.concat([
        pl.scan_csv(str(f)).with_columns(
            condition=pl.lit(cond),
            trait=pl.lit(
                f.name.replace(f'{cond}_', '').replace('.Cauchy.csv.gz', ''))
        ) for f, cond in files
    ])
    if cell_types is not None:
        df = df.filter(pl.col('annotation').is_in(list(cell_types)))
    return df

def cauchy_table(output_dir, conditions, cell_types=None, traits=None,
                 subdir='cauchy_combination'):
    df = load_cauchy(output_dir, conditions, cell_types, subdir)
    if df is None:
        return pd.DataFrame()
    agg = (df
        .with_columns(p_log=(-pl.col('p_cauchy').log10()).fill_null(0.0))
        .group_by(['condition', 'trait', 'annotation'])
        .agg(pl.col('p_log').median())
        .collect()
        .to_pandas())
    avg = (agg.groupby(['trait', 'annotation'])['p_log']
           .mean().reset_index()
           .rename(columns={'p_log': 'avg_score'}))
    wide = agg.pivot(
        index=['trait', 'annotation'], columns='condition', values='p_log'
    ).reset_index()
    result = pd.merge(wide, avg, on=['trait', 'annotation'])
    if traits:
        result = result[result['trait'].isin(traits)]
    return result.sort_values('avg_score', ascending=False)

def trait_associations(output_dir, conditions, contrasts, cell_types, traits):
    files = [
        p for cond in conditions for trait in traits
        if (p := Path(output_dir) / cond / 'report' / trait /
                'gsMap_plot' / f'{cond}_{trait}_gsMap_plot.csv').exists()
    ]
    if not files:
        return pd.DataFrame()
    df = (pl.concat([
            pl.scan_csv(str(f)).with_columns(
                condition=pl.lit(f.parts[-5]),
                trait=pl.lit(f.parts[-3])
            ) for f in files
        ])
        .with_columns(gsmap_score=-pl.col('p').log10())
        .select(['gsmap_score', 'annotation', 'trait', 'condition'])
        .filter(pl.col('annotation').is_in(list(cell_types)))
        .collect()
        .to_pandas())
    results = []
    for (trait, annotation), group in df.groupby(['trait', 'annotation']):
        for treat, ctrl in contrasts:
            t_vals = group.loc[
                group['condition'] == treat, 'gsmap_score'].values
            c_vals = group.loc[
                group['condition'] == ctrl, 'gsmap_score'].values
            if len(t_vals) < 3 or len(c_vals) < 3:
                continue
            _, p = mannwhitneyu(t_vals, c_vals, alternative='two-sided')
            results.append({
                'trait': trait, 'cell_type': annotation,
                'comparison': f'{treat}_vs_{ctrl}',
                'median_diff': np.median(t_vals) - np.median(c_vals),
                'p_value': p,
                'n_treat': len(t_vals), 'n_ctrl': len(c_vals),
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

def gwas_de_overlap(de_path, gwas_hits_path, logfc_threshold=0.5):
    de = pl.read_csv(de_path)
    hits = pl.read_csv(gwas_hits_path)
    homolog = pl.read_csv(homolog_file, separator='\t', has_header=True)
    hits_mouse = hits.join(
        homolog.select(['HUMAN_GENE_SYM', 'MOUSE_GENE_SYM']),
        left_on='Gene Name', right_on='HUMAN_GENE_SYM', how='left')
    gwas_de = (de
        .filter(pl.col('logFC').abs().gt(logfc_threshold))
        .join(
            hits_mouse.select([
                'Gene Name', 'MOUSE_GENE_SYM', 'Z Statistic', 'P-value']),
            left_on='gene', right_on='MOUSE_GENE_SYM', how='inner')
        .sort('FDR'))
    gwas_genes = (hits_mouse
        .filter(pl.col('MOUSE_GENE_SYM').is_not_null())
        ['MOUSE_GENE_SYM'].unique().to_list())
    return gwas_de, hits_mouse, gwas_genes

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
        print(f'[{name}] {cond}: {adata_cond.shape[0]:,} cells → {target}')

if not os.path.exists(resource_dir):
    os.makedirs(gsmap_input, exist_ok=True)
    url = 'https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz'
    subprocess.run(f'wget -q {url} -P {gsmap_input}', shell=True, check=True)
    subprocess.run(
        f'tar -xzf {gsmap_input}/gsMap_resource.tar.gz -C {gsmap_input}',
        shell=True, check=True)
    os.remove(f'{gsmap_input}/gsMap_resource.tar.gz')
    print('downloaded gsMap resources')

os.makedirs(gwas_formatted_dir, exist_ok=True)
for f in sorted(os.listdir(gwas_dir)):
    if not f.endswith('.sumstats.gz'):
        continue
    basename = f.replace('.sumstats.gz', '')
    if os.path.exists(f'{gwas_formatted_dir}/{basename}.sumstats.gz'):
        continue
    subprocess.run(
        f"gsmap format_sumstats "
        f"--sumstats '{gwas_dir}/{f}' "
        f"--out '{gwas_formatted_dir}/{basename}'",
        shell=True, check=True)
    print(f'formatted {basename}')

with open(gwas_config_path, 'w') as fh:
    for gwas_file in sorted(os.listdir(gwas_formatted_dir)):
        if gwas_file.endswith('.sumstats.gz'):
            trait = gwas_file.replace('.sumstats.gz', '')
            path = os.path.abspath(f'{gwas_formatted_dir}/{gwas_file}')
            fh.write(f'{trait}: {path}\n')

with open(gwas_config_path) as fh:
    config = yaml.safe_load(fh)
all_traits = list(config.keys()) if config else []
print(f'{len(all_traits)} GWAS traits')

#endregion

#region run gsmap ##############################################################

for name in datasets:
    conditions = sorted(adatas[name].obs['condition'].unique())
    output = f'{working_dir}/output/{name}/gsmap'
    os.makedirs(output, exist_ok=True)

    for cond in conditions:
        cauchy_dir = f'{output}/{cond}/cauchy_combination'
        if os.path.isdir(cauchy_dir):
            n_done = len(list(Path(cauchy_dir).glob('*.Cauchy.csv.gz')))
            if n_done >= len(all_traits):
                print(f'[{name}] {cond} quick_mode done ({n_done} traits)')
                continue

        subprocess.run(
            f"gsmap quick_mode "
            f"--workdir '{output}' "
            f"--homolog_file '{homolog_file}' "
            f"--sample_name '{cond}' "
            f"--gsMap_resource_dir '{resource_dir}' "
            f"--hdf5_path '{gsmap_input}/{name}/ST/{cond}.h5ad' "
            f"--annotation '{cell_type_col}' "
            f"--data_layer 'counts' "
            f"--sumstats_config_file '{gwas_config_path}' "
            f"--max_processes {max(1, os.cpu_count() - 1)}",
            shell=True, check=True)
        print(f'[{name}] {cond} quick_mode done')

for name in datasets:
    conditions = sorted(adatas[name].obs['condition'].unique())
    output = f'{working_dir}/output/{name}/gsmap'
    for cond in conditions:
        d = f'{output}/{cond}/cauchy_combination'
        files = list(Path(d).glob('*.Cauchy.csv.gz')) \
            if os.path.isdir(d) else []
        if files:
            n = pl.read_csv(files[0])['annotation'].n_unique()
            print(f'[{name}] {cond}: {n} annotations, '
                  f'{len(files)} traits')

#endregion

#region analysis ###############################################################

for name, cfg in datasets.items():
    conditions = sorted(adatas[name].obs['condition'].unique())
    output = f'{working_dir}/output/{name}/gsmap'
    out_dir = f'{working_dir}/output/{name}'
    os.makedirs(out_dir, exist_ok=True)

    cell_types = set(adatas[name].obs[cell_type_col].unique())

    ct = cauchy_table(output, conditions, cell_types, traits_selected)
    if not ct.empty:
        ct.to_csv(f'{out_dir}/gsmap_cauchy_table.csv', index=False)
        print(f'[{name}] cauchy table: {ct.shape[0]} rows')

    ta = trait_associations(
        output, conditions, cfg['contrasts'], cell_types, all_traits)
    if not ta.empty:
        ta.to_csv(f'{out_dir}/gsmap_associations.csv', index=False)
        n_sig = int((ta['p_adj'] < FDR_THRESHOLD).sum())
        print(f'[{name}] associations: {ta.shape[0]} tests, '
              f'{n_sig} sig (FDR<{FDR_THRESHOLD})')
        ta_sig = ta[ta['p_adj'] < FDR_THRESHOLD].sort_values('p_adj')
        if not ta_sig.empty:
            ta_sig.to_csv(
                f'{out_dir}/gsmap_associations_sig.csv', index=False)

#endregion

#region plot — trait ranking ###################################################

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

BONFERRONI_ALPHA = 0.05

rankings = {}
thresholds = {}
for name in datasets:
    conditions = sorted(adatas[name].obs['condition'].unique())
    cell_types = set(adatas[name].obs[cell_type_col].unique())
    output = f'{working_dir}/output/{name}/gsmap'
    df = load_cauchy(output, conditions, cell_types)
    if df is None:
        continue
    stats = df.select(['trait', 'annotation']).collect()
    n_traits = stats['trait'].n_unique()
    n_ann = stats['annotation'].n_unique()
    thresholds[name] = -np.log10(BONFERRONI_ALPHA / (n_traits * n_ann))
    rankings[name] = (df
        .with_columns(p_log=(-pl.col('p_cauchy').log10()))
        .group_by('trait', 'annotation', 'condition')
        .agg(pl.median('p_log').alias('median_log_p'))
        .group_by('trait')
        .agg(pl.max('median_log_p').alias('peak_score'))
        .sort('peak_score', descending=True)
        .collect()
        .to_pandas())

trait_scores = {}
for ranking in rankings.values():
    for _, row in ranking.iterrows():
        trait_scores.setdefault(row['trait'], []).append(row['peak_score'])
trait_order = sorted(trait_scores, key=lambda t: np.mean(trait_scores[t]))

ds_list = [n for n in datasets if n in rankings]
fig, axes = plt.subplots(
    1, len(ds_list), figsize=(2.5 * len(ds_list), 3.5),
    sharey=True, squeeze=False)

for i, name in enumerate(ds_list):
    ax = axes[0, i]
    ranking = (rankings[name].set_index('trait')
               .reindex(trait_order).reset_index())
    ranking['peak_score'] = ranking['peak_score'].fillna(0)
    scores = ranking['peak_score'].values
    norm = mcolors.Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.get_cmap('GnBu')
    sns.barplot(
        x='peak_score', y='trait', data=ranking,
        hue='trait', palette=[cmap(norm(s)) for s in scores],
        ax=ax, orient='h', legend=False)
    ax.axvline(x=thresholds[name], color='black', linestyle='--',
               linewidth=1)
    ax.set_xlabel('Peak Score')
    ax.set_ylabel('')
    ax.set_title(name, fontsize=9)
    sns.despine(ax=ax)
    ax.tick_params(axis='y', length=0)

fig.tight_layout()
os.makedirs(f'{working_dir}/figures', exist_ok=True)
fig.savefig(f'{working_dir}/figures/gsmap_trait_ranking.png',
            dpi=300, bbox_inches='tight')
fig.savefig(f'{working_dir}/figures/gsmap_trait_ranking.svg',
            bbox_inches='tight')
plt.close(fig)

#endregion
