#region imports and setup ######################################################

import os
import subprocess
import warnings
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

warnings.filterwarnings('ignore')

working_dir = 'spatial-pregnancy'
cell_type_col = 'subclass'
trait = 'MDD'
datasets = ['slidetags', 'xenium']
conditions = ['CTRL', 'PREG']

gsmap_input = f'{working_dir}/input/gsmap'
gwas_formatted_dir = f'{gsmap_input}/GWAS_formatted'
resource_dir = f'{gsmap_input}/gsMap_resource'
homolog_file = f'{resource_dir}/homologs/mouse_human_homologs.txt'
sumstats_file = f'{gwas_formatted_dir}/{trait}.sumstats.gz'

#endregion

#region prepare decontam h5ads (corrected_counts from 05_postprocessing) #######

for name in datasets:
    for cond in conditions:
        st_dir = f'{gsmap_input}/{name}_decontam/ST'
        target = f'{st_dir}/{cond}.h5ad'
        if os.path.exists(target):
            print(f'[{name}] {cond} decontam h5ad exists')
            continue
        os.makedirs(st_dir, exist_ok=True)
        adata = sc.read_h5ad(
            f'{working_dir}/output/{name}'
            f'/03_adata_query_{name}.h5ad')
        adata = adata[adata.obs['condition'] == cond].copy()
        assert 'corrected_counts' in adata.layers, \
            'run 05_postprocessing first'
        adata.obsm['spatial'] = \
            adata.obs[['x_affine', 'y_affine']].to_numpy()
        adata.obs_names_make_unique()
        adata.write_h5ad(target)
        print(f'[{name}] {cond}: wrote {target} '
              f'({adata.shape[0]:,} cells)')

#endregion

#region run gsmap on decontam (single-trait, both conditions) ##################

for name in datasets:
    for cond in conditions:
        output = f'{working_dir}/output/{name}/gsmap_decontam'
        ldsc_file = f'{output}/{cond}/spatial_ldsc' \
                    f'/{cond}_{trait}.csv.gz'
        if os.path.exists(ldsc_file):
            print(f'[{name}] {cond} decontam gsmap already done')
            continue
        os.makedirs(output, exist_ok=True)
        result = subprocess.run(
            f"gsmap quick_mode "
            f"--workdir '{output}' "
            f"--homolog_file '{homolog_file}' "
            f"--sample_name '{cond}' "
            f"--gsMap_resource_dir '{resource_dir}' "
            f"--hdf5_path '{gsmap_input}/{name}_decontam"
                f"/ST/{cond}.h5ad' "
            f"--annotation '{cell_type_col}' "
            f"--data_layer 'corrected_counts' "
            f"--sumstats_file '{sumstats_file}' "
            f"--trait_name '{trait}' "
            f"--max_processes {max(1, os.cpu_count() - 1)}",
            shell=True)
        if not os.path.exists(ldsc_file):
            raise RuntimeError(
                f'[{name}] {cond} decontam gsmap failed')
        if result.returncode != 0:
            print(f'[{name}] {cond} report failed (results OK)')
        else:
            print(f'[{name}] {cond} decontam gsmap complete')

#endregion

#region load per-subclass medians across all (dataset, cond, run) ##############

def load_medians(name, cond, subdir):
    ldsc_path = f'{working_dir}/output/{name}/{subdir}' \
                f'/{cond}/spatial_ldsc/{cond}_{trait}.csv.gz'
    if not os.path.exists(ldsc_path):
        return None
    adata = sc.read_h5ad(
        f'{working_dir}/output/{name}'
        f'/03_adata_query_{name}.h5ad')
    obs = adata.obs[adata.obs['condition'] == cond]
    df = pl.read_csv(ldsc_path) \
        .with_columns(gsmap_score=-pl.col('p').log10())
    ann = pl.DataFrame({
        'spot': obs.index.tolist(),
        cell_type_col: obs[cell_type_col].astype(str).tolist(),
    })
    return df.join(ann, on='spot', how='inner') \
        .to_pandas() \
        .groupby(cell_type_col)['gsmap_score'].median()

runs = {'raw': 'gsmap', 'decontam': 'gsmap_decontam'}
medians = {}  # medians[run][name][cond] -> pd.Series
for run, subdir in runs.items():
    medians[run] = {}
    for name in datasets:
        medians[run][name] = {}
        for cond in conditions:
            m = load_medians(name, cond, subdir)
            medians[run][name][cond] = m
            tag = f'{m.shape[0]} subclasses' if m is not None \
                else 'MISSING'
            print(f'  [{run}] {name} {cond}: {tag}')

#endregion

#region cross-dataset concordance: per-condition + differential ################

def concord(a, b, label, run):
    if a is None or b is None:
        return None
    common = a.index.intersection(b.index)
    if len(common) < 5:
        return None
    av, bv = a.loc[common].values, b.loc[common].values
    rho, p_s = spearmanr(av, bv)
    r, p_p = pearsonr(av, bv)
    return {
        'run': run, 'test': label, 'n': len(common),
        'spearman_rho': rho, 'spearman_p': p_s,
        'pearson_r': r, 'pearson_p': p_p,
    }

rows = []
deltas = {}  # deltas[run][name] -> pd.Series of PREG - CTRL
for run in runs:
    # within-condition: slidetags vs xenium per subclass
    for cond in conditions:
        rows.append(concord(
            medians[run]['slidetags'][cond],
            medians[run]['xenium'][cond],
            cond, run))
    # differential (PREG - CTRL) per dataset
    deltas[run] = {}
    for name in datasets:
        preg = medians[run][name]['PREG']
        ctrl = medians[run][name]['CTRL']
        if preg is None or ctrl is None:
            continue
        common = preg.index.intersection(ctrl.index)
        deltas[run][name] = preg.loc[common] - ctrl.loc[common]
    # key test: cross-dataset concordance of differential signal
    rows.append(concord(
        deltas[run].get('slidetags'),
        deltas[run].get('xenium'),
        'PREG - CTRL', run))

summary = pd.DataFrame([r for r in rows if r])
print('\ncross-dataset concordance (slidetags vs xenium, per subclass):')
print(summary.to_string(
    index=False,
    float_format=lambda x: f'{x:.3f}'))
summary.to_csv(
    f'{working_dir}/output'
    f'/gsmap_decontam_concordance_{trait}.csv', index=False)

#endregion

#region per-subclass delta table for inspection ################################

dlt_rows = []
for run in runs:
    for name in datasets:
        if name not in deltas[run]:
            continue
        for ct, d in deltas[run][name].items():
            dlt_rows.append({
                'run': run, 'dataset': name,
                cell_type_col: ct,
                'delta_PREG_CTRL': d,
            })
dlt = pd.DataFrame(dlt_rows)
if not dlt.empty:
    wide = dlt.pivot_table(
        index=cell_type_col, columns=['run', 'dataset'],
        values='delta_PREG_CTRL')
    wide.columns = [f'{r}_{n}' for r, n in wide.columns]
    wide = wide.sort_values(
        wide.columns[0] if len(wide.columns) else 0,
        ascending=False)
    wide.to_csv(
        f'{working_dir}/output'
        f'/gsmap_decontam_delta_{trait}.csv')
    print(f'\nper-subclass delta table: '
          f'{wide.shape[0]} subclasses x {wide.shape[1]} runs')

#endregion
