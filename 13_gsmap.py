#region imports and setup ######################################################

import os
import pickle
import subprocess
import time
import warnings
from collections import defaultdict
from itertools import combinations, product
from math import comb, factorial
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import polars as pl
import scanpy as sc
from scipy.stats import t as t_dist

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

#endregion

#region prep data ##############################################################

adatas = {}
for name, cfg in datasets.items():
    adata = sc.read_h5ad(cfg['path'])
    if drop := cfg.get('drop_samples'):
        adata = adata[~adata.obs['sample'].isin(drop)].copy()
        print(f'[{name}] dropped samples: {drop}')
    adatas[name] = adata
    print(f'[{name}] {adata.shape[0]:,} cells, '
          f'{adata.obs[cell_type_col].nunique()} subclasses, '
          f'{adata.obs["sample"].nunique()} samples')

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
    url = ('https://yanglab.westlake.edu.cn/data/gsMap'
           '/gsMap_resource.tar.gz')
    tarball = f'{gsmap_input}/gsMap_resource.tar.gz'
    subprocess.run(f'wget -q {url} -P {gsmap_input}',
                   shell=True, check=True)
    subprocess.run(f'tar -xzf {tarball} -C {gsmap_input}',
                   shell=True, check=True)
    os.remove(tarball)
    print('downloaded gsMap resources')

os.makedirs(gwas_formatted_dir, exist_ok=True)
for f in sorted(os.listdir(gwas_dir)):
    if not f.endswith('.sumstats.gz'):
        continue
    basename = f.replace('.sumstats.gz', '')
    if os.path.exists(f'{gwas_formatted_dir}/{basename}.sumstats.gz'):
        continue
    subprocess.run(
        f"python -m gsMap format_sumstats "
        f"--sumstats '{gwas_dir}/{f}' "
        f"--out '{gwas_formatted_dir}/{basename}'",
        shell=True, check=True)
    print(f'formatted {basename}')

all_traits = sorted(
    f.removesuffix('.sumstats.gz')
    for f in os.listdir(gwas_formatted_dir)
    if f.endswith('.sumstats.gz'))
print(f'{len(all_traits)} GWAS traits')

#endregion

#region run gsmap ##############################################################

def submit_slurm(cmd, *, job_name, log_file, depends=None, hours=24):
    is_trillium = os.environ.get('CLUSTER', '').startswith('trillium')
    sbatch = '.sbatch' if is_trillium else 'sbatch'
    lines = ['#!/bin/bash']
    if is_trillium:
        lines.append('#SBATCH -p compute')
    lines.append('#SBATCH --account=rrg-shreejoy')
    if not is_trillium:
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

try:
    active = set(subprocess.check_output(
        'squeue -h -u "$USER" -o "%500j"',
        shell=True, text=True).split())
except subprocess.CalledProcessError:
    active = set()
print(f'{len(active)} active jobs')

for name in datasets:
    conditions = sorted(adatas[name].obs['condition'].unique())
    output = f'{working_dir}/output/{name}/gsmap'
    os.makedirs(output, exist_ok=True)

    for cond in conditions:
        hdf5 = f'{gsmap_input}/{name}/ST/{cond}.h5ad'
        done_marker = (f'{output}/{cond}/generate_ldscore/'
                       f'{cond}_generate_ldscore.done')
        ldsc_done = os.path.exists(done_marker)

        def tag_for(t):
            return (f'{ds_abbr.get(name, name[:2])}_'
                    f'{cond_abbr.get(cond, cond)}_{t}')

        done, queued, pending = [], [], []
        for t in all_traits:
            if os.path.exists(f'{output}/{cond}/cauchy_combination/'
                              f'{cond}_{t}.Cauchy.csv.gz'):
                done.append(t)
            elif tag_for(t) in active:
                queued.append(t)
            else:
                pending.append(t)
        print(f'[{name}] {cond}: {len(done)} done, '
              f'{len(queued)} in-flight, {len(pending)} to submit')
        if not pending:
            continue

        def submit(trait, depends=None):
            cauchy = (f'{output}/{cond}/cauchy_combination/'
                      f'{cond}_{trait}.Cauchy.csv.gz')
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
                f"--max_processes $(nproc) "
                f"|| {{ test -f '{done_marker}' && test -f '{cauchy}'; }}")
            return submit_slurm(
                cmd, job_name=tag_for(trait),
                log_file=f'{log_dir}/{name}_{cond}_{trait}.log',
                depends=depends)

        anchor_jid = None
        if not ldsc_done:
            anchor = pending.pop(0)
            anchor_jid = submit(anchor)
            print(f'  {anchor} (prep+ldsc) -> {anchor_jid}')
        for trait in pending:
            jid = submit(trait, depends=anchor_jid)
            tail = f' (waits {anchor_jid})' if anchor_jid else ''
            print(f'  {trait} -> {jid}{tail}')

#endregion

#region meta analysis ##########################################################
# Sumrank-style cross-platform meta-analysis of differential gsMap enrichment.
# Adapted from 06_sumrank.py: cell-types replace genes as the ranked unit,
# datasets replace platforms. Primary stat per (dataset, trait, cell-type):
# Welch t on per-sample mean -log10(p) (sample-level inference, not per-cell).
# Ranks within (dataset, trait), summed across datasets, Irwin-Hall analytical
# p, label-permutation empirical p.

META_CONTRAST_PLATFORMS = {
    'PREG_vs_CTRL':     ['slidetags', 'xenium'],
    'POSTPART_vs_PREG': ['slidetags'],
    'POSTPART_vs_CTRL': ['slidetags'],
}
META_MIN_CELLS_PER_SAMPLE = 20  # (sample, cell-type) included only if >= N cells
META_MIN_SAMPLES_PER_COND = 2   # need >= N samples per condition to test
META_PERM_CAP = 10000           # exhaustive if total combos <= this, else random
META_SEED = 12345
META_FDR = FDR_THRESHOLD

meta_out = f'{working_dir}/output'
meta_perm_dir = f'{meta_out}/gsmap_meta_perms'
os.makedirs(meta_perm_dir, exist_ok=True)

def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    valid = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    if not valid.any():
        return out
    pv = p[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    q = (pv * n / ranks)[order]
    q = np.minimum.accumulate(q[::-1])[::-1]
    q_final = np.empty(n)
    q_final[order] = np.clip(q, 0, 1)
    out[valid] = q_final
    return out

def irwin_hall_cdf(x, n):
    # CDF of sum of n iid Uniform(0,1). Analytical null for sum of normalized
    # ranks across n platforms.
    x = np.clip(np.atleast_1d(x).astype(float), 0.0, float(n))
    out = np.zeros_like(x)
    for k in range(n + 1):
        diff = x - k
        m = diff > 0
        if m.any():
            out[m] += ((-1) ** k) * comb(n, k) * diff[m] ** n
    return out / factorial(n)

def load_ldsc_long(name, adata):
    # Per-cell long frame: (spot, trait, dataset, score, sample, condition, cell_type)
    obs = adata.obs[['sample', 'condition', cell_type_col]].astype(str)
    obs_pl = pl.from_pandas(
        obs.reset_index().rename(columns={
            'index': 'spot', cell_type_col: 'cell_type'}))
    base = f'{meta_out}/{name}/gsmap'
    frames = []
    for cond in sorted(obs['condition'].unique()):
        d = Path(base) / cond / 'spatial_ldsc'
        if not d.is_dir():
            continue
        for f in sorted(d.glob(f'{cond}_*.csv.gz')):
            trait = f.name.replace(f'{cond}_', '').replace('.csv.gz', '')
            frames.append(
                pl.scan_csv(str(f))
                  .with_columns(
                      score=-pl.col('p').log10(),
                      trait=pl.lit(trait),
                      dataset=pl.lit(name))
                  .select(['spot', 'trait', 'dataset', 'score']))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).collect().join(obs_pl, on='spot', how='inner')

def compute_sample_means(long_df):
    # Collapse cells -> per-sample mean -log10(p), min-cells filter.
    return long_df \
        .group_by(['dataset', 'trait', 'cell_type', 'sample', 'condition']) \
        .agg([pl.col('score').mean().alias('mean_score'),
              pl.col('score').count().alias('n_cells')]) \
        .filter(pl.col('n_cells') >= META_MIN_CELLS_PER_SAMPLE)

def per_dataset_diff(means, contrast):
    # Welch t on sample means between treat and base of `contrast`, per
    # (dataset, trait, cell_type). Vectorized.
    treat, base = contrast.split('_vs_')
    agg = means \
        .filter(pl.col('condition').is_in([treat, base])) \
        .group_by(['dataset', 'trait', 'cell_type', 'condition']) \
        .agg([pl.col('mean_score').mean().alias('m'),
              pl.col('mean_score').var(ddof=1).fill_null(0.0).alias('v'),
              pl.col('mean_score').count().alias('n')])
    treat_df = agg.filter(pl.col('condition') == treat) \
        .drop('condition') \
        .rename({'m': 'm_t', 'v': 'v_t', 'n': 'n_t'})
    base_df = agg.filter(pl.col('condition') == base) \
        .drop('condition') \
        .rename({'m': 'm_b', 'v': 'v_b', 'n': 'n_b'})
    wide = treat_df.join(
        base_df, on=['dataset', 'trait', 'cell_type'], how='inner') \
        .filter((pl.col('n_t') >= META_MIN_SAMPLES_PER_COND) &
                (pl.col('n_b') >= META_MIN_SAMPLES_PER_COND))
    if wide.height == 0:
        return pl.DataFrame()
    m_t = wide['m_t'].to_numpy()
    m_b = wide['m_b'].to_numpy()
    v_t = np.nan_to_num(wide['v_t'].to_numpy())
    v_b = np.nan_to_num(wide['v_b'].to_numpy())
    n_t = wide['n_t'].to_numpy().astype(float)
    n_b = wide['n_b'].to_numpy().astype(float)
    beta = m_t - m_b
    se2 = v_t / n_t + v_b / n_b
    with np.errstate(divide='ignore', invalid='ignore'):
        se = np.sqrt(se2)
        t = np.where(se > 0, beta / se, np.nan)
        num = se2 ** 2
        den = (v_t ** 2) / (n_t ** 2 * np.maximum(n_t - 1, 1)) \
            + (v_b ** 2) / (n_b ** 2 * np.maximum(n_b - 1, 1))
        df = np.where(den > 0, num / den, 1.0)
    p = np.where(np.isfinite(t), 2 * t_dist.sf(np.abs(t), df), np.nan)
    return wide.with_columns([
        pl.Series('beta', beta),
        pl.Series('t', t),
        pl.Series('p', p),
        pl.lit(contrast).alias('contrast'),
    ]).select([
        'dataset', 'contrast', 'trait', 'cell_type',
        'beta', 't', 'p',
        pl.col('n_t').alias('n_treat'),
        pl.col('n_b').alias('n_base'),
    ])

def sumrank_gsmap(stats_df, platforms, contrast):
    # Within each trait, rank cell-types per dataset by signed -log10(p),
    # sum normalized ranks across datasets. Cell-types present in only one
    # dataset (D<2) are dropped.
    out = []
    for trait in stats_df['trait'].unique().to_list():
        sub_t = stats_df.filter(pl.col('trait') == trait)
        rank_dfs, active = [], []
        for ds in platforms:
            s = sub_t.filter(pl.col('dataset') == ds)
            if s.height == 0:
                continue
            pv = np.clip(s['p'].to_numpy().astype(float), 1e-300, 1.0)
            sign = np.sign(s['beta'].to_numpy())
            score = np.nan_to_num(-np.log10(pv) * sign)
            order = np.argsort(-score, kind='stable')
            rank = np.empty(len(score), dtype=np.int64)
            rank[order] = np.arange(1, len(score) + 1)
            nrank = (rank - 1) / max(len(score) - 1, 1)
            rank_dfs.append(pl.DataFrame({
                'cell_type': s['cell_type'],
                f'nrank_{ds}': nrank}))
            active.append(ds)
        if len(rank_dfs) < 2:
            continue
        merged = rank_dfs[0]
        for rd in rank_dfs[1:]:
            merged = merged.join(rd, on='cell_type', how='full',
                                 coalesce=True)
        arr = merged.select(
            [f'nrank_{p}' for p in active]).to_numpy()
        d = (~np.isnan(arr)).sum(axis=1)
        s_sum = np.nansum(arr, axis=1)
        keep = d >= 2
        if not keep.any():
            continue
        nlp_up = np.full(len(d), np.nan)
        nlp_dn = np.full(len(d), np.nan)
        for k in np.unique(d[keep]):
            idx = (d == int(k)) & keep
            cdf = irwin_hall_cdf(s_sum[idx], int(k))
            nlp_up[idx] = -np.log10(np.clip(cdf, 1e-300, 1.0))
            nlp_dn[idx] = -np.log10(np.clip(1 - cdf, 1e-300, 1.0))
        out.append(pl.DataFrame({
            'contrast': contrast,
            'trait': trait,
            'cell_type': merged['cell_type'],
            'D': d.astype(np.int64),
            'sum_stat': s_sum,
            'nlp_up': nlp_up,
            'nlp_down': nlp_dn,
        }).filter(pl.col('D') >= 2))
    return pl.concat(out) if out else pl.DataFrame()

def enum_perm_space(means, contrast):
    # For each dataset, enumerate all C(n_samples, n_treat) unique label
    # partitions. Returns (datasets_order, per_ds_samples, per_ds_cond_tuples,
    # total_combos).
    treat, base = contrast.split('_vs_')
    per_ds_samples, per_ds_cond_tuples = {}, {}
    for ds in sorted(means['dataset'].unique().to_list()):
        meta = means.filter(pl.col('dataset') == ds) \
            .filter(pl.col('condition').is_in([treat, base])) \
            .select(['sample', 'condition']).unique() \
            .sort('sample')
        samples = meta['sample'].to_list()
        n_treat = meta.filter(pl.col('condition') == treat).height
        combos = [
            tuple(treat if i in idx else base for i in range(len(samples)))
            for idx in combinations(range(len(samples)), n_treat)]
        per_ds_samples[ds] = samples
        per_ds_cond_tuples[ds] = combos
    datasets_order = list(per_ds_samples.keys())
    total = 1
    for ds in datasets_order:
        total *= len(per_ds_cond_tuples[ds])
    return datasets_order, per_ds_samples, per_ds_cond_tuples, total

def iter_perm_maps(means, contrast, cap=META_PERM_CAP, seed=META_SEED):
    # Exhaustive if total combos <= cap, else random sample `cap` times.
    # First yield: dict with mode ('exhaustive'|'random'), n_perms, total.
    ds_order, samples_by_ds, combos_by_ds, total = enum_perm_space(
        means, contrast)
    if total <= cap:
        mode = 'exhaustive'
        n_perms = total
        combo_iter = product(*[combos_by_ds[ds] for ds in ds_order])
    else:
        mode = 'random'
        n_perms = cap
        rng = np.random.default_rng(seed)
        def _sampler():
            for _ in range(cap):
                yield tuple(
                    combos_by_ds[ds][rng.integers(len(combos_by_ds[ds]))]
                    for ds in ds_order)
        combo_iter = _sampler()
    yield {'mode': mode, 'n_perms': n_perms, 'total_unique': total}
    for combo in combo_iter:
        rows = []
        for ds, cond_tuple in zip(ds_order, combo):
            for s, c in zip(samples_by_ds[ds], cond_tuple):
                rows.append({'dataset': ds, 'sample': s,
                             'perm_condition': c})
        yield pl.DataFrame(rows)

def apply_perm_map(means, perm_map):
    return means.drop('condition').join(
        perm_map, on=['dataset', 'sample'], how='inner'
    ).rename({'perm_condition': 'condition'})

def calibrate_emp_p(real, null_by_ct):
    # Empirical p per (trait, cell-type): tail fraction of null for that cell-
    # type. Pooled up+down under exchangeability doubles null resolution.
    emp_up = np.full(real.height, np.nan)
    emp_dn = np.full(real.height, np.nan)
    nlp_up = real['nlp_up'].to_numpy()
    nlp_dn = real['nlp_down'].to_numpy()
    cts = real['cell_type'].to_numpy()
    for ct in np.unique(cts):
        nu = null_by_ct.get(ct, np.array([]))
        if len(nu) == 0:
            continue
        msk = cts == ct
        for vals, emp in [(nlp_up[msk], emp_up), (nlp_dn[msk], emp_dn)]:
            idx = np.searchsorted(nu, vals, side='left')
            emp[msk] = np.maximum((len(nu) - idx) / len(nu),
                                  1.0 / len(nu))
    return emp_up, emp_dn

def summarize_null_by_ct(sr_k, null_by_ct):
    for ct in sr_k['cell_type'].unique().to_list():
        sub = sr_k.filter(pl.col('cell_type') == ct)
        u = sub['nlp_up'].to_numpy()
        d = sub['nlp_down'].to_numpy()
        null_by_ct[ct].append(u[~np.isnan(u)])
        null_by_ct[ct].append(d[~np.isnan(d)])

# ---- main pipeline ----------------------------------------------------------

# 1. per-cell scores -> per-sample means (cached)
means_cache = f'{meta_out}/gsmap_sample_means.parquet'
if os.path.exists(means_cache):
    means = pl.read_parquet(means_cache)
    print(f'[meta] sample means cached: {means.height:,} rows')
else:
    long_frames = []
    for name in datasets:
        df = load_ldsc_long(name, adatas[name])
        if df.height:
            long_frames.append(df)
            print(f'[meta] {name}: {df.height:,} cell-trait rows loaded')
    if long_frames:
        means = compute_sample_means(pl.concat(long_frames, how='diagonal'))
        means.write_parquet(means_cache)
        print(f'[meta] computed {means.height:,} sample means')
    else:
        means = pl.DataFrame()

# 2. per-dataset Welch t per contrast
per_ds_frames = []
for contrast, platforms in META_CONTRAST_PLATFORMS.items():
    for ds in platforms:
        if ds not in datasets:
            continue
        ds_contrasts = {f'{t}_vs_{c}' for t, c in datasets[ds]['contrasts']}
        if contrast not in ds_contrasts:
            continue
        d = per_dataset_diff(
            means.filter(pl.col('dataset') == ds), contrast)
        if d.height:
            per_ds_frames.append(d)
            print(f'[meta] {ds} {contrast}: '
                  f'{d.height} (trait, cell-type) tests')

if per_ds_frames:
    per_dataset = pl.concat(per_ds_frames)
    per_dataset = per_dataset.with_columns(
        pl.Series('fdr', bh_fdr(per_dataset['p'].to_numpy())))
    per_dataset.write_csv(f'{meta_out}/gsmap_per_dataset.csv')
else:
    per_dataset = pl.DataFrame()

# 3. meta-analysis + permutation null for contrasts with >=2 platforms
meta_frames = []
for contrast, platforms in META_CONTRAST_PLATFORMS.items():
    if len(platforms) < 2 or per_dataset.height == 0:
        continue
    stats_c = per_dataset.filter(pl.col('contrast') == contrast)
    if stats_c.height == 0:
        continue
    real = sumrank_gsmap(stats_c, platforms, contrast)
    if real.height == 0:
        print(f'[meta] {contrast}: no (trait, cell-type) with D>=2')
        continue

    # permutation null (cached per contrast)
    cache_path = f'{meta_perm_dir}/null_{contrast}.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as fh:
            null_by_ct = pickle.load(fh)
        print(f'[meta] {contrast}: null cached')
    else:
        treat, base = contrast.split('_vs_')
        ds_means = means.filter(
            pl.col('dataset').is_in(platforms) &
            pl.col('condition').is_in([treat, base]))
        perm_iter = iter_perm_maps(ds_means, contrast)
        info = next(perm_iter)
        n_perms = info['n_perms']
        print(f'[meta] {contrast}: {info["mode"]} perms '
              f'({n_perms} of {info["total_unique"]} unique combos)')
        null_by_ct = defaultdict(list)
        t0 = time.time()
        step = max(n_perms // 10, 1)
        for k, perm_map in enumerate(perm_iter):
            perm_means = apply_perm_map(ds_means, perm_map)
            per_ds_k = []
            for ds in platforms:
                d = per_dataset_diff(
                    perm_means.filter(pl.col('dataset') == ds), contrast)
                if d.height:
                    per_ds_k.append(d)
            if not per_ds_k:
                continue
            sr_k = sumrank_gsmap(
                pl.concat(per_ds_k), platforms, contrast)
            if sr_k.height:
                summarize_null_by_ct(sr_k, null_by_ct)
            if (k + 1) % step == 0 or k + 1 == n_perms:
                el = time.time() - t0
                print(f'[meta] {contrast}: perm {k+1}/{n_perms} '
                      f'({el:.0f}s, eta '
                      f'{el*(n_perms-k-1)/(k+1):.0f}s)',
                      flush=True)
        null_by_ct = {
            ct: np.sort(np.concatenate(arrs)) if arrs else np.array([])
            for ct, arrs in null_by_ct.items()}
        with open(cache_path, 'wb') as fh:
            pickle.dump(null_by_ct, fh)
        print(f'[meta] {contrast}: null saved '
              f'({(time.time()-t0)/60:.1f} min)')

    emp_up, emp_dn = calibrate_emp_p(real, null_by_ct)
    # Analytical p from Irwin-Hall, direct from nlp columns.
    ana_up = np.clip(10.0 ** (-real['nlp_up'].to_numpy()), 0, 1)
    ana_dn = np.clip(10.0 ** (-real['nlp_down'].to_numpy()), 0, 1)
    real = real.with_columns([
        pl.Series('p_up', ana_up),
        pl.Series('p_down', ana_dn),
        pl.Series('fdr_up', bh_fdr(ana_up)),
        pl.Series('fdr_down', bh_fdr(ana_dn)),
        pl.Series('emp_p_up', emp_up),
        pl.Series('emp_p_down', emp_dn),
        pl.Series('emp_fdr_up', bh_fdr(emp_up)),
        pl.Series('emp_fdr_down', bh_fdr(emp_dn)),
    ])
    meta_frames.append(real)

if meta_frames:
    meta = pl.concat(meta_frames)
    # attach per-dataset betas for interpretability
    beta_wide = per_dataset.select(
        ['contrast', 'trait', 'cell_type', 'dataset', 'beta']) \
        .pivot(on='dataset', index=['contrast', 'trait', 'cell_type'],
               values='beta')
    beta_wide = beta_wide.rename({
        c: f'beta_{c}' for c in beta_wide.columns
        if c not in ('contrast', 'trait', 'cell_type')})
    meta = meta.join(
        beta_wide, on=['contrast', 'trait', 'cell_type'], how='left')
    meta.write_csv(f'{meta_out}/gsmap_meta.csv')
else:
    meta = pl.DataFrame()

# 4. Cauchy per-condition enrichment with BH-FDR (for Fig 5A/5B plotting)
#    Scope: FDR across (trait x cell-type) within each (dataset, condition).
cauchy_frames = []
for name in datasets:
    base = f'{meta_out}/{name}/gsmap'
    for cond in sorted(adatas[name].obs['condition'].unique()):
        d = Path(base) / cond / 'cauchy_combination'
        if not d.is_dir():
            continue
        files = sorted(d.glob(f'{cond}_*.Cauchy.csv.gz'))
        if not files:
            continue
        parts = [
            pl.scan_csv(str(f))
              .with_columns(
                  trait=pl.lit(f.name
                    .replace(f'{cond}_', '')
                    .replace('.Cauchy.csv.gz', '')),
                  dataset=pl.lit(name),
                  condition=pl.lit(cond))
              .rename({'annotation': 'cell_type'})
              .select(['dataset', 'condition', 'trait', 'cell_type',
                       'p_cauchy', 'p_median'])
            for f in files]
        df = pl.concat(parts).collect()
        df = df.with_columns(
            pl.Series('fdr_cauchy', bh_fdr(df['p_cauchy'].to_numpy())),
            pl.Series('neg_log10_p_cauchy',
                      -np.log10(np.clip(
                          df['p_cauchy'].to_numpy(), 1e-300, 1.0))))
        cauchy_frames.append(df)
        n_sig = df.filter(pl.col('fdr_cauchy') < META_FDR).height
        print(f'[meta] cauchy {name} {cond}: {df.height} tests, '
              f'{n_sig} FDR<{META_FDR}')
if cauchy_frames:
    pl.concat(cauchy_frames).write_csv(f'{meta_out}/gsmap_cauchy_fdr.csv')

# 5. summary
for contrast in META_CONTRAST_PLATFORMS:
    if per_dataset.height:
        s = per_dataset.filter(pl.col('contrast') == contrast)
        if s.height:
            n_sig = s.filter(pl.col('fdr') < META_FDR).height
            print(f'[meta] {contrast} per-dataset: {s.height} tests, '
                  f'{n_sig} FDR<{META_FDR}')
    if meta.height:
        s = meta.filter(pl.col('contrast') == contrast)
        if s.height:
            n_up_a = s.filter(pl.col('fdr_up') < META_FDR).height
            n_dn_a = s.filter(pl.col('fdr_down') < META_FDR).height
            n_up_e = s.filter(pl.col('emp_fdr_up') < META_FDR).height
            n_dn_e = s.filter(pl.col('emp_fdr_down') < META_FDR).height
            n_ct = s['cell_type'].n_unique()
            n_tr = s['trait'].n_unique()
            print(f'[meta] {contrast} meta: {n_tr} traits x {n_ct} cts; '
                  f'analytical {n_up_a} up / {n_dn_a} down; '
                  f'empirical {n_up_e} up / {n_dn_e} down '
                  f'(FDR<{META_FDR})')

#endregion

