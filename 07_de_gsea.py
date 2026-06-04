#region imports & CLI #########################################################

import os
import gc
import re
import sys
import glob
import time
import argparse
import subprocess
import pickle as pkl
import warnings
from math import comb, factorial
from collections import defaultdict
from tempfile import NamedTemporaryFile
sys.path.insert(0, os.path.expanduser('~'))

os.environ.setdefault('POLARS_MAX_THREADS', '16')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r, to_py

warnings.filterwarnings('ignore')

from single_cell import SingleCell
from single_cell import _voomByGroup_source_code

_p = argparse.ArgumentParser()
_p.add_argument('--perm-job', choices=['de', 'gsea'], default=None,
                help='worker mode: run perms for one (platform, contrast)')
_p.add_argument('--platform', default=None)
_p.add_argument('--contrast', default=None)
_p.add_argument('--perm-start', type=int, default=None,
                help='0-indexed inclusive start of perm range')
_p.add_argument('--perm-end', type=int, default=None,
                help='0-indexed exclusive end of perm range')
_a, _ = _p.parse_known_args()

IS_WORKER = _a.perm_job is not None
WORKER_STAGE = _a.perm_job
WORKER_PLATFORM = _a.platform
WORKER_CONTRAST = _a.contrast
WORKER_PERM_START = _a.perm_start
WORKER_PERM_END = _a.perm_end
if IS_WORKER and (WORKER_PLATFORM is None or WORKER_CONTRAST is None
                  or WORKER_PERM_START is None or WORKER_PERM_END is None):
    raise SystemExit(
        '--perm-job requires --platform, --contrast, '
        '--perm-start, --perm-end')

#endregion

#region config ################################################################

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
de_suffix = f'_{cell_type_col}' if cell_type_col != 'subclass' else ''

datasets = {
    'slidetags': {
        'path': f'{working_dir}/output/slidetags'
                f'/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
    },
    'merfish': {
        'path': f'{working_dir}/output/merfish'
                f'/03_adata_query_merfish.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium'
                f'/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
    },
}

DESIGN_FORMULAS = {
    'slidetags': '~ 0 + condition + log2(num_cells) + log2(library_size)',
    'merfish':   '~ 0 + condition + log2(num_cells) + log2(library_size)',
    'xenium':    '~ 0 + condition + log2(num_cells)',
}
SUMRANK_CONTRAST_PLATFORMS = {
    'PREG_vs_CTRL': ['slidetags', 'merfish', 'xenium'],
    'POSTPART_vs_PREG': ['slidetags', 'merfish'],
    'POSTPART_vs_CTRL': ['slidetags', 'merfish'],
}

SUMRANK_N_PERM = 1000
SUMRANK_PERM_BATCH = 100   # perms per SLURM job
SUMRANK_CHUNK_SIZE = 20    # perms per chunk parquet
SUMRANK_N_CORES = os.cpu_count() if IS_WORKER else 16
SUMRANK_GSEA_CHUNK = 20
SUMRANK_GSEA_PERM_BATCH = 100
SUMRANK_GSEA_N_CORES = os.cpu_count() if IS_WORKER else 16
SUMRANK_GSEA_NPERM_SIMPLE = 100
PERM_SEED_BASE = 12345
MIN_N_CELLS_EXPR = 10

sumrank_cache_dir = f'{working_dir}/output/de/perms{de_suffix}'
SUMRANK_GSEA_CACHE_DIR = \
    f'{working_dir}/output/gsea/perms{de_suffix}'
PERM_LOG_DIR = f'{working_dir}/output/de/perm_logs'
de_path = f'{working_dir}/output/de/de_results{de_suffix}.csv'
sumrank_path = f'{working_dir}/output/de/sumrank_results{de_suffix}.csv'
sumrank_gsea_path = \
    f'{working_dir}/output/gsea/sumrank_gsea_results{de_suffix}.csv'
for d in (sumrank_cache_dir, SUMRANK_GSEA_CACHE_DIR, PERM_LOG_DIR):
    os.makedirs(d, exist_ok=True)

# Worker mode: restrict datasets to its one target (platform, contrast)
if IS_WORKER:
    if WORKER_PLATFORM not in datasets:
        raise SystemExit(f'unknown platform {WORKER_PLATFORM}')
    for _ds in list(datasets):
        if _ds != WORKER_PLATFORM:
            del datasets[_ds]
    treat, base = WORKER_CONTRAST.split('_vs_')
    datasets[WORKER_PLATFORM]['contrasts'] = [(treat, base)]
    print(f'[worker] stage={WORKER_STAGE} platform={WORKER_PLATFORM} '
          f'contrast={WORKER_CONTRAST} '
          f'perms={WORKER_PERM_START}:{WORKER_PERM_END}', flush=True)

#endregion

#region slurm #################################################################

PLATFORM_ABBR = {'slidetags': 'sl', 'merfish': 'mf', 'xenium': 'xn'}
CONTRAST_ABBR = {'PREG_vs_CTRL': 'PvC',
                 'POSTPART_vs_PREG': 'PPvP',
                 'POSTPART_vs_CTRL': 'PPvC'}

def _batch_tag(stage, plt_name, contrast, start, end):
    return (f'{stage}_{PLATFORM_ABBR.get(plt_name, plt_name[:2])}_'
            f'{CONTRAST_ABBR.get(contrast, contrast)}_{start}_{end}')

def _active_slurm_jobs():
    try:
        out = subprocess.check_output(
            'squeue -h -u "$USER" -o "%j"', shell=True, text=True)
    except subprocess.CalledProcessError:
        return set()
    return {ln.strip() for ln in out.splitlines() if ln.strip()}

LOGIN_HOST = 'tri-login01'

def submit_slurm(cmd, *, job_name, log_file, hours=6):
    script = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p compute',
        '#SBATCH --account=rrg-shreejoy',
        '#SBATCH -N 1',
        '#SBATCH -n 1',
        f'#SBATCH -t {hours}:00:00',
        f'#SBATCH -J {job_name}',
        f'#SBATCH -o {log_file}',
        f"bash -i -c 'export POLARS_MAX_THREADS=192; "
        f"set -euo pipefail; {cmd}'",
    ]) + '\n'
    scratch = os.environ.get('SCRATCH', '.')
    with NamedTemporaryFile(
        'w', dir=scratch, suffix='.sh', delete=False) as fh:
        fh.write(script)
        sp = fh.name
    try:
        subprocess.check_call(
            f'ssh -o BatchMode=yes {LOGIN_HOST} '
            f'CLUSTER=trillium /opt/slurm/bin/sbatch '
            f'--export=NONE --get-user-env=L {sp}',
            shell=True, executable='/bin/bash')
    finally:
        os.unlink(sp)

def _batch_ranges(n_perm, batch_perms):
    return [(s, min(s + batch_perms, n_perm))
            for s in range(0, n_perm, batch_perms)]

def chunk_map(pattern):
    return {int(m.group(1)): p for p in glob.glob(pattern)
            if (m := re.search(r'_chunk_(\d+)\.parquet$', p))}

def _merge_chunks_if_ready(
        pair, parquet_stage, cache_dir, chunk_size, n_perm, unique_cols):
    name, contrast = pair
    final_path = f'{cache_dir}/{parquet_stage}_{name}_{contrast}.parquet'
    if os.path.exists(final_path):
        return False
    chunk_glob = (
        f'{cache_dir}/{parquet_stage}_{name}_{contrast}_chunk_*.parquet')
    existing = chunk_map(chunk_glob)
    needed = list(range(0, n_perm, chunk_size))
    if not all(s in existing for s in needed):
        return False
    full = pl.concat([pl.read_parquet(existing[s]) for s in needed])\
        .unique(subset=unique_cols)
    full.write_parquet(final_path)
    for p in glob.glob(chunk_glob):
        os.remove(p)
    print(f'[merge] {parquet_stage} {name} {contrast}: '
          f'{full.height:,} rows -> {final_path}', flush=True)
    return True

def _sumrank_pairs():
    out = []
    for c, plts in SUMRANK_CONTRAST_PLATFORMS.items():
        for n in plts:
            if n not in datasets:
                continue
            ds_ctrs = {f'{t}_vs_{cc}' for t, cc in datasets[n]['contrasts']}
            if c in ds_ctrs:
                out.append((n, c))
    return out

def _submit_missing_batches(
        *, parquet_stage, job_stage, perm_job, pairs, cache_dir,
        batch_perms, chunk_size, active_jobs):
    submitted, pending = [], []
    for name, contrast in pairs:
        final_path = (
            f'{cache_dir}/{parquet_stage}_{name}_{contrast}.parquet')
        if os.path.exists(final_path):
            continue
        existing = chunk_map(
            f'{cache_dir}/{parquet_stage}_{name}_{contrast}'
            f'_chunk_*.parquet')
        for start, end in _batch_ranges(SUMRANK_N_PERM, batch_perms):
            batch_chunks = list(range(start, end, chunk_size))
            if all(s in existing for s in batch_chunks):
                continue
            tag = _batch_tag(job_stage, name, contrast, start, end)
            if tag in active_jobs:
                pending.append(tag)
                continue
            cmd = (f'{sys.executable} {os.path.abspath(__file__)} '
                   f'--perm-job={perm_job} --platform={name} '
                   f'--contrast={contrast} --perm-start={start} '
                   f'--perm-end={end}')
            submit_slurm(cmd, job_name=tag,
                         log_file=f'{PERM_LOG_DIR}/{tag}.log', hours=6)
            submitted.append(tag)
            pending.append(tag)
    return submitted, pending

def _gate_on_missing(parquet_stage, cache_dir, label):
    missing = [(n, c) for n, c in _sumrank_pairs()
               if not os.path.exists(
                   f'{cache_dir}/{parquet_stage}_{n}_{c}.parquet')]
    if missing:
        print(f'[{label}] waiting on {len(missing)} {parquet_stage} '
              f'parquets:')
        for n, c in missing:
            print(f'  {n} {c}')
        print(f'[{label}] re-run once SLURM jobs complete')
        sys.exit(0)

#endregion

#region adatas & reference pct detected #######################################

pct_file = f'{working_dir}/output/de/ref_pct_detected{de_suffix}.pkl'
if not os.path.exists(pct_file):
    ref = sc.read_h5ad(os.path.expanduser(
        '~/single-cell/ABC/zeng_combined_10Xv3.h5ad'))
    ref.var_names_make_unique()
    pct_detected = {}
    for s in ref.obs[cell_type_col].dropna().unique():
        mask = ref.obs[cell_type_col] == s
        n_cells = mask.sum()
        if n_cells < 10:
            continue
        detected = np.ravel((ref[mask].X > 0).sum(axis=0)) / n_cells
        pct_detected[s] = pd.Series(detected, index=ref.var_names)
    with open(pct_file, 'wb') as f:
        pkl.dump(pct_detected, f)
    del ref; gc.collect()
else:
    with open(pct_file, 'rb') as f:
        pct_detected = pkl.load(f)

def get_ref_pct(cell_type, gene):
    if cell_type in pct_detected and gene in pct_detected[cell_type].index:
        return round(pct_detected[cell_type][gene] * 100, 1)
    return None

def add_ref_pct(df):
    return df.with_columns(
        pl.struct(['cell_type', 'gene']).map_elements(
            lambda r: get_ref_pct(r['cell_type'], r['gene']),
            return_dtype=pl.Float64
        ).alias('ref_pct_detected'))

def compute_n_cells_expr(adata, cell_type_col):
    # per (cell_type, condition, gene): count cells with counts > 0
    out = []
    ct_arr = adata.obs[cell_type_col].astype(str).to_numpy()
    cond_arr = adata.obs['condition'].astype(str).to_numpy()
    genes = adata.var_names.to_numpy()
    for ct in np.unique(ct_arr):
        ct_mask = ct_arr == ct
        for cond in np.unique(cond_arr[ct_mask]):
            mask = ct_mask & (cond_arr == cond)
            n_expr = np.asarray((adata.X[mask] > 0).sum(axis=0)).ravel()
            out.append(pl.DataFrame({
                'cell_type': ct, 'condition': cond,
                'gene': genes, 'n_cells_expr': n_expr.astype(np.int64)}))
    return pl.concat(out)

protein_coding_genes = pd.read_csv(
    f'{working_dir}/input/MRK_ENSEMBL.csv', header=None)
protein_coding_genes = protein_coding_genes[
    protein_coding_genes[8] == 'protein coding gene'][1].to_list()

adatas = {}
for name, cfg in datasets.items():
    adata = sc.read_h5ad(cfg['path'])
    adata.var_names_make_unique()
    if 'gene_symbol' in adata.var.columns:
        adata.var.index = adata.var['gene_symbol']
        adata.var_names_make_unique()
        adata.var.drop(columns='gene_symbol', inplace=True)
    adata.var.index.name = None
    g = adata.var_names
    adata.var['mt'] = g.str.match(r'^(mt-|MT-)')
    adata.var['ribo'] = g.str.match(r'^(Rps|Rpl)')
    adata.var['protein_coding'] = g.isin(protein_coding_genes)
    drop = cfg.get('drop_samples', [])
    if drop:
        adata = adata[~adata.obs['sample'].isin(drop)].copy()
        print(f'[{name}] dropped samples: {drop}')
    adatas[name] = adata
    print(f'[{name}] {adata.shape[0]:,} cells, '
          f'{adata.obs[cell_type_col].nunique()} cell types, '
          f'{adata.obs["condition"].nunique()} conditions')

#endregion

#region pseudobulk & real DE ##################################################

def make_pseudobulk(adata, name):
    sc_obj = SingleCell(adata).skip_qc()
    if name == 'slidetags':
        sc_obj = sc_obj.filter_var(
            pl.col('protein_coding') &
            pl.col('mt').not_() & pl.col('ribo').not_())
    return sc_obj\
        .pseudobulk('sample', cell_type_col)\
        .qc('condition',
            min_samples=2, min_cells=20,
            max_standard_deviations=None,
            min_nonzero_fraction=0.3, verbose=False)\
        .library_size(allow_float=True)

def n_cells_mask(name, contrast, thresh=MIN_N_CELLS_EXPR):
    # (cell_type, gene) pairs with >=thresh cells expressing in each real
    # condition of the contrast; applied to real and perm DE alike
    treat, base = contrast.split('_vs_')
    path = f'{working_dir}/output/de/n_cells_expr_{name}{de_suffix}.parquet'
    if not os.path.exists(path):
        compute_n_cells_expr(adatas[name], cell_type_col).write_parquet(path)
    long = pl.read_parquet(path)\
        .filter(pl.col('condition').is_in([treat, base]))
    wide = long.pivot(
        on='condition', index=['cell_type', 'gene'],
        values='n_cells_expr')
    return wide\
        .with_columns(
            pl.min_horizontal(pl.col(treat), pl.col(base)).alias('min_n'))\
        .filter(pl.col('min_n') >= thresh)\
        .select(['cell_type', 'gene'])

def load_n_cells_long(name):
    path = f'{working_dir}/output/de/n_cells_expr_{name}{de_suffix}.parquet'
    if not os.path.exists(path):
        compute_n_cells_expr(adatas[name], cell_type_col).write_parquet(path)
    return pl.read_parquet(path).with_columns(pl.lit(name).alias('dataset'))

def add_n_cells_contrast(df, n_long):
    df = df.with_columns(
        pl.col('contrast').str.split('_vs_').list.get(0).alias('treat'),
        pl.col('contrast').str.split('_vs_').list.get(1).alias('base'))
    for side in ('treat', 'base'):
        df = df.join(
            n_long.rename({'condition': side,
                           'n_cells_expr': f'n_cells_{side}'}),
            on=['dataset', 'cell_type', 'gene', side], how='left')
    return df.drop(['treat', 'base'])

def run_de(pb_sub, name, contrast, num_threads):
    treat, ctrl = contrast.split('_vs_')
    # voomByGroup requires >=2 samples per group per cell type
    valid_cts = [
        ct for ct, obs in zip(pb_sub.keys(), pb_sub.iter_obs())
        if (obs.group_by('condition').agg(pl.len().alias('n'))
              .filter(pl.col('n') >= 2).height) >= 2]
    de_obj = pb_sub.DE(
        DESIGN_FORMULAS[name],
        contrasts={contrast: f'condition{treat} - condition{ctrl}'},
        categorical_columns='condition', group='condition',
        cell_types=valid_cts, strict=False,
        return_voom_info=False, allow_float=True,
        verbose=False, num_threads=num_threads)
    mask = n_cells_mask(name, contrast)
    return de_obj.table\
        .rename({'p': 'PValue'})\
        .drop(['coefficient', 'Bonferroni'])\
        .join(mask, on=['cell_type', 'gene'], how='inner')

real_cached = os.path.exists(de_path)
if IS_WORKER and not real_cached:
    raise SystemExit(
        f'[worker] {de_path} missing; run driver first to cache real DE')

# Build pseudobulks only for pairs that need real DE or perms
pairs_to_run = set()
for name, cfg in datasets.items():
    for treat, ctrl in cfg['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        perm_cached = os.path.exists(
            f'{sumrank_cache_dir}/perm_{name}_{contrast}.parquet')
        in_sumrank = name in SUMRANK_CONTRAST_PLATFORMS.get(contrast, [])
        if not real_cached or (in_sumrank and not perm_cached):
            pairs_to_run.add((name, contrast))

all_pbs = {}
for name in {n for n, _ in pairs_to_run}:
    pb = make_pseudobulk(adatas[name], name)
    for treat, ctrl in datasets[name]['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        if (name, contrast) in pairs_to_run:
            all_pbs[(name, contrast)] = \
                pb.filter_obs(pl.col('condition').is_in([treat, ctrl]))
            print(f'[{name}] {contrast}: pseudobulk built')

if not real_cached:
    de_frames = []
    for (name, contrast), pb_sub in all_pbs.items():
        print(f'[{name}] {contrast}: running voomByGroup DE', flush=True)
        t0 = time.time()
        df = run_de(pb_sub, name, contrast, num_threads=SUMRANK_N_CORES)\
            .with_columns(pl.lit(contrast).alias('contrast'),
                          pl.lit(name).alias('dataset'))
        de_frames.append(df)
        n_sig = df.filter(pl.col('FDR') < 0.10).height
        print(f'[{name}] {contrast}: {df.height:,} tests, '
              f'{n_sig} DEGs (FDR<0.10), {time.time()-t0:.0f}s', flush=True)
    de_results = add_ref_pct(pl.concat(de_frames))
    n_long = pl.concat(
        [load_n_cells_long(n) for n in de_results['dataset'].unique()],
        how='diagonal_relaxed')
    de_results = add_n_cells_contrast(de_results, n_long)
    de_results.write_csv(de_path)
    de_results.filter(pl.col('FDR') < 0.10)\
        .write_csv(f'{working_dir}/output/de/de_results_sig{de_suffix}.csv')
else:
    de_results = pl.read_csv(de_path)
    print(f'[de] cached: {de_results.height:,} rows, '
          f'{de_results["dataset"].n_unique()} datasets, '
          f'{de_results["contrast"].n_unique()} contrasts')

#endregion

#region sumrank math ##########################################################

# CDF of sum of n iid Uniform(0,1); null for sum of normalized ranks
def irwin_hall_cdf(x, n):
    x = np.clip(np.atleast_1d(x).astype(float), 0.0, float(n))
    out = np.zeros_like(x)
    for k in range(n + 1):
        diff = x - k
        m = diff > 0
        if m.any():
            out[m] += ((-1) ** k) * comb(n, k) * diff[m] ** n
    return out / factorial(n)

def signed_norm_rank(gene, logfc, pval):
    pv = np.clip(pval.to_numpy().astype(float), 1e-300, 1.0)
    lf = np.nan_to_num(logfc.to_numpy().astype(float))
    score = np.nan_to_num(-np.log10(pv) * np.sign(lf))
    order = np.argsort(-score, kind='stable')
    rank = np.empty(len(score), dtype=np.int64)
    rank[order] = np.arange(1, len(score) + 1)
    return pl.DataFrame({
        'gene': gene,
        'nrank': (rank - 1) / max(len(score) - 1, 1)})

def sumrank_one(de_frame, platforms):
    out = []
    for ct in de_frame['cell_type'].unique().to_list():
        rank_dfs, active = [], []
        for plt in platforms:
            sub = de_frame.filter(
                (pl.col('dataset') == plt) & (pl.col('cell_type') == ct))
            if sub.height == 0:
                continue
            rank_dfs.append(signed_norm_rank(
                sub['gene'], sub['logFC'], sub['PValue']
            ).rename({'nrank': f'nrank_{plt}'}))
            active.append(plt)
        if len(rank_dfs) < 2:
            continue
        merged = rank_dfs[0]
        for rd in rank_dfs[1:]:
            merged = merged.join(rd, on='gene', how='full', coalesce=True)
        arr = merged.select([f'nrank_{p}' for p in active]).to_numpy()
        # d = number of platforms that measured this gene (panels differ)
        d = (~np.isnan(arr)).sum(axis=1)
        s = np.nansum(arr, axis=1)
        # require >=2 platforms so the combined signal is cross-validated
        keep = d >= 2
        if not keep.any():
            continue
        # Irwin-Hall parameter = number of uniforms (d varies per gene)
        nlp_up = np.full(len(d), np.nan)
        nlp_dn = np.full(len(d), np.nan)
        for k in np.unique(d[keep]):
            idx = (d == int(k)) & keep
            cdf = irwin_hall_cdf(s[idx], int(k))
            nlp_up[idx] = -np.log10(np.clip(cdf, 1e-300, 1.0))
            nlp_dn[idx] = -np.log10(np.clip(1 - cdf, 1e-300, 1.0))
        out.append(pl.DataFrame({
            'cell_type': ct, 'gene': merged['gene'],
            'D': d.astype(np.int64), 'sum_stat': s,
            'nlp_up': nlp_up, 'nlp_down': nlp_dn,
        }).filter(pl.col('D') >= 2))
    return pl.concat(out) if out else pl.DataFrame()

def sumrank_one_pw(gsea_frame, platforms):
    renamed = gsea_frame.rename({
        'pathway': 'gene', 'NES': 'logFC', 'pvalue': 'PValue'})
    out = sumrank_one(renamed, platforms)
    return out.rename({'gene': 'pathway'}) if out.height > 0 else out

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

def null_pool_vec(perm_frames, *, item_col='gene', fc_col='logFC',
                  p_col='PValue', chunk_perms=100, label='[null]'):
    if len(perm_frames) < 2:
        return {}
    max_k = min(int(pf['perm'].max()) for pf in perm_frames.values())
    null_by_ct = defaultdict(list)
    score_expr = (((-pl.col(p_col).clip(lower_bound=1e-300).log10()) *
                   pl.col(fc_col).sign())
                  .fill_nan(0.0).fill_null(0.0))
    t0 = time.time()
    for lo in range(1, max_k + 1, chunk_perms):
        hi = min(lo + chunk_perms - 1, max_k)
        parts = [pf.filter(pl.col('perm').is_between(lo, hi))
                 for pf in perm_frames.values()]
        sub = pl.concat(parts, how='diagonal')
        if sub.height == 0:
            continue
        scored = sub.with_columns(score=score_expr).with_columns(
            n_items=pl.len().over(['perm', 'cell_type', 'dataset']),
            rk=pl.col('score').rank('ordinal', descending=True)
                .over(['perm', 'cell_type', 'dataset'])
        ).with_columns(
            nrank=(pl.col('rk') - 1).cast(pl.Float64) /
                  pl.max_horizontal(pl.col('n_items') - 1, 1)
                    .cast(pl.Float64)
        ).select(['perm', 'cell_type', item_col, 'dataset', 'nrank'])
        wide = scored.pivot(
            on='dataset',
            index=['perm', 'cell_type', item_col],
            values='nrank')
        nrank_cols = [c for c in wide.columns
                      if c not in ('perm', 'cell_type', item_col)]
        if not nrank_cols:
            continue
        arr = wide.select(nrank_cols).to_numpy()
        d = (~np.isnan(arr)).sum(axis=1)
        s = np.nansum(arr, axis=1)
        keep = d >= 2
        if not keep.any():
            continue
        nlp_up = np.full(len(d), np.nan)
        nlp_dn = np.full(len(d), np.nan)
        for kk in np.unique(d[keep]):
            idx = (d == int(kk)) & keep
            cdf = irwin_hall_cdf(s[idx], int(kk))
            nlp_up[idx] = -np.log10(np.clip(cdf, 1e-300, 1.0))
            nlp_dn[idx] = -np.log10(np.clip(1 - cdf, 1e-300, 1.0))
        cts = wide['cell_type'].to_numpy()
        valid_up = keep & ~np.isnan(nlp_up)
        valid_dn = keep & ~np.isnan(nlp_dn)
        for ct in np.unique(cts[keep]):
            mask = (cts == ct)
            null_by_ct[ct].append(nlp_up[mask & valid_up])
            null_by_ct[ct].append(nlp_dn[mask & valid_dn])
        elapsed = time.time() - t0
        eta = (max_k - hi) * elapsed / max(hi, 1)
        print(f'{label} perms {lo}:{hi} done '
              f'({elapsed:.0f}s, eta {eta:.0f}s)', flush=True)
    return {ct: np.sort(np.concatenate(arrs)) if arrs else np.array([])
            for ct, arrs in null_by_ct.items()}

def calibrate_emp_p(real, null_by_ct):
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
            emp[msk] = np.maximum((len(nu) - idx) / len(nu), 1.0 / len(nu))
    return emp_up, emp_dn

def run_sumrank_meta(*, output_path, cache_dir, parquet_stage,
                     real_for_contrast, sumrank_fn, item_col,
                     fc_col, p_col, post_process, log_prefix):
    # Compute or load the sumrank meta-analysis for one stage.
    # real_for_contrast(c): polars DF of real-data results for contrast c
    # sumrank_fn: sumrank_one (gene) or sumrank_one_pw (pathway)
    # item_col/fc_col/p_col: column names for the unit + log-fc + p-value
    # post_process(out): final column re-ordering hook
    if os.path.exists(output_path):
        out = pl.read_csv(output_path)
        print(f'{log_prefix} cached: {out.height:,} rows, '
              f'{out["contrast"].n_unique()} contrasts')
        return out
    parts = []
    for contrast, platforms in SUMRANK_CONTRAST_PLATFORMS.items():
        real_c = real_for_contrast(contrast)
        if real_c.height == 0:
            continue
        real = sumrank_fn(real_c, platforms)
        if real.height == 0:
            print(f'{log_prefix} {contrast}: '
                  f'no cell types with >=2 platforms')
            continue
        perm_frames = {}
        for plt in platforms:
            p = f'{cache_dir}/{parquet_stage}_{plt}_{contrast}.parquet'
            if os.path.exists(p):
                perm_frames[plt] = pl.read_parquet(p).with_columns(
                    pl.lit(plt).alias('dataset'))
        if len(perm_frames) < 2:
            print(f'{log_prefix} {contrast}: skip, <2 perm caches')
            continue
        null_final = null_pool_vec(
            perm_frames, item_col=item_col,
            fc_col=fc_col, p_col=p_col,
            label=f'{log_prefix} {contrast} null:')
        emp_up, emp_dn = calibrate_emp_p(real, null_final)
        parts.append(real.with_columns(
            pl.lit(contrast).alias('contrast'),
            pl.Series('emp_p_up', emp_up),
            pl.Series('emp_p_down', emp_dn),
            pl.Series('emp_fdr_up', bh_fdr(emp_up)),
            pl.Series('emp_fdr_down', bh_fdr(emp_dn))))
    out = pl.concat(parts) if parts else pl.DataFrame()
    if out.height > 0:
        out = post_process(out)
    out.write_csv(output_path)
    return out

def print_sumrank_summary(out, *, item_col, log_prefix):
    for contrast in SUMRANK_CONTRAST_PLATFORMS:
        if out.height == 0:
            return
        sub = out.filter(pl.col('contrast') == contrast)
        if sub.height == 0:
            continue
        n_up = sub.filter(pl.col('emp_fdr_up') < 0.10).height
        n_dn = sub.filter(pl.col('emp_fdr_down') < 0.10).height
        print(f'{log_prefix} {contrast}: '
              f'{sub["cell_type"].n_unique()} cell types, '
              f'{sub[item_col].n_unique():,} {item_col}s, '
              f'{n_up} up / {n_dn} down (emp_fdr<0.10)')

#endregion

#region DE perms — R code #####################################################

r('''
suppressPackageStartupMessages({
    library(limma); library(dplyr); library(tibble)
    library(purrr); library(parallel)
})
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
           MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1")
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(1)
    RhpcBLASctl::omp_set_num_threads(1)
}
''')
r(_voomByGroup_source_code)
r('''
run_voom_perms <- function(pseudobulks, treat, base, design_formula,
                            n_perm, n_cores, start_perm = 1L,
                            seed_base = 12345L) {
    sample_cond <- unique(do.call(rbind, lapply(pseudobulks, function(el) {
        unique(data.frame(sample = as.character(el$obs$sample),
                          condition = as.character(el$obs$condition),
                          stringsAsFactors = FALSE))
    })))
    perm_maps <- vector("list", n_perm)
    for (k in seq_len(n_perm)) {
        gk <- start_perm + k - 1L
        set.seed(seed_base + gk)
        m <- sample(sample_cond$condition)
        names(m) <- sample_cond$sample
        perm_maps[[k]] <- m
    }
    ct_names <- names(pseudobulks)
    tasks <- expand.grid(k = seq_len(n_perm),
                         ct_idx = seq_along(ct_names),
                         KEEP.OUT.ATTRS = FALSE)
    treat_col <- paste0("condition", treat)
    base_col  <- paste0("condition", base)
    contrast_expr <- paste0(treat_col, " - ", base_col)
    worker <- function(i) {
        k <- tasks$k[i]; ct_idx <- tasks$ct_idx[i]
        ct <- ct_names[ct_idx]; el <- pseudobulks[[ct_idx]]
        gk <- start_perm + k - 1L
        tryCatch({
            targets <- el$obs
            targets$condition <- factor(
                perm_maps[[k]][as.character(targets$sample)],
                levels = c(base, treat))
            if (length(unique(targets$condition)) < 2) return(NULL)
            if (min(table(targets$condition)) < 2) return(NULL)
            targets$library_size <- colSums(el$counts)
            design <- model.matrix(as.formula(design_formula),
                                    data = targets)
            colnames(design) <- make.names(colnames(design))
            if (qr(design)$rank < ncol(design)) return(NULL)
            vbg <- voomByGroup(el$counts, as.character(targets$condition),
                                design, targets$library_size,
                                save.plot = FALSE, print = FALSE)
            fit <- lmFit(vbg, design)
            cmat <- makeContrasts(contrasts = contrast_expr, levels = design)
            fit2 <- eBayes(contrasts.fit(fit, cmat),
                            trend = FALSE, robust = FALSE)
            tt <- topTable(fit2, coef = 1, number = Inf, sort.by = "none")
            data.frame(gene = rownames(tt), logFC = tt$logFC,
                        PValue = tt$P.Value, cell_type = ct,
                        perm = gk, stringsAsFactors = FALSE)
        }, error = function(e) NULL)
    }
    results <- if (n_cores > 1)
        mclapply(seq_len(nrow(tasks)), worker, mc.cores = n_cores)
    else
        lapply(seq_len(nrow(tasks)), worker)
    bind_rows(results)
}
''')

def populate_pb_r(pb_sub, r_var='pb_cur'):
    r(f'{r_var} <- list()')
    for ct, (X, obs, var) in pb_sub.items():
        gene_names = var['_index'] if '_index' in var.columns \
            else pl.Series(var[:, 0])
        to_r(obs, 'obs_tmp')
        to_r(X, 'X_tmp', colnames=gene_names)
        to_r(ct, 'ct_tmp')
        r(f'''
        {r_var}[[ct_tmp]] <- list(
            counts = t(X_tmp),
            obs = obs_tmp)
        ''')
    r('rm(obs_tmp, X_tmp, ct_tmp); invisible(gc())')

#endregion

#region DE perms — submit & worker ############################################

# Driver: merge any ready pairs, then submit missing batches
if not IS_WORKER:
    for pair in _sumrank_pairs():
        _merge_chunks_if_ready(
            pair, 'perm', sumrank_cache_dir,
            SUMRANK_CHUNK_SIZE, SUMRANK_N_PERM,
            unique_cols=['perm', 'cell_type', 'gene'])
    active = _active_slurm_jobs()
    submitted, pending = _submit_missing_batches(
        parquet_stage='perm', job_stage='de', perm_job='de',
        pairs=[p for p in _sumrank_pairs() if p in all_pbs],
        cache_dir=sumrank_cache_dir,
        batch_perms=SUMRANK_PERM_BATCH,
        chunk_size=SUMRANK_CHUNK_SIZE,
        active_jobs=active)
    if submitted:
        print(f'[sumrank] submitted {len(submitted)} DE perm batches:')
        for tag in submitted:
            print(f'  {tag}')
    elif pending:
        print(f'[sumrank] {len(pending)} DE perm batches in flight')
    else:
        print('[sumrank] all DE perms cached')

def run_de_worker(name, contrast, pb_sub, perm_start, perm_end):
    # Compute DE perm chunks for one (platform, contrast, perm-range).
    if name not in SUMRANK_CONTRAST_PLATFORMS.get(contrast, []):
        return
    final_path = f'{sumrank_cache_dir}/perm_{name}_{contrast}.parquet'
    if os.path.exists(final_path):
        print(f'[sumrank] perm cached: {name} {contrast}', flush=True)
        return
    chunk_glob = (
        f'{sumrank_cache_dir}/perm_{name}_{contrast}_chunk_*.parquet')
    existing = chunk_map(chunk_glob)
    starts = list(range(perm_start, perm_end, SUMRANK_CHUNK_SIZE))
    range_label = f'{perm_start}:{perm_end}'
    print(f'[sumrank] {name} {contrast} [{range_label}]: '
          f'{len(starts)} chunks of {SUMRANK_CHUNK_SIZE} '
          f'({sum(s in existing for s in starts)} cached), '
          f'n_cores={SUMRANK_N_CORES}', flush=True)
    need_any = any(s not in existing for s in starts)
    if need_any:
        t_pop = time.time()
        populate_pb_r(pb_sub, 'pb_cur')
        print(f'[sumrank] {name} {contrast}: R pseudobulks populated in '
              f'{time.time()-t_pop:.0f}s', flush=True)
        treat, base = contrast.split('_vs_')
        to_r(treat, 'treat'); to_r(base, 'base')
        to_r(DESIGN_FORMULAS[name], 'design_formula')
        to_r(SUMRANK_N_CORES, 'n_cores')
        to_r(PERM_SEED_BASE, 'seed_base')
    perm_mask = n_cells_mask(name, contrast)
    t_pair = time.time()
    for i, start in enumerate(starts, 1):
        if start in existing:
            print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
                  f'(cached)', flush=True)
            continue
        n_this = min(SUMRANK_CHUNK_SIZE, SUMRANK_N_PERM - start)
        t0 = time.time()
        to_r(n_this, 'n_perm')
        to_r(start + 1, 'start_perm')
        r('chunk_df <- run_voom_perms(pb_cur, treat, base, '
          'design_formula, n_perm, n_cores, start_perm, seed_base)')
        chunk_df = to_py('chunk_df')
        dt = time.time() - t0
        if chunk_df is None or chunk_df.height == 0:
            print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
                  f'returned 0 rows in {dt:.0f}s', flush=True)
            continue
        chunk_df = chunk_df\
            .with_columns(pl.col('perm').cast(pl.Int64))\
            .join(perm_mask, on=['cell_type', 'gene'], how='inner')
        chunk_path = (f'{sumrank_cache_dir}/'
                      f'perm_{name}_{contrast}_chunk_{start}.parquet')
        chunk_df.write_parquet(chunk_path)
        eta = dt * (len(starts) - i) / 60
        print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
              f'done ({chunk_df["perm"].n_unique()} perms, '
              f'{chunk_df.height:,} rows, {dt:.0f}s; eta {eta:.1f} min)',
              flush=True)
    print(f'[sumrank] {name} {contrast} [{range_label}]: chunks written '
          f'({(time.time() - t_pair) / 60:.1f} min)', flush=True)
    if need_any:
        r('rm(pb_cur); invisible(gc())')

if IS_WORKER and WORKER_STAGE == 'de':
    pb_sub = all_pbs.get((WORKER_PLATFORM, WORKER_CONTRAST))
    if pb_sub is None:
        raise SystemExit(
            f'[worker] no pseudobulk for {WORKER_PLATFORM} '
            f'{WORKER_CONTRAST}')
    run_de_worker(WORKER_PLATFORM, WORKER_CONTRAST, pb_sub,
                  WORKER_PERM_START, WORKER_PERM_END)
    print(f'[worker] de perm job done for {WORKER_PLATFORM} '
          f'{WORKER_CONTRAST}', flush=True)
    sys.exit(0)

# Driver: gate on all DE perm parquets being present
if not IS_WORKER:
    _gate_on_missing('perm', sumrank_cache_dir, 'sumrank')

#endregion

#region gene-level sumrank meta ###############################################

def _gene_post_process(out):
    # sum n_cells across datasets per (contrast, cell_type, gene)
    n_cells_agg = de_results\
        .group_by(['contrast', 'cell_type', 'gene'])\
        .agg(pl.col('n_cells_treat').sum(),
             pl.col('n_cells_base').sum())
    return add_ref_pct(out)\
        .join(n_cells_agg, on=['contrast', 'cell_type', 'gene'],
              how='left')\
        .select([
            'contrast', 'cell_type', 'gene', 'D', 'sum_stat',
            'nlp_up', 'nlp_down', 'emp_p_up', 'emp_p_down',
            'emp_fdr_up', 'emp_fdr_down', 'ref_pct_detected',
            'n_cells_treat', 'n_cells_base'])

# Driver-only: workers exit before reaching this via the DE worker
# dispatch above; without this guard a GSEA worker (which falls through)
# would race the driver writing sumrank_results.csv.
if not IS_WORKER:
    sumrank_out = run_sumrank_meta(
        output_path=sumrank_path,
        cache_dir=sumrank_cache_dir,
        parquet_stage='perm',
        real_for_contrast=lambda c: de_results.filter(
            pl.col('contrast') == c),
        sumrank_fn=sumrank_one,
        item_col='gene', fc_col='logFC', p_col='PValue',
        post_process=_gene_post_process,
        log_prefix='[sumrank]')
    print_sumrank_summary(sumrank_out, item_col='gene',
                          log_prefix='[sumrank]')

#endregion

#region GSEA — R code #########################################################

r(f'''
suppressPackageStartupMessages({{
    library(fgsea); library(msigdbr); library(dplyr); library(tibble)
    library(purrr); library(parallel)
}})
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
           MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1")
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {{
    RhpcBLASctl::blas_set_num_threads(1)
    RhpcBLASctl::omp_set_num_threads(1)
}}

cache_file <- "{working_dir}/input/m_df_themed_v2.rds"
if (!file.exists(cache_file)) {{
    m_df <- msigdbr(species = "Mus musculus", category = "C5",
                    subcategory = "GO:BP")
    # Themes ordered most-specific first; first match wins.
    theme_keywords <- list(
        'Maternal_Reproduction' = c(
            'MATERNAL','PREGNAN','GESTATION','PARTURITION','LACTATION',
            'MAMMARY','PLACENTA','_MILK_','OVULATION','OVARIAN',
            'ESTROUS','ESTRUS','FEMALE_GONAD','FEMALE_SEX',
            'MENSTRUAL','UTERINE','OOGENESIS','EMBRYO_IMPLANTATION',
            'REPRODUCTIVE_BEHAVIOR','REPRODUCTIVE_STRUCTURE',
            'MULTICELLULAR_ORGANISMAL_REPRODUCTIVE'),
        'Glial_Myelination' = c(
            'ASTROCYT','GLIAL_','GLIOGENESIS','OLIGODENDRO',
            'MYELIN','SCHWANN','MICROGLIA','MICROGLIAL',
            'ENSHEATHMENT','NEUROINFLAMM','REMYELINATION'),
        'Plasticity' = c(
            'NEUROGENESIS','DENDRITIC_SPINE',
            'AXON_GUIDANCE','AXONOGENESIS',
            'SYNAPSE_ORGANIZATION','SYNAPSE_ASSEMBLY',
            'SYNAPSE_PRUNING','SYNAPTIC_PLASTICITY',
            'LONG_TERM_SYNAPTIC','LONG_TERM_POTENTIATION',
            'LONG_TERM_DEPRESSION',
            'NEURON_PROJECTION_DEVELOPMENT',
            'NEURON_PROJECTION_MORPHOGENESIS',
            'NEURON_DIFFERENTIATION','NEURON_MIGRATION',
            'NEURON_FATE','NEURAL_PRECURSOR','NEURAL_PROGENITOR',
            'SEMAPHORIN','SLIT','EPHRIN',
            'CELL_MORPHOGENESIS_INVOLVED_IN_NEURON'),
        'Hormonal' = c(
            'HORMONE','STEROID','ESTROGEN','PROGESTERONE',
            'ANDROGEN','PROLACTIN','CORTICOSTERON',
            'CORTICOTROPIN','GONADOTROPIN','VASOPRESSIN',
            'NEUROPEPTIDE','LUTEIN','FOLLICLE_STIMULAT',
            'GLUCOCORTICOID','MINERALOCORTICOID',
            'ENDOCRINE_','INSULIN_SECR','LEPTIN','GHRELIN'),
        'Growth_Factors' = c(
            'NEUROTROPH','NERVE_GROWTH_FACTOR',
            'INSULIN_LIKE_GROWTH_FACTOR',
            'FIBROBLAST_GROWTH_FACTOR',
            'EPIDERMAL_GROWTH_FACTOR',
            'VASCULAR_ENDOTHELIAL_GROWTH_FACTOR',
            'HEPATOCYTE_GROWTH_FACTOR',
            'PLATELET_DERIVED_GROWTH_FACTOR',
            'TRANSFORMING_GROWTH_FACTOR',
            'GLIAL_CELL_LINE_DERIVED_NEUROTROPH',
            'BONE_MORPHOGENETIC_PROTEIN',
            'GROWTH_FACTOR_ACTIVITY',
            'GROWTH_FACTOR_PRODUCTION',
            'RESPONSE_TO_GROWTH_FACTOR'),
        'Immune' = c(
            'IMMUNE','IMMUNO','INFLAMMAT','CYTOKINE','INTERFERON',
            'INTERLEUKIN','INNATE','COMPLEMENT_ACTIVATION','ANTIGEN',
            'T_CELL','B_CELL','LYMPHOCYTE','LEUKOCYTE',
            'MACROPHAGE','NK_CELL','TOLL_LIKE','CHEMOKINE'),
        'Behavior_Cognition' = c(
            'BEHAVIOR','LEARNING','MEMORY','COGNITION','LOCOMOT',
            'FEEDING_BEHAVIOR','SUCKLING','VOCALIZATION',
            'CIRCADIAN','RHYTHMIC','SLEEP','AROUSAL'),
        'Ion_Transport' = c(
            'CALCIUM','POTASSIUM','SODIUM','CHLORIDE','ZINC',
            'METAL_ION_TRANSPORT','ION_TRANSPORT','ION_HOMEOSTASIS',
            'CATION_CHANNEL','ANION_CHANNEL','MEMBRANE_POTENTIAL',
            'ACTION_POTENTIAL','CHANNEL_ACTIVITY','_ION_CHANNEL'),
        'Vascular' = c(
            'VASCUL','ANGIOGEN','ENDOTHELIAL',
            'BLOOD_BRAIN_BARRIER','VASOCONSTR','VASODILA',
            'ARTERY','ARTERIOGEN','PERICYTE',
            'BLOOD_CIRCULATION','MICROVASCUL'),
        'Protein_Dynamics' = c(
            'TRANSLATION','RIBOSOM','PROTEASOME','UBIQUITIN',
            'AUTOPHAG','PROTEIN_FOLD','UNFOLDED_PROTEIN',
            'CHAPERONE','ER_STRESS',
            'ENDOPLASMIC_RETICULUM_STRESS',
            'TOPOLOGICALLY_INCORRECT_PROTEIN'),
        'Stress_Apoptosis' = c(
            'APOPTO','NECROP','PYROPTO','FERROPTO','CELL_DEATH',
            'NEURON_DEATH','PROGRAMMED_CELL_DEATH',
            'INTRINSIC_APOPTOTIC','CASPASE',
            'OXIDATIVE_STRESS','REACTIVE_OXYGEN',
            'RESPONSE_TO_STRESS','CELLULAR_RESPONSE_TO_STRESS',
            'RESPONSE_TO_OXIDATIVE','DNA_DAMAGE_RESPONSE',
            'RESPONSE_TO_HYPOXIA','HEAT_SHOCK',
            'RESPONSE_TO_HEAT','RESPONSE_TO_COLD'),
        'Metabolic' = c(
            'METABOLI','LIPID','CHOLESTEROL','GLUCOSE','GLYCOLY',
            'ATP_','CELLULAR_RESPIRATION','AEROBIC_RESPIRATION',
            'ELECTRON_TRANSPORT','OXIDATIVE_PHOSPHOR',
            'MITOCHONDR','MITOPHAG','TRICARBOX','NADH_',
            'FATTY_ACID','BETA_OXID','AMINO_ACID_METAB',
            'NUCLEOTIDE_METAB','GLYCOGEN','KETONE_BODY'),
        'Neuronal' = c(
            'NEURON_','NEUROTRANS','SYNAP','AXON_','DENDRIT',
            'GLUTAMATE','GABA','CHOLINERGIC','DOPAMINERGIC',
            'SEROTONERGIC','SEROTONIN','NEUROMODUL',
            'NEUROSECRETORY','NEUROMUSCUL'),
        'Structural' = c(
            'CELL_ADHESION','CELL_CELL_ADHESION','CELL_JUNCTION',
            'EXTRACELLULAR_MATRIX','BASEMENT_MEMBRANE',
            'GAP_JUNCTION','TIGHT_JUNCTION','INTEGRIN',
            'CYTOSKELET','ACTIN_FILAMENT','INTERMEDIATE_FILAMENT',
            'MICROTUBULE','COLLAGEN')
    )
    theme_patterns <- vapply(
        theme_keywords,
        function(kws) paste(kws, collapse = "|"),
        character(1))
    assign_theme <- function(names_vec) {{
        res <- rep(NA_character_, length(names_vec))
        for (tn in names(theme_patterns)) {{
            todo <- is.na(res)
            if (!any(todo)) break
            hit <- grepl(theme_patterns[[tn]], names_vec[todo],
                         ignore.case = TRUE, perl = TRUE)
            res[which(todo)[hit]] <- tn
        }}
        res
    }}
    m_df_themed <- m_df %>%
        mutate(theme = assign_theme(gs_name)) %>%
        filter(!is.na(theme))
    cat("[theme_v2] pathway counts per theme:\\n")
    print(m_df_themed %>% distinct(gs_name, theme) %>%
          count(theme, sort = TRUE))
    cat(sprintf(
        "[theme_v2] kept %d / %d GO:BP pathways (%.1f%%)\\n",
        length(unique(m_df_themed$gs_name)),
        length(unique(m_df$gs_name)),
        100 * length(unique(m_df_themed$gs_name)) /
              length(unique(m_df$gs_name))))
    saveRDS(m_df_themed, cache_file)
}} else {{
    m_df_themed <- readRDS(cache_file)
}}
filtered_pathways_sr <- split(m_df_themed$gene_symbol, m_df_themed$gs_name)

fgsea_ranked_groups <- function(ranked_df, keys, pathways, n_cores,
                                minSize = 15, nperm_simple = NULL,
                                with_leading_edge = FALSE) {{
    rl <- rle(keys)
    ends <- cumsum(rl$lengths)
    starts <- ends - rl$lengths + 1L
    n_grp <- length(rl$values)
    gene_vec <- ranked_df$gene
    rank_vec <- ranked_df$rank
    worker <- function(i) {{
        idx <- starts[i]:ends[i]
        if (length(idx) < 100) return(NULL)
        ranks <- setNames(rank_vec[idx], gene_vec[idx])
        res <- tryCatch(
            if (is.null(nperm_simple)) {{
                fgsea(pathways = pathways, stats = ranks,
                      minSize = minSize)
            }} else {{
                fgseaSimple(pathways = pathways, stats = ranks,
                            minSize = minSize, nperm = nperm_simple)
            }},
            error = function(e) NULL)
        if (is.null(res) || nrow(res) == 0) {{
            rm(ranks); gc(); return(NULL)
        }}
        out <- data.frame(pathway = res$pathway, NES = res$NES,
                          pvalue = res$pval,
                          key = rl$values[i],
                          stringsAsFactors = FALSE)
        if (with_leading_edge && !is.null(res$leadingEdge)) {{
            out$leading_edge <- vapply(
                res$leadingEdge,
                function(g) paste(g, collapse = ","),
                character(1))
        }}
        rm(ranks, res); gc()
        out
    }}
    results <- if (n_cores > 1)
        mclapply(seq_len(n_grp), worker, mc.cores = n_cores)
    else
        lapply(seq_len(n_grp), worker)
    valid <- vapply(results,
                    function(x) is.null(x) || is.data.frame(x),
                    logical(1))
    if (!all(valid)) {{
        warning(sprintf(
            "fgsea_ranked_groups: %d/%d workers crashed",
            sum(!valid), length(results)))
    }}
    bind_rows(results[valid])
}}

fgsea_perms <- function(perm_df, pathways, n_cores, minSize = 15,
                        nperm_simple = 100) {{
    keys <- paste(perm_df$perm, perm_df$cell_type, sep = "___")
    out <- fgsea_ranked_groups(perm_df, keys, pathways, n_cores, minSize,
                               nperm_simple = nperm_simple)
    if (nrow(out) == 0) return(out)
    parts <- do.call(rbind, strsplit(out$key, "___", fixed = TRUE))
    out$perm <- as.integer(parts[, 1L])
    out$cell_type <- parts[, 2L]
    out$key <- NULL
    out
}}

fgsea_real <- function(ranked_df, pathways, n_cores, minSize = 15) {{
    out <- fgsea_ranked_groups(ranked_df, ranked_df$cell_type,
                               pathways, n_cores, minSize,
                               with_leading_edge = TRUE)
    if (nrow(out) == 0) return(out)
    out$cell_type <- out$key
    out$key <- NULL
    out
}}
''')

def prerank_for_fgsea(df, group_cols):
    return df\
        .with_columns(
            ((-pl.col('PValue').log10()) * pl.col('logFC').sign())
            .alias('rank'))\
        .filter(pl.col('rank').is_finite())\
        .select(group_cols + ['gene', 'rank'])\
        .sort(group_cols + ['rank'],
              descending=[False] * len(group_cols) + [True])

#endregion

#region GSEA real fgsea #######################################################

real_gsea_cache = f'{SUMRANK_GSEA_CACHE_DIR}/real_gsea.parquet'
if IS_WORKER and WORKER_STAGE == 'gsea' and \
        not os.path.exists(real_gsea_cache):
    raise SystemExit(
        f'[worker] {real_gsea_cache} missing; run driver first')
if os.path.exists(real_gsea_cache):
    real_gsea = pl.read_parquet(real_gsea_cache)
    print(f'[sumrank_gsea] real fgsea cached: {real_gsea.height:,} rows')
else:
    pairs = de_results.select(['dataset', 'contrast'])\
        .unique().sort(['dataset', 'contrast']).rows()
    print(f'[sumrank_gsea] running real fgsea on {len(pairs)} pairs',
          flush=True)
    real_parts = []
    t_real = time.time()
    to_r(SUMRANK_GSEA_N_CORES, 'n_cores')
    for j, (ds, ctr) in enumerate(pairs, 1):
        sub = de_results.filter(
            (pl.col('dataset') == ds) & (pl.col('contrast') == ctr))
        ranked = prerank_for_fgsea(sub, ['cell_type'])
        n_cts = ranked['cell_type'].n_unique() if ranked.height > 0 else 0
        t0 = time.time()
        to_r(ranked, 'ranked_df')
        r('real_gsea_sub <- fgsea_real(ranked_df, '
          'filtered_pathways_sr, n_cores)')
        part = to_py('real_gsea_sub')
        dt = time.time() - t0
        if part is not None and part.height > 0:
            part = part.with_columns(
                pl.lit(ds).alias('dataset'),
                pl.lit(ctr).alias('contrast'))
            real_parts.append(part)
        n_rows = 0 if part is None else part.height
        print(f'[sumrank_gsea] real fgsea: {j}/{len(pairs)} {ds} {ctr} '
              f'({n_cts} cts, {n_rows:,} rows, {dt:.0f}s)', flush=True)
    real_gsea = pl.concat(real_parts) if real_parts else pl.DataFrame()
    real_gsea.write_parquet(real_gsea_cache)
    print(f'[sumrank_gsea] real fgsea: {real_gsea.height:,} rows cached '
          f'({(time.time() - t_real) / 60:.1f} min total)', flush=True)

#endregion

#region GSEA perms — submit & worker ##########################################

# Driver: merge any ready pairs, then submit missing batches
if not IS_WORKER:
    for n, c in _sumrank_pairs():
        if not os.path.exists(
                f'{sumrank_cache_dir}/perm_{n}_{c}.parquet'):
            continue
        _merge_chunks_if_ready(
            (n, c), 'fgsea', SUMRANK_GSEA_CACHE_DIR,
            SUMRANK_GSEA_CHUNK, SUMRANK_N_PERM,
            unique_cols=['perm', 'cell_type', 'pathway'])
    active = _active_slurm_jobs()
    # Only submit GSEA batches for pairs whose gene perm parquet exists
    gsea_ready = [(n, c) for n, c in _sumrank_pairs()
                  if os.path.exists(
                      f'{sumrank_cache_dir}/perm_{n}_{c}.parquet')]
    submitted, pending = _submit_missing_batches(
        parquet_stage='fgsea', job_stage='gs', perm_job='gsea',
        pairs=gsea_ready, cache_dir=SUMRANK_GSEA_CACHE_DIR,
        batch_perms=SUMRANK_GSEA_PERM_BATCH,
        chunk_size=SUMRANK_GSEA_CHUNK,
        active_jobs=active)
    if submitted:
        print(f'[sumrank_gsea] submitted {len(submitted)} GSEA perm '
              f'batches:')
        for tag in submitted:
            print(f'  {tag}')
    elif pending:
        print(f'[sumrank_gsea] {len(pending)} GSEA perm batches in flight')
    else:
        print('[sumrank_gsea] all GSEA perms cached')

def run_gsea_worker(name, contrast, perm_start, perm_end):
    # Compute GSEA perm chunks for one (platform, contrast, perm-range).
    if name not in datasets:
        return
    ds_ctrs = {f'{t}_vs_{c}' for t, c in datasets[name]['contrasts']}
    if contrast not in ds_ctrs:
        return
    gene_perm_path = (
        f'{sumrank_cache_dir}/perm_{name}_{contrast}.parquet')
    if not os.path.exists(gene_perm_path):
        print(f'[sumrank_gsea] skip {name} {contrast}: '
              f'gene perms not cached')
        return
    final_path = (
        f'{SUMRANK_GSEA_CACHE_DIR}/fgsea_{name}_{contrast}.parquet')
    if os.path.exists(final_path):
        print(f'[sumrank_gsea] fgsea perm cached: {name} {contrast}')
        return
    chunk_glob = (f'{SUMRANK_GSEA_CACHE_DIR}/'
                  f'fgsea_{name}_{contrast}_chunk_*.parquet')
    existing = chunk_map(chunk_glob)
    n_perms = int(pl.scan_parquet(gene_perm_path)
                  .select(pl.col('perm').max()).collect().item())
    lo, hi = perm_start, min(perm_end, n_perms)
    starts = list(range(lo, hi, SUMRANK_GSEA_CHUNK))
    range_label = f'{lo}:{hi}'
    print(f'[sumrank_gsea] {name} {contrast} [{range_label}]: '
          f'{len(starts)} chunks of {SUMRANK_GSEA_CHUNK} '
          f'({sum(s in existing for s in starts)} cached), '
          f'n_cores={SUMRANK_GSEA_N_CORES}', flush=True)
    to_r(SUMRANK_GSEA_N_CORES, 'n_cores')
    to_r(SUMRANK_GSEA_NPERM_SIMPLE, 'nperm_simple')
    t_pair = time.time()
    for i, start in enumerate(starts, 1):
        if start in existing:
            print(f'[sumrank_gsea] {name} {contrast}: chunk '
                  f'{i}/{len(starts)} (cached)', flush=True)
            continue
        end = min(start + SUMRANK_GSEA_CHUNK, n_perms)
        ks = list(range(start + 1, end + 1))
        sub = pl.scan_parquet(gene_perm_path)\
            .filter(pl.col('perm').is_in(ks)).collect()
        if sub.height == 0:
            continue
        ranked = prerank_for_fgsea(
            sub.with_columns(pl.col('perm').cast(pl.Int64)),
            ['perm', 'cell_type'])
        t0 = time.time()
        to_r(ranked, 'ranked_df')
        try:
            r('chunk_df <- fgsea_perms(ranked_df, '
              'filtered_pathways_sr, n_cores, '
              'nperm_simple = nperm_simple)')
            chunk_df = to_py('chunk_df')
        except Exception as e:
            dt = time.time() - t0
            print(f'[sumrank_gsea] {name} {contrast}: chunk '
                  f'{i}/{len(starts)} FAILED in {dt:.0f}s ({e})',
                  flush=True)
            r('rm(ranked_df); gc()')
            continue
        dt = time.time() - t0
        if chunk_df is None or chunk_df.height == 0:
            print(f'[sumrank_gsea] {name} {contrast}: chunk '
                  f'{i}/{len(starts)} returned 0 rows in {dt:.0f}s',
                  flush=True)
            continue
        chunk_path = (
            f'{SUMRANK_GSEA_CACHE_DIR}/'
            f'fgsea_{name}_{contrast}_chunk_{start}.parquet')
        chunk_df.write_parquet(chunk_path)
        eta = dt * (len(starts) - i) / 60
        print(f'[sumrank_gsea] {name} {contrast}: chunk '
              f'{i}/{len(starts)} done '
              f'({chunk_df["perm"].n_unique()} perms, '
              f'{chunk_df.height:,} rows, {dt:.0f}s; '
              f'eta {eta:.1f} min)', flush=True)
    print(f'[sumrank_gsea] {name} {contrast} [{range_label}]: '
          f'chunks written ({(time.time() - t_pair) / 60:.1f} min)',
          flush=True)

if IS_WORKER and WORKER_STAGE == 'gsea':
    run_gsea_worker(WORKER_PLATFORM, WORKER_CONTRAST,
                    WORKER_PERM_START, WORKER_PERM_END)
    print(f'[worker] gsea perm job done for {WORKER_PLATFORM} '
          f'{WORKER_CONTRAST}', flush=True)
    sys.exit(0)

# Driver: gate on all GSEA perm parquets being present
if not IS_WORKER:
    _gate_on_missing('fgsea', SUMRANK_GSEA_CACHE_DIR, 'sumrank_gsea')

#endregion

#region pathway-level sumrank meta ############################################

def _gsea_post_process(out):
    le = real_gsea.select([
        'contrast', 'cell_type', 'pathway', 'dataset', 'leading_edge'])\
        .pivot(values='leading_edge', on='dataset',
               index=['contrast', 'cell_type', 'pathway'])
    le_cols = [c for c in le.columns
               if c not in ('contrast', 'cell_type', 'pathway')]
    le = le.rename({c: f'leading_edge_{c}' for c in le_cols})
    out = out.join(le, on=['contrast', 'cell_type', 'pathway'], how='left')
    return out.select([
        'contrast', 'cell_type', 'pathway', 'D', 'sum_stat',
        'nlp_up', 'nlp_down', 'emp_p_up', 'emp_p_down',
        'emp_fdr_up', 'emp_fdr_down',
        *sorted(f'leading_edge_{c}' for c in le_cols)])

# null_pool_vec produces NaN for platform-missing (cell_type, pathway)
# pairs; D>=2 filter handles missingness exactly like the gene-level path
# (no NES=0 backfill, which would have biased the null toward the center
# for pathways occasionally filtered by fgseaSimple's minSize).
sumrank_gsea_out = run_sumrank_meta(
    output_path=sumrank_gsea_path,
    cache_dir=SUMRANK_GSEA_CACHE_DIR,
    parquet_stage='fgsea',
    real_for_contrast=lambda c: real_gsea.filter(
        pl.col('contrast') == c),
    sumrank_fn=sumrank_one_pw,
    item_col='pathway', fc_col='NES', p_col='pvalue',
    post_process=_gsea_post_process,
    log_prefix='[sumrank_gsea]')
print_sumrank_summary(sumrank_gsea_out, item_col='pathway',
                      log_prefix='[sumrank_gsea]')

#endregion
