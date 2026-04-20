#region imports and setup ######################################################

import os
import gc
import re
import glob
import time
import pickle as pkl
import warnings
from math import comb, factorial
from collections import defaultdict
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r, to_py

warnings.filterwarnings('ignore')

from single_cell import SingleCell, Pseudobulk

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
de_suffix = f'_{cell_type_col}' if cell_type_col != 'subclass' else ''

datasets = {
    'slidetags': {
        'path': f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
    },
    'merfish': {
        'path': f'{working_dir}/output/merfish/03_adata_query_merfish.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
    },
}

pct_file = f'{working_dir}/output/ref_pct_detected.pkl'
if not os.path.exists(pct_file):
    ref = sc.read_h5ad('single-cell/ABC/zeng_combined_10Xv3.h5ad')
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
          f'{adata.obs[cell_type_col].nunique()} subclasses, '
          f'{adata.obs["condition"].nunique()} conditions')

#endregion

#region run de #################################################################

def make_pseudobulk(adata, name):
    sc_obj = SingleCell(adata).skip_qc()
    if name == 'slidetags':
        sc_obj = sc_obj.filter_var(
            pl.col('protein_coding') &
            pl.col('mt').not_() & pl.col('ribo').not_())
    return sc_obj\
        .pseudobulk('sample', cell_type_col)\
        .qc('condition',
            min_samples=2,
            min_cells=20,
            max_standard_deviations=None,
            min_nonzero_fraction=0.3,
            verbose=False)\
        .library_size(allow_float=True)

MIN_N_CELLS_EXPR = 10

def n_cells_mask(name, contrast, thresh=MIN_N_CELLS_EXPR):
    # (cell_type, gene) pairs with >=thresh cells expressing in each real
    # condition of the contrast. Used to suppress sparse-detection hits
    # consistently across real and permuted DE.
    treat, base = contrast.split('_vs_')
    path = f'{working_dir}/output/n_cells_expr_{name}.parquet'
    if not os.path.exists(path):
        compute_n_cells_expr(adatas[name], cell_type_col)\
        .write_parquet(path)
    long = pl.read_parquet(path)\
        .filter(pl.col('condition').is_in([treat, base]))
    wide = long.pivot(
        on='condition', index=['cell_type', 'gene'],
        values='n_cells_expr')
    return wide\
        .with_columns(
            pl.min_horizontal(pl.col(treat), pl.col(base))
            .alias('min_n'))\
        .filter(pl.col('min_n') >= thresh)\
        .select(['cell_type', 'gene'])

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
        categorical_columns='condition',
        group='condition',
        cell_types=valid_cts,
        strict=False,
        return_voom_info=False,
        allow_float=True,
        verbose=False,
        num_threads=num_threads)
    mask = n_cells_mask(name, contrast)
    return de_obj.table\
        .rename({'p': 'PValue'})\
        .drop(['coefficient', 'Bonferroni'])\
        .join(mask, on=['cell_type', 'gene'], how='inner')

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
SUMRANK_N_CORES = os.cpu_count()
SUMRANK_CHUNK_SIZE = 20
PERM_SEED_BASE = 12345

sumrank_cache_dir = f'{working_dir}/output/sumrank_cache'
os.makedirs(sumrank_cache_dir, exist_ok=True)
os.makedirs(f'{working_dir}/output', exist_ok=True)
de_path = f'{working_dir}/output/de_results{de_suffix}.csv'
real_cached = os.path.exists(de_path)

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

def load_n_cells_long(name):
    path = f'{working_dir}/output/n_cells_expr_{name}.parquet'
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
    de_results\
        .filter(pl.col('FDR') < 0.10)\
        .write_csv(f'{working_dir}/output/de_results_sig{de_suffix}.csv')
else:
    de_results = pl.read_csv(de_path)
    print(f'[de] cached: {de_results.height:,} rows, '
          f'{de_results["dataset"].n_unique()} datasets, '
          f'{de_results["contrast"].n_unique()} contrasts')

#endregion

#region sumrank meta-analysis ##################################################

# CDF of the sum of n iid Uniform(0,1); null distribution of the sum of
# n normalized ranks. Using CDF (not density as in Nakatsuka et al.) gives
# proper two-tail p-values without needing the N/2 cap workaround.
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
                (pl.col('dataset') == plt) &
                (pl.col('cell_type') == ct))
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
        # stratify p-values by d: Irwin-Hall parameter = number of uniforms
        nlp_up = np.full(len(d), np.nan)
        nlp_dn = np.full(len(d), np.nan)
        for k in np.unique(d[keep]):
            idx = (d == int(k)) & keep
            cdf = irwin_hall_cdf(s[idx], int(k))
            nlp_up[idx] = -np.log10(np.clip(cdf, 1e-300, 1.0))
            nlp_dn[idx] = -np.log10(np.clip(1 - cdf, 1e-300, 1.0))
        out.append(pl.DataFrame({
            'cell_type': ct,
            'gene': merged['gene'],
            'D': d.astype(np.int64),
            'sum_stat': s,
            'nlp_up': nlp_up,
            'nlp_down': nlp_dn,
        }).filter(pl.col('D') >= 2))
    return pl.concat(out) if out else pl.DataFrame()

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

def chunk_map(pattern):
    return {int(m.group(1)): p for p in glob.glob(pattern)
            if (m := re.search(r'_chunk_(\d+)\.parquet$', p))}

# perms run via R-side mclapply
r('''
suppressPackageStartupMessages({
    library(limma)
    library(dplyr)
    library(tibble)
    library(purrr)
    library(parallel)
})
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
           MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1")
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(1)
    RhpcBLASctl::omp_set_num_threads(1)
}
''')
r(Pseudobulk._voomByGroup_source_code)
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

for (name, contrast), pb_sub in all_pbs.items():
    if name not in SUMRANK_CONTRAST_PLATFORMS.get(contrast, []):
        continue
    final_path = f'{sumrank_cache_dir}/perm_{name}_{contrast}.parquet'
    if os.path.exists(final_path):
        print(f'[sumrank] perm cached: {name} {contrast}', flush=True)
        continue

    chunk_glob = (
        f'{sumrank_cache_dir}/perm_{name}_{contrast}_chunk_*.parquet')
    existing = chunk_map(chunk_glob)

    starts = list(range(0, SUMRANK_N_PERM, SUMRANK_CHUNK_SIZE))
    print(f'[sumrank] {name} {contrast}: {SUMRANK_N_PERM} perms in '
          f'{len(starts)} chunks of {SUMRANK_CHUNK_SIZE} '
          f'({len(existing)} cached), n_cores={SUMRANK_N_CORES}',
          flush=True)

    need_any = any(start not in existing for start in starts)
    if need_any:
        print(f'[sumrank] {name} {contrast}: populating R pseudobulks...',
              flush=True)
        t_pop = time.time()
        populate_pb_r(pb_sub, 'pb_cur')
        print(f'[sumrank] {name} {contrast}: populated in '
              f'{time.time()-t_pop:.0f}s', flush=True)
        treat, base = contrast.split('_vs_')
        to_r(treat, 'treat')
        to_r(base, 'base')
        to_r(DESIGN_FORMULAS[name], 'design_formula')
        to_r(SUMRANK_N_CORES, 'n_cores')
        to_r(PERM_SEED_BASE, 'seed_base')
    # same detection-threshold mask applied to real DE, used to suppress
    # sparse-detection hits consistently across real and all perms
    perm_mask = n_cells_mask(name, contrast)

    chunks = []
    t_pair = time.time()
    for i, start in enumerate(starts, 1):
        if start in existing:
            chunks.append(pl.read_parquet(existing[start]))
            print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
                  f'(cached)', flush=True)
            continue
        n_this = min(SUMRANK_CHUNK_SIZE, SUMRANK_N_PERM - start)
        print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
              f'starting ({n_this} perms)', flush=True)
        t0 = time.time()
        to_r(n_this, 'n_perm')
        to_r(start + 1, 'start_perm')
        r('chunk_df <- run_voom_perms(pb_cur, treat, base, design_formula, '
          'n_perm, n_cores, start_perm, seed_base)')
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
        chunks.append(chunk_df)
        eta_min = dt * (len(starts) - i) / 60
        print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
              f'done ({chunk_df["perm"].n_unique()} perms, '
              f'{chunk_df.height:,} rows, '
              f'{chunk_df["cell_type"].n_unique()} cell types, '
              f'{dt:.0f}s; eta {eta_min:.1f} min)', flush=True)

    if not chunks:
        print(f'[sumrank] WARN {name} {contrast}: no permutations produced',
              flush=True)
        continue
    full = pl.concat(chunks).unique(subset=['perm', 'cell_type', 'gene'])
    full.write_parquet(final_path)
    for p in glob.glob(chunk_glob):
        os.remove(p)
    print(f'[sumrank] {name} {contrast}: saved {full.height:,} rows '
          f'({full["perm"].n_unique()}/{SUMRANK_N_PERM} perms, '
          f'{(time.time() - t_pair) / 60:.1f} min total)', flush=True)
    if need_any:
        r('rm(pb_cur); invisible(gc())')

sumrank_final = []
for contrast, platforms in SUMRANK_CONTRAST_PLATFORMS.items():
    de_c = de_results.filter(pl.col('contrast') == contrast)
    if de_c.height == 0:
        continue
    real = sumrank_one(de_c, platforms)
    if real.height == 0:
        print(f'[sumrank] {contrast}: no cell types with >=2 platforms')
        continue

    perm_frames = {}
    for plt in platforms:
        p = f'{sumrank_cache_dir}/perm_{plt}_{contrast}.parquet'
        if os.path.exists(p):
            perm_frames[plt] = pl.read_parquet(p).with_columns(
                pl.lit(plt).alias('dataset'))
    if len(perm_frames) < 2:
        print(f'[sumrank] {contrast}: skip calibration, <2 perm caches')
        continue

    null_by_ct = defaultdict(list)
    max_k = min(int(pf['perm'].max()) for pf in perm_frames.values())
    t_null = time.time()
    for k in range(1, max_k + 1):
        dfs = [pf.filter(pl.col('perm') == k) for pf in perm_frames.values()]
        dfs = [d for d in dfs if d.height > 0]
        if len(dfs) < 2:
            continue
        sr_k = sumrank_one(pl.concat(dfs, how='diagonal'), platforms)
        if sr_k.height == 0:
            continue
        # pool nlp_up + nlp_down into one null per cell type: under the
        # null hypothesis the two directions are exchangeable (2x resolution)
        for ct in sr_k['cell_type'].unique().to_list():
            sub = sr_k.filter(pl.col('cell_type') == ct)
            u = sub['nlp_up'].to_numpy()
            d = sub['nlp_down'].to_numpy()
            null_by_ct[ct].append(u[~np.isnan(u)])
            null_by_ct[ct].append(d[~np.isnan(d)])
        if k % 10 == 0 or k == max_k:
            elapsed = time.time() - t_null
            eta = (max_k - k) * elapsed / max(k, 1)
            print(f'[sumrank] {contrast} null: {k}/{max_k} '
                  f'({elapsed:.0f}s elapsed, eta {eta:.0f}s)', flush=True)
    null_final = {
        ct: np.sort(np.concatenate(arrs)) if arrs else np.array([])
        for ct, arrs in null_by_ct.items()}
    emp_up, emp_dn = calibrate_emp_p(real, null_final)

    sumrank_final.append(real.with_columns(
        pl.lit(contrast).alias('contrast'),
        pl.Series('emp_p_up', emp_up),
        pl.Series('emp_p_down', emp_dn),
        pl.Series('emp_fdr_up', bh_fdr(emp_up)),
        pl.Series('emp_fdr_down', bh_fdr(emp_dn))))

sumrank_out = pl.concat(sumrank_final) if sumrank_final else pl.DataFrame()
if sumrank_out.height > 0:
    # sum n_cells across contributing datasets per (contrast, cell_type, gene)
    n_cells_agg = de_results\
        .group_by(['contrast', 'cell_type', 'gene'])\
        .agg(pl.col('n_cells_treat').sum(),
             pl.col('n_cells_base').sum())
    sumrank_out = add_ref_pct(sumrank_out)\
        .join(n_cells_agg, on=['contrast', 'cell_type', 'gene'], how='left')\
        .select([
            'contrast', 'cell_type', 'gene', 'D', 'sum_stat',
            'nlp_up', 'nlp_down', 'emp_p_up', 'emp_p_down',
            'emp_fdr_up', 'emp_fdr_down', 'ref_pct_detected',
            'n_cells_treat', 'n_cells_base'])
sumrank_out.write_csv(
    f'{working_dir}/output/sumrank_results{de_suffix}.csv')

for contrast in SUMRANK_CONTRAST_PLATFORMS:
    sub = sumrank_out.filter(pl.col('contrast') == contrast)
    if sub.height == 0:
        continue
    n_up = sub.filter(pl.col('emp_fdr_up') < 0.10).height
    n_dn = sub.filter(pl.col('emp_fdr_down') < 0.10).height
    n_ct = sub['cell_type'].n_unique()
    n_gn = sub['gene'].n_unique()
    print(f'[sumrank] {contrast}: {n_ct} cell types, {n_gn:,} genes, '
          f'{n_up} up / {n_dn} down DEGs (emp_fdr<0.10)')

#endregion

#region sumrank gsea pathways ##################################################

SUMRANK_GSEA_CACHE_DIR = f'{working_dir}/output/sumrank_gsea_cache'
os.makedirs(SUMRANK_GSEA_CACHE_DIR, exist_ok=True)
SUMRANK_GSEA_CHUNK = 20
SUMRANK_GSEA_N_CORES = os.cpu_count()
sumrank_gsea_path = f'{working_dir}/output/sumrank_gsea_results{de_suffix}.csv'

r(f'''
suppressPackageStartupMessages({{
    library(fgsea)
    library(msigdbr)
    library(dplyr)
    library(tibble)
    library(purrr)
}})

cache_file <- "{working_dir}/input/m_df_themed.rds"
if (!file.exists(cache_file)) {{
    m_df <- msigdbr(species = "Mus musculus", category = "C5",
                    subcategory = "GO:BP")
    theme_keywords <- list(
        'Neuronal' = c(
            'NEURO','SYNAP','AXON','DENDRITE','GLUTAMATE','GABA',
            'CHOLINERGIC','DOPAMINERGIC','SEROTONERGIC',
            'ACTION_POTENTIAL','REGULATION_NEUROTRANSMITTER_LEVELS',
            'REGULATION_SYNAPTIC_PLASTICITY'),
        'Metabolic' = c(
            'METABOLIC','LIPID','CHOLESTEROL','GLUCOSE_METABOLIC',
            'ATP_METABOLIC','CELLULAR_RESPIRATION',
            'ELECTRON_TRANSPORT','OXIDATIVE_PHOSPHORYLATION'),
        'Vascular' = c(
            'VASCULAR','VASCULATURE','ANGIOGENESIS','ENDOTHELIAL',
            'BLOOD_BRAIN_BARRIER','BLOOD_VESSEL',
            'ENDOTHELIAL_CELL_MIGRATION'),
        'Immune' = c(
            'IMMUNE','INFLAMMATORY','CYTOKINE','INTERFERON',
            'INNATE_IMMUNE','MICROGLIAL'),
        'Hormonal' = c(
            'HORMONE','STEROID','ESTROGEN','PROGESTERONE',
            'GLUCOCORTICOID','MINERALOCORTICOID',
            'CELLULAR_RESPONSE_HORMONE_STIMULUS'),
        'Growth_Factors' = c(
            'GROWTH_FACTOR','NEUROTROPHIC','BDNF','NGF','IGF',
            'FIBROBLAST_GROWTH',
            'CELLULAR_RESPONSE_GROWTH_FACTOR'),
        'Plasticity' = c(
            'NEUROGENESIS','DENDRITIC_SPINE','AXON_GUIDANCE',
            'SYNAPSE_ORGANIZATION',
            'NEURON_PROJECTION_DEVELOPMENT'),
        'Structural' = c(
            'ADHESION','EXTRACELLULAR_MATRIX','CELL_JUNCTION',
            'CELL_ADHESION'),
        'Protein_Dynamics' = c(
            'TRANSLATION','RIBOSOMAL','PROTEASOME',
            'UBIQUITIN','AUTOPHAGY','PROTEIN_FOLDING',
            'CHAPERONE'),
        'Ion_Transport' = c('CALCIUM','POTASSIUM','ION_TRANSPORT',
                            'MEMBRANE_POTENTIAL','ION_HOMEOSTASIS')
    )
    all_keywords <- unlist(theme_keywords)
    regex_pattern <- paste(all_keywords, collapse = "|")
    get_theme <- function(gs_name, themes) {{
        for (tn in names(themes)) {{
            if (any(sapply(themes[[tn]], grepl, gs_name, ignore.case=TRUE)))
                return(tn)
        }}
        NA_character_
    }}
    m_df_themed <- m_df %>%
        filter(grepl(regex_pattern, gs_name, ignore.case = TRUE)) %>%
        rowwise() %>%
        mutate(theme = get_theme(gs_name, theme_keywords)) %>%
        ungroup() %>%
        filter(!is.na(theme))
    saveRDS(m_df_themed, cache_file)
}} else {{
    m_df_themed <- readRDS(cache_file)
}}
filtered_pathways_sr <- split(m_df_themed$gene_symbol, m_df_themed$gs_name)

fgsea_ranked_groups <- function(ranked_df, keys, pathways, n_cores,
                                minSize = 15) {{
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
            fgsea(pathways = pathways, stats = ranks, minSize = minSize),
            error = function(e) NULL)
        if (is.null(res) || nrow(res) == 0) return(NULL)
        data.frame(pathway = res$pathway, NES = res$NES,
                   pvalue = res$pval,
                   key = rl$values[i],
                   stringsAsFactors = FALSE)
    }}
    results <- if (n_cores > 1)
        mclapply(seq_len(n_grp), worker, mc.cores = n_cores)
    else
        lapply(seq_len(n_grp), worker)
    bind_rows(results)
}}

fgsea_perms <- function(perm_df, pathways, n_cores, minSize = 15) {{
    keys <- paste(perm_df$perm, perm_df$cell_type, sep = "___")
    out <- fgsea_ranked_groups(perm_df, keys, pathways, n_cores, minSize)
    if (nrow(out) == 0) return(out)
    parts <- do.call(rbind, strsplit(out$key, "___", fixed = TRUE))
    out$perm <- as.integer(parts[, 1L])
    out$cell_type <- parts[, 2L]
    out$key <- NULL
    out
}}

fgsea_real <- function(ranked_df, pathways, n_cores, minSize = 15) {{
    out <- fgsea_ranked_groups(ranked_df, ranked_df$cell_type,
                               pathways, n_cores, minSize)
    if (nrow(out) == 0) return(out)
    out$cell_type <- out$key
    out$key <- NULL
    out
}}
''')

def sumrank_one_pw(gsea_frame, platforms):
    renamed = gsea_frame.rename({
        'pathway': 'gene', 'NES': 'logFC', 'pvalue': 'PValue'})
    out = sumrank_one(renamed, platforms)
    if out.height > 0:
        out = out.rename({'gene': 'pathway'})
    return out

def prerank_for_fgsea(df, group_cols):
    return df\
        .with_columns(
            ((-pl.col('PValue').log10()) * pl.col('logFC').sign())
            .alias('rank'))\
        .filter(pl.col('rank').is_finite())\
        .select(group_cols + ['gene', 'rank'])\
        .sort(group_cols + ['rank'],
              descending=[False] * len(group_cols) + [True])

real_gsea_cache = f'{SUMRANK_GSEA_CACHE_DIR}/real_gsea.parquet'
if os.path.exists(real_gsea_cache):
    real_gsea = pl.read_parquet(real_gsea_cache)
    print(f'[sumrank_gsea] real fgsea cached: {real_gsea.height:,} rows')
else:
    pairs = de_results\
        .select(['dataset', 'contrast'])\
        .unique()\
        .sort(['dataset', 'contrast'])\
        .rows()
    print(f'[sumrank_gsea] running real fgsea on {len(pairs)} '
          f'dataset/contrast pairs', flush=True)
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
        r('real_gsea_sub <- fgsea_real(ranked_df, filtered_pathways_sr, '
          'n_cores)')
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

for contrast, platforms in SUMRANK_CONTRAST_PLATFORMS.items():
    for name in platforms:
        if name not in datasets:
            continue
        ds_contrasts = {f'{t}_vs_{c}' for t, c in datasets[name]['contrasts']}
        if contrast not in ds_contrasts:
            continue
        gene_perm_path = f'{sumrank_cache_dir}/perm_{name}_{contrast}.parquet'
        if not os.path.exists(gene_perm_path):
            print(f'[sumrank_gsea] skip {name} {contrast}: gene perms '
                  f'not cached')
            continue
        final_path = f'{SUMRANK_GSEA_CACHE_DIR}/fgsea_{name}_{contrast}.parquet'
        if os.path.exists(final_path):
            print(f'[sumrank_gsea] fgsea perm cached: {name} {contrast}')
            continue

        chunk_glob = (f'{SUMRANK_GSEA_CACHE_DIR}/'
                      f'fgsea_{name}_{contrast}_chunk_*.parquet')
        existing = chunk_map(chunk_glob)

        n_perms = int(pl.scan_parquet(gene_perm_path)\
            .select(pl.col('perm').max())\
            .collect()\
            .item())
        starts = list(range(0, n_perms, SUMRANK_GSEA_CHUNK))
        print(f'[sumrank_gsea] {name} {contrast}: {n_perms} perms in '
              f'{len(starts)} chunks of {SUMRANK_GSEA_CHUNK} '
              f'({len(existing)} cached), n_cores={SUMRANK_GSEA_N_CORES}',
              flush=True)
        to_r(SUMRANK_GSEA_N_CORES, 'n_cores')

        chunks = []
        t_pair = time.time()
        for i, start in enumerate(starts, 1):
            if start in existing:
                chunks.append(pl.read_parquet(existing[start]))
                print(f'[sumrank_gsea] {name} {contrast}: chunk '
                      f'{i}/{len(starts)} (cached)', flush=True)
                continue
            end = min(start + SUMRANK_GSEA_CHUNK, n_perms)
            ks = list(range(start + 1, end + 1))
            sub = pl.scan_parquet(gene_perm_path)\
                .filter(pl.col('perm').is_in(ks))\
                .collect()
            if sub.height == 0:
                continue
            ranked = prerank_for_fgsea(
                sub.with_columns(pl.col('perm').cast(pl.Int64)),
                ['perm', 'cell_type'])
            t0 = time.time()
            to_r(ranked, 'ranked_df')
            r('chunk_df <- fgsea_perms(ranked_df, filtered_pathways_sr, '
              'n_cores)')
            chunk_df = to_py('chunk_df')
            dt = time.time() - t0
            if chunk_df is None or chunk_df.height == 0:
                print(f'[sumrank_gsea] {name} {contrast}: chunk '
                      f'{i}/{len(starts)} returned no rows in {dt:.0f}s',
                      flush=True)
                continue
            chunk_path = (f'{SUMRANK_GSEA_CACHE_DIR}/'
                          f'fgsea_{name}_{contrast}_chunk_{start}.parquet')
            chunk_df.write_parquet(chunk_path)
            chunks.append(chunk_df)
            remaining = len(starts) - i
            eta_min = dt * remaining / 60
            print(f'[sumrank_gsea] {name} {contrast}: chunk '
                  f'{i}/{len(starts)} done ({chunk_df["perm"].n_unique()} '
                  f'perms, {chunk_df.height:,} rows, {dt:.0f}s; '
                  f'eta {eta_min:.1f} min)', flush=True)

        if not chunks:
            print(f'[sumrank_gsea] WARN {name} {contrast}: no fgsea results')
            continue
        full = pl.concat(chunks).unique(
            subset=['perm', 'cell_type', 'pathway'])
        full.write_parquet(final_path)
        for p in glob.glob(chunk_glob):
            os.remove(p)
        print(f'[sumrank_gsea] {name} {contrast}: saved {full.height:,} rows '
              f'({(time.time() - t_pair) / 60:.1f} min total)')

sumrank_gsea_final = []
for contrast, platforms in SUMRANK_CONTRAST_PLATFORMS.items():
    real_c = real_gsea.filter(pl.col('contrast') == contrast)
    if real_c.height == 0:
        continue
    real = sumrank_one_pw(real_c, platforms)
    if real.height == 0:
        print(f'[sumrank_gsea] {contrast}: no cell types with >=2 platforms')
        continue

    perm_frames = {}
    for plt in platforms:
        p = f'{SUMRANK_GSEA_CACHE_DIR}/fgsea_{plt}_{contrast}.parquet'
        if os.path.exists(p):
            perm_frames[plt] = pl.read_parquet(p).with_columns(
                pl.lit(plt).alias('dataset'))
    if len(perm_frames) < 2:
        print(f'[sumrank_gsea] {contrast}: skip calibration, <2 perm caches')
        continue

    for plt, pf in list(perm_frames.items()):
        master_plt = real_c\
            .filter(pl.col('dataset') == plt)\
            .select(['cell_type', 'pathway'])\
            .unique()
        if master_plt.height == 0:
            continue
        perms_plt = pf.select('perm').unique()
        expected = perms_plt.join(master_plt, how='cross')
        pf_m = pf.join(master_plt, on=['cell_type', 'pathway'], how='inner')
        filled = expected.join(
            pf_m.select(['perm', 'cell_type', 'pathway', 'NES', 'pvalue']),
            on=['perm', 'cell_type', 'pathway'], how='left')
        filled = filled.with_columns(
            pl.col('NES').fill_null(0.0),
            pl.col('pvalue').fill_null(1.0),
            pl.lit(plt).alias('dataset'))
        perm_frames[plt] = filled

    null_by_ct = defaultdict(list)
    max_k = min(int(pf['perm'].max()) for pf in perm_frames.values())
    t_null = time.time()
    for k in range(1, max_k + 1):
        dfs = [pf.filter(pl.col('perm') == k) for pf in perm_frames.values()]
        dfs = [d for d in dfs if d.height > 0]
        if len(dfs) < 2:
            continue
        sr_k = sumrank_one_pw(pl.concat(dfs, how='diagonal'), platforms)
        if sr_k.height == 0:
            continue
        for ct in sr_k['cell_type'].unique().to_list():
            sub = sr_k.filter(pl.col('cell_type') == ct)
            u = sub['nlp_up'].to_numpy()
            d = sub['nlp_down'].to_numpy()
            null_by_ct[ct].append(u[~np.isnan(u)])
            null_by_ct[ct].append(d[~np.isnan(d)])
        if k % 10 == 0 or k == max_k:
            elapsed = time.time() - t_null
            eta = (max_k - k) * elapsed / max(k, 1)
            print(f'[sumrank_gsea] {contrast} null: {k}/{max_k} '
                  f'({elapsed:.0f}s elapsed, eta {eta:.0f}s)', flush=True)
    null_final = {
        ct: np.sort(np.concatenate(arrs)) if arrs else np.array([])
        for ct, arrs in null_by_ct.items()}
    emp_up, emp_dn = calibrate_emp_p(real, null_final)

    sumrank_gsea_final.append(real.with_columns(
        pl.lit(contrast).alias('contrast'),
        pl.Series('emp_p_up', emp_up),
        pl.Series('emp_p_down', emp_dn),
        pl.Series('emp_fdr_up', bh_fdr(emp_up)),
        pl.Series('emp_fdr_down', bh_fdr(emp_dn))))

sumrank_gsea_out = pl.concat(sumrank_gsea_final) if sumrank_gsea_final \
    else pl.DataFrame()
if sumrank_gsea_out.height > 0:
    sumrank_gsea_out = sumrank_gsea_out.select([
        'contrast', 'cell_type', 'pathway', 'D', 'sum_stat',
        'nlp_up', 'nlp_down', 'emp_p_up', 'emp_p_down',
        'emp_fdr_up', 'emp_fdr_down'])
sumrank_gsea_out.write_csv(sumrank_gsea_path)

for contrast in SUMRANK_CONTRAST_PLATFORMS:
    if sumrank_gsea_out.height == 0:
        break
    sub = sumrank_gsea_out.filter(pl.col('contrast') == contrast)
    if sub.height == 0:
        continue
    n_up = sub.filter(pl.col('emp_fdr_up') < 0.10).height
    n_dn = sub.filter(pl.col('emp_fdr_down') < 0.10).height
    n_ct = sub['cell_type'].n_unique()
    n_pw = sub['pathway'].n_unique()
    print(f'[sumrank_gsea] {contrast}: {n_ct} cell types, {n_pw:,} pathways, '
          f'{n_up} up / {n_dn} down (emp_fdr<0.10)')

#endregion

