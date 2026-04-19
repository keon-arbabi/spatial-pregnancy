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

from single_cell import SingleCell

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
de_suffix = f'_{cell_type_col}' if cell_type_col != 'subclass' else ''

REF_PCT_THRESHOLD = 0

datasets = {
    'slidetags': {
        'path': f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
        'design': '~ group + log_num_cells + log_lib_size',
    },
    'merfish': {
        'path': f'{working_dir}/output/merfish/03_adata_query_merfish.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
        'design': '~ group + log_num_cells + log_lib_size',
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
        'design': '~ group + log_num_cells',
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
        X_sub = ref[mask].X
        detected = np.ravel((X_sub > 0).sum(axis=0)) / n_cells
        pct_detected[s] = pd.Series(detected, index=ref.var_names)
    pkl.dump(pct_detected, open(pct_file, 'wb'))
    del ref; gc.collect()
else:
    pct_detected = pkl.load(open(pct_file, 'rb'))

def get_expressed_genes(cell_type, gene_list):
    if cell_type not in pct_detected:
        return gene_list
    ref = pct_detected[cell_type]
    return [g for g in gene_list
            if g not in ref.index or ref[g] >= REF_PCT_THRESHOLD / 100]

def get_ref_pct(cell_type, gene):
    if cell_type in pct_detected and gene in pct_detected[cell_type].index:
        return round(pct_detected[cell_type][gene] * 100, 1)
    return None

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
        .library_size(allow_float=True, num_threads=1)

def populate_r(pb, r_list, cfg):
    r(f'{r_list} <- list()')
    for cell_type, (X, obs, var) in pb.items():
        gene_names = (
            var['_index'] if '_index' in var.columns
            else pl.Series(var.to_pandas().index.tolist()))
        all_genes = gene_names.to_list()
        keep_genes = get_expressed_genes(cell_type, all_genes)
        keep_idx = [i for i, g in enumerate(all_genes) if g in keep_genes]
        if len(keep_idx) == 0:
            continue
        X_filt = X[:, keep_idx]
        gene_names_filt = pl.Series([all_genes[i] for i in keep_idx])

        to_r(obs, 'obs')
        to_r(cell_type, 'cell_type')
        to_r(X_filt, 'X', colnames=gene_names_filt)
        r(f'''
        counts <- t(X)
        element <- list(counts = counts, obs = obs)
        {r_list}[[cell_type]] <- element
        ''')

SUMRANK_CONTRAST_PLATFORMS = {
    'PREG_vs_CTRL': ['slidetags', 'merfish', 'xenium'],
    'POSTPART_vs_PREG': ['slidetags', 'merfish'],
    'POSTPART_vs_CTRL': ['slidetags', 'merfish'],
}
sumrank_cache_dir = f'{working_dir}/output/sumrank_cache'
os.makedirs(sumrank_cache_dir, exist_ok=True)
os.makedirs(f'{working_dir}/output', exist_ok=True)
de_path = f'{working_dir}/output/de_results{de_suffix}.csv'
real_cached = os.path.exists(de_path)

pairs_to_populate = set()
if not real_cached:
    for name, cfg in datasets.items():
        for treat, ctrl in cfg['contrasts']:
            pairs_to_populate.add((name, f'{treat}_vs_{ctrl}'))
for contrast, platforms in SUMRANK_CONTRAST_PLATFORMS.items():
    for name in platforms:
        if name not in datasets:
            continue
        ds_contrasts = {f'{t}_vs_{c}' for t, c in datasets[name]['contrasts']}
        if contrast not in ds_contrasts:
            continue
        if not os.path.exists(
                f'{sumrank_cache_dir}/perm_{name}_{contrast}.parquet'):
            pairs_to_populate.add((name, contrast))

all_r_lists = {}
names_to_pb = {n for n, _ in pairs_to_populate}
for name in names_to_pb:
    pb = make_pseudobulk(adatas[name], name)
    for treat, ctrl in datasets[name]['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        if (name, contrast) not in pairs_to_populate:
            continue
        r_list = f'pb_{name}_{contrast}'
        pb_sub = pb.filter_obs(pl.col('condition').is_in([treat, ctrl]))
        populate_r(pb_sub, r_list, datasets[name])
        all_r_lists[(name, contrast)] = (r_list, ctrl)
        print(f'[{name}] {contrast}: sent to R')

if pairs_to_populate:
    r('''
    suppressPackageStartupMessages({
        library(edgeR)
        library(dplyr)
        library(tibble)
        library(purrr)
    })

    run_edgeR <- function(
        pseudobulks, ref_level,
        design_formula = "~ group + log_num_cells + log_lib_size") {
        imap(pseudobulks, function(element, cell_type_name) {
            tryCatch({
                targets <- element$obs
                all_levels <- unique(as.character(targets$condition))
                other_level <- all_levels[all_levels != ref_level]
                targets$group <- factor(
                    targets$condition, levels = c(ref_level, other_level))
                if (n_distinct(targets$group) < 2) return(NULL)

                targets$log_num_cells <- log2(targets$num_cells)
                targets$log_lib_size <- log2(colSums(element$counts))
                design <- model.matrix(
                    as.formula(design_formula), data = targets)
                y <- DGEList(counts = element$counts, samples = targets)
                y <- calcNormFactors(y, method = "TMM")

                y <- estimateDisp(y, design, robust = TRUE)
                fit <- glmFit(y, design = design)
                test <- glmLRT(fit, coef = 2)

                tt <- topTags(test, n = Inf) %>%
                    as.data.frame() %>%
                    rownames_to_column("gene")
                tt
            }, error = function(e) {
                warning(paste("Error in", cell_type_name, ":", e$message))
                return(NULL)
            })
        }) %>%
        bind_rows(.id = "cell_type")
    }
    ''')

if not real_cached:
    de_frames = []
    for (name, contrast), (r_list, ref_level) in all_r_lists.items():
        design_formula = datasets[name]['design']
        to_r(ref_level, 'ref_level')
        to_r(design_formula, 'design_formula')
        r(f'de_tmp <- run_edgeR({r_list}, ref_level, design_formula)')
        df = to_py('de_tmp')
        if df is not None and df.height > 0:
            df = df.with_columns(
                pl.lit(contrast).alias('contrast'),
                pl.lit(name).alias('dataset'))
            de_frames.append(df)
            n_sig = df.filter(pl.col('FDR') < 0.10).height
            print(f'[{name}] {contrast}: {df.height:,} tests, '
                  f'{n_sig} DEGs (FDR<0.10)')

    de_results = pl.concat(de_frames)
    de_results = de_results.with_columns(
        pl.struct(['cell_type', 'gene']).map_elements(
            lambda r: get_ref_pct(r['cell_type'], r['gene']),
            return_dtype=pl.Float64
        ).alias('ref_pct_detected')
    )

    de_results.write_csv(de_path)
    de_results\
        .filter(pl.col('FDR') < 0.10)\
        .write_csv(f'{working_dir}/output/de_results_sig{de_suffix}.csv')

    for name in datasets:
        df = de_results.filter(pl.col('dataset') == name)
        for contrast in df['contrast'].unique().to_list():
            sub = df.filter(pl.col('contrast') == contrast)
            n_sig = sub.filter(pl.col('FDR') < 0.10).height
            n_ct = sub['cell_type'].n_unique()
            n_genes = sub['gene'].n_unique()
            print(f'[{name}] {contrast}: {n_ct} cell types, '
                  f'{n_genes:,} genes tested, {n_sig} DEGs')
else:
    de_results = pl.read_csv(de_path)
    print(f'[de] cached: {de_results.height:,} rows, '
          f'{de_results["dataset"].n_unique()} datasets, '
          f'{de_results["contrast"].n_unique()} contrasts')

#endregion

#region sumrank meta-analysis ##################################################

SUMRANK_N_PERM = 200
SUMRANK_N_CORES = 32
SUMRANK_CHUNK_SIZE = 32

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
        d = (~np.isnan(arr)).sum(axis=1)
        s = np.nansum(arr, axis=1)
        keep = d >= 2
        if not keep.any():
            continue
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

r('''
suppressPackageStartupMessages({
    library(parallel)
})
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1",
           MKL_NUM_THREADS = "1", BLIS_NUM_THREADS = "1")
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::blas_set_num_threads(1)
    RhpcBLASctl::omp_set_num_threads(1)
}

sumrank_errors <- new.env()

diag_pseudobulks <- function(pseudobulks, label) {
    cat("[R diag]", label, ": n_cell_types =", length(pseudobulks), "\\n")
    sc <- unique(do.call(rbind, lapply(pseudobulks, function(el) {
        unique(data.frame(sample = as.character(el$obs$sample),
                          condition = as.character(el$obs$condition),
                          stringsAsFactors = FALSE))
    })))
    cat("[R diag]", label, ": n_samples =", nrow(sc),
        "cond_counts =",
        paste(names(table(sc$condition)), table(sc$condition),
              sep = ":", collapse = ","), "\\n")
    sizes <- sapply(pseudobulks, function(el) nrow(el$obs))
    gene_counts <- sapply(pseudobulks, function(el) nrow(el$counts))
    cat("[R diag]", label, ": samples/ct min/med/max =",
        min(sizes), "/", median(sizes), "/", max(sizes), "\\n")
    cat("[R diag]", label, ": genes/ct min/med/max =",
        min(gene_counts), "/", median(gene_counts), "/",
        max(gene_counts), "\\n")
    flush.console()
    invisible(NULL)
}

edgeR_perm_one <- function(element, perm_cond, ref_level, design_formula) {
    tryCatch({
        targets <- element$obs
        samps <- as.character(targets$sample)
        levs_all <- unique(perm_cond[samps])
        other <- setdiff(levs_all, ref_level)
        if (length(other) == 0) return(NULL)
        targets$group <- factor(perm_cond[samps],
                                levels = c(ref_level, other))
        if (n_distinct(targets$group) < 2) return(NULL)
        if (min(table(targets$group)) < 2) return(NULL)
        targets$log_num_cells <- log2(targets$num_cells)
        targets$log_lib_size <- log2(colSums(element$counts))
        design <- model.matrix(as.formula(design_formula), data = targets)
        y <- DGEList(counts = element$counts, samples = targets)
        y <- calcNormFactors(y, method = "TMM")
        y <- estimateDisp(y, design, robust = TRUE)
        fit <- glmFit(y, design = design)
        test <- glmLRT(fit, coef = 2)
        tt <- topTags(test, n = Inf)$table
        data.frame(gene = rownames(tt), logFC = tt$logFC,
                   PValue = tt$PValue, stringsAsFactors = FALSE)
    }, error = function(e) {
        assign(as.character(runif(1)), conditionMessage(e),
               envir = sumrank_errors)
        NULL
    })
}

run_edgeR_perms <- function(pseudobulks, ref_level, design_formula,
                             n_perm, n_cores, start_perm = 1,
                             seed_base = 12345) {
    sample_cond <- unique(do.call(rbind, lapply(pseudobulks, function(el) {
        unique(data.frame(sample = as.character(el$obs$sample),
                          condition = as.character(el$obs$condition),
                          stringsAsFactors = FALSE))
    })))
    worker <- function(k) {
        gk <- start_perm + k - 1
        set.seed(seed_base + gk)
        perm_cond <- sample(sample_cond$condition)
        names(perm_cond) <- sample_cond$sample
        bind_rows(imap(pseudobulks, function(el, ct) {
            res <- edgeR_perm_one(el, perm_cond, ref_level, design_formula)
            if (is.null(res)) return(NULL)
            res$cell_type <- ct; res$perm <- gk; res
        }))
    }
    results <- if (n_cores > 1)
        mclapply(1:n_perm, worker, mc.cores = n_cores)
    else
        lapply(1:n_perm, worker)
    bind_rows(results)
}
''')

def flush_r_errors():
    r('''
    if (length(ls(sumrank_errors)) > 0) {
        msgs <- unlist(as.list(sumrank_errors))
        tbl <- sort(table(msgs), decreasing = TRUE)
        for (i in seq_len(min(5, length(tbl)))) {
            cat("[R diag] err x", as.integer(tbl[i]), ":",
                names(tbl)[i], "\\n")
        }
        rm(list = ls(sumrank_errors), envir = sumrank_errors)
    }
    flush.console()
    ''')

for (name, contrast), (r_list, ref_level) in all_r_lists.items():
    if name not in SUMRANK_CONTRAST_PLATFORMS.get(contrast, []):
        continue
    final_path = f'{sumrank_cache_dir}/perm_{name}_{contrast}.parquet'
    if os.path.exists(final_path):
        print(f'[sumrank] perm cached: {name} {contrast}', flush=True)
        continue

    chunk_glob = (
        f'{sumrank_cache_dir}/perm_{name}_{contrast}_chunk_*.parquet')
    existing = {}
    for p in glob.glob(chunk_glob):
        m = re.search(r'_chunk_(\d+)\.parquet$', p)
        if m:
            existing[int(m.group(1))] = p

    to_r(ref_level, 'ref_level')
    to_r(datasets[name]['design'], 'design_formula')
    to_r(SUMRANK_N_CORES, 'n_cores')

    starts = list(range(0, SUMRANK_N_PERM, SUMRANK_CHUNK_SIZE))
    print(f'[sumrank] {name} {contrast}: {SUMRANK_N_PERM} perms in '
          f'{len(starts)} chunks of {SUMRANK_CHUNK_SIZE} '
          f'({len(existing)} already on disk), n_cores={SUMRANK_N_CORES}',
          flush=True)

    chunks = []
    t_pair = time.time()
    last_dt = None
    for i, start in enumerate(starts, 1):
        if start in existing:
            chunks.append(pl.read_parquet(existing[start]))
            print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
                  f'(cached)', flush=True)
            continue
        n_this = min(SUMRANK_CHUNK_SIZE, SUMRANK_N_PERM - start)
        to_r(n_this, 'n_perm')
        to_r(start + 1, 'start_perm')
        print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
              f'starting ({n_this} perms, {SUMRANK_N_CORES} cores)',
              flush=True)
        t0 = time.time()
        r(f'perm_chunk <- run_edgeR_perms({r_list}, ref_level, '
          f'design_formula, n_perm, n_cores, start_perm)')
        chunk_df = to_py('perm_chunk')
        last_dt = time.time() - t0
        flush_r_errors()
        if chunk_df is None or chunk_df.height == 0:
            print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
                  f'returned 0 rows in {last_dt:.0f}s', flush=True)
            continue
        chunk_path = (f'{sumrank_cache_dir}/'
                      f'perm_{name}_{contrast}_chunk_{start}.parquet')
        chunk_df.write_parquet(chunk_path)
        chunks.append(chunk_df)
        remaining = len(starts) - i
        eta_min = last_dt * remaining / 60
        print(f'[sumrank] {name} {contrast}: chunk {i}/{len(starts)} '
              f'done ({chunk_df["perm"].n_unique()} perms, '
              f'{chunk_df.height:,} rows, '
              f'{chunk_df["cell_type"].n_unique()} cell types, '
              f'{last_dt:.0f}s; eta {eta_min:.1f} min)', flush=True)

    if not chunks:
        print(f'[sumrank] WARN {name} {contrast}: no permutations produced',
              flush=True)
        continue
    full = pl.concat(chunks).unique(
        subset=['perm', 'cell_type', 'gene'])
    full.write_parquet(final_path)
    for p in glob.glob(chunk_glob):
        os.remove(p)
    print(f'[sumrank] {name} {contrast}: saved {full.height:,} rows '
          f'({full["perm"].n_unique()}/{SUMRANK_N_PERM} perms, '
          f'{(time.time() - t_pair) / 60:.1f} min total)', flush=True)

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

    emp_up = np.full(real.height, np.nan)
    emp_dn = np.full(real.height, np.nan)
    nlp_up = real['nlp_up'].to_numpy()
    nlp_dn = real['nlp_down'].to_numpy()
    cts = real['cell_type'].to_numpy()
    for ct in np.unique(cts):
        msk = cts == ct
        nu = null_final.get(ct, np.array([]))
        if len(nu) > 0:
            idx_up = np.searchsorted(nu, nlp_up[msk], side='left')
            emp_up[msk] = np.maximum(
                (len(nu) - idx_up) / len(nu), 1.0 / len(nu))
            idx_dn = np.searchsorted(nu, nlp_dn[msk], side='left')
            emp_dn[msk] = np.maximum(
                (len(nu) - idx_dn) / len(nu), 1.0 / len(nu))

    sumrank_final.append(real.with_columns(
        pl.lit(contrast).alias('contrast'),
        pl.Series('emp_p_up', emp_up),
        pl.Series('emp_p_down', emp_dn),
        pl.Series('emp_fdr_up', bh_fdr(emp_up)),
        pl.Series('emp_fdr_down', bh_fdr(emp_dn))))

sumrank_out = pl.concat(sumrank_final) if sumrank_final else pl.DataFrame()
if sumrank_out.height > 0:
    sumrank_out = sumrank_out.select([
        'contrast', 'cell_type', 'gene', 'D', 'sum_stat',
        'nlp_up', 'nlp_down', 'emp_p_up', 'emp_p_down',
        'emp_fdr_up', 'emp_fdr_down'])
sumrank_out.write_csv(f'{working_dir}/output/sumrank_results{de_suffix}.csv')

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
SUMRANK_GSEA_CHUNK = 32
SUMRANK_GSEA_N_CORES = 32
SUMRANK_GSEA_INNER_BATCHES = 4
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

fgsea_one <- function(de_tab, pathways, minSize = 15) {{
    tryCatch({{
        ranks <- de_tab %>%
            mutate(rank = -log10(PValue) * sign(logFC)) %>%
            filter(is.finite(rank)) %>%
            arrange(desc(rank)) %>%
            select(gene, rank) %>%
            tibble::deframe()
        if (length(ranks) < 100) return(NULL)
        res <- fgsea(pathways = pathways, stats = ranks, minSize = minSize)
        if (nrow(res) == 0) return(NULL)
        data.frame(pathway = res$pathway, NES = res$NES,
                   pvalue = res$pval, stringsAsFactors = FALSE)
    }}, error = function(e) NULL)
}}

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
    return (df
        .with_columns(
            ((-pl.col('PValue').log10()) * pl.col('logFC').sign())
            .alias('rank'))
        .filter(pl.col('rank').is_finite())
        .select(group_cols + ['gene', 'rank'])
        .sort(group_cols + ['rank'],
              descending=[False] * len(group_cols) + [True]))

real_gsea_cache = f'{SUMRANK_GSEA_CACHE_DIR}/real_gsea.parquet'
if os.path.exists(real_gsea_cache):
    real_gsea = pl.read_parquet(real_gsea_cache)
    print(f'[sumrank_gsea] real fgsea cached: {real_gsea.height:,} rows')
else:
    pairs = (de_results.select(['dataset', 'contrast']).unique()
             .sort(['dataset', 'contrast']).rows())
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
        existing = {}
        for p in glob.glob(chunk_glob):
            m = re.search(r'_chunk_(\d+)\.parquet$', p)
            if m:
                existing[int(m.group(1))] = p

        n_perms = int(pl.scan_parquet(gene_perm_path)
                      .select(pl.col('perm').max()).collect().item())
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
                .filter(pl.col('perm').is_in(ks)).collect()
            if sub.height == 0:
                continue
            ranked = prerank_for_fgsea(
                sub.with_columns(pl.col('perm').cast(pl.Int64)),
                ['perm', 'cell_type'])

            unique_ks = sorted(ranked['perm'].unique().to_list())
            n_batches = min(SUMRANK_GSEA_INNER_BATCHES, len(unique_ks))
            batch_groups = np.array_split(unique_ks, n_batches)
            n_cts_chunk = ranked['cell_type'].n_unique()
            print(f'[sumrank_gsea][dbg] {name} {contrast}: chunk '
                  f'{i}/{len(starts)} ranked={ranked.height:,} rows, '
                  f'{len(unique_ks)} perms x {n_cts_chunk} cts '
                  f'-> {n_batches} batches', flush=True)
            sub_results = []
            t_chunk = time.time()
            for bi, pb in enumerate(batch_groups, 1):
                part_in = ranked.filter(pl.col('perm').is_in(pb.tolist()))
                print(f'[sumrank_gsea][dbg] chunk {i}/{len(starts)} '
                      f'batch {bi}/{n_batches}: sending '
                      f'{part_in.height:,} rows to R', flush=True)
                t0 = time.time()
                to_r(part_in, 'ranked_df')
                t_to_r = time.time() - t0
                print(f'[sumrank_gsea][dbg] chunk {i}/{len(starts)} '
                      f'batch {bi}/{n_batches}: to_r done ({t_to_r:.0f}s), '
                      f'launching fgsea on {n_cts_chunk * len(pb)} tasks '
                      f'across {SUMRANK_GSEA_N_CORES} cores', flush=True)
                t1 = time.time()
                r('fgsea_batch <- fgsea_perms(ranked_df, '
                  'filtered_pathways_sr, n_cores)')
                t_fgsea = time.time() - t1
                print(f'[sumrank_gsea][dbg] chunk {i}/{len(starts)} '
                      f'batch {bi}/{n_batches}: fgsea done '
                      f'({t_fgsea:.0f}s), collecting', flush=True)
                batch_df = to_py('fgsea_batch')
                dt = time.time() - t0
                n_rows = 0 if batch_df is None else batch_df.height
                if batch_df is not None and batch_df.height > 0:
                    sub_results.append(batch_df)
                print(f'[sumrank_gsea] {name} {contrast}: chunk '
                      f'{i}/{len(starts)} batch {bi}/{n_batches} '
                      f'({len(pb)} perms, {n_rows:,} rows, {dt:.0f}s)',
                      flush=True)
            if not sub_results:
                print(f'[sumrank_gsea] {name} {contrast}: chunk '
                      f'{i}/{len(starts)} returned no rows', flush=True)
                continue
            chunk_df = pl.concat(sub_results)
            chunk_path = (f'{SUMRANK_GSEA_CACHE_DIR}/'
                          f'fgsea_{name}_{contrast}_chunk_{start}.parquet')
            chunk_df.write_parquet(chunk_path)
            chunks.append(chunk_df)
            remaining = len(starts) - i
            chunk_dt = time.time() - t_chunk
            eta_min = chunk_dt * remaining / 60
            print(f'[sumrank_gsea] {name} {contrast}: chunk '
                  f'{i}/{len(starts)} done ({chunk_df["perm"].n_unique()} '
                  f'perms, {chunk_dt:.0f}s; eta {eta_min:.1f} min)',
                  flush=True)

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
        master_plt = (real_c.filter(pl.col('dataset') == plt)
                      .select(['cell_type', 'pathway']).unique())
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

    emp_up = np.full(real.height, np.nan)
    emp_dn = np.full(real.height, np.nan)
    nlp_up = real['nlp_up'].to_numpy()
    nlp_dn = real['nlp_down'].to_numpy()
    cts = real['cell_type'].to_numpy()
    for ct in np.unique(cts):
        msk = cts == ct
        nu = null_final.get(ct, np.array([]))
        if len(nu) > 0:
            idx_up = np.searchsorted(nu, nlp_up[msk], side='left')
            emp_up[msk] = np.maximum(
                (len(nu) - idx_up) / len(nu), 1.0 / len(nu))
            idx_dn = np.searchsorted(nu, nlp_dn[msk], side='left')
            emp_dn[msk] = np.maximum(
                (len(nu) - idx_dn) / len(nu), 1.0 / len(nu))

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

#region de barplot #############################################################

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

FDR_STRICT = 0.10
FDR_LOOSE = 0.20
MIN_DEGS_TO_SHOW = 10
seismic_cmap = plt.get_cmap('seismic')
UP_COLOR = seismic_cmap(0.9)
DN_COLOR = seismic_cmap(0.1)

de_fig = pl.read_csv(f'{working_dir}/output/de_results{de_suffix}.csv')

def get_type(ct):
    if 'Glut' in ct:
        return 'Glut'
    elif any(x in ct for x in ['Gaba', 'IMN', 'Chol']):
        return 'Gaba'
    return 'NN'

contrasts = ['PREG_vs_CTRL', 'POSTPART_vs_PREG', 'POSTPART_vs_CTRL']
contrast_titles = {
    'PREG_vs_CTRL': 'Pregnant vs\nNulliparous',
    'POSTPART_vs_PREG': 'Postpartum vs\nPregnant',
    'POSTPART_vs_CTRL': 'Postpartum vs\nNulliparous',
}

deg_counts = de_fig\
    .filter((pl.col('FDR') < FDR_LOOSE) &
            (pl.col('dataset').is_in(['slidetags', 'xenium'])))\
    .group_by(['cell_type', 'contrast', 'dataset'])\
    .agg(((pl.col('FDR') < FDR_STRICT) &
          (pl.col('logFC') > 0)).sum().alias('up_strict'),
         ((pl.col('FDR') < FDR_STRICT) &
          (pl.col('logFC') < 0)).sum().alias('down_strict'),
         ((pl.col('FDR') < FDR_LOOSE) &
          (pl.col('logFC') > 0)).sum().alias('up_loose'),
         ((pl.col('FDR') < FDR_LOOSE) &
          (pl.col('logFC') < 0)).sum().alias('down_loose'))

ct_totals = deg_counts\
    .group_by('cell_type')\
    .agg((pl.sum('up_strict') + pl.sum('down_strict')).alias('total'))\
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

st_cell_types = set(deg_counts.filter(
    pl.col('dataset') == 'slidetags')['cell_type'].unique().to_list())

BAR_H = 0.42
st_offset = -BAR_H / 2
xn_offset = BAR_H / 2

fig = plt.figure(figsize=(6, 12))
outer_gs = gridspec.GridSpec(
    len(major_types), len(contrasts), figure=fig,
    height_ratios=height_ratios, hspace=0.06, wspace=0.04,
    width_ratios=[2, 1, 1])

for i, group_type in enumerate(major_types):
    group_cts = groups.filter(
        pl.col('type') == group_type)['cell_type'].explode().to_list()

    for j, contrast in enumerate(contrasts):
        ax = fig.add_subplot(outer_gs[i, j])

        show_xn = (contrast == 'PREG_vs_CTRL')

        st_data = deg_counts.filter(
            (pl.col('contrast') == contrast) &
            (pl.col('dataset') == 'slidetags') &
            (pl.col('cell_type').is_in(group_cts)))
        st_dict = {r['cell_type']: r for r in st_data.to_dicts()}

        def draw_stacked(y, row, color_up, color_dn, alpha):
            u_strict = row.get('up_strict', 0)
            d_strict = row.get('down_strict', 0)
            u_loose = row.get('up_loose', 0)
            d_loose = row.get('down_loose', 0)
            u_extra = max(u_loose - u_strict, 0)
            d_extra = max(d_loose - d_strict, 0)

            lw = 0.4
            ax.barh(y, u_strict, height=BAR_H, align='center',
                    facecolor=color_up, edgecolor=color_up,
                    alpha=alpha, linewidth=lw, zorder=5)
            ax.barh(y, -d_strict, height=BAR_H, align='center',
                    facecolor=color_dn, edgecolor=color_dn,
                    alpha=alpha, linewidth=lw, zorder=5)

            if u_extra > 0:
                ax.barh(y, u_extra, left=u_strict, height=BAR_H,
                        align='center',
                        facecolor='none', edgecolor=color_up,
                        linewidth=lw, hatch='////',
                        alpha=alpha, zorder=6)
            if d_extra > 0:
                ax.barh(y, -d_extra, left=-d_strict, height=BAR_H,
                        align='center',
                        facecolor='none', edgecolor=color_dn,
                        linewidth=lw, hatch='////',
                        alpha=alpha, zorder=6)

        if show_xn:
            xn_data = deg_counts.filter(
                (pl.col('contrast') == contrast) &
                (pl.col('dataset') == 'xenium') &
                (pl.col('cell_type').is_in(group_cts)))
            xn_dict = {r['cell_type']: r for r in xn_data.to_dicts()}

            for idx, ct in enumerate(group_cts):
                draw_stacked(idx + st_offset,
                             st_dict.get(ct, {}), UP_COLOR, DN_COLOR, 0.9)
                draw_stacked(idx + xn_offset,
                             xn_dict.get(ct, {}), UP_COLOR, DN_COLOR, 0.45)
        else:
            for idx, ct in enumerate(group_cts):
                draw_stacked(idx, st_dict.get(ct, {}),
                             UP_COLOR, DN_COLOR, 0.9)

        all_up, all_dn = [], []
        for ct in group_cts:
            r_st = st_dict.get(ct, {})
            all_up.append(r_st.get('up_loose', 0))
            all_dn.append(r_st.get('down_loose', 0))
            if show_xn:
                r_xn = xn_dict.get(ct, {})
                all_up.append(r_xn.get('up_loose', 0))
                all_dn.append(r_xn.get('down_loose', 0))
        xlim = max(max(all_up + [1]), max(all_dn + [1])) * 1.25

        ax.axvline(0, color='grey', linewidth=0.5, zorder=0)
        ax.grid(True, 'major', 'y', ls='-', lw=0.3, c='lightgray', zorder=0)
        ax.set_xlim(-xlim, xlim)
        ax.set_yticks(range(len(group_cts)))
        ax.set_ylim(len(group_cts) - 0.5, -0.5)
        ax.tick_params(length=0, labelsize=7)

        if j == 0:
            ax.set_yticklabels(group_cts, fontsize=7.5)
            ax.tick_params(axis='y', pad=8)
            for idx, ct in enumerate(group_cts):
                if ct not in st_cell_types:
                    ax.plot(-0.01, idx, 's', color='#555555', markersize=3,
                            transform=ax.get_yaxis_transform(), clip_on=False,
                            zorder=20)
        else:
            ax.set_yticklabels([])

        if i == 0:
            ax.set_title(contrast_titles[contrast], fontsize=9, pad=6)

        if i == len(major_types) - 1:
            ax.set_xlabel('DEGs', fontsize=8.5)
        else:
            ax.set_xticklabels([])

legend_elements = [
    Patch(facecolor=UP_COLOR, edgecolor=UP_COLOR, alpha=0.9, linewidth=0.4,
          label='Upregulated'),
    Patch(facecolor=DN_COLOR, edgecolor=DN_COLOR, alpha=0.9, linewidth=0.4,
          label='Downregulated'),
    Patch(facecolor='lightgray', edgecolor='black', linewidth=0.4,
          label='Slide-tags'),
    Patch(facecolor='lightgray', edgecolor='black', alpha=0.45,
          linewidth=0.4, label='Xenium'),
    Patch(facecolor='lightgray', edgecolor='black', linewidth=0.4,
          label='FDR<0.10'),
    Patch(facecolor='none', edgecolor='black', linewidth=0.4,
          hatch='////', label='FDR<0.20'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#555555',
           markersize=5, label='Xenium only'),
]
fig.legend(handles=legend_elements, loc='lower right',
           bbox_to_anchor=(0.98, 0.02), fontsize=7,
           frameon=False, ncol=4)

os.makedirs(f'{working_dir}/figures', exist_ok=True)
plt.savefig(f'{working_dir}/figures/deg_barplot.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/deg_barplot.svg',
            bbox_inches='tight')
plt.close()

#endregion

#region de exemplars ###########################################################

EXEMPLAR_GENES = [
    ('Grm1',    '058 PAL-STR Gaba-Chol'),
    ('Slc17a8', '058 PAL-STR Gaba-Chol'),
    ('Trpc5',   '009 L2/3 IT PIR-ENTl Glut'),
    ('Dusp7',   '006 L4/5 IT CTX Glut'),
    ('Nefl',    '064 STR-PAL Chst9 Gaba'),
    ('Kdr',     '333 Endo NN'),
    ('Igfbp3',  '333 Endo NN'),
    ('Idi1',    '319 Astro-TE NN'),
    ('Gjb6',    '319 Astro-TE NN'),
    ('Grm8',    '085 SI-MPO-LPO Lhx8 Gaba'),
    ('Calcr',   '085 SI-MPO-LPO Lhx8 Gaba'),
    ('Sod2',    '085 SI-MPO-LPO Lhx8 Gaba'),
    ('Cntnap4', '118 ADP-MPO Trp73 Glut'),
    ('Grb10',   '118 ADP-MPO Trp73 Glut'),
]
condition_colors = {
    'CTRL': '#7209b7',
    'PREG': '#b5179e',
    'POSTPART': '#f72585',
}

def get_pseudobulk_expr(adata_src, gene, cell_type):
    if gene not in adata_src.var_names:
        return {}
    cell_mask = adata_src.obs[cell_type_col] == cell_type
    if cell_mask.sum() == 0:
        return {}
    subset = adata_src[cell_mask, gene]
    groups = subset.obs.groupby('sample')['condition'].first()
    result = {}
    for sample, cond in groups.items():
        s_mask = subset.obs['sample'] == sample
        total = subset[s_mask].X.sum()
        n_cells = s_mask.sum()
        result[sample] = {
            'cond': cond,
            'cpm': np.log2(total / n_cells * 1e4 + 1) if n_cells > 0 else 0,
        }
    return result

def zscore_pb(pb):
    if not pb:
        return
    vals = [v['cpm'] for v in pb.values()]
    mu = np.mean(vals)
    sd = max(np.std(vals), 1e-6)
    for v in pb.values():
        v['z'] = (v['cpm'] - mu) / sd

def draw_panel(ax, pb, conditions, marker, ms, alpha, seed):
    cond_pos = {c: i for i, c in enumerate(conditions)}
    rng = np.random.default_rng(seed)
    for vals in pb.values():
        cond = vals['cond']
        if cond not in cond_pos:
            continue
        x = cond_pos[cond] + rng.uniform(-0.18, 0.18)
        y = vals['z'] + rng.uniform(-0.05, 0.05)
        ax.scatter(x, y, marker=marker,
                   c=condition_colors[cond], s=ms, alpha=alpha,
                   linewidths=0, zorder=10)
    means = {}
    for cond in conditions:
        zvals = [v['z'] for v in pb.values() if v['cond'] == cond]
        if zvals:
            m = np.mean(zvals)
            se = np.std(zvals) / np.sqrt(len(zvals)) \
                if len(zvals) > 1 else 0
            means[cond] = m
            ax.errorbar(cond_pos[cond], m, yerr=se,
                        fmt='none', color='black', capsize=1.5,
                        capthick=0.5, linewidth=0.5, zorder=5,
                        alpha=alpha)
    if len(means) > 1:
        xs = [cond_pos[c] for c in conditions if c in means]
        ys = [means[c] for c in conditions if c in means]
        ax.plot(xs, ys, '-', color='black', linewidth=0.6,
                alpha=0.3, zorder=1)

def add_sig_star(ax, gene, cell_type, dataset, cond_pos):
    is_sig = de_fig.filter(
        (pl.col('gene') == gene) &
        (pl.col('cell_type') == cell_type) &
        (pl.col('dataset') == dataset) &
        (pl.col('contrast') == 'PREG_vs_CTRL') &
        (pl.col('FDR') < 0.10)).height > 0
    if not is_sig or 'CTRL' not in cond_pos or 'PREG' not in cond_pos:
        return
    x = (cond_pos['CTRL'] + cond_pos['PREG']) / 2
    ylim = ax.get_ylim()
    ax.text(x, ylim[1] - (ylim[1] - ylim[0]) * 0.02, '*',
            ha='center', va='top', fontsize=9, fontweight='bold',
            clip_on=False)

def format_ax(ax, linewidth=0.5):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)

adata_xn_norm = adatas['xenium'].copy()
sc.pp.normalize_total(adata_xn_norm, target_sum=1e4)
sc.pp.log1p(adata_xn_norm)

xn_ct_subsets = {}
for ct in set(ct for _, ct in EXEMPLAR_GENES):
    mask = adata_xn_norm.obs[cell_type_col] == ct
    if mask.sum() > 0:
        xn_ct_subsets[ct] = adata_xn_norm[mask]

all_ffd = adata_xn_norm.obs[['x_ffd', 'y_ffd']].values
fov_cx = (all_ffd[:, 0].min() + all_ffd[:, 0].max()) / 2
fov_cy = (all_ffd[:, 1].min() + all_ffd[:, 1].max()) / 2
fov_half = max(np.ptp(all_ffd[:, 0]), np.ptp(all_ffd[:, 1])) / 2 * 1.05

n_genes = len(EXEMPLAR_GENES)
n_col_genes = 2
n_rows = (n_genes + n_col_genes - 1) // n_col_genes

fig = plt.figure(figsize=(n_col_genes * 4.0, n_rows * 2.0))
outer_gs = gridspec.GridSpec(n_rows, n_col_genes, figure=fig,
                              hspace=0.35, wspace=0.2)

for idx, (gene, cell_type) in enumerate(EXEMPLAR_GENES):
    row_i = idx // n_col_genes
    col_i = idx % n_col_genes
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer_gs[row_i, col_i],
        hspace=0.05, wspace=0.04,
        width_ratios=[0.7, 1.0, 1.0])
    ax_st = fig.add_subplot(inner_gs[0, 0])
    ax_xn = fig.add_subplot(inner_gs[1, 0], sharex=ax_st)
    ax_sp_ctrl = fig.add_subplot(inner_gs[:, 1])
    ax_sp_preg = fig.add_subplot(inner_gs[:, 2])

    pb_st = get_pseudobulk_expr(adatas['slidetags'], gene, cell_type)
    pb_xn = get_pseudobulk_expr(adatas['xenium'], gene, cell_type)

    if not pb_st and not pb_xn:
        for a in [ax_st, ax_xn, ax_sp_ctrl, ax_sp_preg]:
            a.set_visible(False)
        continue

    has_postpart = any(v['cond'] == 'POSTPART' for v in pb_st.values())
    st_conditions = ['CTRL', 'PREG', 'POSTPART'] if has_postpart \
        else ['CTRL', 'PREG']
    xn_conditions = ['CTRL', 'PREG']

    zscore_pb(pb_st)
    zscore_pb(pb_xn)

    draw_panel(ax_st, pb_st, st_conditions, 'o', 12, 0.90, idx)
    draw_panel(ax_xn, pb_xn, xn_conditions, 'D', 9, 0.65, idx + 1000)

    for ax in [ax_st, ax_xn]:
        yl = ax.get_ylim()
        pad = (yl[1] - yl[0]) * 0.08
        ax.set_ylim(yl[0] - pad, yl[1] + pad)

    st_cond_pos = {c: i for i, c in enumerate(st_conditions)}
    xn_cond_pos = {c: i for i, c in enumerate(xn_conditions)}
    add_sig_star(ax_st, gene, cell_type, 'slidetags', st_cond_pos)
    add_sig_star(ax_xn, gene, cell_type, 'xenium', xn_cond_pos)

    n_x = max(len(st_conditions), len(xn_conditions))
    ax_st.set_xlim(-0.5, n_x - 0.5)

    ct_label = re.sub(r'^\d+\s+', '', cell_type)
    parent = fig.add_subplot(outer_gs[row_i, col_i])
    parent.axis('off')
    parent.patch.set_alpha(0)
    parent.set_title(f'{gene}\n{ct_label}', fontsize=6.5, pad=3)

    plt.setp(ax_st.get_xticklabels(), visible=False)
    ax_st.tick_params(axis='x', length=0)
    ax_xn.set_xticks(list(range(n_x)))
    ax_xn.set_xticklabels(['N', 'P', 'PP'][:n_x], fontsize=5)
    ax_xn.tick_params(axis='x', length=1.5)

    for ax in [ax_st, ax_xn]:
        ax.tick_params(axis='y', labelsize=5.5, length=1.5)
        format_ax(ax)

    subset = xn_ct_subsets.get(cell_type)
    if subset is not None and gene in subset.var_names:
        expr = np.asarray(subset[:, gene].X.toarray()).ravel()
        coords_x = subset.obs['x_ffd'].values
        coords_y = subset.obs['y_ffd'].values
        conds = subset.obs['condition'].values

        nonzero = expr[expr > 0]
        vmin = np.quantile(nonzero, 0.05) if len(nonzero) > 0 else 0
        vmax = np.quantile(nonzero, 0.95) if len(nonzero) > 0 else 1
        if vmax <= vmin:
            vmax = vmin + 1

        for ax_sp, cond, sp_label in [
                (ax_sp_ctrl, 'CTRL', 'Nulliparous'),
                (ax_sp_preg, 'PREG', 'Pregnant')]:
            c_mask = conds == cond
            if c_mask.sum() == 0:
                ax_sp.set_visible(False)
                continue
            order = np.argsort(expr[c_mask])
            ax_sp.scatter(
                coords_x[c_mask][order], coords_y[c_mask][order],
                c=expr[c_mask][order], cmap='viridis', s=0.2,
                vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
            ax_sp.set_xlim(fov_cx - fov_half, fov_cx + fov_half)
            ax_sp.set_ylim(fov_cy - fov_half, fov_cy + fov_half)
            ax_sp.set_xticks([])
            ax_sp.set_yticks([])
            ax_sp.set_facecolor('black')
            ax_sp.set_xlabel(sp_label, fontsize=5, labelpad=2)
            format_ax(ax_sp)
    else:
        ax_sp_ctrl.set_visible(False)
        ax_sp_preg.set_visible(False)

fig.text(0.04, 0.5, 'Expression (z-score)',
         va='center', ha='center', rotation='vertical', fontsize=7)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=4, markeredgecolor='none', label='Slide-tags'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
           markersize=3.5, markeredgecolor='none',
           label='Xenium', alpha=0.65),
]
last_row_y = 1 / n_rows * 0.15
fig.legend(handles=legend_elements, loc='lower center',
           bbox_to_anchor=(0.4, last_row_y), fontsize=6, frameon=False,
           ncol=2)

cbar_ax = fig.add_axes([0.55, last_row_y + 0.003, 0.06, 0.006])
sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['low', 'high'], fontsize=5)
cbar.ax.set_xlabel('log₁p expression', fontsize=5, labelpad=1)
cbar.outline.set_linewidth(0.4)

plt.savefig(f'{working_dir}/figures/deg_exemplar_pseudobulk.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/deg_exemplar_pseudobulk.svg',
            bbox_inches='tight')
plt.close()
del adata_xn_norm, xn_ct_subsets

#endregion
