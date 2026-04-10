#region imports and setup #######################################################

import os
import gc
import pickle as pkl
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r, to_py

import warnings
warnings.filterwarnings('ignore')

from single_cell import SingleCell

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'

datasets = {
    'slidetags': {
        'path': f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
        'norm': 'tmm',
    },
    'merfish': {
        'path': f'{working_dir}/output/merfish/03_adata_query_merfish.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
        'norm': 'volume',
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'norm': 'area',
    },
}

#endregion
#region reference detection rates ##############################################

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

#endregion
#region load data ##############################################################

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
    adatas[name] = adata
    print(f'[{name}] {adata.shape[0]:,} cells, '
          f'{adata.obs[cell_type_col].nunique()} subclasses, '
          f'{adata.obs["condition"].nunique()} conditions')

#endregion
#region pseudobulk #############################################################

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
            min_cells=10,
            max_standard_deviations=None,
            min_nonzero_fraction=0,
            verbose=False)\
        .library_size(allow_float=True, num_threads=1)

def populate_r(pb, r_list, cfg, adata):
    r(f'{r_list} <- list()')
    for cell_type, (X, obs, var) in pb.items():
        to_r(obs, 'obs')
        to_r(cell_type, 'cell_type')
        gene_names = (
            var['_index'] if '_index' in var.columns
            else pl.Series(var.to_pandas().index.tolist()))
        to_r(X, 'X', colnames=gene_names)
        size_col = {'volume': 'volume', 'area': 'cell_area'}\
            .get(cfg['norm'])
        if size_col:
            size_sums = []
            for row in obs.iter_rows(named=True):
                sample = row['sample']
                mask = ((adata.obs['sample'] == sample) &
                        (adata.obs[cell_type_col] == cell_type))
                size_sums.append(adata.obs.loc[mask, size_col].sum())
            to_r(np.array(size_sums), 'size_sums')
            r(f'''
            counts <- t(X)
            element <- list(counts = counts, obs = obs, size_sums = size_sums)
            {r_list}[[cell_type]] <- element
            ''')
        else:
            r(f'''
            counts <- t(X)
            element <- list(counts = counts, obs = obs)
            {r_list}[[cell_type]] <- element
            ''')

all_r_lists = {}
for name, cfg in datasets.items():
    adata = adatas[name]
    pb = make_pseudobulk(adata, name)
    for treat, ctrl in cfg['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        r_list = f'pb_{name}_{contrast}'
        pb_sub = pb.filter_obs(pl.col('condition').is_in([treat, ctrl]))
        populate_r(pb_sub, r_list, cfg, adata)
        all_r_lists[(name, contrast)] = (r_list, ctrl)
        print(f'[{name}] {contrast}: sent to R')

#endregion
#region edgeR ##################################################################

r('''
suppressPackageStartupMessages({
    library(edgeR)
    library(dplyr)
    library(tibble)
    library(purrr)
})

run_edgeR <- function(pseudobulks, ref_level, norm_method) {
    imap(pseudobulks, function(element, cell_type_name) {
        tryCatch({
            targets <- element$obs
            all_levels <- unique(as.character(targets$condition))
            other_level <- all_levels[all_levels != ref_level]
            targets$group <- factor(
                targets$condition, levels = c(ref_level, other_level))
            if (n_distinct(targets$group) < 2) return(NULL)

            design <- model.matrix(~ group, data = targets)
            y <- DGEList(counts = element$counts, samples = targets)

            if (norm_method %in% c("volume", "area")) {
                log_size <- log(element$size_sums)
                y$offset <- matrix(log_size, nrow = nrow(y$counts),
                                   ncol = ncol(y$counts), byrow = TRUE)
            } else {
                y <- calcNormFactors(y, method = "TMM")
            }

            y <- estimateDisp(y, design)
            fit <- glmFit(y, design = design)
            test <- glmLRT(fit, coef = 2)

            topTags(test, n = Inf) %>%
                as.data.frame() %>%
                rownames_to_column("gene")
        }, error = function(e) {
            warning(paste("Error in", cell_type_name, ":", e$message))
            return(NULL)
        })
    }) %>%
    bind_rows(.id = "cell_type")
}
''')

de_frames = []
for (name, contrast), (r_list, ref_level) in all_r_lists.items():
    norm = datasets[name]['norm']
    to_r(ref_level, 'ref_level')
    to_r(norm, 'norm_method')
    r(f'de_tmp <- run_edgeR({r_list}, ref_level, norm_method)')
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

#endregion
#region add reference detection rates ##########################################

def get_ref_pct(cell_type, gene):
    if cell_type in pct_detected and gene in pct_detected[cell_type].index:
        return round(pct_detected[cell_type][gene] * 100, 1)
    return None

REF_PCT_THRESHOLD = 10

de_results = de_results.with_columns(
    pl.struct(['cell_type', 'gene']).map_elements(
        lambda r: get_ref_pct(r['cell_type'], r['gene']),
        return_dtype=pl.Float64
    ).alias('ref_pct_detected')
).with_columns(
    (pl.col('ref_pct_detected').is_not_null() &
     (pl.col('ref_pct_detected') >= REF_PCT_THRESHOLD))
    .alias('expressed_in_ref')
)

#endregion
#region save ###################################################################

os.makedirs(f'{working_dir}/output', exist_ok=True)
de_results.write_csv(f'{working_dir}/output/de_results.csv')
de_results\
    .filter((pl.col('FDR') < 0.10) & pl.col('expressed_in_ref'))\
    .write_csv(f'{working_dir}/output/de_results_sig.csv')

for name in datasets:
    df = de_results.filter(pl.col('dataset') == name)
    for contrast in df['contrast'].unique().to_list():
        sub = df.filter(pl.col('contrast') == contrast)
        n_all = sub.filter(pl.col('FDR') < 0.10).height
        n_filt = sub.filter(
            (pl.col('FDR') < 0.10) & pl.col('expressed_in_ref')).height
        n_ct = sub['cell_type'].n_unique()
        print(f'[{name}] {contrast}: {n_ct} cell types, '
              f'{n_filt} DEGs ({n_all} before ref filter)')

#endregion
