"""
Test whether per-A eBayes moderation is biased by fit size.

variancePartition isn't installed locally; use limma directly on
sample-pseudobulked CLR values. The *question* (does eBayes variance-
borrowing across B features within a per-A fit systematically bias the
t-statistics?) applies regardless of whether the underlying fit is dream
or limma — eBayes is the same routine either way.

For each center cell type A:
  1. Aggregate cell-level (b_count, all_count) to sample level via sum.
  2. CLR at sample level: 0.5*(log(b+0.5) - log((all-b)+0.5)).
  3. Compute feature weights per crumblr (4*n*p*(1-p)).
  4. limma::lmFit with weights; contrasts.fit; then both:
       - unmoderated classical t = coef / (stdev.unscaled * sigma)
       - eBayes moderated t
  5. Export both for comparison.

Dataset: xenium PREG_vs_CTRL (strongest n_B/|LFC| coupling, so strongest
expected bias signal).
"""
import os, warnings, numpy as np, pandas as pd
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r, to_py
from tqdm.auto import tqdm
from statsmodels.stats.multitest import fdrcorrection
warnings.filterwarnings('ignore')

working_dir = '/home/karbabi/spatial-pregnancy'
MIN_NONZERO = 5
name = 'xenium'
contrast_pair = ('PREG', 'CTRL')
contrast_name = f'{contrast_pair[0]}_vs_{contrast_pair[1]}'

print(f'[{name}] loading cached spatial stats', flush=True)
S = pd.read_pickle(f'{working_dir}/output/{name}/spatial_stats.pkl')

# same pair filter as production
pair_filter = (
    S.groupby(['sample_id','cell_type_a','cell_type_b'], observed=True)
     ['b_count'].apply(lambda x: int((x > 0).sum()))
     .groupby(['cell_type_a','cell_type_b']).min())
valid_pairs = pair_filter[pair_filter >= MIN_NONZERO].index.tolist()
valid_b_by_a = {}
for a, b in valid_pairs:
    valid_b_by_a.setdefault(a, []).append(b)
print(f'  {len(valid_pairs)} pairs across {len(valid_b_by_a)} A-types',
      flush=True)

r('''
suppressPackageStartupMessages({ library(limma) })

classical_stats <- function(cf, coef_name) {
    b  <- cf$coefficients[, coef_name]
    su <- cf$stdev.unscaled[, coef_name]
    sg <- cf$sigma
    df <- cf$df.residual
    t_raw <- b / (su * sg)
    p_raw <- 2 * pt(-abs(t_raw), df = df)
    data.frame(feature = rownames(cf$coefficients),
               logFC_raw = b, t_raw = t_raw, p_raw = p_raw,
               df_res = df, sigma = sg,
               stringsAsFactors = FALSE)
}
''')

groups_by_a = S.groupby('cell_type_a', observed=True, sort=False)
mod_rows, raw_rows, diag_rows = [], [], []

for ct_a in tqdm(sorted(valid_b_by_a.keys()), desc=f'[{name}] per-A'):
    if ct_a not in groups_by_a.groups:
        continue
    sub = groups_by_a.get_group(ct_a)
    valid_b = valid_b_by_a[ct_a]

    # Sample-level aggregation: sum b_count and all_count over cells of A
    # within each (sample_id, cell_type_b). all_count is the per-cell
    # total neighbors; summing over A-cells in a sample gives the total
    # edges from A-cells of that sample to any neighbor. b_count summed
    # over A-cells gives total A→B edges in that sample.
    agg = (sub.groupby(['sample_id','condition','cell_type_b'],
                        observed=True)
              .agg(b_count=('b_count','sum'),
                   all_count=('all_count','sum'))
              .reset_index())
    agg = agg[agg['condition'].astype(str).isin(list(contrast_pair))]
    agg = agg[agg['cell_type_b'].astype(str).isin(
                [str(x) for x in valid_b])]
    if agg.empty:
        continue

    b_piv = agg.pivot(index='cell_type_b', columns='sample_id',
                      values='b_count').fillna(0.0)
    # all_count is per A-cell total; the denominator for CLR is
    # the total edges from A-cells in that sample. Aggregate separately:
    total_edges = (sub.drop_duplicates('cell_id')
                      .groupby(['sample_id','condition'], observed=True)
                      ['all_count'].sum().reset_index())
    total_edges = total_edges[
        total_edges['condition'].astype(str).isin(list(contrast_pair))]
    samp_cond = (total_edges.set_index('sample_id')
                 [['condition']].astype(str))
    # align columns
    samples = [s for s in b_piv.columns if s in samp_cond.index]
    if len(samples) < 3:
        continue
    b_piv = b_piv[samples]
    all_vec = total_edges.set_index('sample_id').loc[samples,
                                                      'all_count'].values
    b_arr = b_piv.values.astype(np.float64)
    all_row = all_vec[None, :]
    other_arr = np.maximum(all_row - b_arr, 0.0)
    # CLR at sample level per (B feature, sample)
    clr_matrix = 0.5 * (np.log(b_arr + 0.5) - np.log(other_arr + 0.5))

    # crumblr-style precision weights
    n_total = all_row + 1.0
    p = (b_arr + 0.5) / n_total
    w = 4.0 * n_total * p * (1.0 - p)
    weights_matrix = w.astype(np.float64)

    feature_names = [str(x) for x in b_piv.index]
    conditions = samp_cond.loc[samples, 'condition'].values
    # require both levels present
    if not set(contrast_pair).issubset(set(conditions)):
        continue
    if clr_matrix.shape[0] < 2:
        continue

    try:
        to_r(clr_matrix, 'clr_matrix')
        to_r(weights_matrix, 'weights_matrix')
        to_r(feature_names, 'feature_names')
        to_r(list(samples), 'sample_ids')
        to_r(pd.DataFrame({'sample_id': samples,
                           'condition': conditions}),
             'meta', format='data.frame')
        to_r(f'condition{contrast_pair[0]} - condition{contrast_pair[1]}',
             'contrast_str')
        r('''
        rownames(clr_matrix) <- feature_names
        colnames(clr_matrix) <- sample_ids
        rownames(weights_matrix) <- feature_names
        colnames(weights_matrix) <- sample_ids
        rownames(meta) <- sample_ids
        meta$condition <- factor(meta$condition)

        design <- model.matrix(~ 0 + condition, data = meta)
        colnames(design) <- sub("^condition", "condition",
                                colnames(design))
        L <- makeContrasts(contrasts = contrast_str, levels = design)

        fit  <- lmFit(clr_matrix, design, weights = weights_matrix)
        cf   <- contrasts.fit(fit, L)
        coef_name <- colnames(cf$coefficients)[1]

        raw_df <- classical_stats(cf, coef_name)
        eb     <- eBayes(cf)
        tt_mod <- topTable(eb, coef = coef_name, number = Inf,
                           sort.by = "none")
        tt_mod$feature <- rownames(tt_mod)

        diag_df <- data.frame(
            s2_prior = as.numeric(eb$s2.prior[1]),
            df_prior = as.numeric(eb$df.prior[1]),
            n_features = nrow(cf$coefficients),
            n_samples  = ncol(clr_matrix),
            df_residual_median = as.numeric(
                stats::median(cf$df.residual)),
            stringsAsFactors = FALSE)
        ''')
        raw_df = to_py('raw_df', format='pandas')
        mod_df = to_py('tt_mod', format='pandas')
        diag_df = to_py('diag_df', format='pandas')

        raw_df['cell_type_a'] = ct_a
        mod_df['cell_type_a'] = ct_a
        diag_df['cell_type_a'] = ct_a
        raw_rows.append(raw_df)
        mod_rows.append(mod_df)
        diag_rows.append(diag_df)
    except Exception as e:
        print(f'  {ct_a}: {type(e).__name__}: {e}', flush=True)

raw_all = pd.concat(raw_rows, ignore_index=True)
mod_all = pd.concat(mod_rows, ignore_index=True)
diag_all = pd.concat(diag_rows, ignore_index=True)

raw_all = raw_all.rename(columns={'feature':'cell_type_b'})
mod_all = mod_all.rename(columns={
    'feature':'cell_type_b', 'P.Value':'p_mod',
    't':'t_mod', 'logFC':'logFC_mod', 'adj.P.Val':'q_mod'})

merged = raw_all.merge(
    mod_all[['cell_type_a','cell_type_b','logFC_mod','t_mod','p_mod',
             'q_mod']],
    on=['cell_type_a','cell_type_b'])

merged['q_raw'] = fdrcorrection(merged['p_raw'].fillna(1.0))[1]
# Recompute moderated FDR on merged subset for fair comparison
merged['q_mod_recalc'] = fdrcorrection(merged['p_mod'].fillna(1.0))[1]

out_path = f'{working_dir}/output/_ebayes_test_{name}_{contrast_name}.csv'
merged.to_csv(out_path, index=False)
diag_all.to_csv(
    f'{working_dir}/output/_ebayes_test_{name}_{contrast_name}_diag.csv',
    index=False)
print(f'saved {merged.shape[0]:,} rows → {out_path}', flush=True)
