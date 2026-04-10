import polars as pl
import numpy as np

working_dir = '/home/karbabi/spatial-pregnancy'
de = pl.read_csv(f'{working_dir}/output/de_results.csv')
preg = de.filter(pl.col('contrast') == 'PREG_vs_CTRL')

# --- 1. Overview ---
print('=== 1. Overview (with log2 num_cells + log2 lib_size covariates) ===\n')
for ds in ['slidetags', 'xenium', 'merfish']:
    sub = preg.filter(pl.col('dataset') == ds)
    sig = sub.filter(pl.col('FDR') < 0.10)
    if sig.height == 0:
        print(f'{ds}: 0 DEGs')
        continue
    up = (sig['logFC'] > 0).sum()
    down = sig.height - up
    pct_up = up / sig.height * 100
    n_ct = sig['cell_type'].n_unique()
    print(f'{ds:10s} {sig.height:>5d} DEGs ({up} up, {down} down, '
          f'{pct_up:.0f}% up) in {n_ct} cell types')

# --- 2. Conditional concordance ---
print('\n=== 2. Conditional concordance ===\n')

pairs = [('slidetags', 'xenium', 'st', 'xn')]
for ds1, ds2, lab1, lab2 in pairs:
    d1 = preg.filter((pl.col('dataset') == ds1) & (pl.col('PValue') < 0.05))\
        .select(['gene', 'cell_type', pl.col('logFC').alias('lfc1')])
    d2 = preg.filter(pl.col('dataset') == ds2)\
        .select(['gene', 'cell_type', pl.col('logFC').alias('lfc2')])
    joined = d1.join(d2, on=['gene', 'cell_type'])
    if joined.height > 0:
        x, y = joined['lfc1'].to_numpy(), joined['lfc2'].to_numpy()
        conc = (x * y > 0).sum() / len(x) * 100
        up_mask = x > 0
        dn_mask = x < 0
        up_c = (y[up_mask] > 0).sum() / up_mask.sum() * 100
        dn_c = (y[dn_mask] < 0).sum() / dn_mask.sum() * 100
        print(f'{lab1} sig→{lab2}: n={joined.height}, conc={conc:.0f}% '
              f'(up:{up_c:.0f}%, down:{dn_c:.0f}%)')

    # Reverse
    d1r = preg.filter(pl.col('dataset') == ds1)\
        .select(['gene', 'cell_type', pl.col('logFC').alias('lfc1')])
    d2r = preg.filter((pl.col('dataset') == ds2) & (pl.col('PValue') < 0.05))\
        .select(['gene', 'cell_type', pl.col('logFC').alias('lfc2')])
    joined_r = d1r.join(d2r, on=['gene', 'cell_type'])
    if joined_r.height > 0:
        xr, yr = joined_r['lfc1'].to_numpy(), joined_r['lfc2'].to_numpy()
        conc_r = (xr * yr > 0).sum() / len(xr) * 100
        print(f'{lab2} sig→{lab1}: n={joined_r.height}, conc={conc_r:.0f}%')

# --- 3. Replication rate ---
print('\n=== 3. Replication rate (FDR<0.10 in A, p<0.05 same dir in B) ===\n')

for ds1, ds2, lab1, lab2 in pairs:
    degs = preg.filter(
        (pl.col('dataset') == ds1) & (pl.col('FDR') < 0.10))\
        .select(['gene', 'cell_type', pl.col('logFC').alias('lfc1')])
    d2 = preg.filter(pl.col('dataset') == ds2)\
        .select(['gene', 'cell_type',
                 pl.col('logFC').alias('lfc2'),
                 pl.col('PValue').alias('p2')])
    joined = degs.join(d2, on=['gene', 'cell_type'])
    if joined.height > 0:
        repl = joined.filter(
            (pl.col('p2') < 0.05) &
            ((pl.col('lfc1') * pl.col('lfc2')) > 0)).height
        anti = joined.filter(
            (pl.col('p2') < 0.05) &
            ((pl.col('lfc1') * pl.col('lfc2')) < 0)).height
        print(f'{lab1}→{lab2}: {degs.height} DEGs, '
              f'{joined.height} testable, '
              f'{repl} replicated ({repl/max(joined.height,1)*100:.1f}%), '
              f'{anti} anti ({anti/max(joined.height,1)*100:.1f}%)')

# --- 4. Top cell types ---
print('\n=== 4. Top cell types by DEG count ===\n')
for ds in ['slidetags', 'xenium']:
    top = preg.filter((pl.col('dataset') == ds) & (pl.col('FDR') < 0.10))\
        .group_by('cell_type').agg(
            pl.len().alias('n'),
            (pl.col('logFC') > 0).sum().alias('up'),
            (pl.col('logFC') < 0).sum().alias('down'),
        ).sort('n', descending=True).head(10)
    print(f'  {ds}:')
    for row in top.iter_rows(named=True):
        print(f'    {row["cell_type"]:45s} {row["n"]:>4d} '
              f'({row["up"]} up, {row["down"]} down)')
    print()

# --- 5. Compare with previous run ---
print('=== 5. Comparison to previous model ===')
print('(paste previous values manually for comparison)\n')
print('Previous (log_num_cells only):')
print('  slidetags: 1553 DEGs, 36% up')
print('  xenium: 12150 DEGs, 41% up')
print('  st sig→xn: conc=50%')
print('  xn sig→st: conc=51%')
print('  st→xn replication: 44/590 (7.5%)')
print()
print('Current (log2_num_cells + log2_lib_size):')
# Will be filled in by the run
