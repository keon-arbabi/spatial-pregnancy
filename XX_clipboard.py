import numpy as np
import scanpy as sc

working_dir = '/home/karbabi/spatial-pregnancy'

datasets = {
    'slidetags': f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
    'merfish':   f'{working_dir}/output/merfish/03_adata_query_merfish.h5ad',
    'xenium':    f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
}

for ds_name, path in datasets.items():
    print('=' * 110)
    print(f'{ds_name.upper()}')
    print('=' * 110)

    adata = sc.read_h5ad(path)
    samples = sorted(adata.obs['sample'].unique())
    subclasses = sorted(adata.obs['subclass'].unique())

    print(f'\n{"sample":14s} {"n_cells":>8s} {"med_cts":>8s} {"mean_cts":>9s} '
          f'{"q10":>6s} {"q25":>6s} {"q75":>6s} {"q90":>6s}')
    for s in samples:
        mask = adata.obs['sample'] == s
        cts = np.asarray(adata[mask].X.sum(axis=1)).ravel()
        print(f'{s:14s} {mask.sum():>8d} '
              f'{np.median(cts):>8.0f} {np.mean(cts):>9.0f} '
              f'{np.quantile(cts, 0.10):>6.0f} '
              f'{np.quantile(cts, 0.25):>6.0f} '
              f'{np.quantile(cts, 0.75):>6.0f} '
              f'{np.quantile(cts, 0.90):>6.0f}')

    print(f'\n--- Per-sample "extreme" fraction across subclasses (n>=30) ---\n')
    print(f'{"sample":14s} {"n_sub":>6s} {"low":>6s} {"high":>6s} {"extreme":>8s}')

    summary = {}
    for target in samples:
        low_count, high_count, total = 0, 0, 0
        for sub in subclasses:
            row = {}
            for s in samples:
                mask = (adata.obs['subclass'] == sub) & \
                       (adata.obs['sample'] == s)
                if mask.sum() < 30:
                    row[s] = np.nan
                    continue
                cts = np.asarray(adata[mask].X.sum(axis=1)).ravel()
                row[s] = np.median(cts)
            if np.isnan(row.get(target, np.nan)):
                continue
            others = [v for k, v in row.items()
                      if k != target and not np.isnan(v)]
            if len(others) < 2:
                continue
            med_oth = np.median(others)
            if med_oth == 0:
                continue
            ratio = row[target] / med_oth
            total += 1
            if ratio < 0.7:
                low_count += 1
            elif ratio > 1.3:
                high_count += 1
        extreme = low_count + high_count
        summary[target] = {'n': total, 'low': low_count,
                           'high': high_count, 'extreme': extreme}
        ext_pct = extreme / total * 100 if total > 0 else 0
        low_pct = low_count / total * 100 if total > 0 else 0
        high_pct = high_count / total * 100 if total > 0 else 0
        print(f'{target:14s} {total:>6d} '
              f'{low_count:>3d}({low_pct:>2.0f}%) '
              f'{high_count:>3d}({high_pct:>2.0f}%) '
              f'{extreme:>3d}({ext_pct:>2.0f}%)')

    max_extreme = max(v['extreme'] / v['n'] * 100 if v['n'] > 0 else 0
                       for v in summary.values())
    med_extreme = np.median([v['extreme'] / v['n'] * 100 if v['n'] > 0 else 0
                              for v in summary.values()])
    print(f'\n  Max extreme %: {max_extreme:.0f}%')
    print(f'  Median extreme %: {med_extreme:.0f}%')
    worst = max(summary.items(),
                key=lambda x: x[1]['extreme'] / max(x[1]['n'], 1))
    w_low = worst[1]['low']
    w_high = worst[1]['high']
    w_bidir = min(w_low, w_high) / max(w_low, w_high, 1)
    print(f'  Worst sample: {worst[0]} '
          f'(low={w_low}, high={w_high}, '
          f'bidirectionality={w_bidir:.2f})')

    print()
    del adata
