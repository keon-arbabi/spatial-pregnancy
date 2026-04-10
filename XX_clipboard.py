import polars as pl
import numpy as np

working_dir = '/home/karbabi/spatial-pregnancy'
de = pl.read_csv(f'{working_dir}/output/de_results.csv')

###############################################################################
# For each FDR<0.10 DEG: show what the other platforms say
###############################################################################

contrast = 'PREG_vs_CTRL'
sub = de.filter(pl.col('contrast') == contrast)

# Get all FDR<0.10 hits
fdr_hits = sub.filter(pl.col('FDR') < 0.10)\
    .select(['gene', 'cell_type', 'dataset', 'logFC', 'PValue', 'FDR',
             'ref_pct_detected'])\
    .sort(['dataset', 'FDR'])

# Get all results (for lookup)
all_results = sub.select(['gene', 'cell_type', 'dataset', 'logFC', 'PValue',
                           'FDR', 'ref_pct_detected'])

# For each FDR hit, find matching gene-cell in other platforms
rows = []
for row in fdr_hits.iter_rows(named=True):
    others = all_results.filter(
        (pl.col('gene') == row['gene']) &
        (pl.col('cell_type') == row['cell_type']) &
        (pl.col('dataset') != row['dataset']))

    other_data = {}
    for o in others.iter_rows(named=True):
        other_data[o['dataset']] = o

    same_dir_any = False
    same_dir_sig = False
    for ds, o in other_data.items():
        if (row['logFC'] > 0) == (o['logFC'] > 0):
            same_dir_any = True
            if o['PValue'] < 0.05:
                same_dir_sig = True

    rows.append({
        'gene': row['gene'],
        'cell_type': row['cell_type'],
        'lead_ds': row['dataset'],
        'lead_lfc': row['logFC'],
        'lead_fdr': row['FDR'],
        'lead_ref': row['ref_pct_detected'],
        'n_other': len(other_data),
        'other_data': other_data,
        'same_dir_any': same_dir_any,
        'same_dir_sig': same_dir_sig,
    })

# --- Summary stats ---
total = len(rows)
has_other = sum(1 for r in rows if r['n_other'] > 0)
same_dir = sum(1 for r in rows if r['same_dir_any'])
same_dir_sig = sum(1 for r in rows if r['same_dir_sig'])
opp_dir_all = sum(1 for r in rows if r['n_other'] > 0 and not r['same_dir_any'])

print(f'=== {contrast}: {total} FDR<0.10 DEGs ===\n')
print(f'  Testable in other platform(s): {has_other}')
print(f'  Same direction in ≥1 other:    {same_dir} '
      f'({same_dir/max(has_other,1)*100:.0f}%)')
print(f'  Same dir + p<0.05 in ≥1:       {same_dir_sig} '
      f'({same_dir_sig/max(has_other,1)*100:.0f}%)')
print(f'  Opposite dir in ALL others:    {opp_dir_all} '
      f'({opp_dir_all/max(has_other,1)*100:.0f}%)')

# --- By lead platform ---
for lead_ds in ['slidetags', 'xenium', 'merfish']:
    ds_rows = [r for r in rows if r['lead_ds'] == lead_ds]
    if not ds_rows:
        continue
    ds_has = sum(1 for r in ds_rows if r['n_other'] > 0)
    ds_same = sum(1 for r in ds_rows if r['same_dir_any'])
    ds_sig = sum(1 for r in ds_rows if r['same_dir_sig'])
    ds_opp = sum(1 for r in ds_rows if r['n_other'] > 0 and not r['same_dir_any'])

    print(f'\n--- {lead_ds} ({len(ds_rows)} FDR DEGs) ---')
    print(f'  Testable: {ds_has}, Same dir: {ds_same} '
          f'({ds_same/max(ds_has,1)*100:.0f}%), '
          f'Same+sig: {ds_sig} ({ds_sig/max(ds_has,1)*100:.0f}%), '
          f'All opposite: {ds_opp} ({ds_opp/max(ds_has,1)*100:.0f}%)')

# --- Print all hits with other-platform detail ---
for lead_ds in ['slidetags', 'xenium', 'merfish']:
    ds_rows = sorted([r for r in rows if r['lead_ds'] == lead_ds],
                     key=lambda r: r['lead_fdr'])
    if not ds_rows:
        continue

    print(f'\n{"=" * 100}')
    print(f'{lead_ds} FDR<0.10 DEGs with cross-platform evidence')
    print(f'{"=" * 100}\n')

    for r in ds_rows:
        dir_str = '↑' if r['lead_lfc'] > 0 else '↓'
        ref_str = f'{r["lead_ref"]:.0f}%' if r['lead_ref'] is not None else '-'

        # Status
        if r['same_dir_sig']:
            status = '✓ REPLICATED'
        elif r['same_dir_any']:
            status = '~ concordant'
        elif r['n_other'] > 0:
            status = '✗ discordant'
        else:
            status = '- no overlap'

        print(f'{dir_str} {r["gene"]:15s} {r["cell_type"]:40s} '
              f'{lead_ds[:2]}:FDR={r["lead_fdr"]:.1e},lfc={r["lead_lfc"]:+.2f} '
              f'ref={ref_str}  {status}')

        for ds in ['merfish', 'slidetags', 'xenium']:
            if ds == lead_ds or ds not in r['other_data']:
                continue
            o = r['other_data'][ds]
            same = (r['lead_lfc'] > 0) == (o['logFC'] > 0)
            arrow = '→' if same else '←'
            sig = '*' if o['PValue'] < 0.05 else ' '
            print(f'    {arrow}{sig} {ds[:2]}: lfc={o["logFC"]:+.3f}, '
                  f'p={o["PValue"]:.1e}, FDR={o["FDR"]:.2f}')
