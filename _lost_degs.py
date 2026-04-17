"""Are the 701 slidetags DEGs we lost under SoupX contamination-like or real?
Key test: do 'lost' DEGs have worse xenium concordance than 'kept' DEGs?
"""
import polars as pl
import numpy as np

c = pl.read_csv('/home/karbabi/spatial-pregnancy/output/de_results.csv')
d = pl.read_csv('/home/karbabi/spatial-pregnancy/output/de_results_dirty.csv')

def classify(ct):
    if 'NN' in ct or any(x in ct for x in ['Astro', 'Oligo', 'OPC', 'Endo',
                                            'Microglia', 'VLMC', 'Epen',
                                            'Peri', 'SMC', 'Macrophage']):
        return 'NN'
    if 'Glut' in ct:
        return 'Glut'
    return 'Gaba'

cls_map = lambda df: df.with_columns(
    pl.col('cell_type').map_elements(classify, return_dtype=pl.Utf8)
    .alias('cls'))

con = 'PREG_vs_CTRL'
st_d = cls_map(d.filter((pl.col('dataset') == 'slidetags')
                        & (pl.col('contrast') == con)))
st_c = cls_map(c.filter((pl.col('dataset') == 'slidetags')
                        & (pl.col('contrast') == con)))
xn_d = cls_map(d.filter((pl.col('dataset') == 'xenium')
                        & (pl.col('contrast') == con)))
xn_c = cls_map(c.filter((pl.col('dataset') == 'xenium')
                        & (pl.col('contrast') == con)))

# Merge dirty + clean on (cell_type, gene) for slidetags
merged = st_d.rename({
    'logFC': 'lfc_d', 'FDR': 'fdr_d', 'PValue': 'p_d'}).select(
    ['cell_type', 'gene', 'lfc_d', 'fdr_d', 'p_d',
     'ref_pct_detected', 'cls']).join(
    st_c.rename({'logFC': 'lfc_c', 'FDR': 'fdr_c', 'PValue': 'p_c'}).select(
        ['cell_type', 'gene', 'lfc_c', 'fdr_c', 'p_c']),
    on=['cell_type', 'gene'], how='outer_coalesce')

# Categorize: lost (sig d, not c), kept (sig both), new (sig c, not d)
def status(r):
    sd = r['fdr_d'] is not None and r['fdr_d'] < 0.10
    sc = r['fdr_c'] is not None and r['fdr_c'] < 0.10
    if sd and sc: return 'kept'
    if sd and not sc: return 'lost'
    if not sd and sc: return 'new'
    return 'ns'

merged = merged.with_columns(
    pl.struct(['fdr_d', 'fdr_c']).map_elements(
        status, return_dtype=pl.Utf8).alias('status'))

print('='*78)
print('A. STATUS COUNTS (slidetags PREG_vs_CTRL)')
print('='*78)
print(merged.group_by('status').len().sort('len', descending=True))

print()
print('='*78)
print('B. CLASS x STATUS')
print('='*78)
ctab = merged.filter(pl.col('status').is_in(['lost', 'kept', 'new']))\
    .group_by(['cls', 'status']).len()\
    .pivot(index='cls', on='status', values='len', aggregate_function='sum')\
    .fill_null(0).with_columns(
        (pl.col('lost') / (pl.col('lost') + pl.col('kept')))
        .alias('lost_frac'))
print(ctab.sort('cls'))

print()
print('='*78)
print('C. ref_pct_detected BY STATUS (lower = more likely contamination)')
print('='*78)
rp = merged.filter(pl.col('status').is_in(['lost', 'kept', 'new'])
                    & pl.col('ref_pct_detected').is_not_null())
print(rp.group_by(['cls', 'status']).agg(
    pl.col('ref_pct_detected').mean().alias('mean_refpct'),
    pl.col('ref_pct_detected').median().alias('med_refpct'),
    (pl.col('ref_pct_detected') < 10).mean().alias('pct_low_refpct'),
    pl.len().alias('n')).sort(['cls', 'status']))

print()
print('='*78)
print('D. |logFC| (dirty) BY STATUS — were lost DEGs marginal hits?')
print('='*78)
lfc = merged.filter(pl.col('status').is_in(['lost', 'kept']))\
    .with_columns(pl.col('lfc_d').abs().alias('abs_lfc_d'))
print(lfc.group_by(['cls', 'status']).agg(
    pl.col('abs_lfc_d').mean().alias('mean_abs_lfc'),
    pl.col('abs_lfc_d').median().alias('med_abs_lfc'),
    pl.col('p_d').log10().mul(-1).mean().alias('mean_neglog10p'),
    pl.len().alias('n')).sort(['cls', 'status']))

print()
print('='*78)
print('E. KEY TEST: xenium concordance by status')
print('='*78)
print('For slidetags DEGs that have a xenium logFC in the same cell_type,')
print('does lfc_d (dirty slidetags) have same SIGN as lfc_xn (xenium)?')
print('If "lost" DEGs have LOWER sign-concordance, SoupX is removing noise.')
print('If similar or higher, SoupX is removing real biology.')
print()
xn_map = cls_map(xn_d).select(['cell_type', 'gene', 'logFC', 'PValue'])\
    .rename({'logFC': 'lfc_xn', 'PValue': 'p_xn'})
merged_xn = merged.join(xn_map, on=['cell_type', 'gene'], how='inner')

for cls in ['Glut', 'Gaba', 'NN']:
    print(f'\n  [{cls}]')
    for st in ['lost', 'kept', 'new']:
        sub = merged_xn.filter(
            (pl.col('cls') == cls) & (pl.col('status') == st))
        if sub.height < 10:
            print(f'    {st:>5} n={sub.height}  (too few)')
            continue
        lfc_d = sub['lfc_d'].to_numpy()
        lfc_xn = sub['lfc_xn'].to_numpy()
        mask = ~(np.isnan(lfc_d) | np.isnan(lfc_xn))
        if mask.sum() < 10:
            continue
        lfc_d = lfc_d[mask]
        lfc_xn = lfc_xn[mask]
        sign_agree = (np.sign(lfc_d) == np.sign(lfc_xn)).mean()
        pear = np.corrcoef(lfc_d, lfc_xn)[0, 1]
        # stricter: xenium also nominally sig (p<0.05)
        p_xn = sub['p_xn'].to_numpy()[mask]
        strict = (p_xn < 0.05)
        if strict.sum() >= 5:
            strict_agree = (np.sign(lfc_d[strict])
                           == np.sign(lfc_xn[strict])).mean()
        else:
            strict_agree = float('nan')
        print(f'    {st:>5} n={mask.sum():>4}  '
              f'sign_agree={sign_agree:.2%}  '
              f'pearson={pear:+.3f}  '
              f'sign_agree@xn_p<0.05={strict_agree:.2%} '
              f'(n={strict.sum()})')

print()
print('='*78)
print('F. TOP 25 LOST DEGs IN NN (candidate contamination)')
print('='*78)
top_lost_nn = merged.filter(
    (pl.col('status') == 'lost') & (pl.col('cls') == 'NN'))\
    .sort('fdr_d').head(25)
print(top_lost_nn.select(['cell_type', 'gene', 'lfc_d', 'fdr_d',
                          'lfc_c', 'fdr_c', 'ref_pct_detected']))

print()
print('='*78)
print('G. GENE OVERLAP: are lost DEGs recurrent "contamination genes"?')
print('='*78)
print('Genes appearing as "lost" DEG across multiple cell types:')
lost_gene_counts = merged.filter(pl.col('status') == 'lost')\
    .group_by('gene').agg(
        pl.len().alias('n_types'),
        pl.col('cell_type').unique().alias('in_types'),
        pl.col('ref_pct_detected').mean().alias('mean_refpct'),
        pl.col('lfc_d').mean().alias('mean_lfc_d'))\
    .filter(pl.col('n_types') >= 3).sort('n_types', descending=True)
print(lost_gene_counts.head(30))
