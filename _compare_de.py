"""Systematic comparison of DE results: corrected vs uncorrected."""
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
    if 'Gaba' in ct or 'GABA' in ct or 'IMN' in ct or 'Chol' in ct:
        return 'Gaba'
    return 'Other'

def enrich(df):
    return df.with_columns(
        pl.col('cell_type').map_elements(classify, return_dtype=pl.Utf8)
        .alias('cls'))

c = enrich(c)
d = enrich(d)

print('='*78)
print('1. OVERALL DEG COUNTS (FDR<0.10)')
print('='*78)
for name, df in [('dirty', d), ('clean', c)]:
    sig = df.filter(pl.col('FDR') < 0.10)
    print(f'  {name}: {sig.height:>7,} / {df.height:>9,} '
          f'({sig.height/df.height:.2%})')

print()
print('='*78)
print('2. DEGs BY dataset x contrast (FDR<0.10)')
print('='*78)
print(f"{'dataset':<10} {'contrast':<20} {'dirty':>8} {'clean':>8} {'Δ':>8} {'pct':>6}")
for ds in ['xenium', 'merfish', 'slidetags']:
    for con in ['PREG_vs_CTRL', 'POSTPART_vs_PREG', 'POSTPART_vs_CTRL']:
        nd = d.filter((pl.col('dataset') == ds) & (pl.col('contrast') == con)
                      & (pl.col('FDR') < 0.10)).height
        nc = c.filter((pl.col('dataset') == ds) & (pl.col('contrast') == con)
                      & (pl.col('FDR') < 0.10)).height
        if nd == 0 and nc == 0:
            continue
        delta = nc - nd
        pct = (delta/nd*100) if nd > 0 else float('inf')
        print(f'{ds:<10} {con:<20} {nd:>8,} {nc:>8,} {delta:>+8,} {pct:>+5.0f}%')

print()
print('='*78)
print('3. DEGs BY cell class (FDR<0.10)')
print('='*78)
print('where contamination was expected to matter (NN > Glut/Gaba)')
print()
print(f"{'dataset':<10} {'class':<6} {'dirty':>8} {'clean':>8} {'Δ':>8} {'pct':>6}")
for ds in ['xenium', 'merfish', 'slidetags']:
    for cls in ['Glut', 'Gaba', 'NN']:
        nd = d.filter((pl.col('dataset') == ds) & (pl.col('cls') == cls)
                      & (pl.col('FDR') < 0.10)).height
        nc = c.filter((pl.col('dataset') == ds) & (pl.col('cls') == cls)
                      & (pl.col('FDR') < 0.10)).height
        if nd == 0 and nc == 0:
            continue
        delta = nc - nd
        pct = (delta/nd*100) if nd > 0 else float('inf')
        print(f'{ds:<10} {cls:<6} {nd:>8,} {nc:>8,} {delta:>+8,} {pct:>+5.0f}%')

print()
print('='*78)
print('4. SIGNIFICANT DEG OVERLAP (direction-preserving)')
print('='*78)
print('For each dataset x contrast: overlap of FDR<0.10 DEGs between dirty/clean')
print()
key_cols = ['dataset', 'contrast', 'cell_type', 'gene']
for ds in ['xenium', 'merfish', 'slidetags']:
    for con in ['PREG_vs_CTRL', 'POSTPART_vs_PREG']:
        dd = d.filter((pl.col('dataset') == ds) & (pl.col('contrast') == con)
                      & (pl.col('FDR') < 0.10))
        cc = c.filter((pl.col('dataset') == ds) & (pl.col('contrast') == con)
                      & (pl.col('FDR') < 0.10))
        if dd.height == 0 and cc.height == 0:
            continue
        dkey = dd.select(key_cols + ['logFC']).rename({'logFC': 'lfc_d'})
        ckey = cc.select(key_cols + ['logFC']).rename({'logFC': 'lfc_c'})
        inter = dkey.join(ckey, on=key_cols, how='inner')
        same_sign = inter.filter(
            pl.col('lfc_d') * pl.col('lfc_c') > 0).height
        only_d = dd.height - inter.height
        only_c = cc.height - inter.height
        flipped = inter.height - same_sign
        print(f'  {ds:<10} {con:<20} '
              f'dirty={dd.height:,} clean={cc.height:,} '
              f'both={inter.height:,} (same_sign={same_sign:,}, '
              f'flipped={flipped}) '
              f'only_d={only_d:,} only_c={only_c:,}')

print()
print('='*78)
print('5. CROSS-PLATFORM CONCORDANCE (slidetags vs xenium, PREG_vs_CTRL)')
print('='*78)
print('Overlap of (cell_type, gene) DEGs (same cell type; FDR<0.10 in both)')
print()
for name, df in [('dirty', d), ('clean', c)]:
    st = df.filter((pl.col('dataset') == 'slidetags')
                   & (pl.col('contrast') == 'PREG_vs_CTRL')
                   & (pl.col('FDR') < 0.10))
    xn = df.filter((pl.col('dataset') == 'xenium')
                   & (pl.col('contrast') == 'PREG_vs_CTRL')
                   & (pl.col('FDR') < 0.10))
    inter = st.select(['cell_type', 'gene', 'logFC']).rename(
        {'logFC': 'lfc_st'}).join(
        xn.select(['cell_type', 'gene', 'logFC']).rename(
            {'logFC': 'lfc_xn'}),
        on=['cell_type', 'gene'], how='inner')
    same = inter.filter(pl.col('lfc_st') * pl.col('lfc_xn') > 0).height
    flipped = inter.height - same
    print(f'  {name}: st={st.height:,} xn={xn.height:,} '
          f'shared={inter.height:,} (same_sign={same:,}, flipped={flipped})')

print()
print('='*78)
print('6. CROSS-PLATFORM LOGFC CORRELATION (slidetags vs xenium, PREG_vs_CTRL)')
print('='*78)
print('On union of genes tested in both; by cell class')
print()
for name, df in [('dirty', d), ('clean', c)]:
    st = df.filter((pl.col('dataset') == 'slidetags')
                   & (pl.col('contrast') == 'PREG_vs_CTRL'))\
        .select(['cell_type', 'gene', 'logFC', 'PValue', 'cls'])\
        .rename({'logFC': 'lfc_st', 'PValue': 'p_st'})
    xn = df.filter((pl.col('dataset') == 'xenium')
                   & (pl.col('contrast') == 'PREG_vs_CTRL'))\
        .select(['cell_type', 'gene', 'logFC', 'PValue'])\
        .rename({'logFC': 'lfc_xn', 'PValue': 'p_xn'})
    m = st.join(xn, on=['cell_type', 'gene'], how='inner')
    print(f'  {name} (n={m.height:,}):')
    for cls in ['Glut', 'Gaba', 'NN']:
        sub = m.filter(pl.col('cls') == cls)
        if sub.height < 10:
            continue
        st_v = sub['lfc_st'].to_numpy()
        xn_v = sub['lfc_xn'].to_numpy()
        pear = np.corrcoef(st_v, xn_v)[0, 1]
        spear = pl.DataFrame({'a': st_v, 'b': xn_v}).select(
            pl.corr('a', 'b', method='spearman'))[0, 0]
        sign_agree = (np.sign(st_v) == np.sign(xn_v)).mean()
        top = sub.filter(
            (pl.col('p_st') < 0.05) & (pl.col('p_xn') < 0.05))
        top_sign = ((top['lfc_st'].to_numpy() * top['lfc_xn'].to_numpy())
                    > 0).mean() if top.height > 0 else float('nan')
        print(f'    {cls:<5} n={sub.height:>7,} '
              f'pearson={pear:+.3f} spearman={spear:+.3f} '
              f'sign_agree={sign_agree:.2%} '
              f'sign_agree@p<0.05_both={top_sign:.2%} (n={top.height})')

print()
print('='*78)
print('7. REF_PCT_DETECTED: are DEGs more cell-type-appropriate after?')
print('='*78)
print('Mean ref_pct for significant DEGs (higher = more specific/expressed)')
print()
print(f"{'dataset':<10} {'class':<6} {'dirty':>8} {'clean':>8} {'Δ':>8}")
for ds in ['xenium', 'merfish', 'slidetags']:
    for cls in ['Glut', 'Gaba', 'NN']:
        dsub = d.filter((pl.col('dataset') == ds) & (pl.col('cls') == cls)
                        & (pl.col('FDR') < 0.10)
                        & pl.col('ref_pct_detected').is_not_null())
        csub = c.filter((pl.col('dataset') == ds) & (pl.col('cls') == cls)
                        & (pl.col('FDR') < 0.10)
                        & pl.col('ref_pct_detected').is_not_null())
        if dsub.height == 0 and csub.height == 0:
            continue
        dm = dsub['ref_pct_detected'].mean() if dsub.height > 0 else np.nan
        cm = csub['ref_pct_detected'].mean() if csub.height > 0 else np.nan
        delta = cm - dm if not (np.isnan(cm) or np.isnan(dm)) else np.nan
        print(f'{ds:<10} {cls:<6} {dm:>8.2f} {cm:>8.2f} {delta:>+8.2f}')

print()
print('='*78)
print('8. LOW REF_PCT DEGS (suspected contamination-driven)')
print('='*78)
print('Fraction of sig DEGs with ref_pct < 10 (= gene barely expressed in that cell type)')
print()
print(f"{'dataset':<10} {'class':<6} {'dirty':>12} {'clean':>12}")
for ds in ['xenium', 'merfish', 'slidetags']:
    for cls in ['Glut', 'Gaba', 'NN']:
        def frac(df):
            sub = df.filter(
                (pl.col('dataset') == ds) & (pl.col('cls') == cls)
                & (pl.col('FDR') < 0.10)
                & pl.col('ref_pct_detected').is_not_null())
            if sub.height == 0:
                return None
            low = sub.filter(pl.col('ref_pct_detected') < 10).height
            return (low, sub.height, low/sub.height)
        fd = frac(d)
        fc = frac(c)
        if fd is None and fc is None:
            continue
        fd_s = f'{fd[0]:,}/{fd[1]:,} ({fd[2]:.1%})' if fd else '-'
        fc_s = f'{fc[0]:,}/{fc[1]:,} ({fc[2]:.1%})' if fc else '-'
        print(f'{ds:<10} {cls:<6} {fd_s:>12} {fc_s:>12}')

print()
print('='*78)
print('9. XENIUM: NN TYPES PER-SUBCLASS (where contamination was highest)')
print('='*78)
print('Sig DEG counts by NN subclass in xenium PREG_vs_CTRL')
print()
for ct in sorted(c.filter(pl.col('cls') == 'NN')['cell_type'].unique()
                 .to_list()):
    ds = 'xenium'
    con = 'PREG_vs_CTRL'
    nd = d.filter((pl.col('dataset') == ds) & (pl.col('contrast') == con)
                  & (pl.col('cell_type') == ct)
                  & (pl.col('FDR') < 0.10)).height
    nc = c.filter((pl.col('dataset') == ds) & (pl.col('contrast') == con)
                  & (pl.col('cell_type') == ct)
                  & (pl.col('FDR') < 0.10)).height
    if nd + nc == 0:
        continue
    total_d = d.filter((pl.col('dataset') == ds) & (pl.col('contrast') == con)
                       & (pl.col('cell_type') == ct)).height
    # mean ref_pct of sig DEGs
    rd = d.filter((pl.col('dataset') == ds) & (pl.col('contrast') == con)
                  & (pl.col('cell_type') == ct) & (pl.col('FDR') < 0.10))\
        ['ref_pct_detected'].mean()
    rc = c.filter((pl.col('dataset') == ds) & (pl.col('contrast') == con)
                  & (pl.col('cell_type') == ct) & (pl.col('FDR') < 0.10))\
        ['ref_pct_detected'].mean()
    rd_s = f'{rd:.1f}' if rd is not None else '-'
    rc_s = f'{rc:.1f}' if rc is not None else '-'
    print(f'  {ct[:50]:<52} '
          f'dirty={nd:>4} (refpct={rd_s}) '
          f'clean={nc:>4} (refpct={rc_s}) '
          f'Δ={nc-nd:+}')
