import polars as pl
import numpy as np
from scipy.stats import pearsonr

working_dir = '/home/karbabi/spatial-pregnancy'
de = pl.read_csv(f'{working_dir}/output/de_results.csv')
preg = de.filter(pl.col('contrast') == 'PREG_vs_CTRL')

# --- 1. What does ref_pct_detected actually measure? ---
# The scRNA reference is male, adult, non-pregnant.
# Low ref_pct could mean:
#   (a) Gene truly not expressed in that cell type → spillover artifact
#   (b) Gene expressed but pregnancy-induced → absent in male reference
#   (c) Gene expressed at low level, below scRNA detection → real but sparse
#   (d) Gene expressed but subclass boundaries differ between ref and query

print('=== 1. Pregnancy-relevant genes at risk of being filtered ===\n')

# Genes we KNOW are biologically relevant to pregnancy
pregnancy_genes = {
    # Hormone receptors upregulated during pregnancy
    'Esr1': 'estrogen receptor, MPOA priming',
    'Pgr': 'progesterone receptor, broadly upregulated',
    'Prlr': 'prolactin receptor, MPOA/LSX',
    'Oxtr': 'oxytocin receptor, maternal behavior',
    'Lepr': 'leptin receptor, HPA axis',
    'Ar': 'androgen receptor',
    # Neuropeptides
    'Gal': 'galanin, MPOA Lhx8+ neurons',
    'Oxt': 'oxytocin',
    'Crh': 'CRH, stress axis',
    'Prl': 'prolactin (secreted)',
    'Gnrh1': 'GnRH',
    'Cartpt': 'CART peptide',
    'Nts': 'neurotensin',
    'Tac2': 'tachykinin/NkB',
    'Sst': 'somatostatin',
    'Trh': 'TRH',
    'Pdyn': 'prodynorphin',
    'Penk': 'proenkephalin',
    # Trophic/vascular
    'Vegfa': 'VEGF-A, angiogenesis',
    'Bdnf': 'BDNF, neuroplasticity',
    'Ntrk2': 'TrkB receptor',
    'Igf1': 'IGF1',
    'Igf2': 'IGF2',
    'Fgf14': 'FGF14',
    'Kdr': 'VEGFR2, endothelial',
    'Flt4': 'VEGFR3, endothelial',
    'Flt1': 'VEGFR1',
    # Stress/glucocorticoid
    'Nr3c1': 'glucocorticoid receptor',
    'Nr3c2': 'mineralocorticoid receptor',
    'Fkbp5': 'GR co-chaperone',
    'Sgk1': 'GR target',
    # Key DEGs from manuscript
    'Ckb': 'creatine kinase, neuronal suppression',
    'Tshz2': 'TF, neuronal identity',
    'Cnr1': 'CB1 receptor',
    'Hif3a': 'hypoxia factor, oligo',
    'Ptgds': 'prostaglandin synthase',
    'Ccnd3': 'cyclin D3, microglia',
    'Mfsd2a': 'BBB transporter',
}

print(f'{"gene":10s} {"role":40s} {"mf_ref%":>8s} {"st_ref%":>8s} '
      f'{"xn_ref%":>8s} {"at_risk":>8s}')
print('-' * 85)

at_risk_genes = []
for gene, role in pregnancy_genes.items():
    refs = {}
    for ds in ['merfish', 'slidetags', 'xenium']:
        sub = preg.filter((pl.col('dataset') == ds) &
                          (pl.col('gene') == gene) &
                          pl.col('ref_pct_detected').is_not_null())
        if sub.height > 0:
            # median ref_pct across cell types where it's tested
            refs[ds] = sub['ref_pct_detected'].median()
        else:
            refs[ds] = None

    # Check if any significant hits would be lost
    hits_lost = 0
    hits_kept = 0
    for ds in ['merfish', 'slidetags', 'xenium']:
        sub = preg.filter((pl.col('dataset') == ds) &
                          (pl.col('gene') == gene) &
                          (pl.col('PValue') < 0.05))
        lost = sub.filter(
            pl.col('ref_pct_detected').is_null() |
            (pl.col('ref_pct_detected') < 10)).height
        kept = sub.filter(
            pl.col('ref_pct_detected').is_not_null() &
            (pl.col('ref_pct_detected') >= 10)).height
        hits_lost += lost
        hits_kept += kept

    at_risk = 'YES' if hits_lost > hits_kept and hits_lost > 0 else ''
    if at_risk:
        at_risk_genes.append(gene)

    def fmt(v):
        return f'{v:>7.1f}%' if v is not None else '      -'

    print(f'{gene:10s} {role:40s} {fmt(refs.get("merfish"))} '
          f'{fmt(refs.get("slidetags"))} {fmt(refs.get("xenium"))} '
          f'{at_risk:>8s}')
    if at_risk:
        # Show what would be lost
        for ds in ['merfish', 'slidetags', 'xenium']:
            lost = preg.filter(
                (pl.col('dataset') == ds) &
                (pl.col('gene') == gene) &
                (pl.col('PValue') < 0.05) &
                ((pl.col('ref_pct_detected') < 10) |
                 pl.col('ref_pct_detected').is_null()))
            if lost.height > 0:
                for row in lost.sort('PValue').head(3).iter_rows(named=True):
                    ref = row['ref_pct_detected']
                    ref_str = f'{ref:.1f}%' if ref is not None else 'null'
                    print(f'  LOST: {ds:10s} {row["cell_type"]:40s} '
                          f'logFC={row["logFC"]:+.2f} p={row["PValue"]:.1e} '
                          f'ref={ref_str}')

print(f'\n{len(at_risk_genes)} genes at risk: {at_risk_genes}')

# --- 2. Distribution of ref_pct for all significant hits ---
print('\n=== 2. Where do significant DEGs fall in ref_pct? ===\n')

for ds in ['slidetags', 'merfish', 'xenium']:
    sig = preg.filter((pl.col('dataset') == ds) & (pl.col('PValue') < 0.05) &
                       pl.col('ref_pct_detected').is_not_null())
    pcts = sig['ref_pct_detected'].to_numpy()
    bins = [0, 1, 5, 10, 20, 50, 100]
    counts = np.histogram(pcts, bins=bins)[0]
    print(f'{ds}:')
    cumulative_lost = 0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        n = counts[i]
        cumulative_lost += n
        up = sig.filter(
            (pl.col('ref_pct_detected') >= lo) &
            (pl.col('ref_pct_detected') < hi) &
            (pl.col('logFC') > 0)).height
        pct_up = up / n * 100 if n > 0 else 0
        print(f'  ref {lo:>3d}-{hi:<3d}%: {n:>5d} hits ({pct_up:.0f}% up)  '
              f'cumulative filtered: {cumulative_lost}')
    print()

# --- 3. Case studies: genes with mixed ref_pct across cell types ---
print('=== 3. Genes with genuine expression in SOME cell types '
      'but not others ===\n')
print('(gene has sig hits in both ref≥10% and ref<10% cell types)\n')

for ds in ['slidetags', 'xenium']:
    sig = preg.filter((pl.col('dataset') == ds) & (pl.col('PValue') < 0.01) &
                       pl.col('ref_pct_detected').is_not_null())
    gene_split = sig.group_by('gene').agg(
        (pl.col('ref_pct_detected') >= 10).sum().alias('n_native'),
        (pl.col('ref_pct_detected') < 10).sum().alias('n_spillover'),
        pl.len().alias('n_total')
    ).filter((pl.col('n_native') > 0) & (pl.col('n_spillover') > 0))\
        .sort('n_spillover', descending=True)

    print(f'{ds}: {gene_split.height} genes with mixed native/spillover hits')
    for row in gene_split.head(15).iter_rows(named=True):
        gene = row['gene']
        native = sig.filter(
            (pl.col('gene') == gene) & (pl.col('ref_pct_detected') >= 10))
        spill = sig.filter(
            (pl.col('gene') == gene) & (pl.col('ref_pct_detected') < 10))
        nat_dir = '+' if (native['logFC'] > 0).sum() > native.height / 2 else '-'
        spl_dir = '+' if (spill['logFC'] > 0).sum() > spill.height / 2 else '-'
        print(f'  {gene:15s} native: {row["n_native"]:>2d} ({nat_dir})  '
              f'spillover: {row["n_spillover"]:>2d} ({spl_dir})  '
              f'{"SAME" if nat_dir == spl_dir else "OPPOSITE"} direction')
    print()

# --- 4. Neuropeptides specifically ---
# These are a concern because many neuropeptides are expressed at low levels
# in the scRNA reference but ARE genuinely expressed (MERFISH was designed
# to detect them)
print('=== 4. Neuropeptide ref_pct by cell type (would filter remove '
      'real biology?) ===\n')

neuropeptides = ['Gal', 'Sst', 'Nts', 'Cartpt', 'Tac2', 'Crh', 'Oxt',
                 'Trh', 'Pdyn', 'Penk', 'Prl']

for gene in neuropeptides:
    # Show all cell types where this gene is sig in any platform
    sig = preg.filter((pl.col('gene') == gene) & (pl.col('PValue') < 0.05))
    if sig.height == 0:
        continue
    # Get unique cell types
    cts = sig['cell_type'].unique().to_list()
    print(f'--- {gene} ({len(cts)} cell types with p<0.05 in any platform) ---')
    for ct in sorted(cts)[:8]:
        row_data = {}
        for ds in ['merfish', 'slidetags', 'xenium']:
            sub = preg.filter(
                (pl.col('dataset') == ds) &
                (pl.col('gene') == gene) &
                (pl.col('cell_type') == ct))
            if sub.height > 0:
                r = sub.row(0, named=True)
                ref = r['ref_pct_detected']
                ref_str = f'{ref:.0f}%' if ref is not None else '-'
                p_str = f'p={r["PValue"]:.0e}' if r['PValue'] < 0.05 else ''
                row_data[ds] = (f'{r["logFC"]:+.2f}', ref_str, p_str)
            else:
                row_data[ds] = ('', '', '')
        ct_short = ct[:42]
        mf = row_data.get('merfish', ('','',''))
        st = row_data.get('slidetags', ('','',''))
        xn = row_data.get('xenium', ('','',''))
        ref_val = None
        for ds in ['merfish', 'slidetags', 'xenium']:
            sub = preg.filter(
                (pl.col('dataset') == ds) &
                (pl.col('gene') == gene) &
                (pl.col('cell_type') == ct) &
                pl.col('ref_pct_detected').is_not_null())
            if sub.height > 0:
                ref_val = sub['ref_pct_detected'][0]
                break
        ref_flag = ' FILTERED' if ref_val is not None and ref_val < 10 else ''
        ref_str = f'{ref_val:.0f}%' if ref_val is not None else '-'
        print(f'  {ct_short:42s} ref={ref_str:>5s}{ref_flag}')
        for ds, (lfc, _, p) in [('mf', mf), ('st', st), ('xn', xn)]:
            if lfc:
                print(f'    {ds}: logFC={lfc} {p}')
    if len(cts) > 8:
        print(f'  ... and {len(cts) - 8} more cell types')
    print()

# --- 5. What fraction of the manuscript's claims survive the filter? ---
print('=== 5. Manuscript claim survival (ref≥10% filter) ===\n')

# Key claims from Fig 2 text
claims = [
    ('Esr1', 'merfish', ['057 NDB-SI-MA-STRv Lhx8 Gaba',
                          '086 MPO-ADP Lhx8 Gaba']),
    ('Pgr', 'merfish', ['007 L2/3 IT CTX Glut', '006 L4/5 IT CTX Glut',
                         '005 L5 IT CTX Glut']),
    ('Prlr', 'merfish', ['086 MPO-ADP Lhx8 Gaba', '085 SI-MPO-LPO Lhx8 Gaba']),
    ('Gal', 'merfish', ['086 MPO-ADP Lhx8 Gaba', '085 SI-MPO-LPO Lhx8 Gaba']),
    ('Sst', 'merfish', ['052 Pvalb Gaba', '053 Sst Gaba',
                         '046 Vip Gaba']),
    ('Vegfa', 'merfish', ['319 Astro-TE NN', '333 Endo NN']),
    ('Nr3c1', 'slidetags', ['007 L2/3 IT CTX Glut', '327 Oligo NN']),
    ('Ntrk2', 'slidetags', ['007 L2/3 IT CTX Glut', '006 L4/5 IT CTX Glut']),
    ('Bdnf', 'slidetags', ['007 L2/3 IT CTX Glut']),
    ('Ckb', 'slidetags', ['007 L2/3 IT CTX Glut', '006 L4/5 IT CTX Glut']),
]

for gene, ds, cell_types in claims:
    print(f'{gene} in {ds}:')
    for ct in cell_types:
        sub = preg.filter(
            (pl.col('dataset') == ds) &
            (pl.col('gene') == gene) &
            (pl.col('cell_type') == ct))
        if sub.height > 0:
            r = sub.row(0, named=True)
            ref = r['ref_pct_detected']
            status = 'PASS' if ref is not None and ref >= 10 else 'FAIL'
            ref_str = f'{ref:.1f}%' if ref is not None else 'null'
            print(f'  {status} {ct:42s} ref={ref_str:>6s} '
                  f'logFC={r["logFC"]:+.3f} p={r["PValue"]:.1e}')
        else:
            print(f'  MISS {ct:42s} (not tested)')
    print()
