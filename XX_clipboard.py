import polars as pl
import numpy as np

working_dir = '/home/karbabi/spatial-pregnancy'
pw = pl.read_csv(f'{working_dir}/output/pathway_results_gsea_sig.csv')
pw_full = pl.read_parquet(f'{working_dir}/output/pathway_results_gsea.parquet')

def get_type(ct):
    if 'Glut' in ct:
        return 'Glut'
    elif any(x in ct for x in ['Gaba', 'IMN', 'Chol']):
        return 'Gaba'
    return 'NN'

preg = pw_full.filter(pl.col('contrast') == 'PREG_vs_CTRL')
preg_sig = pw.filter(pl.col('contrast') == 'PREG_vs_CTRL')

print('=' * 100)
print('PATHWAY DEEP ANALYSIS')
print('=' * 100)

print('\n=== 1. Slide-tags: top pathways per cell class ===\n')

st = preg_sig.filter(pl.col('dataset') == 'slidetags')
st = st.with_columns(
    pl.col('cell_type').map_elements(get_type, return_dtype=pl.Utf8)
    .alias('class'))

for cls in ['Glut', 'Gaba', 'NN']:
    sub = st.filter(pl.col('class') == cls)
    freq = sub.group_by('pathway').agg([
        pl.len().alias('n_ct'),
        pl.col('NES').mean().alias('mean_NES'),
        pl.col('theme').first().alias('theme'),
    ]).sort('n_ct', descending=True)
    print(f'  --- {cls} ({sub["cell_type"].n_unique()} cell types) ---')
    for row in freq.head(8).iter_rows(named=True):
        d = '↑' if row['mean_NES'] > 0 else '↓'
        print(f'    {row["pathway"][:50]:50s} {row["n_ct"]:>2d} ct '
              f'{d} NES={row["mean_NES"]:+.2f} [{row["theme"]}]')
    print()

print('=== 2. Select pathways for heatmap ===\n')

PATHWAY_SELECTION = {
    'Neuronal': [
        'GOBP_SYNAPTIC_SIGNALING',
        'GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY',
        'GOBP_REGULATION_OF_TRANS_SYNAPTIC_SIGNALING',
        'GOBP_GLUTAMATE_RECEPTOR_SIGNALING_PATHWAY',
        'GOBP_NEUROTRANSMITTER_TRANSPORT',
        'GOBP_LONG_TERM_SYNAPTIC_POTENTIATION',
    ],
    'Metabolic': [
        'GOBP_OXIDATIVE_PHOSPHORYLATION',
        'GOBP_ELECTRON_TRANSPORT_CHAIN',
        'GOBP_CELLULAR_RESPIRATION',
        'GOBP_ATP_SYNTHESIS_COUPLED_ELECTRON_TRANSPORT',
    ],
    'Vascular': [
        'GOBP_VASCULATURE_DEVELOPMENT',
        'GOBP_BLOOD_VESSEL_MORPHOGENESIS',
        'GOBP_ENDOTHELIAL_CELL_MIGRATION',
        'GOBP_REGULATION_OF_ENDOTHELIAL_CELL_MIGRATION',
    ],
    'Structural': [
        'GOBP_CELL_ADHESION',
        'GOBP_CELL_SUBSTRATE_ADHESION',
    ],
    'Immune': [
        'GOBP_CYTOKINE_PRODUCTION',
        'GOBP_POSITIVE_REGULATION_OF_INFLAMMATORY_RESPONSE',
    ],
    'Hormonal': [
        'GOBP_CELLULAR_RESPONSE_TO_HORMONE_STIMULUS',
        'GOBP_NEUROPEPTIDE_SIGNALING_PATHWAY',
    ],
    'Growth_Factors': [
        'GOBP_CELLULAR_RESPONSE_TO_GROWTH_FACTOR_STIMULUS',
    ],
}

all_selected = []
for theme, pws in PATHWAY_SELECTION.items():
    for p in pws:
        all_selected.append(p)

print(f'Selected {len(all_selected)} pathways across {len(PATHWAY_SELECTION)} themes')

print('\n=== 3. NES matrix for selected pathways × cell types ===\n')

glut_cts = sorted(set(
    st.filter(pl.col('class') == 'Glut')['cell_type'].unique().to_list()))
gaba_cts = sorted(set(
    st.filter(pl.col('class') == 'Gaba')['cell_type'].unique().to_list()))
nn_cts = sorted(set(
    st.filter(pl.col('class') == 'NN')['cell_type'].unique().to_list()))

for cls_name, cts in [('Glut', glut_cts), ('Gaba', gaba_cts), ('NN', nn_cts)]:
    print(f'\n  --- {cls_name}: {len(cts)} cell types ---')
    for pw_name in all_selected[:5]:
        row_data = []
        for ct in cts:
            hit = preg.filter(
                (pl.col('dataset') == 'slidetags') &
                (pl.col('pathway') == pw_name) &
                (pl.col('cell_type') == ct))
            if hit.height > 0:
                nes = hit['NES'][0]
                padj = hit['padj'][0]
                sig = '*' if padj < 0.10 else ''
                row_data.append(f'{nes:+.1f}{sig}')
            else:
                row_data.append('  --')
        short = pw_name.replace('GOBP_', '')[:35]
        print(f'    {short:35s} {" ".join(f"{x:>6s}" for x in row_data[:6])}')

print('\n=== 4. Cross-platform pathway concordance detail ===\n')

for pw_name in all_selected:
    st_hits = preg_sig.filter(
        (pl.col('dataset') == 'slidetags') &
        (pl.col('pathway') == pw_name))
    xn_hits = preg_sig.filter(
        (pl.col('dataset') == 'xenium') &
        (pl.col('pathway') == pw_name))
    if st_hits.height == 0 and xn_hits.height == 0:
        continue
    st_cts = set(st_hits['cell_type'].to_list())
    xn_cts = set(xn_hits['cell_type'].to_list())
    shared_cts = st_cts & xn_cts
    short = pw_name.replace('GOBP_', '')[:45]

    st_dir = '↓' if st_hits.height > 0 and st_hits['NES'].mean() < 0 else '↑'
    xn_dir = '↓' if xn_hits.height > 0 and xn_hits['NES'].mean() < 0 else '↑'

    concordant = st_dir == xn_dir if st_hits.height > 0 and xn_hits.height > 0 \
        else None
    conc_str = 'CONCORDANT' if concordant == True else \
               'DISCORDANT' if concordant == False else 'ONE-PLATFORM'

    print(f'  {short:45s} ST:{st_hits.height:>2d}ct{st_dir} '
          f'XN:{xn_hits.height:>2d}ct{xn_dir} '
          f'shared:{len(shared_cts)} [{conc_str}]')

print('\n=== 5. Postpartum reversal check ===\n')

for pw_name in ['GOBP_OXIDATIVE_PHOSPHORYLATION',
                 'GOBP_SYNAPTIC_SIGNALING',
                 'GOBP_VASCULATURE_DEVELOPMENT',
                 'GOBP_CELL_ADHESION',
                 'GOBP_CELLULAR_RESPIRATION',
                 'GOBP_ENDOTHELIAL_CELL_MIGRATION']:
    preg_nes = preg_sig.filter(
        (pl.col('dataset') == 'slidetags') &
        (pl.col('pathway') == pw_name))
    pp = pw.filter(
        (pl.col('dataset') == 'slidetags') &
        (pl.col('contrast') == 'POSTPART_vs_PREG') &
        (pl.col('pathway') == pw_name))
    short = pw_name.replace('GOBP_', '')[:40]
    preg_mean = preg_nes['NES'].mean() if preg_nes.height > 0 else 0
    pp_mean = pp['NES'].mean() if pp.height > 0 else 0
    reversed = (preg_mean * pp_mean < 0) if pp.height > 0 else None
    rev_str = 'REVERSED' if reversed else \
              'PERSISTENT' if reversed == False else 'NO PP DATA'
    print(f'  {short:40s} PREG={preg_mean:+.2f}({preg_nes.height}ct) '
          f'PP={pp_mean:+.2f}({pp.height}ct) [{rev_str}]')

print('\n=== 6. Figure panel design ===\n')

print('Proposed layout (3 heatmap blocks, 2 contrast columns each):')
print()
print('Block 1: Glutamatergic neurons')
glut_show = [ct for ct in glut_cts
             if st.filter(pl.col('cell_type') == ct).height >= 3]
print(f'  Cell types with >=3 sig pathways: {len(glut_show)}')
for ct in glut_show[:8]:
    n = st.filter(pl.col('cell_type') == ct).height
    print(f'    {ct}: {n} pathways')

print('\nBlock 2: GABAergic neurons')
gaba_show = [ct for ct in gaba_cts
             if st.filter(pl.col('cell_type') == ct).height >= 3]
print(f'  Cell types with >=3 sig pathways: {len(gaba_show)}')
for ct in gaba_show[:8]:
    n = st.filter(pl.col('cell_type') == ct).height
    print(f'    {ct}: {n} pathways')

print('\nBlock 3: Non-neuronal')
nn_show = [ct for ct in nn_cts
           if st.filter(pl.col('cell_type') == ct).height >= 3]
print(f'  Cell types with >=3 sig pathways: {len(nn_show)}')
for ct in nn_show:
    n = st.filter(pl.col('cell_type') == ct).height
    print(f'    {ct}: {n} pathways')
