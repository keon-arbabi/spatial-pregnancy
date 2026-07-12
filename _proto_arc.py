"""Prototype: peripartum GSEA trajectory arcs (Panel A candidate)."""
import importlib
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

fc = importlib.import_module('12_figure_helper')
fc.setup_style()
wd = '/home/karbabi/spatial-pregnancy'

PATHWAY_BANDS = {
 'Synaptic adhesion': ['GOBP_SYNAPSE_ASSEMBLY', 'GOBP_SYNAPSE_ORGANIZATION',
    'GOBP_HOMOPHILIC_CELL_CELL_ADHESION'],
 'Excitability': ['GOBP_REGULATION_OF_MEMBRANE_POTENTIAL',
    'GOBP_REGULATION_OF_POSTSYNAPTIC_MEMBRANE_POTENTIAL',
    'GOBP_POTASSIUM_ION_TRANSPORT'],
 'GABA & neuropeptide': ['GOBP_SYNAPTIC_TRANSMISSION_GABAERGIC',
    'GOBP_NEUROPEPTIDE_SIGNALING_PATHWAY', 'GOBP_NEUROTRANSMITTER_SECRETION'],
 'Glucocorticoid stress': ['GOBP_RESPONSE_TO_CORTICOSTEROID',
    'GOBP_RESPONSE_TO_STEROID_HORMONE'],
 'Neurotrophic': ['GOBP_RESPONSE_TO_NERVE_GROWTH_FACTOR',
    'GOBP_NEUROTROPHIN_TRK_RECEPTOR_SIGNALING_PATHWAY',
    'GOBP_RESPONSE_TO_GROWTH_FACTOR'],
 'Neuronal development': ['GOBP_REGULATION_OF_NEURON_DIFFERENTIATION',
    'GOBP_NEURON_FATE_COMMITMENT', 'GOBP_AXON_DEVELOPMENT'],
 'Activity-dependent plasticity': ['GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY',
    'GOBP_REGULATION_OF_TRANS_SYNAPTIC_SIGNALING',
    'GOBP_REGULATION_OF_LONG_TERM_SYNAPTIC_POTENTIATION'],
}
band_of = {p: b for b, ps in PATHWAY_BANDS.items() for p in ps}
PW = list(band_of)
BAND_COLORS = {
 'Synaptic adhesion': '#0072B2', 'Excitability': '#E69F00',
 'GABA & neuropeptide': '#009E73', 'Glucocorticoid stress': '#D55E00',
 'Neurotrophic': '#C9A800', 'Neuronal development': '#CC79A7',
 'Activity-dependent plasticity': '#56B4E9'}
SG = [('Synaptic & excitability remodeling',
       ['Synaptic adhesion', 'Excitability', 'GABA & neuropeptide']),
      ('Hormone & trophic signalling',
       ['Glucocorticoid stress', 'Neurotrophic']),
      ('Development & plasticity',
       ['Neuronal development', 'Activity-dependent plasticity'])]
NEUR = lambda c: (' Glut' in c) or (' Gaba' in c) or ('IMN' in c)

# median NES across platforms per (contrast, pathway, ct)
r = (pl.read_parquet(f'{wd}/output/gsea/perms/real_gsea.parquet')
     .filter(pl.col('pathway').is_in(PW))
     .group_by(['contrast', 'pathway', 'cell_type'])
     .agg(pl.col('NES').median().alias('nes')))
nes = {(x['contrast'], x['pathway'], x['cell_type']): x['nes']
       for x in r.iter_rows(named=True)}
# pregnancy-responsive neuronal pathway x ct pairs
g = (pl.read_csv(f'{wd}/output/gsea/sumrank_gsea_results.csv')
     .filter((pl.col('contrast') == 'PREG_vs_CTRL') & (pl.col('D') >= 2)
             & pl.col('pathway').is_in(PW))
     .with_columns(pl.when(pl.col('nlp_up') >= pl.col('nlp_down'))
                   .then(pl.col('emp_p_up')).otherwise(pl.col('emp_p_down'))
                   .alias('emp_p')).filter(pl.col('emp_p') <= 0.05))
pairs = [(x['pathway'], x['cell_type']) for x in g.iter_rows(named=True)
         if NEUR(x['cell_type'])]

XS = np.array([0, 1, 2.0])
XLAB = ['Nulli-\nparous', 'Pregnant', 'Post-\npartum']


def band_traj(b):
    bp = [(p, c) for (p, c) in pairs if band_of[p] == b]
    rows = []
    for p, c in bp:
        pv = nes.get(('PREG_vs_CTRL', p, c))
        qv = nes.get(('POSTPART_vs_CTRL', p, c))
        if pv is not None and qv is not None:
            rows.append((pv, qv))
    a = np.array(rows)
    return a  # columns: preg, postpart


fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.1), sharey=True)
for ax, (sg, bands) in zip(axes, SG):
    ax.axhline(0, color='#999', lw=0.8, ls='--', zorder=1)
    for b in bands:
        a = band_traj(b)
        col = BAND_COLORS[b]
        # faint individual pathway x ct trajectories
        for pv, qv in a:
            ax.plot(XS, [0, pv, qv], color=col, lw=0.4, alpha=0.10, zorder=2)
        # bold band-median arc
        med = [0, np.median(a[:, 0]), np.median(a[:, 1])]
        ax.plot(XS, med, color=col, lw=2.4, marker='o', ms=5,
                zorder=4, solid_capstyle='round')
        ax.annotate(f' {b}', (2, med[2]), color=col, fontsize=7.5,
                    va='center', ha='left', fontweight='bold')
    ax.set_title(sg, fontsize=8.5)
    ax.set_xticks(XS)
    ax.set_xticklabels(XLAB, fontsize=7.5)
    ax.set_xlim(-0.15, 3.3)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('GSEA enrichment (median NES)', fontsize=8.5)
fig.suptitle('Peripartum trajectory of neuronal pathway programs',
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(f'{wd}/figures/_proto_arc.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print('wrote figures/_proto_arc.png')
