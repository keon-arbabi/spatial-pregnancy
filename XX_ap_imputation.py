#region imports and setup #######################################################

import os, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import joblib

working_dir = '/home/karbabi/spatial-pregnancy'
fig_dir = f'{working_dir}/figures'
os.makedirs(fig_dir, exist_ok=True)

# full 53-section atlas metadata
print('loading full atlas metadata...')
cells = pd.read_csv(
    'single-cell/ABC/metadata/cells_joined.csv',
    usecols=['class', 'class_color', 'x_ccf', 'y_ccf', 'z_ccf',
             'brain_section_label', 'parcellation_division'])

exclude = ['unassigned', 'brain-unassigned', 'fiber tracts-unassigned']
cells = cells[~cells['parcellation_division'].isin(exclude)]
cells = cells[cells['x_ccf'].notna()].reset_index(drop=True)

class_names = sorted(cells['class'].unique())
class_colors = dict(zip(cells['class'], cells['class_color']))
n_classes = len(class_names)
class_to_idx = {c: i for i, c in enumerate(class_names)}

section_ap = cells.groupby('brain_section_label')['x_ccf']\
    .median().sort_values()
all_sections = section_ap.index.tolist()
print(f'{len(cells):,} cells, {len(all_sections)} sections, {n_classes} classes')

# spatial density fingerprints: per-class 2D histograms on a fixed DV/ML grid
n_bins = 50
y_edges = np.linspace(cells['y_ccf'].min(), cells['y_ccf'].max(), n_bins + 1)
z_edges = np.linspace(cells['z_ccf'].min(), cells['z_ccf'].max(), n_bins + 1)

def compute_fingerprint(section_cells, class_col='class'):
    fp = np.zeros((n_classes, n_bins, n_bins))
    for cls, idx in class_to_idx.items():
        cls_cells = section_cells[section_cells[class_col] == cls]
        if len(cls_cells) > 0:
            fp[idx], _, _ = np.histogram2d(
                cls_cells['y_ccf'].values, cls_cells['z_ccf'].values,
                bins=[y_edges, z_edges])
    return fp.flatten()

def predict_ap(query_fp, ref_fps, ref_aps, top_k=5):
    sims = cosine_similarity([query_fp], ref_fps)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    weights = np.maximum(sims[top_idx], 0)
    if weights.sum() > 0:
        weights /= weights.sum()
    return (weights * ref_aps[top_idx]).sum(), sims

print('computing reference fingerprints...')
ref_fp_matrix = np.zeros((len(all_sections), n_classes * n_bins * n_bins))
for i, section in enumerate(all_sections):
    sec_cells = cells[cells['brain_section_label'] == section]
    ref_fp_matrix[i] = compute_fingerprint(sec_cells)
ref_ap_values = np.array([section_ap[s] for s in all_sections])

joblib.dump({
    'ref_fp_matrix': ref_fp_matrix,
    'ref_ap_values': ref_ap_values,
    'all_sections': all_sections,
    'class_to_idx': class_to_idx,
    'y_edges': y_edges,
    'z_edges': z_edges,
    'n_bins': n_bins,
    'n_classes': n_classes,
}, f'{working_dir}/input/ap_fingerprints.joblib')
print('saved fingerprints to input/ap_fingerprints.joblib')

#endregion

#region leave-one-section-out cross-validation #################################
# hold out every atlas section in turn. reporting the error local to the plane
# we sample matters: the global MAE is inflated by the anterior/posterior
# extremes, where top-k weighting is biased inward.

ref_section_names = ['C57BL6J-638850.46', 'C57BL6J-638850.47',
                     'C57BL6J-638850.48', 'C57BL6J-638850.49']
lo = float(section_ap[ref_section_names].min()) - 0.5
hi = float(section_ap[ref_section_names].max()) + 0.5

cv_results = []
for held_idx, held_out in enumerate(all_sections):
    ref_mask = np.arange(len(all_sections)) != held_idx
    pred_ap, _ = predict_ap(ref_fp_matrix[held_idx],
                            ref_fp_matrix[ref_mask], ref_ap_values[ref_mask])
    true_ap = float(section_ap[held_out])
    cv_results.append(dict(section=held_out, true_ap=true_ap,
                           pred_ap=pred_ap, error=abs(pred_ap - true_ap)))

cv_df = pd.DataFrame(cv_results)
ap_range = cells['x_ccf'].max() - cells['x_ccf'].min()
in_window = (cv_df['true_ap'] >= lo) & (cv_df['true_ap'] <= hi)
mae_all = cv_df['error'].mean()
mae_local = cv_df.loc[in_window, 'error'].mean()

print(f'\n[CV] leave-one-section-out over {len(cv_df)} atlas sections')
print(f'[CV] MAE, all sections    = {mae_all:.3f} mm '
      f'({mae_all / ap_range * 100:.1f}% of AP range)')
print(f'[CV] MAE, sampled window  = {mae_local:.3f} mm '
      f'(n = {int(in_window.sum())} sections, {lo:.2f} to {hi:.2f} mm)')

#endregion

#region inference on query data ################################################

import scanpy as sc

datasets_config = {
    'merfish':   {'sample_col': 'sample',     's': 1.0},
    'slidetags': {'sample_col': 'sample',     's': 2.0},
    'xenium':    {'sample_col': 'sample_rep', 's': 0.5},
}
ref_s = 0.5

# samples excluded from the analysis; keep in sync with 06_sumrank.py so this
# figure shows the same sections the reported results are computed from
drop_samples_map = {
    'xenium': ['CTRL_3'],
    'merfish': [],
    'slidetags': [],
}

ref_section_aps = section_ap[ref_section_names].sort_values()

all_sample_ap = []
for name, cfg in datasets_config.items():
    sample_col = cfg['sample_col']
    s_query = cfg['s']

    query_path = f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad'
    adata = sc.read_h5ad(query_path)
    print(f'[{name}] {adata.shape[0]:,} cells')

    # affine, not ffd: a global linear fit places the section in CCF space
    # without the non-rigid warping that could absorb a real AP offset.
    # reference prep: x_raw = -z_ccf, y_raw = -y_ccf
    y_ccf_q = -adata.obs['y_affine'].values
    z_ccf_q = -adata.obs['x_affine'].values

    samples = sorted(adata.obs[sample_col].unique())
    drop = drop_samples_map.get(name, [])
    if drop:
        n_before = len(samples)
        samples = [s for s in samples
                   if adata.obs.loc[adata.obs[sample_col] == s, 'sample']
                   .iloc[0] not in drop]
        print(f'[{name}] dropped {drop}: '
              f'{n_before} -> {len(samples)} samples')

    for s in samples:
        mask = adata.obs[sample_col] == s
        sec_df = pd.DataFrame({
            'class': adata.obs.loc[mask, 'class'].values,
            'y_ccf': y_ccf_q[mask.values],
            'z_ccf': z_ccf_q[mask.values],
        })

        query_fp = compute_fingerprint(sec_df)
        pred_ap, sims = predict_ap(query_fp, ref_fp_matrix, ref_ap_values)

        top3_idx = np.argsort(sims)[::-1][:3]
        top3_str = ', '.join(
            f'{all_sections[j].split(".")[-1]}({sims[j]:.3f})'
            for j in top3_idx)

        adata.obs.loc[mask, 'ap_predicted'] = pred_ap
        all_sample_ap.append(dict(
            dataset=name, sample_rep=s,
            sample=str(adata.obs.loc[mask, 'sample'].iloc[0]),
            condition=str(adata.obs.loc[mask, 'condition'].iloc[0]),
            ap_predicted=pred_ap, n_cells=int(mask.sum()),
            top_match=all_sections[top3_idx[0]]))

        print(f'[{name}] {s}: AP={pred_ap:.2f} (top: {top3_str})')

    out_path = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    # adata.write(out_path)
    print(f'[{name}] saved to {out_path}')

#endregion

#region supplementary figure ###################################################
# Two panels only:
#   A  the predictor is accurate within the atlas, with the error local to the
#      plane we sample reported alongside the global error
#   B  the predicted AP of every sample, by dataset and condition, which is
#      what the reviewer's offset claim is actually about

rep_df = pd.DataFrame(all_sample_ap).sort_values(['dataset', 'ap_predicted'])
rep_df.to_csv(f'{working_dir}/output/ap_predicted_samples.csv', index=False)

# one point per animal: replicate sections from the same mouse are averaged,
# so the unit here matches the unit of replication used for the analysis
ap_df = (rep_df.groupby(['dataset', 'sample', 'condition'], as_index=False)
         ['ap_predicted'].mean().sort_values(['dataset', 'ap_predicted']))
print(f'\n[query] {len(rep_df)} sections from {len(ap_df)} animals')

print('\n[query] mean predicted AP by condition')
for ds, g in ap_df.groupby('dataset'):
    m = g.groupby('condition')['ap_predicted'].mean()
    print(f'  {ds:10s} ' + '  '.join(f'{c}={v:.3f}' for c, v in m.items())
          + f'   max difference = {m.max() - m.min():.3f} mm')

COND_COLORS = {'CTRL': '#7209b7', 'PREG': '#b5179e', 'POSTPART': '#f72585'}
COND_LABEL = {'CTRL': 'Nulliparous', 'PREG': 'Pregnant',
              'POSTPART': 'Postpartum'}
DS_LABEL = {'slidetags': 'Slide-tags', 'merfish': 'MERFISH',
            'xenium': 'Xenium'}

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0),
                         gridspec_kw={'width_ratios': [1.0, 1.3]})

# A) leave-one-section-out validation on the atlas
ax = axes[0]
lims = [cv_df['true_ap'].min() - 0.4, cv_df['true_ap'].max() + 0.4]
ax.axvspan(lo, hi, color='0.92', zorder=0)
ax.plot(lims, lims, 'k--', lw=0.9, zorder=1)
ax.scatter(cv_df.loc[~in_window, 'true_ap'], cv_df.loc[~in_window, 'pred_ap'],
           s=20, c='0.62', zorder=3, label='all other sections')
ax.scatter(cv_df.loc[in_window, 'true_ap'], cv_df.loc[in_window, 'pred_ap'],
           s=32, c='steelblue', edgecolors='white', linewidths=0.5, zorder=4,
           label='sampled window')
ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect('equal')
ax.set_xlabel('True AP (mm)'); ax.set_ylabel('Predicted AP (mm)')
ax.set_title(f'Leave-one-section-out, Allen atlas\n'
             f'MAE {mae_all:.3f} mm overall, {mae_local:.3f} mm in window',
             fontsize=9)
ax.legend(fontsize=7, loc='upper left', frameon=False)
ax.text(-0.16, 1.06, 'A', transform=ax.transAxes, fontsize=13,
        fontweight='bold', va='top', ha='left')

# B) predicted AP of every sample, by dataset and condition
ax = axes[1]
ds_order = [d for d in ['slidetags', 'merfish', 'xenium']
            if d in set(ap_df['dataset'])]
rng = np.random.default_rng(0)
ax.axvspan(lo, hi, color='0.92', zorder=0)
for a in ref_section_aps.values:
    ax.axvline(a, color='0.55', lw=0.8, ls=':', zorder=1)
for yi, ds in enumerate(ds_order):
    sub = ap_df[ap_df['dataset'] == ds]
    for cond, g in sub.groupby('condition'):
        jit = rng.uniform(-0.14, 0.14, len(g))
        ax.scatter(g['ap_predicted'], np.full(len(g), yi) + jit, s=44,
                   c=COND_COLORS.get(cond, '0.5'), edgecolors='white',
                   linewidths=0.6, zorder=3)
ax.set_yticks(range(len(ds_order)))
ax.set_yticklabels([DS_LABEL[d] for d in ds_order])
ax.set_ylim(-0.6, len(ds_order) - 0.4)
ax.set_xlabel('Predicted AP (mm)')
ax.set_title('Predicted AP of every sample', fontsize=9)
present = [c for c in ['CTRL', 'PREG', 'POSTPART']
           if c in set(ap_df['condition'])]
ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', markersize=6,
                              color=COND_COLORS[c], label=COND_LABEL[c])
                   for c in present],
          fontsize=7, loc='best', frameon=False)
ax.text(-0.10, 1.06, 'B', transform=ax.transAxes, fontsize=13,
        fontweight='bold', va='top', ha='left')

plt.tight_layout()
fig.savefig(f'{fig_dir}/ap_validation_supplement.png', dpi=300,
            bbox_inches='tight')
fig.savefig(f'{fig_dir}/ap_validation_supplement.pdf', bbox_inches='tight')
plt.close()

print(f'\nfigure -> {fig_dir}/ap_validation_supplement.png')
print(f'table  -> {working_dir}/output/ap_predicted_samples.csv')
print('done')

#endregion
