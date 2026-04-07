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

#region leave-section-out cross-validation #####################################

# evenly spaced sections + sections 46-49 (query reference range)
cv_idx = np.linspace(0, len(all_sections) - 1, 8, dtype=int)
cv_sections = list(dict.fromkeys(
    [all_sections[i] for i in cv_idx] +
    ['C57BL6J-638850.46', 'C57BL6J-638850.47',
     'C57BL6J-638850.48', 'C57BL6J-638850.49']))
n_cv = len(cv_sections)
print(f'\n[CV] hold-out sections: {[s.split(".")[-1] for s in cv_sections]}')

cv_results = []
cv_sims = {}

for i, held_out in enumerate(cv_sections):
    held_idx = all_sections.index(held_out)
    query_fp = ref_fp_matrix[held_idx]

    ref_mask = np.arange(len(all_sections)) != held_idx
    ref_fps = ref_fp_matrix[ref_mask]
    ref_aps = ref_ap_values[ref_mask]
    ref_names = [s for j, s in enumerate(all_sections) if j != held_idx]

    pred_ap, sims = predict_ap(query_fp, ref_fps, ref_aps)
    true_ap = section_ap[held_out]
    error = abs(pred_ap - true_ap)

    cv_sims[held_out] = (ref_aps, sims, ref_names)

    top3 = np.argsort(sims)[::-1][:3]
    top_str = ', '.join(
        f'{ref_names[j].split(".")[-1]}({sims[j]:.3f})' for j in top3)

    cv_results.append(dict(
        section=held_out, true_ap=true_ap,
        pred_ap=pred_ap, error=error))

    print(f'[CV {i+1}/{n_cv}] section {held_out.split(".")[-1]}: '
          f'pred={pred_ap:.2f}, true={true_ap:.2f}, error={error:.3f} '
          f'(top: {top_str})')

cv_df = pd.DataFrame(cv_results)
ap_range = cells['x_ccf'].max() - cells['x_ccf'].min()
mean_err = cv_df['error'].mean()
print(f'\n[CV] template matching: MAE={mean_err:.3f} ± '
      f'{cv_df["error"].std():.3f} '
      f'({mean_err / ap_range * 100:.1f}% of AP range)')

# predicted vs true AP
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 13], [0, 13], 'k--', lw=1, zorder=1)
ax.scatter(cv_df['true_ap'], cv_df['pred_ap'],
           s=100, c='steelblue', marker='D', zorder=5,
           edgecolors='white', linewidths=1)
for _, row in cv_df.iterrows():
    ax.annotate(row['section'].split('.')[-1],
                (row['true_ap'], row['pred_ap']),
                textcoords='offset points', xytext=(8, 4), fontsize=8)
ax.set_xlabel('True AP')
ax.set_ylabel('Predicted AP')
ax.set_title(f'Template Matching (leave-section-out CV)\n'
             f'MAE={mean_err:.3f} ({mean_err / ap_range * 100:.1f}% '
             f'of AP range)')
ax.set_aspect('equal')
plt.tight_layout()
fig.savefig(f'{fig_dir}/ap_template_pred_vs_true.png', dpi=200)
plt.close()

# similarity profiles
ncols = 4
nrows = int(np.ceil(n_cv / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                         squeeze=False)
fig.suptitle('Similarity to Reference Sections', fontsize=14)

for i, held_out in enumerate(cv_sections):
    ax = axes[i // ncols, i % ncols]
    ref_aps, sims, ref_names = cv_sims[held_out]
    ax.scatter(ref_aps, sims, s=8, alpha=0.6, c='steelblue')
    true_ap = section_ap[held_out]
    ax.axvline(true_ap, color='red', lw=1, ls='--', label='true AP')
    pred_ap = cv_df.loc[cv_df['section'] == held_out, 'pred_ap'].values[0]
    ax.axvline(pred_ap, color='green', lw=1, ls='--', label='pred AP')
    ax.set_title(f'section {held_out.split(".")[-1]}', fontsize=10)
    ax.set_xlabel('Reference AP')
    ax.set_ylabel('Cosine similarity')
    if i == 0:
        ax.legend(fontsize=7)

for j in range(n_cv, nrows * ncols):
    axes[j // ncols, j % ncols].set_visible(False)

plt.tight_layout()
fig.savefig(f'{fig_dir}/ap_template_similarity_profiles.png', dpi=200)
plt.close()

print('figures saved to figures/')

#endregion

#region inference on query data ################################################

import scanpy as sc

datasets_config = {
    'merfish':   {'sample_col': 'sample',     's': 1.0},
    'slidetags': {'sample_col': 'sample',     's': 2.0},
    'xenium':    {'sample_col': 'sample_rep', 's': 0.5},
}
ref_s = 0.5

ref_section_names = ['C57BL6J-638850.46', 'C57BL6J-638850.47',
                     'C57BL6J-638850.48', 'C57BL6J-638850.49']
ref_section_aps = section_ap[ref_section_names].sort_values()

all_sample_ap = []
for name, cfg in datasets_config.items():
    sample_col = cfg['sample_col']
    s_query = cfg['s']

    query_path = f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad'
    adata = sc.read_h5ad(query_path)
    print(f'[{name}] {adata.shape[0]:,} cells')

    # convert CAST-aligned coords to original CCF space
    # reference prep: x_raw = -z_ccf, y_raw = -y_ccf
    y_ccf_q = -adata.obs['y_ffd'].values
    z_ccf_q = -adata.obs['x_ffd'].values

    samples = sorted(adata.obs[sample_col].unique())
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
            dataset=name, sample=s, ap_predicted=pred_ap,
            n_cells=int(mask.sum()),
            top_match=all_sections[top3_idx[0]]))

        print(f'[{name}] {s}: AP={pred_ap:.2f} (top: {top3_str})')

    out_path = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    # adata.write(out_path)
    print(f'[{name}] saved to {out_path}')

    # per-dataset figure: row 0 = 4 ref sections, rows 1-2 = query slices
    # all colored by class, ordered anterior to posterior
    sample_aps_q = (adata.obs.groupby(sample_col)['ap_predicted']
                    .first().sort_values())
    query_samples_ord = list(sample_aps_q.index)
    n_query = len(query_samples_ord)
    half = int(np.ceil(n_query / 2))
    ncols = max(len(ref_section_names), half)
    nrows = 3

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle(f'{name}: Reference + Query Slices (anterior→posterior)',
                 fontsize=14)

    # row 0: reference sections
    for i, sec_name in enumerate(ref_section_aps.index):
        ax = axes[0, i]
        sec_cells = cells[cells['brain_section_label'] == sec_name]
        ax.scatter(sec_cells['z_ccf'], sec_cells['y_ccf'],
                   s=ref_s, c=sec_cells['class_color'],
                   alpha=0.5, rasterized=True)
        ax.set_title(f'ref {sec_name.split(".")[-1]} '
                     f'(AP={ref_section_aps[sec_name]:.2f})', fontsize=10)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    # rows 1-2: query slices, split anterior half / posterior half
    for q_idx, samp in enumerate(query_samples_ord):
        r = 1 if q_idx < half else 2
        c = q_idx if q_idx < half else q_idx - half
        ax = axes[r, c]
        sm = (adata.obs[sample_col] == samp).values
        cls = adata.obs.loc[sm, 'class'].values
        colors = [class_colors.get(c_, '#d3d3d3') for c_ in cls]
        ax.scatter(z_ccf_q[sm], y_ccf_q[sm],
                   s=s_query, c=colors, alpha=0.5, rasterized=True)
        ap_val = sample_aps_q[samp]
        ax.set_title(f'{samp} (AP={ap_val:.2f})', fontsize=10)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')

    # hide unused axes
    n_row1, n_row2 = half, n_query - half
    for c in range(ncols):
        if c >= len(ref_section_names):
            axes[0, c].set_visible(False)
        if c >= n_row1:
            axes[1, c].set_visible(False)
        if c >= n_row2:
            axes[2, c].set_visible(False)

    plt.tight_layout()
    fig.savefig(f'{fig_dir}/ap_imputation_{name}.png',
                dpi=200, bbox_inches='tight')
    plt.close()

if all_sample_ap:
    ap_df = pd.DataFrame(all_sample_ap).sort_values('ap_predicted')
    print(f'\n[summary] {len(ap_df)} samples across '
          f'{ap_df["dataset"].nunique()} datasets')

print('done')

#endregion
