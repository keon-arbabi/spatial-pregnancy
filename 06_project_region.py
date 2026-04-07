#region imports and setup ######################################################

import os, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy.spatial import cKDTree

working_dir = '/home/karbabi/spatial-pregnancy'
fig_dir = f'{working_dir}/figures'
os.makedirs(fig_dir, exist_ok=True)

# tunable parameters
PRIOR_KEY = 'subclass' # 'class' (~24) or 'subclass' (~300)
K_LOCAL = 20           # k-NN among query cells for the local composition window
MIN_REF_CELLS = 3000   # drop divisions with too little support in this AP slice
ALPHA = 0.5            # weight on the Bayesian division-frequency prior;
                       # 0 = no prior, larger = pulls borderline cells toward
                       # the more common divisions (STR, Iso)

#endregion

#region build per-division reference fingerprints #############################

# the reference contributes ONLY aggregate cell-type composition per
# division -- no positions. each query cell will be classified by how well
# its local cell-type neighborhood matches each fingerprint.

print('loading reference...')
ref = sc.read_h5ad(f'{working_dir}/input/adata_ref_zeng_raw.h5ad')

section_col = ('brain_section_label' if 'brain_section_label' in ref.obs.columns
               else 'brain_section_label_x')
print(f'  {ref.shape[0]:,} cells across '
      f'{ref.obs[section_col].nunique()} sections')
print(f'  {ref.obs[PRIOR_KEY].nunique()} {PRIOR_KEY} categories')

# keep all divisions with enough reference cells in this AP slice -- grey
# matter, fiber tracts (lfbs, mfbs, ...) and ventricles (VL, V3) all matter
div_counts = ref.obs['parcellation_division'].value_counts()
divisions = sorted(div_counts[div_counts >= MIN_REF_CELLS].index)
ref = ref[ref.obs['parcellation_division'].isin(divisions)].copy()
print(f'  kept {len(divisions)} divisions (>= {MIN_REF_CELLS} cells): '
      f'{divisions}')

div_colors = (ref.obs[['parcellation_division',
                       'parcellation_division_color']]
              .drop_duplicates()
              .set_index('parcellation_division')
              ['parcellation_division_color'].to_dict())

#endregion

#region inference helper ######################################################

def transfer_division(adata, ref, divisions,
                      k=K_LOCAL, prior_key=PRIOR_KEY):
    """
    Local cell-type composition matching.

    For each query cell:
      1. Find its k nearest neighbors in (x_ffd, y_ffd), per sample.
      2. Build a local subclass histogram from those neighbors.
      3. Score each reference division by the log-likelihood that this
         histogram was sampled from the division's aggregate cell-type
         composition (its "fingerprint").
      4. Softmax + argmax -> predicted division.

    The reference contributes only its aggregate per-division compositions.
    No reference positions are used, so query-specific structural variation
    is preserved. Fiber tracts and ventricles are identified because oligo
    and ependymal clusters in the query produce histograms that match the
    fiber-tract / VS fingerprints from the reference.
    """
    n = adata.n_obs
    n_div = len(divisions)

    # build a shared subclass vocabulary from ref + query
    sublabels = sorted(set(ref.obs[prior_key].astype(str).unique()) |
                       set(adata.obs[prior_key].astype(str).unique()))
    sub_to_idx = {s: i for i, s in enumerate(sublabels)}
    n_sub = len(sublabels)

    # 1. reference fingerprints: P(subclass | division), Laplace-smoothed
    div_to_idx = {d: i for i, d in enumerate(divisions)}
    fingerprints = np.zeros((n_div, n_sub), dtype=np.float32)
    ref_div = ref.obs['parcellation_division'].values
    ref_sub = ref.obs[prior_key].astype(str).values
    for d, s in zip(ref_div, ref_sub):
        fingerprints[div_to_idx[d], sub_to_idx[s]] += 1.0
    div_totals = fingerprints.sum(axis=1)              # cells per division
    fingerprints += 1.0
    fingerprints /= fingerprints.sum(axis=1, keepdims=True)
    log_fp = np.log(fingerprints).astype(np.float32)   # (n_div, n_sub)

    # division frequency prior P(div) for the Bayesian update
    div_freq = div_totals / div_totals.sum()
    log_div_prior = np.log(div_freq + 1e-12).astype(np.float32)  # (n_div,)

    # 2. per query cell -- local k-NN subclass histograms, score divisions
    query_sub_idx = np.array(
        [sub_to_idx[str(s)] for s in adata.obs[prior_key].values])

    P = np.full((n, n_div), 1.0 / n_div, dtype=np.float32)
    for s in adata.obs['sample'].unique():
        mask = (adata.obs['sample'] == s).values
        m = int(mask.sum())
        coords = adata.obs.loc[mask, ['x_ffd', 'y_ffd']].values
        if m <= k:
            continue
        tree = cKDTree(coords)
        _, idxs = tree.query(coords, k=k)
        nbr_sub = query_sub_idx[mask][idxs]  # (m, k)

        # local histograms (m, n_sub)
        H = np.zeros((m, n_sub), dtype=np.float32)
        rows = np.repeat(np.arange(m), k)
        cols = nbr_sub.ravel()
        np.add.at(H, (rows, cols), 1.0)

        # log-posterior: average per-neighbor log-likelihood + Bayesian
        # division-frequency prior. dividing by k brings the likelihood
        # onto the same scale as the prior so ALPHA actually has bite.
        avg_log_lik = (H @ log_fp.T) / k                    # (m, n_div)
        log_post = avg_log_lik + ALPHA * log_div_prior[None, :]
        log_post -= log_post.max(axis=1, keepdims=True)
        Pi = np.exp(log_post)
        Pi /= Pi.sum(axis=1, keepdims=True)
        P[mask] = Pi

    best = np.argmax(P, axis=1)
    pred = np.array([divisions[i] for i in best])
    conf = P[np.arange(n), best]
    return pred, conf

#endregion

#region plotting helpers ######################################################

def plot_panel(ax, x, y, colors, title):
    ax.scatter(x, y, s=0.3, alpha=0.4, rasterized=True, c=colors)
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal')
    ax.axis('off')

def plot_grid(panels, fig_path, suptitle):
    """panels = [(title, x, y, colors), ...]; lays out in a row-major grid."""
    n = len(panels)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 4 * nrows),
                             squeeze=False)
    fig.suptitle(suptitle, fontsize=14)
    for i, (title, x, y, c) in enumerate(panels):
        plot_panel(axes[i // ncols, i % ncols], x, y, c, title)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

#endregion

#region reference figure ######################################################

ref_sections = sorted(ref.obs[section_col].unique())
ref_panels = []
for s in ref_sections:
    m = (ref.obs[section_col] == s).values
    ref_panels.append((
        f'{s.split(".")[-1]} (n={m.sum():,})',
        ref.obs.loc[m, 'x_raw'].values,
        ref.obs.loc[m, 'y_raw'].values,
        ref.obs.loc[m, 'parcellation_division_color'].values,
    ))
plot_grid(ref_panels, f'{fig_dir}/region_ref.png',
          'reference: parcellation_division')
print('reference figure saved')

#endregion

#region inference on query data ###############################################

datasets = ['merfish', 'slidetags', 'xenium']

for name in datasets:
    query_path = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    if not os.path.exists(query_path):
        print(f'\n[{name}] missing {query_path}')
        continue

    print(f'\n[{name}] loading...')
    adata = sc.read_h5ad(query_path)
    print(f'[{name}] {adata.shape[0]:,} cells, '
          f'{adata.obs["sample"].nunique()} samples')

    pred, conf = transfer_division(adata, ref, divisions)
    adata.obs['parcellation_division'] = pred
    adata.obs['parcellation_division_color'] = [
        div_colors.get(d, '#888888') for d in pred]
    adata.obs['division_confidence'] = conf

    print(f'[{name}] divisions:')
    for d, c in (adata.obs['parcellation_division']
                 .value_counts().items()):
        print(f'  {d}: {c:,} ({c/len(adata)*100:.1f}%)')
    print(f'[{name}] median confidence: {np.median(conf):.2f}')

    # adata.write(query_path)  # disabled during iteration

    # per-sample query plot in CAST-aligned coords (same frame as ref)
    samples = sorted(adata.obs['sample'].unique())
    panels = []
    for s in samples:
        m = (adata.obs['sample'] == s).values
        panels.append((
            f'{s} (n={m.sum():,})',
            adata.obs.loc[m, 'x_ffd'].values,
            adata.obs.loc[m, 'y_ffd'].values,
            adata.obs.loc[m, 'parcellation_division_color'].values,
        ))
    plot_grid(panels, f'{fig_dir}/region_query_{name}.png',
              f'{name}: parcellation_division')
    print(f'[{name}] figure saved')

print('\ndone')

#endregion
