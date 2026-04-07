#region imports and setup #######################################################

import subprocess
import sys

import numpy as np
import scanpy as sc

working_dir = '/home/karbabi/spatial-pregnancy'
datasets = {
    'merfish': 'sample',
    'slidetags': 'sample',
    'xenium': 'sample_rep',
}
thr = dict(
    subclass_confidence=0.6,
    subclass_margin=0.2,
    min_cos_dist=0.3,
    n_spatial_candidates=1,
)
min_cells_per_sample = 10

#endregion
#region filter ##################################################################

for name, sample_col in datasets.items():
    in_path = f'{working_dir}/output/{name}/02_adata_query_{name}.h5ad'
    out_path = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    adata = sc.read_h5ad(in_path)
    n_total = len(adata)
    obs = adata.obs
    print(f'\n[{name}] {n_total:,} cells')

    masks = {
        'subclass_confidence':
            obs['subclass_confidence'] >= thr['subclass_confidence'],
        'subclass_margin':
            obs['subclass_margin'] >= thr['subclass_margin'],
        'min_cos_dist':
            obs['min_cos_dist'] < thr['min_cos_dist'],
        'n_spatial_candidates':
            obs['n_spatial_candidates'] >= thr['n_spatial_candidates'],
    }
    for k, m in masks.items():
        n_drop = (~m).sum()
        print(f'  drop {k}: {n_drop:,} ({n_drop/n_total*100:.1f}%)')

    # drop cells of subclasses with < min_cells_per_sample in their sample
    counts = obs.groupby([sample_col, 'subclass'], observed=True).size()
    rare = counts[counts < min_cells_per_sample]
    rare_pairs = set(rare.index)
    rare_mask = np.array([
        (s, c) not in rare_pairs
        for s, c in zip(obs[sample_col].values, obs['subclass'].values)])
    n_rare_drop = (~rare_mask).sum()
    print(f'  drop rare (<{min_cells_per_sample}/sample): '
          f'{n_rare_drop:,} ({n_rare_drop/n_total*100:.1f}%)')
    masks['rare_subclass'] = rare_mask

    keep = np.logical_and.reduce(list(masks.values()))
    n_keep = keep.sum()
    n_dropped = n_total - n_keep
    print(f'  total dropped: {n_dropped:,} ({n_dropped/n_total*100:.1f}%)')
    print(f'  keep: {n_keep:,} ({n_keep/n_total*100:.1f}%)')

    adata = adata[keep].copy()
    adata.write(out_path)
    print(f'[{name}] saved {out_path}')

# regenerate viewer_data/ from the freshly written 03 h5ads
print('\n[viewer] regenerating viewer_data/...')
subprocess.run(
    [sys.executable, f'{working_dir}/XX_export_viewer.py'], check=True)

print('\ndone')

'''
[merfish] 990,647 cells
  drop subclass_confidence: 199,817 (20.2%)
  drop subclass_margin: 149,008 (15.0%)
  drop min_cos_dist: 57,259 (5.8%)
  drop n_spatial_candidates: 8 (0.0%)
  total dropped: 241,794 (24.4%)
  keep: 748,853 (75.6%)

[slidetags] 87,491 cells
  drop subclass_confidence: 9,754 (11.1%)
  drop subclass_margin: 7,626 (8.7%)
  drop min_cos_dist: 8,237 (9.4%)
  drop n_spatial_candidates: 299 (0.3%)
  total dropped: 16,062 (18.4%)
  keep: 71,429 (81.6%)

[xenium] 867,704 cells
  drop subclass_confidence: 90,909 (10.5%)
  drop subclass_margin: 72,974 (8.4%)
  drop min_cos_dist: 27,073 (3.1%)
  drop n_spatial_candidates: 123 (0.0%)
  total dropped: 109,740 (12.6%)
  keep: 757,964 (87.4%)
'''
#endregion
