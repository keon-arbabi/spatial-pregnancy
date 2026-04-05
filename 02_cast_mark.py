#region imports and setup #######################################################

import os, sys, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import torch
import scanorama
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sparse
from sklearn.cluster import KMeans
import CAST
from CAST.models.model_GCNII import Args

working_dir = '/home/karbabi/spatial-pregnancy'

def rotate_coords(coords, angle):
    theta = np.radians(angle)
    rot_mat = torch.tensor([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]], dtype=torch.float32)
    return torch.mm(torch.from_numpy(coords).float(), rot_mat).numpy()

def run_cast_mark(name, adata_query, adata_ref, cast_args, k_values,
                  sample_col='sample', rotation_angles=None):
    output_dir = f'{working_dir}/output/{name}'
    os.makedirs(f'{output_dir}/CAST-MARK', exist_ok=True)
    print(f'[{name}] query: {adata_query.shape}, ref: {adata_ref.shape}')

    # normalize query (raw counts -> log-normalized)
    adata_query.X = adata_query.layers['counts'].copy()
    sc.pp.normalize_total(adata_query)
    sc.pp.log1p(adata_query)

    # batch correction with scanorama (cached)
    scanorama_path = f'{output_dir}/scanorama_X.npz'
    if os.path.exists(scanorama_path):

        print(f'[{name}] loading cached scanorama...')
        all_X = sparse.load_npz(scanorama_path)
    else:
        print(f'[{name}] running scanorama...')
        adata_query_s, adata_ref_s = scanorama.correct_scanpy([
            adata_query, adata_ref])
        all_X = ad.concat(
            [adata_query_s, adata_ref_s], axis=0, merge='same').X
        sparse.save_npz(scanorama_path, all_X)
        del adata_query_s, adata_ref_s

    # build coords and expression dicts
    query_samples = sorted(adata_query.obs[sample_col].unique())
    ref_samples = sorted(adata_ref.obs['sample'].unique())
    sample_names = sorted(set(query_samples) | set(ref_samples))

    print(f'[{name}] {len(query_samples)} query + {len(ref_samples)} ref = '
          f'{len(sample_names)} samples')
    all_obs = pd.concat([adata_query.obs, adata_ref.obs])

    coords_raw = {}
    exp_dict = {}
    for s in sample_names:
        if s in query_samples:
            mask = all_obs[sample_col] == s
        else:
            mask = all_obs['sample'] == s
        coords_raw[s] = all_obs.loc[mask, ['x_raw', 'y_raw']].values
        exp_dict[s] = all_X[mask.values].toarray()

    # rotate query coords for registration
    if rotation_angles:
        coords_raw = {
            k: rotate_coords(v, rotation_angles[k])
            if k in rotation_angles else v
            for k, v in coords_raw.items()}

    # run CAST-MARK (or load cached)
    embed_path = f'{output_dir}/embed_dict.pt'
    if os.path.exists(embed_path):
        print(f'[{name}] loading cached embeddings')
        embed_dict = torch.load(embed_path, weights_only=False)
    else:
        print(f'[{name}] running CAST-MARK ({cast_args.epochs} epochs, '
              f'{cast_args.n_layers} layers)...')
        embed_dict = CAST.CAST_MARK(
            coords_raw, exp_dict,
            f'{output_dir}/CAST-MARK',
            graph_strategy='delaunay',
            args=cast_args)
        torch.save(embed_dict, embed_path)

    # detach and stack embeddings
    embed_dict = {k.split('_dup')[0]: v.cpu().detach()
                  for k, v in embed_dict.items()}
    embed_stack = np.vstack([embed_dict[s].numpy() for s in sample_names])

    # kmeans clustering plots
    for n_clust in k_values:
        print(f'[{name}] Clustering k={n_clust}')
        labels = KMeans(n_clusters=n_clust, random_state=0).fit_predict(
            embed_stack)
        colors = sns.color_palette('Set3', n_clust)

        num_plot = len(sample_names)
        plot_row = int(np.ceil(num_plot / 5))
        fig, axes = plt.subplots(plot_row, 5, figsize=(30, 3.5 * plot_row),
                                 squeeze=False)
        cell_start = 0
        for j, s in enumerate(sample_names):
            ax = axes[j // 5, j % 5]
            coords = coords_raw[s]
            n = coords.shape[0]
            size = np.log(1e4 / n) + 3
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=labels[cell_start:cell_start + n],
                       cmap=plt.cm.colors.ListedColormap(colors),
                       s=size, edgecolors='none')
            ax.set_title(f'{s} (k={n_clust})', fontsize=14)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            cell_start += n
        for j in range(num_plot, plot_row * 5):
            axes[j // 5, j % 5].set_visible(False)
        plt.tight_layout()
        fig.savefig(f'{working_dir}/figures/{name}_k{n_clust}.png', dpi=150)
        plt.close(fig)

    # save artifacts
    torch.save(coords_raw, f'{output_dir}/coords_raw.pt', pickle_protocol=4)
    torch.save(exp_dict, f'{output_dir}/exp_dict.pt', pickle_protocol=4)
    total_cells = sum(c.shape[0] for c in coords_raw.values())
    print(f'[{name}] done — {total_cells:,} cells, '
          f'{embed_stack.shape[1]} embedding dims')

#endregion
#region run ####################################################################

import sys

def run_merfish():
    adata_query = sc.read_h5ad(
        f'{working_dir}/output/merfish/01_adata_query_merfish.h5ad')
    adata_ref = sc.read_h5ad(
        f'{working_dir}/input/adata_ref_zeng_imputed.h5ad')
    run_cast_mark(
        'merfish', adata_query, adata_ref,
        cast_args=Args(
            dataname='merfish', gpu=0,
            epochs=400, lr1=1e-3, wd1=0, lambd=1e-3,
            n_layers=9, der=0.5, dfr=0.3,
            use_encoder=False, encoder_dim=512),
        k_values=[10],
        sample_col='sample',
        rotation_angles={
            'CTRL_1': 72, 'CTRL_2': 110, 'CTRL_3': -33,
            'PREG_1': 3, 'PREG_2': -98, 'PREG_3': -138,
            'POSTPART_1': 75, 'POSTPART_2': 115, 'POSTPART_3': -65})

def run_slidetags():
    adata_query = sc.read_h5ad(
        f'{working_dir}/output/slidetags/01_adata_query_slidetags.h5ad')
    adata_ref = sc.read_h5ad(
        f'{working_dir}/input/adata_ref_zeng_raw.h5ad')
    sc.pp.normalize_total(adata_ref)
    sc.pp.log1p(adata_ref)
    run_cast_mark(
        'slidetags', adata_query, adata_ref,
        cast_args=Args(
            dataname='slidetags', gpu=0,
            epochs=100, lr1=1e-3, wd1=0, lambd=1e-3,
            n_layers=9, der=0.5, dfr=0.3,
            use_encoder=False, encoder_dim=512),
        k_values=[10],
        sample_col='sample',
        rotation_angles={
            'CTRL_1': 35, 'CTRL_2': -130, 'CTRL_3': 100,
            'PREG_1': -90, 'PREG_2': 95, 'PREG_3': 95,
            'POSTPART_1': -20, 'POSTPART_2': -180})

def run_xenium():
    adata_query = sc.read_h5ad(
        f'{working_dir}/output/xenium/01_adata_query_xenium.h5ad')
    adata_ref = sc.read_h5ad(
        f'{working_dir}/input/adata_ref_zeng_imputed.h5ad')
    run_cast_mark(
        'xenium', adata_query, adata_ref,
        cast_args=Args(
            dataname='xenium', gpu=0,
            epochs=400, lr1=1e-3, wd1=0, lambd=1e-3,
            n_layers=9, der=0.5, dfr=0.3,
            use_encoder=True, encoder_dim=512),
        k_values=[10],
        sample_col='sample_rep',
        rotation_angles={
            'CTRL_2_1': -180, 'CTRL_2_2': -180,
            'CTRL_3_1': -180, 'CTRL_3_2': -180,
            'PREG_2_1': -180, 'PREG_2_2': -180,
            'PREG_3_1': -180, 'PREG_3_2': -180})

gpu_map = {'merfish': '0', 'slidetags': '1', 'xenium': '2'}
runners = {'merfish': run_merfish, 'slidetags': run_slidetags,
           'xenium': run_xenium}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <dataset>')
        print(f'Datasets: {", ".join(runners.keys())}')
        sys.exit(1)
    t = sys.argv[1]
    if t not in runners:
        print(f'Unknown dataset: {t}. Choose from {list(runners.keys())}')
        sys.exit(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_map[t]
    runners[t]()

#endregion
