#region imports and setup #######################################################

import os, sys, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.expanduser('~/CAST-keon'))
import CAST

working_dir = '/home/karbabi/spatial-pregnancy'
ref_section = 'C57BL6J-638850.46'

def split_dicts(coords_raw, embed_dict, n_split, seed=42):
    torch.manual_seed(seed)
    indices_dict = {}
    new_coords = {}
    new_embeds = {}
    query_reference_list = {}
    for key in coords_raw:
        if key.startswith('C57BL6J-638850'):
            new_coords[key] = coords_raw[key]
            new_embeds[key] = embed_dict[key]
        else:
            indices = torch.randperm(coords_raw[key].shape[0])
            indices_dict[key] = indices
            splits = torch.tensor_split(indices, n_split)
            for i, split in enumerate(splits, 1):
                new_key = f'{key}_{i}'
                new_coords[new_key] = coords_raw[key][split]
                new_embeds[new_key] = embed_dict[key][split]
                query_reference_list[new_key] = [new_key, ref_section]
    return new_coords, new_embeds, indices_dict, query_reference_list

def collapse_dicts(coords_final, indices_dict, n_split):
    collapsed = {}
    for base_key, indices in indices_dict.items():
        full_array = np.zeros((len(indices), 2), dtype=np.float32)
        start_idx = 0
        for i in range(1, n_split + 1):
            key = f'{base_key}_{i}'
            if key in coords_final:
                split_data = next(v for k, v in coords_final[key].items()
                                  if not k.startswith('C57BL6J'))
                split_data = np.asarray(split_data, dtype=np.float32)
                end_idx = start_idx + len(split_data)
                split_indices = indices[start_idx:end_idx]
                full_array[split_indices] = split_data
                start_idx = end_idx
        collapsed[base_key] = full_array
    return collapsed

def run_cast_stack(name, n_split=None, stack_params=None):
    output_dir = f'{working_dir}/output/{name}'
    stack_dir = f'{output_dir}/CAST-STACK'
    os.makedirs(stack_dir, exist_ok=True)

    sp = {
        'iterations': 100,
        'dist_penalty1': 0.1,
        'bleeding': 500,
        'd_list': [3, 2, 1, 1/2, 1/3],
        'attention_params': [None, 3, 1, 0],
        'dist_penalty2': [0.1],
        'alpha_basis_bs': [500],
        'meshsize': [8],
        'iterations_bs': [50],
        'attention_params_bs': [[None, 3, 1, 0]],
        'mesh_weight': [None],
    }
    if stack_params:
        sp.update(stack_params)

    coords_raw = torch.load(f'{output_dir}/coords_raw.pt', weights_only=False)
    embed_dict = torch.load(f'{output_dir}/embed_dict.pt', weights_only=False)

    # detach embeddings
    embed_dict = {k.split('_dup')[0]: v.cpu().detach()
                  if hasattr(v, 'detach') else v
                  for k, v in embed_dict.items()}
    query_keys = [k for k in coords_raw if not k.startswith('C57BL6J-638850')]
    print(f'[{name}] {len(query_keys)} query samples, '
          f'n_split={n_split or "none"}')

    if n_split:
        coords_raw, embed_dict, indices_dict, query_reference_list = \
            split_dicts(coords_raw, embed_dict, n_split)
    else:
        indices_dict = None
        query_reference_list = {
            k: [k, ref_section] for k in query_keys}

    coords_affine = {}
    coords_ffd = {}
    for sample in sorted(query_reference_list.keys()):
        cache_path = f'{stack_dir}/{sample}.pt'
        if os.path.exists(cache_path):
            print(f'[{name}] loading cached {sample}')
            coords_affine[sample], coords_ffd[sample] = \
                torch.load(cache_path, weights_only=False)
            continue

        print(f'[{name}] aligning {sample}...')
        os.makedirs(f'{stack_dir}/{sample}', exist_ok=True)
        params_dist = CAST.reg_params(
            dataname=query_reference_list[sample],
            gpu=0 if torch.cuda.is_available() else -1,
            diff_step=5, **sp)
        params_dist.alpha_basis = torch.Tensor(
            [1/1000, 1/1000, 1/50, 5, 5]).reshape(5, 1).to(params_dist.device)

        coords_affine[sample], coords_ffd[sample] = \
            CAST.CAST_STACK(
                coords_raw, embed_dict,
                f'{stack_dir}/{sample}',
                query_reference_list[sample],
                params_dist,
                mid_visual=False,
                rescale=True)
        torch.save((coords_affine[sample], coords_ffd[sample]), cache_path)

    # collapse splits back to sample level
    if n_split:
        coords_affine = collapse_dicts(coords_affine, indices_dict, n_split)
        coords_ffd = collapse_dicts(coords_ffd, indices_dict, n_split)
    else:
        coords_affine = {k: np.asarray(
            next(v for kk, v in coords_affine[k].items()
                 if not kk.startswith('C57BL6J')), dtype=np.float32)
            for k in query_keys}
        coords_ffd = {k: np.asarray(
            next(v for kk, v in coords_ffd[k].items()
                 if not kk.startswith('C57BL6J')), dtype=np.float32)
            for k in query_keys}

    torch.save(coords_affine, f'{output_dir}/coords_affine.pt')
    torch.save(coords_ffd, f'{output_dir}/coords_ffd.pt')

    # plot overlays
    ref_coords = coords_raw[ref_section]
    fig_dir = f''

    for coord_type, coord_dict in [('affine', coords_affine),
                                    ('ffd', coords_ffd)]:
        samples = sorted(coord_dict.keys())
        ncols = min(5, len(samples))
        nrows = int(np.ceil(len(samples) / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5 * ncols, 5 * nrows), squeeze=False)
        fig.suptitle(f'{name} CAST-STACK {coord_type}', fontsize=16)
        for i, s in enumerate(samples):
            ax = axes[i // ncols, i % ncols]
            ax.scatter(ref_coords[:, 0], ref_coords[:, 1],
                       s=0.3, c='lightgray', alpha=0.2)
            ax.scatter(coord_dict[s][:, 0], coord_dict[s][:, 1],
                       s=0.5, c='red', alpha=0.3)
            ax.set_title(s)
            ax.set_aspect('equal')
            ax.axis('off')
        for j in range(len(samples), nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)
        plt.tight_layout()
        fig.savefig(f'{working_dir}/figures/{name}_stack_{coord_type}.png',
                    dpi=200)
        plt.close()

    total = sum(c.shape[0] for c in coords_ffd.values())
    print(f'[{name}] done — {total:,} cells aligned')

#endregion
#region run ####################################################################

def run_merfish():
    run_cast_stack('merfish', n_split=5,
                   stack_params={'iterations_bs': [100]})

def run_slidetags():
    run_cast_stack('slidetags', n_split=None)

def run_xenium():
    run_cast_stack('xenium', n_split=5,
                   stack_params={'iterations_bs': [200]})

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
