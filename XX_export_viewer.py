#region imports and setup ######################################################

# Builds a compact, web-friendly export of the postprocessed spatial datasets
# (merfish / slidetags / xenium) plus the Zeng MERFISH reference for use by
# viewer.html.
#
# Output layout under viewer_data/:
#   manifest.json                      schema, palettes, per-sample index
#   reference/{section}.bin            packed binary per ref section
#   {dataset}/{sample_rep}.bin         packed binary per query sample
#
# Per-sample binary layout (little-endian, n = n_cells for that sample):
#   bytes [        0 ..  4n)   x_ffd            float32
#   bytes [       4n ..  8n)   y_ffd            float32
#   bytes [       8n .. 12n)   x_affine         float32   (= ffd for ref)
#   bytes [      12n .. 16n)   y_affine         float32   (= ffd for ref)
#   bytes [      16n .. 20n)   total_counts     float32   per-cell raw library size
#   bytes [      20n .. 24n)   avg_pdist        float32   avg distance to ref neighbors
#   bytes [      24n .. 26n)   subclass_id      uint16    index into palette
#   bytes [      26n .. 27n)   class_id         uint8     index into palette
#   bytes [      27n .. 28n)   subclass_conf    uint8     round(255*v) (1.0 for ref)
#   bytes [      28n .. 29n)   subclass_margin  uint8     round(255*v) (1.0 for ref)
#   bytes [      29n .. 30n)   min_cos_dist     uint8     round(255*v) (0.0 for ref)
# Total size = 30 * n bytes.

import os
import json
import h5py
import numpy as np
import pandas as pd
import torch

working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/viewer_data'
os.makedirs(out_dir, exist_ok=True)

# one-shot env diagnostic so permission failures are easy to debug
import getpass, socket
_um = os.umask(0o022); os.umask(_um)
print(f'[env] host={socket.gethostname()} '
      f'user={getpass.getuser()} uid={os.getuid()} euid={os.geteuid()} '
      f'umask={oct(_um)}')
for _d in ['', '/reference', '/xenium', '/merfish', '/slidetags']:
    _p = out_dir + _d
    print(f'[env] writable {_p}: {os.access(_p, os.W_OK) if os.path.exists(_p) else "missing"}')

datasets = ['xenium', 'merfish', 'slidetags']
ref_path = f'{working_dir}/input/adata_ref_zeng_raw.h5ad'
metadata_csv = '/home/karbabi/single-cell/ABC/metadata/cells_joined.csv'

# which obs column was used as the matching key during CAST-STACK alignment.
# (must match `sample_col` in 04_project_cell_types.py per dataset)
sample_col_map = {
    'xenium':    'sample_rep',
    'merfish':   'sample',
    'slidetags': 'sample',
}

# per-dataset samples to drop (matched against obs['sample'])
# keep in sync with drop_samples in 06_sumrank.py
drop_samples_map = {
    'xenium': ['CTRL_3'],
    'merfish': [],
    'slidetags': [],
}

# per-gene expression export. viewer.html keys its "gene expression" panel off
# manifest['datasets'][name]['genes'], so with this False the binaries are not
# written, the key is omitted, and the panel is hidden in the viewer.
export_genes = False

#endregion
#region helpers ################################################################

def read_obs_column(obs_grp, col):
    """Return a numpy array of obs[col]. Decodes categoricals to strings."""
    v = obs_grp[col]
    if isinstance(v, h5py.Group):
        cats = v['categories'][:]
        codes = v['codes'][:]
        if cats.dtype.kind in ('S', 'O'):
            cats = np.array(
                [c.decode() if isinstance(c, bytes) else str(c) for c in cats])
        return np.where(codes >= 0, cats[np.clip(codes, 0, None)], '')
    return v[:]


def read_obs(path, columns):
    with h5py.File(path, 'r') as f:
        obs = f['obs']
        return {c: read_obs_column(obs, c) for c in columns if c in obs}


def quantize_unit(arr, lo=0.0, hi=1.0):
    """Map float [lo,hi] -> uint8 [0,255], clipping out-of-range."""
    a = np.clip(np.asarray(arr, dtype=np.float32), lo, hi)
    return np.round((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def write_atomic(path, data):
    """Write to a temp sibling and os.replace over the destination.
    Works even when the existing file isn't writable by us, as long as
    the parent directory is."""
    tmp = f'{path}.tmp.{os.getpid()}'
    with open(tmp, 'wb') as fh:
        fh.write(data)
    os.replace(tmp, path)


def f32pair_to_u64(x, y):
    """Bit-exact (float32, float32) -> uint64 key for hashing."""
    xb = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    yb = np.ascontiguousarray(y, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return (xb << np.uint64(32)) | yb


def gather_affine_coords(name, sample_key_arr, x_ffd, y_ffd):
    """For each query cell (with its dataset-level sample key, x_ffd, y_ffd),
    return matching (x_affine, y_affine) by looking up the row in
    coords_ffd[s] then reading from coords_affine[s] at that index.

    Falls back to ffd coords for unmatched cells (and prints a warning)."""
    n = len(sample_key_arr)
    x_aff = x_ffd.copy()
    y_aff = y_ffd.copy()

    ffd_path = f'{working_dir}/output/{name}/coords_ffd.pt'
    aff_path = f'{working_dir}/output/{name}/coords_affine.pt'
    if not (os.path.exists(ffd_path) and os.path.exists(aff_path)):
        print(f'  [{name}] no coords_*.pt — affine view will mirror ffd')
        return x_aff, y_aff

    cffd = torch.load(ffd_path, weights_only=False)
    caff = torch.load(aff_path, weights_only=False)

    matched = 0
    for s in np.unique(sample_key_arr):
        if s not in cffd or s not in caff:
            continue
        cf = np.asarray(cffd[s], dtype=np.float32)
        ca = np.asarray(caff[s], dtype=np.float32)
        # bit-exact (x_ffd, y_ffd) -> row index in coords_ffd[s]
        ref_keys = f32pair_to_u64(cf[:, 0], cf[:, 1])
        idx_map = dict(zip(ref_keys.tolist(), range(ref_keys.shape[0])))

        m = (sample_key_arr == s)
        cell_idx = np.where(m)[0]
        q_keys = f32pair_to_u64(x_ffd[cell_idx], y_ffd[cell_idx]).tolist()
        for i, k in zip(cell_idx, q_keys):
            j = idx_map.get(int(k))
            if j is not None:
                x_aff[i] = ca[j, 0]
                y_aff[i] = ca[j, 1]
                matched += 1

    if matched < n:
        print(f'  [{name}] affine match: {matched:,}/{n:,} '
              f'({100*matched/max(n,1):.1f}%) — unmatched cells fall back to ffd')
    else:
        print(f'  [{name}] affine match: {matched:,}/{n:,} (100%)')
    return x_aff, y_aff


def pack_sample(x_ffd, y_ffd, x_aff, y_aff, total_counts, avg_pdist,
                class_ids, subclass_ids, conf, margin, cos):
    """Pack one sample into a 30*n byte buffer with the layout above."""
    n = len(x_ffd)
    buf = bytearray(30 * n)
    view = memoryview(buf)
    np.frombuffer(view[0  * n:4  * n], dtype=np.float32)[:] = \
        x_ffd.astype(np.float32, copy=False)
    np.frombuffer(view[4  * n:8  * n], dtype=np.float32)[:] = \
        y_ffd.astype(np.float32, copy=False)
    np.frombuffer(view[8  * n:12 * n], dtype=np.float32)[:] = \
        x_aff.astype(np.float32, copy=False)
    np.frombuffer(view[12 * n:16 * n], dtype=np.float32)[:] = \
        y_aff.astype(np.float32, copy=False)
    np.frombuffer(view[16 * n:20 * n], dtype=np.float32)[:] = \
        total_counts.astype(np.float32, copy=False)
    np.frombuffer(view[20 * n:24 * n], dtype=np.float32)[:] = \
        avg_pdist.astype(np.float32, copy=False)
    np.frombuffer(view[24 * n:26 * n], dtype=np.uint16)[:] = \
        subclass_ids.astype(np.uint16, copy=False)
    np.frombuffer(view[26 * n:27 * n], dtype=np.uint8)[:] = \
        class_ids.astype(np.uint8, copy=False)
    np.frombuffer(view[27 * n:28 * n], dtype=np.uint8)[:] = \
        conf.astype(np.uint8, copy=False)
    np.frombuffer(view[28 * n:29 * n], dtype=np.uint8)[:] = \
        margin.astype(np.uint8, copy=False)
    np.frombuffer(view[29 * n:30 * n], dtype=np.uint8)[:] = \
        cos.astype(np.uint8, copy=False)
    return buf

#endregion
#region build palettes from cells_joined.csv ###################################
# uses the project's exact color_mappings convention

print('loading class/subclass colors from ABC metadata...')
cells_joined = pd.read_csv(
    metadata_csv,
    usecols=['class', 'class_color', 'subclass', 'subclass_color'])

color_mappings = {
    'class': dict(zip(
        cells_joined['class'].str.replace('/', '_'),
        cells_joined['class_color'])),
    'subclass': {k.replace('_', '/'): v for k, v in dict(zip(
        cells_joined['subclass'].str.replace('/', '_'),
        cells_joined['subclass_color'])).items()}
}
for level in color_mappings:
    color_mappings[level]['Unlabelled'] = '#d3d3d3'
del cells_joined


def lookup_class_color(name):
    # color_mappings['class'] keys have '/' replaced by '_'
    return color_mappings['class'].get(
        name.replace('/', '_'), '#d3d3d3')


def lookup_subclass_color(name):
    # color_mappings['subclass'] keys are normalized round-trip; equivalent
    # to keying by the original name in our data
    return color_mappings['subclass'].get(name, '#d3d3d3')


# collect every class/subclass actually present in any of the datasets + ref
present_classes = set()
present_subclasses = set()

for name in datasets:
    p = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    obs = read_obs(p, ['class', 'subclass'])
    present_classes.update(np.unique(obs['class']).tolist())
    present_subclasses.update(np.unique(obs['subclass']).tolist())

ref_obs_lc = read_obs(ref_path, ['class', 'subclass'])
present_classes.update(np.unique(ref_obs_lc['class']).tolist())
present_subclasses.update(np.unique(ref_obs_lc['subclass']).tolist())

print(f'  {len(present_classes)} classes, {len(present_subclasses)} subclasses '
      f'present (data + ref)')


def sort_by_prefix(name):
    """Sort '01 IT-ET Glut' / '001 ...' by leading numeric prefix."""
    head = name.split(' ', 1)[0]
    try:
        return (int(head), name)
    except ValueError:
        return (10**9, name)


class_list = sorted(present_classes, key=sort_by_prefix)
subclass_list = sorted(present_subclasses, key=sort_by_prefix)

# index 0 reserved for 'Unlabelled' so palette lookup is always direct
class_palette = [{'name': 'Unlabelled', 'color': '#d3d3d3'}] + [
    {'name': c, 'color': lookup_class_color(c)} for c in class_list]
subclass_palette = [{'name': 'Unlabelled', 'color': '#d3d3d3'}] + [
    {'name': s, 'color': lookup_subclass_color(s)} for s in subclass_list]

class_to_id = {p['name']: i for i, p in enumerate(class_palette)}
subclass_to_id = {p['name']: i for i, p in enumerate(subclass_palette)}

assert len(subclass_palette) < 2**16, 'too many subclasses for u16'
assert len(class_palette) < 2**8, 'too many classes for u8'

#endregion
#region manifest skeleton ######################################################

manifest = {
    'version': 5,
    'class_palette': class_palette,
    'subclass_palette': subclass_palette,
    'reference': None,
    'datasets': {},
    'binary_layout': {
        'fields': [
            {'name': 'x_ffd',    'dtype': 'float32', 'count': 'n'},
            {'name': 'y_ffd',    'dtype': 'float32', 'count': 'n'},
            {'name': 'x_affine', 'dtype': 'float32', 'count': 'n'},
            {'name': 'y_affine', 'dtype': 'float32', 'count': 'n'},
            {'name': 'total_counts', 'dtype': 'float32', 'count': 'n'},
            {'name': 'avg_pdist',    'dtype': 'float32', 'count': 'n'},
            {'name': 'subclass_id', 'dtype': 'uint16', 'count': 'n'},
            {'name': 'class_id',    'dtype': 'uint8',  'count': 'n'},
            {'name': 'subclass_confidence', 'dtype': 'uint8',
             'count': 'n', 'scale': 1 / 255},
            {'name': 'subclass_margin', 'dtype': 'uint8',
             'count': 'n', 'scale': 1 / 255},
            {'name': 'min_cos_dist', 'dtype': 'uint8',
             'count': 'n', 'scale': 1 / 255},
        ],
        'bytes_per_cell': 30,
    },
}

#endregion
#region reference export #######################################################

print('\n[reference] reading Zeng MERFISH ref obs...')
ref_obs = read_obs(
    ref_path, ['sample', 'class', 'subclass', 'x_raw', 'y_raw'])
ref_sample = ref_obs['sample'].astype(str)
ref_cls = ref_obs['class'].astype(str)
ref_sub = ref_obs['subclass'].astype(str)
ref_x = ref_obs['x_raw'].astype(np.float32)
ref_y = ref_obs['y_raw'].astype(np.float32)

ref_class_ids = np.array(
    [class_to_id.get(c, 0) for c in ref_cls], dtype=np.uint8)
ref_subclass_ids = np.array(
    [subclass_to_id.get(s, 0) for s in ref_sub], dtype=np.uint16)

ref_dir = f'{out_dir}/reference'
os.makedirs(ref_dir, exist_ok=True)

ref_section_list = []
for section in sorted(np.unique(ref_sample).tolist()):
    m = ref_sample == section
    n = int(m.sum())
    x_s = ref_x[m]
    y_s = ref_y[m]

    # ref has no QC scores; fill with full-confidence sentinels
    full = np.full(n, 255, dtype=np.uint8)
    zero = np.zeros(n, dtype=np.uint8)

    # ref isn't transformed: ffd == affine == raw
    # ref has no library size / pdist; fill with zeros
    zero_counts = np.zeros(n, dtype=np.float32)
    zero_pdist = np.zeros(n, dtype=np.float32)
    buf = pack_sample(
        x_s, y_s, x_s, y_s, zero_counts, zero_pdist,
        ref_class_ids[m], ref_subclass_ids[m],
        full, full, zero)

    short = section.split('.')[-1]  # e.g. '46'
    rel = f'reference/{short}.bin'
    write_atomic(f'{out_dir}/{rel}', buf)

    rng = {
        'ffd':    [[float(x_s.min()), float(x_s.max())],
                   [float(y_s.min()), float(y_s.max())]],
        'affine': [[float(x_s.min()), float(x_s.max())],
                   [float(y_s.min()), float(y_s.max())]],
    }
    ref_section_list.append({
        'section': section,
        'short': short,
        'n_cells': n,
        'x_range_ffd':    rng['ffd'][0],    'y_range_ffd':    rng['ffd'][1],
        'x_range_affine': rng['affine'][0], 'y_range_affine': rng['affine'][1],
        'file': rel,
    })

manifest['reference'] = {
    'n_cells': int(len(ref_x)),
    'x_range_ffd':    [float(ref_x.min()), float(ref_x.max())],
    'y_range_ffd':    [float(ref_y.min()), float(ref_y.max())],
    'x_range_affine': [float(ref_x.min()), float(ref_x.max())],
    'y_range_affine': [float(ref_y.min()), float(ref_y.max())],
    'sections': ref_section_list,
}
print(f'[reference] wrote {len(ref_section_list)} sections '
      f'({len(ref_x):,} cells)')

#endregion
#region per-dataset query export ###############################################

cols_to_read = [
    'sample_rep', 'sample', 'condition', 'class', 'subclass',
    'subclass_confidence', 'subclass_margin', 'min_cos_dist',
    'x_ffd', 'y_ffd', 'total_counts', 'avg_pdist',
]

for name in datasets:
    print(f'\n[{name}] reading obs...')
    p = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    obs = read_obs(p, cols_to_read)
    n_total = len(obs['x_ffd'])
    print(f'[{name}] {n_total:,} cells')

    sample_rep = obs['sample_rep'].astype(str)
    sample = obs['sample'].astype(str)
    condition = obs['condition'].astype(str)
    cls = obs['class'].astype(str)
    sub = obs['subclass'].astype(str)
    x_ffd_arr = obs['x_ffd'].astype(np.float32)
    y_ffd_arr = obs['y_ffd'].astype(np.float32)
    total_counts = obs['total_counts'].astype(np.float32)
    avg_pdist = obs['avg_pdist'].astype(np.float32)
    conf = obs['subclass_confidence'].astype(np.float32)
    margin = obs['subclass_margin'].astype(np.float32)
    cos = obs['min_cos_dist'].astype(np.float32)

    # drop samples listed for this dataset (keep in sync with 06_sumrank.py)
    drop = drop_samples_map.get(name, [])
    if drop:
        keep_mask = ~np.isin(sample, drop)
        n_before = len(sample)
        sample_rep = sample_rep[keep_mask]
        sample = sample[keep_mask]
        condition = condition[keep_mask]
        cls = cls[keep_mask]
        sub = sub[keep_mask]
        x_ffd_arr = x_ffd_arr[keep_mask]
        y_ffd_arr = y_ffd_arr[keep_mask]
        total_counts = total_counts[keep_mask]
        avg_pdist = avg_pdist[keep_mask]
        conf = conf[keep_mask]
        margin = margin[keep_mask]
        cos = cos[keep_mask]
        n_total = len(sample)
        print(f'[{name}] dropped samples {drop}: '
              f'{n_before:,} → {n_total:,} cells')

    # match each cell to its row in coords_affine.pt via coords_ffd.pt
    sample_col = sample_col_map.get(name, 'sample')
    sample_key = (sample_rep if sample_col == 'sample_rep' else sample)
    x_aff_arr, y_aff_arr = gather_affine_coords(
        name, sample_key, x_ffd_arr, y_ffd_arr)

    class_ids = np.array(
        [class_to_id.get(c, 0) for c in cls], dtype=np.uint8)
    subclass_ids = np.array(
        [subclass_to_id.get(s, 0) for s in sub], dtype=np.uint16)
    conf_q = quantize_unit(conf)
    margin_q = quantize_unit(margin)
    cos_q = quantize_unit(cos)

    ds_dir = f'{out_dir}/{name}'
    os.makedirs(ds_dir, exist_ok=True)

    samples = []
    unique_reps = sorted(np.unique(sample_rep).tolist(), key=str)
    for rep in unique_reps:
        m = sample_rep == rep
        n = int(m.sum())
        if n == 0:
            continue

        xf = x_ffd_arr[m]; yf = y_ffd_arr[m]
        xa = x_aff_arr[m]; ya = y_aff_arr[m]
        tc = total_counts[m]
        pdist_s = avg_pdist[m]

        buf = pack_sample(
            xf, yf, xa, ya, tc, pdist_s,
            class_ids[m], subclass_ids[m],
            conf_q[m], margin_q[m], cos_q[m])

        rel = f'{name}/{rep}.bin'
        write_atomic(f'{out_dir}/{rel}', buf)

        first = int(np.argmax(m))
        samples.append({
            'sample_rep': rep,
            'sample': str(sample[first]),
            'condition': str(condition[first]),
            'n_cells': n,
            'x_range_ffd':    [float(xf.min()), float(xf.max())],
            'y_range_ffd':    [float(yf.min()), float(yf.max())],
            'x_range_affine': [float(xa.min()), float(xa.max())],
            'y_range_affine': [float(ya.min()), float(ya.max())],
            'file': rel,
        })

    manifest['datasets'][name] = {
        'n_cells': int(n_total),
        'x_range_ffd':    [float(x_ffd_arr.min()), float(x_ffd_arr.max())],
        'y_range_ffd':    [float(y_ffd_arr.min()), float(y_ffd_arr.max())],
        'x_range_affine': [float(x_aff_arr.min()), float(x_aff_arr.max())],
        'y_range_affine': [float(y_aff_arr.min()), float(y_aff_arr.max())],
        'samples': samples,
    }

    print(f'[{name}] wrote {len(samples)} samples '
          f'({sum(s["n_cells"] for s in samples):,} cells)')

#endregion
#region per-gene expression export #############################################

if not export_genes:
    print('\n[genes] export disabled (export_genes = False); no per-gene '
          'binaries written and no gene lists added to the manifest, so the '
          'gene expression panel is hidden in viewer.html')
else:
    import re
    import scipy.sparse as sp

    TARGET_SUM = 1e4  # library-size normalize to this

    # filter genes the same way 06_sumrank.py does for slidetags
    # (protein-coding AND NOT mt AND NOT ribo). for merfish/xenium the gene panel
    # is curated so no filter is applied.
    filter_genes_map = {
        'xenium':    False,
        'merfish':   False,
        'slidetags': True,
    }

    # load protein-coding gene list (same source as 06_sumrank.py)
    _pc_df = pd.read_csv(
        f'{working_dir}/input/MRK_ENSEMBL.csv', header=None)
    PROTEIN_CODING = set(
        _pc_df[_pc_df[8] == 'protein coding gene'][1].astype(str).tolist())
    del _pc_df
    print(f'[filter] {len(PROTEIN_CODING):,} protein-coding gene symbols loaded')

    _mt_re = re.compile(r'^(mt-|MT-)')
    _ribo_re = re.compile(r'^(Rps|Rpl)')

    def is_keep_gene(g):
        return (g in PROTEIN_CODING
                and not _mt_re.match(g)
                and not _ribo_re.match(g))

    for name in datasets:
        print(f'\n[{name}] exporting gene expression...')
        p = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
        gene_dir = f'{out_dir}/{name}/genes'
        os.makedirs(gene_dir, exist_ok=True)

        with h5py.File(p, 'r') as f:
            # read gene names from var
            var = f['var']
            if '_index' in var:
                gene_names = var['_index'][:]
            else:
                gene_names = var[list(var.keys())[0]][:]
            gene_names = [g.decode() if isinstance(g, bytes) else str(g)
                          for g in gene_names]
            n_genes = len(gene_names)

            # read sample_rep + sample for ordering and drop filter
            obs = f['obs']
            sample_rep_arr = read_obs_column(obs, 'sample_rep').astype(str)
            sample_arr_full = read_obs_column(obs, 'sample').astype(str)
            n_cells = len(sample_rep_arr)

            # keep mask (drop samples listed for this dataset)
            drop = drop_samples_map.get(name, [])
            keep = ~np.isin(sample_arr_full, drop) \
                if drop else np.ones(n_cells, dtype=bool)
            # build cell ordering: within-dataset sorted by sample_rep, KEEP cells
            sorted_reps = sorted(np.unique(sample_rep_arr[keep]))
            gene_order = np.concatenate(
                [np.where((sample_rep_arr == rep) & keep)[0]
                 for rep in sorted_reps])

            # read sparse X matrix (CSR) — RAW counts
            X_grp = f['X']
            data = X_grp['data'][:].astype(np.float32)
            indices = X_grp['indices'][:]
            indptr = X_grp['indptr'][:]
            X = sp.csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))
            del data, indices, indptr

        # filter genes (06_sumrank.py convention: protein_coding AND NOT mt
        # AND NOT ribo). only applied to datasets flagged in filter_genes_map.
        if filter_genes_map.get(name, False):
            keep_gene_idx = [j for j, g in enumerate(gene_names) if is_keep_gene(g)]
            kept_names = [gene_names[j] for j in keep_gene_idx]
            print(f'[{name}] gene filter: '
                  f'{len(kept_names):,}/{n_genes:,} kept '
                  f'(protein_coding ∧ ¬mt ∧ ¬ribo)')
            X = X[:, keep_gene_idx]
            gene_names = kept_names
            n_genes = len(gene_names)

        # convert to CSC for efficient column access
        print(f'[{name}] converting to CSC ({n_genes:,} genes, '
              f'{X.nnz:,} nnz)...')
        X = X.tocsc()

        # write per-gene binary = raw counts as float32, in sample_rep order.
        # client derives: per-cell log2(count/total_counts*1e4+1) for coloring,
        # per-sample pseudobulk log2(sum_count/n_cells*1e4+1) for violin.
        for j in range(n_genes):
            gene = gene_names[j]
            col = X[:, j].toarray().ravel().astype(np.float32)
            expr_ordered = col[gene_order]
            write_atomic(f'{gene_dir}/{gene}.bin', expr_ordered.tobytes())
            if (j + 1) % 500 == 0 or j + 1 == n_genes:
                print(f'[{name}]   {j+1:,}/{n_genes:,} genes')

        # add gene list to manifest
        manifest['datasets'][name]['genes'] = gene_names
        del X
        print(f'[{name}] wrote {n_genes:,} gene files to {gene_dir}/')

write_atomic(
    f'{out_dir}/manifest.json',
    json.dumps(manifest, indent=2).encode('utf-8'))

print(f'\nmanifest written to {out_dir}/manifest.json')
print('done')

#endregion
