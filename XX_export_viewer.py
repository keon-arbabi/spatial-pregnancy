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
#   bytes [        0 ..  4n)   x                float32   (ref: x_raw, query: x_ffd)
#   bytes [       4n ..  8n)   y                float32   (ref: y_raw, query: y_ffd)
#   bytes [       8n .. 10n)   subclass_id      uint16    index into palette
#   bytes [      10n .. 11n)   class_id         uint8     index into palette
#   bytes [      11n .. 12n)   subclass_conf    uint8     round(255*v) (1.0 for ref)
#   bytes [      12n .. 13n)   subclass_margin  uint8     round(255*v) (1.0 for ref)
#   bytes [      13n .. 14n)   min_cos_dist     uint8     round(255*v) (0.0 for ref)
# Total size = 14 * n bytes.

import os
import json
import h5py
import numpy as np
import pandas as pd

working_dir = '/home/karbabi/spatial-pregnancy'
out_dir = f'{working_dir}/viewer_data'
os.makedirs(out_dir, exist_ok=True)

datasets = ['merfish', 'slidetags', 'xenium']
ref_path = f'{working_dir}/input/adata_ref_zeng_raw.h5ad'
metadata_csv = '/home/karbabi/single-cell/ABC/metadata/cells_joined.csv'

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


def pack_sample(x, y, class_ids, subclass_ids, conf, margin, cos):
    """Pack one sample into a 14*n byte buffer with the layout above."""
    n = len(x)
    buf = bytearray(14 * n)
    view = memoryview(buf)
    np.frombuffer(view[0 * n:4 * n], dtype=np.float32)[:] = \
        x.astype(np.float32, copy=False)
    np.frombuffer(view[4 * n:8 * n], dtype=np.float32)[:] = \
        y.astype(np.float32, copy=False)
    np.frombuffer(view[8 * n:10 * n], dtype=np.uint16)[:] = \
        subclass_ids.astype(np.uint16, copy=False)
    np.frombuffer(view[10 * n:11 * n], dtype=np.uint8)[:] = \
        class_ids.astype(np.uint8, copy=False)
    np.frombuffer(view[11 * n:12 * n], dtype=np.uint8)[:] = \
        conf.astype(np.uint8, copy=False)
    np.frombuffer(view[12 * n:13 * n], dtype=np.uint8)[:] = \
        margin.astype(np.uint8, copy=False)
    np.frombuffer(view[13 * n:14 * n], dtype=np.uint8)[:] = \
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
    'version': 2,
    'class_palette': class_palette,
    'subclass_palette': subclass_palette,
    'reference': None,
    'datasets': {},
    'binary_layout': {
        'fields': [
            {'name': 'x', 'dtype': 'float32', 'count': 'n'},
            {'name': 'y', 'dtype': 'float32', 'count': 'n'},
            {'name': 'subclass_id', 'dtype': 'uint16', 'count': 'n'},
            {'name': 'class_id', 'dtype': 'uint8', 'count': 'n'},
            {'name': 'subclass_confidence', 'dtype': 'uint8',
             'count': 'n', 'scale': 1 / 255},
            {'name': 'subclass_margin', 'dtype': 'uint8',
             'count': 'n', 'scale': 1 / 255},
            {'name': 'min_cos_dist', 'dtype': 'uint8',
             'count': 'n', 'scale': 1 / 255},
        ],
        'bytes_per_cell': 14,
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

    buf = pack_sample(
        x_s, y_s,
        ref_class_ids[m], ref_subclass_ids[m],
        full, full, zero)

    short = section.split('.')[-1]  # e.g. '46'
    rel = f'reference/{short}.bin'
    with open(f'{out_dir}/{rel}', 'wb') as fh:
        fh.write(buf)

    ref_section_list.append({
        'section': section,
        'short': short,
        'n_cells': n,
        'x_range': [float(x_s.min()), float(x_s.max())],
        'y_range': [float(y_s.min()), float(y_s.max())],
        'file': rel,
    })

manifest['reference'] = {
    'n_cells': int(len(ref_x)),
    'x_range': [float(ref_x.min()), float(ref_x.max())],
    'y_range': [float(ref_y.min()), float(ref_y.max())],
    'sections': ref_section_list,
}
print(f'[reference] wrote {len(ref_section_list)} sections '
      f'({len(ref_x):,} cells)')

#endregion
#region per-dataset query export ###############################################

cols_to_read = [
    'sample_rep', 'sample', 'condition', 'class', 'subclass',
    'subclass_confidence', 'subclass_margin', 'min_cos_dist',
    'x_ffd', 'y_ffd',
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
    x = obs['x_ffd'].astype(np.float32)
    y = obs['y_ffd'].astype(np.float32)
    conf = obs['subclass_confidence'].astype(np.float32)
    margin = obs['subclass_margin'].astype(np.float32)
    cos = obs['min_cos_dist'].astype(np.float32)

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

        x_s = x[m]
        y_s = y[m]

        buf = pack_sample(
            x_s, y_s,
            class_ids[m], subclass_ids[m],
            conf_q[m], margin_q[m], cos_q[m])

        rel = f'{name}/{rep}.bin'
        with open(f'{out_dir}/{rel}', 'wb') as fh:
            fh.write(buf)

        first = int(np.argmax(m))
        samples.append({
            'sample_rep': rep,
            'sample': str(sample[first]),
            'condition': str(condition[first]),
            'n_cells': n,
            'x_range': [float(x_s.min()), float(x_s.max())],
            'y_range': [float(y_s.min()), float(y_s.max())],
            'file': rel,
        })

    manifest['datasets'][name] = {
        'n_cells': int(n_total),
        'x_range': [float(x.min()), float(x.max())],
        'y_range': [float(y.min()), float(y.max())],
        'samples': samples,
    }

    print(f'[{name}] wrote {len(samples)} samples '
          f'({sum(s["n_cells"] for s in samples):,} cells)')

with open(f'{out_dir}/manifest.json', 'w') as fh:
    json.dump(manifest, fh, indent=2)

print(f'\nmanifest written to {out_dir}/manifest.json')
print('done')

#endregion
