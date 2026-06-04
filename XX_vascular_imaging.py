"""From-scratch vascular morphometrics from the Kalish CD31 imaging.

Reads per-ROI binary masks + skeletons in-memory from the zip, computes
scale-free vessel metrics (no manual tissue ROI needed), cross-checks
against the collaborator CSV, and exports a representative pipeline
montage for the figure.
"""
import os
import io
import re
import zipfile

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import mannwhitneyu, spearmanr
from skimage.morphology import (skeletonize, convex_hull_image,
                                 binary_dilation, disk)
from skan import Skeleton, summarize

working_dir = '/home/karbabi/spatial-pregnancy'
ZIP = f'{working_dir}/input/vascular_imaging.zip'
OUT = f'{working_dir}/output/vascular_imaging'
os.makedirs(OUT, exist_ok=True)
BASE = ('SDP_vascular_analysis_KalishCollaboration/'
        'SDP_KalishCollaboration/')
DEFAULT_PX_UM = 0.5687
FOV_UM = 500.0
METRICS = ['junctions_per_mm_vessel', 'junctions_per_mm2_hull',
           'mean_tortuosity', 'mean_vessel_diameter_um',
           'vessel_area_fraction_hull', 'lacunarity_mean']

z = zipfile.ZipFile(ZIP)


def parse(name):
    p = name.split('/')
    idx = [k for k, x in enumerate(p)
           if 'mouse' in x and x.split()[0] in ('nulliparous', 'pregnant')]
    if not idx:
        return None
    i = idx[0]
    cm, roi, fname = p[i], p[i + 1] if i + 1 < len(p) else '', p[-1]
    cond = 'Nulliparous' if cm.startswith('nulliparous') else 'Pregnant'
    mnum = int(re.search(r'mouse\s*(\d+)', cm).group(1))
    rm = re.search(r'ROI\s*(\d+)', roi) or re.search(r'ROI\s*(\d+)', fname)
    return cond, mnum, int(rm.group(1)), fname


rois = {}
for n in z.namelist():
    if not n.lower().endswith('.tif') or 'DS_Store' in n:
        continue
    pr = parse(n)
    if pr is None:
        continue
    cond, mnum, rnum, fname = pr
    d = rois.setdefault((cond, mnum, rnum), {})
    fl = fname.lower()
    if 'binary' in fl or 'threshold' in fl:
        d['mask'] = n
    elif 'skeleton' in fl:
        d['skel'] = n
    elif 'processed' in fl:
        d['proc'] = n
print(f'found {len(rois)} ROIs')


def read_tif(name):
    arr = tifffile.imread(io.BytesIO(z.read(name)))
    return arr[..., 0] if arr.ndim > 2 else arr


def px_um_of(name):
    with tifffile.TiffFile(io.BytesIO(z.read(name))) as tf:
        t = tf.pages[0].tags.get('XResolution')
        if t is not None:
            num, den = t.value
            if num and den and num / den > 0:
                return den / num
    return None


def lacunarity(mask, hull):
    m, h = mask.astype(np.float64), hull.astype(np.float64)
    vals = []
    for r in [4, 8, 16, 32, 64, 128]:
        if r >= min(mask.shape):
            break
        s = ndi.uniform_filter(m, size=r, mode='constant') * (r * r)
        inside = ndi.uniform_filter(h, size=r, mode='constant') >= 0.999
        s = s[inside]
        if s.size < 10 or s.mean() <= 0:
            continue
        vals.append(1.0 + s.var() / s.mean() ** 2)
    return float(np.mean(vals)) if vals else np.nan


def skel_col(df, base):
    for c in (base, base.replace('-', '_')):
        if c in df.columns:
            return c
    raise KeyError(base)


def compute(mask_name, skel_name, pxum):
    mask = read_tif(mask_name) > 0
    skel = skeletonize(mask)
    if skel.sum() < 10:
        return None
    nb = (ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8),
                       mode='constant') - skel.astype(np.uint8))
    n_junc = ndi.label(skel & (nb >= 3), structure=np.ones((3, 3)))[1]
    n_end = int((skel & (nb == 1)).sum())
    sk = Skeleton(skel, spacing=pxum)
    try:
        sdf = summarize(sk, separator='-')
    except TypeError:
        sdf = summarize(sk)
    arc = sdf[skel_col(sdf, 'branch-distance')].to_numpy()
    eu = sdf[skel_col(sdf, 'euclidean-distance')].to_numpy()
    length_mm = float(arc.sum()) / 1000.0
    ok = (eu > 2 * pxum) & (arc > 5 * pxum)
    tort = (float(np.average(arc[ok] / eu[ok], weights=arc[ok]))
            if ok.any() else np.nan)
    hull = convex_hull_image(mask)
    hull_mm2 = hull.sum() * (pxum ** 2) / 1e6
    dt = ndi.distance_transform_edt(mask) * pxum
    fiji = ndi.label(read_tif(skel_name) == 70,
                     structure=np.ones((3, 3)))[1]
    return dict(
        n_junctions=n_junc, n_endpoints=n_end, skeleton_length_mm=length_mm,
        junctions_per_mm_vessel=n_junc / length_mm if length_mm else np.nan,
        junctions_per_mm2_hull=n_junc / hull_mm2 if hull_mm2 else np.nan,
        mean_tortuosity=tort,
        mean_vessel_diameter_um=2 * float(dt[skel].mean()),
        vessel_area_fraction_hull=mask.sum() / hull.sum() * 100,
        vessel_area_fraction_frame=float(mask.mean()) * 100,
        lacunarity_mean=lacunarity(mask, hull),
        px_um=pxum, fiji_junctions=fiji,
        img_h=mask.shape[0], img_w=mask.shape[1])


records = []
for (cond, mnum, rnum), f in sorted(rois.items()):
    if 'mask' not in f or 'skel' not in f:
        print(f'  skip {cond} m{mnum} ROI{rnum}: missing files')
        continue
    pxum = px_um_of(f['skel']) or px_um_of(f.get('proc', f['skel'])) \
        or DEFAULT_PX_UM
    m = compute(f['mask'], f['skel'], pxum)
    if m is None:
        continue
    m.update(condition=cond, mouse=mnum, roi=rnum)
    records.append(m)
    print(f'  {cond[:4]} m{mnum} ROI{rnum}: '
          f'junc/mm={m["junctions_per_mm_vessel"]:.2f} '
          f'junc/mm2={m["junctions_per_mm2_hull"]:.0f} '
          f'tort={m["mean_tortuosity"]:.3f} '
          f'(skan_junc={m["n_junctions"]} fiji_junc={m["fiji_junctions"]})')

df = pd.DataFrame(records)
df = df[['condition', 'mouse', 'roi'] +
        [c for c in df.columns if c not in ('condition', 'mouse', 'roi')]]
df.to_csv(f'{OUT}/imaging_metrics.csv', index=False)
print(f'\nwrote {OUT}/imaging_metrics.csv ({len(df)} ROIs)')

stats = []
for met in METRICS:
    a = df[df.condition == 'Nulliparous'][met].dropna()
    b = df[df.condition == 'Pregnant'][met].dropna()
    p_roi = mannwhitneyu(a, b, alternative='two-sided').pvalue
    mm = df.groupby(['condition', 'mouse'])[met].mean().reset_index()
    am = mm[mm.condition == 'Nulliparous'][met].dropna()
    bm = mm[mm.condition == 'Pregnant'][met].dropna()
    p_mouse = (mannwhitneyu(am, bm, alternative='two-sided').pvalue
               if len(am) >= 1 and len(bm) >= 1 else np.nan)
    stats.append(dict(metric=met, mean_null=a.mean(), mean_preg=b.mean(),
                      direction='up' if b.mean() > a.mean() else 'down',
                      p_roi=p_roi, p_mouse=p_mouse))
stats = pd.DataFrame(stats)
stats.to_csv(f'{OUT}/imaging_stats.csv', index=False)
print('\n=== stats (PREG vs CTRL) ===')
print(stats.to_string(index=False))

csv = pd.read_csv(io.BytesIO(z.read(BASE + 'vascular_data.csv')))
csv['condition'] = csv['Condition']
csv['mouse'] = csv['Sample'].str.extract(r'Mouse (\d+)').astype(int)
csv['roi'] = csv['Sample'].str.extract(r'ROI(\d+)').astype(int)
mg = df.merge(csv, on=['condition', 'mouse', 'roi'], how='inner')
cc = []
for ours, theirs in [('junctions_per_mm2_hull', 'Junction_Density'),
                     ('junctions_per_mm_vessel', 'Junction_Density'),
                     ('mean_tortuosity', 'Mean_Tortuosity'),
                     ('vessel_area_fraction_hull', 'Percent_Vessel_Area'),
                     ('lacunarity_mean', 'Average_Lacunarity')]:
    rho, p = spearmanr(mg[ours], mg[theirs])
    cc.append(dict(ours=ours, theirs=theirs, spearman_rho=rho, p=p, n=len(mg)))
cc = pd.DataFrame(cc)
cc.to_csv(f'{OUT}/crosscheck_vs_csv.csv', index=False)
print('\n=== cross-check vs collaborator CSV (Spearman) ===')
print(cc.to_string(index=False))


def save_png(path, arr):
    Image.fromarray(arr.astype(np.uint8)).save(path)


_D4 = [lambda a: a, lambda a: np.rot90(a, 1), lambda a: np.rot90(a, 2),
       lambda a: np.rot90(a, 3), lambda a: a[:, ::-1], lambda a: a[::-1],
       lambda a: a.T, lambda a: a.T[:, ::-1]]


def _gray(p):
    return p[..., :3].max(-1) if p.ndim == 3 else p


def orient_to_mask(g, mask):
    best, best_g = -1.0, g
    for op in _D4:
        gg = op(g)
        h = min(gg.shape[0], mask.shape[0])
        w = min(gg.shape[1], mask.shape[1])
        if h < mask.shape[0] * 0.7 or w < mask.shape[1] * 0.7:
            continue
        gc = gg[:h, :w].astype(float)
        mb = mask[:h, :w] > 0
        if mb.sum() < 100:
            continue
        s = gc[mb].mean() / max(gc[~mb].mean(), 1e-9)
        if s > best:
            best, best_g = s, gg
    h = min(best_g.shape[0], mask.shape[0])
    w = min(best_g.shape[1], mask.shape[1])
    return best_g[:h, :w], best


def export_crop(f, pxum, prefix):
    mask = read_tif(f['mask']) > 0
    g, ratio = orient_to_mask(
        _gray(tifffile.imread(io.BytesIO(z.read(f['proc'])))), mask)
    h, w = g.shape
    mask = mask[:h, :w]
    half = int(round(FOV_UM / pxum / 2))
    half = min(half, h // 2, w // 2)
    box = 2 * half
    dens = ndi.uniform_filter(mask.astype(np.float32), size=box,
                              mode='constant')
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    cy = int(np.clip(cy, half, h - half))
    cx = int(np.clip(cx, half, w - half))
    sl = (slice(cy - half, cy + half), slice(cx - half, cx + half))
    pc = g[sl].astype(np.float32)
    lo, hi = np.percentile(pc, (1, 99.5))
    pc = np.clip((pc - lo) / (hi - lo + 1e-9), 0, 1)
    save_png(f'{OUT}/cd31_{prefix}.png', pc * 255)
    return dict(fov_um=2 * half * pxum, px_um=pxum, crop_px=2 * half,
                orient_ratio=round(float(ratio), 2))


SHORT = {'Nulliparous': 'null', 'Pregnant': 'preg'}
meta = []
samples = (df[['condition', 'mouse']].drop_duplicates()
           .sort_values(['condition', 'mouse']))
for _, srow in samples.iterrows():
    cond, mouse = srow['condition'], int(srow['mouse'])
    sub = df[(df.condition == cond) & (df.mouse == mouse)]
    med = sub['junctions_per_mm_vessel'].median()
    pick = sub.iloc[(sub['junctions_per_mm_vessel'] - med).abs().argmin()]
    prefix = f'{SHORT[cond]}{mouse}'
    m = export_crop(rois[(cond, mouse, int(pick['roi']))],
                    float(pick['px_um']), prefix)
    m.update(prefix=prefix, condition=cond, mouse=mouse, roi=int(pick['roi']))
    meta.append(m)
    print(f'montage {prefix}: ROI{pick["roi"]} '
          f'orient={m["orient_ratio"]} fov={m["fov_um"]:.0f}um')
pd.DataFrame(meta).to_csv(f'{OUT}/montage_meta.csv', index=False)
print(f'wrote {len(meta)} CD31 crops + {OUT}/montage_meta.csv')
