"""CD31 vascular morphometrics for the Figure 4 validation (MPOA).

Cohort: the 20260720 imaging set, which supersedes the May cohort (acquired
with unstable laser power). Four nulliparous animals (NULL2-5; no NULL1, whose
slides were unusable) and five pregnant animals (PREG1-5), contributing 12 and
16 ROIs. ROIs are technical replicates within an animal.

Per ROI the collaborator provided <A>_<R>_composite.png (CD31 red + DAPI grey),
<A>_<R>_CD31.png (red only) and <A>_<R>_CD31_mask.png (a manually thresholded
binary vessel mask, cropped so that ROIs from one animal do not overlap). All
three share pixel dimensions; the scale is 1.7583 px/um throughout.

Three properties of the raw data drive the design:

1. The rectangular field is not all tissue. Fields contain off-section space,
   ventricle lumen and, for stitched fields, tile gutters, and nulliparous
   fields are ~40% larger than pregnant ones. Densities are therefore computed
   inside a signal-derived tissue mask, never per field.
2. The manual mask is cropped to a sub-region of the imaged tissue. CD31+
   vessels outside that crop are real (comparable intensity and auto-detected
   vessel fraction), so scoring the manual mask over the whole tissue would
   deflate density by 13-40%, unevenly across ROIs. The analysed domain is
   therefore recovered per ROI as the region where the manual mask accounts
   for the vessels an independent segmentation finds.
3. A scale bar and its text are burned into every image and are excluded.

Two segmentations are quantified over the identical domain: the collaborator's
manual mask ("manual"), and an operator-independent Li threshold of the CD31
channel ("auto"). Agreement between them is reported per ROI, so the result
does not rest on one person's threshold choice.

Because the comparison comes out null, every image-analysis choice that could
manufacture a false negative is also varied here, in the same pass over the
pixels: the denominator the density is scored against (field, tissue, domain,
guarded core), the despeckling threshold, the split into capillary and larger
vessels, and the within-ROI heterogeneity of vessel density rather than its
mean alone. A second pass re-thresholds every image to a common vessel area
fraction, which separates a genuine caliber difference from the thinning a
stricter threshold produces on its own. The group statistics over all of these
are computed downstream.

Reads  input/vascular_imaging_20260720/<animal>/ROI<n>/*.png
Writes output/vascular_imaging_20260720/{roi_metrics.csv,
       robustness_recomputed_roi.csv, robustness_area_matched.csv,
       params.json, masks/*.npz, qc/*.png}

Statistics, robustness tests and the supplementary workbook are built by
23_vascular_stats.py; the QC supplement by 21_supp_figure_5.py.
"""

#region imports and config #####################################################

import os
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from skimage.filters import threshold_li, threshold_otsu
from skimage.morphology import skeletonize
from skimage.transform import rescale
from skan import Skeleton, summarize

Image.MAX_IMAGE_PIXELS = None

working_dir = '/home/karbabi/spatial-pregnancy'
IMG_DIR = f'{working_dir}/input/vascular_imaging_20260720'
OUT = f'{working_dir}/output/vascular_imaging_20260720'
QC = f'{OUT}/qc'
MASKS = f'{OUT}/masks'
for d in (OUT, QC, MASKS):
    os.makedirs(d, exist_ok=True)

# stated by the imaging core for every image in this set
# (Additional_Information.html); matches the 0.5687 um/px carried in the TIFF
# tags of the previous cohort, i.e. the same 40x objective.
PX_PER_UM = 1.7583
PX_UM = 1.0 / PX_PER_UM

TISSUE_SCALE = 0.25          # work at 1/4 resolution for the tissue field
TISSUE_BLUR_UM = 15.0
TISSUE_FLOOR = 2.0           # grey levels; off-section space is ~0 after the
TISSUE_REL = 0.10            # rolling-ball subtraction, tissue is >> this
TISSUE_CLOSE_UM = 20.0
TISSUE_MIN_AREA_UM2 = 5e4
BURN_SAT = 230               # burned-in scale bar / text: saturated in RGB
BURN_MIN_PX = 400
BURN_PAD_UM = 25.0

DOMAIN_CLOSE_UM = 100.0      # envelope radius recovering the manual crop
DOMAIN_CLOSE_ALT = [150.0, 200.0]   # radii reported as a sensitivity analysis
DOMAIN_FILL_UM2 = 5e4
DOMAIN_BOX_UM = 200.0        # coverage-ratio window, cross-check only
DOMAIN_RATIO = 0.5
CORE_GUARD_UM = 25.0         # counts are made inside this erosion of the domain

SPECK_MIN_AREA_UM2 = 20.0    # despeckle both segmentations identically
HOLE_MAX_AREA_UM2 = 20.0

LACUNARITY_BOX_UM = [10, 20, 40, 80]
TORT_MIN_EUCL_UM = 5.0

# DAPI nuclei: reviewer 1 asked whether endothelial cell number changes, so
# nuclei are detected and split into vessel-associated and parenchymal. These
# are 30 um max projections, in which nuclei overlap in z, so counts are a
# relative measure applied identically to both groups, not an absolute density.
NUC_BLUR_UM = 1.2
NUC_MIN_DIST_UM = 3.0
NUC_PERIVASC_UM = 3.0        # centroid within this of the vessel mask

# robustness sweeps, measured alongside the primary metrics
SPECKLE_SWEEP_UM2 = [0.0, 5.0, 20.0, 50.0, 100.0]
CAPILLARY_MAX_UM = 6.0        # diameter cut separating capillaries from larger
LOCAL_WINDOW_UM = 100.0       # window for within-ROI density heterogeneity

# imaging session per (animal, roi), from the .lif filenames in the source zip
SESSION = {
    ('NULL2', 1): '2026-07-12', ('NULL2', 2): '2026-07-12',
    ('NULL3', 1): '2026-07-08', ('NULL3', 2): '2026-07-08',
    ('NULL3', 3): '2026-07-08',
    ('NULL4', 1): '2026-06-28', ('NULL4', 2): '2026-06-28',
    ('NULL4', 3): '2026-07-12',
    ('NULL5', 1): '2026-06-24', ('NULL5', 2): '2026-06-24',
    ('NULL5', 3): '2026-06-24', ('NULL5', 4): '2026-06-24',
    ('PREG1', 1): '2026-07-15', ('PREG1', 2): '2026-07-15',
    ('PREG1', 3): '2026-07-15', ('PREG1', 4): '2026-07-15',
    ('PREG2', 1): '2026-06-24', ('PREG2', 2): '2026-06-24',
    ('PREG2', 3): '2026-06-24',
    ('PREG3', 1): '2026-07-19', ('PREG3', 2): '2026-07-19',
    ('PREG3', 3): '2026-07-19',
    ('PREG4', 1): '2026-07-17', ('PREG4', 2): '2026-07-17',
    ('PREG4', 3): '2026-07-17',
    ('PREG5', 1): '2026-07-19', ('PREG5', 2): '2026-07-19',
    ('PREG5', 3): '2026-07-19',
}

#endregion

#region morphology helpers #####################################################
# Disk structuring elements at these radii (40-450 px) are prohibitively slow;
# every dilation/erosion/closing is done with a distance transform instead.


def um_px(um):
    return max(1.0, um * PX_PER_UM)


def um2_px(um2):
    return max(1, int(round(um2 * PX_PER_UM ** 2)))


def dilate(mask, r_px):
    return ndi.distance_transform_edt(~mask) <= r_px


def erode(mask, r_px):
    return ndi.distance_transform_edt(mask) > r_px


def closing(mask, r_px):
    return erode(dilate(mask, r_px), r_px)


def largest_component(mask):
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def drop_small(mask, min_px):
    if min_px <= 1:
        return mask
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    keep = np.nonzero(sizes >= min_px)[0] + 1
    return np.isin(lab, keep)


def fill_small_holes(mask, max_px):
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    lab, n = ndi.label(holes)
    if n == 0:
        return mask
    sizes = ndi.sum(holes, lab, range(1, n + 1))
    small = np.nonzero(sizes <= max_px)[0] + 1
    return mask | np.isin(lab, small)


def skeleton_stats(mask, core):
    """Skeleton, its 3x3 neighbour count and the junction/length densities
    inside `core`. Shared by the primary metrics and the sweeps."""
    skel = skeletonize(mask)
    nb = (ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8),
                       mode='constant') - skel.astype(np.uint8))
    core_mm2 = max(core.sum() * PX_UM ** 2 / 1e6, 1e-9)
    length_mm = skel[core].sum() * PX_UM / 1000.0
    n_junc = ndi.label(skel & (nb >= 3) & core, structure=np.ones((3, 3)))[1]
    return skel, nb, length_mm / core_mm2, n_junc / core_mm2

#endregion

#region io and segmentation ####################################################


def inventory():
    rows = []
    for animal in sorted(os.listdir(IMG_DIR)):
        adir = f'{IMG_DIR}/{animal}'
        if not os.path.isdir(adir):
            continue
        for roi in sorted(os.listdir(adir)):
            rdir = f'{adir}/{roi}'
            if not os.path.isdir(rdir):
                continue
            stem = f'{rdir}/{animal}_{roi}'
            rows.append(dict(
                condition='Nulliparous' if animal.startswith('NULL')
                          else 'Pregnant',
                animal=animal, roi=int(roi.replace('ROI', '')),
                mask_path=f'{stem}_CD31_mask.png',
                cd31_path=f'{stem}_CD31.png',
                comp_path=f'{stem}_composite.png'))
    df = pd.DataFrame(rows)
    df['session'] = [SESSION.get((a, r), '')
                     for a, r in zip(df.animal, df.roi)]
    for c in ('mask_path', 'cd31_path', 'comp_path'):
        missing = [p for p in df[c] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f'{c}: {missing}')
    return df


def read_planes(row):
    """CD31 plane, DAPI plane and the manual mask, all as arrays of one shape.

    The composite renders CD31 in red and DAPI in grey, so the green channel is
    DAPI. CD31 is taken from the dedicated red-only PNG, which is the image the
    collaborator actually thresholded.
    """
    comp = np.asarray(Image.open(row.comp_path).convert('RGB'))
    cd31 = np.asarray(Image.open(row.cd31_path).convert('RGB'))[..., 0]
    raw = np.asarray(Image.open(row.mask_path))
    if raw.ndim == 3:
        raw = raw[..., :3].max(-1)
    mask = raw > (raw.max() / 2 if raw.max() > 0 else 0)
    if comp.shape[:2] != mask.shape or cd31.shape != mask.shape:
        raise ValueError(f'{row.animal}_ROI{row.roi}: shape mismatch')
    dapi = comp[..., 1]
    # cross-check that the red-only PNG agrees with red-minus-grey
    from_comp = np.clip(comp[..., 0].astype(np.int16) - dapi, 0, 255)
    corr = float(np.corrcoef(from_comp.ravel()[::37],
                             cd31.ravel()[::37].astype(np.int16))[0, 1])
    return (cd31.astype(np.float32), dapi.astype(np.float32), mask, comp,
            corr)


def load_masks(name):
    """The masks saved by the main pass, for the second (area-matched) pass."""
    z = np.load(f'{MASKS}/{name}.npz')
    sh = tuple(z['shape'])
    n = sh[0] * sh[1]
    return {k: np.unpackbits(z[k])[:n].reshape(sh).astype(bool)
            for k in z.files if k != 'shape'}


def burn_in_mask(comp):
    """Scale bar and its label, burned into every image."""
    sat = ((comp[..., 0] > BURN_SAT) & (comp[..., 1] > BURN_SAT)
           & (comp[..., 2] > BURN_SAT))
    sat = drop_small(sat, BURN_MIN_PX)
    if not sat.any():
        return sat
    return dilate(sat, um_px(BURN_PAD_UM))


def tissue_mask(cd31, dapi):
    """Imaged tissue: anywhere with DAPI or CD31 signal above the near-zero
    off-section floor. A histogram-based threshold (Otsu/triangle) is wrong
    here because many fields are entirely tissue, i.e. unimodal."""
    sig = np.maximum(dapi, cd31)
    small = rescale(sig, TISSUE_SCALE, anti_aliasing=True, preserve_range=True)
    sm = ndi.gaussian_filter(small, um_px(TISSUE_BLUR_UM) * TISSUE_SCALE)
    thr = max(TISSUE_FLOOR, TISSUE_REL * float(np.median(sm)))
    m = sm > thr
    m = closing(m, um_px(TISSUE_CLOSE_UM) * TISSUE_SCALE)
    m = drop_small(m, um2_px(TISSUE_MIN_AREA_UM2) * TISSUE_SCALE ** 2)
    m = ndi.binary_fill_holes(m)
    m = ndi.zoom(m.astype(np.uint8),
                 (cd31.shape[0] / m.shape[0], cd31.shape[1] / m.shape[1]),
                 order=0).astype(bool)
    out = np.zeros(cd31.shape, dtype=bool)
    h, w = min(m.shape[0], out.shape[0]), min(m.shape[1], out.shape[1])
    out[:h, :w] = m[:h, :w]
    return out


def condition_mask(mask):
    """Despeckle and pinhole-fill a binary vessel mask."""
    m = drop_small(mask, um2_px(SPECK_MIN_AREA_UM2))
    return fill_small_holes(m, um2_px(HOLE_MAX_AREA_UM2))


def auto_segment(cd31, domain, rule='li'):
    """Operator-independent vessel segmentation of the CD31 plane."""
    v = cd31[domain]
    v = v[::7] if v.size > 2_000_000 else v
    if v.size < 1000:
        return np.zeros_like(domain), np.nan
    thr = threshold_li(v) if rule == 'li' else threshold_otsu(v)
    return condition_mask((cd31 > thr) & domain), float(thr)


def analysis_domain(manual, tissue, radius_um=DOMAIN_CLOSE_UM):
    """Recover the sub-region the collaborator actually thresholded.

    Outside the crop the manual mask is empty, so closing it at a radius that
    spans the inter-capillary spacing reconstructs the analysed region and
    stops within one radius of the last masked vessel. Radius sensitivity is
    reported alongside the metrics.
    """
    dom = closing(manual, um_px(radius_um)) & tissue
    dom = fill_small_holes(dom, um2_px(DOMAIN_FILL_UM2))
    dom = ndi.binary_fill_holes(dom)
    return largest_component(dom)


def domain_by_coverage(manual, auto_full, tissue):
    """Independent estimate of the same region, from the local ratio of manual
    to automatic vessel coverage. Used only as a QC cross-check: it is fragile
    where the automatic threshold over-segments."""
    box = int(round(um_px(DOMAIN_BOX_UM)))
    num = ndi.uniform_filter(manual.astype(np.float32), box)
    den = ndi.uniform_filter(auto_full.astype(np.float32), box)
    ratio = np.where(den > 1e-4, num / np.maximum(den, 1e-6), 0.0)
    dom = ratio > DOMAIN_RATIO
    if not dom.any():
        return tissue.copy()
    dom = closing(dom, um_px(DOMAIN_CLOSE_UM)) & tissue
    return largest_component(fill_small_holes(dom, um2_px(DOMAIN_FILL_UM2)))

#endregion

#region morphometrics ##########################################################


def lacunarity(mask, core):
    """Gliding-box lacunarity, over boxes lying wholly inside the core."""
    m = mask.astype(np.float32)
    c = core.astype(np.float32)
    vals = {}
    for box_um in LACUNARITY_BOX_UM:
        r = int(round(um_px(box_um)))
        if r >= min(mask.shape) // 2:
            continue
        s = ndi.uniform_filter(m, size=r, mode='constant') * (r * r)
        inside = ndi.uniform_filter(c, size=r, mode='constant') >= 0.999
        v = s[inside]
        if v.size < 50 or v.mean() <= 0:
            continue
        vals[box_um] = float(1.0 + v.var() / v.mean() ** 2)
    return vals


def fractal_dimension(mask, core):
    m = mask & core
    if m.sum() < 100:
        return np.nan
    ys, xs = np.nonzero(core)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sizes, counts = [], []
    for box_um in [5, 10, 20, 40, 80, 160]:
        b = int(round(um_px(box_um)))
        h, w = m.shape[0] // b * b, m.shape[1] // b * b
        if h == 0 or w == 0:
            break
        blocks = m[:h, :w].reshape(h // b, b, w // b, b).any(axis=(1, 3))
        if blocks.sum() > 0:
            sizes.append(b * PX_UM)
            counts.append(int(blocks.sum()))
    if len(sizes) < 3:
        return np.nan
    return float(np.polyfit(np.log(1 / np.array(sizes)), np.log(counts), 1)[0])


def nuclei_metrics(dapi, vessel, core):
    """Nuclei density, split by association with the vessel mask.

    Vessel-associated nuclei per mm of vessel is the closest imaging analogue
    of endothelial (plus mural) cell number per unit vessel.
    """
    from skimage.feature import peak_local_max
    out = {}
    core_mm2 = core.sum() * PX_UM ** 2 / 1e6
    if core_mm2 <= 0:
        return out
    sm = ndi.gaussian_filter(dapi, um_px(NUC_BLUR_UM))
    v = sm[core]
    if v.size < 1000:
        return out
    thr = threshold_li(v[::7] if v.size > 2_000_000 else v)
    nuc = (sm > thr) & core
    dist = ndi.distance_transform_edt(nuc)
    peaks = peak_local_max(dist, min_distance=int(round(um_px(
        NUC_MIN_DIST_UM))), labels=nuc, exclude_border=False)
    if len(peaks) == 0:
        return out
    peri_zone = dilate(vessel, um_px(NUC_PERIVASC_UM))
    is_peri = peri_zone[peaks[:, 0], peaks[:, 1]]
    n_tot, n_peri = len(peaks), int(is_peri.sum())
    skel = skeletonize(vessel)
    length_mm = skel[core].sum() * PX_UM / 1000.0
    out['nuclei_density_per_mm2'] = n_tot / core_mm2
    out['perivascular_nuclei_density_per_mm2'] = n_peri / core_mm2
    out['parenchymal_nuclei_density_per_mm2'] = (n_tot - n_peri) / core_mm2
    out['perivascular_nuclei_fraction'] = n_peri / n_tot * 100
    out['perivascular_nuclei_per_mm_vessel'] = (
        n_peri / length_mm if length_mm else np.nan)
    out['nuclear_area_fraction'] = nuc[core].sum() / core.sum() * 100
    return out


def morphometrics(vessel, core, prefix):
    """Metrics for one segmentation. The skeleton and distance transforms are
    computed on the full mask and only counted inside `core`, so the domain
    boundary does not truncate branches or inflate endpoint counts."""
    out = {}
    core_mm2 = core.sum() * PX_UM ** 2 / 1e6
    if core_mm2 <= 0:
        return out
    out[f'{prefix}_vessel_area_fraction'] = (
        (vessel & core).sum() / core.sum() * 100)

    skel = skeletonize(vessel)
    if skel.sum() < 50:
        return out
    nb = (ndi.convolve(skel.astype(np.uint8), np.ones((3, 3), np.uint8),
                       mode='constant') - skel.astype(np.uint8))
    n_junc = ndi.label(skel & (nb >= 3) & core,
                       structure=np.ones((3, 3)))[1]
    n_end = int((skel & (nb == 1) & core).sum())
    length_mm = skel[core].sum() * PX_UM / 1000.0

    out[f'{prefix}_length_density_mm_per_mm2'] = length_mm / core_mm2
    out[f'{prefix}_junction_density_per_mm2'] = n_junc / core_mm2
    out[f'{prefix}_junctions_per_mm_vessel'] = (
        n_junc / length_mm if length_mm else np.nan)
    out[f'{prefix}_endpoint_density_per_mm2'] = n_end / core_mm2

    sk = Skeleton(skel, spacing=PX_UM)
    try:
        sdf = summarize(sk, separator='-')
    except TypeError:
        sdf = summarize(sk)
    cols = {c.replace('_', '-'): c for c in sdf.columns}
    arc = sdf[cols['branch-distance']].to_numpy()
    eu = sdf[cols['euclidean-distance']].to_numpy()
    ok = (eu > TORT_MIN_EUCL_UM) & (arc > TORT_MIN_EUCL_UM)
    out[f'{prefix}_branch_density_per_mm2'] = len(arc) / core_mm2
    out[f'{prefix}_mean_branch_length_um'] = (
        float(np.mean(arc[ok])) if ok.any() else np.nan)
    out[f'{prefix}_mean_tortuosity'] = (
        float(np.average(arc[ok] / eu[ok], weights=arc[ok]))
        if ok.any() else np.nan)

    rad = (ndi.distance_transform_edt(vessel) * PX_UM)[skel & core]
    if rad.size:
        out[f'{prefix}_mean_vessel_diameter_um'] = float(2 * rad.mean())
        out[f'{prefix}_p90_vessel_diameter_um'] = float(
            2 * np.percentile(rad, 90))
    d_out = (ndi.distance_transform_edt(~vessel) * PX_UM)[core]
    if d_out.size:
        out[f'{prefix}_mean_dist_to_vessel_um'] = float(d_out.mean())
        out[f'{prefix}_p95_dist_to_vessel_um'] = float(
            np.percentile(d_out, 95))

    lac = lacunarity(vessel & core, core)
    for k, v in lac.items():
        out[f'{prefix}_lacunarity_{k}um'] = v
    out[f'{prefix}_lacunarity_mean'] = (
        float(np.mean(list(lac.values()))) if lac else np.nan)
    out[f'{prefix}_fractal_dimension'] = fractal_dimension(vessel, core)
    return out

#endregion

#region robustness sweeps ######################################################
# Everything an alternative analysis of the same images would need: the four
# candidate denominators, the despeckling sweep, the vessel-size split and the
# within-ROI density distribution. Measured from the masks already in memory,
# so no second decode of the images is required.


def sweeps(raw, manual, core, tissue, domain):
    rec = {}
    # --- denominators --------------------------------------------------------
    # `raw` is the collaborator's mask as delivered, before despeckling and
    # before the burnt-in scale bar is cut out, so vaf_field is exactly what a
    # naive whole-field quantification would report.
    rec['vaf_field'] = float(raw.mean()) * 100
    rec['vaf_tissue'] = (raw & tissue).sum() / max(tissue.sum(), 1) * 100
    rec['vaf_domain'] = (raw & domain).sum() / max(domain.sum(), 1) * 100
    rec['vaf_core'] = (manual & core).sum() / max(core.sum(), 1) * 100

    # --- despeckling sweep ---------------------------------------------------
    for thr in SPECKLE_SWEEP_UM2:
        v = drop_small(raw, um2_px(thr))
        tag = int(thr)
        rec[f'vaf_speck{tag}'] = (v & core).sum() / max(core.sum(), 1) * 100
        _, _, len_den, junc_den = skeleton_stats(v, core)
        rec[f'len_speck{tag}'] = len_den
        rec[f'junc_speck{tag}'] = junc_den

    # --- vessel size classes and radius distribution -------------------------
    skel = skeletonize(manual)
    edt = ndi.distance_transform_edt(manual) * PX_UM
    rad = edt[skel & core]
    core_mm2 = max(core.sum() * PX_UM ** 2 / 1e6, 1e-9)
    if rad.size:
        for q in (10, 25, 50, 75, 90):
            rec[f'radius_p{q}'] = float(np.percentile(rad, q))
        cap = rad < CAPILLARY_MAX_UM / 2
        rec['capillary_length_frac'] = float(cap.mean()) * 100
        rec['capillary_length_density'] = (
            cap.sum() * PX_UM / 1000.0 / core_mm2)
        rec['large_length_density'] = (
            (~cap).sum() * PX_UM / 1000.0 / core_mm2)
    dia = 2 * edt
    rec['vaf_capillary'] = (
        (manual & core & (dia < CAPILLARY_MAX_UM)).sum()
        / max(core.sum(), 1) * 100)
    rec['vaf_large'] = (
        (manual & core & (dia >= CAPILLARY_MAX_UM)).sum()
        / max(core.sum(), 1) * 100)

    # --- within-ROI heterogeneity --------------------------------------------
    box = int(round(um_px(LOCAL_WINDOW_UM)))
    local = ndi.uniform_filter(manual.astype(np.float32), box) * 100
    inside = ndi.uniform_filter(core.astype(np.float32), box) >= 0.999
    lv = local[inside]
    if lv.size > 100:
        rec['local_vaf_sd'] = float(lv.std())
        rec['local_vaf_cv'] = float(lv.std() / max(lv.mean(), 1e-9))
        rec['local_vaf_skew'] = float(stats.skew(lv))
        rec['local_vaf_p10'] = float(np.percentile(lv, 10))
        rec['local_vaf_p90'] = float(np.percentile(lv, 90))
    return rec


def area_matched(man, target_pct):
    """Re-threshold every image to the SAME vessel area fraction, then re-measure
    the geometry.

    Vessel area -4%, length density +1% and diameter -6% is the exact signature
    a stricter threshold produces: it thins vessels without moving the skeleton.
    If pregnant vessels are genuinely thinner and more numerous, matched-area
    length density is higher in pregnancy; if the caliber difference was
    threshold-driven, everything equalises.
    """
    rows = []
    for _, r in man.iterrows():
        name = f'{r.animal}_ROI{int(r.roi)}'
        core = load_masks(name)['core']
        cd = np.asarray(Image.open(
            f'{IMG_DIR}/{r.animal}/ROI{int(r.roi)}/{name}_CD31.png'
        ).convert('RGB'))[..., 0].astype(np.float32)
        thr = float(np.quantile(cd[core], 1 - target_pct / 100.0))
        mask = drop_small((cd > thr) & core, um2_px(SPECK_MIN_AREA_UM2))
        skel, _, len_den, junc_den = skeleton_stats(mask, core)
        rad = (ndi.distance_transform_edt(mask) * PX_UM)[skel]
        rows.append(dict(
            condition=r.condition, animal=r.animal, roi=int(r.roi),
            matched_thr=thr,
            matched_vaf=mask[core].sum() / max(core.sum(), 1) * 100,
            matched_len=len_den, matched_junc=junc_den,
            matched_diam=float(2 * rad.mean()) if rad.size else np.nan))
        print(f'  {name:12s} thr={thr:6.1f} vaf={rows[-1]["matched_vaf"]:5.2f} '
              f'len={rows[-1]["matched_len"]:5.1f} '
              f'diam={rows[-1]["matched_diam"]:.2f}')
    return pd.DataFrame(rows)

#endregion

#region qc #####################################################################


def save_qc(name, cd31, manual, auto, tissue, domain, core, scale=0.18):
    """CD31 in grey, manual mask green, auto-only pixels red, tissue outline
    magenta, analysed domain yellow, guarded core cyan."""
    def ds(a, order=0):
        return rescale(a.astype(np.float32), scale, order=order,
                       anti_aliasing=(order > 0), preserve_range=True)
    g = ds(cd31, 1)
    g = g / max(g.max(), 1e-6)
    rgb = np.dstack([g, g, g]) * 0.7
    a_only = ds(auto & ~manual) > 0.5
    rgb[a_only] = [0.95, 0.25, 0.25]
    rgb[ds(manual) > 0.5] = [0.10, 0.95, 0.35]
    for m, col in ((tissue, [1.0, 0.25, 0.9]), (domain, [1.0, 0.85, 0.1]),
                   (core, [0.15, 0.7, 1.0])):
        mm = ds(m) > 0.5
        rgb[mm ^ ndi.binary_erosion(mm, np.ones((3, 3)))] = col
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
        f'{QC}/{name}.png')


def save_masks(name, shape, **masks):
    np.savez_compressed(
        f'{MASKS}/{name}.npz', shape=np.array(shape),
        **{k: np.packbits(v) for k, v in masks.items()})

#endregion

#region run ####################################################################

if __name__ == '__main__':
    man = inventory()
    print(f'{len(man)} ROIs | '
          f'{(man.condition == "Nulliparous").sum()} nulliparous / '
          f'{(man.condition == "Pregnant").sum()} pregnant | '
          f'{man.animal.nunique()} animals')

    records, sweep_records = [], []
    for _, row in man.iterrows():
        name = f'{row.animal}_ROI{row.roi}'
        cd31, dapi, raw_mask, comp, split_corr = read_planes(row)

        burn = burn_in_mask(comp)
        tissue = tissue_mask(cd31, dapi) & ~burn
        manual = condition_mask(raw_mask) & ~burn
        auto_full, thr_tissue = auto_segment(cd31, tissue, 'li')
        domain = analysis_domain(manual, tissue)
        core = erode(domain, um_px(CORE_GUARD_UM))
        if core.sum() < um2_px(1e5):        # never expected; guard anyway
            core = domain.copy()
        # the reported automatic segmentation is thresholded on the same
        # domain the manual mask covers, so the two differ only in operator
        auto, thr_li = auto_segment(cd31, domain, 'li')
        auto_otsu, thr_otsu = auto_segment(cd31, domain, 'otsu')
        dom_cov = domain_by_coverage(manual, auto_full, tissue)

        rec = dict(
            condition=row.condition, animal=row.animal, roi=row.roi,
            session=row.session,
            field_area_mm2=cd31.size * PX_UM ** 2 / 1e6,
            tissue_area_mm2=tissue.sum() * PX_UM ** 2 / 1e6,
            domain_area_mm2=domain.sum() * PX_UM ** 2 / 1e6,
            core_area_mm2=core.sum() * PX_UM ** 2 / 1e6,
            tissue_fraction_of_field=float(tissue.mean()),
            domain_fraction_of_tissue=domain.sum() / max(tissue.sum(), 1),
            burn_in_px=int(burn.sum()), split_corr=split_corr,
            thr_li=thr_li, thr_otsu=thr_otsu, thr_tissue=thr_tissue,
            despeckle_removed_frac=float(
                1 - manual.sum() / max(raw_mask.sum(), 1)),
            manual_auto_dice=float(
                2 * (manual & auto & core).sum()
                / max((manual & core).sum() + (auto & core).sum(), 1)),
            img_h=cd31.shape[0], img_w=cd31.shape[1])
        rec.update(morphometrics(manual, core, 'manual'))
        rec.update(morphometrics(auto, core, 'auto'))
        rec.update(nuclei_metrics(dapi, manual, core))
        rec['otsu_vessel_area_fraction'] = (
            (auto_otsu & core).sum() / core.sum() * 100)
        # domain-construction sensitivity: envelope radius, and the
        # independent coverage-ratio estimate of the same region
        rec['cov_domain_area_mm2'] = dom_cov.sum() * PX_UM ** 2 / 1e6
        rec['cov_domain_jaccard'] = float(
            (dom_cov & domain).sum() / max((dom_cov | domain).sum(), 1))
        for r_alt in DOMAIN_CLOSE_ALT:
            d_alt = analysis_domain(manual, tissue, r_alt)
            c_alt = erode(d_alt, um_px(CORE_GUARD_UM))
            tag = f'r{int(r_alt)}'
            rec[f'{tag}_domain_area_mm2'] = d_alt.sum() * PX_UM ** 2 / 1e6
            rec[f'{tag}_vessel_area_fraction'] = (
                (manual & c_alt).sum() / max(c_alt.sum(), 1) * 100)
            sk_alt = skeletonize(manual)
            rec[f'{tag}_length_density_mm_per_mm2'] = (
                sk_alt[c_alt].sum() * PX_UM / 1000.0
                / max(c_alt.sum() * PX_UM ** 2 / 1e6, 1e-9))
        records.append(rec)

        sw = dict(condition=row.condition, animal=row.animal, roi=row.roi)
        sw.update(sweeps(raw_mask, manual, core, tissue, domain))
        sweep_records.append(sw)

        save_qc(name, cd31, manual, auto, tissue, domain, core)
        save_masks(name, cd31.shape, manual=manual, auto=auto,
                   domain=domain, core=core, tissue=tissue)
        print(f'  {name:12s} tis={rec["tissue_fraction_of_field"]*100:4.0f}% '
              f'dom={rec["domain_area_mm2"]:.2f}mm2 '
              f'dice={rec["manual_auto_dice"]:.3f} | '
              f'VAF {rec["manual_vessel_area_fraction"]:5.2f}/'
              f'{rec["auto_vessel_area_fraction"]:5.2f}%  '
              f'len {rec["manual_length_density_mm_per_mm2"]:5.1f}  '
              f'junc {rec["manual_junction_density_per_mm2"]:5.0f}  '
              f'diam {rec["manual_mean_vessel_diameter_um"]:.2f}um  '
              f'nuc {rec.get("nuclei_density_per_mm2", float("nan")):5.0f}/mm2 '
              f'({rec.get("perivascular_nuclei_fraction", float("nan")):.0f}% '
              f'perivascular)')
        print(f'  {"":12s} VAF by denominator: field={sw["vaf_field"]:5.2f} '
              f'tissue={sw["vaf_tissue"]:5.2f} core={sw["vaf_core"]:5.2f}  |  '
              f'cap={sw["vaf_capillary"]:5.2f} large={sw["vaf_large"]:5.2f}')

    df = pd.DataFrame(records)
    lead = ['condition', 'animal', 'roi', 'session']
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(f'{OUT}/roi_metrics.csv', index=False)
    pd.DataFrame(sweep_records).to_csv(
        f'{OUT}/robustness_recomputed_roi.csv', index=False)

    # second pass: geometry at a common vessel area fraction, which needs the
    # cohort median from the first pass
    target = float(df['manual_vessel_area_fraction'].median())
    print(f'\nre-thresholding every field to {target:.2f}% vessel area')
    area_matched(man, target).to_csv(
        f'{OUT}/robustness_area_matched.csv', index=False)

    with open(f'{OUT}/params.json', 'w') as f:
        json.dump({k: v for k, v in globals().items()
                   if k.isupper() and isinstance(
                       v, (int, float, str, list))}, f, indent=2)
    print(f'\nwrote {OUT}/'
          '{roi_metrics,robustness_recomputed_roi,robustness_area_matched}'
          f'.csv ({len(df)} ROIs x {df.shape[1]} cols), '
          f'{MASKS}/*.npz, {QC}/*.png')

#endregion
