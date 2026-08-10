"""Statistics for the CD31 vascular morphometrics (Figure 4 validation).

Unit of analysis is the animal: ROIs are technical replicates and were treated
as independent in the collaborator's original t-tests, which inflates n from 9
to 28. Every metric is therefore reduced to an animal mean and tested three
ways: a Welch t-test (matching how the Fig. 3 and Fig. 5 IF panels are
reported), an exact permutation test over all 126 ways of splitting 9 animals
into groups of 4 and 5 (assumption-free at this sample size), and a
random-intercept mixed model fitted to all 28 ROIs.

No endpoint is pre-specified, so the full battery is reported with
Benjamini-Hochberg FDR and an estimate of the effective number of independent
tests. Because a null result at n = 4 vs 5 is a statement about precision, the
confidence interval of every difference and the minimum detectable effect are
reported alongside the p-values.

The comparison comes out null, so the second half of this script asks what
would have to be true for that null to be an artifact. Every analytic choice
made here or in 22_vascular_imaging.py is varied in turn, alongside the
sampling and outlier structure of the data:

  1. influence      leave-one-animal-out for every metric
  2. outliers       robust z-scores at animal and ROI level
  3. normalization  field / tissue / domain / core denominators
  4. processing     despeckle and hole-fill thresholds, including none
  5. size classes   capillary vs larger-vessel area, vessel radius distribution
  6. heterogeneity  local vessel density within ROI, not just its mean
  7. multivariate   whether the battery shifts jointly when no metric does
  8. old cohort     the May imaging, analysed the same way, for direction
  9. test choice    ROI-level vs animal-level vs mixed model
 10. power          what effect sizes this design could and could not detect
 11. matched area   geometry re-measured at a common vessel area fraction

Blocks 3-6 and 11 are scored from the per-image sweeps written by
22_vascular_imaging.py; the rest are computed here. Everything then lands in
one supplementary workbook holding what a reader needs to check the reported
result and nothing else: the analysis settings, the per-field and per-animal
measurements, the statistics, and the sensitivity analyses. Intermediate tables
kept on disk for the pipeline (per-metric outlier z-scores, the full
leave-one-out grid, multivariate components) are summarised or omitted there
rather than reproduced.

Reads  output/vascular_imaging_20260720/{roi_metrics,
       robustness_recomputed_roi,robustness_area_matched}.csv, params.json
       input/vascular_imaging.zip  (the May cohort's own quantification)
Writes output/vascular_imaging_20260720/{animal_means,stats_primary,
       stats_sensitivity,confounders,variance_components,robustness_*}.csv
       and Supplementary_Data_vascular.xlsx
"""

#region imports and config #####################################################

import io
import os
import json
import zipfile
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

working_dir = '/home/karbabi/spatial-pregnancy'
OUT = f'{working_dir}/output/vascular_imaging_20260720'
OLD_ZIP = f'{working_dir}/input/vascular_imaging.zip'
XLSX = f'{OUT}/Supplementary_Data_vascular.xlsx'

COND = ['Nulliparous', 'Pregnant']
AUTO_DICE_MIN = 0.80          # below this the automatic segmentation failed

# metric battery, grouped by what they measure. Labels are figure-ready.
# Vessel metrics exist for both segmentations (manual_/auto_ prefixes); the
# DAPI-derived cellularity metrics are computed once, against the manual mask.
BATTERY = [
    ('Density', 'vessel_area_fraction', 'Vessel area (% of tissue)'),
    ('Density', 'length_density_mm_per_mm2', 'Vessel length density (mm/mm²)'),
    ('Density', 'junction_density_per_mm2', 'Junction density (/mm²)'),
    ('Density', 'branch_density_per_mm2', 'Branch density (/mm²)'),
    ('Density', 'endpoint_density_per_mm2', 'Endpoint density (/mm²)'),
    ('Architecture', 'junctions_per_mm_vessel', 'Junctions per mm vessel'),
    ('Architecture', 'mean_branch_length_um', 'Mean branch length (µm)'),
    ('Architecture', 'mean_tortuosity', 'Mean tortuosity'),
    ('Architecture', 'fractal_dimension', 'Fractal dimension'),
    ('Architecture', 'lacunarity_mean', 'Lacunarity'),
    ('Caliber', 'mean_vessel_diameter_um', 'Mean vessel diameter (µm)'),
    ('Caliber', 'p90_vessel_diameter_um', '90th pct diameter (µm)'),
    ('Supply', 'mean_dist_to_vessel_um', 'Mean distance to vessel (µm)'),
    ('Supply', 'p95_dist_to_vessel_um', '95th pct distance to vessel (µm)'),
]
METRICS = [m for _, m, _ in BATTERY]
LABELS = {m: lab for _, m, lab in BATTERY}
FAMILY = {m: fam for fam, m, _ in BATTERY}

# DAPI cellularity, addressing whether vessel-associated cell number changes
NUCLEI = [
    ('Cellularity', 'perivascular_nuclei_per_mm_vessel',
     'Perivascular nuclei per mm vessel'),
    ('Cellularity', 'perivascular_nuclei_density_per_mm2',
     'Perivascular nuclei (/mm²)'),
    ('Cellularity', 'perivascular_nuclei_fraction',
     'Perivascular nuclei (% of all)'),
    ('Cellularity', 'nuclei_density_per_mm2', 'All nuclei (/mm²)'),
]
LABELS.update({m: lab for _, m, lab in NUCLEI})
FAMILY.update({m: fam for fam, m, _ in NUCLEI})
NUC_METRICS = [m for _, m, _ in NUCLEI]
# column name in roi_metrics.csv for each battery entry
COLUMN = {m: f'manual_{m}' for m in METRICS}
COLUMN.update({m: m for m in NUC_METRICS})
ALL_METRICS = METRICS + NUC_METRICS

#endregion

#region statistics #############################################################


def hedges_g(a, b):
    """Bias-corrected standardised difference (b - a) with a 95% interval."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1)
                  + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan
    d = (np.mean(b) - np.mean(a)) / sp
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2 - 2)))
    return g, g - 1.96 * se, g + 1.96 * se


def welch(a, b):
    """Welch t-test plus the confidence interval of the mean difference."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    n1, n2 = len(a), len(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    df = (se ** 4 / (v1 ** 2 / (n1 ** 2 * (n1 - 1))
                     + v2 ** 2 / (n2 ** 2 * (n2 - 1)))) if se > 0 else np.nan
    diff = np.mean(b) - np.mean(a)
    crit = stats.t.ppf(0.975, df) if np.isfinite(df) else np.nan
    return t, p, diff, diff - crit * se, diff + crit * se, df, se


def welch_p(a, b):
    return float(welch(a, b)[1])


def group_arrays(df, col, cond_col='condition'):
    return (df.loc[df[cond_col] == COND[0], col].dropna().to_numpy(),
            df.loc[df[cond_col] == COND[1], col].dropna().to_numpy())


def perm_p(values, in_group1):
    """Two-sided p over every assignment of units to the observed group sizes.
    With 4 and 5 animals this enumerates all 126 partitions, so the smallest
    attainable p is 1/126 = 0.0079."""
    values = np.asarray(values, float)
    in_group1 = np.asarray(in_group1, bool)
    obs = abs(welch(values[in_group1], values[~in_group1])[0])
    stat = []
    for combo in combinations(range(len(values)), int(in_group1.sum())):
        m = np.zeros(len(values), bool)
        m[list(combo)] = True
        stat.append(abs(welch(values[m], values[~m])[0]))
    stat = np.asarray(stat)
    return float(np.mean(stat >= obs - 1e-12)), int(len(stat))


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    q = p[ok]
    order = np.argsort(q)
    ranked = q[order] * len(q) / (np.arange(len(q)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty(len(q))
    res[order] = np.clip(ranked, 0, 1)
    out[ok] = res
    return out


def effective_n_tests(mat):
    """Li & Ji (2005) effective number of independent tests from the
    correlation matrix of the metrics."""
    c = np.corrcoef(np.asarray(mat, float).T)
    c = np.nan_to_num(c, nan=0.0)
    ev = np.abs(np.linalg.eigvalsh(c))
    return float(sum((e >= 1) + (e - np.floor(e)) for e in ev))


def mde(a, b, power=0.80, alpha=0.05):
    """Smallest difference in means detectable at the observed variance."""
    n1, n2 = len(a), len(b)
    sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1)
                  + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    df = n1 + n2 - 2
    nc = stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)
    return nc * sp * np.sqrt(1 / n1 + 1 / n2)


def mixed_model(roi, metric):
    """metric ~ condition with a random intercept per animal, on all ROIs."""
    import statsmodels.formula.api as smf
    d = roi[['animal', 'condition', metric]].dropna().copy()
    d = d.rename(columns={metric: 'y'})
    d['preg'] = (d.condition == 'Pregnant').astype(float)
    try:
        fit = smf.mixedlm('y ~ preg', d, groups=d['animal']).fit(reml=True)
        icc = float(fit.cov_re.iloc[0, 0]
                    / (fit.cov_re.iloc[0, 0] + fit.scale))
        return (float(fit.params['preg']), float(fit.bse['preg']),
                float(fit.pvalues['preg']), icc)
    except Exception:
        return (np.nan,) * 4


def require(path):
    if not os.path.exists(path):
        raise SystemExit(
            f'missing {path} - run 22_vascular_imaging.py first')
    return path

#endregion

#region primary statistics #####################################################

if __name__ == '__main__':
    roi = pd.read_csv(require(f'{OUT}/roi_metrics.csv'))
    roi['auto_qc_fail'] = roi['manual_auto_dice'] < AUTO_DICE_MIN
    roi['session_ord'] = pd.to_datetime(roi['session']).map(
        lambda t: t.toordinal())

    print(f'{len(roi)} ROIs, {roi.animal.nunique()} animals '
          f'({roi.groupby("condition").animal.nunique().to_dict()})')
    print(f'manual vs automatic segmentation Dice: '
          f'{roi.manual_auto_dice.mean():.3f} '
          f'(range {roi.manual_auto_dice.min():.3f}-'
          f'{roi.manual_auto_dice.max():.3f}); '
          f'{int(roi.auto_qc_fail.sum())} ROI(s) below {AUTO_DICE_MIN}')

    # ---- animal means -------------------------------------------------------
    keep_cols = [f'manual_{m}' for m in METRICS] \
        + [f'auto_{m}' for m in METRICS] + NUC_METRICS \
        + ['domain_area_mm2', 'tissue_fraction_of_field', 'thr_li',
           'manual_auto_dice', 'session_ord', 'field_area_mm2']
    keep_cols = [c for c in keep_cols if c in roi.columns]
    animal = (roi.groupby(['condition', 'animal'], as_index=False)[keep_cols]
              .mean())
    animal['n_roi'] = roi.groupby(['condition', 'animal']).size().values
    animal.to_csv(f'{OUT}/animal_means.csv', index=False)

    # ---- primary battery on the manual masks --------------------------------
    rows = []
    for metric in ALL_METRICS:
        col = COLUMN[metric]
        if col not in animal.columns:
            continue
        a = animal.loc[animal.condition == COND[0], col].dropna().to_numpy()
        b = animal.loc[animal.condition == COND[1], col].dropna().to_numpy()
        t, p, diff, lo, hi, df, se = welch(a, b)
        g, glo, ghi = hedges_g(a, b)
        pp, n_perm = perm_p(animal[col].to_numpy(),
                            animal.condition.to_numpy() == COND[0])
        u_p = stats.mannwhitneyu(a, b, alternative='two-sided').pvalue
        beta, bse, p_lmm, icc = mixed_model(roi, col)
        rows.append(dict(
            family=FAMILY[metric], metric=metric, label=LABELS[metric],
            mean_null=a.mean(), sd_null=a.std(ddof=1),
            mean_preg=b.mean(), sd_preg=b.std(ddof=1),
            diff=diff, diff_lo=lo, diff_hi=hi,
            pct_change=diff / a.mean() * 100 if a.mean() else np.nan,
            hedges_g=g, g_lo=glo, g_hi=ghi,
            t=t, df=df, p_welch=p, p_perm=pp, n_perm=n_perm,
            p_mannwhitney=u_p, lmm_beta=beta, lmm_se=bse, lmm_p=p_lmm,
            icc_animal=icc, mde=mde(a, b),
            mde_pct=mde(a, b) / a.mean() * 100 if a.mean() else np.nan,
            n_distinct_roi=int(roi[col].nunique()),
            quantized=bool(roi[col].nunique() < len(roi) / 2)))
    primary = pd.DataFrame(rows)
    for c in ('p_welch', 'p_perm', 'lmm_p'):
        primary[f'fdr_{c.replace("p_", "").replace("_p", "")}'] = \
            bh_fdr(primary[c])
    meff = effective_n_tests(
        animal[[COLUMN[m] for m in ALL_METRICS
                if COLUMN[m] in animal.columns]].dropna(axis=1, how='all'))
    primary['n_metrics'] = len(primary)
    primary['effective_n_tests'] = round(meff, 2)
    primary.to_csv(f'{OUT}/stats_primary.csv', index=False)
    DONE = primary.metric.tolist()      # metrics that had a usable column

    print(f'\n=== primary battery: animal means, {len(primary)} metrics '
          f'(effective independent tests {meff:.1f}) ===')
    qz = primary[primary.quantized]
    if len(qz):
        print('  note: ' + ', '.join(
            f'{r.label} takes only {r.n_distinct_roi} distinct values across '
            f'{len(roi)} ROIs (pixel-grid quantisation), so its p-values are '
            'tie-dominated and not interpretable'
            for _, r in qz.iterrows()))
    show = primary[['label', 'mean_null', 'mean_preg', 'pct_change',
                    'hedges_g', 'p_welch', 'p_perm', 'lmm_p', 'fdr_welch',
                    'mde_pct']].copy()
    show.columns = ['metric', 'null', 'preg', '%chg', 'g', 'p_t', 'p_perm',
                    'p_lmm', 'FDR', 'MDE%']
    print(show.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # ---- sensitivity: segmentation, domain radius, threshold rule -----------
    sens = []
    variants = [('manual', 'manual_{}', roi),
                ('auto', 'auto_{}', roi),
                ('auto_qc_pass', 'auto_{}', roi[~roi.auto_qc_fail])]
    for tag, pat, src in variants:
        am = src.groupby(['condition', 'animal'], as_index=False).mean(
            numeric_only=True)
        for metric in METRICS:
            col = pat.format(metric)
            if col not in am.columns:
                continue
            a = am.loc[am.condition == COND[0], col].dropna().to_numpy()
            b = am.loc[am.condition == COND[1], col].dropna().to_numpy()
            if len(a) < 2 or len(b) < 2:
                continue
            t, p, diff, lo, hi, _, _ = welch(a, b)
            g = hedges_g(a, b)[0]
            sens.append(dict(variant=tag, metric=metric, mean_null=a.mean(),
                             mean_preg=b.mean(), diff=diff, hedges_g=g,
                             p_welch=p))
    # envelope-radius and Otsu variants (vessel area / length only)
    for tag, cols in [('envelope_r150', ['r150_vessel_area_fraction',
                                         'r150_length_density_mm_per_mm2']),
                      ('envelope_r200', ['r200_vessel_area_fraction',
                                         'r200_length_density_mm_per_mm2']),
                      ('otsu', ['otsu_vessel_area_fraction'])]:
        am = roi.groupby(['condition', 'animal'], as_index=False).mean(
            numeric_only=True)
        for col in cols:
            if col not in am.columns:
                continue
            a = am.loc[am.condition == COND[0], col].to_numpy()
            b = am.loc[am.condition == COND[1], col].to_numpy()
            t, p, diff, lo, hi, _, _ = welch(a, b)
            sens.append(dict(variant=tag, metric=col, mean_null=a.mean(),
                             mean_preg=b.mean(), diff=diff,
                             hedges_g=hedges_g(a, b)[0], p_welch=p))
    sens = pd.DataFrame(sens)
    sens.to_csv(f'{OUT}/stats_sensitivity.csv', index=False)
    print('\n=== sensitivity: vessel area fraction across variants ===')
    sv = sens[sens.metric.str.contains('vessel_area_fraction')]
    print(sv.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # ---- confounders --------------------------------------------------------
    conf_vars = ['domain_area_mm2', 'tissue_fraction_of_field', 'thr_li',
                 'session_ord', 'field_area_mm2']
    crows = []
    for metric in ALL_METRICS:
        col = COLUMN[metric]
        if col not in roi.columns:
            continue
        for cv in conf_vars:
            rho, p = stats.spearmanr(roi[col], roi[cv], nan_policy='omit')
            crows.append(dict(metric=metric, covariate=cv, spearman_rho=rho,
                              p=p, level='roi'))
    for cv in conf_vars:
        a, b = group_arrays(animal, cv)
        crows.append(dict(metric='(group difference in covariate)',
                          covariate=cv, spearman_rho=np.nan,
                          p=welch_p(a, b), level='animal'))
    conf = pd.DataFrame(crows)
    conf.to_csv(f'{OUT}/confounders.csv', index=False)

    print('\n=== covariate balance between groups (Welch p) ===')
    for cv in conf_vars:
        a, b = group_arrays(animal, cv)
        print(f'  {cv:26s} null={a.mean():8.3f}  preg={b.mean():8.3f}  '
              f'p={welch_p(a, b):.3f}')

    # ---- variance components ------------------------------------------------
    # the mixed model was already fitted per metric for the primary battery, so
    # its ICC is reused rather than refitted
    icc_of = dict(zip(primary.metric, primary.icc_animal))
    vrows = []
    for metric in DONE:
        col = COLUMN[metric]
        vrows.append(dict(
            metric=metric, icc_animal=icc_of[metric],
            sd_within_animal=roi.groupby('animal')[col].std(ddof=1).mean(),
            sd_between_animal=roi.groupby('animal')[col].mean().std(ddof=1)))
    varcomp = pd.DataFrame(vrows)
    varcomp.to_csv(f'{OUT}/variance_components.csv', index=False)
    icc_mean = np.nanmean([v['icc_animal'] for v in vrows])
    print(f'\nbetween-animal ICC (mean over metrics): {icc_mean:.2f} '
          f'- ROIs within an animal are highly correlated, so additional ROIs '
          f'add little; power is set by animal number')

#endregion

#region robustness #############################################################

    # ---- 1. leave-one-animal-out --------------------------------------------
    print('\n=== 1. leave-one-animal-out influence ===')
    rows = []
    for m in DONE:
        col = COLUMN[m]
        a_full, b_full = group_arrays(animal, col)
        p_full, g_full = welch_p(a_full, b_full), hedges_g(a_full, b_full)[0]
        for _, dropped in animal.iterrows():
            sub = animal[animal.animal != dropped.animal]
            a, b = group_arrays(sub, col)
            if len(a) < 2 or len(b) < 2:
                continue
            rows.append(dict(metric=m, dropped=dropped.animal,
                             p_full=p_full, g_full=g_full,
                             p_loo=welch_p(a, b), g_loo=hedges_g(a, b)[0]))
    loo = pd.DataFrame(rows)
    loo.to_csv(f'{OUT}/robustness_leave_one_out.csv', index=False)
    summ = (loo.groupby('metric')
            .agg(p_full=('p_full', 'first'), g_full=('g_full', 'first'),
                 p_min=('p_loo', 'min'), p_max=('p_loo', 'max'),
                 g_min=('g_loo', 'min'), g_max=('g_loo', 'max'))
            .sort_values('p_min'))
    summ['most_influential'] = [
        loo[loo.metric == m].sort_values('p_loo').dropped.iloc[0]
        for m in summ.index]
    print(summ.to_string(float_format=lambda v: f'{v:.3f}'))
    print(f'metrics that would reach p<0.05 if one animal were removed: '
          f'{(summ.p_min < 0.05).sum()}/{len(summ)}')

    # ---- 2. outliers --------------------------------------------------------
    print('\n=== 2. outlier structure ===')
    orows = []
    for m in DONE:
        col = COLUMN[m]
        for cond in COND:
            v = animal.loc[animal.condition == cond, col]
            med = v.median()
            mad = stats.median_abs_deviation(v, scale='normal')
            for an, val in zip(animal.loc[animal.condition == cond, 'animal'],
                               v):
                z = (val - med) / mad if mad > 0 else 0.0
                orows.append(dict(level='animal', metric=m, condition=cond,
                                  unit=an, value=val, robust_z=z))
        for an, grp in roi.groupby('animal'):
            med = grp[col].median()
            mad = stats.median_abs_deviation(grp[col], scale='normal')
            for _, rr in grp.iterrows():
                z = (rr[col] - med) / mad if mad > 0 else 0.0
                orows.append(dict(level='roi', metric=m,
                                  condition=rr.condition,
                                  unit=f'{an}_ROI{int(rr.roi)}',
                                  value=rr[col], robust_z=z))
    outl = pd.DataFrame(orows)
    outl.to_csv(f'{OUT}/robustness_outliers.csv', index=False)
    ext = outl[(outl.level == 'animal') & (outl.robust_z.abs() > 3.5)]
    print(f'animal-level |robust z| > 3.5: {len(ext)} of '
          f'{(outl.level == "animal").sum()} animal-metric values')
    if len(ext):
        print(ext.sort_values('robust_z', key=abs, ascending=False)
              .head(10).to_string(index=False,
                                  float_format=lambda v: f'{v:.2f}'))
        cnt = ext.groupby(['condition', 'unit']).size()
        print('\nby animal:')
        print(cnt.to_string())

    # ---- 3-6. alternative normalizations, processing, size classes ----------
    print('\n=== 3-6. alternative normalizations, processing, size classes '
          '===')
    rc = pd.read_csv(require(f'{OUT}/robustness_recomputed_roi.csv'))
    rc_animal = rc.groupby(['condition', 'animal'], as_index=False).mean(
        numeric_only=True)
    arows = []
    for col in [c for c in rc.columns
                if c not in ('condition', 'animal', 'roi')]:
        a, b = group_arrays(rc_animal, col)
        if len(a) < 2 or len(b) < 2 or not np.isfinite(a).all():
            continue
        arows.append(dict(variant=col, mean_null=a.mean(), mean_preg=b.mean(),
                          pct_change=(b.mean() - a.mean()) / a.mean() * 100,
                          hedges_g=hedges_g(a, b)[0], p_welch=welch_p(a, b)))
    alt = pd.DataFrame(arows).sort_values('p_welch')
    alt.to_csv(f'{OUT}/robustness_alternatives.csv', index=False)
    print(alt.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # ---- 7. multivariate ----------------------------------------------------
    print('\n=== 7. multivariate shift across the whole battery ===')
    X = animal[[COLUMN[m] for m in DONE]].to_numpy(float)
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    lab = (animal.condition == COND[0]).to_numpy()
    u, s, vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    pcs = u * s
    mvrows = []
    for k in range(3):
        mvrows.append(dict(component=f'PC{k+1}',
                           var_explained=float(s[k] ** 2 / (s ** 2).sum()),
                           hedges_g=hedges_g(pcs[lab, k], pcs[~lab, k])[0],
                           p_perm=perm_p(pcs[:, k], lab)[0]))
    cent = np.linalg.norm(X[lab].mean(0) - X[~lab].mean(0))
    null_cent = []
    for combo in combinations(range(len(X)), int(lab.sum())):
        mm = np.zeros(len(X), bool)
        mm[list(combo)] = True
        null_cent.append(np.linalg.norm(X[mm].mean(0) - X[~mm].mean(0)))
    p_cent = float(np.mean(np.asarray(null_cent) >= cent - 1e-12))
    mvrows.append(dict(component='centroid distance', var_explained=np.nan,
                       hedges_g=cent, p_perm=p_cent))
    mv = pd.DataFrame(mvrows)
    mv.to_csv(f'{OUT}/robustness_multivariate.csv', index=False)
    print(mv.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    # ---- 8. the May cohort --------------------------------------------------
    print('\n=== 8. May cohort (independent acquisition), same statistics '
          '===')
    may = pd.DataFrame(columns=['metric', 'n_null', 'n_preg', 'mean_null',
                                'mean_preg', 'pct_change', 'hedges_g',
                                'p_animal', 'p_roi'])
    try:
        z = zipfile.ZipFile(OLD_ZIP)
        member = [n for n in z.namelist() if n.endswith('vascular_data.csv')][0]
        old = pd.read_csv(io.BytesIO(z.read(member)))
        old['animal'] = (old.Condition.str[:4].str.upper() + '_'
                         + old.Sample.str.extract(r'Mouse (\d+)')[0])
        old['condition'] = old.Condition
        oa = old.groupby(['condition', 'animal'], as_index=False).mean(
            numeric_only=True)
        rows = []
        for col in ['Percent_Vessel_Area', 'Junction_Density',
                    'Mean_Tortuosity', 'Average_Lacunarity']:
            a, b = group_arrays(oa, col)
            a_r, b_r = group_arrays(old, col)
            rows.append(dict(metric=col, n_null=len(a), n_preg=len(b),
                             mean_null=a.mean(), mean_preg=b.mean(),
                             pct_change=(b.mean() - a.mean()) / a.mean() * 100,
                             hedges_g=hedges_g(a, b)[0],
                             p_animal=welch_p(a, b), p_roi=welch_p(a_r, b_r)))
        may = pd.DataFrame(rows)
        may.to_csv(f'{OUT}/robustness_may_cohort.csv', index=False)
        print(may.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
        new_map = {'Percent_Vessel_Area': 'vessel_area_fraction',
                   'Junction_Density': 'junction_density_per_mm2',
                   'Mean_Tortuosity': 'mean_tortuosity',
                   'Average_Lacunarity': 'lacunarity_mean'}
        print('\ndirection agreement between cohorts:')
        for k, v in new_map.items():
            o = may.loc[may.metric == k, 'pct_change'].iloc[0]
            n = primary.loc[primary.metric == v, 'pct_change'].iloc[0]
            agree = 'same' if np.sign(o) == np.sign(n) else 'OPPOSITE'
            print(f'  {k:22s} May {o:+7.1f}%   July {n:+7.1f}%   {agree}')
    except Exception as e:
        print(f'  skipped: {e}')

    # ---- 9. unit of analysis ------------------------------------------------
    print('\n=== 9. what the unit of analysis does to the p-value ===')
    lmm_of = dict(zip(primary.metric, primary.lmm_p))
    rows = []
    for m in DONE:
        col = COLUMN[m]
        a_r, b_r = group_arrays(roi, col)
        a_a, b_a = group_arrays(animal, col)
        rows.append(dict(metric=m, p_roi_pseudorep=welch_p(a_r, b_r),
                         p_animal=welch_p(a_a, b_a), p_lmm=lmm_of[m]))
    unit = pd.DataFrame(rows).sort_values('p_roi_pseudorep')
    unit.to_csv(f'{OUT}/robustness_unit_of_analysis.csv', index=False)
    print(unit.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print(f'metrics "significant" if ROIs are treated as independent: '
          f'{(unit.p_roi_pseudorep < 0.05).sum()}/{len(unit)}; '
          f'at the animal level: {(unit.p_animal < 0.05).sum()}/{len(unit)}')

    # ---- 10. power ----------------------------------------------------------
    print('\n=== 10. detectable effect, and the n needed for smaller ones ===')
    rows = []
    for m in DONE:
        a, b = group_arrays(animal, COLUMN[m])
        sp = np.sqrt(((len(a) - 1) * np.var(a, ddof=1)
                      + (len(b) - 1) * np.var(b, ddof=1))
                     / (len(a) + len(b) - 2))
        need = {}
        for pct in (10, 20, 30):
            delta = pct / 100 * a.mean()
            n = 2
            while n < 400:
                df = 2 * n - 2
                nc = stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)
                if delta >= nc * sp * np.sqrt(2 / n):
                    break
                n += 1
            need[pct] = n
        rows.append(dict(metric=m, between_animal_cv_pct=sp / a.mean() * 100,
                         n_per_group_for_10pct=need[10],
                         n_per_group_for_20pct=need[20],
                         n_per_group_for_30pct=need[30]))
    power = pd.DataFrame(rows).sort_values('between_animal_cv_pct')
    power.to_csv(f'{OUT}/robustness_power.csv', index=False)
    print(power.to_string(index=False, float_format=lambda v: f'{v:.1f}'))

    # ---- 11. geometry at matched vessel area --------------------------------
    print('\n=== 11. geometry at matched vessel area (threshold control) ===')
    am = pd.read_csv(require(f'{OUT}/robustness_area_matched.csv'))
    am_animal = am.groupby(['condition', 'animal'], as_index=False).mean(
        numeric_only=True)
    rows = []
    for col in ['matched_thr', 'matched_vaf', 'matched_len', 'matched_junc',
                'matched_diam']:
        a, b = group_arrays(am_animal, col)
        rows.append(dict(metric=col, mean_null=a.mean(), mean_preg=b.mean(),
                         pct_change=(b.mean() - a.mean()) / a.mean() * 100,
                         hedges_g=hedges_g(a, b)[0], p_welch=welch_p(a, b)))
    matched = pd.DataFrame(rows)
    matched.to_csv(f'{OUT}/robustness_area_matched_stats.csv', index=False)
    print(matched.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('  interpretation: a real caliber shift predicts higher matched-area '
          'length density in pregnancy; a threshold artifact predicts no '
          'difference in any matched-area measure')

    print(f'\nwrote {OUT}/'
          '{animal_means,stats_primary,stats_sensitivity,confounders,'
          'variance_components,robustness_*}.csv')

#endregion

#region supplementary workbook #################################################

    params = json.load(open(require(f'{OUT}/params.json')))

    # --- per-field -----------------------------------------------------------
    fields = pd.DataFrame({
        'Condition': roi.condition, 'Animal': roi.animal, 'Field': roi.roi,
        'Imaging session': roi.session,
        'Imaged field (mm²)': roi.field_area_mm2,
        'Tissue (mm²)': roi.tissue_area_mm2,
        'Analysed domain (mm²)': roi.domain_area_mm2,
        'Guarded core (mm²)': roi.core_area_mm2,
        'Manual vs automatic Dice': roi.manual_auto_dice,
    })
    for m in DONE:
        fields[LABELS[m]] = roi[COLUMN[m]]
    fields['Condition'] = pd.Categorical(fields['Condition'], COND,
                                         ordered=True)
    fields = fields.sort_values(['Condition', 'Animal', 'Field'])

    # --- per-animal ----------------------------------------------------------
    animals = pd.DataFrame({
        'Condition': animal.condition, 'Animal': animal.animal,
        'Fields': animal.n_roi})
    for m in DONE:
        animals[LABELS[m]] = animal[COLUMN[m]]
    animals['Condition'] = pd.Categorical(animals['Condition'], COND,
                                          ordered=True)
    animals = animals.sort_values(['Condition', 'Animal'])

    # --- statistics ----------------------------------------------------------
    pw = power.set_index('metric')
    vc = varcomp.set_index('metric')
    statistics = pd.DataFrame({
        'Measure': primary.label, 'Family': primary.family,
        'Nulliparous mean': primary.mean_null,
        'Nulliparous SD': primary.sd_null,
        'Pregnant mean': primary.mean_preg, 'Pregnant SD': primary.sd_preg,
        'Difference': primary['diff'],
        'Difference 95% CI low': primary.diff_lo,
        'Difference 95% CI high': primary.diff_hi,
        'Change (%)': primary['pct_change'],
        "Hedges' g": primary.hedges_g,
        "g 95% CI low": primary.g_lo, "g 95% CI high": primary.g_hi,
        'p (Welch t-test, animal means)': primary.p_welch,
        'p (exact permutation, 126 assignments)': primary.p_perm,
        'p (mixed model, 28 fields)': primary.lmm_p,
        'FDR (Benjamini-Hochberg, 18 measures)': primary.fdr_welch,
        'Between-animal ICC': primary.icc_animal,
        'Between-animal CV (%)': [pw.loc[m, 'between_animal_cv_pct']
                                  for m in DONE],
        'Minimum detectable difference (%)': primary.mde_pct,
        'n per group for 10% change': [pw.loc[m, 'n_per_group_for_10pct']
                                       for m in DONE],
        'n per group for 20% change': [pw.loc[m, 'n_per_group_for_20pct']
                                       for m in DONE],
        'SD within animal': [vc.loc[m, 'sd_within_animal'] for m in DONE],
        'SD between animals': [vc.loc[m, 'sd_between_animal'] for m in DONE],
        'Note': ['Pixel-grid quantised (5 distinct values across 28 fields); '
                 'p-values tie-dominated and not interpretable' if q else ''
                 for q in primary.quantized],
    })

    # --- robustness ----------------------------------------------------------
    rb = []

    def add(block, variant, mean_null, mean_preg, g, p, note=''):
        rb.append(dict(Block=block, Variant=variant,
                       **{'Nulliparous mean': mean_null,
                          'Pregnant mean': mean_preg,
                          "Hedges' g": g, 'p (Welch)': p, 'Note': note}))

    DENOM = {'vaf_field': 'Whole imaged field (no tissue mask)',
             'vaf_tissue': 'Tissue mask',
             'vaf_domain': 'Analysed domain',
             'vaf_core': 'Guarded core (used throughout)'}
    for k, lab_ in DENOM.items():
        r = alt[alt.variant == k].iloc[0]
        add('Vessel area: denominator', lab_, r.mean_null, r.mean_preg,
            r.hedges_g, r.p_welch)

    for thr in (0, 5, 20, 50, 100):
        r = alt[alt.variant == f'vaf_speck{thr}'].iloc[0]
        add('Vessel area: despeckle threshold', f'Objects < {thr} µm² removed',
            r.mean_null, r.mean_preg, r.hedges_g, r.p_welch,
            'Used throughout' if thr == 20 else '')
    for thr in (0, 5, 20, 50, 100):
        r = alt[alt.variant == f'len_speck{thr}'].iloc[0]
        add('Vessel length density: despeckle threshold',
            f'Objects < {thr} µm² removed', r.mean_null, r.mean_preg,
            r.hedges_g, r.p_welch, 'Used throughout' if thr == 20 else '')

    SEG = {'manual': 'Manual mask (used throughout)',
           'auto': 'Automatic Li threshold',
           'auto_qc_pass': 'Automatic Li threshold, 1 failed field excluded'}
    for var, lab_ in SEG.items():
        for m in ('vessel_area_fraction', 'junction_density_per_mm2',
                  'mean_vessel_diameter_um'):
            r = sens[(sens.variant == var) & (sens.metric == m)]
            if len(r):
                r = r.iloc[0]
                add(f'Segmentation: {LABELS[m]}', lab_, r.mean_null,
                    r.mean_preg, r.hedges_g, r.p_welch)

    for var, lab_ in [('envelope_r150', 'Domain radius 150 µm'),
                      ('envelope_r200', 'Domain radius 200 µm'),
                      ('otsu', 'Otsu threshold')]:
        for _, r in sens[sens.variant == var].iterrows():
            nm = ('Vessel area' if 'vessel_area' in r.metric
                  else 'Vessel length density')
            add(f'Domain and threshold: {nm}', lab_, r.mean_null, r.mean_preg,
                r.hedges_g, r.p_welch)

    MATCH = {'matched_len': 'Vessel length density',
             'matched_junc': 'Junction density',
             'matched_diam': 'Mean vessel diameter'}
    for k, lab_ in MATCH.items():
        r = matched[matched.metric == k].iloc[0]
        add('Geometry at matched vessel area', lab_, r.mean_null, r.mean_preg,
            r.hedges_g, r.p_welch,
            'Every field re-thresholded to the same vessel area, isolating '
            'morphology from threshold choice')

    for _, r in may.iterrows():
        add('May cohort (independent acquisition, 3 vs 2 animals)',
            r.metric.replace('_', ' '), r.mean_null, r.mean_preg, r.hedges_g,
            r.p_animal, 'Superseded cohort; direction check only')

    for m in DONE:
        s = loo[loo.metric == m]
        worst = s.loc[s.p_loo.idxmin()]
        add('Leave one animal out', LABELS[m], np.nan, np.nan,
            worst.g_loo, worst.p_loo,
            f'Most influential animal: {worst.dropped} '
            f'(p {worst.p_full:.3f} with all 9 animals)')

    for _, r in unit.iterrows():
        add('Unit of analysis', LABELS[r.metric], np.nan, np.nan, np.nan,
            r.p_animal,
            f'p = {r.p_roi_pseudorep:.4f} if the 28 fields are treated as '
            f'independent; p = {r.p_lmm:.4f} by mixed model')

    cb = conf[conf.level == 'animal']
    CBAL = {'domain_area_mm2': 'Analysed domain (mm²)',
            'tissue_fraction_of_field': 'Tissue fraction of field',
            'thr_li': 'Automatic threshold (grey level)',
            'session_ord': 'Imaging session (date)',
            'field_area_mm2': 'Imaged field (mm²)'}
    for k, lab_ in CBAL.items():
        r = cb[cb.covariate == k].iloc[0]
        a = animal.loc[animal.condition == COND[0], k].mean()
        b = animal.loc[animal.condition == COND[1], k].mean()
        add('Covariate balance between groups', lab_, a, b, np.nan, r.p,
            'Technical covariate, not a measurement')

    robustness = pd.DataFrame(rb)

    # --- parameters ----------------------------------------------------------
    PARAM_LABELS = [
        ('PX_PER_UM', 'Image scale (pixels per µm)'),
        ('TISSUE_BLUR_UM', 'Tissue mask: blur (µm)'),
        ('TISSUE_CLOSE_UM', 'Tissue mask: morphological closing (µm)'),
        ('TISSUE_MIN_AREA_UM2', 'Tissue mask: minimum component (µm²)'),
        ('DOMAIN_CLOSE_UM', 'Analysed domain: closing radius (µm)'),
        ('DOMAIN_FILL_UM2', 'Analysed domain: hole fill (µm²)'),
        ('CORE_GUARD_UM', 'Guarded core: border erosion (µm)'),
        ('SPECK_MIN_AREA_UM2', 'Vessel mask: despeckle (µm²)'),
        ('HOLE_MAX_AREA_UM2', 'Vessel mask: pinhole fill (µm²)'),
        ('CAPILLARY_MAX_UM', 'Capillary diameter cut (µm)'),
        ('NUC_PERIVASC_UM', 'Perivascular nucleus distance (µm)'),
        ('TORT_MIN_EUCL_UM', 'Tortuosity: minimum branch chord (µm)'),
        ('BURN_PAD_UM', 'Burnt-in scale bar exclusion pad (µm)'),
    ]
    parameters = pd.DataFrame(
        [dict(Parameter=lab_, Value=params[k]) for k, lab_ in PARAM_LABELS
         if k in params])

    # --- readme --------------------------------------------------------------
    n_null = (animal.condition == COND[0]).sum()
    n_preg = (animal.condition == COND[1]).sum()
    va = primary[primary.metric == 'vessel_area_fraction'].iloc[0]
    readme = pd.DataFrame([
        ('Supplementary Data: CD31 vascular quantification (Fig. 4F)', ''),
        ('', ''),
        ('Cohort',
         f'{n_null} nulliparous and {n_preg} pregnant mice, '
         f'{(roi.condition == COND[0]).sum()} and '
         f'{(roi.condition == COND[1]).sum()} imaged fields, preoptic area. '
         'Independent of the animals used for spatial transcriptomics.'),
        ('Unit of analysis',
         'The animal. Fields within an animal are technical replicates '
         f'(vessel-area ICC {va.icc_animal:.2f}), so they are averaged before '
         'testing.'),
        ('Headline result',
         f'Vessels occupied {va.mean_null:.2f}% of the tissue in nulliparous '
         f'and {va.mean_preg:.2f}% in pregnant mice (difference '
         f'{va["diff"]:+.2f} percentage points, 95% CI {va.diff_lo:+.2f} to '
         f'{va.diff_hi:+.2f}, p = {va.p_welch:.2f}). No measure differed after '
         f'correction (minimum FDR {primary.fdr_welch.min():.2f}).'),
        ('Segmentation robustness',
         f'The manual masks and a fully automatic Li threshold agree at a mean '
         f'Dice of {roi.manual_auto_dice.mean():.2f} '
         f'(range {roi.manual_auto_dice.min():.2f}-'
         f'{roi.manual_auto_dice.max():.2f}) across '
         f'{len(roi)} fields.'),
        ('Multiple testing',
         f'{len(primary)} measures reported without a pre-specified endpoint; '
         'the effective number of independent tests is '
         f'{primary.effective_n_tests.iloc[0]:.1f} (Li and Ji 2005).'),
        ('', ''),
        ('Sheet', 'Contents'),
        ('Parameters', 'Image scale and every image-analysis threshold.'),
        ('Fields',
         'Per-field measurements and the areas they were computed in.'),
        ('Animals', 'Per-animal means; the unit of analysis.'),
        ('Statistics',
         'Group means, differences with confidence intervals, effect sizes, '
         'three tests per measure, FDR, variance components and detectable '
         'effects.'),
        ('Robustness',
         'Every analytic choice varied in turn: denominator, despeckling, '
         'segmentation, domain radius, threshold rule, matched-area geometry, '
         'the superseded May cohort, leave-one-animal-out, unit of analysis, '
         'and covariate balance.'),
    ], columns=['Item', 'Detail'])

    # --- write ---------------------------------------------------------------
    SHEETS = [('README', readme), ('Parameters', parameters),
              ('Fields', fields), ('Animals', animals),
              ('Statistics', statistics), ('Robustness', robustness)]

    with pd.ExcelWriter(XLSX, engine='xlsxwriter') as xw:
        book = xw.book
        hdr = book.add_format({'bold': True, 'valign': 'top',
                               'text_wrap': True, 'bottom': 1,
                               'font_size': 10})
        wrap = book.add_format({'valign': 'top', 'text_wrap': True,
                                'font_size': 10})
        plain = book.add_format({'valign': 'top', 'font_size': 10})
        num3 = book.add_format({'num_format': '0.000', 'font_size': 10})
        for name, df_ in SHEETS:
            df_.to_excel(xw, sheet_name=name, index=False, startrow=1,
                         header=False)
            ws = xw.sheets[name]
            for j, c in enumerate(df_.columns):
                ws.write(0, j, c, hdr)
            if name == 'README':
                ws.set_column(0, 0, 26, plain)
                ws.set_column(1, 1, 110, wrap)
                continue
            if name == 'Parameters':
                ws.set_column(0, 0, 42, plain)
                ws.set_column(1, 1, 14, plain)
                continue
            for j, c in enumerate(df_.columns):
                width = max(len(str(c)) * 0.95,
                            df_[c].astype(str).str.len().max() * 0.95 + 1)
                width = min(max(width, 9), 46)
                is_num = pd.api.types.is_numeric_dtype(df_[c])
                ws.set_column(j, j, width,
                              num3 if is_num and c != 'Field' else
                              (wrap if c == 'Note' else plain))
            ws.freeze_panes(1, 2 if name in ('Fields', 'Animals') else 1)
            ws.autofilter(0, 0, len(df_), len(df_.columns) - 1)

    print(f'\nwrote {XLSX}')
    for name, df_ in SHEETS:
        print(f'  {name:12s} {df_.shape[0]:3d} rows x {df_.shape[1]:2d} cols')

#endregion
