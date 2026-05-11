#region imports & CLI #########################################################

import os
import sys
import argparse
import subprocess
import pickle as pkl
import warnings
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.distance import pdist
from scipy.spatial import KDTree
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r, to_py
sys.path.insert(0, os.path.expanduser('~'))
from single_cell import SingleCell

warnings.filterwarnings('ignore')

_p = argparse.ArgumentParser()
_p.add_argument('--cellchat-job', action='store_true',
                help='worker mode: compute one (platform, condition[, sample])')
_p.add_argument('--stage', choices=['pre', 'sample'], default=None,
                help='worker stage: pre (build _pre.rds, MERINGUE on pooled '
                     'condition) or sample (load _pre.rds, computeCommunProb '
                     'on one sample)')
_p.add_argument('--platform', default=None)
_p.add_argument('--condition', default=None)
_p.add_argument('--sample', default=None,
                help='sample id (required for --stage=sample)')
_p.add_argument('--scale-distance', type=float, default=None,
                help='scale.distance value calibrated by the driver')
_a, _ = _p.parse_known_args()

IS_WORKER = _a.cellchat_job
WORKER_STAGE = _a.stage
WORKER_PLATFORM = _a.platform
WORKER_CONDITION = _a.condition
WORKER_SAMPLE = _a.sample
WORKER_SCALE_DISTANCE = _a.scale_distance
if IS_WORKER and (WORKER_PLATFORM is None or WORKER_CONDITION is None
                  or WORKER_SCALE_DISTANCE is None or WORKER_STAGE is None):
    raise SystemExit(
        '--cellchat-job requires --stage, --platform, --condition, '
        '--scale-distance')
if IS_WORKER and WORKER_STAGE == 'sample' and WORKER_SAMPLE is None:
    raise SystemExit('--stage=sample requires --sample')

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
cellphonedb_path = f'{working_dir}/input/cellphonedb'
ortholog_cache = f'{cellphonedb_path}/gprofiler_orthologs.pkl'

# computed per dataset as conversion_factor / interaction_range
SCALE_DISTANCE = None
CONTACT_RANGE = 10
SPOT_SIZE = 15

datasets = {
    'slidetags': {
        'path': f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
        'interaction_range': 500,
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
        'interaction_range': 300,
    },
}

# Worker mode: restrict datasets to its single platform.
if IS_WORKER:
    if WORKER_PLATFORM not in datasets:
        raise SystemExit(f'unknown platform {WORKER_PLATFORM}')
    for _ds in list(datasets):
        if _ds != WORKER_PLATFORM:
            del datasets[_ds]
    print(f'[worker] stage={WORKER_STAGE} platform={WORKER_PLATFORM} '
          f'condition={WORKER_CONDITION} '
          f'sample={WORKER_SAMPLE or "-"}', flush=True)

#endregion

#region SLURM helpers #########################################################

PLATFORM_ABBR = {'slidetags': 'sl', 'xenium': 'xn'}
CONDITION_ABBR = {'CTRL': 'ctl', 'PREG': 'prg', 'POSTPART': 'ppt'}
CC_OUT_DIR = f'{working_dir}/output/cellchat'
CC_LOG_DIR = f'{CC_OUT_DIR}/logs'
os.makedirs(CC_LOG_DIR, exist_ok=True)
for _name in datasets:
    os.makedirs(f'{CC_OUT_DIR}/{_name}', exist_ok=True)

def _cc_tag_pre(name, cond):
    return (f'ccp_{PLATFORM_ABBR.get(name, name[:2])}_'
            f'{CONDITION_ABBR.get(cond, cond[:3].lower())}')

def _cc_tag_sample(name, cond, sample):
    # samples like "CTRL_1" -> "1", or use full name if no underscore
    suffix = sample.split('_')[-1] if '_' in sample else sample
    return (f'ccs_{PLATFORM_ABBR.get(name, name[:2])}_'
            f'{CONDITION_ABBR.get(cond, cond[:3].lower())}_{suffix}')

def _active_slurm_jobs():
    try:
        out = subprocess.check_output(
            'squeue -h -u "$USER" -o "%j"', shell=True, text=True)
    except subprocess.CalledProcessError:
        return set()
    return {ln.strip() for ln in out.splitlines() if ln.strip()}

LOGIN_HOST = 'tri-login01'

def submit_slurm(cmd, *, job_name, log_file, hours=24):
    script = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p compute',
        '#SBATCH --account=rrg-shreejoy',
        '#SBATCH -N 1',
        '#SBATCH -n 1',
        f'#SBATCH -t {hours}:00:00',
        f'#SBATCH -J {job_name}',
        f'#SBATCH -o {log_file}',
        f"bash -i -c 'export POLARS_MAX_THREADS=192; "
        f"set -euo pipefail; {cmd}'",
    ]) + '\n'
    scratch = os.environ.get('SCRATCH', '.')
    with NamedTemporaryFile(
        'w', dir=scratch, suffix='.sh', delete=False) as fh:
        fh.write(script)
        sp = fh.name
    try:
        subprocess.check_call(
            f'ssh -o BatchMode=yes {LOGIN_HOST} '
            f'CLUSTER=trillium /opt/slurm/bin/sbatch '
            f'--export=NONE --get-user-env=L {sp}',
            shell=True, executable='/bin/bash')
    finally:
        os.unlink(sp)

#endregion

#region load data ##############################################################

adatas = {}
for name, cfg in datasets.items():
    adata = sc.read_h5ad(cfg['path'])
    if 'gene_symbol' in adata.var.columns:
        adata.var.index = adata.var['gene_symbol']
        adata.var_names_make_unique()
        adata.var.drop(columns='gene_symbol', inplace=True)
    adata.var.index.name = None
    drop = cfg.get('drop_samples', [])
    if drop:
        adata = adata[~adata.obs['sample'].isin(drop)].copy()
        print(f'[{name}] dropped samples: {drop}')
    adatas[name] = adata
    print(f'[{name}] {adata.shape[0]:,} cells, '
          f'{adata.obs[cell_type_col].nunique()} subclasses, '
          f'{adata.var.shape[0]:,} genes')

#endregion

#region build extended cellchat database ######################################

r('''
options(future.globals.maxSize = Inf)
suppressPackageStartupMessages({
    library(SpatialCellChat)
    library(NeuronChat)
    library(dplyr)
    library(purrr)
    library(Seurat)
    library(tidyverse)
    library(future)
})

# multisession (separate R processes) over multicore (fork): forking
# crashes Matrix.so on large sparse matrices (future#390). parallelly's
# localhost-worker safety check reads /proc/self/status which reports 1
# CPU on Trillium tasks; we own the full node, so disable the check.
options(parallelly.maxWorkers.localhost = c(Inf, Inf))
n_workers <- 16L
plan("multisession", workers = n_workers)
cat(sprintf("[future] strategy=multisession, workers=%d\\n", n_workers))
''')

with open(ortholog_cache, 'rb') as f:
    ortho = pkl.load(f)
to_r(ortho['interaction_input'], 'cpdb_interaction_input')
to_r(ortho['complex_input'], 'cpdb_complex_input')
to_r(ortho['cpdb_gene_info_mouse'].reset_index(drop=True),
     'cpdb_gene_info_mouse')

r('''
base_interaction <- CellChatDB.mouse$interaction
base_complex <- CellChatDB.mouse$complex
base_geneInfo <- CellChatDB.mouse$geneInfo
geneInfo_cpdb <- cpdb_gene_info_mouse

cpdb_filtered <- cpdb_interaction_input %>%
    filter(partner_a %in% geneInfo_cpdb$uniprot &
           partner_b %in% geneInfo_cpdb$uniprot)
cpdb_filtered$ligand <- geneInfo_cpdb$Symbol[
    match(cpdb_filtered$partner_a, geneInfo_cpdb$uniprot)]
cpdb_filtered$receptor <- geneInfo_cpdb$Symbol[
    match(cpdb_filtered$partner_b, geneInfo_cpdb$uniprot)]

cpdb_formatted <- cpdb_filtered %>%
    rename(interaction_name = interactors,
           pathway_name = classification) %>%
    mutate(annotation = case_when(
               directionality == 'secreted' ~ 'Secreted Signaling',
               TRUE ~ 'Cell-Cell Contact'),
           interaction_name_2 = interaction_name)

data(list = 'interactionDB_mouse')
neuron_chat_db_list <- eval(parse(text = 'interactionDB_mouse'))
neuron_chat_formatted <- purrr::map_dfr(
    neuron_chat_db_list,
    ~ expand.grid(interaction_name = .x$interaction_name,
                  ligand = .x$lig_contributor,
                  receptor = .x$receptor_subunit,
                  stringsAsFactors = FALSE)
) %>% mutate(pathway_name = ligand, annotation = 'Secreted Signaling')

required_cols <- colnames(base_interaction)
for (col in required_cols) {
    cls <- class(base_interaction[[col]])
    if (!col %in% names(cpdb_formatted))
        cpdb_formatted[[col]] <- as(NA, cls)
    else
        cpdb_formatted[[col]] <- as(cpdb_formatted[[col]], cls)
    if (!col %in% names(neuron_chat_formatted))
        neuron_chat_formatted[[col]] <- as(NA, cls)
    else
        neuron_chat_formatted[[col]] <- as(
            neuron_chat_formatted[[col]], cls)
}

final_interactions <- bind_rows(
    base_interaction,
    cpdb_formatted[, required_cols],
    neuron_chat_formatted[, required_cols]) %>%
    mutate(ligand_upper = toupper(ligand),
           receptor_upper = toupper(receptor)) %>%
    distinct(ligand_upper, receptor_upper, .keep_all = TRUE) %>%
    select(-ligand_upper, -receptor_upper)

final_geneInfo <- bind_rows(base_geneInfo, geneInfo_cpdb) %>%
    distinct(Symbol, .keep_all = TRUE)

CellChatDB_ext <- list(
    interaction = final_interactions,
    complex = base_complex,
    cofactor = CellChatDB.mouse$cofactor,
    geneInfo = final_geneInfo)
''')

n_interactions = to_py('nrow(CellChatDB_ext$interaction)')
print(f'Extended CellChat DB: {n_interactions} interactions')

#endregion

#region compute / dispatch ####################################################

def hemisphere_split(adata_cond):
    keep_mask = pd.Series(False, index=adata_cond.obs.index)
    for samp, samp_obs in adata_cond.obs.groupby('sample', observed=True):
        med_x = samp_obs['x_affine'].median()
        keep_mask.loc[
            samp_obs.index[samp_obs['x_affine'] < med_x]] = True
    return adata_cond[keep_mask.values]

def deduplicate_cells(adata, min_dist_um, conversion_factor):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    coords = adata.obs[['x_affine', 'y_affine']].values
    min_dist_aff = min_dist_um / conversion_factor
    tree = KDTree(coords)
    pairs = tree.query_pairs(min_dist_aff, output_type='ndarray')
    if len(pairs) == 0:
        return adata, 0
    n = len(coords)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=bool)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    seen = set()
    keep = np.zeros(n, dtype=bool)
    for i in range(n):
        if labels[i] not in seen:
            seen.add(labels[i])
            keep[i] = True
    return adata[keep], int(n - keep.sum())

DEDUP_MIN_DIST_UM = 1.0
MAX_CELLS_COMPUTE = 25000
MAX_CELLS_COMPUTE_SAMPLE = 25000

def subsample_uniform(adata, max_cells, seed=0):
    # Uniform random subsample. Unlike per-cell-type random sampling,
    # this preserves spatial structure (every cell has equal keep
    # probability), preserves cell-type proportions, and just thins the
    # point process — communication probabilities scale globally without
    # distorting relative patterns.
    if adata.n_obs <= max_cells:
        return adata, 0
    rng = np.random.default_rng(seed)
    keep_idx = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    return adata[keep_idx], int(adata.n_obs - max_cells)

def calibrate_scale_distance(adata, contrasts, conversion_factor,
                             interaction_range, name):
    # Calibrate from the SAME data slice the per-sample workers will see
    # (full sample, dedup, MAX_CELLS_COMPUTE_SAMPLE cap; no hemisphere
    # split). Take the smallest min(NN) across all (cond, sample) — the
    # densest sample sets the scale, guaranteeing every per-sample
    # computeCommunProb call lands min(d × scale.distance) >= TARGET.
    TARGET = 1.5
    conditions_all = sorted({c for pair in contrasts for c in pair})
    d_min_um_per_sample = {}
    for cond in conditions_all:
        for samp in sorted(
                adata.obs[adata.obs['condition'] == cond]['sample']
                .astype(str).unique()):
            a = adata[(adata.obs['condition'] == cond) &
                      (adata.obs['sample'] == samp)]
            a, n_dd = deduplicate_cells(
                a, DEDUP_MIN_DIST_UM, conversion_factor)
            a, n_sub = subsample_uniform(a, MAX_CELLS_COMPUTE_SAMPLE)
            coords = a.obs[['x_affine', 'y_affine']].values
            tree = KDTree(coords)
            nn_dist_aff = tree.query(coords, k=2)[0][:, 1]
            d_min_um = float(np.min(nn_dist_aff) * conversion_factor)
            d_min_um_per_sample[(cond, samp)] = d_min_um
            print(f'[{name}] {cond}/{samp}: dedup -{n_dd}, '
                  f'subsample -{n_sub} → {a.n_obs:,} cells, '
                  f'min(NN) = {d_min_um:.4f} μm')
    d_min_smallest = min(d_min_um_per_sample.values())
    d_min_largest = max(d_min_um_per_sample.values())
    density_ratio = d_min_largest / d_min_smallest
    scale_distance = TARGET / d_min_smallest
    print(f'[{name}] density ratio (sparsest/densest, by min d_μm): '
          f'{density_ratio:.2f}x')
    print(f'[{name}] calibrated scale.distance: {scale_distance:.4f}')
    for (cond, samp), d in d_min_um_per_sample.items():
        d_min_post_pred = d * scale_distance
        flag = '' if 1.0 <= d_min_post_pred <= 2.0 else ' (OUT OF [1,2])'
        print(f'  [{cond}/{samp}] predicted d.min_post: '
              f'{d_min_post_pred:.3f}{flag}')
    if density_ratio > 2.0 / TARGET:
        print(f'[{name}] WARNING: density ratio > '
              f'{2.0/TARGET:.2f}x — sparsest sample will exceed '
              f'd.min_post=2 and may error.')
    return scale_distance

def prep_adata_cond_pooled(adata, cond, conversion_factor):
    # Build the pooled (all-samples-in-condition) hemisphere-split,
    # deduped, subsampled adata used for the pre-step (MERINGUE on a
    # spatially coherent representation of the condition).
    a = adata[adata.obs['condition'] == cond].copy()
    a.obs = a.obs[
        ['sample', 'condition', cell_type_col, 'x_affine', 'y_affine']]
    a.uns, a.obsm, a.obsp = {}, {}, {}
    n0 = a.n_obs
    a = hemisphere_split(a).copy()
    n1 = a.n_obs
    a, n_dd = deduplicate_cells(a, DEDUP_MIN_DIST_UM, conversion_factor)
    n2 = a.n_obs
    a, n_sub = subsample_uniform(a, MAX_CELLS_COMPUTE)
    a = a.copy()
    print(f'  pooled: {n0:,} → hemi {n1:,} → dedup {n2:,} (-{n_dd}) '
          f'→ subsample {a.n_obs:,} (-{n_sub}) '
          f'across {a.obs["sample"].nunique()} samples')
    return a

# Path bookkeeping (built in driver and reused in worker).
all_pre_paths = {}      # (name, cond) -> _pre.rds
all_sample_paths = {}   # (name, cond, sample) -> _sample.rds
all_cond_paths = {}     # (name, cond) -> _cond.rds (averaged)
samples_per = {}        # (name, cond) -> [sample, ...]
scale_distance_per_dataset = {}

for name, cfg in datasets.items():
    adata = adatas[name]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(
        adata.n_obs, size=min(1000, adata.n_obs), replace=False)
    raw = adata.obs.iloc[sample_idx][['x_raw', 'y_raw']].values
    aff = adata.obs.iloc[sample_idx][['x_affine', 'y_affine']].values
    conversion_factor = float(np.median(pdist(raw)) / np.median(pdist(aff)))
    print(f'[{name}] conversion factor: {conversion_factor:.2f}')

    if IS_WORKER:
        scale_distance = WORKER_SCALE_DISTANCE
        print(f'[{name}] worker scale.distance (from --scale-distance): '
              f'{scale_distance:.4f}')
    else:
        scale_distance = calibrate_scale_distance(
            adata, cfg['contrasts'], conversion_factor,
            cfg['interaction_range'], name)
    scale_distance_per_dataset[name] = scale_distance

    conditions_all = sorted({
        c for pair in cfg['contrasts'] for c in pair})
    for cond in conditions_all:
        all_pre_paths[(name, cond)] = (
            f'{CC_OUT_DIR}/{name}/'
            f'cellchat_pre_{cond}_{cell_type_col}.rds')
        all_cond_paths[(name, cond)] = (
            f'{CC_OUT_DIR}/{name}/'
            f'cellchat_cond_{cond}_{cell_type_col}.rds')
        samples_per[(name, cond)] = sorted(
            adata.obs[adata.obs['condition'] == cond]['sample']
            .astype(str).unique().tolist())
        for samp in samples_per[(name, cond)]:
            all_sample_paths[(name, cond, samp)] = (
                f'{CC_OUT_DIR}/{name}/'
                f'cellchat_sample_{cond}_{samp}_{cell_type_col}.rds')

    if IS_WORKER and WORKER_CONDITION not in conditions_all:
        raise SystemExit(
            f'condition {WORKER_CONDITION} not valid for {name}')

# === Worker: pre-step (build _pre.rds for one condition) ====================
if IS_WORKER and WORKER_STAGE == 'pre':
    name = WORKER_PLATFORM
    cfg = datasets[name]
    cond = WORKER_CONDITION
    pre_path = all_pre_paths[(name, cond)]
    if os.path.exists(pre_path):
        print(f'[worker pre] {pre_path} cached, exiting')
        sys.exit(0)

    adata = adatas[name]
    # recompute conversion_factor (same RNG seed → same value as driver)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(
        adata.n_obs, size=min(1000, adata.n_obs), replace=False)
    raw = adata.obs.iloc[sample_idx][['x_raw', 'y_raw']].values
    aff = adata.obs.iloc[sample_idx][['x_affine', 'y_affine']].values
    conversion_factor = float(np.median(pdist(raw)) / np.median(pdist(aff)))

    adata_cond = prep_adata_cond_pooled(adata, cond, conversion_factor)
    SingleCell(adata_cond).to_seurat('sobj', v3=True)
    # shim to avoid layer issues with seurat
    r('''
    .gad_orig <- getFromNamespace("GetAssayData.Seurat", "SeuratObject")
    assignInNamespace("GetAssayData.Seurat", function(object,
            assay = NULL, slot = NULL, layer = NULL, ...) {
        if (!is.null(slot) && is.null(layer)) {
            layer <- slot; slot <- NULL
        }
        .gad_orig(object = object, assay = assay, layer = layer, ...)
    }, ns = "SeuratObject")
    ''')
    spatial_locs = adata_cond.obs[['x_affine', 'y_affine']]
    to_r(spatial_locs, 'spatial_locs', format='data.frame')
    to_r(cell_type_col, 'cell_type_col')
    to_r(conversion_factor, 'conversion_factor')
    to_r(SPOT_SIZE, 'spot_size')
    to_r(pre_path, 'pre_path')
    to_r(cond, 'cond_label')

    r('''
    sobj$samples <- sobj$sample
    sobj <- NormalizeData(sobj)
    sobj[[cell_type_col]] <- droplevels(sobj[[cell_type_col]])

    sf <- data.frame(ratio = conversion_factor, tol = spot_size / 2)
    cobj <- createSpatialCellChat(
        object = sobj, group.by = cell_type_col,
        assay = "RNA", datatype = "spatial",
        coordinates = spatial_locs[colnames(sobj), ],
        spatial.factors = sf)
    cobj@DB <- CellChatDB_ext
    cobj <- subsetData(cobj)
    cobj <- preProcessing(cobj)

    # MERINGUE on the pooled hemisphere data — gives us the variable
    # features and LR pair set that ALL per-sample workers will reuse,
    # so the per-sample @net$prob tensors share dim names and can be
    # averaged element-wise without union/zero-padding.
    n_cells <- ncol(cobj@data.signaling)
    MERINGUE_MAX <- 30000
    if (n_cells > MERINGUE_MAX) {
        cat(sprintf("  [%s] subsampling %d -> %d cells for MERINGUE\\n",
            cond_label, n_cells, MERINGUE_MAX))
        set.seed(0)
        cell_idx <- sort(sample.int(n_cells, MERINGUE_MAX))
        norm_mat_sub <- cobj@data.signaling[, cell_idx]
        coord_sub <- cobj@images$coordinates[cell_idx, , drop = FALSE]
    } else {
        norm_mat_sub <- cobj@data.signaling
        coord_sub <- cobj@images$coordinates
    }
    cat(sprintf("  [%s] MERINGUE on %d cells x %d genes\\n",
        cond_label, ncol(norm_mat_sub), nrow(norm_mat_sub)))
    w_sub <- MERINGUE::getSpatialNeighbors(coord_sub, filterDist = NA)
    markers_meringue <- MERINGUE::getSpatialPatterns(norm_mat_sub, w_sub)
    features_sig <- rownames(markers_meringue)[
        markers_meringue$p.adj < 0.05 &
        !is.nan(markers_meringue$observed)]
    cobj@var.features[["features"]] <- features_sig
    cobj@var.features[["features.info"]] <- markers_meringue
    cat(sprintf("  [%s] MERINGUE found %d variable features\\n",
        cond_label, length(features_sig)))
    rm(norm_mat_sub, coord_sub, w_sub, markers_meringue); gc()

    cobj <- identifyOverExpressedInteractions(cobj, variable.both = FALSE)
    cat(sprintf("  [%s] LR sig pairs: %d\\n",
        cond_label, nrow(cobj@LR$LRsig)))

    saveRDS(cobj, file = pre_path)
    rm(sobj, cobj); gc()
    cat(sprintf("  [%s] saved pre.rds: %s\\n", cond_label, pre_path))
    ''')
    print(f'[{name}] {cond}: pre-step done')
    sys.exit(0)

# === Worker: sample-step (computeCommunProb on one full sample) ==============
if IS_WORKER and WORKER_STAGE == 'sample':
    name = WORKER_PLATFORM
    cfg = datasets[name]
    cond = WORKER_CONDITION
    samp = WORKER_SAMPLE
    pre_path = all_pre_paths[(name, cond)]
    sample_path = all_sample_paths[(name, cond, samp)]
    if os.path.exists(sample_path):
        print(f'[worker sample] {sample_path} cached, exiting')
        sys.exit(0)
    if not os.path.exists(pre_path):
        raise SystemExit(
            f'[worker sample] _pre.rds missing: {pre_path}')

    adata = adatas[name]
    # recompute conversion_factor (same RNG seed → same value as driver)
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(
        adata.n_obs, size=min(1000, adata.n_obs), replace=False)
    raw = adata.obs.iloc[sample_idx][['x_raw', 'y_raw']].values
    aff = adata.obs.iloc[sample_idx][['x_affine', 'y_affine']].values
    conversion_factor = float(np.median(pdist(raw)) / np.median(pdist(aff)))

    # Build adata for this (condition, sample): full sample, dedup only,
    # no hemisphere split. Spatial structure is preserved within the
    # sample; cross-sample overlap is moot because each worker handles
    # one sample in isolation.
    adata_cs = adata[(adata.obs['condition'] == cond) &
                     (adata.obs['sample'] == samp)].copy()
    adata_cs.obs = adata_cs.obs[
        ['sample', 'condition', cell_type_col, 'x_affine', 'y_affine']]
    adata_cs.uns, adata_cs.obsm, adata_cs.obsp = {}, {}, {}
    n_before = adata_cs.n_obs
    adata_cs, n_dropped = deduplicate_cells(
        adata_cs, DEDUP_MIN_DIST_UM, conversion_factor)
    n_dedup = adata_cs.n_obs
    # Uniform per-sample cap. Thins the spatial point process
    # homogeneously (preserves relative density between cell types and
    # within-tissue architecture). Slidetags samples (~5-8k after dedup)
    # are unaffected; xenium samples (>100k) get capped to keep
    # per-sample computeCommunProb under the 24h cluster job limit.
    adata_cs, n_sub = subsample_uniform(
        adata_cs, MAX_CELLS_COMPUTE_SAMPLE)
    adata_cs = adata_cs.copy()
    print(f'[{name}] {cond}/{samp}: full sample {n_before:,} → '
          f'dedup {n_dedup:,} (-{n_dropped}) → '
          f'subsample {adata_cs.n_obs:,} (-{n_sub})')

    SingleCell(adata_cs).to_seurat('sobj', v3=True)
    # shim to avoid layer issues with seurat
    r('''
    .gad_orig <- getFromNamespace("GetAssayData.Seurat", "SeuratObject")
    assignInNamespace("GetAssayData.Seurat", function(object,
            assay = NULL, slot = NULL, layer = NULL, ...) {
        if (!is.null(slot) && is.null(layer)) {
            layer <- slot; slot <- NULL
        }
        .gad_orig(object = object, assay = assay, layer = layer, ...)
    }, ns = "SeuratObject")
    ''')
    spatial_locs = adata_cs.obs[['x_affine', 'y_affine']]
    to_r(spatial_locs, 'spatial_locs', format='data.frame')
    to_r(pre_path, 'pre_path')
    to_r(sample_path, 'sample_path')
    to_r(samp, 'sample_label')
    to_r(cond, 'cond_label')
    to_r(cell_type_col, 'cell_type_col')
    to_r(conversion_factor, 'conversion_factor')
    to_r(SPOT_SIZE, 'spot_size')
    to_r(cfg['interaction_range'], 'interaction_range')
    to_r(scale_distance_per_dataset[name], 'scale_distance')
    to_r(CONTACT_RANGE, 'contact_range')

    # Build a per-sample cobj from raw expression, then graft the
    # variable-feature set and LR pair list from pre.rds so all per-sample
    # @net$prob tensors share the same LR dimension. preProcessing on the
    # per-sample data computes @data.signaling for the DB-relevant genes;
    # @LR$LRsig from pre selects which subset of those drives the LR loop
    # in computeCommunProb.
    r('''
    pre <- readRDS(pre_path)
    cat(sprintf("[%s/%s] pre.rds: %d var.features, %d LR sig pairs\\n",
        cond_label, sample_label,
        length(pre@var.features$features), nrow(pre@LR$LRsig)))

    sobj$samples <- sobj$sample
    sobj <- NormalizeData(sobj)
    sobj[[cell_type_col]] <- droplevels(sobj[[cell_type_col]])

    sf <- data.frame(ratio = conversion_factor, tol = spot_size / 2)
    cobj <- createSpatialCellChat(
        object = sobj, group.by = cell_type_col,
        assay = "RNA", datatype = "spatial",
        coordinates = spatial_locs[colnames(sobj), ],
        spatial.factors = sf)
    cobj@DB <- pre@DB
    cobj <- subsetData(cobj)
    cobj <- preProcessing(cobj)

    # Inherit the spatially-variable gene set and LR pair list from
    # pre.rds (which ran MERINGUE on pooled-condition data); ensures all
    # per-sample workers in this condition produce @net$prob tensors with
    # the same LR axis so the aggregator can average element-wise.
    cobj@var.features <- pre@var.features
    cobj@LR <- pre@LR
    rm(pre); gc()

    # CellChat requires @idents factor levels to match present types.
    cobj@idents <- droplevels(cobj@idents)
    cat(sprintf("[%s/%s] %d cells, %d cell types, %d LR pairs\\n",
        cond_label, sample_label,
        ncol(cobj@data.signaling),
        length(levels(cobj@idents)), nrow(cobj@LR$LRsig)))

    cat(sprintf("[%s/%s] computeCommunProb scale.distance=%.4f\\n",
        cond_label, sample_label, scale_distance))
    cobj <- computeCommunProb(
        cobj, distance.use = TRUE,
        interaction.range = interaction_range,
        scale.distance = scale_distance,
        contact.dependent = TRUE,
        contact.range = contact_range)

    # Drop multisession workers before any heavy serialization step;
    # filterProbability is deferred to the condition-aggregation step
    # (after averaging across samples) so we keep all entries here.
    plan("sequential")

    cat(sprintf("[%s/%s] prob sum: %.6f, nonzero: %d\\n",
        cond_label, sample_label,
        sum(cobj@net$prob, na.rm = TRUE),
        sum(cobj@net$prob > 0, na.rm = TRUE)))

    saveRDS(cobj, file = sample_path)
    rm(cobj, sobj); gc()
    ''')
    print(f'[{name}] {cond}/{samp}: saved {sample_path}')
    sys.exit(0)

# === Driver: dispatch + gate ================================================
if not IS_WORKER:
    active = _active_slurm_jobs()

    # Phase 1: pre-step jobs (one per (dataset, condition)) ------------------
    submitted_pre, pending_pre = [], []
    for (name, cond), pre_path in all_pre_paths.items():
        if os.path.exists(pre_path):
            continue
        tag = _cc_tag_pre(name, cond)
        if tag in active:
            pending_pre.append(tag); continue
        cmd = (f'{sys.executable} {os.path.abspath(__file__)} '
               f'--cellchat-job --stage=pre --platform={name} '
               f'--condition={cond} '
               f'--scale-distance={scale_distance_per_dataset[name]}')
        submit_slurm(cmd, job_name=tag,
                     log_file=f'{CC_LOG_DIR}/{tag}.log', hours=24)
        submitted_pre.append(tag); pending_pre.append(tag)

    if submitted_pre:
        print(f'[cellchat] submitted {len(submitted_pre)} pre-step jobs:')
        for t in submitted_pre: print(f'  {t}')
    elif pending_pre:
        print(f'[cellchat] {len(pending_pre)} pre-step jobs in flight:')
        for t in pending_pre: print(f'  {t}')

    missing_pre = [k for k, p in all_pre_paths.items()
                   if not os.path.exists(p)]
    if missing_pre:
        print(f'[cellchat] gating phase 1: {len(missing_pre)} pre.rds '
              f'still missing; rerun once jobs finish')
        sys.exit(0)
    print('[cellchat] all pre.rds cached')

    # Phase 2: per-sample jobs (one per (dataset, condition, sample)) --------
    submitted_smp, pending_smp = [], []
    for (name, cond, samp), sample_path in all_sample_paths.items():
        # Skip if final cond.rds already exists (no need to recompute).
        if os.path.exists(all_cond_paths[(name, cond)]):
            continue
        if os.path.exists(sample_path):
            continue
        tag = _cc_tag_sample(name, cond, samp)
        if tag in active:
            pending_smp.append(tag); continue
        cmd = (f'{sys.executable} {os.path.abspath(__file__)} '
               f'--cellchat-job --stage=sample --platform={name} '
               f'--condition={cond} --sample={samp} '
               f'--scale-distance={scale_distance_per_dataset[name]}')
        submit_slurm(cmd, job_name=tag,
                     log_file=f'{CC_LOG_DIR}/{tag}.log', hours=24)
        submitted_smp.append(tag); pending_smp.append(tag)

    if submitted_smp:
        print(f'[cellchat] submitted {len(submitted_smp)} sample jobs:')
        for t in submitted_smp: print(f'  {t}')
    elif pending_smp:
        print(f'[cellchat] {len(pending_smp)} sample jobs in flight:')
        for t in pending_smp: print(f'  {t}')

    # gate: missing per-sample RDS for any (name, cond) whose cond.rds
    # also doesn't exist
    missing_smp = []
    for (name, cond, samp), p in all_sample_paths.items():
        if os.path.exists(all_cond_paths[(name, cond)]):
            continue
        if not os.path.exists(p):
            missing_smp.append((name, cond, samp))
    if missing_smp:
        print(f'[cellchat] gating phase 2: {len(missing_smp)} sample.rds '
              f'still missing; rerun once jobs finish')
        sys.exit(0)
    print('[cellchat] all sample.rds cached (or condition aggregates exist)')

# === Driver: aggregate per-sample → per-condition ===========================
for (name, cond), cond_path in all_cond_paths.items():
    if os.path.exists(cond_path):
        print(f'[{name}] {cond}: cond.rds cached')
        continue
    sample_paths = [all_sample_paths[(name, cond, s)]
                    for s in samples_per[(name, cond)]]
    missing = [p for p in sample_paths if not os.path.exists(p)]
    if missing:
        # shouldn't happen due to gate, but defensive
        raise SystemExit(
            f'[{name}] {cond}: cannot aggregate, missing sample.rds: '
            f'{missing}')
    print(f'[{name}] {cond}: aggregating {len(sample_paths)} samples')
    to_r(sample_paths, 'sample_paths')
    to_r(cond_path, 'cond_path')
    to_r(f'{name}/{cond}', 'agg_label')
    r('''
    plan("sequential")
    n <- length(sample_paths)

    # Spatial computeCommunProb only sets @net$prob.cell (per-cell ×
    # per-cell × LR sparse3Darray); cell-type-aggregated @net$prob is
    # populated by the official computeAvgCommunProb step. Run that here
    # per sample (without permutation testing — we don't need pval), then
    # average the small (n_ct × n_ct × n_lr) tensors across samples.
    sum_prob <- NULL
    all_ct <- character(0)
    all_lr <- NULL
    all_idents_chr <- character(0)
    template <- NULL

    for (i in seq_along(sample_paths)) {
        cat(sprintf("[%s] loading sample %d/%d: %s\\n",
            agg_label, i, n, sample_paths[[i]]))
        co <- readRDS(sample_paths[[i]])

        cat(sprintf("[%s] sample %d: computeAvgCommunProb...\\n",
            agg_label, i))
        co <- computeAvgCommunProb(co, do.permutation = FALSE,
                                    avg.type = "avg")
        # @net$prob is now (n_ct_sample × n_ct_sample × n_lr)

        if (is.null(all_lr)) {
            all_lr <- dimnames(co@net$prob)[[3]]
        } else if (!identical(dimnames(co@net$prob)[[3]], all_lr)) {
            stop(sprintf("[%s] sample %d LR dim mismatch", agg_label, i))
        }

        sample_levs <- levels(co@idents)
        new_ct <- union(all_ct, sample_levs)

        # Re-allocate sum_prob with extended dim if cell-type union grew
        if (is.null(sum_prob) || length(new_ct) > length(all_ct)) {
            new_sum <- array(0,
                dim = c(length(new_ct), length(new_ct), length(all_lr)),
                dimnames = list(new_ct, new_ct, all_lr))
            if (!is.null(sum_prob)) {
                new_sum[rownames(sum_prob), colnames(sum_prob), ] <- sum_prob
            }
            sum_prob <- new_sum
            all_ct <- new_ct
        }
        sum_prob[sample_levs, sample_levs, ] <-
            sum_prob[sample_levs, sample_levs, ] + co@net$prob

        all_idents_chr <- c(all_idents_chr, as.character(co@idents))

        if (i == 1L) {
            # Save first sample as template, but strip the huge per-cell
            # slots (we keep only what downstream diff blocks need).
            template <- co
            template@net$prob.cell <- NULL
            template@net$tmp <- list()
        }

        rm(co); gc()
    }

    avg_prob <- sum_prob / n
    n_ct <- length(all_ct)

    template@net$prob <- avg_prob
    template@net$pval <- array(0, dim = dim(avg_prob),
                                dimnames = dimnames(avg_prob))
    template@net$weight <- apply(avg_prob, c(1, 2), sum)
    template@net$count <- apply(avg_prob > 0, c(1, 2), sum)
    # Pool @idents across all samples so filterCommunication's
    # min.cells threshold uses condition-wide cell counts per type.
    template@idents <- factor(all_idents_chr, levels = all_ct)

    cat(sprintf(
        "[%s] avg prob sum: %.6f, nonzero: %d (across %d samples, %d cell types, %d LR)\\n",
        agg_label, sum(avg_prob, na.rm = TRUE),
        sum(avg_prob > 0, na.rm = TRUE),
        n, n_ct, length(all_lr)))

    saveRDS(template, file = cond_path)
    rm(template, avg_prob, sum_prob); gc()
    ''')
    print(f'[{name}] {cond}: saved {cond_path}')

# === Driver: combine per-contrast pair RDS files (fast) =====================
for name, cfg in datasets.items():
    for treat, ctrl in cfg['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        rds_path = (f'{CC_OUT_DIR}/{name}/'
                    f'cellchat_{contrast}_{cell_type_col}.rds')
        if os.path.exists(rds_path):
            print(f'[{name}] {contrast}: cached')
            continue
        to_r(all_cond_paths[(name, ctrl)], 'ctrl_path')
        to_r(all_cond_paths[(name, treat)], 'treat_path')
        to_r(rds_path, 'rds_path')
        r('''
        saveRDS(list(readRDS(ctrl_path), readRDS(treat_path)),
                file = rds_path)
        ''')
        print(f'[{name}] {contrast}: combined {rds_path}')

#endregion

#region differential signaling — cell-type level ##############################

celltype_diff_frames = []
for name, cfg in datasets.items():
    for treat, ctrl in cfg['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        rds_path = (f'{CC_OUT_DIR}/{name}/'
                    f'cellchat_{contrast}_{cell_type_col}.rds')
        to_r(rds_path, 'rds_path')
        to_r(ctrl, 'cond_ctrl')
        to_r(treat, 'cond_treat')

        r('''
        # Diff steps inherit the multisession future plan from setup;
        # filterCommunication / aggregateNet / computeCommunProbPathway /
        # mergeSpatialCellChat all clone the cobj per worker, blowing
        # memory on pooled-sample cobjs. These steps are seconds-to-
        # minutes sequentially; drop workers.
        plan("sequential")
        cobjs <- readRDS(rds_path)
        cobj_1 <- cobjs[[1]]
        cobj_2 <- cobjs[[2]]

        # filterCommunication() unconditionally errors when @net$prob.cell
        # is absent (aggregator stripped it). We only need its first
        # block (zero out cell types with <10 cells); inline it.
        zero_low_cell_types <- function(co, min.cells = 10) {
            ex <- which(as.numeric(table(co@idents)) < min.cells)
            if (length(ex) > 0L) {
                cat(sprintf(
                    "  zeroing %d cell types with <%d cells: %s\\n",
                    length(ex), min.cells,
                    paste(levels(co@idents)[ex], collapse = ", ")))
                co@net$prob[ex, , ] <- 0
                co@net$prob[, ex, ] <- 0
                if (!is.null(co@net$pval)) {
                    co@net$pval[co@net$prob == 0] <- 1
                }
            }
            co
        }
        cobj_1 <- zero_low_cell_types(cobj_1, 10)
        cobj_2 <- zero_low_cell_types(cobj_2, 10)
        cobj_1 <- aggregateNet(cobj_1)
        cobj_2 <- aggregateNet(cobj_2)

        cc <- mergeSpatialCellChat(list(cobj_1, cobj_2),
                            add.names = c(cond_ctrl, cond_treat),
                            show.plot = FALSE)

        g1_w <- cc@net[[cond_ctrl]]$weight
        g2_w <- cc@net[[cond_treat]]$weight
        all_ct <- sort(unique(c(rownames(g1_w), rownames(g2_w),
                                colnames(g1_w), colnames(g2_w))))

        pad <- function(m, ct) {
            out <- matrix(0, length(ct), length(ct),
                          dimnames = list(ct, ct))
            if (!is.null(m)) out[rownames(m), colnames(m)] <- m
            out
        }

        diff_w <- pad(g2_w, all_ct) - pad(g1_w, all_ct)
        net_df <- reshape2::melt(diff_w, value.name = "weight_diff")
        colnames(net_df)[1:2] <- c("sender", "receiver")
        net_df$contrast <- paste0(cond_treat, "_vs_", cond_ctrl)
        ''')
        df = to_py('net_df', format='pandas')
        df['dataset'] = name
        celltype_diff_frames.append(df)
        n_nonzero = (df['weight_diff'].abs() > 1e-10).sum()
        print(f'[{name}] {contrast}: {n_nonzero} non-zero cell-type pairs')

celltype_diff = pd.concat(celltype_diff_frames, ignore_index=True)
# Catch return of the zero-prob bug: each (dataset, contrast) must have
# weight_diff with both positive and negative values.
for (ds, ctr), sub in celltype_diff.groupby(['dataset', 'contrast']):
    has_pos = bool((sub['weight_diff'] > 0).any())
    has_neg = bool((sub['weight_diff'] < 0).any())
    assert has_pos and has_neg, (
        f'[{ds}] {ctr}: weight_diff has uniform sign '
        f'(pos={has_pos}, neg={has_neg}); zero-prob bug suspected')
celltype_diff.to_csv(
    f'{CC_OUT_DIR}/celltype_diff.csv', index=False)

#endregion

#region differential signaling — pathway level ################################

pathway_diff_frames = []
for name, cfg in datasets.items():
    for treat, ctrl in cfg['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        rds_path = (f'{CC_OUT_DIR}/{name}/'
                    f'cellchat_{contrast}_{cell_type_col}.rds')
        to_r(rds_path, 'rds_path')
        to_r(ctrl, 'cond_ctrl')
        to_r(treat, 'cond_treat')

        r('''
        plan("sequential")
        cobjs <- readRDS(rds_path)
        cobj_1 <- cobjs[[1]]
        cobj_2 <- cobjs[[2]]

        # filterCommunication() unconditionally errors when @net$prob.cell
        # is absent (aggregator stripped it). Inline the min.cells filter.
        zero_low_cell_types <- function(co, min.cells = 10) {
            ex <- which(as.numeric(table(co@idents)) < min.cells)
            if (length(ex) > 0L) {
                co@net$prob[ex, , ] <- 0
                co@net$prob[, ex, ] <- 0
                if (!is.null(co@net$pval)) {
                    co@net$pval[co@net$prob == 0] <- 1
                }
            }
            co
        }
        cobj_1 <- zero_low_cell_types(cobj_1, 10)
        cobj_2 <- zero_low_cell_types(cobj_2, 10)
        # do.cell = FALSE: aggregator stripped @net$prob.cell to keep
        # cond.rds small. The do.group path (default) uses @net$prob
        # which we populated via computeAvgCommunProb.
        cobj_1 <- tryCatch(computeCommunProbPathway(cobj_1, do.cell = FALSE),
                           error = function(e) cobj_1)
        cobj_2 <- tryCatch(computeCommunProbPathway(cobj_2, do.cell = FALSE),
                           error = function(e) cobj_2)

        prob1 <- cobj_1@netP$prob
        prob2 <- cobj_2@netP$prob
        pw1 <- if (!is.null(prob1)) dimnames(prob1)[[3]] else character(0)
        pw2 <- if (!is.null(prob2)) dimnames(prob2)[[3]] else character(0)
        all_pw <- unique(c(pw1, pw2))
        all_ct <- sort(unique(c(
            levels(cobj_1@idents), levels(cobj_2@idents))))

        cat(sprintf("[R DIAG] prob1 null: %s, prob2 null: %s\n",
            is.null(prob1), is.null(prob2)))
        if (!is.null(prob1)) cat(sprintf(
            "[R DIAG] prob1 dim: %s, pathways: %d\n",
            paste(dim(prob1), collapse="x"), length(pw1)))
        if (!is.null(prob2)) cat(sprintf(
            "[R DIAG] prob2 dim: %s, pathways: %d\n",
            paste(dim(prob2), collapse="x"), length(pw2)))
        cat(sprintf("[R DIAG] all_pw: %d, all_ct: %d\n",
            length(all_pw), length(all_ct)))

        pw_df <- data.frame(sender = character(),
                            receiver = character(),
                            prob_diff = numeric(),
                            pathway = character(),
                            contrast = character(),
                            stringsAsFactors = FALSE)

        if (length(all_pw) > 0 && length(all_ct) > 0) {
            pad_pw <- function(prob, pw, ct) {
                m <- matrix(0, length(ct), length(ct),
                            dimnames = list(ct, ct))
                if (!is.null(prob) && pw %in% dimnames(prob)[[3]]) {
                    p <- prob[, , pw]
                    if (is.matrix(p)) {
                        m[rownames(p), colnames(p)] <- p
                    } else {
                        m[names(p)[1], names(p)[2]] <- p
                    }
                }
                m
            }
            pw_diffs <- list()
            for (pw in all_pw) {
                m1 <- pad_pw(prob1, pw, all_ct)
                m2 <- pad_pw(prob2, pw, all_ct)
                d <- reshape2::melt(m2 - m1, value.name = "prob_diff")
                colnames(d)[1:2] <- c("sender", "receiver")
                d$pathway <- pw
                d$contrast <- paste0(cond_treat, "_vs_", cond_ctrl)
                pw_diffs[[pw]] <- d
            }
            pw_df <- do.call(rbind, pw_diffs)
            cat(sprintf("[R DIAG] pw_df: %d rows\n", nrow(pw_df)))
        }
        ''')
        df = to_py('pw_df', format='pandas')
        if df is not None and len(df) > 0:
            df['dataset'] = name
            pathway_diff_frames.append(df)
            n_pw = df['pathway'].nunique()
            print(f'[{name}] {contrast}: {n_pw} pathways')
        else:
            print(f'[{name}] {contrast}: no pathways detected')

pathway_diff = pd.concat(pathway_diff_frames, ignore_index=True)
pathway_diff.to_csv(
    f'{CC_OUT_DIR}/pathway_diff.csv', index=False)

#endregion
