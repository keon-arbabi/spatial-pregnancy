#region imports and setup ######################################################

import os
import sys
import pickle as pkl
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.spatial.distance import pdist
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r, to_py
sys.path.insert(0, os.path.expanduser('~'))
from single_cell import SingleCell

warnings.filterwarnings('ignore')

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
cellphonedb_path = f'{working_dir}/input/cellphonedb'
ortholog_cache = f'{cellphonedb_path}/gprofiler_orthologs.pkl'

SCALE_DISTANCE = None  # computed per dataset as conversion_factor / interaction_range
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
})

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

#region compute cellchat per dataset x condition ###############################

for name, cfg in datasets.items():
    adata = adatas[name]
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(
        adata.n_obs, size=min(1000, adata.n_obs), replace=False)
    raw = adata.obs.iloc[sample_idx][['x_raw', 'y_raw']].values
    aff = adata.obs.iloc[sample_idx][['x_affine', 'y_affine']].values
    conversion_factor = float(np.median(pdist(raw)) / np.median(pdist(aff)))
    scale_distance = conversion_factor / cfg['interaction_range']
    print(f'[{name}] conversion factor: {conversion_factor:.2f}, '
          f'scale.distance: {scale_distance:.4f}')

    cond_rds = {}
    conditions = sorted({
        c for pair in cfg['contrasts'] for c in pair})
    for cond in conditions:
        cond_path = (
            f'{working_dir}/output/{name}/'
            f'cellchat_cond_{cond}_{cell_type_col}.rds')
        cond_rds[cond] = cond_path
        if os.path.exists(cond_path):
            print(f'[{name}] {cond}: cached {cond_path}')
            continue

        adata_cond = adata[adata.obs['condition'] == cond].copy()
        adata_cond.obs = adata_cond.obs[
            ['sample', 'condition', cell_type_col, 'x_affine', 'y_affine']]
        adata_cond.uns, adata_cond.obsm, adata_cond.obsp = {}, {}, {}

        SingleCell(adata_cond).to_seurat('sobj', v3=True)
        # shim to avoid layer issues with seurat
        r('''
        .gad_orig <- getFromNamespace("GetAssayData.Seurat", "SeuratObject")
        assignInNamespace("GetAssayData.Seurat", function(object,
                assay = NULL, slot = NULL, layer = NULL, ...) {
            if (!is.null(slot) && is.null(layer)) {
                layer <- slot
                slot <- NULL
            }
            .gad_orig(object = object, assay = assay, layer = layer, ...)
        }, ns = "SeuratObject")
        ''')
        spatial_locs = adata_cond.obs[['x_affine', 'y_affine']]
        to_r(spatial_locs, 'spatial_locs', format='data.frame')
        to_r(cell_type_col, 'cell_type_col')
        to_r(conversion_factor, 'conversion_factor')
        to_r(cfg['interaction_range'], 'interaction_range')
        to_r(scale_distance, 'scale_distance')
        to_r(CONTACT_RANGE, 'contact_range')
        to_r(SPOT_SIZE, 'spot_size')
        to_r(cond_path, 'cond_path')
        to_r(cond, 'cond_label')

        r('''
        sobj$samples <- sobj$sample
        sobj <- NormalizeData(sobj)
        sobj[[cell_type_col]] <- droplevels(sobj[[cell_type_col]])

        sf <- data.frame(ratio = conversion_factor,
                         tol = spot_size / 2)

        cobj <- createSpatialCellChat(
            object = sobj, group.by = cell_type_col,
            assay = "RNA", datatype = "spatial",
            coordinates = spatial_locs[colnames(sobj), ],
            spatial.factors = sf)
        cobj@DB <- CellChatDB_ext
        cobj <- subsetData(cobj)
        cobj <- preProcessing(cobj)
        cobj <- identifyOverExpressedGenes(
            cobj, selection.method = "meringue", do.grid = FALSE)
        cobj <- identifyOverExpressedInteractions(
            cobj, variable.both = FALSE)

        cat(sprintf("  [%s] pre-computeCommunProb: data.signaling dim %s, LRsig %d pairs\\n",
            cond_label, paste(dim(cobj@data.signaling), collapse="x"),
            nrow(cobj@LR$LRsig)))

        cobj <- computeCommunProb(
            cobj, distance.use = TRUE,
            interaction.range = interaction_range,
            scale.distance = scale_distance,
            contact.dependent = TRUE,
            contact.range = contact_range)
        cobj <- filterProbability(cobj)

        cat(sprintf("  [%s] prob sum: %.6f, nonzero: %d\\n",
            cond_label, sum(cobj@net$prob, na.rm=TRUE),
            sum(cobj@net$prob > 0, na.rm=TRUE)))

        saveRDS(cobj, file = cond_path)
        rm(cobj, sobj); gc()
        ''')
        print(f'[{name}] {cond}: saved {cond_path}')

    for treat, ctrl in cfg['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        rds_path = (f'{working_dir}/output/{name}/'
                    f'cellchat_{contrast}_{cell_type_col}.rds')
        if os.path.exists(rds_path):
            print(f'[{name}] {contrast}: cached')
            continue
        to_r(cond_rds[ctrl], 'ctrl_path')
        to_r(cond_rds[treat], 'treat_path')
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
        rds_path = (f'{working_dir}/output/{name}/'
                    f'cellchat_{contrast}_{cell_type_col}.rds')
        to_r(rds_path, 'rds_path')
        to_r(ctrl, 'cond_ctrl')
        to_r(treat, 'cond_treat')

        r('''
        cobjs <- readRDS(rds_path)
        cobj_1 <- cobjs[[1]]
        cobj_2 <- cobjs[[2]]

        cobj_1 <- filterCommunication(cobj_1, min.cells = 10)
        cobj_2 <- filterCommunication(cobj_2, min.cells = 10)
        cobj_1 <- aggregateNet(cobj_1)
        cobj_2 <- aggregateNet(cobj_2)

        cc <- mergeSpatialCellChat(list(cobj_1, cobj_2),
                            add.names = c(cond_ctrl, cond_treat))

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
celltype_diff.to_csv(
    f'{working_dir}/output/cellchat_celltype_diff.csv', index=False)

#endregion

#region differential signaling — pathway level ################################

pathway_diff_frames = []
for name, cfg in datasets.items():
    for treat, ctrl in cfg['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        rds_path = (f'{working_dir}/output/{name}/'
                    f'cellchat_{contrast}_{cell_type_col}.rds')
        to_r(rds_path, 'rds_path')
        to_r(ctrl, 'cond_ctrl')
        to_r(treat, 'cond_treat')

        r('''
        cobjs <- readRDS(rds_path)
        cobj_1 <- cobjs[[1]]
        cobj_2 <- cobjs[[2]]

        cobj_1 <- filterCommunication(cobj_1, min.cells = 10)
        cobj_2 <- filterCommunication(cobj_2, min.cells = 10)
        cobj_1 <- tryCatch(computeCommunProbPathway(cobj_1),
                           error = function(e) cobj_1)
        cobj_2 <- tryCatch(computeCommunProbPathway(cobj_2),
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
    f'{working_dir}/output/cellchat_pathway_diff.csv', index=False)

#endregion

#region plot — interaction range visualization ################################

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

from scipy.spatial import KDTree
from matplotlib.colors import Normalize
from matplotlib import cm

n_exemplars = 5
fig, axes = plt.subplots(
    len(datasets), n_exemplars + 1,
    figsize=(3 * (n_exemplars + 1), 2.8 * len(datasets)),
    squeeze=False)
rng = np.random.default_rng(42)

for row, (name, cfg) in enumerate(datasets.items()):
    adata = adatas[name]
    sample = sorted(adata.obs['sample'].unique())[0]
    sub = adata.obs[adata.obs['sample'] == sample]
    coords_raw = sub[['x_raw', 'y_raw']].to_numpy(dtype=np.float64)
    coords_aff = sub[['x_affine', 'y_affine']].to_numpy(dtype=np.float64)
    sample_idx = rng.choice(min(1000, len(coords_raw)),
                            size=min(1000, len(coords_raw)), replace=False)
    cf = float(np.median(pdist(coords_raw[sample_idx])) /
               np.median(pdist(coords_aff[sample_idx])))
    r_affine = cfg['interaction_range'] / cf
    r_contact = CONTACT_RANGE / cf
    sd = cf / cfg['interaction_range']
    ax = axes[row, 0]
    ax.scatter(coords_aff[:, 0], coords_aff[:, 1], s=0.5, c='lightgray',
               alpha=0.5, linewidth=0, rasterized=True)
    exemplar_idx = rng.integers(len(coords_aff), size=n_exemplars)
    exemplars = coords_aff[exemplar_idx]
    ax.scatter(exemplars[:, 0], exemplars[:, 1], s=20, c='red', zorder=5)
    for cell in exemplars:
        ax.add_patch(Circle(cell, r_affine, fill=False, color='red',
                            linewidth=0.8, linestyle='--'))
        ax.add_patch(Circle(cell, r_contact, fill=False, color='orange',
                            linewidth=0.8, linestyle='-'))
    ax.set_aspect('equal')
    ax.set_title(f'{name} — {sample}\n'
                 f'interaction={cfg["interaction_range"]}μm, '
                 f'contact={CONTACT_RANGE}μm\n'
                 f'scale.distance={sd:.4f}',
                 fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    tree = KDTree(coords_aff)
    cmap = cm.Blues
    for col, (idx, cell) in enumerate(zip(exemplar_idx, exemplars), start=1):
        neighbor_idx = tree.query_ball_point(cell, r=r_affine)
        neighbor_idx = [i for i in neighbor_idx if i != idx]
        zoom = r_affine * 2.5
        ax = axes[row, col]
        in_zoom = ((np.abs(coords_aff[:, 0] - cell[0]) < zoom * 1.5) &
                   (np.abs(coords_aff[:, 1] - cell[1]) < zoom * 1.5))
        ax.scatter(coords_aff[in_zoom, 0], coords_aff[in_zoom, 1],
                   s=6, c='lightgray', alpha=0.7, linewidth=0,
                   rasterized=True)
        if neighbor_idx:
            nb_coords = coords_aff[neighbor_idx]
            dists_aff = np.sqrt(((nb_coords - cell) ** 2).sum(axis=1))
            dists_um = dists_aff * cf
            weights = np.exp(-sd * dists_um)
            colors = cmap(Normalize(0, 1)(weights))
            ax.scatter(nb_coords[:, 0], nb_coords[:, 1],
                       s=10, c=colors, linewidth=0, rasterized=True)
        ax.scatter(cell[0], cell[1], s=80, c='red', zorder=5,
                   edgecolors='white', linewidths=1)
        ax.add_patch(Circle(cell, r_affine, fill=False, color='red',
                            linewidth=1.2, linestyle='--'))
        ax.add_patch(Circle(cell, r_contact, fill=False, color='orange',
                            linewidth=1.2, linestyle='-'))
        ax.set_xlim(cell[0] - zoom, cell[0] + zoom)
        ax.set_ylim(cell[1] - zoom, cell[1] + zoom)
        ax.set_aspect('equal')
        ax.set_title(f'{len(neighbor_idx):,} in range', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)

plt.tight_layout()
os.makedirs(f'{working_dir}/figures', exist_ok=True)
plt.savefig(f'{working_dir}/figures/cellchat_interaction_range.png',
            dpi=200, bbox_inches='tight')
plt.close()

#endregion

'''
[slidetags] correction: mean=12.3%, median=9.8%
  01 IT-ET Glut: 12.3% (20,495 cells)
  02 NP-CT-L6b Glut: 12.6% (5,079 cells)
  03 OB-CR Glut: 15.9% (25 cells)
  04 DG-IMN Glut: 8.2% (20 cells)
  05 OB-IMN GABA: 12.2% (543 cells)
  06 CTX-CGE GABA: 12.7% (2,153 cells)
  07 CTX-MGE GABA: 12.5% (2,768 cells)
  08 CNU-MGE GABA: 12.8% (2,920 cells)
  09 CNU-LGE GABA: 12.6% (16,725 cells)
  10 LSX GABA: 13.5% (1,798 cells)
  11 CNU-HYa GABA: 13.2% (5,345 cells)
  12 HY GABA: 12.6% (887 cells)
  13 CNU-HYa Glut: 13.4% (2,395 cells)
  14 HY Glut: 12.1% (586 cells)
  15 HY Gnrh1 Glut: 18.5% (3 cells)
  18 TH Glut: 14.9% (17 cells)
  19 MB Glut: 14.6% (116 cells)
  20 MB GABA: 15.7% (30 cells)
  23 P Glut: 15.5% (4 cells)
  24 MY Glut: 13.6% (10 cells)
  30 Astro-Epen: 11.9% (7,684 cells)
  31 OPC-Oligo: 11.4% (9,769 cells)
  33 Vascular: 11.0% (6,020 cells)
  34 Immune: 11.8% (2,099 cells)
[slidetags] X: [[3. 1. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]
 [2. 0. 0. 0. 0. 0.]]
  drop subclass_confidence: 9,886 (11.3%)
  drop subclass_margin: 7,726 (8.8%)
  drop min_cos_dist: 8,379 (9.6%)
  drop n_spatial_candidates: 283 (0.3%)
  drop rare (<10/sample): 1,248 (1.4%)
  total dropped: 16,911 (19.3%)
  keep: 70,580 (80.7%)

[xenium] correction: mean=32.6%, median=28.6%
  01 IT-ET Glut: 24.7% (138,417 cells)
  02 NP-CT-L6b Glut: 25.4% (40,379 cells)
  03 OB-CR Glut: 46.2% (45 cells)
  04 DG-IMN Glut: 36.0% (945 cells)
  05 OB-IMN GABA: 27.9% (6,392 cells)
  06 CTX-CGE GABA: 35.9% (9,984 cells)
  07 CTX-MGE GABA: 24.6% (17,540 cells)
  08 CNU-MGE GABA: 24.8% (14,117 cells)
  09 CNU-LGE GABA: 28.7% (165,437 cells)
  10 LSX GABA: 27.6% (16,546 cells)
  11 CNU-HYa GABA: 26.1% (34,777 cells)
  12 HY GABA: 23.6% (9,788 cells)
  13 CNU-HYa Glut: 26.6% (18,053 cells)
  14 HY Glut: 25.2% (6,951 cells)
  15 HY Gnrh1 Glut: 36.6% (16 cells)
  18 TH Glut: 20.8% (128 cells)
  19 MB Glut: 25.8% (912 cells)
  20 MB GABA: 24.0% (56 cells)
  23 P Glut: 34.5% (6 cells)
  24 MY Glut: 26.8% (111 cells)
  30 Astro-Epen: 39.3% (113,937 cells)
  31 OPC-Oligo: 33.5% (117,895 cells)
  33 Vascular: 45.9% (122,222 cells)
  34 Immune: 41.9% (33,050 cells)
[xenium] X: [[0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1.]]
  drop subclass_confidence: 76,340 (8.8%)
  drop subclass_margin: 61,714 (7.1%)
  drop min_cos_dist: 6,805 (0.8%)
  drop n_spatial_candidates: 67 (0.0%)
  drop rare (<10/sample): 681 (0.1%)
  total dropped: 82,294 (9.5%)
  keep: 785,410 (90.5%)

[merfish] correction: mean=13.9%, median=12.7%
  01 IT-ET Glut: 12.8% (165,994 cells)
  02 NP-CT-L6b Glut: 11.8% (43,622 cells)
  03 OB-CR Glut: 15.0% (123 cells)
  04 DG-IMN Glut: 13.5% (273 cells)
  05 OB-IMN GABA: 13.5% (2,520 cells)
  06 CTX-CGE GABA: 13.0% (27,729 cells)
  07 CTX-MGE GABA: 13.6% (19,974 cells)
  08 CNU-MGE GABA: 14.0% (30,668 cells)
  09 CNU-LGE GABA: 11.2% (186,565 cells)
  10 LSX GABA: 14.6% (12,188 cells)
  11 CNU-HYa GABA: 13.9% (22,580 cells)
  12 HY GABA: 17.6% (6,697 cells)
  13 CNU-HYa Glut: 16.0% (13,680 cells)
  14 HY Glut: 17.9% (2,817 cells)
  15 HY Gnrh1 Glut: 24.0% (45 cells)
  18 TH Glut: 19.5% (28 cells)
  19 MB Glut: 17.3% (93 cells)
  20 MB GABA: 16.2% (79 cells)
  23 P Glut: 10.0% (1 cells)
  24 MY Glut: 16.9% (80 cells)
  30 Astro-Epen: 14.9% (187,116 cells)
  31 OPC-Oligo: 13.0% (136,385 cells)
  33 Vascular: 18.6% (78,460 cells)
  34 Immune: 18.7% (52,930 cells)
[merfish] X: [[14.  0.  0.  0.  1.  1.]
 [ 0.  0.  0.  0.  0.  3.]
 [ 0.  0.  0.  0.  0.  4.]]
  drop subclass_confidence: 139,926 (14.1%)
  drop subclass_margin: 105,999 (10.7%)
  drop min_cos_dist: 41,805 (4.2%)
  drop n_spatial_candidates: 7 (0.0%)
  drop rare (<10/sample): 703 (0.1%)
  total dropped: 166,848 (16.8%)
  keep: 823,799 (83.2%)
'''
