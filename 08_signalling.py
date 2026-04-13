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

SCALE_DISTANCE = 100
CONTACT_RANGE = 30
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
suppressPackageStartupMessages({
    library(CellChat)
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
    if (!col %in% names(cpdb_formatted)) cpdb_formatted[[col]] <- ''
    if (!col %in% names(neuron_chat_formatted)) neuron_chat_formatted[[col]] <- ''
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

#region compute cellchat per dataset x contrast ################################

for name, cfg in datasets.items():
    adata = adatas[name]

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(adata.n_obs, size=min(1000, adata.n_obs),
                            replace=False)
    raw = adata.obs.iloc[sample_idx][['x_raw', 'y_raw']].values
    aff = adata.obs.iloc[sample_idx][['x_affine', 'y_affine']].values
    conversion_factor = float(np.median(pdist(raw)) / np.median(pdist(aff)))
    print(f'[{name}] conversion factor: {conversion_factor:.2f}')

    for treat, ctrl in cfg['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        rds_path = (f'{working_dir}/output/{name}/'
                    f'cellchat_{contrast}_{cell_type_col}.rds')
        if os.path.exists(rds_path):
            print(f'[{name}] {contrast}: cached {rds_path}')
            continue

        adata_sub = adata.copy()
        adata_sub = adata_sub[
            adata_sub.obs['condition'].isin([treat, ctrl])].copy()
        adata_sub.obs = adata_sub.obs[
            ['sample', 'condition', cell_type_col,
             'x_raw', 'y_raw', 'x_affine', 'y_affine']]
        adata_sub.uns, adata_sub.obsm, adata_sub.obsp = {}, {}, {}

        SingleCell(adata_sub).to_seurat('sobj', v3=True)
        spatial_locs = adata_sub.obs[['x_affine', 'y_affine']]
        to_r(spatial_locs, 'spatial_locs', format='data.frame')
        to_r(ctrl, 'cond_ctrl')
        to_r(treat, 'cond_treat')
        to_r(cell_type_col, 'cell_type_col')
        to_r(conversion_factor, 'conversion_factor')
        to_r(rds_path, 'rds_path')
        to_r(cfg['interaction_range'], 'interaction_range')
        to_r(SCALE_DISTANCE, 'scale_distance')
        to_r(CONTACT_RANGE, 'contact_range')
        to_r(SPOT_SIZE, 'spot_size')

        r('''
        sobj$samples <- sobj$sample
        sobj <- NormalizeData(sobj)

        sobj_1 <- subset(sobj, subset = condition == cond_ctrl)
        sobj_2 <- subset(sobj, subset = condition == cond_treat)
        sobj_1[[cell_type_col]] <- droplevels(sobj_1[[cell_type_col]])
        sobj_2[[cell_type_col]] <- droplevels(sobj_2[[cell_type_col]])
        rm(sobj); gc()

        sf <- data.frame(ratio = conversion_factor,
                         tol = spot_size / 2)
        cc_params <- list(type = "truncatedMean", trim = 0.1,
                          distance.use = TRUE,
                          interaction.range = interaction_range,
                          scale.distance = scale_distance,
                          contact.range = contact_range)
        ''')

        for cond_label, sobj_name in [('ctrl', 'sobj_1'),
                                       ('treat', 'sobj_2')]:
            to_r(cond_label, 'cond_label')
            to_r(sobj_name, 'sobj_name')
            r(f'''
            cobj <- createCellChat(
                object = get(sobj_name), group.by = cell_type_col,
                assay = "RNA", datatype = "spatial",
                coordinates = spatial_locs[colnames(get(sobj_name)), ],
                spatial.factors = sf)
            cobj@DB <- CellChatDB_ext
            cobj <- subsetData(cobj)
            cobj <- identifyOverExpressedGenes(cobj)
            cobj <- identifyOverExpressedInteractions(cobj)
            cobj <- do.call(computeCommunProb, c(list(object = cobj),
                            cc_params))
            cat(sprintf("  [%s] prob sum: %.6f, nonzero: %d\\n",
                cond_label, sum(cobj@net$prob, na.rm=TRUE),
                sum(cobj@net$prob > 0, na.rm=TRUE)))
            assign(paste0("cobj_", cond_label), cobj)
            rm(cobj); gc()
            ''')

        r('saveRDS(list(cobj_ctrl, cobj_treat), file = rds_path)')
        print(f'[{name}] {contrast}: saved {rds_path}')

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

        cc <- mergeCellChat(list(cobj_1, cobj_2),
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

    sample_idx = rng.choice(min(1000, len(coords_raw)), size=min(1000, len(coords_raw)),
                            replace=False)
    cf = float(np.median(pdist(coords_raw[sample_idx])) /
               np.median(pdist(coords_aff[sample_idx])))
    r_affine = cfg['interaction_range'] / cf

    ax = axes[row, 0]
    ax.scatter(coords_aff[:, 0], coords_aff[:, 1], s=0.5, c='lightgray',
               alpha=0.5, linewidth=0, rasterized=True)
    exemplar_idx = rng.integers(len(coords_aff), size=n_exemplars)
    exemplars = coords_aff[exemplar_idx]
    ax.scatter(exemplars[:, 0], exemplars[:, 1], s=20, c='red', zorder=5)
    for cell in exemplars:
        ax.add_patch(Circle(cell, r_affine, fill=False, color='red',
                            linewidth=1, linestyle='--'))
    ax.set_aspect('equal')
    ax.set_title(f'{name} — {sample}\n'
                 f'range={cfg["interaction_range"]}μm '
                 f'(affine r={r_affine:.4f})',
                 fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    from scipy.spatial import KDTree
    tree = KDTree(coords_aff)
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
            ax.scatter(coords_aff[neighbor_idx, 0],
                       coords_aff[neighbor_idx, 1],
                       s=10, c='steelblue', alpha=0.85, linewidth=0,
                       rasterized=True)
        ax.scatter(cell[0], cell[1], s=80, c='red', zorder=5,
                   edgecolors='white', linewidths=1)
        ax.add_patch(Circle(cell, r_affine, fill=False, color='red',
                            linewidth=1.5, linestyle='--'))
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
plt.savefig(f'{working_dir}/figures/cellchat_interaction_range.svg',
            bbox_inches='tight')
plt.close()

#endregion
