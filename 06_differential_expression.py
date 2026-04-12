#region imports and setup ######################################################

import os
import gc
import re
import pickle as pkl
import warnings
import numpy as np
import pandas as pd
import polars as pl
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
from ryp import r, to_r, to_py

warnings.filterwarnings('ignore')

from single_cell import SingleCell

working_dir = '/home/karbabi/spatial-pregnancy'
cell_type_col = 'subclass'
REF_PCT_THRESHOLD = 5

datasets = {
    'slidetags': {
        'path': f'{working_dir}/output/slidetags/03_adata_query_slidetags.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
        'design': '~ group + log_num_cells + log_lib_size',
    },
    'merfish': {
        'path': f'{working_dir}/output/merfish/03_adata_query_merfish.h5ad',
        'contrasts': [('PREG', 'CTRL'), ('POSTPART', 'PREG'),
                      ('POSTPART', 'CTRL')],
        'design': '~ group + log_num_cells + log_lib_size',
    },
    'xenium': {
        'path': f'{working_dir}/output/xenium/03_adata_query_xenium.h5ad',
        'contrasts': [('PREG', 'CTRL')],
        'drop_samples': ['CTRL_3'],
        'design': '~ group + log_num_cells',
    },
}

pct_file = f'{working_dir}/output/ref_pct_detected.pkl'
if not os.path.exists(pct_file):
    ref = sc.read_h5ad('single-cell/ABC/zeng_combined_10Xv3.h5ad')
    ref.var_names_make_unique()
    pct_detected = {}
    for s in ref.obs[cell_type_col].dropna().unique():
        mask = ref.obs[cell_type_col] == s
        n_cells = mask.sum()
        if n_cells < 10:
            continue
        X_sub = ref[mask].X
        detected = np.ravel((X_sub > 0).sum(axis=0)) / n_cells
        pct_detected[s] = pd.Series(detected, index=ref.var_names)
    pkl.dump(pct_detected, open(pct_file, 'wb'))
    del ref; gc.collect()
else:
    pct_detected = pkl.load(open(pct_file, 'rb'))

def get_expressed_genes(cell_type, gene_list):
    if cell_type not in pct_detected:
        return gene_list
    ref = pct_detected[cell_type]
    return [g for g in gene_list
            if g not in ref.index or ref[g] >= REF_PCT_THRESHOLD / 100]

def get_ref_pct(cell_type, gene):
    if cell_type in pct_detected and gene in pct_detected[cell_type].index:
        return round(pct_detected[cell_type][gene] * 100, 1)
    return None

protein_coding_genes = pd.read_csv(
    f'{working_dir}/input/MRK_ENSEMBL.csv', header=None)
protein_coding_genes = protein_coding_genes[
    protein_coding_genes[8] == 'protein coding gene'][1].to_list()

adatas = {}
for name, cfg in datasets.items():
    adata = sc.read_h5ad(cfg['path'])
    adata.var_names_make_unique()
    if 'gene_symbol' in adata.var.columns:
        adata.var.index = adata.var['gene_symbol']
        adata.var_names_make_unique()
        adata.var.drop(columns='gene_symbol', inplace=True)
    adata.var.index.name = None
    g = adata.var_names
    adata.var['mt'] = g.str.match(r'^(mt-|MT-)')
    adata.var['ribo'] = g.str.match(r'^(Rps|Rpl)')
    adata.var['protein_coding'] = g.isin(protein_coding_genes)
    drop = cfg.get('drop_samples', [])
    if drop:
        adata = adata[~adata.obs['sample'].isin(drop)].copy()
        print(f'[{name}] dropped samples: {drop}')
    adatas[name] = adata
    print(f'[{name}] {adata.shape[0]:,} cells, '
          f'{adata.obs[cell_type_col].nunique()} subclasses, '
          f'{adata.obs["condition"].nunique()} conditions')

#endregion

#region run de #################################################################

def make_pseudobulk(adata, name):
    sc_obj = SingleCell(adata).skip_qc()
    if name == 'slidetags':
        sc_obj = sc_obj.filter_var(
            pl.col('protein_coding') &
            pl.col('mt').not_() & pl.col('ribo').not_())
    return sc_obj\
        .pseudobulk('sample', cell_type_col)\
        .qc('condition',
            min_samples=2,
            min_cells=20,
            max_standard_deviations=None,
            min_nonzero_fraction=0.3,
            verbose=False)\
        .library_size(allow_float=True, num_threads=1)

def populate_r(pb, r_list, cfg):
    r(f'{r_list} <- list()')
    for cell_type, (X, obs, var) in pb.items():
        gene_names = (
            var['_index'] if '_index' in var.columns
            else pl.Series(var.to_pandas().index.tolist()))
        all_genes = gene_names.to_list()
        keep_genes = get_expressed_genes(cell_type, all_genes)
        keep_idx = [i for i, g in enumerate(all_genes) if g in keep_genes]
        if len(keep_idx) == 0:
            continue
        X_filt = X[:, keep_idx]
        gene_names_filt = pl.Series([all_genes[i] for i in keep_idx])

        to_r(obs, 'obs')
        to_r(cell_type, 'cell_type')
        to_r(X_filt, 'X', colnames=gene_names_filt)
        r(f'''
        counts <- t(X)
        element <- list(counts = counts, obs = obs)
        {r_list}[[cell_type]] <- element
        ''')

all_r_lists = {}
for name, cfg in datasets.items():
    adata = adatas[name]
    pb = make_pseudobulk(adata, name)
    for treat, ctrl in cfg['contrasts']:
        contrast = f'{treat}_vs_{ctrl}'
        r_list = f'pb_{name}_{contrast}'
        pb_sub = pb.filter_obs(pl.col('condition').is_in([treat, ctrl]))
        populate_r(pb_sub, r_list, cfg)
        all_r_lists[(name, contrast)] = (r_list, ctrl)
        print(f'[{name}] {contrast}: sent to R')

r('''
suppressPackageStartupMessages({
    library(edgeR)
    library(dplyr)
    library(tibble)
    library(purrr)
})

run_edgeR <- function(pseudobulks, ref_level,
                      design_formula = "~ group + log_num_cells + log_lib_size") {
    imap(pseudobulks, function(element, cell_type_name) {
        tryCatch({
            targets <- element$obs
            all_levels <- unique(as.character(targets$condition))
            other_level <- all_levels[all_levels != ref_level]
            targets$group <- factor(
                targets$condition, levels = c(ref_level, other_level))
            if (n_distinct(targets$group) < 2) return(NULL)

            targets$log_num_cells <- log2(targets$num_cells)
            targets$log_lib_size <- log2(colSums(element$counts))
            design <- model.matrix(
                as.formula(design_formula), data = targets)
            y <- DGEList(counts = element$counts, samples = targets)
            y <- calcNormFactors(y, method = "TMM")

            y <- estimateDisp(y, design, robust = TRUE)
            fit <- glmFit(y, design = design)
            test <- glmLRT(fit, coef = 2)

            tt <- topTags(test, n = Inf) %>%
                as.data.frame() %>%
                rownames_to_column("gene")
            tt
        }, error = function(e) {
            warning(paste("Error in", cell_type_name, ":", e$message))
            return(NULL)
        })
    }) %>%
    bind_rows(.id = "cell_type")
}
''')

de_frames = []
for (name, contrast), (r_list, ref_level) in all_r_lists.items():
    design_formula = datasets[name]['design']
    to_r(ref_level, 'ref_level')
    to_r(design_formula, 'design_formula')
    r(f'de_tmp <- run_edgeR({r_list}, ref_level, design_formula)')
    df = to_py('de_tmp')
    if df is not None and df.height > 0:
        df = df.with_columns(
            pl.lit(contrast).alias('contrast'),
            pl.lit(name).alias('dataset'))
        de_frames.append(df)
        n_sig = df.filter(pl.col('FDR') < 0.10).height
        print(f'[{name}] {contrast}: {df.height:,} tests, '
              f'{n_sig} DEGs (FDR<0.10)')

de_results = pl.concat(de_frames)
de_results = de_results.with_columns(
    pl.struct(['cell_type', 'gene']).map_elements(
        lambda r: get_ref_pct(r['cell_type'], r['gene']),
        return_dtype=pl.Float64
    ).alias('ref_pct_detected')
)

os.makedirs(f'{working_dir}/output', exist_ok=True)
de_results.write_csv(f'{working_dir}/output/de_results.csv')
de_results\
    .filter(pl.col('FDR') < 0.10)\
    .write_csv(f'{working_dir}/output/de_results_sig.csv')

for name in datasets:
    df = de_results.filter(pl.col('dataset') == name)
    for contrast in df['contrast'].unique().to_list():
        sub = df.filter(pl.col('contrast') == contrast)
        n_sig = sub.filter(pl.col('FDR') < 0.10).height
        n_ct = sub['cell_type'].n_unique()
        n_genes = sub['gene'].n_unique()
        print(f'[{name}] {contrast}: {n_ct} cell types, '
              f'{n_genes:,} genes tested, {n_sig} DEGs')

#endregion

#region de barplot #############################################################

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 400

FDR_STRICT = 0.10
FDR_LOOSE = 0.20
MIN_DEGS_TO_SHOW = 10
seismic_cmap = plt.get_cmap('seismic')
UP_COLOR = seismic_cmap(0.9)
DN_COLOR = seismic_cmap(0.1)

de_fig = pl.read_csv(f'{working_dir}/output/de_results.csv')

def get_type(ct):
    if 'Glut' in ct:
        return 'Glut'
    elif any(x in ct for x in ['Gaba', 'IMN', 'Chol']):
        return 'Gaba'
    return 'NN'

contrasts = ['PREG_vs_CTRL', 'POSTPART_vs_PREG', 'POSTPART_vs_CTRL']
contrast_titles = {
    'PREG_vs_CTRL': 'Pregnant vs\nNulliparous',
    'POSTPART_vs_PREG': 'Postpartum vs\nPregnant',
    'POSTPART_vs_CTRL': 'Postpartum vs\nNulliparous',
}

deg_counts = de_fig\
    .filter((pl.col('FDR') < FDR_LOOSE) &
            (pl.col('dataset').is_in(['slidetags', 'xenium'])))\
    .group_by(['cell_type', 'contrast', 'dataset'])\
    .agg(((pl.col('FDR') < FDR_STRICT) &
          (pl.col('logFC') > 0)).sum().alias('up_strict'),
         ((pl.col('FDR') < FDR_STRICT) &
          (pl.col('logFC') < 0)).sum().alias('down_strict'),
         ((pl.col('FDR') < FDR_LOOSE) &
          (pl.col('logFC') > 0)).sum().alias('up_loose'),
         ((pl.col('FDR') < FDR_LOOSE) &
          (pl.col('logFC') < 0)).sum().alias('down_loose'))

ct_totals = deg_counts\
    .group_by('cell_type')\
    .agg((pl.sum('up_strict') + pl.sum('down_strict')).alias('total'))\
    .filter(pl.col('total') >= MIN_DEGS_TO_SHOW)\
    .with_columns(
        pl.col('cell_type').map_elements(
            get_type, return_dtype=pl.Utf8).alias('type'))\
    .with_columns(
        pl.col('type').replace_strict(
            {'Glut': 0, 'Gaba': 1, 'NN': 2}).alias('type_order'))\
    .sort(['type_order', 'total'], descending=[False, True])

groups = ct_totals.group_by('type', maintain_order=True).all()
major_types = groups['type'].to_list()
height_ratios = groups['cell_type'].list.len().to_list()

st_cell_types = set(deg_counts.filter(
    pl.col('dataset') == 'slidetags')['cell_type'].unique().to_list())

BAR_H = 0.42
st_offset = -BAR_H / 2
xn_offset = BAR_H / 2

fig = plt.figure(figsize=(6, 12))
outer_gs = gridspec.GridSpec(
    len(major_types), len(contrasts), figure=fig,
    height_ratios=height_ratios, hspace=0.06, wspace=0.04,
    width_ratios=[2, 1, 1])

for i, group_type in enumerate(major_types):
    group_cts = groups.filter(
        pl.col('type') == group_type)['cell_type'].explode().to_list()

    for j, contrast in enumerate(contrasts):
        ax = fig.add_subplot(outer_gs[i, j])

        show_xn = (contrast == 'PREG_vs_CTRL')

        st_data = deg_counts.filter(
            (pl.col('contrast') == contrast) &
            (pl.col('dataset') == 'slidetags') &
            (pl.col('cell_type').is_in(group_cts)))
        st_dict = {r['cell_type']: r for r in st_data.to_dicts()}

        def draw_stacked(y, row, color_up, color_dn, alpha):
            u_strict = row.get('up_strict', 0)
            d_strict = row.get('down_strict', 0)
            u_loose = row.get('up_loose', 0)
            d_loose = row.get('down_loose', 0)
            u_extra = max(u_loose - u_strict, 0)
            d_extra = max(d_loose - d_strict, 0)

            lw = 0.4
            ax.barh(y, u_strict, height=BAR_H, align='center',
                    facecolor=color_up, edgecolor=color_up,
                    alpha=alpha, linewidth=lw, zorder=5)
            ax.barh(y, -d_strict, height=BAR_H, align='center',
                    facecolor=color_dn, edgecolor=color_dn,
                    alpha=alpha, linewidth=lw, zorder=5)

            if u_extra > 0:
                ax.barh(y, u_extra, left=u_strict, height=BAR_H,
                        align='center',
                        facecolor='none', edgecolor=color_up,
                        linewidth=lw, hatch='////',
                        alpha=alpha, zorder=6)
            if d_extra > 0:
                ax.barh(y, -d_extra, left=-d_strict, height=BAR_H,
                        align='center',
                        facecolor='none', edgecolor=color_dn,
                        linewidth=lw, hatch='////',
                        alpha=alpha, zorder=6)

        if show_xn:
            xn_data = deg_counts.filter(
                (pl.col('contrast') == contrast) &
                (pl.col('dataset') == 'xenium') &
                (pl.col('cell_type').is_in(group_cts)))
            xn_dict = {r['cell_type']: r for r in xn_data.to_dicts()}

            for idx, ct in enumerate(group_cts):
                draw_stacked(idx + st_offset,
                             st_dict.get(ct, {}), UP_COLOR, DN_COLOR, 0.9)
                draw_stacked(idx + xn_offset,
                             xn_dict.get(ct, {}), UP_COLOR, DN_COLOR, 0.45)
        else:
            for idx, ct in enumerate(group_cts):
                draw_stacked(idx, st_dict.get(ct, {}),
                             UP_COLOR, DN_COLOR, 0.9)

        all_up, all_dn = [], []
        for ct in group_cts:
            r_st = st_dict.get(ct, {})
            all_up.append(r_st.get('up_loose', 0))
            all_dn.append(r_st.get('down_loose', 0))
            if show_xn:
                r_xn = xn_dict.get(ct, {})
                all_up.append(r_xn.get('up_loose', 0))
                all_dn.append(r_xn.get('down_loose', 0))
        xlim = max(max(all_up + [1]), max(all_dn + [1])) * 1.25

        ax.axvline(0, color='grey', linewidth=0.5, zorder=0)
        ax.grid(True, 'major', 'y', ls='-', lw=0.3, c='lightgray', zorder=0)
        ax.set_xlim(-xlim, xlim)
        ax.set_yticks(range(len(group_cts)))
        ax.set_ylim(len(group_cts) - 0.5, -0.5)
        ax.tick_params(length=0, labelsize=7)

        if j == 0:
            ax.set_yticklabels(group_cts, fontsize=7.5)
            ax.tick_params(axis='y', pad=8)
            for idx, ct in enumerate(group_cts):
                if ct not in st_cell_types:
                    ax.plot(-0.01, idx, 's', color='#555555', markersize=3,
                            transform=ax.get_yaxis_transform(), clip_on=False,
                            zorder=20)
        else:
            ax.set_yticklabels([])

        if i == 0:
            ax.set_title(contrast_titles[contrast], fontsize=9, pad=6)

        if i == len(major_types) - 1:
            ax.set_xlabel('DEGs', fontsize=8.5)
        else:
            ax.set_xticklabels([])

legend_elements = [
    Patch(facecolor=UP_COLOR, edgecolor=UP_COLOR, alpha=0.9, linewidth=0.4,
          label='Upregulated'),
    Patch(facecolor=DN_COLOR, edgecolor=DN_COLOR, alpha=0.9, linewidth=0.4,
          label='Downregulated'),
    Patch(facecolor='lightgray', edgecolor='black', linewidth=0.4,
          label='Slide-tags'),
    Patch(facecolor='lightgray', edgecolor='black', alpha=0.45,
          linewidth=0.4, label='Xenium'),
    Patch(facecolor='lightgray', edgecolor='black', linewidth=0.4,
          label='FDR<0.10'),
    Patch(facecolor='none', edgecolor='black', linewidth=0.4,
          hatch='////', label='FDR<0.20'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#555555',
           markersize=5, label='Xenium only'),
]
fig.legend(handles=legend_elements, loc='lower right',
           bbox_to_anchor=(0.98, 0.02), fontsize=7,
           frameon=False, ncol=4)

os.makedirs(f'{working_dir}/figures', exist_ok=True)
plt.savefig(f'{working_dir}/figures/deg_barplot.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/deg_barplot.svg',
            bbox_inches='tight')
plt.close()

#endregion

#region DEG exemplar violins ###################################################

EXEMPLAR_GENES = [
    ('Grm1', '058 PAL-STR Gaba-Chol'),
    ('Slc17a8', '058 PAL-STR Gaba-Chol'),
    ('Gpr88', '009 L2/3 IT PIR-ENTl Glut'),
    ('Dusp7', '006 L4/5 IT CTX Glut'),
    ('Kdr', '333 Endo NN'),
    ('Igfbp3', '333 Endo NN'),
    ('Idi1', '319 Astro-TE NN'),
    ('Gjb6', '319 Astro-TE NN'),
    ('Grm8', '085 SI-MPO-LPO Lhx8 Gaba'),
    ('Calcr', '085 SI-MPO-LPO Lhx8 Gaba'),
    ('Gal', '086 MPO-ADP Lhx8 Gaba'),
    ('Brs3', '106 PVpo-VMPO-MPN Hmx2 Gaba'),
    ('Rxfp1', '124 MPN-MPO-PVpo Hmx2 Glut'),
]

condition_colors = {
    'CTRL': '#7209b7',
    'PREG': '#b5179e',
    'POSTPART': '#f72585',
}

def get_pseudobulk_expr(adata_src, gene, cell_type):
    if gene not in adata_src.var_names:
        return {}
    cell_mask = adata_src.obs[cell_type_col] == cell_type
    if cell_mask.sum() == 0:
        return {}
    subset = adata_src[cell_mask, gene]
    groups = subset.obs.groupby('sample')['condition'].first()
    result = {}
    for sample, cond in groups.items():
        s_mask = subset.obs['sample'] == sample
        total = subset[s_mask].X.sum()
        n_cells = s_mask.sum()
        result[sample] = {
            'cond': cond,
            'cpm': np.log2(total / n_cells * 1e4 + 1) if n_cells > 0 else 0,
        }
    return result

def zscore_pb(pb):
    if not pb:
        return
    vals = [v['cpm'] for v in pb.values()]
    mu = np.mean(vals)
    sd = max(np.std(vals), 1e-6)
    for v in pb.values():
        v['z'] = (v['cpm'] - mu) / sd

def draw_panel(ax, pb, conditions, marker, ms, alpha, seed):
    cond_pos = {c: i for i, c in enumerate(conditions)}
    rng = np.random.default_rng(seed)
    for vals in pb.values():
        cond = vals['cond']
        if cond not in cond_pos:
            continue
        x = cond_pos[cond] + rng.uniform(-0.18, 0.18)
        ax.scatter(x, vals['z'], marker=marker,
                   c=condition_colors[cond], s=ms, alpha=alpha,
                   linewidths=0, zorder=10)
    means = {}
    for cond in conditions:
        zvals = [v['z'] for v in pb.values() if v['cond'] == cond]
        if zvals:
            m = np.mean(zvals)
            se = np.std(zvals) / np.sqrt(len(zvals)) \
                if len(zvals) > 1 else 0
            means[cond] = m
            ax.errorbar(cond_pos[cond], m, yerr=se,
                        fmt='none', color='black', capsize=1.5,
                        capthick=0.5, linewidth=0.5, zorder=5,
                        alpha=alpha)
    if len(means) > 1:
        xs = [cond_pos[c] for c in conditions if c in means]
        ys = [means[c] for c in conditions if c in means]
        ax.plot(xs, ys, '-', color='black', linewidth=0.6,
                alpha=0.3, zorder=1)

def add_sig_star(ax, gene, cell_type, dataset, cond_pos):
    is_sig = de_fig.filter(
        (pl.col('gene') == gene) &
        (pl.col('cell_type') == cell_type) &
        (pl.col('dataset') == dataset) &
        (pl.col('contrast') == 'PREG_vs_CTRL') &
        (pl.col('FDR') < 0.10)).height > 0
    if not is_sig or 'CTRL' not in cond_pos or 'PREG' not in cond_pos:
        return
    x = (cond_pos['CTRL'] + cond_pos['PREG']) / 2
    ylim = ax.get_ylim()
    ax.text(x, ylim[1] - (ylim[1] - ylim[0]) * 0.02, '*',
            ha='center', va='top', fontsize=9, fontweight='bold',
            clip_on=False)

def format_ax(ax, linewidth=0.5):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)

adata_xn_norm = adatas['xenium'].copy()
sc.pp.normalize_total(adata_xn_norm, target_sum=1e4)
sc.pp.log1p(adata_xn_norm)

xn_ct_subsets = {}
for ct in set(ct for _, ct in EXEMPLAR_GENES):
    mask = adata_xn_norm.obs[cell_type_col] == ct
    if mask.sum() > 0:
        xn_ct_subsets[ct] = adata_xn_norm[mask]

all_ffd = adata_xn_norm.obs[['x_ffd', 'y_ffd']].values
fov_cx = (all_ffd[:, 0].min() + all_ffd[:, 0].max()) / 2
fov_cy = (all_ffd[:, 1].min() + all_ffd[:, 1].max()) / 2
fov_half = max(np.ptp(all_ffd[:, 0]), np.ptp(all_ffd[:, 1])) / 2 * 1.05

n_genes = len(EXEMPLAR_GENES)
n_col_genes = 2
n_rows = (n_genes + n_col_genes - 1) // n_col_genes

fig = plt.figure(figsize=(n_col_genes * 4.0, n_rows * 2.0))
outer_gs = gridspec.GridSpec(n_rows, n_col_genes, figure=fig,
                              hspace=0.5, wspace=0.25)

for idx, (gene, cell_type) in enumerate(EXEMPLAR_GENES):
    row_i = idx // n_col_genes
    col_i = idx % n_col_genes
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer_gs[row_i, col_i],
        hspace=0.05, wspace=0.04,
        width_ratios=[1.0, 0.9, 0.9])
    ax_st = fig.add_subplot(inner_gs[0, 0])
    ax_xn = fig.add_subplot(inner_gs[1, 0], sharex=ax_st)
    ax_sp_ctrl = fig.add_subplot(inner_gs[:, 1])
    ax_sp_preg = fig.add_subplot(inner_gs[:, 2])

    pb_st = get_pseudobulk_expr(adatas['slidetags'], gene, cell_type)
    pb_xn = get_pseudobulk_expr(adatas['xenium'], gene, cell_type)

    if not pb_st and not pb_xn:
        for a in [ax_st, ax_xn, ax_sp_ctrl, ax_sp_preg]:
            a.set_visible(False)
        continue

    has_postpart = any(v['cond'] == 'POSTPART' for v in pb_st.values())
    st_conditions = ['CTRL', 'PREG', 'POSTPART'] if has_postpart \
        else ['CTRL', 'PREG']
    xn_conditions = ['CTRL', 'PREG']

    zscore_pb(pb_st)
    zscore_pb(pb_xn)

    draw_panel(ax_st, pb_st, st_conditions, 'o', 12, 1.0, idx)
    draw_panel(ax_xn, pb_xn, xn_conditions, 'D', 9, 0.65, idx + 1000)

    for ax in [ax_st, ax_xn]:
        yl = ax.get_ylim()
        pad = (yl[1] - yl[0]) * 0.08
        ax.set_ylim(yl[0] - pad, yl[1] + pad)

    st_cond_pos = {c: i for i, c in enumerate(st_conditions)}
    xn_cond_pos = {c: i for i, c in enumerate(xn_conditions)}
    add_sig_star(ax_st, gene, cell_type, 'slidetags', st_cond_pos)
    add_sig_star(ax_xn, gene, cell_type, 'xenium', xn_cond_pos)

    n_x = max(len(st_conditions), len(xn_conditions))
    ax_st.set_xlim(-0.5, n_x - 0.5)

    ct_label = re.sub(r'^\d+\s+', '', cell_type)
    parent = fig.add_subplot(outer_gs[row_i, col_i])
    parent.axis('off')
    parent.patch.set_alpha(0)
    parent.set_title(f'{gene}\n{ct_label}', fontsize=6.5, pad=3)

    ax_st.set_xticks(list(range(n_x)))
    ax_st.set_xticklabels(['N', 'P', 'PP'][:n_x], fontsize=5)
    ax_st.tick_params(axis='x', length=1.5)
    ax_xn.set_xticks(list(range(n_x)))
    ax_xn.set_xticklabels(['N', 'P', 'PP'][:n_x], fontsize=5)
    ax_xn.tick_params(axis='x', length=1.5)

    for ax in [ax_st, ax_xn]:
        ax.tick_params(axis='y', labelsize=5.5, length=1.5)
        format_ax(ax)

    subset = xn_ct_subsets.get(cell_type)
    if subset is not None and gene in subset.var_names:
        expr = np.asarray(subset[:, gene].X.toarray()).ravel()
        coords_x = subset.obs['x_ffd'].values
        coords_y = subset.obs['y_ffd'].values
        conds = subset.obs['condition'].values

        nonzero = expr[expr > 0]
        vmin = np.quantile(nonzero, 0.05) if len(nonzero) > 0 else 0
        vmax = np.quantile(nonzero, 0.95) if len(nonzero) > 0 else 1
        if vmax <= vmin:
            vmax = vmin + 1

        for ax_sp, cond, sp_label in [
                (ax_sp_ctrl, 'CTRL', 'Nulliparous'),
                (ax_sp_preg, 'PREG', 'Pregnant')]:
            c_mask = conds == cond
            if c_mask.sum() == 0:
                ax_sp.set_visible(False)
                continue
            order = np.argsort(expr[c_mask])
            ax_sp.scatter(
                coords_x[c_mask][order], coords_y[c_mask][order],
                c=expr[c_mask][order], cmap='viridis', s=0.2,
                vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
            ax_sp.set_xlim(fov_cx - fov_half, fov_cx + fov_half)
            ax_sp.set_ylim(fov_cy - fov_half, fov_cy + fov_half)
            ax_sp.set_xticks([])
            ax_sp.set_yticks([])
            ax_sp.set_facecolor('black')
            ax_sp.set_xlabel(sp_label, fontsize=5, labelpad=2)
            format_ax(ax_sp)
    else:
        ax_sp_ctrl.set_visible(False)
        ax_sp_preg.set_visible(False)

fig.text(0.04, 0.5, 'Expression (z-score)',
         va='center', ha='center', rotation='vertical', fontsize=7)

ax_legend = fig.add_subplot(outer_gs[n_rows - 1, n_col_genes - 1])
ax_legend.axis('off')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markersize=4, markeredgecolor='none', label='Slide-tags'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gray',
           markersize=3.5, markeredgecolor='none',
           label='Xenium', alpha=0.65),
]
ax_legend.legend(handles=legend_elements, loc='upper left',
                 fontsize=6, frameon=False)

cbar_ax = ax_legend.inset_axes([0.05, 0.25, 0.35, 0.06])
sm = ScalarMappable(cmap='viridis', norm=Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['low', 'high'], fontsize=5)
cbar.ax.set_xlabel('log₁p expression', fontsize=5, labelpad=1)
cbar.outline.set_linewidth(0.4)

plt.savefig(f'{working_dir}/figures/deg_exemplar_pseudobulk.png',
            dpi=300, bbox_inches='tight')
plt.savefig(f'{working_dir}/figures/deg_exemplar_pseudobulk.svg',
            bbox_inches='tight')
plt.close()
del adata_xn_norm, xn_ct_subsets

#endregion
