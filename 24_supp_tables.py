"""Supplementary Data workbooks.

One entry point per numbered Supplementary Data file, so the workbooks are
regenerated from the pipeline rather than hand-edited.

Supplementary Data 1 - samples and gene panels
    Slide-tags Samples   per sample and library, with the raw FASTQ manifest
    MERFISH Samples      per sample
    Xenium Samples       per sample, with capture regions and what was excluded
    MERFISH Gene Panel   496 genes of the custom panel
    Xenium Gene Panel    5,006 genes of the Xenium Prime 5K pan-tissue panel
    Figure 1C Marker Expression
                         source data for the marker dot plot: 45 markers x 84
                         subclasses x 3 platforms

Supplementary Data 2 - cell-type composition and local proximity
    Description          sheet contents and column definitions
    Composition          crumblr + dream differential abundance, per platform
    Composition Cross-Platform
                         the same tests side by side, with direction agreement
    Local Proximity      limma test on neighborhood enrichment z-scores, per
                         platform, for every center-surround subclass pair
    Local Proximity Meta MERFISH + Xenium sum-of-ranks meta-analysis of the
                         pregnant-versus-nulliparous contrast
    Sample Composition   source data: cells, proportion and CLR abundance per
                         sample and subclass

Supplementary Data 3 - differential expression
    Description          sheet contents and column definitions
    Differential Expression
                         voomByGroup and limma per platform and contrast
    Cross-Platform Meta-analysis
                         sum-of-ranks across platforms, permutation calibrated
    Genes Changed per Cell Type
                         source data for Figure 2A

Supplementary Data 4 - pathway enrichment
    Description          sheet contents and column definitions
    Pathway Enrichment   fgsea per platform and contrast
    Cross-Platform Meta-analysis
                         the same meta-analysis at the pathway level
    Pathway Themes       the keyword rules that grouped the GO terms

The per-platform sheets of Data 3 and 4 carry the rows behind a meta-analysis
result, which keeps each workbook under the 30 MB the journal allows per
supplementary file. write_deposit() writes the unrestricted tables to
output/supplementary/deposit for a repository, cited from Data availability.

Sample provenance (internal IDs, tile IDs, strain, stage) is fixed
experimental fact and is held in this file. Everything derivable - post-QC cell
counts, capture regions, which samples survived - is read back from the
processed objects, so the table cannot drift from the analysis. The Slide-tags
FASTQ manifest is read from the original hand-built workbook (LEGACY_XLSX)
rather than transcribed; if that file is absent the column is left empty and a
warning is printed.

Reads  output/{slidetags,merfish,xenium}/03_adata_query_*.h5ad
       input/XeniumPrimeMouse5Kpan_tissue_pathways_metadata.csv
       input/markers.csv, input/m_df_themed_v2.rds (gene rationale)
       output/proximity/{global_props,global_norm_props,local_diff,
       local_sumrank}.csv (written by 07_prop_prox.py)
       LEGACY_XLSX (Slide-tags FASTQ manifest, MERFISH panel Ensembl IDs)
Writes output/supplementary/Supplementary_Data_<n>.xlsx
"""

#region imports and config #####################################################

import os
import re
import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
import scanpy as sc
import xlsxwriter

working_dir = '/home/karbabi/spatial-pregnancy'
OUT = f'{working_dir}/output/supplementary'
PROX = f'{working_dir}/output/proximity'
XENIUM_PANEL_CSV = (f'{working_dir}/input/'
                    'XeniumPrimeMouse5Kpan_tissue_pathways_metadata.csv')
THEMED_GO_RDS = f'{working_dir}/input/m_df_themed_v2.rds'
LEGACY_XLSX = '/home/karbabi/Supplementary_Dataset_1.xlsx'
os.makedirs(OUT, exist_ok=True)

MARKERS_CSV = f'{working_dir}/input/markers.csv'
MARKER_FC = 5.0           # min fold change for a reference marker call
PLATFORMS = ['slidetags', 'merfish', 'xenium']
PLATFORM_LABELS = {'slidetags': 'Slide-tags', 'merfish': 'MERFISH',
                   'xenium': 'Xenium'}
MIN_CELLS, MIN_PLATFORMS = 100, 2   # the 84-subclass study set

STRAIN = 'C57BL/6J'
CONDITION = {'CTRL': 'Nulliparous', 'PREG': 'Pregnant',
             'POSTPART': 'Postpartum'}
STAGE = {'CTRL': '8 wks', 'PREG': 'gestational day 18',
         'POSTPART': 'day 20, pre-weaning'}

# the three contrasts, in the order they are reported throughout
CONTRASTS = ['PREG_vs_CTRL', 'POSTPART_vs_PREG', 'POSTPART_vs_CTRL']
CONTRAST_LABELS = {
    'PREG_vs_CTRL': 'Pregnant vs Nulliparous',
    'POSTPART_vs_PREG': 'Postpartum vs Pregnant',
    'POSTPART_vs_CTRL': 'Postpartum vs Nulliparous',
}
PROX_FDR = 0.10          # significance threshold of 07_prop_prox.py
SIG_COL = 'Significant (FDR ≤ 0.10)'

DE_DIR = f'{working_dir}/output/de'
GSEA_DIR = f'{working_dir}/output/gsea'
DE_GSEA_SCRIPT = f'{working_dir}/06_de_gsea.py'
DE_FDR = 0.10            # per-platform threshold of 06_de_gsea.py
EMP_P = 0.05             # empirical threshold of figures 2-5

# Slide-tags provenance: sample -> (internal ID, tile ID). Fixed experimental
# fact, not derivable from the processed objects.
SLIDETAGS_PROV = {
    'CTRL_1':     ('Virg1', 'C0005_013'),
    'CTRL_2':     ('Virg2', 'C0003_004'),
    'CTRL_3':     ('Virg3', 'U0009_014'),
    'PREG_1':     ('Preg1', 'C0003_003'),
    'PREG_2':     ('Preg2', 'C0005_014'),
    'PREG_3':     ('Preg3', 'U0009_016'),
    'POSTPART_1': ('PP1',   'U0009_013'),
    'POSTPART_2': ('PP2',   'U0009_015'),
}
SLIDETAGS_LIBRARIES = ['cDNA', 'Surface protein']

# samples that were collected but do not reach the analysis, and why
EXCLUSIONS = {
    ('slidetags', 'POSTPART_3'):
        'Brain unavailable for tissue processing; not profiled',
    ('xenium', 'PREG_1'):
        'Both capture regions excluded at quality control',
    ('xenium', 'CTRL_3'):
        'Annotated but excluded from all downstream analyses. In both '
        'capture regions non-neuronal cells recovered about half the '
        'transcripts per cell of the other samples (median 465 against '
        '928-1,047), while neuronal cells were unaffected and segmented '
        'cell area was normal. The sample was the most discordant of the '
        'five in pseudobulk expression.',
}

# Established cell-type markers carried by the panels, from the Figure 1
# dotplot. These are the "established marker genes to identify major brain cell
# types" of the Methods; every other panel gene is a pathway selection.
CURATED_MARKERS = {
    'Pallium glutamatergic': ['Slc17a7', 'Cux2', 'Rorb', 'Tle4', 'Foxp2',
                              'Fezf2'],
    'Pallium GABAergic': ['Gad2', 'Vip', 'Sncg', 'Lamp5', 'Pvalb', 'Lhx6',
                          'Sst'],
    'Subpallium GABAergic': ['Dlx2', 'Cyp26b1', 'Tac1', 'Ppp1r1b', 'Drd1',
                             'Drd2', 'Pax6', 'Nr3c2'],
    'Hypothalamus-extended amygdala': ['Bnc2', 'Pdyn', 'Lhx8', 'Tac2', 'Six3',
                                       'Gal', 'Ebf3', 'Barhl2', 'Bsx', 'Esr1',
                                       'Otp', 'Trh'],
    'Non-neuronal': ['Lrig1', 'Aqp4', 'Foxj1', 'Prlr', 'Pdgfra', 'Mog',
                     'Col1a1', 'Pdgfrb', 'Myh11', 'Flt1', 'Cx3cr1', 'Mrc1'],
}
MARKER_GROUP = {g: grp for grp, gs in CURATED_MARKERS.items() for g in gs}
SHARED_MARKERS = [g for gs in CURATED_MARKERS.values() for g in gs]

# the five cellular neighborhoods, as Allen class numeric prefixes; the row
# blocks of figures 1 and 2. Keys match CURATED_MARKERS so a gene and a
# subclass are labeled with the same vocabulary.
NEIGHBORHOODS = {
    'Pallium glutamatergic': ['01', '02', '03', '04'],
    'Pallium GABAergic': ['06', '07'],
    'Subpallium GABAergic': ['05', '08', '09', '10'],
    'Hypothalamus-extended amygdala': ['11', '12', '13', '14', '15', '18',
                                       '19', '20', '24'],
    'Non-neuronal': ['30', '31', '33', '34'],
}

#endregion

#region gene selection rationale ###############################################
# Both panels are annotated the same way, so the two sheets are directly
# comparable. A gene counts as a cell-type marker if it discriminates an Allen
# subclass in the reference marker table (MARKERS_CSV, fold change >=
# MARKER_FC, restricted to the 84 subclasses analyzed here) or appears in the
# Figure 1 marker set; its pathway rationale is
# membership in the 14 themed GO biological-process collections used for the
# enrichment analysis. Neither column is a transcription of a design record -
# the MERFISH panel was designed in-house and no per-gene record is held here,
# and the Xenium panel is an off-the-shelf product.


@lru_cache(maxsize=1)
def themed_go_map():
    """gene symbol -> sorted themes, from the curated GO BP collection."""
    os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
    from ryp import r, to_py
    r(f'.m <- unique(readRDS("{THEMED_GO_RDS}")[, c("gene_symbol", "theme")])')
    df = to_py('.m', format='pandas')
    return (df.groupby('gene_symbol')['theme']
            .agg(lambda s: '; '.join(sorted(set(s)))).to_dict())


@lru_cache(maxsize=1)
def study_subclasses():
    """The 84 subclasses the analyses run on: >= MIN_CELLS cells in at least
    MIN_PLATFORMS of the three platforms, the same rule as figures 1 and 2."""
    n_pass = {}
    for name in PLATFORMS:
        a = sc.read_h5ad(
            f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad',
            backed='r')
        vc = a.obs['subclass'].astype(str).value_counts()
        a.file.close()
        for s in vc.index[vc >= MIN_CELLS]:
            n_pass[s] = n_pass.get(s, 0) + 1
    return frozenset(s for s, k in n_pass.items() if k >= MIN_PLATFORMS)


@lru_cache(maxsize=1)
def reference_markers():
    """gene symbol -> every study subclass it discriminates, strongest first.

    Restricted to the subclasses actually analyzed here, so the table never
    cites a subclass outside the sampled coronal plane. Lists are complete:
    a gene that marks 40 of the 84 gets all 40.
    """
    keep = study_subclasses()
    m = pd.read_csv(MARKERS_CSV, usecols=['cell_type', 'gene', 'fold_change'])
    m = m[(m.fold_change >= MARKER_FC) & m.cell_type.isin(keep)]
    m = m.sort_values('fold_change', ascending=False)
    return m.groupby('gene')['cell_type'].apply(list).to_dict()


@lru_cache(maxsize=1)
def subclass_neighborhood():
    """subclass -> cellular neighborhood, via its Allen class prefix.

    Pooled over the three platforms: the proximity tables carry subclasses that
    fall below the study-set threshold and are not annotated on every platform.
    The class of a subclass is a property of the Allen taxonomy, so the three
    platforms never disagree.
    """
    pairs = []
    for name in PLATFORMS:
        a = sc.read_h5ad(
            f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad',
            backed='r')
        pairs.append(
            a.obs[['class', 'subclass']].astype(str).drop_duplicates())
        a.file.close()
    pairs = pd.concat(pairs).drop_duplicates('subclass')
    of_prefix = {p: n for n, ps in NEIGHBORHOODS.items() for p in ps}
    return {r.subclass: of_prefix.get(r['class'].split(' ', 1)[0], '')
            for _, r in pairs.iterrows()}


def figure_1_row_order():
    """The 84 subclasses in the Figure 1C row order: by neighborhood, then by
    numeric prefix within it."""
    keep, hood = study_subclasses(), subclass_neighborhood()
    order = []
    for name in NEIGHBORHOODS:
        order += sorted((s for s in keep if hood.get(s) == name),
                        key=lambda s: int(s.split(' ', 1)[0]))
    return order


def gene_matrix(name, genes, chunk=50_000):
    """Dense cells x genes for a handful of genes, plus the subclass of each
    cell.

    `adata.to_memory()[:, genes]` would also work, but it materialises X plus
    both count layers; Xenium's X alone is 461M non-zeros (~3.7 GB). Streaming
    the CSR in row blocks and keeping only the wanted columns bounds peak
    memory to one block.
    """
    import h5py
    path = f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad'
    a = sc.read_h5ad(path, backed='r')
    names = list(a.var_names.astype(str))
    subclass = a.obs['subclass'].astype(str).to_numpy()
    a.file.close()

    at = {g: i for i, g in enumerate(names)}
    present = [g for g in genes if g in at]
    want = np.full(len(names), -1)
    want[[at[g] for g in present]] = np.arange(len(present))

    with h5py.File(path, 'r') as f:
        X = f['X']
        indptr = X['indptr'][:]
        n = len(indptr) - 1
        out = np.zeros((n, len(present)), dtype=np.float32)
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            lo, hi = indptr[i0], indptr[i1]
            if hi == lo:
                continue
            cols = want[X['indices'][lo:hi]]
            keep = cols >= 0
            rows = np.repeat(np.arange(i0, i1), np.diff(indptr[i0:i1 + 1]))
            out[rows[keep], cols[keep]] = X['data'][lo:hi][keep]
    return present, out, subclass


def marker_expression():
    """Source data for the Figure 1C dot plot: per subclass, gene and platform,
    the z-scored mean expression and the fraction of cells expressing.

    Reproduces the computation of compute_dotplot() in 10_figure_1.py - the
    mean is
    z-scored per gene across the 84 subclasses, so a value is a gene's relative
    enrichment in that subclass, not an absolute level. Genes outside a
    platform's panel are kept as rows with Measured = No, which is the grey
    wedge in the figure.
    """
    order = figure_1_row_order()
    hood = subclass_neighborhood()
    rows = []
    for name in PLATFORMS:
        present, X, subc = gene_matrix(name, SHARED_MARKERS)
        df = pd.DataFrame(X, columns=present)
        mean = df.groupby(subc)[present].mean().reindex(order)
        frac = (df > 0).astype(float).groupby(subc)[present].mean()\
            .reindex(order)
        M, F = mean.to_numpy().T, frac.to_numpy().T
        mu = np.nanmean(M, axis=1, keepdims=True)
        sd = np.nanstd(M, axis=1, keepdims=True)
        sd[sd == 0] = 1
        Z = (M - mu) / sd
        counts = pd.Series(subc).value_counts().reindex(order, fill_value=0)
        at = {g: k for k, g in enumerate(present)}
        for ci, s in enumerate(order):
            for g in SHARED_MARKERS:
                k = at.get(g)
                rows.append({
                    'Cell Type': s,
                    'Cell Type Neighborhood': hood.get(s, ''),
                    'Gene Symbol': g,
                    'Marker Gene Neighborhood': MARKER_GROUP[g],
                    'Platform': PLATFORM_LABELS[name],
                    'Cells': int(counts[s]),
                    'Measured': 'Yes' if k is not None else 'No',
                    'Mean Expression (z-scored)':
                        Z[k, ci] if k is not None else np.nan,
                    'Fraction Expressing (%)':
                        F[k, ci] * 100 if k is not None else np.nan})
    df = pd.DataFrame(rows)
    df['Cell Type'] = pd.Categorical(df['Cell Type'], order, ordered=True)
    df['Gene Symbol'] = pd.Categorical(df['Gene Symbol'], SHARED_MARKERS,
                                       ordered=True)
    df['Platform'] = pd.Categorical(
        df['Platform'], [PLATFORM_LABELS[p] for p in PLATFORMS], ordered=True)
    return (df.sort_values(['Cell Type', 'Gene Symbol', 'Platform'])
            .reset_index(drop=True))


def annotate_panel(genes):
    """Selection rationale, marker cell types and pathway themes per gene."""
    themes = themed_go_map()
    markers = reference_markers()
    rows = []
    for g in genes:
        marker = []
        if g in MARKER_GROUP:
            marker.append(f'{MARKER_GROUP[g]} (Figure 1 marker set)')
        if g in markers:
            marker.append('; '.join(markers[g]))
        pathway = themes.get(g, '')
        why = '; '.join((['Cell-type marker'] if marker else [])
                        + (['pathway'] if pathway else [])) or 'Not annotated'
        rows.append((why[0].upper() + why[1:], ' | '.join(marker), pathway))
    return pd.DataFrame(
        rows, columns=['Selection Rationale', 'Marker For', 'Pathway Themes'])

#endregion

#region helpers ################################################################


def platform_counts(name):
    """Post-QC cells per sample, and per capture region where recorded."""
    a = sc.read_h5ad(f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad',
                     backed='r')
    obs = a.obs[['sample'] + (['sample_rep'] if 'sample_rep' in a.obs else [])]
    obs = obs.astype(str).copy()
    a.file.close()
    cells = obs.groupby('sample').size()
    reps = (obs.groupby('sample')['sample_rep'].nunique()
            if 'sample_rep' in obs else None)
    return cells, reps


def sample_sort_key(s):
    order = {'CTRL': 0, 'PREG': 1, 'POSTPART': 2}
    stem, _, num = s.rpartition('_')
    return order.get(stem, 9), int(num) if num.isdigit() else 0


def subclass_sort_key(s):
    """Allen numeric prefix, so '032 ...' sorts before '318 ...'."""
    p = s.split(' ', 1)[0]
    return (int(p) if p.isdigit() else 10_000, s)


def round_sig(df, digits=4):
    """Round every float column to `digits` significant figures, in place.

    Keeps small P values readable (3.2e-05 stays 3.2e-05) where a fixed number
    of decimals would flatten them to zero.
    """
    for c in df.select_dtypes('float').columns:
        df[c] = df[c].map(
            lambda v: v if pd.isna(v) else float(f'{v:.{digits}g}'))
    return df


def yesno(mask):
    return np.where(mask, 'Yes', 'No')


def subclass_counts(name):
    """Cells per sample and subclass, for one platform."""
    a = sc.read_h5ad(f'{working_dir}/output/{name}/03_adata_query_{name}.h5ad',
                     backed='r')
    obs = a.obs[['sample', 'subclass']].astype(str)
    a.file.close()
    return (obs.value_counts().rename('Cells').reset_index()
            .rename(columns={'sample': 'Sample ID', 'subclass': 'Cell Type'}))


def base_rows(samples, cells):
    """The four columns every sample sheet shares."""
    return pd.DataFrame({
        'Sample ID': samples,
        'Condition': [CONDITION[s.rpartition('_')[0]] for s in samples],
        'Stage': [STAGE[s.rpartition('_')[0]] for s in samples],
        'Mouse Strain': STRAIN,
        'Cells (post-QC)': [cells.get(s, pd.NA) for s in samples],
    })


def legacy_sheet(name):
    if not os.path.exists(LEGACY_XLSX):
        return None
    try:
        df = pd.read_excel(LEGACY_XLSX, sheet_name=name)
    except ValueError:
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df


def slidetags_raw_files():
    """(sample, library) -> FASTQ manifest, from the original workbook.

    In that file some manifests overflowed into an unnamed ninth column; the
    two are concatenated back into one cell here.
    """
    df = legacy_sheet('Slide-tags Samples')
    if df is None:
        print(f'  ! {LEGACY_XLSX} not found - Raw Files left empty')
        return {}
    extra = [c for c in df.columns if c.startswith('Unnamed')]
    out = {}
    for _, r in df.dropna(subset=['Sample ID']).iterrows():
        parts = [str(r['Raw Files'])] if pd.notna(r.get('Raw Files')) else []
        parts += [str(r[c]) for c in extra if pd.notna(r.get(c))]
        out[(r['Sample ID'], r['Library'])] = ', '.join(parts)
    return out


def write_workbook(path, sheets, wide_cols=(), width_sample=50_000):
    """Write sheets with a bold header row, autofilter and frozen header.

    xlsxwriter runs in constant-memory mode, flushing each row to disk as it
    is written: the differential-expression sheets reach 549,000 rows, which
    in the default mode would hold tens of millions of cell objects in RAM.
    That mode requires the header and the column widths to be set before any
    data and the rows to be written in order, so this does not use
    DataFrame.to_excel. Widths are measured on the first `width_sample` rows.
    """
    with xlsxwriter.Workbook(path, {'constant_memory': True}) as book:
        hdr = book.add_format({'bold': True, 'valign': 'top',
                               'text_wrap': True, 'bottom': 1,
                               'font_size': 10})
        wrap = book.add_format({'valign': 'top', 'text_wrap': True,
                                'font_size': 10})
        plain = book.add_format({'valign': 'top', 'font_size': 10})
        for name, df in sheets:
            ws = book.add_worksheet(name)
            for j, c in enumerate(df.columns):
                body = df[c].head(width_sample).astype(str)
                width = max(len(str(c)) * 1.05,
                            (body.str.len().max() or 0) * 1.05 + 1)
                ws.set_column(j, j, min(max(width, 10), 60),
                              wrap if c in wide_cols else plain)
            ws.write_row(0, 0, list(df.columns), hdr)
            # NaN and NA become None so xlsxwriter writes an empty cell
            for i, row in enumerate(
                    df.astype(object).where(df.notna(), None)
                    .itertuples(index=False, name=None), start=1):
                ws.write_row(i, 0, row)
            ws.freeze_panes(1, 1)
            ws.autofilter(0, 0, len(df), len(df.columns) - 1)
    print(f'wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)')
    for name, df in sheets:
        print(f'  {name:32s} {df.shape[0]:9,d} rows x {df.shape[1]:2d} cols')

#endregion

#region Supplementary Data 1 ###################################################


def build_dataset_1():
    sheets = []

    # --- Slide-tags: one row per sample and library -------------------------
    cells, _ = platform_counts('slidetags')
    raw = slidetags_raw_files()
    rows = []
    for s in sorted(SLIDETAGS_PROV, key=sample_sort_key):
        internal, tile = SLIDETAGS_PROV[s]
        for lib in SLIDETAGS_LIBRARIES:
            rows.append(dict(
                **base_rows([s], cells).iloc[0].to_dict(),
                **{'Internal Sample ID': internal, 'Tile ID': tile,
                   'Library': lib, 'Raw Files': raw.get((s, lib), ''),
                   'Notes': ''}))
    for (plat, s), note in EXCLUSIONS.items():
        if plat == 'slidetags':
            rows.append({'Sample ID': s,
                         'Condition': CONDITION[s.rpartition('_')[0]],
                         'Stage': STAGE[s.rpartition('_')[0]],
                         'Mouse Strain': STRAIN, 'Cells (post-QC)': pd.NA,
                         'Internal Sample ID': '', 'Tile ID': '',
                         'Library': '', 'Raw Files': '', 'Notes': note})
    sheets.append(('Slide-tags Samples', pd.DataFrame(rows)))

    # --- MERFISH ------------------------------------------------------------
    cells, _ = platform_counts('merfish')
    samples = sorted(cells.index, key=sample_sort_key)
    df = base_rows(samples, cells)
    df['Notes'] = ''
    sheets.append(('MERFISH Samples', df))

    # --- Xenium -------------------------------------------------------------
    cells, reps = platform_counts('xenium')
    samples = sorted(cells.index, key=sample_sort_key)
    df = base_rows(samples, cells)
    df.insert(4, 'Capture Regions', [int(reps.get(s, 0)) for s in samples])
    df['Notes'] = [EXCLUSIONS.get(('xenium', s), '') for s in samples]
    for s, note in ((s, n) for (p, s), n in EXCLUSIONS.items()
                    if p == 'xenium' and s not in set(samples)):
        df.loc[len(df)] = {
            'Sample ID': s, 'Condition': CONDITION[s.rpartition('_')[0]],
            'Stage': STAGE[s.rpartition('_')[0]], 'Mouse Strain': STRAIN,
            'Capture Regions': 0, 'Cells (post-QC)': pd.NA, 'Notes': note}
    df = df.sort_values('Sample ID', key=lambda c: c.map(sample_sort_key))
    sheets.append(('Xenium Samples', df.reset_index(drop=True)))

    # --- MERFISH gene panel -------------------------------------------------
    panel = legacy_sheet('MERFISH Gene Panel')
    if panel is None:
        print(f'  ! {LEGACY_XLSX} not found - MERFISH panel taken from the '
              'processed object, without Ensembl IDs')
        a = sc.read_h5ad(f'{working_dir}/output/merfish/'
                         '03_adata_query_merfish.h5ad', backed='r')
        panel = pd.DataFrame({'Gene Symbol': sorted(a.var_names),
                              'Gene ID': ''})
        a.file.close()
    panel = (panel[['Gene Symbol', 'Gene ID']]
             .sort_values('Gene Symbol').reset_index(drop=True))
    panel = pd.concat(
        [panel, annotate_panel(panel['Gene Symbol'])], axis=1)
    sheets.append(('MERFISH Gene Panel', panel))

    # --- Xenium gene panel, simplified from the vendor metadata -------------
    xp = pd.read_csv(XENIUM_PANEL_CSV)
    xp = (xp[['gene_name', 'gene_id']]
          .rename(columns={'gene_name': 'Gene Symbol', 'gene_id': 'Gene ID'})
          .drop_duplicates()
          .sort_values('Gene Symbol')
          .reset_index(drop=True))
    xp = pd.concat(
        [xp, annotate_panel(xp['Gene Symbol'])], axis=1)
    sheets.append(('Xenium Gene Panel', xp))

    # --- Figure 1C marker dot plot, as source data -------------------------
    sheets.append(('Figure 1C Marker Expression',
                   marker_expression()))

    return sheets


#endregion

#region Supplementary Data 2 ###################################################
# The composition and proximity results behind "Cell-type composition and
# spatial organization show no coordinated change in pregnancy", as computed by
# 07_prop_prox.py. Every test that was run is reported, including the
# subclasses outside the 84-subclass study set: the false discovery rate of a
# platform and contrast is computed over all of its rows, so dropping any of
# them would leave the reported FDR unreproducible. The In Study Set column
# marks the rows that appear in Supplementary Figure 3.

NOMINAL_P = 0.05
PLATFORM_ORDER = [PLATFORM_LABELS[p] for p in PLATFORMS]
CONTRAST_ORDER = [CONTRAST_LABELS[c] for c in CONTRASTS]

STUDY_SET_DEF = (
    'Yes if the subclass is one of the 84 analyzed throughout the study (at '
    'least 100 cells on at least two platforms); for a pair, Yes only if both '
    'members qualify. Rows marked No were tested but are not shown in '
    'Supplementary Figure 3, and are kept here because the FDR of a test is '
    'computed over all of its rows.')

DESCRIPTION_2 = [
    ('All sheets', '',
     'Cell-type composition and local spatial proximity across reproductive '
     'states, the source data for "Cell-type composition and spatial '
     'organization show no coordinated change in pregnancy" and for '
     'Supplementary Figure 3. Composition was tested on all three platforms, '
     'proximity on the two imaging platforms, which resolve single-cell '
     'coordinates.'),
    ('All sheets', 'Platform', 'Slide-tags, MERFISH or Xenium.'),
    ('All sheets', 'Contrast',
     'Treatment versus reference state. Xenium did not sample the postpartum '
     'state, so it contributes the pregnant versus nulliparous contrast '
     'only.'),
    ('All sheets', 'Cell Type',
     'Subclass of the Allen whole-brain atlas taxonomy, with its numeric '
     'code.'),
    ('All sheets', 'Cell Type Neighborhood',
     'The cellular neighborhood the subclass belongs to, as in Figures 1 '
     'and 2.'),
    ('All sheets', 'In Study Set', STUDY_SET_DEF),
    ('All sheets', 'P Value',
     'Unadjusted P value of the test described for that sheet.'),
    ('All sheets', SIG_COL,
     'Yes if the row passes the false discovery rate threshold used '
     'throughout the study.'),

    ('Composition', '',
     'Differential abundance of each cell subclass between reproductive '
     'states, tested separately on each platform. Subclass counts per sample '
     'were centered log-ratio (CLR) transformed with precision weights '
     '(crumblr) and fitted with variancePartition::dream, condition as a '
     'fixed effect. One row per platform, contrast and subclass, ordered by '
     'P value within a platform and contrast.'),
    ('Composition', 'Mean CLR Abundance',
     'Average CLR-transformed abundance across the samples of the contrast '
     '(limma AveExpr).'),
    ('Composition', 'CLR Difference',
     'Difference in CLR-transformed abundance, treatment minus reference, on '
     'the natural-log scale. Positive means the subclass makes up a larger '
     'share of the section in the treatment state.'),
    ('Composition', 't', 'Moderated t statistic.'),
    ('Composition', 'FDR',
     'Benjamini-Hochberg adjusted P value, computed within a platform and '
     'contrast across all subclasses tested there.'),

    ('Composition Cross-Platform', '',
     'The same tests laid side by side, one row per contrast and subclass, so '
     'that the direction of every shift can be compared across platforms. '
     'Ordered by the lowest P value within a contrast. Blank platform columns '
     'mean the subclass was not annotated on that platform, or that the '
     'platform did not sample the contrast.'),
    ('Composition Cross-Platform', 'Platforms Tested',
     'Number of platforms contributing an estimate for this subclass and '
     'contrast (1 to 3).'),
    ('Composition Cross-Platform', f'Platforms with P < {NOMINAL_P}',
     'Number of those platforms reaching a nominal P value below '
     f'{NOMINAL_P}, before correction for multiple testing.'),
    ('Composition Cross-Platform', 'Direction Concordant',
     'Yes if every platform that tested the subclass gives the shift the same '
     'sign, irrespective of significance. Blank where only one platform '
     'tested it.'),
    ('Composition Cross-Platform', 'Lowest P Value',
     'Smallest P value across the platforms tested.'),
    ('Composition Cross-Platform', 'Lowest FDR',
     'Smallest adjusted P value across the platforms tested.'),

    ('Local Proximity', '',
     'Change in local co-occurrence between every ordered pair of subclasses, '
     'tested separately on the two imaging platforms. For each cell, '
     'neighbors of each subclass within 20 times the sample\'s median '
     'nearest-neighbor distance (~200 um) were counted and converted to an '
     'enrichment z-score against 100 permutations of the subclass labels. '
     'z-scores were averaged per sample and pair and tested with limma, '
     'weighted by the number of contributing cells. A pair had to co-occur in '
     'at least five cells in every sample of the contrast to be tested. One '
     'row per platform, contrast and pair, ordered by P value within a '
     'platform and contrast.'),
    ('Local Proximity', 'Center Cell Type',
     'The subclass whose neighborhood is being described.'),
    ('Local Proximity', 'Surround Cell Type',
     'The subclass counted among those neighbors.'),
    ('Local Proximity', 'Mean Enrichment z',
     'Average enrichment z-score across the samples of the contrast (limma '
     'AveExpr). Positive means the surround subclass is over-represented '
     'among the center subclass\'s neighbors relative to the permuted null.'),
    ('Local Proximity', 'Enrichment z Difference',
     'Difference in mean enrichment z-score, treatment minus reference. '
     'Positive means the two subclasses became more likely to neighbor one '
     'another in the treatment state.'),
    ('Local Proximity', 'FDR',
     'Benjamini-Hochberg adjusted P value, computed within a platform and '
     'contrast across all pairs tested there.'),

    ('Local Proximity Meta', '',
     'MERFISH and Xenium combined for the pregnant versus nulliparous '
     'contrast by sum-of-ranks meta-analysis, the same procedure used for '
     'differential expression. Within each platform, pairs are ranked by '
     'signed significance; the sum of the two normalized ranks is calibrated '
     'against 1,000 permutations of the condition labels to give an empirical '
     'P value per direction. Only pairs tested on both platforms are '
     'reported. Ordered by the smaller of the two empirical P values.'),
    ('Local Proximity Meta', 'Direction Concordant',
     'Yes if the MERFISH and Xenium differences share a sign.'),
    ('Local Proximity Meta', 'Platforms',
     'Number of platforms contributing to the meta-analysis (D); always 2 '
     'here, because only MERFISH and Xenium provide single-cell coordinates '
     'and both sampled this contrast.'),
    ('Local Proximity Meta', 'Sum of Normalized Ranks',
     'Sum of the two within-platform normalized ranks, from 0 (top-ranked '
     'increase on both platforms) to 2 (top-ranked decrease on both).'),
    ('Local Proximity Meta', 'Empirical P (Increased)',
     'Fraction of the permutation null at least as extreme, for an increase '
     'in co-occurrence on both platforms.'),
    ('Local Proximity Meta', 'Empirical P (Decreased)',
     'As above, for a decrease on both platforms.'),
    ('Local Proximity Meta', 'Empirical FDR (Increased)',
     'Benjamini-Hochberg adjustment of the empirical P values for an '
     'increase, across all pairs in the meta-analysis.'),
    ('Local Proximity Meta', 'Empirical FDR (Decreased)',
     'As above, for a decrease.'),

    ('Sample Composition', '',
     'Source data for the composition analysis: the cells of each subclass in '
     'each sample, and the transformed abundance they were tested on. Samples '
     'excluded from the analysis are absent.'),
    ('Sample Composition', 'Cells',
     'Cells of this subclass in this sample, after quality control and label '
     'transfer.'),
    ('Sample Composition', 'Proportion (%)',
     'Cells of this subclass as a percentage of the sample.'),
    ('Sample Composition', 'CLR Abundance',
     'Centered log-ratio transformed abundance (natural log, pseudocount '
     '0.5), the value the differential-abundance model was fitted to.'),
]


def _as_ordered(df):
    """Platform and contrast as ordered categories, to sort as reported."""
    if 'Platform' in df:
        df['Platform'] = pd.Categorical(df['Platform'], PLATFORM_ORDER,
                                        ordered=True)
    if 'Contrast' in df:
        df['Contrast'] = pd.Categorical(df['Contrast'], CONTRAST_ORDER,
                                        ordered=True)
    return df


def _ordered(df, sort_by, ascending=True):
    return _as_ordered(df).sort_values(sort_by, ascending=ascending)\
        .reset_index(drop=True)


def description_sheet():
    return pd.DataFrame(DESCRIPTION_2,
                        columns=['Sheet', 'Column', 'Definition'])


def composition_sheet():
    g = pd.read_csv(f'{PROX}/global_props.csv')
    hood, keep = subclass_neighborhood(), study_subclasses()
    out = pd.DataFrame({
        'Platform': g['dataset'].map(PLATFORM_LABELS),
        'Contrast': g['contrast'].map(CONTRAST_LABELS),
        'Cell Type': g['cell_type'],
        'Cell Type Neighborhood': g['cell_type'].map(hood).fillna(''),
        'In Study Set': yesno(g['cell_type'].isin(keep)),
        'Mean CLR Abundance': g['AveExpr'],
        'CLR Difference': g['logFC'],
        't': g['t'],
        'P Value': g['P.Value'],
        'FDR': g['adj.P.Val'],
        SIG_COL: yesno(g['adj.P.Val'] <= PROX_FDR),
    })
    return round_sig(_ordered(out, ['Platform', 'Contrast', 'P Value']))


def composition_cross_platform_sheet():
    g = pd.read_csv(f'{PROX}/global_props.csv')
    hood, keep = subclass_neighborhood(), study_subclasses()
    wide = g.pivot_table(index=['contrast', 'cell_type'], columns='dataset',
                         values=['logFC', 'P.Value', 'adj.P.Val'],
                         aggfunc='first')
    idx = wide.index.to_frame(index=False)
    lf = np.column_stack([wide[('logFC', p)] for p in PLATFORMS])
    pv = np.column_stack([wide[('P.Value', p)] for p in PLATFORMS])
    fdr = np.column_stack([wide[('adj.P.Val', p)] for p in PLATFORMS])

    tested = ~np.isnan(lf)
    n_tested = tested.sum(axis=1)
    same_sign = (((lf > 0) & tested).sum(axis=1) == 0) | \
                (((lf < 0) & tested).sum(axis=1) == 0)

    out = pd.DataFrame({
        'Contrast': idx['contrast'].map(CONTRAST_LABELS),
        'Cell Type': idx['cell_type'],
        'Cell Type Neighborhood': idx['cell_type'].map(hood).fillna(''),
        'In Study Set': yesno(idx['cell_type'].isin(keep)),
    })
    for j, p in enumerate(PLATFORMS):
        out[f'{PLATFORM_LABELS[p]} CLR Difference'] = lf[:, j]
        out[f'{PLATFORM_LABELS[p]} P Value'] = pv[:, j]
    out['Platforms Tested'] = n_tested
    out[f'Platforms with P < {NOMINAL_P}'] = (pv < NOMINAL_P).sum(axis=1)
    out['Direction Concordant'] = np.where(
        n_tested < 2, '', yesno(same_sign))
    out['Lowest P Value'] = np.nanmin(pv, axis=1)
    out['Lowest FDR'] = np.nanmin(fdr, axis=1)
    out[SIG_COL] = yesno(out['Lowest FDR'] <= PROX_FDR)
    return round_sig(_ordered(out, ['Contrast', 'Lowest P Value']))


def local_proximity_sheet():
    d = pd.read_csv(f'{PROX}/local_diff.csv')
    hood, keep = subclass_neighborhood(), study_subclasses()
    out = pd.DataFrame({
        'Platform': d['dataset'].map(PLATFORM_LABELS),
        'Contrast': d['contrast'].map(CONTRAST_LABELS),
        'Center Cell Type': d['cell_type_a'],
        'Center Neighborhood': d['cell_type_a'].map(hood).fillna(''),
        'Surround Cell Type': d['cell_type_b'],
        'Surround Neighborhood': d['cell_type_b'].map(hood).fillna(''),
        'In Study Set': yesno(d['cell_type_a'].isin(keep) &
                              d['cell_type_b'].isin(keep)),
        'Mean Enrichment z': d['AveExpr'],
        'Enrichment z Difference': d['logFC'],
        't': d['t'],
        'P Value': d['P.Value'],
        'FDR': d['adj.P.Val'],
        SIG_COL: yesno(d['adj.P.Val'] <= PROX_FDR),
    })
    return round_sig(_ordered(out, ['Platform', 'Contrast', 'P Value']))


def local_proximity_meta_sheet():
    sr = pd.read_csv(f'{PROX}/local_sumrank.csv')
    d = pd.read_csv(f'{PROX}/local_diff.csv')
    hood, keep = subclass_neighborhood(), study_subclasses()
    pair = pd.MultiIndex.from_arrays(
        [sr['cell_type_a'], sr['cell_type_b']])

    out = pd.DataFrame({
        'Contrast': sr['contrast'].map(CONTRAST_LABELS),
        'Center Cell Type': sr['cell_type_a'],
        'Center Neighborhood': sr['cell_type_a'].map(hood).fillna(''),
        'Surround Cell Type': sr['cell_type_b'],
        'Surround Neighborhood': sr['cell_type_b'].map(hood).fillna(''),
        'In Study Set': yesno(sr['cell_type_a'].isin(keep) &
                              sr['cell_type_b'].isin(keep)),
    })
    signs = []
    for p in ['merfish', 'xenium']:
        sub = d[(d['dataset'] == p) &
                (d['contrast'] == 'PREG_vs_CTRL')].set_index(
                    ['cell_type_a', 'cell_type_b'])
        assert sub.index.is_unique, f'{p}: duplicate pairs'
        diff = sub['logFC'].reindex(pair).to_numpy()
        out[f'{PLATFORM_LABELS[p]} Enrichment z Difference'] = diff
        out[f'{PLATFORM_LABELS[p]} P Value'] = \
            sub['P.Value'].reindex(pair).to_numpy()
        signs.append(np.sign(diff))
    out['Direction Concordant'] = yesno(signs[0] == signs[1])
    out['Platforms'] = sr['D']
    out['Sum of Normalized Ranks'] = sr['sum_stat']
    out['Empirical P (Increased)'] = sr['emp_p_up']
    out['Empirical P (Decreased)'] = sr['emp_p_down']
    out['Empirical FDR (Increased)'] = sr['emp_fdr_up']
    out['Empirical FDR (Decreased)'] = sr['emp_fdr_down']
    out[SIG_COL] = yesno(
        sr[['emp_fdr_up', 'emp_fdr_down']].min(axis=1) <= PROX_FDR)
    out['_sort'] = sr[['emp_p_up', 'emp_p_down']].min(axis=1)
    out = _ordered(out, ['Contrast', '_sort'])
    return round_sig(out.drop(columns='_sort'))


def sample_composition_sheet():
    gn = pd.read_csv(f'{PROX}/global_norm_props.csv')
    hood, keep = subclass_neighborhood(), study_subclasses()
    counts = pd.concat([subclass_counts(p).assign(dataset=p)
                        for p in PLATFORMS], ignore_index=True)
    counts = counts.rename(columns={'Sample ID': 'sample',
                                    'Cell Type': 'cell_type'})
    df = gn.merge(counts, on=['dataset', 'sample', 'cell_type'], how='left')
    df['Cells'] = df['Cells'].fillna(0).astype(int)
    total = df.groupby(['dataset', 'sample'])['Cells'].transform('sum')

    out = pd.DataFrame({
        'Platform': df['dataset'].map(PLATFORM_LABELS),
        'Sample ID': df['sample'],
        'Condition': df['condition'].map(CONDITION),
        'Cell Type': df['cell_type'],
        'Cell Type Neighborhood': df['cell_type'].map(hood).fillna(''),
        'In Study Set': yesno(df['cell_type'].isin(keep)),
        'Cells': df['Cells'],
        'Proportion (%)': 100 * df['Cells'] / total,
        'CLR Abundance': df['normalized_prop'],
    })
    out['Platform'] = pd.Categorical(out['Platform'], PLATFORM_ORDER,
                                     ordered=True)
    out = out.sort_values(
        ['Platform', 'Sample ID', 'Cell Type'],
        key=lambda c: (c.map(sample_sort_key) if c.name == 'Sample ID' else
                       c.map(subclass_sort_key) if c.name == 'Cell Type' else
                       c)).reset_index(drop=True)
    return round_sig(out)


def build_dataset_2():
    return [
        ('Description', description_sheet()),
        ('Composition', composition_sheet()),
        ('Composition Cross-Platform', composition_cross_platform_sheet()),
        ('Local Proximity', local_proximity_sheet()),
        ('Local Proximity Meta', local_proximity_meta_sheet()),
        ('Sample Composition', sample_composition_sheet()),
    ]


#endregion

#region Supplementary Data 3 and 4 #############################################
# Differential expression and pathway enrichment, as computed by 06_de_gsea.py.
# Two workbooks rather than one: the per-platform differential-expression table
# alone is 1.08 million rows.
#
# Significance follows the figures, not the false discovery rate. The
# permutation-calibrated meta-analysis has no row below an empirical FDR of
# 0.10 at either level, and figures 2-5 call a gene or pathway changed at an
# empirical P below 0.05 with at least two platforms measuring it. Both the
# empirical P values and their FDR are reported, and the flag column follows
# the figure so the tables and the figures agree.

GENE_SIG_COL = 'Significant (Empirical P < 0.05)'
PATH_SIG_COL = 'Significant (Empirical P ≤ 0.05)'

# The journal caps a supplementary file at 30 MB and a submission at 150 MB.
# The unrestricted per-platform tables are 1,080,385 rows of differential
# expression and 212,809 of enrichment, which makes a 102 MB workbook. The
# sheets therefore carry the platform-level rows behind a meta-analysis
# result - 19% and 49% of the rows - and write_deposit() writes the full
# tables for a repository, cited from Data availability.
DEPOSIT = f'{OUT}/deposit'
DEPOSIT_README = """Supplementary Data 3, 4 and 5: unrestricted per-platform
tables
================================================================

Supplementary Data 3, 4 and 5 report the platform-level results behind every
cross-platform result. These three files are the same analyses without that
restriction: every test that was run, including the Slide-tags genes no other
panel measures and the postpartum contrasts that only one platform sampled.

differential_expression_all_platforms.csv.gz
    {n_de:,} rows. Differential expression per platform, contrast, cell
    subclass and gene. Columns are those of the Differential Expression
    sheet of Supplementary Data 3, which defines them.

pathway_enrichment_all_platforms.csv.gz
    {n_gsea:,} rows. Gene set enrichment per platform, contrast, cell
    subclass and pathway. Columns are those of the Pathway Enrichment sheet
    of Supplementary Data 4, which defines them.

intercellular_signalling_all_platforms.csv.gz
    {n_sig:,} rows. Differential ligand-receptor signalling per platform,
    contrast, ordered pair of cell subclasses and interaction, for every
    interaction with a non-zero change; combinations that never fired in
    either condition are omitted. Columns follow the Reproducible Signaling
    sheet of Supplementary Data 5, one platform at a time.

All are UTF-8 comma-separated, gzip compressed, with one header row.
Numeric values carry four significant figures, as in the workbooks.
Generated by 24_supp_tables.py from the output of 06_de_gsea.py and
08_signalling.py.
"""


def theme_label(theme):
    """Theme as it is shown: the code names it Maternal_Reproduction."""
    return theme.replace('_', ' ')


@lru_cache(maxsize=1)
def pathway_theme():
    """GO:BP pathway -> theme, from the curated collection."""
    os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
    from ryp import r, to_py
    r(f'.p <- unique(readRDS("{THEMED_GO_RDS}")[, c("gs_name", "theme")])')
    df = to_py('.p', format='pandas')
    return {g: theme_label(t) for g, t in zip(df['gs_name'], df['theme'])}


@lru_cache(maxsize=1)
def theme_keywords():
    """[(theme, [keyword, ...])] in the order the themes are applied.

    Parsed out of the theme_keywords list in 06_de_gsea.py rather than
    transcribed, so the table cannot drift from the code that built the
    collection. Order is part of the definition: a pathway takes the first
    theme whose keywords its name matches.
    """
    src = open(DE_GSEA_SCRIPT).read()
    block = re.search(r'theme_keywords <- list\((.*?)\n    \)\n', src, re.S)
    if block is None:
        raise RuntimeError(f'theme_keywords not found in {DE_GSEA_SCRIPT}')
    return [(theme, re.findall(r"'([^']+)'", kws))
            for theme, kws in re.findall(
                r"'([A-Za-z_]+)' = c\((.*?)\)", block.group(1), re.S)]


def meta_direction(df):
    """Tested direction and its empirical P, as figures 2-5 choose them."""
    up = df['nlp_up'] >= df['nlp_down']
    return (np.where(up, 'Increased', 'Decreased'),
            np.where(up, df['emp_p_up'], df['emp_p_down']))


def meta_keys(path, item):
    """The (contrast, cell type, item) triples that reached a meta-analysis."""
    k = pd.read_csv(path, usecols=['contrast', 'cell_type', item])
    return pd.MultiIndex.from_frame(k.drop_duplicates())


def _de_frame(restrict=True):
    """Per-platform differential expression, with display columns."""
    d = pd.read_csv(f'{DE_DIR}/de_results.csv')
    if restrict:
        idx = pd.MultiIndex.from_frame(
            d[['contrast', 'cell_type', 'gene']])
        d = d[idx.isin(meta_keys(f'{DE_DIR}/sumrank_results.csv', 'gene'))]
    keep = study_subclasses()
    return pd.DataFrame({
        'Platform': d['dataset'].map(PLATFORM_LABELS),
        'Contrast': d['contrast'].map(CONTRAST_LABELS),
        'Cell Type': d['cell_type'],
        'In Study Set': yesno(d['cell_type'].isin(keep)),
        'Gene Symbol': d['gene'],
        'log2 Fold Change': d['logFC'],
        'Standard Error': d['SE'],
        '95% CI Lower': d['LCI'],
        '95% CI Upper': d['UCI'],
        'Mean Expression (log2 CPM)': d['AveExpr'],
        'P Value': d['PValue'],
        'FDR': d['FDR'],
        SIG_COL: yesno(d['FDR'] <= DE_FDR),
        'Reference Detection (%)': d['ref_pct_detected'],
        'Cells Expressing (Treatment)': d['n_cells_treat'],
        'Cells Expressing (Reference)': d['n_cells_base'],
        '_contrast': d['contrast'],
        '_sort': d['cell_type'].map(subclass_sort_key),
    })


def de_sheet(restrict=True):
    df = _ordered(_de_frame(restrict),
                  ['Platform', 'Contrast', '_sort', 'P Value'])
    return round_sig(df.drop(columns=['_contrast', '_sort']))


def gene_meta_sheet():
    sr = pd.read_csv(f'{DE_DIR}/sumrank_results.csv')
    hood, keep = subclass_neighborhood(), study_subclasses()
    direction, emp = meta_direction(sr)
    out = pd.DataFrame({
        'Contrast': sr['contrast'].map(CONTRAST_LABELS),
        'Cell Type': sr['cell_type'],
        'Cell Type Neighborhood': sr['cell_type'].map(hood).fillna(''),
        'In Study Set': yesno(sr['cell_type'].isin(keep)),
        'Gene Symbol': sr['gene'],
        'Platforms': sr['D'],
        'Sum of Normalized Ranks': sr['sum_stat'],
        'Direction': direction,
        'Empirical P (Increased)': sr['emp_p_up'],
        'Empirical P (Decreased)': sr['emp_p_down'],
        'Empirical FDR (Increased)': sr['emp_fdr_up'],
        'Empirical FDR (Decreased)': sr['emp_fdr_down'],
        GENE_SIG_COL: yesno(emp < EMP_P),
        'Reference Detection (%)': sr['ref_pct_detected'],
        'Cells Expressing (Treatment)': sr['n_cells_treat'],
        'Cells Expressing (Reference)': sr['n_cells_base'],
    })
    out['_sort'] = sr['cell_type'].map(subclass_sort_key)
    out['_p'] = emp
    out = _ordered(out, ['Contrast', '_sort', '_p'])
    return round_sig(out.drop(columns=['_sort', '_p']))


def genes_changed_sheet():
    """Source data for Figure 2A: genes changed per cell type and contrast.

    Counts each direction independently at empirical P < 0.05 with at least
    two platforms, the rule the figure applies.
    """
    sr = pd.read_csv(f'{DE_DIR}/sumrank_results.csv')
    hood, keep = subclass_neighborhood(), study_subclasses()
    d2 = sr['D'] >= MIN_PLATFORMS
    g = sr.assign(
        up=((sr['emp_p_up'] < EMP_P) & d2).astype(int),
        dn=((sr['emp_p_down'] < EMP_P) & d2).astype(int),
    ).groupby(['contrast', 'cell_type'], observed=True).agg(
        tested=('gene', 'size'), up=('up', 'sum'), dn=('dn', 'sum')
    ).reset_index()
    out = pd.DataFrame({
        'Contrast': g['contrast'].map(CONTRAST_LABELS),
        'Cell Type': g['cell_type'],
        'Cell Type Neighborhood': g['cell_type'].map(hood).fillna(''),
        'In Study Set': yesno(g['cell_type'].isin(keep)),
        'Genes Tested': g['tested'],
        'Genes Increased': g['up'],
        'Genes Decreased': g['dn'],
        'Genes Changed': g['up'] + g['dn'],
        'Percent Changed': 100 * (g['up'] + g['dn']) / g['tested'],
    })
    return round_sig(_ordered(out, ['Contrast', 'Genes Changed'],
                              ascending=[True, False]))


def pathway_platform_sheet(restrict=True):
    """Per-platform fgsea, from the cache the meta-analysis was built on."""
    g = pd.read_parquet(f'{GSEA_DIR}/perms/real_gsea.parquet')
    if restrict:
        idx = pd.MultiIndex.from_frame(
            g[['contrast', 'cell_type', 'pathway']])
        g = g[idx.isin(meta_keys(
            f'{GSEA_DIR}/sumrank_gsea_results.csv', 'pathway'))]
    theme, keep = pathway_theme(), study_subclasses()
    out = pd.DataFrame({
        'Platform': g['dataset'].map(PLATFORM_LABELS),
        'Contrast': g['contrast'].map(CONTRAST_LABELS),
        'Cell Type': g['cell_type'],
        'In Study Set': yesno(g['cell_type'].isin(keep)),
        'Pathway': g['pathway'],
        'Pathway Theme': g['pathway'].map(theme).fillna(''),
        'Normalized Enrichment Score': g['NES'],
        'P Value': g['pvalue'],
        'Leading Edge Genes': g['leading_edge'],
    })
    out['_sort'] = g['cell_type'].map(subclass_sort_key)
    out = _ordered(out, ['Platform', 'Contrast', '_sort', 'P Value'])
    return round_sig(out.drop(columns='_sort'))


def pathway_meta_sheet():
    g = pd.read_csv(f'{GSEA_DIR}/sumrank_gsea_results.csv')
    hood, keep = subclass_neighborhood(), study_subclasses()
    theme = pathway_theme()
    direction, emp = meta_direction(g)
    out = pd.DataFrame({
        'Contrast': g['contrast'].map(CONTRAST_LABELS),
        'Cell Type': g['cell_type'],
        'Cell Type Neighborhood': g['cell_type'].map(hood).fillna(''),
        'In Study Set': yesno(g['cell_type'].isin(keep)),
        'Pathway': g['pathway'],
        'Pathway Theme': g['pathway'].map(theme).fillna(''),
        'Platforms': g['D'],
        'Sum of Normalized Ranks': g['sum_stat'],
        'Direction': direction,
        'Empirical P (Increased)': g['emp_p_up'],
        'Empirical P (Decreased)': g['emp_p_down'],
        'Empirical FDR (Increased)': g['emp_fdr_up'],
        'Empirical FDR (Decreased)': g['emp_fdr_down'],
        PATH_SIG_COL: yesno(emp <= EMP_P),
    })
    for p in PLATFORMS:
        col = f'leading_edge_{p}'
        if col in g:
            out[f'Leading Edge ({PLATFORM_LABELS[p]})'] = g[col].fillna('')
    out['_sort'] = g['cell_type'].map(subclass_sort_key)
    out['_p'] = emp
    out = _ordered(out, ['Contrast', '_sort', '_p'])
    return round_sig(out.drop(columns=['_sort', '_p']))


def pathway_themes_sheet():
    counts = pd.Series(pathway_theme()).value_counts()
    rows = [{'Match Order': i,
             'Theme': theme_label(theme),
             'Keywords': ', '.join(kws),
             'GO Biological Processes Assigned':
                 int(counts.get(theme_label(theme), 0))}
            for i, (theme, kws) in enumerate(theme_keywords(), start=1)]
    return pd.DataFrame(rows)


DESCRIPTION_3 = [
    ('All sheets', '',
     'Differential expression across reproductive states, the source data for '
     '"Pregnancy remodels the maternal brain through distinct neuronal, '
     'glial-vascular and lipid programs" and for Figure 2A. Each platform was '
     'tested separately and the platforms were then combined by '
     'meta-analysis; only the meta-analysis is used to call a gene changed.'),
    ('All sheets', 'Cell Type',
     'Subclass of the Allen whole-brain atlas taxonomy, with its numeric '
     'code.'),
    ('All sheets', 'In Study Set', STUDY_SET_DEF),
    ('All sheets', 'Reference Detection (%)',
     'Percentage of cells of this subclass expressing the gene in the Allen '
     'single-cell reference, as a guide to whether the gene is expected in '
     'this cell type at all.'),
    ('All sheets', 'Cells Expressing (Treatment)',
     'Cells of this subclass with at least one count of the gene, in the '
     'treatment state of the contrast. A gene was tested only where at least '
     '10 cells expressed it in each state.'),

    ('Differential Expression', '',
     'Differential expression on each platform separately, one row per '
     'platform, contrast, cell subclass and gene, ordered by P value within '
     'a platform and subclass. Counts were pseudobulked per sample and '
     'subclass and fitted with voomByGroup and limma, with condition as a '
     'fixed effect and the number of cells and library size as covariates '
     '(library size on Slide-tags and MERFISH only). Restricted to the '
     'subclass and gene pairs that reached the meta-analysis, so every row '
     'of the next sheet can be traced to the platform estimates behind it. '
     'The unrestricted table of all 1,080,385 tests, which includes the '
     'genes only Slide-tags measures, is deposited with the study data.'),
    ('Differential Expression', 'log2 Fold Change',
     'Treatment relative to reference. Positive means the gene increased in '
     'expression in the treatment state.'),
    ('Differential Expression', 'Mean Expression (log2 CPM)',
     'Average expression across the samples of the contrast, in log2 counts '
     'per million (limma AveExpr).'),
    ('Differential Expression', 'FDR',
     'Benjamini-Hochberg adjusted P value, computed within a platform, '
     'contrast and cell subclass, over all genes tested there rather than '
     'the subset shown.'),
    ('Differential Expression', SIG_COL,
     'Yes if the gene passes the false discovery rate threshold on this one '
     'platform. A gene is called changed in the study only if the '
     'meta-analysis supports it, so this flag is descriptive rather than the '
     'criterion used.'),

    ('Cross-Platform Meta-analysis', '',
     'Genes ranked by signed significance within each platform and combined '
     'by sum of ranks, calibrated against 1,000 permutations of the condition '
     'labels. Only genes measured on at least two platforms enter, so the '
     'pregnancy contrast rests on the 3,881 genes shared by at least two of '
     'the three panels and the postpartum contrasts on roughly 350. One row '
     'per contrast, cell subclass and gene, ordered by empirical P within a '
     'subclass.'),
    ('Cross-Platform Meta-analysis', 'Platforms',
     'Number of platforms that measured this gene in this cell subclass (D); '
     'at least 2 by construction.'),
    ('Cross-Platform Meta-analysis', 'Sum of Normalized Ranks',
     'Sum of the within-platform normalized ranks, from 0 (top-ranked '
     'increase on every platform) to D (top-ranked decrease on every '
     'platform).'),
    ('Cross-Platform Meta-analysis', 'Direction',
     'The direction tested, taken as the stronger of the two tails, as in '
     'figures 2 to 5.'),
    ('Cross-Platform Meta-analysis', 'Empirical P (Increased)',
     'Fraction of the permutation null at least as extreme, for an increase '
     'on every contributing platform. One-sided.'),
    ('Cross-Platform Meta-analysis', 'Empirical FDR (Increased)',
     'Benjamini-Hochberg adjustment of the empirical P values for an '
     'increase, across all genes of the contrast. No gene reaches 0.10, which '
     'is why the figures call significance on the empirical P value.'),
    ('Cross-Platform Meta-analysis', GENE_SIG_COL,
     'Yes if the empirical P value for the tested direction is below 0.05, '
     'the rule figures 2 to 5 apply.'),

    ('Genes Changed per Cell Type', '',
     'Source data for Figure 2A: per contrast and cell subclass, the genes '
     'reaching an empirical P below 0.05 in each direction, counted '
     'independently, against the genes tested in the meta-analysis.'),
]

DESCRIPTION_4 = [
    ('All sheets', '',
     'Gene set enrichment across reproductive states, the source data for '
     'Figure 2B and figures 3 to 5. Genes were preranked within each cell '
     'subclass by signed significance and tested against a curated collection '
     'of 3,043 GO biological processes with fgsea. Each platform was tested '
     'separately and the platforms were then combined by meta-analysis; only '
     'the meta-analysis is used to call a pathway changed.'),
    ('All sheets', 'Cell Type',
     'Subclass of the Allen whole-brain atlas taxonomy, with its numeric '
     'code.'),
    ('All sheets', 'In Study Set', STUDY_SET_DEF),
    ('All sheets', 'Pathway',
     'MSigDB gene set name for the GO biological process.'),
    ('All sheets', 'Pathway Theme',
     'The theme the pathway was assigned to, as defined in the Pathway '
     'Themes sheet.'),

    ('Pathway Enrichment', '',
     'Enrichment on each platform separately, one row per platform, '
     'contrast, cell subclass and pathway, ordered by P value within a '
     'platform and subclass. A cell subclass was tested where at least 100 '
     'genes were ranked, and a pathway where at least 15 of its genes were '
     'present. No adjusted P value is reported at this level: the '
     'per-platform results are the input to the meta-analysis, which carries '
     'the calibrated significance. Restricted to the subclass and pathway '
     'pairs that reached the meta-analysis; the unrestricted table of all '
     '212,809 tests is deposited with the study data.'),
    ('Pathway Enrichment', 'Normalized Enrichment Score',
     'fgsea NES. Positive means the pathway is enriched among genes that '
     'increased in expression in the treatment state.'),
    ('Pathway Enrichment', 'Leading Edge Genes',
     'The genes driving the enrichment signal on that platform.'),

    ('Cross-Platform Meta-analysis', '',
     'Pathways ranked by signed significance within each platform and '
     'combined by sum of ranks, calibrated against 1,000 permutations of the '
     'condition labels, the same procedure used for genes. Only pathways '
     'scored on at least two platforms enter. One row per contrast, cell '
     'subclass and pathway, ordered by empirical P within a subclass.'),
    ('Cross-Platform Meta-analysis', 'Platforms',
     'Number of platforms that scored this pathway in this cell subclass (D); '
     'at least 2 by construction.'),
    ('Cross-Platform Meta-analysis', 'Sum of Normalized Ranks',
     'Sum of the within-platform normalized ranks, from 0 (top-ranked '
     'increase on every platform) to D (top-ranked decrease on every '
     'platform).'),
    ('Cross-Platform Meta-analysis', 'Direction',
     'The direction tested, taken as the stronger of the two tails, as in '
     'figures 2 to 5.'),
    ('Cross-Platform Meta-analysis', 'Empirical P (Increased)',
     'Fraction of the permutation null at least as extreme, for an increase '
     'on every contributing platform. One-sided.'),
    ('Cross-Platform Meta-analysis', 'Empirical FDR (Increased)',
     'Benjamini-Hochberg adjustment of the empirical P values for an '
     'increase, across all pathways of the contrast. No pathway reaches 0.10, '
     'which is why the figures call significance on the empirical P value.'),
    ('Cross-Platform Meta-analysis', PATH_SIG_COL,
     'Yes if the empirical P value for the tested direction is at or below '
     '0.05, the rule figures 2 to 5 apply.'),

    ('Pathway Themes', '',
     'How the GO biological processes were grouped. Every GO:BP gene set was '
     'matched against the keyword lists below in the order shown and took the '
     'first theme it matched; the 3,043 that matched at least one theme form '
     'the collection tested, and the rest were not tested. Keywords match '
     'anywhere in the gene set name and are case-insensitive.'),
    ('Pathway Themes', 'Match Order',
     'The order in which themes were applied. Earlier themes are more '
     'specific and win a tie: a pathway naming both pregnancy and metabolism '
     'is maternal, not metabolic.'),
]


def build_dataset_3():
    return [
        ('Description', pd.DataFrame(
            DESCRIPTION_3, columns=['Sheet', 'Column', 'Definition'])),
        ('Differential Expression', de_sheet()),
        ('Cross-Platform Meta-analysis', gene_meta_sheet()),
        ('Genes Changed per Cell Type', genes_changed_sheet()),
    ]


def build_dataset_4():
    return [
        ('Description', pd.DataFrame(
            DESCRIPTION_4, columns=['Sheet', 'Column', 'Definition'])),
        ('Pathway Enrichment', pathway_platform_sheet()),
        ('Cross-Platform Meta-analysis', pathway_meta_sheet()),
        ('Pathway Themes', pathway_themes_sheet()),
    ]


#endregion

#region Supplementary Data 5 ###################################################
# Intercellular signalling, as computed by 08_signalling.py.
#
# Unlike differential expression and enrichment, a differential signalling
# result carries no P value. The permutation test of 08_signalling.py asks
# whether an interaction is specific to a cell-type pair within a sample, not
# whether it changes between conditions; a change is a difference of condition
# means, and reproducibility is directional agreement between two platforms
# above a magnitude floor. The workbook reports that criterion and says so.

LIANA_DIR = f'{working_dir}/output/liana'
LIANA_KEYS = ['source', 'target', 'ligand_complex', 'receptor_complex']
SIG_PLATFORMS = ['slidetags', 'xenium']
SIG_CONTRAST = ('PREG', 'CTRL')
# the chord panels, with the build_chord arguments that change which edges
# survive; cell_allow and k_neurons only choose which cells are drawn
FIGURE_CHORDS = [
    ('Figure 3F', f'{working_dir}/13_figure_3.py',
     dict(mag_floor=False, neuron_intrinsic=True, include_nn=True)),
    ('Figure 4E', f'{working_dir}/14_figure_4.py', dict()),
    ('Figure 5F', f'{working_dir}/15_figure_5.py', dict()),
]
CURATED_CACHE = f'{LIANA_DIR}/curated_lr_diff.parquet'


def liana_contrast(platform, treat, base):
    """One platform's differential signalling, per interaction and cell pair.

    This is the merge 08_signalling.py writes to inflow_diff.csv. That file is
    6.7 GB and 51.6 million rows, the great majority of them interaction and
    cell-pair combinations that never fired, so it is rebuilt here from the two
    per-condition aggregates rather than read back.
    """
    import polars as pl

    def cond(c):
        return pl.read_parquet(
            f'{LIANA_DIR}/{platform}/inflow_cond_{c}_subclass.parquet')
    m = cond(base).join(cond(treat), on=LIANA_KEYS, how='full',
                        suffix='_treat', coalesce=True)
    for c in ('lr_mean', 'n_sig', 'n_present'):
        m = m.with_columns(pl.col(c).fill_null(0),
                           pl.col(f'{c}_treat').fill_null(0))
    return m.with_columns(
        (pl.col('lr_mean_treat') - pl.col('lr_mean')).alias('diff')
    ).select(LIANA_KEYS + ['lr_mean', 'lr_mean_treat', 'diff'])


@lru_cache(maxsize=1)
def reproducible_signalling():
    """Interactions that change consistently on both platforms in pregnancy.

    The rule of the Methods and of build_chord() in 12_figure_helper.py:
    measured on both platforms, the same direction on each, and at least the
    median absolute change of that platform. Self-signalling is dropped, as in
    the figures.
    """
    import polars as pl
    treat, base = SIG_CONTRAST
    d = {p: liana_contrast(p, treat, base) for p in SIG_PLATFORMS}
    floor = {p: float(v.filter(pl.col('diff') != 0)['diff'].abs().median())
             for p, v in d.items()}
    w = (d['slidetags'].rename({'lr_mean': 'st_base',
                                'lr_mean_treat': 'st_treat', 'diff': 'st'})
         .join(d['xenium'].rename({'lr_mean': 'xn_base',
                                   'lr_mean_treat': 'xn_treat', 'diff': 'xn'}),
               on=LIANA_KEYS, how='inner'))
    return (w.filter((pl.col('st') != 0) & (pl.col('xn') != 0)
                     & (pl.col('st').sign() == pl.col('xn').sign())
                     & (pl.col('st').abs() >= floor['slidetags'])
                     & (pl.col('xn').abs() >= floor['xenium'])
                     & (pl.col('source') != pl.col('target')))
            .with_columns(((pl.col('st') + pl.col('xn')) / 2)
                          .alias('combined'))
            .sort(pl.col('combined').abs(), descending=True))


def figure_lr_pairs(path):
    """The curated ligand-receptor pairs and themes of one figure script.

    Executed out of the figure's own source rather than transcribed: the
    assignments and the loops that extend them are lifted by their names, so
    the table cannot drift from the panel it documents.
    """
    import ast
    want = ('CANONICAL_LR_PAIRS', 'CHORD_THEME_LIGANDS', 'CHORD_THEME_TITLES')

    def defines(node):
        """True where the node binds or mutates one of the names, as opposed
        to merely passing it to build_chord()."""
        if isinstance(node, ast.Assign):
            return any(getattr(t, 'id', None) in want for t in node.targets)
        return any(isinstance(s, ast.Call)
                   and isinstance(s.func, ast.Attribute)
                   and getattr(s.func.value, 'id', None) in want
                   and s.func.attr in ('add', 'update')
                   for s in ast.walk(node))

    tree = ast.parse(open(path).read())
    body = [n for n in tree.body
            if isinstance(n, (ast.Assign, ast.For, ast.Expr)) and defines(n)]
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), path, 'exec'), ns)
    # the figure's theme titles read 'descriptor\nligand(s)'; the ligands have
    # their own column here, so the descriptor alone names the theme
    titles = {k: v.split('\n')[0]
              for k, v in ns['CHORD_THEME_TITLES'].items()}
    return ns['CANONICAL_LR_PAIRS'], ns['CHORD_THEME_LIGANDS'], titles


def curated_lr_cache():
    """inflow_diff.csv, restricted to the ligands and receptors of the chords.

    build_chord() reads either layout; giving it this subset keeps its own
    filters, magnitude floors and theme assignment untouched while avoiding a
    6.7 GB scan for each panel.
    """
    import polars as pl
    if os.path.exists(CURATED_CACHE):
        return CURATED_CACHE
    pairs = set().union(*(figure_lr_pairs(p)[0] for _, p, _ in FIGURE_CHORDS))
    ligs, recs = {p[0] for p in pairs}, {p[1] for p in pairs}
    treat, base = SIG_CONTRAST
    parts = []
    for p in SIG_PLATFORMS:
        parts.append(liana_contrast(p, treat, base)
                     .filter(pl.col('ligand_complex').is_in(list(ligs))
                             & pl.col('receptor_complex').is_in(list(recs)))
                     .rename({'diff': 'lr_mean_diff'})
                     .with_columns(pl.lit(p).alias('dataset'),
                                   pl.lit(f'{treat}_vs_{base}')
                                   .alias('contrast')))
    pl.concat(parts).write_parquet(CURATED_CACHE)
    return CURATED_CACHE


def signalling_sheet():
    import polars as pl
    r = reproducible_signalling()
    hood, keep = subclass_neighborhood(), study_subclasses()
    src, tgt = r['source'].to_list(), r['target'].to_list()
    out = pd.DataFrame({
        'Contrast': CONTRAST_LABELS['PREG_vs_CTRL'],
        'Source Cell Type': src,
        'Source Neighborhood': [hood.get(c, '') for c in src],
        'Target Cell Type': tgt,
        'Target Neighborhood': [hood.get(c, '') for c in tgt],
        'In Study Set': yesno([s in keep and t in keep
                               for s, t in zip(src, tgt)]),
        'Ligand': r['ligand_complex'].to_list(),
        'Receptor': r['receptor_complex'].to_list(),
        'Slide-tags Score (Nulliparous)': r['st_base'].to_list(),
        'Slide-tags Score (Pregnant)': r['st_treat'].to_list(),
        'Slide-tags Difference': r['st'].to_list(),
        'Xenium Score (Nulliparous)': r['xn_base'].to_list(),
        'Xenium Score (Pregnant)': r['xn_treat'].to_list(),
        'Xenium Difference': r['xn'].to_list(),
        'Combined Difference': r['combined'].to_list(),
        'Direction': np.where(np.array(r['combined'].to_list()) > 0,
                              'Increased', 'Decreased'),
    })
    return round_sig(out)


def curated_signalling_sheet():
    """Every reproducible interaction on the ligand-receptor pairs of the
    chord panels, built by the figures' own build_chord()."""
    import importlib
    import polars as pl
    fc = importlib.import_module('12_figure_helper')
    path = curated_lr_cache()
    hood = subclass_neighborhood()
    per_platform = reproducible_signalling().select(LIANA_KEYS + ['st', 'xn'])
    frames = []
    for label, script, kw in FIGURE_CHORDS:
        pairs, themes, titles = figure_lr_pairs(script)
        _, edges = fc.build_chord(path, pairs, themes, k_neurons=10 ** 6, **kw)
        frames.append(edges.with_columns(
            pl.lit(label).alias('figure'),
            pl.col('theme').replace(titles)))
    e = pl.concat(frames).join(per_platform, on=LIANA_KEYS, how='left')
    src, tgt = e['source'].to_list(), e['target'].to_list()
    out = pd.DataFrame({
        'Figure': e['figure'].to_list(),
        'Signaling Theme': e['theme'].to_list(),
        'Source Cell Type': src,
        'Source Neighborhood': [hood.get(c, '') for c in src],
        'Target Cell Type': tgt,
        'Target Neighborhood': [hood.get(c, '') for c in tgt],
        'Ligand': e['ligand_complex'].to_list(),
        'Receptor': e['receptor_complex'].to_list(),
        'Slide-tags Difference': e['st'].to_list(),
        'Xenium Difference': e['xn'].to_list(),
        'Combined Difference': e['meta_diff'].to_list(),
        'Direction': np.where(np.array(e['meta_diff'].to_list()) > 0,
                              'Increased', 'Decreased'),
    })
    out = out.sort_values(['Figure', 'Signaling Theme', 'Combined Difference'],
                          key=lambda c: (c.abs() if c.name ==
                                         'Combined Difference' else c),
                          ascending=[True, True, False])
    return round_sig(out.reset_index(drop=True))


@lru_cache(maxsize=1)
def lr_resource_sheet():
    """The ligand-receptor pairs searched, and where each came from."""
    import liana as li
    os.environ['R_HOME'] = os.path.expanduser('~/miniforge3/lib/R')
    from ryp import r, to_py
    mc = li.rs.select_resource('mouseconsensus')[['ligand', 'receptor']]
    r('''
    suppressPackageStartupMessages(library(NeuronChat))
    data("interactionDB_mouse")
    nc_df <- do.call(rbind, lapply(interactionDB_mouse, function(x)
        expand.grid(ligand = x$lig_contributor, receptor = x$receptor_subunit,
                    stringsAsFactors = FALSE)))
    nc_df <- unique(nc_df[, c("ligand", "receptor")])
    rownames(nc_df) <- NULL
    ''')
    nc = to_py('nc_df', format='pandas', index=False)
    mc_set = set(map(tuple, mc.to_numpy()))
    nc_set = set(map(tuple, nc.to_numpy()))
    both = sorted(mc_set | nc_set)
    tested = {}
    for p in SIG_PLATFORMS:
        d = pd.read_parquet(
            f'{LIANA_DIR}/{p}/inflow_cond_CTRL_subclass.parquet',
            columns=['ligand_complex', 'receptor_complex'])
        tested[p] = set(map(tuple, d.drop_duplicates().to_numpy()))
    src = ['LIANA mouse consensus and NeuronChat'
           if k in mc_set and k in nc_set
           else 'LIANA mouse consensus' if k in mc_set else 'NeuronChat'
           for k in both]
    return pd.DataFrame({
        'Ligand': [k[0] for k in both],
        'Receptor': [k[1] for k in both],
        'Source Database': src,
        'Tested (Slide-tags)': yesno([k in tested['slidetags'] for k in both]),
        'Tested (Xenium)': yesno([k in tested['xenium'] for k in both]),
    })


DESCRIPTION_5 = [
    ('All sheets', '',
     'Intercellular signalling in pregnancy, the source data for the chord '
     'panels of Figures 3F, 4E and 5F. Ligand-receptor interactions were '
     'scored per cell with LIANA on a spatial neighbourhood graph, retaining '
     'spatially structured signals, and aggregated to ordered pairs of cell '
     'subclasses (Methods). Slide-tags and Xenium were analysed separately '
     'and combined by directional agreement.'),
    ('All sheets', 'Source Cell Type',
     'The subclass sending the signal, that is, expressing the ligand.'),
    ('All sheets', 'Target Cell Type',
     'The subclass receiving it, that is, expressing the receptor. '
     'Self-signalling pairs, where source and target are the same subclass, '
     'are excluded here as they are from the figures.'),
    ('All sheets', 'In Study Set', STUDY_SET_DEF),
    ('All sheets', 'Combined Difference',
     'Mean of the two platform differences, the effect the figures draw. '
     'Positive means the interaction was stronger in pregnancy.'),
    ('All sheets', 'Direction',
     'Increased or decreased in pregnancy relative to nulliparous.'),

    ('Reproducible Signaling', '',
     'Every interaction changing consistently across platforms in pregnancy: '
     'measured on both, moving in the same direction on each, and reaching at '
     'least the median absolute change of that platform. One row per '
     'interaction and ordered pair of cell subclasses, ordered by the size of '
     'the combined difference. No P value is reported for a difference: the '
     'permutation test in the pipeline establishes that an interaction is '
     'specific to a pair of cell types within a sample, not that it changes '
     'between conditions, so reproducibility across two independently '
     'acquired platforms is the criterion. Postpartum contrasts were measured '
     'on Slide-tags alone, cannot meet this criterion, and are deposited with '
     'the study data rather than reported here.'),
    ('Reproducible Signaling', 'Slide-tags Score (Nulliparous)',
     'Mean interaction score across that condition\'s samples on that '
     'platform, in the arbitrary units of the LIANA inflow score. The '
     'difference between the two conditions is what is compared; the scores '
     'themselves are given so the size of a change can be read against the '
     'level it moved from.'),
    ('Reproducible Signaling', 'Slide-tags Difference',
     'Pregnant minus nulliparous on that platform.'),

    ('Curated Interactions', '',
     'The subset of the ligand-receptor pairs curated for the chord panels, '
     'reported for every cell-type pair that meets the reproducibility rule. '
     'Built by the same function that draws the panels, so the selection '
     'cannot drift; each panel then displays a subset of these cell types '
     'chosen for legibility, so this sheet is a superset of what is drawn. '
     'Pairs measured on Slide-tags alone, such as ligands outside the Xenium '
     'panel, enter at the top decile of effect magnitude, as in the figures.'),
    ('Curated Interactions', 'Figure', 'The chord panel the pair belongs to.'),
    ('Curated Interactions', 'Signaling Theme',
     'The grouping the panel draws the interaction under.'),

    ('Ligand-Receptor Resource', '',
     'Every ligand-receptor pair searched, from the union of the LIANA mouse '
     'consensus resource and the neurotransmitter and neuropeptide '
     'interactions of NeuronChat. A pair is marked as tested on a platform '
     'where both partners were measured and the interaction survived the '
     'spatial-autocorrelation filter in at least one sample.'),
]


def build_dataset_5():
    return [
        ('Description', pd.DataFrame(
            DESCRIPTION_5, columns=['Sheet', 'Column', 'Definition'])),
        ('Reproducible Signaling', signalling_sheet()),
        ('Curated Interactions', curated_signalling_sheet()),
        ('Ligand-Receptor Resource', lr_resource_sheet()),
    ]


def signalling_deposit():
    """Every platform and contrast, minus interactions that never fired."""
    import polars as pl
    hood = subclass_neighborhood()
    rows = []
    for p, contrasts in (('slidetags', [('PREG', 'CTRL'),
                                        ('POSTPART', 'PREG'),
                                        ('POSTPART', 'CTRL')]),
                         ('xenium', [('PREG', 'CTRL')])):
        for treat, base in contrasts:
            d = liana_contrast(p, treat, base).filter(pl.col('diff') != 0)
            src, tgt = d['source'].to_list(), d['target'].to_list()
            rows.append(pd.DataFrame({
                'Platform': PLATFORM_LABELS[p],
                'Contrast': CONTRAST_LABELS[f'{treat}_vs_{base}'],
                'Source Cell Type': src,
                'Source Neighborhood': [hood.get(c, '') for c in src],
                'Target Cell Type': tgt,
                'Target Neighborhood': [hood.get(c, '') for c in tgt],
                'Ligand': d['ligand_complex'].to_list(),
                'Receptor': d['receptor_complex'].to_list(),
                'Score (Reference)': d['lr_mean'].to_list(),
                'Score (Treatment)': d['lr_mean_treat'].to_list(),
                'Difference': d['diff'].to_list()}))
    return round_sig(pd.concat(rows, ignore_index=True))

#endregion

#region Supplementary Data 6 ###################################################
# The immunofluorescence validations of Figures 3E and 5E, from the hand-built
# quantification workbook. That file lays each target out as two side-by-side
# blocks of animals, one column per image, and carries two further targets
# (APOE in astrocytes, MERTK) that no figure shows; this reads the four that
# are shown and reshapes them into one row per image.
#
# Per-animal means are recomputed from the images rather than read from the
# workbook's own summary column. The two agree for 55 of the 56 animals; the
# exception is noted where it arises.

IF_XLSX = f'{working_dir}/input/IF validation quantification.xlsx'
IF_ANIMAL = r'(?i)(nonp|p)\s*(\d+)'
VALIDATIONS = [
    dict(sheet='NONP_P_FKBP5', panel='Figure 3E', protein='FKBP5',
         cell_type='', marker='', region='Striatum',
         value_row='FKBP5 + area', area_row=None, coloc_row=None,
         measurement='FKBP5-positive area'),
    dict(sheet='NONP_P_APOE_IBA1', panel='Figure 5E', protein='APOE',
         cell_type='Microglia', marker='IBA1',
         region='Subventricular zone',
         value_row='normalized microglia ApoE', area_row='microglia area',
         coloc_row='coloc',
         measurement='APOE signal within IBA1-positive objects, per unit '
                     'IBA1-positive area'),
    dict(sheet='NONP_P_APOE_NG2', panel='Figure 5E', protein='APOE',
         cell_type='Oligodendrocyte precursor cells', marker='NG2',
         region='Subventricular zone',
         value_row='normalized OPC ApoE', area_row='OPC area',
         coloc_row='coloc',
         measurement='APOE signal within NG2-positive objects, per unit '
                     'NG2-positive area'),
    dict(sheet='NONP_P_HMGCR_S100b', panel='Figure 5E', protein='HMGCR',
         cell_type='Astrocytes', marker='S100β', region='Cortex',
         value_row='normalized ASTRO HMGCR', area_row='astrocyte area',
         coloc_row='coloc',
         measurement='HMGCR signal within S100β-positive objects, per unit '
                     'S100β-positive area'),
]
IF_CONDITIONS = [('Nulliparous', 'Non Pregnant'), ('Pregnant', 'Pregnant')]


def _tidy(s):
    """Collapse whitespace; the workbook uses non-breaking spaces."""
    return re.sub(r'\s+', ' ', str(s).replace('\xa0', ' ')).strip()


def read_if_sheet(sheet):
    """One row per image, animal and measured quantity.

    The two condition blocks are found by their titles, and each block is read
    down its own label column: a row whose cells right of that column are all
    text names the images of the next animal, and the numeric rows beneath
    carry that animal's measurements. The block must be bounded on the right
    by the summary column the workbook keeps there, or the first animal of the
    pregnant block is read as part of the summary instead.
    """
    d = pd.read_excel(IF_XLSX, sheet_name=sheet, header=None)
    title = next(i for i in range(6)
                 if len([j for j, v in d.iloc[i].items()
                         if isinstance(v, str)
                         and re.match(r'^(NONP|P)_', v)]) == 2)
    left, right = [j for j, v in d.iloc[title].items()
                   if isinstance(v, str) and re.match(r'^(NONP|P)_', v)]
    summary = [j for i in range(8) for j, v in d.iloc[i].items()
               if isinstance(v, str)
               and _tidy(v) in [b for _, b in IF_CONDITIONS]]
    end = min(summary) if summary else 10 ** 6
    rows = []
    for (cond, _), col, stop in zip(IF_CONDITIONS, (left, right),
                                    (right, end)):
        animal, images = None, {}
        for i in range(title, len(d)):
            cells = {j: v for j, v in d.iloc[i].items()
                     if col < j < stop and pd.notna(v)}
            text = {j: v for j, v in cells.items() if isinstance(v, str)}
            if text and len(text) == len(cells):
                label = d.iat[i, col]
                m = (re.match(IF_ANIMAL, _tidy(label))
                     if isinstance(label, str) else None)
                for v in text.values():
                    if m:
                        break
                    m = re.match(IF_ANIMAL, _tidy(v))
                if m:
                    animal = (m.group(1) + m.group(2)).upper()
                images = text
                continue
            label = d.iat[i, col]
            if not images or not isinstance(label, str):
                continue
            for j, name in images.items():
                v = cells.get(j)
                if isinstance(v, (int, float, np.floating)):
                    rows.append({'Condition': cond, 'Animal': animal,
                                 'Image': _tidy(name),
                                 'measure': _tidy(label), 'value': float(v)})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def if_measurements():
    """Every image of the four validations, one row each."""
    out = []
    for v in VALIDATIONS:
        d = read_if_sheet(v['sheet'])
        wide = d.pivot_table(index=['Condition', 'Animal', 'Image'],
                             columns='measure', values='value',
                             aggfunc='first').reset_index()
        got = {c: wide[c] if c in wide else np.nan
               for c in (v['area_row'], v['coloc_row'], v['value_row'])
               if c is not None}
        out.append(pd.DataFrame({
            'Figure Panel': v['panel'],
            'Target Protein': v['protein'],
            'Cell Type': v['cell_type'],
            'Cell Type Marker': v['marker'],
            'Region': v['region'],
            'Condition': wide['Condition'],
            'Animal': wide['Animal'],
            'Image': wide['Image'],
            'Marker-Positive Area': got.get(v['area_row'], np.nan),
            'Colocalized Target Area': got.get(v['coloc_row'], np.nan),
            'Measurement': v['measurement'],
            'Quantified Value': got[v['value_row']]}))
    df = pd.concat(out, ignore_index=True)
    df['Condition'] = pd.Categorical(
        df['Condition'], [c for c, _ in IF_CONDITIONS], ordered=True)
    return df.sort_values(['Figure Panel', 'Target Protein', 'Cell Type',
                           'Condition', 'Animal', 'Image'],
                          ignore_index=True)


def if_animal_sheet():
    """The per-animal means the figures plot, one row per biological
    replicate."""
    g = (if_measurements()
         .groupby(['Figure Panel', 'Target Protein', 'Cell Type',
                   'Cell Type Marker', 'Region', 'Measurement', 'Condition',
                   'Animal'], observed=True, dropna=False)['Quantified Value']
         .agg(['size', 'mean']).reset_index()
         .rename(columns={'size': 'Images', 'mean': 'Mean Quantified Value'}))
    return round_sig(g, digits=6)


def if_statistics_sheet():
    from scipy import stats
    keys = ['Figure Panel', 'Target Protein', 'Cell Type', 'Cell Type Marker',
            'Region', 'Measurement']
    rows = []
    a = if_animal_sheet()
    for k, sub in a.groupby(keys, observed=True, dropna=False, sort=False):
        g = {c: sub.loc[sub['Condition'] == c, 'Mean Quantified Value']
             .to_numpy() for c, _ in IF_CONDITIONS}
        n0, n1 = len(g['Nulliparous']), len(g['Pregnant'])
        v0, v1 = g['Nulliparous'].var(ddof=1), g['Pregnant'].var(ddof=1)
        t, p = stats.ttest_ind(g['Pregnant'], g['Nulliparous'],
                               equal_var=False)
        df = (v0 / n0 + v1 / n1) ** 2 / (
            (v0 / n0) ** 2 / (n0 - 1) + (v1 / n1) ** 2 / (n1 - 1))
        rows.append(dict(zip(keys, k)) | {
            'Animals (Nulliparous)': n0, 'Animals (Pregnant)': n1,
            'Mean (Nulliparous)': g['Nulliparous'].mean(),
            'SEM (Nulliparous)': g['Nulliparous'].std(ddof=1) / np.sqrt(n0),
            'Mean (Pregnant)': g['Pregnant'].mean(),
            'SEM (Pregnant)': g['Pregnant'].std(ddof=1) / np.sqrt(n1),
            'Fold Change (Pregnant / Nulliparous)':
                g['Pregnant'].mean() / g['Nulliparous'].mean(),
            't': t, 'Degrees of Freedom': df, 'P Value': p,
            'Test': 'Unpaired two-sided Welch t-test on per-animal means'})
    return round_sig(pd.DataFrame(rows), digits=6)


DESCRIPTION_6 = [
    ('All sheets', '',
     'Immunofluorescence validation of four protein changes, the source data '
     'for Figures 3E and 5E. Nulliparous and pregnant mice, seven per group. '
     'APOE and HMGCR were quantified inside a cell-type marker; FKBP5 was '
     'quantified across the striatum without a marker. Two further targets in '
     'the source quantification, APOE in astrocytes and MERTK, are not shown '
     'in any figure and are not reported here.'),
    ('All sheets', 'Cell Type Marker',
     'The marker whose segmented objects define the compartment the target '
     'protein was measured in. Empty for FKBP5, which was measured across the '
     'region.'),
    ('All sheets', 'Measurement',
     'What the quantified value is, for that validation.'),
    ('All sheets', 'Animal',
     'The mouse. NONP1 to NONP7 are nulliparous, P1 to P7 pregnant.'),

    ('Per-Image Measurements', '',
     'One row per image. Three to six images were quantified per mouse. Areas '
     'and signal are in the arbitrary units of the segmentation output, so '
     'they are comparable within a validation but not between validations.'),
    ('Per-Image Measurements', 'Marker-Positive Area',
     'Area of the segmented marker-positive objects in that image. Empty for '
     'FKBP5.'),
    ('Per-Image Measurements', 'Colocalized Target Area',
     'Target-protein signal overlapping those objects. Empty for FKBP5.'),
    ('Per-Image Measurements', 'Quantified Value',
     'The colocalized area divided by the marker-positive area, or, for '
     'FKBP5, the FKBP5-positive area itself. This is the quantity averaged '
     'per mouse and plotted.'),

    ('Per-Animal Summary', '',
     'The per-mouse means that are the biological replicates of the test and '
     'the points drawn on Figures 3E and 5E, recomputed here from every image '
     'of that mouse.'),
    ('Per-Animal Summary', 'Images',
     'Images contributing to the mean. For one pregnant mouse of the HMGCR '
     'validation the summary column of the source quantification averages '
     'four of that mouse\'s five images; the mean reported here uses all '
     'five, as for every other mouse.'),

    ('Statistical Tests', '',
     'One row per validation, comparing the per-mouse means of the two '
     'groups with an unpaired two-sided Welch t-test, as reported in the '
     'Results.'),
    ('Statistical Tests', 'Degrees of Freedom',
     'Welch-Satterthwaite approximation, so not an integer.'),
]


def build_dataset_6():
    return [
        ('Description', pd.DataFrame(
            DESCRIPTION_6, columns=['Sheet', 'Column', 'Definition'])),
        ('Per-Image Measurements', round_sig(if_measurements(), digits=6)),
        ('Per-Animal Summary', if_animal_sheet()),
        ('Statistical Tests', if_statistics_sheet()),
    ]

#endregion

#region Supplementary Data 7 ###################################################
# The CD31 vascular morphometry, from the workbook 23_vascular_stats.py writes.
# That workbook was built before the others and carries a prose README and a
# 25-column statistics sheet; this restates it in the format of the rest,
# with the README as a Description sheet and the statistics reduced to the
# columns a null result rests on: the difference and its interval, the effect
# size, the three tests, the correction, and the change the design could have
# detected.

VASC_XLSX = (f'{working_dir}/output/vascular_imaging_20260720/'
             'Supplementary_Data_vascular.xlsx')
VASC_KEEP = [
    ('Measure', 'Measure'), ('Family', 'Family'),
    ('Nulliparous mean', 'Nulliparous Mean'),
    ('Nulliparous SD', 'Nulliparous SD'),
    ('Pregnant mean', 'Pregnant Mean'),
    ('Pregnant SD', 'Pregnant SD'),
    ('Difference', 'Difference'),
    ('Change (%)', 'Change (%)'),
    ('Difference 95% CI low', 'Difference 95% CI Lower'),
    ('Difference 95% CI high', 'Difference 95% CI Upper'),
    ("Hedges' g", "Hedges' g"),
    ('p (Welch t-test, animal means)', 'P Value (Welch)'),
    ('p (exact permutation, 126 assignments)', 'P Value (Permutation)'),
    ('p (mixed model, 28 fields)', 'P Value (Mixed Model)'),
    ('FDR (Benjamini-Hochberg, 18 measures)', 'FDR'),
    ('Minimum detectable difference (%)', 'Minimum Detectable Change (%)'),
    ('Between-animal ICC', 'Between-Animal ICC'),
]


def vascular_sheet(name):
    return pd.read_excel(VASC_XLSX, sheet_name=name)


def vascular_measures_sheet():
    d = vascular_sheet('Statistics')
    out = d[[c for c, _ in VASC_KEEP]].rename(columns=dict(VASC_KEEP))
    out[SIG_COL] = yesno(out['FDR'] <= PROX_FDR)
    return round_sig(out)


DESCRIPTION_7 = [
    ('All sheets', '',
     'Quantitative morphometry of the CD31-stained vascular bed in the '
     'preoptic area, the source data for Figure 4F and Supplementary Figure '
     '5. Four nulliparous and five pregnant mice, 12 and 16 imaged fields, an '
     'independent cohort from the transcriptomic experiments. The animal is '
     'the unit of analysis: fields within an animal are technical replicates '
     'and are averaged before testing.'),
    ('All sheets', 'Measure',
     'One of the 18 morphometric measures, spanning vessel density, network '
     'architecture, caliber, tissue supply and cellularity.'),
    ('All sheets', 'Family',
     'The group of related measures: Density, Architecture, Caliber, Supply '
     'or Cellularity.'),

    ('Measures', '',
     'The primary comparison, one row per measure. Group means over animals '
     'with their standard deviations, the difference and its 95% confidence '
     'interval, the standardised effect size, and three tests of the same '
     'comparison. No measure differs between the groups, and the confidence '
     'intervals and the minimum detectable change say how large a difference '
     'the design could have found.'),
    ('Measures', 'Difference',
     'Pregnant minus nulliparous, in the units of the measure.'),
    ('Measures', "Hedges' g",
     'Standardised difference between the group means, bias-corrected for '
     'the small sample.'),
    ('Measures', 'P Value (Welch)',
     'Unpaired two-sided Welch t-test on the per-animal means; the primary '
     'test.'),
    ('Measures', 'P Value (Permutation)',
     'Exact permutation test over all 126 assignments of the nine animals to '
     'the two groups, free of any distributional assumption.'),
    ('Measures', 'P Value (Mixed Model)',
     'Linear mixed model over all 28 fields with the animal as a random '
     'effect, using the fields rather than averaging them.'),
    ('Measures', 'FDR',
     'Benjamini-Hochberg adjusted P value across the 18 measures. The '
     'measures are correlated, so this is conservative; the effective number '
     'of independent tests is 6.0 by the Li and Ji eigenvalue method.'),
    ('Measures', 'Minimum Detectable Change (%)',
     'The smallest difference this cohort could have detected at 80% power '
     'and a two-sided alpha of 0.05, as a percentage of the nulliparous '
     'mean. A null result is only as strong as this number is small.'),
    ('Measures', 'Between-Animal ICC',
     'Share of the total variance lying between animals rather than between '
     'fields of the same animal. High values are why fields are averaged '
     'before testing.'),

    ('Per-Animal Values', '',
     'The per-animal means that the tests were run on, one row per mouse.'),
    ('Per-Animal Values', 'Fields',
     'Imaged fields contributing to that animal\'s means.'),

    ('Per-Field Values', '',
     'One row per imaged field, with the areas each measure was computed in '
     'and the agreement between the manual vessel mask and a fully automatic '
     'threshold. Measures are computed inside the guarded core, the analysed '
     'domain eroded from its border so that no measure is affected by the '
     'edge of the field.'),
    ('Per-Field Values', 'Manual vs automatic Dice',
     'Overlap between the manual vessel mask and an automatic Li threshold '
     'of the same field, from 0 to 1. Mean 0.93 across the 28 fields.'),

    ('Robustness', '',
     'Every analytic choice varied in turn, one row per variant: the '
     'denominator area, the despeckling threshold, the segmentation, the '
     'domain and threshold rule, geometry at matched vessel area, dropping '
     'each animal in turn, using the field rather than the animal as the '
     'unit, and the balance of covariates between the groups. The direction '
     'and the absence of a difference hold throughout.'),
    ('Robustness', 'Block', 'The choice being varied.'),
    ('Robustness', 'Variant', 'The setting used, within that choice.'),

    ('Parameters', '',
     'The image scale and every threshold of the image analysis, so the '
     'measurements can be recomputed from the images.'),
]


def build_dataset_7():
    return [
        ('Description', pd.DataFrame(
            DESCRIPTION_7, columns=['Sheet', 'Column', 'Definition'])),
        ('Measures', vascular_measures_sheet()),
        ('Per-Animal Values', round_sig(vascular_sheet('Animals'), digits=6)),
        ('Per-Field Values', round_sig(vascular_sheet('Fields'), digits=6)),
        ('Robustness', round_sig(vascular_sheet('Robustness'))),
        ('Parameters', vascular_sheet('Parameters')),
    ]

#endregion

#region Supplementary Data 8 ###################################################
# gsMap heritability enrichment, as computed by 09_gsmap.py. Everything the
# analysis produced fits in one workbook: the per-condition enrichment, the
# per-platform differences between reproductive states, their cross-platform
# meta-analysis and the per-sample scores all of it rests on.
#
# GWAS provenance is fixed fact supplied with the study and is held here.
# Obvious typographical errors in the source list were corrected; see the
# comment beside each entry.

GSMAP_DIR = f'{working_dir}/output/gsmap'
FIGURE_6_SCRIPT = f'{working_dir}/16_figure_6.py'
GSMAP_SIG_COL = 'Significant (Empirical P < 0.05)'

# abbreviation (the key used throughout the analysis), trait, study, year,
# ancestry, sample size
GWAS_TRAITS = [
    ('AD', 'Alzheimer dementia',
     'Bellenguez et al. Nat Genet 2022', 2022, 'European', 486482),
    ('ADHD', 'Attention deficit hyperactivity disorder',
     'Demontis et al. Nat Genet 2023', 2023, 'European', 225534),
    ('ALS', 'Amyotrophic lateral sclerosis',
     'van Rheenen et al. Nat Genet 2021', 2021, 'Mixed', 80082),
    ('Anxiety', 'Anxiety (GAD symptoms)',
     'Levey et al. Am J Psychiatry 2020', 2020, 'Mixed', 199611),
    ('Autism', 'Autism spectrum disorder',
     'Grove et al. Nat Genet 2019', 2019, 'European', 46351),
    ('BPD', 'Borderline personality disorder',
     'Witt et al. medRxiv 2024', 2024, 'European', 1054056),
    ('Bipolar', 'Bipolar disorder',
     'Mullins et al. Nat Genet 2021', 2021, 'European', 99551),
    ('MDD', 'Major depressive disorder',
     'Howard et al. Nat Neurosci 2019', 2019, 'European', 500199),
    ('Neuroticism', 'Neuroticism',
     'Nagel et al. Nat Genet 2018', 2018, 'European', 375458),
    ('OCD', 'Obsessive compulsive disorder',
     'Strom et al. Nat Genet 2025', 2024, 'European', 2098077),
    ('PD', "Parkinson's disorder",
     'Kim et al. Nat Genet 2024', 2024, 'Mixed', 2525897),
    ('PPD', 'Postpartum depression', '', None, '', 48000),
    ('PTSD', 'Post-traumatic stress disorder',
     'Nievergelt et al. Nat Commun 2019', 2019, 'Mixed', 69279),
    ('SCZ', 'Schizophrenia',
     'Trubetskoy et al. Nature 2022', 2022, 'Mixed', 69856),
    ('Stroke', 'Stroke',
     'Malik et al. Nat Genet 2018', 2018, 'European', 446696),
    ('Suicide', 'Suicide attempt',
     'Docherty et al. Am J Psychiatry 2023', 2023, 'Mixed', 958896),
]


@lru_cache(maxsize=1)
def figure_6d_cell_types():
    """The exemplar subclasses of Figure 6D, out of the figure's own source."""
    import ast
    tree = ast.parse(open(FIGURE_6_SCRIPT).read())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(getattr(t, 'id', None) == 'EXEMPLAR_CELLTYPES'
                        for t in n.targets))
    ns = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 FIGURE_6_SCRIPT, 'exec'), ns)
    return {(trait, ct) for trait, cts in ns['EXEMPLAR_CELLTYPES'].items()
            for ct in cts}


def _cell_cols(frame, col='cell_type'):
    """Neighborhood and study-set columns for a cell-type column."""
    hood, keep = subclass_neighborhood(), study_subclasses()
    ct = frame[col].to_list()
    return ([hood.get(c, '') for c in ct], yesno([c in keep for c in ct]))


def gwas_traits_sheet():
    df = pd.DataFrame(
        [{'Trait': t, 'Abbreviation': a, 'Study': s, 'Year': y,
          'Ancestry': anc, 'Sample Size': n, 'Access': 'Public'}
         for a, t, s, y, anc, n in GWAS_TRAITS]).sort_values('Trait')
    # nullable, so the one trait without a published year stays blank rather
    # than turning the column into floats
    df['Year'] = df['Year'].astype('Int64')
    return df.reset_index(drop=True)


def trait_enrichment_sheet():
    import polars as pl
    c = pl.read_csv(f'{GSMAP_DIR}/cauchy_fdr.csv')
    hood, study = _cell_cols(c)
    out = pd.DataFrame({
        'Platform': [PLATFORM_LABELS[d] for d in c['dataset']],
        'Condition': [CONDITION[x] for x in c['condition']],
        'Trait': c['trait'].to_list(),
        'Cell Type': c['cell_type'].to_list(),
        'Cell Type Neighborhood': hood,
        'In Study Set': study,
        'P Value (Cauchy)': c['p_cauchy'].to_list(),
        'P Value (Median)': c['p_median'].to_list(),
        'FDR': c['fdr_cauchy'].to_list()})
    out[SIG_COL] = yesno(out['FDR'] <= PROX_FDR)
    out['_sort'] = [subclass_sort_key(x) for x in c['cell_type']]
    out['Platform'] = pd.Categorical(out['Platform'], PLATFORM_ORDER,
                                     ordered=True)
    out = out.sort_values(['Platform', 'Condition', 'Trait',
                           'P Value (Cauchy)']).reset_index(drop=True)
    return round_sig(out.drop(columns='_sort'))


def state_difference_sheet():
    import polars as pl
    p = pl.read_csv(f'{GSMAP_DIR}/per_dataset.csv')
    hood, study = _cell_cols(p)
    out = pd.DataFrame({
        'Platform': [PLATFORM_LABELS[d] for d in p['dataset']],
        'Contrast': [CONTRAST_LABELS[c] for c in p['contrast']],
        'Trait': p['trait'].to_list(),
        'Cell Type': p['cell_type'].to_list(),
        'Cell Type Neighborhood': hood,
        'In Study Set': study,
        'Difference in Association': p['beta'].to_list(),
        't': p['t'].to_list(),
        'P Value': p['p'].to_list(),
        'FDR': p['fdr'].to_list(),
        'Samples (Treatment)': p['n_treat'].to_list(),
        'Samples (Reference)': p['n_base'].to_list()})
    out[SIG_COL] = yesno(out['FDR'] <= PROX_FDR)
    return round_sig(_ordered(out, ['Platform', 'Contrast', 'Trait',
                                    'P Value']))


def gsmap_meta_sheet():
    import polars as pl
    m = pl.read_csv(f'{GSMAP_DIR}/meta.csv')
    hood, study = _cell_cols(m)
    up = m['nlp_up'].to_numpy() >= m['nlp_down'].to_numpy()
    emp = np.where(up, m['emp_p_up'].to_numpy(), m['emp_p_down'].to_numpy())
    shown = figure_6d_cell_types()
    out = pd.DataFrame({
        'Contrast': [CONTRAST_LABELS[c] for c in m['contrast']],
        'Trait': m['trait'].to_list(),
        'Cell Type': m['cell_type'].to_list(),
        'Cell Type Neighborhood': hood,
        'In Study Set': study,
        'Platforms': m['D'].to_list(),
        'Sum of Normalized Ranks': m['sum_stat'].to_list(),
        'Direction': np.where(up, 'Strengthened', 'Weakened'),
        'Slide-tags Difference': m['beta_slidetags'].to_list(),
        'Xenium Difference': m['beta_xenium'].to_list(),
        'Empirical P (Strengthened)': m['emp_p_up'].to_list(),
        'Empirical P (Weakened)': m['emp_p_down'].to_list(),
        'Empirical FDR (Strengthened)': m['emp_fdr_up'].to_list(),
        'Empirical FDR (Weakened)': m['emp_fdr_down'].to_list()})
    out[GSMAP_SIG_COL] = yesno(emp < EMP_P)
    out['Shown in Figure 6D'] = yesno(
        [(t, c) in shown for t, c in zip(m['trait'], m['cell_type'])])
    out['_p'] = emp
    return round_sig(_ordered(out, ['Contrast', 'Trait', '_p'])
                     .drop(columns='_p'))


def gsmap_sample_sheet():
    import polars as pl
    s = pl.read_parquet(f'{GSMAP_DIR}/sample_means.parquet')
    hood, study = _cell_cols(s)
    out = pd.DataFrame({
        'Platform': [PLATFORM_LABELS[d] for d in s['dataset']],
        'Trait': s['trait'].to_list(),
        'Cell Type': s['cell_type'].to_list(),
        'Cell Type Neighborhood': hood,
        'In Study Set': study,
        'Sample ID': s['sample'].to_list(),
        'Condition': [CONDITION[c] for c in s['condition']],
        'Cells': s['n_cells'].to_list(),
        'Mean Association Score': s['mean_score'].to_list()})
    out['Platform'] = pd.Categorical(out['Platform'], PLATFORM_ORDER,
                                     ordered=True)
    out = out.sort_values(
        ['Platform', 'Trait', 'Cell Type', 'Condition', 'Sample ID'],
        key=lambda c: (c.map(subclass_sort_key) if c.name == 'Cell Type'
                       else c.map(sample_sort_key) if c.name == 'Sample ID'
                       else c)).reset_index(drop=True)
    return round_sig(out)


DESCRIPTION_8 = [
    ('All sheets', '',
     'Heritability enrichment of 16 human neurological and psychiatric traits '
     'across the maternal brain, the source data for Figure 6. gsMap assigns '
     'every cell a spatially aware gene specificity score, maps it to nearby '
     'SNPs and tests each cell for heritability enrichment by stratified LD '
     'score regression; per-cell association scores were then summarised to '
     'cell subclasses. Run on Slide-tags and Xenium, separately for each '
     'reproductive state.'),
    ('All sheets', 'Trait',
     'Abbreviation of the GWAS trait, as defined in the GWAS Traits sheet.'),
    ('All sheets', 'Cell Type',
     'Subclass of the Allen whole-brain atlas taxonomy, with its numeric '
     'code.'),
    ('All sheets', 'In Study Set', STUDY_SET_DEF),

    ('GWAS Traits', '',
     'The genome-wide association study behind each trait. Sample sizes are '
     'as reported by the source study.'),
    ('GWAS Traits', 'Abbreviation',
     'The key used for the trait in every other sheet.'),

    ('Trait Enrichment', '',
     'Enrichment of each trait in each cell subclass, within one platform and '
     'one reproductive state. Per-cell association P values were combined to '
     'the subclass by the Cauchy combination test. One row per platform, '
     'state, trait and subclass, ordered by P value.'),
    ('Trait Enrichment', 'P Value (Cauchy)',
     'Cauchy combination of the per-cell P values of that subclass; the '
     'primary enrichment statistic, and what Figure 6A and 6B display.'),
    ('Trait Enrichment', 'P Value (Median)',
     'Median per-cell P value of the subclass, reported alongside as a '
     'robustness check on the combination.'),
    ('Trait Enrichment', 'FDR',
     'Benjamini-Hochberg adjusted P value across all trait and subclass pairs '
     'within a platform and state.'),

    ('State Differences', '',
     'Change in a trait\'s enrichment between two reproductive states, one '
     'platform at a time. Per-cell association scores were averaged within '
     'each sample and subclass, requiring at least 20 cells, and the sample '
     'means of the two states compared by Welch t-test. One row per platform, '
     'contrast, trait and subclass.'),
    ('State Differences', 'Difference in Association',
     'Treatment minus reference, in mean -log10 association P per cell. '
     'Positive means the subclass carried more of the trait\'s heritability '
     'in the treatment state.'),
    ('State Differences', 'Samples (Treatment)',
     'Animals contributing a mean for that subclass, which is two or three '
     'per state.'),

    ('Cross-Platform Meta-analysis', '',
     'Slide-tags and Xenium combined for the pregnant versus nulliparous '
     'contrast by sum of ranks, calibrated against label permutations, the '
     'same procedure used for genes and pathways. Only the pregnancy contrast '
     'has two platforms; the postpartum contrasts are reported per platform '
     'in the previous sheet. Ordered by empirical P within a trait.'),
    ('Cross-Platform Meta-analysis', 'Direction',
     'Whether the association strengthened or weakened in pregnancy, taken as '
     'the stronger of the two tails.'),
    ('Cross-Platform Meta-analysis', 'Empirical P (Strengthened)',
     'Fraction of the permutation null at least as extreme, for a '
     'strengthened association on both platforms. One-sided.'),
    ('Cross-Platform Meta-analysis', 'Shown in Figure 6D',
     'The subclasses drawn as exemplars in that panel.'),

    ('Per-Sample Scores', '',
     'Source data for the tests above and for the trajectories of Figure 6D: '
     'the mean per-cell association score of every subclass in every sample.'),
    ('Per-Sample Scores', 'Mean Association Score',
     'Mean -log10 association P across the cells of that subclass in that '
     'sample.'),
    ('Per-Sample Scores', 'Cells',
     'Cells of that subclass in that sample. Sample and subclass pairs below '
     '20 cells were not carried into the tests.'),
]


def build_dataset_8():
    return [
        ('Description', pd.DataFrame(
            DESCRIPTION_8, columns=['Sheet', 'Column', 'Definition'])),
        ('GWAS Traits', gwas_traits_sheet()),
        ('Trait Enrichment', trait_enrichment_sheet()),
        ('State Differences', state_difference_sheet()),
        ('Cross-Platform Meta-analysis', gsmap_meta_sheet()),
        ('Per-Sample Scores', gsmap_sample_sheet()),
    ]

#endregion

#region deposit and run ########################################################


def write_deposit():
    """The unrestricted per-platform tables, for the repository deposit."""
    os.makedirs(DEPOSIT, exist_ok=True)
    de, gsea = de_sheet(restrict=False), pathway_platform_sheet(restrict=False)
    sig = signalling_deposit()
    for name, df in (('differential_expression_all_platforms', de),
                     ('pathway_enrichment_all_platforms', gsea),
                     ('intercellular_signalling_all_platforms', sig)):
        path = f'{DEPOSIT}/{name}.csv.gz'
        df.to_csv(path, index=False, compression='gzip')
        print(f'wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB, '
              f'{len(df):,} rows)')
    with open(f'{DEPOSIT}/README.txt', 'w') as f:
        f.write(DEPOSIT_README.format(n_de=len(de), n_gsea=len(gsea),
                                      n_sig=len(sig)))
    print(f'wrote {DEPOSIT}/README.txt')


DATASETS = {1: build_dataset_1, 2: build_dataset_2,
            3: build_dataset_3, 4: build_dataset_4,
            5: build_dataset_5, 6: build_dataset_6,
            7: build_dataset_7, 8: build_dataset_8}

#endregion

#region run ####################################################################

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', type=int, choices=sorted(DATASETS),
                    default=None, help='build one dataset (default: all)')
    ap.add_argument('--deposit', action='store_true',
                    help='write the unrestricted per-platform tables for the '
                         'repository (implied by a full run)')
    args = ap.parse_args()

    for n in ([args.dataset] if args.dataset else sorted(DATASETS)):
        sheets = DATASETS[n]()
        write_workbook(f'{OUT}/Supplementary_Data_{n}.xlsx', sheets,
                       wide_cols=('Notes', 'Definition'))
    if args.deposit or args.dataset is None:
        write_deposit()

#endregion
