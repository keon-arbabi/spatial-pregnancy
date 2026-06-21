"""READ-ONLY audit-data builder for figure 5 (lipid). Produces compact CSV/JSON
slices in output/audit_fig5/ that downstream reviewers reason over. Does NOT
modify the figure. Mirrors 12_figure_helper conventions for direction/D/emp_p.
"""
import os, re, json, importlib
import numpy as np
import polars as pl

fc = importlib.import_module('12_figure_helper')
wd = '/home/karbabi/spatial-pregnancy'
OUT = f'{wd}/output/audit_fig5'
os.makedirs(OUT, exist_ok=True)
CONTRAST = 'PREG_vs_CTRL'

# ---------------------------------------------------------------- figure config
PATHWAY_BANDS = [
    ('Membrane lipid', ['GOBP_MEMBRANE_LIPID_METABOLIC_PROCESS',
        'GOBP_MEMBRANE_LIPID_BIOSYNTHETIC_PROCESS', 'GOBP_LIPID_TRANSLOCATION']),
    ('Sphingolipid', ['GOBP_SPHINGOLIPID_METABOLIC_PROCESS',
        'GOBP_CERAMIDE_METABOLIC_PROCESS']),
    ('Fatty acid catabolism', ['GOBP_FATTY_ACID_CATABOLIC_PROCESS',
        'GOBP_FATTY_ACID_BETA_OXIDATION']),
    ('Cholesterol & carriers', ['GOBP_REGULATION_OF_LIPID_LOCALIZATION',
        'GOBP_LIPOPROTEIN_METABOLIC_PROCESS', 'GOBP_STEROID_BIOSYNTHETIC_PROCESS']),
]
ordered_pathways = [p for _, ps in PATHWAY_BANDS for p in ps]
pathway_band = {p: b for b, ps in PATHWAY_BANDS for p in ps}
GENE_BANDS = [
    ('Membrane lipid', ['Lpin1','Tecr','Agpat4','Dgat1','Pisd']),
    ('Sphingolipid', ['Cers4','Cers5','Cers6','St6galnac5','Hexa','Glb1']),
    ('Fatty acid catabolism', ['Cpt1a','Ivd','Echs1','Acaa2','Decr1','Acox1','Hacl1']),
    ('Cholesterol & carriers', ['Apoe','Clu','Fabp7','Hmgcs1','Hmgcr','Idi1','Sqle',
        'Pcsk9','Sort1','Sorl1','Cd81','Abca1','Lpl','Srebf1','Mfsd2a']),
]
ordered_genes = [g for _, gs in GENE_BANDS for g in gs]
gene_band = {g: b for b, gs in GENE_BANDS for g in gs}
ct_allowlist = {'334 Microglia NN','327 Oligo NN','323 Ependymal NN',
                '333 Endo NN','119 SI-MA-LPO-LHA Skor1 Glut'}
CORTICAL_GABA = ['Pvalb','Sst','Vip','Lamp5','Sncg','Pax6']
def assign_class(ct):
    if 'NN' in ct: return 'Non-neuronal'
    if 'Glut' in ct: return 'Glutamatergic'
    if any(t in ct for t in CORTICAL_GABA): return 'GABAergic Cortex'
    return 'GABAergic Subcortex'
CLASS_ORDER = ['Glutamatergic','GABAergic Cortex','GABAergic Subcortex','Non-neuronal']
CARD_GENES = ['Lpin1','Glb1','Ivd','Apoe','Hmgcr','Idi1','Mfsd2a']
card_ctx = {'Apoe': ['326 OPC NN','334 Microglia NN']}
max_rows_map = {'Apoe': 5}
CHORD_THEME_LIGANDS = {'Apoe': ['Apoe','Lpl'], 'Pcsk9': ['Pcsk9'],
                       'Pltp': ['Pltp'], 'Reln': ['Reln']}
CANONICAL_LR_PAIRS = set()
for _r in ['Lrp1','Lrp2','Lrp4','Lrp8','Vldlr','Ldlr','Sort1','Sorl1','Trem2','Abca1']:
    CANONICAL_LR_PAIRS.add(('Apoe', _r))
CANONICAL_LR_PAIRS.add(('Lpl','Lrp1'))
for _r in ['Lrp1','Lrp2']:
    CANONICAL_LR_PAIRS.add(('Clu', _r)); CANONICAL_LR_PAIRS.add(('Apod', _r))
CANONICAL_LR_PAIRS.add(('Reln','Vldlr')); CANONICAL_LR_PAIRS.add(('Reln','Lrp8'))
for _r in ['Ldlr','Lrp1','Lrp8','Sort1','Vldlr','Cd81','Aplp2']:
    CANONICAL_LR_PAIRS.add(('Pcsk9', _r))
CANONICAL_LR_PAIRS.add(('Pltp','Abca1'))
for _l in ['Sphk1','Sphk2']:
    for _r in ['S1pr1','S1pr2','S1pr3','S1pr4','S1pr5']:
        CANONICAL_LR_PAIRS.add((_l, _r))
for _r in ['Slc22a17','Lrp2']:
    CANONICAL_LR_PAIRS.add(('Lcn2', _r))
CANONICAL_LR_PAIRS.add(('A2m','Lrp1'))

LIPID_RX = ('LIPID|CHOLESTEROL|STEROL|STEROID|SPHINGO|CERAMIDE|GLYCERO|PHOSPHOLIPID|'
    'PHOSPHATIDYL|FATTY_ACID|ACYL|TRIGLYCERIDE|LIPOPROTEIN|ADIPO|SQUALENE|ISOPRENOID|'
    'MEVALONATE|GANGLIOSIDE|GLYCOLIPID|GLYCEROLIPID|ARACHIDON|EICOSANOID|PROSTAGLANDIN|'
    'LEUKOTRIENE|OXYSTEROL|BILE_ACID|APOLIPOPROTEIN|LECITHIN|CARDIOLIPIN|SPHINGOMYELIN|'
    'CERAMIDASE|MYELIN|INOSITOL_PHOSPHATE|FAT_CELL|LDL_|_LDL|HDL_|VLDL|CHYLOMICRON|'
    'STEROL_|_FAT_|MONOACYLGLYCEROL|DIACYLGLYCEROL|GLUCOSYLCERAMIDE|GALACTOSYLCERAMIDE')

# ------------------------------------------------------------------- load data
gsea_all, gsea, real_nes = fc.load_gsea(wd)
de_sr_all, de_sr, de_pp = fc.load_de(wd)
ordered_cts, ct_class = fc.select_cell_types(
    gsea, ordered_pathways, ct_allowlist, assign_class, CLASS_ORDER, min_hits=2)
col_set = set(ordered_cts)
print(f'displayed columns ({len(ordered_cts)}):')
for c in ordered_cts: print(f'  [{ct_class[c]}] {c}')

# direction helpers (sumrank up/down + median NES/logFC sign)
def dir_sumrank(r):
    return 'up' if (r['nlp_up'] or 0) >= (r['nlp_down'] or 0) else 'down'

# all cell types present in data
all_cts = sorted(set(gsea_all['cell_type'].unique().to_list()))

# lipid candidate pathways (regex) -- defined early so NES covers them too
import re as _re
lip_paths = sorted({p for p in gsea_all['pathway'].unique().to_list()
                    if _re.search(LIPID_RX, p)})
_nes_universe = set(ordered_pathways) | set(lip_paths)
# median NES per (pathway,ct) over PREG real fgsea (ALL lipid + displayed paths)
nes_agg = (real_nes.filter(pl.col('pathway').is_in(list(_nes_universe)))
           .group_by(['pathway','cell_type']).agg(pl.col('NES').median().alias('nes')))
nes_look = {(r['pathway'], r['cell_type']): r['nes'] for r in nes_agg.iter_rows(named=True)}

# ============================================================ 1. PATHWAYS
# 1a. displayed pathways, full per-celltype detail (all cts)
sub = gsea_all.filter(pl.col('pathway').is_in(ordered_pathways))
# collapse to best (max nlp) row per (pathway,ct)
best = {}
for r in sub.iter_rows(named=True):
    k = (r['pathway'], r['cell_type'])
    nlp = max(r['nlp_up'] or 0, r['nlp_down'] or 0)
    if k not in best or nlp > best[k]['nlp']:
        ep = r['emp_p_up'] if (r['nlp_up'] or 0) >= (r['nlp_down'] or 0) else r['emp_p_down']
        best[k] = dict(pathway=r['pathway'], cell_type=r['cell_type'], nlp=nlp,
                       emp_p=ep, D=r['D'], dir_sr=dir_sumrank(r))
rows = []
for (p, ct), v in best.items():
    nes = nes_look.get((p, ct))
    rows.append(dict(band=pathway_band[p], **v,
                     nes=nes, dir_nes=('up' if nes is not None and nes > 0 else
                                       'down' if nes is not None else None),
                     in_cols=ct in col_set, sig=(v['emp_p'] is not None and v['emp_p'] <= 0.05)))
pl.DataFrame(rows).sort(['band','pathway','nlp'], descending=[False,False,True]) \
    .write_csv(f'{OUT}/path_displayed_detail.csv')

# 1b. per displayed pathway aggregate (over cols, and over all cts)
def is_sig(v): return v['emp_p'] is not None and v['emp_p'] <= 0.05
def nes_dir(p, ct):
    nv = nes_look.get((p, ct))
    return 'up' if (nv is not None and nv > 0) else 'down' if nv is not None else None
agg = []
for p in ordered_pathways:
    rs = [v for (pp, ct), v in best.items() if pp == p]
    rs_sig = [v for v in rs if is_sig(v)]
    rs_col = [v for v in rs if v['cell_type'] in col_set]
    rs_col_sig = [v for v in rs_col if is_sig(v)]
    up = sum(nes_dir(p, v['cell_type']) == 'up' for v in rs_col_sig)
    dn = sum(nes_dir(p, v['cell_type']) == 'down' for v in rs_col_sig)
    agg.append(dict(band=pathway_band[p], pathway=p,
        n_sig_all=len(rs_sig), n_sig_cols=len(rs_col_sig),
        peak_nlp_cols=max((v['nlp'] for v in rs_col), default=0.0),
        peak_nlp_all=max((v['nlp'] for v in rs_sig), default=0.0),
        col_sig_up=up, col_sig_down=dn,
        dir_dominant=('up' if up >= dn else 'down') if rs_col_sig else 'none'))
pl.DataFrame(agg).write_csv(f'{OUT}/path_displayed_agg.csv')

# 1c. lipid candidate pathways (regex), aggregate over cols
lip_paths = sorted({p for p in gsea_all['pathway'].unique().to_list() if re.search(LIPID_RX, p)})
gcol = gsea_all.filter(pl.col('cell_type').is_in(list(col_set)))
gcol_best = {}
for r in gcol.filter(pl.col('pathway').is_in(lip_paths)).iter_rows(named=True):
    k = (r['pathway'], r['cell_type']); nlp = max(r['nlp_up'] or 0, r['nlp_down'] or 0)
    if k not in gcol_best or nlp > gcol_best[k]['nlp']:
        ep = r['emp_p_up'] if (r['nlp_up'] or 0) >= (r['nlp_down'] or 0) else r['emp_p_down']
        gcol_best[k] = dict(nlp=nlp, emp_p=ep, D=r['D'])
cand = []
for p in lip_paths:
    rs = [(ct, v) for (pp, ct), v in gcol_best.items() if pp == p]
    sig = [(ct, v) for ct, v in rs if v['emp_p'] is not None and v['emp_p'] <= 0.05]
    nes_up = sum((nes_look.get((p, ct)) or 0) > 0 for ct, _ in sig)
    cand.append(dict(pathway=p, displayed=p in ordered_pathways,
        n_sig_cols=len(sig), peak_nlp_cols=max((v['nlp'] for _, v in rs), default=0.0),
        sig_cts=','.join(sorted(fc.numeric_prefix(ct).__str__().zfill(3) for ct, _ in sig)),
        col_sig_nes_up=nes_up, col_sig_nes_down=len(sig) - nes_up))
pl.DataFrame(cand).sort('n_sig_cols', descending=True).write_csv(f'{OUT}/path_candidates.csv')

# 1d. leading-edge gene union per displayed pathway (for redundancy + core-gene checks)
le_cols = ['leading_edge_merfish','leading_edge_slidetags','leading_edge_xenium']
le = gsea_all.filter(pl.col('pathway').is_in(ordered_pathways)
                     & pl.col('cell_type').is_in(list(col_set)))
le_map = {}
for r in le.iter_rows(named=True):
    s = le_map.setdefault(r['pathway'], {})
    for c in le_cols:
        for x in str(r[c] or '').split(','):
            if x: s[x] = s.get(x, 0) + 1
le_rows = [dict(pathway=p, n_le_genes=len(s),
                top_le=','.join(g for g, _ in sorted(s.items(), key=lambda kv: -kv[1])[:25]))
           for p, s in le_map.items()]
pl.DataFrame(le_rows).write_csv(f'{OUT}/path_leading_edge.csv')

# ============================================================ 2. GENES
de_best = {}  # (gene,ct) -> best sumrank row
for r in de_sr_all.filter(pl.col('gene').is_in(
        list(set(ordered_genes) | {l for l, _ in CANONICAL_LR_PAIRS} |
             {rr for _, rr in CANONICAL_LR_PAIRS}))).iter_rows(named=True):
    k = (r['gene'], r['cell_type']); nlp = max(r['nlp_up'] or 0, r['nlp_down'] or 0)
    if k not in de_best or nlp > de_best[k]['nlp']:
        ep = r['emp_p_up'] if (r['nlp_up'] or 0) >= (r['nlp_down'] or 0) else r['emp_p_down']
        de_best[k] = dict(nlp=nlp, emp_p=ep, D=r['D'], dir_sr=dir_sumrank(r),
                          ref_pct=r['ref_pct_detected'])
# median logFC + platforms from de_pp
lfc_agg = (de_pp.filter(pl.col('logFC').is_not_null())
           .group_by(['gene','cell_type']).agg(
               pl.col('logFC').median().alias('lfc'),
               pl.col('dataset').n_unique().alias('n_plat'),
               pl.col('dataset').sort().str.join(',').alias('plats')))
lfc_look = {(r['gene'], r['cell_type']): r for r in lfc_agg.iter_rows(named=True)}

g_rows = []
for g in ordered_genes:
    for ct in all_cts:
        b = de_best.get((g, ct)); l = lfc_look.get((g, ct))
        if b is None and l is None: continue
        lfc = l['lfc'] if l else None
        g_rows.append(dict(band=gene_band[g], gene=g, cell_type=ct, in_cols=ct in col_set,
            nlp=(b['nlp'] if b else None), emp_p=(b['emp_p'] if b else None),
            D=(b['D'] if b else None), dir_sr=(b['dir_sr'] if b else None),
            ref_pct=(b['ref_pct'] if b else None),
            lfc=lfc, dir_lfc=('up' if lfc is not None and lfc > 0 else 'down' if lfc is not None else None),
            n_plat=(l['n_plat'] if l else 0), plats=(l['plats'] if l else ''),
            sig=(b is not None and b['emp_p'] is not None and b['emp_p'] <= 0.05)))
pl.DataFrame(g_rows).write_csv(f'{OUT}/gene_displayed_detail.csv')

# 2b. displayed gene aggregate over columns
g_agg = []
for g in ordered_genes:
    rs = [r for r in g_rows if r['gene'] == g and r['in_cols']]
    sig = [r for r in rs if r['sig']]
    up = sum(r['dir_lfc'] == 'up' for r in sig); dn = sum(r['dir_lfc'] == 'down' for r in sig)
    g_agg.append(dict(band=gene_band[g], gene=g, n_sig_cols=len(sig),
        peak_nlp_cols=max((r['nlp'] or 0 for r in rs), default=0.0),
        max_D=max((r['D'] or 0 for r in rs), default=0),
        col_sig_lfc_up=up, col_sig_lfc_down=dn,
        dir_dominant=('up' if up >= dn else 'down') if sig else 'none',
        max_ref_pct=max((r['ref_pct'] or 0 for r in rs), default=0.0)))
pl.DataFrame(g_agg).write_csv(f'{OUT}/gene_displayed_agg.csv')

# 2c. lipid candidate genes from leading edges of displayed lipid pathways +
#     all-lipid-pathway LE; rank by sig-cell count over columns
le_pool = {}
le_all = gsea_all.filter(pl.col('pathway').is_in(lip_paths)
                         & pl.col('cell_type').is_in(list(col_set)))
for r in le_all.iter_rows(named=True):
    for c in le_cols:
        for x in str(r[c] or '').split(','):
            if x: le_pool[x] = le_pool.get(x, 0) + 1
cand_genes = sorted(set(le_pool) | set(ordered_genes))
de_best_all = {}
for r in de_sr.filter(pl.col('gene').is_in(cand_genes)
                      & pl.col('cell_type').is_in(list(col_set))).iter_rows(named=True):
    k = (r['gene'], r['cell_type']); nlp = max(r['nlp_up'] or 0, r['nlp_down'] or 0)
    if k not in de_best_all or nlp > de_best_all[k]['nlp']:
        de_best_all[k] = dict(nlp=nlp, dir=dir_sumrank(r))
cg = []
for g in cand_genes:
    rs = [(ct, v) for (gg, ct), v in de_best_all.items() if gg == g]
    if not rs: continue
    up = sum(v['dir'] == 'up' for _, v in rs); dn = len(rs) - up
    cg.append(dict(gene=g, displayed=g in ordered_genes, le_freq=le_pool.get(g, 0),
        n_sig_cols=len(rs), peak_nlp=max(v['nlp'] for _, v in rs),
        sig_up=up, sig_down=dn))
pl.DataFrame(cg).sort(['n_sig_cols','peak_nlp'], descending=True).write_csv(f'{OUT}/gene_candidates.csv')

# ============================================================ 3. CELL-TYPE RANKING
# find strong missing populations: per ct, sig displayed paths + sig lipid paths +
# sig lipid genes; flag whether displayed as a column
ct_rank = []
gall_best = {}
for r in gsea.filter(pl.col('pathway').is_in(lip_paths)).iter_rows(named=True):
    k = (r['pathway'], r['cell_type']); gall_best[k] = True
for ct in all_cts:
    n_disp_sig = sum(1 for (p, c), v in best.items() if c == ct and is_sig(v))
    n_lip_sig = sum(1 for (p, c) in gall_best if c == ct)
    n_lip_gene = de_sr.filter((pl.col('cell_type') == ct)
                              & pl.col('gene').is_in(cand_genes)).height
    ct_rank.append(dict(cell_type=ct, in_cols=ct in col_set,
        n_disp_path_sig=n_disp_sig, n_lipid_path_sig=n_lip_sig, n_lipid_gene_sig=n_lip_gene))
pl.DataFrame(ct_rank).sort(['n_lipid_path_sig','n_lipid_gene_sig'], descending=True) \
    .write_csv(f'{OUT}/celltype_lipid_rank.csv')

# ============================================================ 4. CCC / CHORD
ligs = sorted({p[0] for p in CANONICAL_LR_PAIRS})
recs = sorted({p[1] for p in CANONICAL_LR_PAIRS})
lr = (pl.scan_csv(f'{wd}/output/liana/inflow_diff.csv')
      .filter((pl.col('contrast') == CONTRAST)
              & pl.col('ligand_complex').is_in(ligs)
              & pl.col('receptor_complex').is_in(recs))
      .select(['dataset','source','target','ligand_complex','receptor_complex',
               'lr_mean_ctrl','lr_mean_treat','lr_mean_diff','n_sig_ctrl','n_sig_treat'])
      .collect(engine='streaming'))
lr = lr.filter(pl.struct(['ligand_complex','receptor_complex']).map_elements(
    lambda s: (s['ligand_complex'], s['receptor_complex']) in CANONICAL_LR_PAIRS,
    return_dtype=pl.Boolean))
lr.write_csv(f'{OUT}/ccc_raw_canonical.csv')

# per (source,target,ligand,receptor) cross-platform concordance (slidetags vs xenium)
keys = ['source','target','ligand_complex','receptor_complex']
wide = (lr.group_by(keys + ['dataset']).agg(pl.col('lr_mean_diff').first())
        .pivot(on='dataset', index=keys, values='lr_mean_diff'))
for c in ('slidetags','xenium'):
    if c not in wide.columns: wide = wide.with_columns(pl.lit(None).alias(c))
lig2t = {l: t for t, ls in CHORD_THEME_LIGANDS.items() for l in ls}
wrows = []
for r in wide.iter_rows(named=True):
    st, xn = r.get('slidetags'), r.get('xenium')
    concord = (st is not None and xn is not None and st != 0 and xn != 0
               and (st > 0) == (xn > 0))
    wrows.append(dict(theme=lig2t.get(r['ligand_complex']),
        source=r['source'], target=r['target'],
        ligand=r['ligand_complex'], receptor=r['receptor_complex'],
        st=st, xn=xn, both_present=(st is not None and xn is not None),
        sign_concordant=concord,
        meta_diff=((st + xn) / 2 if concord else None),
        dir=('up' if concord and (st + xn) > 0 else 'down' if concord else None)))
pl.DataFrame(wrows).write_csv(f'{OUT}/ccc_pair_concordance.csv')

# theme-level CCC direction summary (concordant edges only)
th = {}
for w in wrows:
    if not w['sign_concordant']: continue
    t = w['theme']; d = th.setdefault(t, dict(n=0, up=0, dn=0, sumsigned=0.0))
    d['n'] += 1; d['up'] += w['dir'] == 'up'; d['dn'] += w['dir'] == 'down'
    d['sumsigned'] += w['meta_diff']
th_rows = [dict(theme=t, n_concordant_edges=d['n'], edges_up=d['up'], edges_down=d['dn'],
                signed_sum=d['sumsigned'],
                dir_dominant=('up' if d['sumsigned'] > 0 else 'down')) for t, d in th.items()]
pl.DataFrame(th_rows).write_csv(f'{OUT}/ccc_theme_direction.csv')

# the actual edges the figure draws (replicate build_chord) -> direction per theme
cell_set, edges = fc.build_chord(f'{wd}/output/liana/inflow_diff.csv',
    CANONICAL_LR_PAIRS, CHORD_THEME_LIGANDS, k_neurons=10)
edges.write_csv(f'{OUT}/ccc_figure_edges.csv')
with open(f'{OUT}/ccc_cell_set.json', 'w') as f:
    json.dump(sorted(cell_set, key=fc.numeric_prefix), f, indent=1)

# DE direction of each L-R gene (ligand UP in preg? receptor?) over columns + all
lr_genes = sorted(set(ligs) | set(recs))
lrde = []
for g in lr_genes:
    rs = de_sr.filter((pl.col('gene') == g))
    rs_col = rs.filter(pl.col('cell_type').is_in(list(col_set)))
    # direction via de_best (sumrank)
    detail = [(ct, de_best.get((g, ct))) for ct in all_cts if de_best.get((g, ct))]
    sig = [(ct, v) for ct, v in detail if v['emp_p'] is not None and v['emp_p'] <= 0.05]
    up = sum(v['dir_sr'] == 'up' for _, v in sig); dn = len(sig) - up
    lrde.append(dict(gene=g, role=('ligand' if g in ligs else '') + ('receptor' if g in recs else ''),
        in_dotplot=g in ordered_genes, n_sig_all=len(sig), sig_up=up, sig_down=dn,
        dir_dominant=('up' if up >= dn else 'down') if sig else 'none',
        peak_nlp=max((v['nlp'] for _, v in detail), default=0.0)))
pl.DataFrame(lrde).sort('n_sig_all', descending=True).write_csv(f'{OUT}/ccc_lr_gene_de.csv')

# ============================================================ 5. CARDS (row select)
def card_rows(gene, ctx, max_rows):
    hits = de_sr.filter((pl.col('gene') == gene) & pl.col('cell_type').is_in(list(col_set)))
    cts_in = list(hits['cell_type'].to_list())
    for ct in ctx:
        if ct in col_set and ct not in cts_in: cts_in.append(ct)
    rows = []
    for ct in cts_in:
        srr = hits.filter(pl.col('cell_type') == ct)
        ep = srr['emp_p'][0] if srr.height else None
        rows.append(dict(cell_type=ct, emp_p=ep, forced=ct in ctx))
    if max_rows and len(rows) > max_rows:
        keep_r = [r for r in rows if r['forced']]
        others = sorted([r for r in rows if not r['forced']],
                        key=lambda r: r['emp_p'] if r['emp_p'] is not None else 9.0)
        rows = keep_r + others[:max_rows - len(keep_r)]
    return rows
card_out = []
for g in CARD_GENES:
    for r in card_rows(g, card_ctx.get(g, []), max_rows_map.get(g)):
        card_out.append(dict(card_gene=g, **r))
pl.DataFrame(card_out).write_csv(f'{OUT}/cards.csv')

# ============================================================ 6. CROSS-FIGURE
crossfig = {
 'figure_3_neuron': {
   'themes': ['Synaptic adhesion','Excitability','GABA & neuropeptide','Neuronal development'],
   'ligands': {'Synaptic adhesion':['Cntn1','Ncam1'],'Excitability':['Slc17a7'],
               'GABA & neuropeptide':['Gad2'],'Neuronal development':['Sema4a','Sema5a']},
   'pairs': [['Cntn1','Nrcam'],['Ncam1','Robo1'],['Slc17a7','Gria1'],['Slc17a7','Grin1'],
             ['Gad2','Gabbr1'],['Sema4a','Plxnd1'],['Sema5a','Plxna3']]},
 'figure_4_microglia_vascular': {
   'themes': ['TGFb','Notch','Ang2_Cxcl12'],
   'ligands': {'TGFb':['Tgfb1','Tgfb2'],'Notch':['Dll4','Jag1','Jag2'],
               'Ang2_Cxcl12':['Angpt2','Cxcl12']}},
 'figure_5_lipid': {'themes': list(CHORD_THEME_LIGANDS), 'ligands': CHORD_THEME_LIGANDS},
}
with open(f'{OUT}/crossfig.json', 'w') as f:
    json.dump(crossfig, f, indent=1)

# manifest
with open(f'{OUT}/_manifest.json', 'w') as f:
    json.dump(dict(displayed_cts=ordered_cts, ct_class=ct_class,
        ordered_pathways=ordered_pathways, pathway_band=pathway_band,
        ordered_genes=ordered_genes, gene_band=gene_band,
        card_genes=CARD_GENES, card_ctx=card_ctx,
        canonical_pairs=sorted(['__'.join(p) for p in CANONICAL_LR_PAIRS]),
        n_lipid_candidate_pathways=len(lip_paths)), f, indent=1)
print('\nwrote audit slices to', OUT)
for fn in sorted(os.listdir(OUT)): print('  ', fn)
