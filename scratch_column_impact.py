import polars as pl

CONTRAST = "PREG_vs_CTRL"

# ---- load tables (small enough to collect fully) ----
gsea = pl.scan_csv("output/gsea/sumrank_gsea_results.csv").filter(
    pl.col("contrast") == CONTRAST
).collect(engine="streaming")

de = pl.scan_csv("output/de/sumrank_results.csv").filter(
    pl.col("contrast") == CONTRAST
).collect(engine="streaming")

print("gsea rows (PREG_vs_CTRL):", gsea.shape)
print("de rows (PREG_vs_CTRL):", de.shape)

def is_neuronal(col):
    return col.str.contains(" Glut") | col.str.contains(" Gaba") | col.str.contains("IMN")

# ---- current 37 columns (from FIGURE_STATE) ----
COLUMNS_37 = [
"009 L2/3 IT PIR-ENTl Glut","046 Vip Gaba","030 L6 CT CTX Glut","005 L5 IT CTX Glut",
"056 Sst Chodl Gaba","004 L6 IT CTX Glut","007 L2/3 IT CTX Glut","063 STR D1 Sema5a Gaba",
"002 IT EP-CLA Glut","053 Sst Gaba","119 SI-MA-LPO-LHA Skor1 Glut","058 PAL-STR Gaba-Chol",
"082 CEA-BST Ebf1 Pdyn Gaba","055 STR Lhx8 Gaba","022 L5 ET CTX Glut","029 L6b CTX Glut",
"062 STR D2 Gaba","060 OT D3 Folh1 Gaba","064 STR-PAL Chst9 Gaba","001 CLA-EPd-CTX Car3 Glut",
"061 STR D1 Gaba","066 NDB-SI-ant Prdm12 Gaba","032 L5 NP CTX Glut","006 L4/5 IT CTX Glut",
"052 Pvalb Gaba","054 STR Prox1 Lhx6 Gaba",
# allowlists
"085 SI-MPO-LPO Lhx8 Gaba","086 MPO-ADP Lhx8 Gaba","124 MPN-MPO-PVpo Hmx2 Glut",
"318 Astro-NT NN","319 Astro-TE NN","323 Ependymal NN","326 OPC NN","327 Oligo NN","334 Microglia NN","335 BAM NN",
"114 COAa-PAA-MEA Barhl2 Glut",
]
assert len(COLUMNS_37) == 37, len(COLUMNS_37)
COLUMNS_37_SET = set(COLUMNS_37)

CURRENT_26 = [
"009 L2/3 IT PIR-ENTl Glut","046 Vip Gaba","030 L6 CT CTX Glut","005 L5 IT CTX Glut",
"056 Sst Chodl Gaba","004 L6 IT CTX Glut","007 L2/3 IT CTX Glut","063 STR D1 Sema5a Gaba",
"002 IT EP-CLA Glut","053 Sst Gaba","119 SI-MA-LPO-LHA Skor1 Glut","058 PAL-STR Gaba-Chol",
"082 CEA-BST Ebf1 Pdyn Gaba","055 STR Lhx8 Gaba","022 L5 ET CTX Glut","029 L6b CTX Glut",
"062 STR D2 Gaba","060 OT D3 Folh1 Gaba","064 STR-PAL Chst9 Gaba","001 CLA-EPd-CTX Car3 Glut",
"061 STR D1 Gaba","066 NDB-SI-ant Prdm12 Gaba","032 L5 NP CTX Glut","006 L4/5 IT CTX Glut",
"052 Pvalb Gaba","054 STR Prox1 Lhx6 Gaba",
]
assert len(CURRENT_26) == 26

CURRENT_19_GO = [
"GOBP_SYNAPSE_ASSEMBLY","GOBP_SYNAPSE_ORGANIZATION","GOBP_HOMOPHILIC_CELL_CELL_ADHESION",
"GOBP_REGULATION_OF_MEMBRANE_POTENTIAL","GOBP_MONOATOMIC_ION_TRANSPORT","GOBP_POTASSIUM_ION_TRANSPORT",
"GOBP_SYNAPTIC_TRANSMISSION_GABAERGIC","GOBP_NEUROPEPTIDE_SIGNALING_PATHWAY","GOBP_NEUROTRANSMITTER_SECRETION",
"GOBP_RESPONSE_TO_CORTICOSTEROID","GOBP_RESPONSE_TO_STEROID_HORMONE",
"GOBP_RESPONSE_TO_NERVE_GROWTH_FACTOR","GOBP_NEUROTROPHIN_TRK_RECEPTOR_SIGNALING_PATHWAY",
"GOBP_REGULATION_OF_NEURON_DIFFERENTIATION","GOBP_NEURON_FATE_COMMITMENT","GOBP_AXON_DEVELOPMENT",
"GOBP_REGULATION_OF_SYNAPTIC_PLASTICITY","GOBP_REGULATION_OF_TRANS_SYNAPTIC_SIGNALING",
"GOBP_REGULATION_OF_LONG_TERM_SYNAPTIC_POTENTIATION",
]
assert len(CURRENT_19_GO) == 19

CANDIDATE_PATHWAYS = [
"GOBP_INHIBITORY_SYNAPSE_ASSEMBLY","GOBP_DENDRITIC_SPINE_MORPHOGENESIS",
"GOBP_REGULATION_OF_POSTSYNAPTIC_MEMBRANE_POTENTIAL","GOBP_CHLORIDE_TRANSPORT",
"GOBP_ACTION_POTENTIAL","GOBP_REGULATION_OF_SYNAPTIC_TRANSMISSION_GABAERGIC",
"GOBP_NEUROGENESIS","GOBP_NEURON_DEVELOPMENT","GOBP_REGULATION_OF_NEURON_PROJECTION_DEVELOPMENT",
"GOBP_ASSOCIATIVE_LEARNING","GOBP_LONG_TERM_MEMORY",
]

# sanity: confirm all pathway names exist in table
all_paths = set(gsea["pathway"].unique().to_list())
for p in CURRENT_19_GO + CANDIDATE_PATHWAYS:
    if p not in all_paths:
        print("MISSING PATHWAY IN TABLE:", p)

CANDIDATE_GENES = ["Nts","Pdyn","Tac2","Crh","Sst","Ar","Esr1","Esr2","Pgr","Oxtr","Drd1","Drd2"]

print("\n" + "="*100)
print("PART (a): candidate genes -> significant neuronal cell types, DE table (D>=2, ref_pct>=5)")
print("="*100)

de_gate = de.filter(pl.col("D") >= 2).filter(pl.col("ref_pct_detected") >= 5).filter(is_neuronal(pl.col("cell_type")))
de_gate = de_gate.with_columns(
    nlp=pl.max_horizontal("nlp_up","nlp_down"),
    direction=pl.when(pl.col("nlp_up") >= pl.col("nlp_down")).then(pl.lit("UP")).otherwise(pl.lit("DOWN")),
    emp_p=pl.when(pl.col("nlp_up") >= pl.col("nlp_down")).then(pl.col("emp_p_up")).otherwise(pl.col("emp_p_down")),
)
de_gate = de_gate.with_columns(significant=pl.col("emp_p") <= 0.05)

gene_results = {}
for g in CANDIDATE_GENES:
    sub = de_gate.filter(pl.col("gene") == g)
    sig = sub.filter(pl.col("significant")).sort("emp_p")
    n_tested = sub.height
    rows = []
    for r in sig.iter_rows(named=True):
        ct = r["cell_type"]
        status = "ALREADY-IN-37-COLUMNS" if ct in COLUMNS_37_SET else "NEW-COLUMN-NEEDED"
        rows.append((ct, r["direction"], r["D"], r["ref_pct_detected"], r["emp_p"], status))
    gene_results[g] = (n_tested, rows)
    print(f"\n-- {g} -- n_sig={len(rows)} / n_tested_neuronal_D>=2_refpct>=5={n_tested}")
    if not rows:
        print("   (no significant neuronal cell types under gate -> OFF-PANEL / absent)")
    for ct, d, D, rp, p, status in rows:
        print(f"   {ct:35s} dir={d:4s} D={D} ref_pct={rp:5.1f} emp_p={p:.2e}  [{status}]")

print("\n" + "="*100)
print("PART (b): simulate adding each candidate pathway to the 19-term set; recompute >=3-hit neuronal columns")
print("="*100)

gsea_neu = gsea.filter(is_neuronal(pl.col("cell_type"))).filter(pl.col("D") >= 2)
gsea_neu = gsea_neu.with_columns(
    nlp=pl.max_horizontal("nlp_up","nlp_down"),
    direction=pl.when(pl.col("nlp_up") >= pl.col("nlp_down")).then(pl.lit("UP")).otherwise(pl.lit("DOWN")),
    emp_p=pl.when(pl.col("nlp_up") >= pl.col("nlp_down")).then(pl.col("emp_p_up")).otherwise(pl.col("emp_p_down")),
)
gsea_neu = gsea_neu.with_columns(significant=pl.col("emp_p") <= 0.05)

def hitset(pathway_list):
    sub = gsea_neu.filter(pl.col("pathway").is_in(pathway_list)).filter(pl.col("significant"))
    counts = sub.group_by("cell_type").agg(pl.len().alias("n_hits"))
    keep = counts.filter(pl.col("n_hits") >= 3)
    return dict(zip(keep["cell_type"].to_list(), keep["n_hits"].to_list())), dict(zip(counts["cell_type"].to_list(), counts["n_hits"].to_list()))

base_keep, base_all = hitset(CURRENT_19_GO)
base_keep_set = set(base_keep.keys())
print("\nBaseline (19 terms) >=3-hit neuronal cell types:", len(base_keep_set))
mismatch = base_keep_set.symmetric_difference(set(CURRENT_26))
if mismatch:
    print("  *** MISMATCH vs reported CURRENT_26 ***:", mismatch)
    print("  computed set:", sorted(base_keep_set))
else:
    print("  MATCHES reported CURRENT_26 exactly.")

for p in CANDIDATE_PATHWAYS:
    new_keep, new_all = hitset(CURRENT_19_GO + [p])
    new_keep_set = set(new_keep.keys())
    gained = sorted(new_keep_set - base_keep_set)
    lost = sorted(base_keep_set - new_keep_set)
    gained_status = [(ct, "ALREADY-IN-37-COLUMNS" if ct in COLUMNS_37_SET else "NEW-COLUMN-NEEDED") for ct in gained]
    print(f"\n-- +{p} --")
    print(f"   new >=3-hit set size: {len(new_keep_set)} (baseline {len(base_keep_set)})")
    print(f"   GAINED columns ({len(gained)}): {gained_status}")
    print(f"   LOST columns ({len(lost)}): {lost}")
    # also show this pathway's own per-celltype significance in neuronal cts, for transparency
    own = gsea_neu.filter(pl.col("pathway")==p).filter(pl.col("significant")).sort("emp_p")
    print(f"   pathway's own significant neuronal cell types ({own.height}):")
    for r in own.iter_rows(named=True):
        print(f"     {r['cell_type']:35s} dir={r['direction']:4s} emp_p={r['emp_p']:.2e}")

print("\n" + "="*100)
print("PART (c): hypothalamic/preoptic/BNST/extended-amygdala/septal neuronal subclasses")
print("="*100)

KEYWORDS = ["MPO","MPN","LPO","LHA","PVpo","BST","CEA","MEA","PVH","ARH","SI","NDB","LS","MS"]

neuronal_cts = sorted(set(gsea.filter(is_neuronal(pl.col("cell_type")))["cell_type"].to_list()))
def matches_kw(ct):
    hits = []
    for kw in KEYWORDS:
        # match as a token: kw bounded by non-alnum or string edges to avoid accidental substrings
        import re
        if re.search(rf'(?<![A-Za-z0-9]){kw}(?![a-z0-9])', ct):
            hits.append(kw)
    return hits

hypo_cts = []
for ct in neuronal_cts:
    kws = matches_kw(ct)
    if kws:
        hypo_cts.append((ct, kws))

print(f"Neuronal subclasses matching hypothalamic/BNST/septal keywords: {len(hypo_cts)}")
for ct, kws in hypo_cts:
    n_hits = base_all.get(ct, 0)
    is_col = ct in COLUMNS_37_SET
    print(f"  {ct:35s} keywords={kws}  n_sig_hits(19 GO)={n_hits}  already_column={is_col}")
