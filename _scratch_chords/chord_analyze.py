import polars as pl

d = pl.read_parquet("/home/karbabi/spatial-pregnancy/_scratch_chords/lr_raw.parquet")

cell_allow = set("""
009 L2/3 IT PIR-ENTl Glut
046 Vip Gaba
030 L6 CT CTX Glut
005 L5 IT CTX Glut
056 Sst Chodl Gaba
004 L6 IT CTX Glut
007 L2/3 IT CTX Glut
063 STR D1 Sema5a Gaba
002 IT EP-CLA Glut
053 Sst Gaba
119 SI-MA-LPO-LHA Skor1 Glut
058 PAL-STR Gaba-Chol
082 CEA-BST Ebf1 Pdyn Gaba
055 STR Lhx8 Gaba
022 L5 ET CTX Glut
029 L6b CTX Glut
062 STR D2 Gaba
060 OT D3 Folh1 Gaba
064 STR-PAL Chst9 Gaba
001 CLA-EPd-CTX Car3 Glut
061 STR D1 Gaba
066 NDB-SI-ant Prdm12 Gaba
032 L5 NP CTX Glut
006 L4/5 IT CTX Glut
052 Pvalb Gaba
054 STR Prox1 Lhx6 Gaba
085 SI-MPO-LPO Lhx8 Gaba
086 MPO-ADP Lhx8 Gaba
124 MPN-MPO-PVpo Hmx2 Glut
318 Astro-NT NN
319 Astro-TE NN
323 Ependymal NN
326 OPC NN
327 Oligo NN
334 Microglia NN
335 BAM NN
114 COAa-PAA-MEA Barhl2 Glut
""".strip().split("\n"))
assert len(cell_allow) == 37

def is_nn(c): return ' NN' in c or c.endswith(' NN')
def is_neuron(c): return ('Glut' in c) or ('Gaba' in c) or ('IMN' in c)

d = d.filter(pl.col("source") != pl.col("target"))
d = d.filter(pl.col("source").is_in(cell_allow) & pl.col("target").is_in(cell_allow))

PAIRS = [
    ("Cntn1","Nrcam"), ("Ncam1","Robo1"), ("Slc17a7","Gria1"), ("Slc17a7","Grin1"), ("Gad2","Gabbr1"),
    ("Ncam1","Ncam2"), ("Gad1","Gabbr1"), ("Slc32a1","Gabbr1"), ("Ncam1","Gfra1"),
]

keys = ["source","target","ligand_complex","receptor_complex"]

for lig, rec in PAIRS:
    sub = d.filter((pl.col("ligand_complex")==lig) & (pl.col("receptor_complex")==rec))
    wide = (sub.group_by(keys+["dataset"]).agg(pl.col("lr_mean_diff").first())
            .pivot(on="dataset", index=keys, values="lr_mean_diff"))
    for c in ("slidetags","xenium"):
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    wide = wide.rename({"slidetags":"st","xenium":"xn"})
    xp = wide.filter(pl.col("st").is_not_null() & pl.col("xn").is_not_null()
                      & (pl.col("st")!=0) & (pl.col("xn")!=0)
                      & (pl.col("st").sign()==pl.col("xn").sign()))
    xp = xp.with_columns(((pl.col("st")+pl.col("xn"))/2).alias("meta_diff"))
    n_edges = xp.height
    n_neuron_neuron = xp.filter(pl.col("source").map_elements(is_neuron, return_dtype=pl.Boolean)
                                  & pl.col("target").map_elements(is_neuron, return_dtype=pl.Boolean)).height
    n_neuron_nn = n_edges - n_neuron_neuron
    src_classes = set(xp["source"].to_list())
    tgt_classes = set(xp["target"].to_list())
    distinct_pairs = xp.select(["source","target"]).unique().height
    # top 8 by |meta_diff|
    top8 = xp.with_columns(pl.col("meta_diff").abs().alias("mag")).sort("mag", descending=True).head(8)
    top8_rows = list(top8.select(["source","target","meta_diff"]).iter_rows())
    up = xp.filter(pl.col("meta_diff")>0).height
    down = xp.filter(pl.col("meta_diff")<0).height
    total_mag = float(xp["meta_diff"].abs().sum())
    print(f"\n=== {lig} -> {rec} ===")
    print(f"  xp(concordant) edges={n_edges}  neuron-neuron={n_neuron_neuron}  neuron-NN={n_neuron_nn}  distinct(src,tgt) pairs={distinct_pairs}")
    print(f"  UP={up} DOWN={down}  total|mag|={total_mag:.3f}")
    print(f"  n distinct source cts={len(src_classes)}  n distinct target cts={len(tgt_classes)}")
    print(f"  top8 by |meta_diff|:")
    for s,t,md in top8_rows:
        print(f"    {s} -> {t}  meta_diff={md:.4f}  src_nn={is_nn(s)} tgt_nn={is_nn(t)}")
