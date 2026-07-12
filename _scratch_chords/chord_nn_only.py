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

def is_neuron(c): return ('Glut' in c) or ('Gaba' in c) or ('IMN' in c)
neuron_set = {c for c in cell_allow if is_neuron(c)}

d = d.filter(pl.col("source") != pl.col("target"))
d = d.filter(pl.col("source").is_in(cell_allow) & pl.col("target").is_in(cell_allow))

PAIRS = [
    ("Cntn1","Nrcam"), ("Ncam1","Robo1"), ("Slc17a7","Gria1"), ("Slc17a7","Grin1"), ("Gad2","Gabbr1"),
    ("Ncam1","Ncam2"), ("Gad1","Gabbr1"), ("Slc32a1","Gabbr1"), ("Ncam1","Gfra1"),
]
keys = ["source","target","ligand_complex","receptor_complex"]

print(f"{'pair':22s} {'n_nn':>5s} {'tot|mag|':>9s} {'top1_frac':>9s} {'UP':>5s} {'DOWN':>5s}")
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
    # restrict to neuron-neuron only
    xp_nn = xp.filter(pl.col("source").is_in(neuron_set) & pl.col("target").is_in(neuron_set))
    n = xp_nn.height
    mags = xp_nn["meta_diff"].abs().to_numpy()
    total_mag = float(mags.sum())
    top1_frac = float(mags.max()/total_mag) if total_mag>0 else 0.0
    up = xp_nn.filter(pl.col("meta_diff")>0).height
    down = xp_nn.filter(pl.col("meta_diff")<0).height
    print(f"{lig+'->'+rec:22s} {n:5d} {total_mag:9.3f} {top1_frac:9.3f} {up:5d} {down:5d}")
