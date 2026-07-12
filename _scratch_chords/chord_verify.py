import polars as pl

path = "/home/karbabi/spatial-pregnancy/output/liana/inflow_diff.csv"

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
print("n cell_allow =", len(cell_allow))

def is_nn(c): return ' NN' in c or c.endswith(' NN')
def is_neuron(c): return ('Glut' in c) or ('Gaba' in c) or ('IMN' in c)

PAIRS = {
    "current": [("Cntn1","Nrcam"),("Ncam1","Robo1"),("Slc17a7","Gria1"),("Slc17a7","Grin1"),("Gad2","Gabbr1")],
    "candidates": [("Ncam1","Ncam2"),("Gad1","Gabbr1"),("Slc32a1","Gabbr1"),("Ncam1","Gfra1")],
}
all_pairs = PAIRS["current"] + PAIRS["candidates"]
ligs = sorted({p[0] for p in all_pairs})
recs = sorted({p[1] for p in all_pairs})

d = (pl.scan_csv(path)
     .filter(pl.col("contrast") == "PREG_vs_CTRL")
     .filter(pl.col("ligand_complex").is_in(ligs) & pl.col("receptor_complex").is_in(recs))
     .select(["dataset","source","target","ligand_complex","receptor_complex","lr_mean_diff"])
     .collect(engine="streaming"))
print("rows pulled:", d.height)
d.write_parquet("/home/karbabi/spatial-pregnancy/_scratch_chords/lr_raw.parquet")
