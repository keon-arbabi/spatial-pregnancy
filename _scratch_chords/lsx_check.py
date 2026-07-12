import polars as pl
path = "/home/karbabi/spatial-pregnancy/output/liana/inflow_diff.csv"

# unrestricted (whole atlas, self-loops included) check for Gad1->Gabbr1 and Slc32a1->Gabbr1
d = (pl.scan_csv(path)
     .filter(pl.col("contrast") == "PREG_vs_CTRL")
     .filter(pl.col("ligand_complex").is_in(["Gad1","Slc32a1"]) & (pl.col("receptor_complex")=="Gabbr1"))
     .select(["dataset","source","target","ligand_complex","receptor_complex","lr_mean_diff"])
     .collect(engine="streaming"))
print("rows:", d.height)
# self loops?
self_loops = d.filter(pl.col("source")==pl.col("target"))
print("self-loop rows:", self_loops.height)
lsx_rows = d.filter(pl.col("source").str.contains("LSX") | pl.col("target").str.contains("LSX"))
print("rows involving any LSX cell type:", lsx_rows.height, "/ total", d.height)
print(lsx_rows.select(["source","target","dataset","lr_mean_diff"]).unique().sort("lr_mean_diff", descending=True).head(10))
