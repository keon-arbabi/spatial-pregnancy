# Figure 1 dotplot — marker coverage & name-gene suitability (deep report)

**Question.** (1) Do any of the 84 displayed cell types lack a usable marker among
the 34 genes in the dotplot? (2) For each subclass, is the gene that defines its
*name* present in the panels and actually specific/high for it?

**Method.** Computed directly from the three `03_adata_query_*.h5ad` files (read-only).
For each platform, per-subclass mean expression was z-scored across the 84 included
subclasses (same set/ordering as the figure). A platform "carries" a subclass only if
it has ≥20 cells of it (so absence ≠ low signal). Name-defining genes were parsed from
each subclass label (region abbreviations excluded by testing every token against the
whole-transcriptome slide-tags symbol set; `D1/D2/D3`→`Drd1/2/3`). A gene is a **usable
marker** for a subclass if, in ≥2 carrying panels, the subclass has z≥1.5 **and** ranks
in that gene's top-5 subclasses (specificity, not just height). "one-panel" = strong but
only one platform measures both gene and cell type.

---

## Headline

- **62/84 displayed rows have a specific marker; 22/84 do not** — either faint across
  all 34 columns, or they only share a lineage column with sibling types (not uniquely
  resolved). The deficit is concentrated in **HY-EA (12 of 27)** and fine **cortical IT
  (5)** subtypes — exactly the groups the MERFISH 496-panel does not serve.
- **31/84 subclasses have no eponymous gene at all** — by design they are named
  anatomically/by lineage (all cortical IT/ET/CT/L6b layers; all 13 non-neuronal;
  a few regional GABA). These are covered by canonical class/layer markers, not a
  namesake gene, and the non-neuronal 13 are all well covered.
- Of the **53 subclasses that do carry an eponymous gene: 26 good, 7 partial,
  9 strong-but-single-platform, 11 weak.** The 11 weak namesakes are broad transcription
  factors or genes that peak elsewhere — the gene names them but does not mark them well.
- **Three clean cross-platform additions would close most of the resolvable HY-EA gap:**
  **Bnc2** (078), **Barhl2** (114), **Pdyn** (082). None are currently displayed.

---

## Part A — the 22 rows without a specific displayed marker

`best` = strongest displayed gene for that row (mean z across carrying panels).

| group | subclass | best displayed gene (z) | status |
|---|---|---|---|
| Pallium Glut | 002 IT EP-CLA Glut | Slc17a7 (1.71) | shares pan-IT markers; no unique column |
| Pallium Glut | 004 L6 IT CTX Glut | Slc17a7 (1.95) | shares pan-IT markers |
| Pallium Glut | 010 IT AON-TT-DP Glut | — (≈0) | **faint** — no displayed gene fits |
| Pallium Glut | 020 L2/3 IT RSP Glut | Lamp5 (2.77) | Lamp5 high but shared with 049 |
| Pallium Glut | 028 L6b/CT ENT Glut | Fezf2 (1.64) | shares L6b markers |
| Pallium GABA | 050 Lamp5 Lhx6 Gaba | Lhx6 (1.55) | Lamp5 & Lhx6 both weak here |
| Subpallium GABA | 041 OB-in Frmd7 Gaba | Ppp1r1b (1.23) | faint; Frmd7 weak |
| Subpallium GABA | 059 GPe-SI Sox6 Cyp26b1 | Gad2 (1.10) | fixable → **Sox6** (xe +2.4) |
| Subpallium GABA | 068 LSX Otx2 Gaba | Gad2 (1.32) | Otx2 weak everywhere |
| Subpallium GABA | 070 LSX Prdm12 Slit2 | Nr3c2 (1.89) | fixable → **Prdm12** (sl +3.9) |
| HY-EA | 073 MEA-BST Sox6 Gaba | Tac2 (0.44) | **faint**; Sox6 weak |
| HY-EA | 074 MEA-BST Lhx6 Sp9 | Gad2 (0.75) | **faint**; Lhx6 broad |
| HY-EA | 078 SI-MA-ACB Ebf1 Bnc2 | Pax6 (1.13) | fixable → **Bnc2** (sl +5.4 / xe +7.9) ★ |
| HY-EA | 079 CEA-BST Six3 Cyp26b1 | Nr3c2 (1.53) | fixable → **Six3** (partial) |
| HY-EA | 080 CEA-AAA-BST Six3 Sp9 | Foxp2 (1.64) | fixable → **Six3** (partial) |
| HY-EA | 082 CEA-BST Ebf1 Pdyn | Drd2 (1.14) | fixable → **Pdyn** (sl +2.4 / xe +3.9) ★ |
| HY-EA | 085 SI-MPO-LPO Lhx8 | — (≈0) | Lhx8 present but shared with 055/057/086 |
| HY-EA | 090 BST-MPN Six3 Nrgn | Esr1 (1.36) | fixable → **Six3** (partial) |
| HY-EA | 111 TRS-BAC Sln Glut | Cux2 (0.74) | fixable → **Sln** (sl +8.5, 1 panel) |
| HY-EA | 114 COAa-PAA-MEA Barhl2 | Slc17a7 (0.37) | fixable → **Barhl2** (sl +3.5 / xe +7.8) ★ |
| HY-EA | 116 AVPV-MEPO-SFO Tbr1 | Trh (1.12) | Tbr1 weak; no good fix |
| HY-EA | 119 SI-MA-LPO-LHA Skor1 | Lhx8 (0.50) | fixable → **Skor1** (sl +4.0, 1 panel) |

**Read:** "uncovered" ≠ markerless cell. Many of these express a displayed *lineage*
column (Slc17a7, Lhx8, Lhx6, Gad2) but share it with siblings, so no single column is
unique to them. Identity for those rows is carried by the dendrogram + subclass colour
bar + neighbourhood grouping rather than a dedicated gene column.

---

## Part B — are the name-defining (eponymous) genes available and well-suited?

**31/84 have no eponymous gene** (named by region/lineage), and rely on canonical
markers — correctly:
- cortical glutamatergic layers (all `L#/IT/ET/CT/L6b …`) → Slc17a7 + Cux2/Rorb/Fezf2/Tle4/Foxp2;
- all 13 non-neuronal (Astro/Oligo/OPC/Ependymal/CHOR/VLMC/Peri/SMC/Endo/Microglia/BAM)
  → Aqp4/Foxj1/Pdgfra/Mog/Pdgfrb/Myh11/Flt1/Cx3cr1 — **all 13 well covered**;
- a few regional GABA (045 OB-STR-CTX Inh IMN, 058 PAL-STR Gaba-Chol).

**53/84 carry an eponymous gene:**

| verdict | n | meaning | subclasses |
|---|---|---|---|
| good | 26 | namesake is a strong, cross-platform marker | 046 Vip, 049 Lamp5, 051/052 Pvalb, 053/056 Sst, 054 Prox1, 057/085/086 Lhx8, 060 Drd3, 061/063/081 Drd1, 062 Drd2, 067 Pax6, 075/076 Lhx6, 078 Bnc2, 082 Pdyn, 088 Tac2, 089 Six3, 106/107 Hmx2, 114 Barhl2, 118 Trp73 |
| partial | 7 | namesake moderate / one strong platform | 055 Lhx8, 079/080/090 Six3, 101 Pax6, 115 Bsx, 124 Hmx2 |
| one-panel | 9 | namesake excellent but only one platform measures it | 001 Car3, 059 Sox6, 064 Chst9, 070/071 Prdm12, 072 Lmo1, 111 Sln, 119 Skor1, 131 Otp |
| weak | 11 | namesake present in ≥1 panel but a poor marker here | see below |

**The 11 weak namesakes — the gene names the type but does not mark it:**

| subclass | namesake | z (sl/me/xe) | why weak |
|---|---|---|---|
| 047 Sncg Gaba | Sncg | +0.5 / −0.0 / +1.6 | γ-synuclein peaks in SMC/vascular, not Sncg interneurons |
| 050 Lamp5 Lhx6 Gaba | Lamp5 | +0.5 / – / +0.7 | a Lamp5-low neurogliaform type; Lamp5 itself faint |
| 041 OB-in Frmd7 Gaba | Frmd7 | +0.5 / – / −0.4 | low detection; OB-restricted |
| 065 IA Mgp Gaba | Mgp | −0.2 / – / – | one panel, no signal |
| 068 LSX Otx2 Gaba | Otx2 | +0.3 / +0.6 / +0.4 | broad developmental TF |
| 069 LSX Nkx2-1 Gaba | Nkx2-1 | −0.4 / – / +0.1 | broad MGE TF, low detection |
| 066 NDB-SI-ant Prdm12 | Prdm12 | +0.4 (1p) | low here (cf. strong in 070/071) |
| 073 MEA-BST Sox6 Gaba | Sox6 | −0.2 / – / −0.2 | Sox6 broad across GABA lineages |
| 074 MEA-BST Lhx6 Sp9 | Lhx6 | −0.1 / −0.1 / −0.1 | Lhx6 is a pan-MGE factor, not specific |
| 116 AVPV-MEPO-SFO Tbr1 | Tbr1 | −0.5 / +0.9 / +0.7 | Tbr1 broad pallial/preoptic |
| 126 ARH-PVp Tbx3 Glut | Tbx3 | −0.2 (xe) | low detection |

These are mostly **broad transcription factors** (Lhx6, Sox6, Otx2, Nkx2-1, Tbr1) or
genes whose **peak is another cell type** (Sncg). Per the standing rule, the canonical
ones stay (they are the atlas naming gene); the table just documents that the wedge will
read pale for these rows because the biology, not the panel, makes them poor markers.

---

## Part C — opportunities (not applied)

Adding the row's own eponymous gene closes most of the *resolvable* gap. Ranked by value:

**Clean cross-platform wins (light up ≥2 wedges, high specificity):**
- **Bnc2** → 078 SI-MA-ACB Ebf1 Bnc2 (sl +5.4 / xe +7.9)
- **Barhl2** → 114 COAa-PAA-MEA Barhl2 (sl +3.5 / xe +7.8)
- **Pdyn** → 082 CEA-BST Ebf1 Pdyn (sl +2.4 / xe +3.9)
- **Six3** → 079 / 080 / 090 CEA/BST/MPN Six3 (partial, one column serves three rows)

**Single-platform helps (one strong wedge):**
- **Sln** → 111 (sl +8.5) · **Skor1** → 119 (sl +4.0) · **Prdm12** → 070 (sl +3.9)
  · **Sox6** → 059 (xe +2.4)

**No eponymous fix (intrinsically hard — leave to dendrogram/colour identity):**
- cortical IT/L6b 002, 004, 010, 020, 028 (no namesake gene; would need fine IT markers)
- 050 Lamp5 Lhx6 · 073/074 MEA-BST · 116 Tbr1 · 068 Otx2 · 041 Frmd7 · 085 Lhx8(shared)

**Trade-off:** each addition is one more of 34→up to ~38 columns. Bnc2 + Barhl2 + Pdyn
(+ optionally Six3) would raise specific coverage from 62/84 to ~66–69/84 with three to
four columns, all in the HY-EA block where the gap is worst. The cortical-IT and
broad-TF rows cannot be fixed by a marker swap and are correctly left to the
neighbourhood/dendrogram/colour-bar encoding.

---

## One-line answer

Yes — 22/84 displayed rows lack a *specific* marker (mostly HY-EA preoptic/BST/MEA and
fine cortical IT), and 11 subclasses whose name is a broad TF (or Sncg) are named by a
gene that does not mark them well. The most useful, low-cost remedy is adding **Bnc2,
Barhl2, Pdyn** (and optionally **Six3**) — the rows' own namesake genes, all strong and
cross-platform, none currently shown.
