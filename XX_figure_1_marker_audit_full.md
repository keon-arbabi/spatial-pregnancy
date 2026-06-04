# Figure 1 dotplot — comprehensive per-cell-type marker audit

Every one of the 84 displayed cell types is stepped through below and judged on **two
independent axes**:

1. **Reference specificity** — from `ref_pct_detected.pkl` (ABC 10Xv3 whole-transcriptome).
   For each gene I report the fraction of the subclass's cells that detect it (`pct`,
   sensitivity), its **rank among the 84** displayed subclasses, and the **gap** =
   `pct(this) − pct(next-highest of the 84)` (specificity; >0 means no other displayed
   subclass detects it more).
2. **Spatial signal** — z-scored mean expression across the 84 in each platform
   (`sl`=slide-tags, `me`=MERFISH, `xe`=Xenium), and the subclass's spatial rank for that
   gene.

**Marker tags per (cell type × gene):**
- **specific** — spatial peak/near-peak (z≥2, rank≤2) **and** reference top-3 with pct≥0.3.
- **supporting** — strong on one axis only (real but shared with relatives).
- **shared** — present but ranks below several other subclasses (lineage/spillover).
- **anchor** — a deliberate pan-class gene (Slc17a7, Gad2).

**Cell-type verdict:** ACCURATE (≥1 *specific* marker) · SUPPORTED (only *supporting*) ·
ANCHOR-ONLY (only a pan-class gene) · WEAK (nothing reaches the row specifically).

## Headline

| verdict | n | meaning |
|---|---|---|
| ACCURATE | 37 | a displayed gene specifically and reproducibly marks it |
| SUPPORTED | 20 | marked, but by genes shared across a small relative-cluster |
| ANCHOR-ONLY | 4 | only a pan-class anchor (Slc17a7 / Gad2) reaches it |
| WEAK | 23 | no displayed gene marks it specifically |

**Two genuine mismarks to fix (the only correctness errors found):**
- **335 BAM** is "marked" by **Bsx**, but Bsx is detected in **0%** of BAM cells in the
  reference (rank 70/84) — the slide-tags z+8.7 is a low-count **artifact**. The correct,
  textbook marker is **Mrc1** (ref 1.00, gap +0.94, spatial +8.6, 3-panel).
- **330 VLMC** is "marked" by **Bnc2 / Cyp26b1** spillover (the namesakes of 078 / 059;
  ref 0.11 in VLMC). The correct marker is **Col1a1** (ref 0.91, gap +0.77, spatial +7.6,
  3-panel).

**Two pan-only rows worth upgrading (currently fine but not specific):**
- **318 Astro-NT** rests on pan-astro **Aqp4** → **Lrig1** (gap +0.23, +6.8, 3-panel).
- **325 CHOR** rests on shared **Foxj1** → **Ttr/Ace** (Ace ref 1.00, gap +0.64, 2-panel).
- (**111 TRS-BAC** leans on Barhl2 borrowed from 114 → **Ebf3**, ref 0.99, +8.7.)

**The 23 WEAK + 4 ANCHOR rows are overwhelmingly *not* a panel gap.** For almost all of
them the **best gene in the whole transcriptome has gap ≈ 0.00** (Mef2c, Kcnma1, Nrg3,
Gria2, Peg3, Syt1, Fgf14 …) — i.e. the reference itself has no single specific marker.
These are **combinatorially-defined** types (fine cortical IT; LSX Prdm12/Otx2/Nkx2-1;
MEA-BST; ZI; deep preoptic Hmx2/Skor1) that are correctly left to the dendrogram + colour
bar + neighbourhood grouping. This is a real biological resolution limit, not a fixable
omission.

---

## Pallium Glut (17)

| cell type | verdict | primary displayed marker (spatial · reference) | rationale |
|---|---|---|---|
| 001 CLA-EPd-CTX Car3 | SUPPORTED | Cux2 z+4.6 r1 · ref 0.88 r6 | Car3 (namesake) not in panels; Cux2 is a fair upper/claustrum proxy but pan |
| 002 IT EP-CLA | WEAK | — | best ref Thra gap 0.00: **no specific gene exists**; IT subtype |
| 003 L5/6 IT TPE-ENT | ANCHOR | Slc17a7 r2 · ref 1.0 r1 | pan-glut only; Grm8 gap 0.01 |
| 004 L6 IT CTX | WEAK | — | Mef2c gap 0.00; IT continuum |
| 005 L5 IT CTX | WEAK | — | Mef2c gap 0.00; IT continuum |
| 006 L4/5 IT CTX | **ACCURATE** | Rorb z+6.9 r1 · ref 0.97 r2 | textbook L4; Cux2 co-marks (upper, reasonable) |
| 007 L2/3 IT CTX | **ACCURATE** | Cux2 z+4.2 · ref 0.98 r1; Lamp5 z+4.1 | upper-layer Cux2 + Lamp5, both reasonable |
| 008 L2/3 IT ENT | ANCHOR | Slc17a7 · ref 1.0 r1 | entorhinal L2/3; Adgrl1 gap 0.00 |
| 009 L2/3 IT PIR-ENTl | ANCHOR | Slc17a7 · ref 1.0 r1 | Kcnma1 gap 0.00 |
| 010 IT AON-TT-DP | WEAK | — | olfactory-areas IT; Kcnma1 gap 0.00 |
| 020 L2/3 IT RSP | WEAK | Lamp5 r4 (shared) | retrosplenial; Kif1b gap 0.00 |
| 022 L5 ET CTX | **ACCURATE** | Fezf2 z+4.9 · ref 0.99 r1 | ET marker; Lamp5 co (reasonable) |
| 027 L6b EPd | **ACCURATE** | Tle4 z+2.7 · ref 1.0 r2 | deep-layer Tle4; Slc17a7 anchor |
| 028 L6b/CT ENT | SUPPORTED | Fezf2 r4 · ref 0.89 r3 | shares deep markers; borderline |
| 029 L6b CTX | **ACCURATE** | Tle4 z+4.2 r1 · ref 1.0 r3 | clean L6b |
| 030 L6 CT CTX | WEAK | Tle4/Fezf2 r3 (shared with L6b) | Nrg3 gap 0.00; overlaps L6b |
| 032 L5 NP CTX | **ACCURATE** | Fezf2 z+6.0 r1 · ref 0.93 r2 | NP peaks Fezf2 |

**Glut summary:** the layer/projection backbone (Slc17a7→Cux2→Rorb→Tle4→Foxp2→Fezf2)
cleanly resolves L4 (Rorb), upper (Cux2), L6b (Tle4), ET/NP (Fezf2). The unresolved rows
are all **fine IT subtypes** (002/004/005/010/020/030) that the reference itself cannot
separate with one gene.

## Pallium GABA (7)

| cell type | verdict | primary displayed marker (spatial · reference) | rationale |
|---|---|---|---|
| 046 Vip | **ACCURATE** | Vip z+8.1 r1 · ref 0.88 r1 gap+0.44 | textbook; Tac2 co (CGE, reasonable) |
| 047 Sncg | SUPPORTED | Vip z+3.5 r2; Tac2 r3 | **Sncg namesake is a poor marker** (peaks in SMC); Vip carries it (Sncg/Vip CGE overlap) |
| 049 Lamp5 | ANCHOR | Lamp5 r3 (shared); Gad2 anchor | Lamp5 shared with 007/020; not uniquely resolved |
| 050 Lamp5 Lhx6 | WEAK | Dlx2 r4 | Grik2 gap 0.00; Lamp5-low neurogliaform |
| 051 Pvalb chandelier | **ACCURATE** | Pvalb z+2.9 · ref 0.86 r1 | chandelier subtype, Pvalb specific here |
| 052 Pvalb | **ACCURATE** | Pvalb z+7.6 r1 · ref 0.82 r2 | textbook |
| 053 Sst | **ACCURATE** | Sst z+4.9 · ref 0.98 r2 | textbook; Pdyn co (shared) |

## Subpallium GABA (20)

| cell type | verdict | primary displayed marker (spatial · reference) | rationale |
|---|---|---|---|
| 041 OB-in Frmd7 | WEAK | — | OB interneuron; Pcbp3 gap 0.01; shares neuroblast program |
| 045 OB-STR-CTX IMN | **ACCURATE** | Dlx2 z+5.8 r1 (added); Pax6 co | immature-neuron program; Sox4 even stronger (ref) |
| 054 STR Prox1 Lhx6 | WEAK | Pvalb r3 (shared) | Nrg3 gap 0.00 |
| 055 STR Lhx8 | WEAK | — | Peg3 gap 0.00 |
| 056 Sst Chodl | **ACCURATE** | Sst z+7.0 r1; Lhx6 z+4.8 r1 | Sst+Lhx6 both specific (long-range Sst), reasonable pair |
| 057 NDB-SI-MA-STRv Lhx8 | WEAK | Lhx6 r4 (shared) | Cntn1 gap 0.00 |
| 058 PAL-STR Gaba-Chol | **ACCURATE** | Lhx8 z+4.1 · ref 0.99 r1 | cholinergic Lhx8; Drd2 co (pallidal, reasonable) |
| 059 GPe-SI Sox6 Cyp26b1 | **ACCURATE** | Cyp26b1 z+8.2 r1 (added) | namesake, highly specific |
| 060 OT D3 Folh1 | **ACCURATE** | Tac1 z+5.2 r1 · ref 0.99; Foxp2 co | striatal/OT; **Cux2 (me+4.8) is a spillover artifact** (ref r11) |
| 061 STR D1 | **ACCURATE** | Drd1 z+5.1 r1; Ppp1r1b r1 | D1-MSN; Tac1/Pdyn co (reasonable striatal) |
| 062 STR D2 | **ACCURATE** | Drd2 z+7.5 r1 gap+0.38; Ppp1r1b | textbook D2; Tle4 (r4) minor spillover |
| 063 STR D1 Sema5a | SUPPORTED | Tac1/Drd1/Foxp2 all r3 | D1 subtype overlapping 061; cluster-marked, all reasonable |
| 064 STR-PAL Chst9 | SUPPORTED | Foxp2 z+2.8 · ref 1.0 r1 | Foxp2 ref-specific; Lypd1 best ref |
| 065 IA Mgp | WEAK | — | intercalated amygdala; Gria2 gap 0.00 |
| 067 LSX Sall3 Pax6 | **ACCURATE** | Pax6 z+5.7 r1 · ref 0.98 gap+0.21 | septal Pax6, clean |
| 068 LSX Otx2 | WEAK | Six3 r3 (shared) | Ank2 gap 0.00 |
| 069 LSX Nkx2-1 | WEAK | Nr3c2 r3 (shared) | LSX cluster gene only |
| 070 LSX Prdm12 Slit2 | WEAK | Nr3c2 r4 (shared) | LSX cluster gene only |
| 071 LSX Prdm12 Zeb2 | SUPPORTED | Nr3c2 z+4.4 r1 (ref r13) | pan-LSX Nr3c2; not unique |
| 072 LSX Sall3 Lmo1 | SUPPORTED | Nr3c2 r2 (ref r21) | pan-LSX; not unique |

**Subpallium summary:** striatal MSNs (Drd1/Drd2/Ppp1r1b/Tac1) and septal Pax6/Nr3c2 are
excellent; the **LSX Prdm12/Otx2/Nkx2-1 subtypes share one pan-LSX program** (Nr3c2/Pax6/
Six3) and are not individually separable — confirmed by the reference (gap ≈ 0).

## HY-EA (27)

| cell type | verdict | primary displayed marker (spatial · reference) | rationale |
|---|---|---|---|
| 066 NDB-SI-ant Prdm12 | SUPPORTED | Pax6 r3 · ref 0.65 r3 | weak; Ncam1 gap 0.00 |
| 073 MEA-BST Sox6 | WEAK | — | Fgf14 gap 0.00 |
| 074 MEA-BST Lhx6 Sp9 | WEAK | — | Rtn3 gap 0.00 |
| 075 MEA-BST Lhx6 Nr2e1 | SUPPORTED | Lhx6 z+2.8 r2 (ref r7) | shared MGE program |
| 076 MEA-BST Lhx6 Nfib | SUPPORTED | Esr1 r3; Lhx6/Dlx2 | MEA Esr1+, reasonable; cluster |
| 078 SI-MA-ACB Ebf1 Bnc2 | **ACCURATE** | Bnc2 z+6.6 r1 gap+0.61 (added) | namesake, very specific |
| 079 CEA-BST Six3 Cyp26b1 | SUPPORTED | Six3 r4; Cyp26b1 r3 | CEA/BST Six3 cluster |
| 080 CEA-AAA-BST Six3 Sp9 | WEAK | — | Syt1 gap 0.00 |
| 081 ACB-BST-FS D1 | SUPPORTED | Drd1/Pdyn/Tac1/Foxp2 | D1-like BST/FS; all reasonable, shared with striatum |
| 082 CEA-BST Ebf1 Pdyn | SUPPORTED | Pdyn z+3.1 r2 · ref 0.60 (added) | Drd2 (r4) minor; reasonable |
| 085 SI-MPO-LPO Lhx8 | WEAK | Lhx8 r4 (shared) | shares Lhx8 with 058/086 |
| 086 MPO-ADP Lhx8 | **ACCURATE** | Lhx8 z+5.3 r1 | clean preoptic Lhx8; Gad2 anchor |
| 088 BST Tac2 | **ACCURATE** | Tac2 z+5.4 r1 gap+0.22 | namesake, specific (me/xe) |
| 089 PVR Six3 Sox3 | SUPPORTED | Six3 z+2.8 r1 (ref r5) | PVR Six3 cluster; Gad2 anchor |
| 090 BST-MPN Six3 Nrgn | SUPPORTED | Six3 r2 | Six3 cluster |
| 101 ZI Pax6 | WEAK | Lhx8 r3 **(ref 0.01 — spurious)** | ZI; Peg3 gap 0.00; the Lhx8 hit is spatial spillover |
| 106 PVpo-VMPO-MPN Hmx2 | **ACCURATE** | Gal z+5.3; Esr1 z+4.5 (both specific) | preoptic Gal/Esr1, reasonable pair |
| 107 DMH Hmx2 | **ACCURATE** | Gal z+6.7 r1 | DMH galanin; Esr1/Rorb/Tac2 minor |
| 111 TRS-BAC Sln | ACCURATE* | Barhl2 r2 · ref 0.37 r3 | **Barhl2 borrowed from 114**; true marker **Ebf3** (ref 0.99, +8.7) or Sln |
| 114 COAa-PAA-MEA Barhl2 | **ACCURATE** | Barhl2 z+5.7 r1 (added) | namesake, specific |
| 115 MS-SF Bsx | **ACCURATE** | Bsx z+3.6 (xe+7.3) · ref 0.60 r1 | Xenium-specific (slide-tags lacks signal) |
| 116 AVPV-MEPO-SFO Tbr1 | WEAK | Trh r2 (shared) | App gap 0.00 |
| 118 ADP-MPO Trp73 | SUPPORTED | Barhl2 r3; Sncg r3 | both spillover; Trp73 namesake not in panels |
| 119 SI-MA-LPO-LHA Skor1 | WEAK | — | Skor1 (slide-tags-only) not displayed; Nrg3 gap 0.00 |
| 124 MPN-MPO-PVpo Hmx2 | WEAK | — | Stxbp1 gap 0.00 |
| 126 ARH-PVp Tbx3 | **ACCURATE** | Esr1 z+5.6 r1 (me+7.0/xe+4.2); Bsx co | ARH estrogen-receptor, clean |
| 131 LHA-AHN-PVH Otp Trh | **ACCURATE** | Otp z+8.8 r1 gap+0.92; Trh z+8.6 r1 gap+0.52 | the strongest pair in the figure |

**HY-EA summary:** the well-resolved rows are the named-gene types (Lhx8 086, Tac2 088,
Gal/Esr1 106/107, Barhl2 114, Bsx 115, Esr1 126, Otp/Trh 131) plus the new Bnc2 078. The
**Six3 cluster (079/089/090)** and **MEA-BST (073–076)** are collectively, not
individually, marked; and several deep preoptic types (080/116/119/124) have no specific
gene anywhere (gap ≈ 0).

## Non-neuronal (13)

| cell type | verdict | primary displayed marker (spatial · reference) | rationale |
|---|---|---|---|
| 318 Astro-NT | SUPPORTED | Aqp4 r4 (pan-astro) | upgrade → **Lrig1** (ref 0.96 gap+0.23, +6.8) |
| 319 Astro-TE | **ACCURATE** | Rorb z+2.8 · ref 0.99 r1 | telencephalic astro; Aqp4 co |
| 321 Astroependymal | SUPPORTED | Aqp4+Foxj1 | mixed identity; Rfx4 marginal |
| 323 Ependymal | **ACCURATE** | Foxj1 z+8.1 r1 | clean (Cx3cr1/Sncg/Barhl2 hits are noise) |
| 325 CHOR | SUPPORTED | Foxj1 r3 (shared w/ ependymal) | upgrade → **Ttr/Ace** (Ace ref 1.0 gap+0.64) |
| 326 OPC | **ACCURATE** | Pdgfra z+8.7 r1 | textbook |
| 327 Oligo | **ACCURATE** | Mog z+8.9 r1 gap+0.86 | textbook |
| 330 VLMC | **MISMARK** | Bnc2/Cyp26b1 spillover (ref 0.11) | **→ Col1a1** (ref 0.91 gap+0.77, +7.6) |
| 331 Peri | **ACCURATE** | Pdgfrb z+8.2 r1 | textbook; Flt1 minor (endo adjacency) |
| 332 SMC | **ACCURATE** | Myh11 z+9.0 gap+0.70 | textbook; Sncg co (real in SMC, ref 0.61) |
| 333 Endo | **ACCURATE** | Flt1 z+8.0 gap+0.56 | textbook |
| 334 Microglia | **ACCURATE** | Cx3cr1 z+7.3 gap+0.53 | textbook |
| 335 BAM | **MISMARK** | Bsx ref 0.00 r70 (slide-tags artifact) | **→ Mrc1** (ref 1.0 gap+0.94, +8.6) |

---

## Validation of the recently-added markers

| gene | target | result |
|---|---|---|
| Dlx2 | 045 IMN | **specific** (z+5.8 r1) — confirmed |
| Cyp26b1 | 059 GPe-SI | **specific** (z+8.2 r1 gap+0.24) — confirmed |
| Bnc2 | 078 SI-MA-ACB | **specific** (z+6.6 r1 gap+0.61) — confirmed |
| Barhl2 | 114 COAa-PAA-MEA | **specific** (z+5.7 r1) — confirmed; also (weakly) lights 111/118 |
| Six3 | 079/089/090 | **supporting/cluster** — marks the CEA/BST-Six3 group collectively, not one row |
| Pdyn | 082 CEA-BST | **supporting** (z+3.1 r2, ref 0.60) — modest but valid |

All six are validated as useful; Six3 and Pdyn should be understood as *cluster* markers
(they tag a small group of relatives) rather than single-row-specific.

## Recommendations

**Fix (correctness):**
1. **335 BAM → Mrc1** (replace the Bsx artifact).
2. **330 VLMC → Col1a1** (replace Bnc2/Cyp26b1 spillover).

**Upgrade (specificity, optional):**
3. **318 Astro-NT → add Lrig1**; **325 CHOR → add Ttr/Ace**; **111 TRS-BAC → Ebf3**.

**Accept as unresolvable (no single-gene marker exists in the reference):** the fine
cortical IT set (002/004/005/010/020/030, plus anchors 003/008/009), LSX
Otx2/Nkx2-1/Prdm12 (068/069/070), MEA-BST (073/074), 050/054/055/057/065, and deep
preoptic 080/116/119/124. These remain identified by neighbourhood + dendrogram + colour,
which is the appropriate encoding for combinatorially-defined types.
