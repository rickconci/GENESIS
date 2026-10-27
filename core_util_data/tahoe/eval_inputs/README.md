# Frozen inputs

Small tables the level-3 drivers (`scripts/figures/figureN/score/`) read as shipped: condition lists, neighbour
tables and hand-curated gene sets. Each was made once, by the rule below, and no driver rebuilds it.

| File | What it is, and the rule that made it | Read by |
|---|---|---|
| `ladder_cohort.csv` | the replication-ladder cohort: 100 of the 175 eligible two-plate conditions, drawn with seed 17 | Figure 1 main |
| `multi_plate_test_conditions.tsv` | every test condition of the bio split measured on two or more plates: 545 conditions, 556 plate pairs. Figure 2a keeps the 530 pairs (500 conditions) with at least 50 cells on each plate whose cell line never saw the compound in training | Figure 2 main, ED_objective |
| `benchmark_pairs.csv` | Figure 2a's 530 plate pairs (500 conditions) | ED_program_summary |
| `copy_closest_line__cosine_v2.tsv`, `copy_closest_line__cosine_pca100__hbss015.tsv` | the neighbour line of each held-out line (`score.py copy_lines`): copy_closest's, and Figure 4's | Figure 2 main, ED_program_summary; Figure 4 main |
| `copy_closest_pert_neighbors.tsv` | copy_closest's neighbour for the held-out-perturbation test conditions | ED_pert_generalization |
| `two_plate_conditions.tsv`, `two_plate_conditions__min50.tsv`, `two_plate_counts.parquet` | the 423 held-out-perturbation test conditions measured on exactly two plates, the 403 with ≥ 50 cells on each, and each plate's cell count | ED_pert_generalization |
| `train614_conds.tsv` | the 3,941 training conditions measured on plates 6 and 14 | ED_12cells |
| `neighbors.parquet` | Copy closest's neighbour for each test condition of the 15%-conditions split | ED_tahoe_hallmark_gsea |
| `gene_sets.tsv`, `stories/*.tsv`, `drug_expectations.tsv` | Figure 4's a priori gene sets and mechanism stories, hand-curated; each compound's expected pathway, from those tables (`ctx_apriori.py`, whose line-feature checks need DepMap 23Q4) | Figure 4 main |
| `fig4_checks/` | the published pathway tables each Figure 4 producer checks its truth against | Figure 4 main, ED_program_summary |

Two more frozen inputs ship with the data release rather than in git:

| File | What it is | Read by |
|---|---|---|
| `core_results/figure4/progeny/library_size__tahoe.tsv`, `…/progeny_omnipath_academic.tsv` | transcripts per cell of every condition and its controls, per plate; the PROGENy weights, an OmniPath download (URL in `figure4/extract/progeny.py`) | Figure 4 main (and ED_tahoe_hallmark_gsea) |
