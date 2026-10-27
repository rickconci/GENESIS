# The datasets, their caches and the splits

Training and scoring read two public datasets through caches built beside them. Drawing the figures reads neither.

## The datasets

**Tahoe-100M.** 50 cancer cell lines exposed to 379 compounds at three doses across 14 plates; a condition is a cell
line, compound, dose and plate. The experiments use the per-plate h5ad files of
[Tahoe-100M](https://www.biorxiv.org/content/10.1101/2025.02.20.639398) (2025-02-25 release,
`plate*_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad`), distributed through the
[Arc Virtual Cell Atlas](https://docs.lamin.ai/arc-virtual-cell-atlas) and on
[Hugging Face](https://huggingface.co/datasets/tahoebio/Tahoe-100M).

**PBMC cytokine atlas.** 12 donors, each on its own 96-well plate of 90 cytokine stimulations and six PBS control
wells, annotated into 18 cell types; a condition is a donor, cytokine, cell type and well. This is Parse Biosciences'
[10 million human PBMCs in a single experiment](https://www.parsebiosciences.com/datasets/10-million-human-pbmcs-in-a-single-experiment/).

`configs/roots.yaml` is the one machine-specific file. It ships with placeholders; set the two dataset roots (or point
`PERTGEN_ROOTS` at your own copy):

```yaml
data_tahoe: /path/to/tahoe100M/h5ad            # Tahoe-100M plate h5ads; build_cache.py writes the caches here
data_parse: /path/to/parse_pbmc_cytokines/10m  # PBMC cytokine h5ads
```

Every other path is relative to the repository root, or names a root as a template (`{data:tahoe}/…`), which
`uv run python -m src.paths '<template>'` expands.

## Build the caches

`build_cache.py` builds a dataset's caches from its h5ads and a training config, which names the data, the gene set,
the condition columns, the control and the split:

| Subcommand | Builds |
|---|---|
| `train` | the HVG training cache (memory-mapped CSR shards) |
| `eval` | the evaluation truth: one NPZ per condition, plus the control pools every scorer reads |
| `summary` | one row per condition with its cell count, the input to `make_splits.py` |
| `stats` | per-condition log1p means and cell counts in two seeded halves (copy_closest's input) |

Training builds the caches its own config names when they are missing. The level-3 drivers need these nine, built
once into the dataset roots (no builder records its disk or time cost). `--all-conditions` writes every condition, so
one namespace serves every split.

```bash
B="uv run python scripts/data/build_cache.py"
T=$(uv run python -m src.paths '{data:tahoe}'); P=$(uv run python -m src.paths '{data:parse}')
MAIN=configs/tahoe/bio/main/mlp_h4096x8192_nrs256_vg05_5ep_state2k.yaml
U5750=configs/tahoe/bio/biology/union5750_conds015/mlp_h4096x8192_nrs256_vg03rs256_5ep_union5750_conds015.yaml
P3072=configs/tahoe/bio/biology/progeny3072_cells005_keepctrl/mlp_h4096x8192_nrs256_vg05rs256_25ep_progeny3072_cells005_keepctrl.yaml
PBMC=configs/parse/bio/main/mlp_h4096x8192_nrs256_vg05rs256_5ep_parse_bio_cytokines.yaml
PBMC_EVAL=configs/parse/bio/eval_cache/eval_cache_parse_pbmc2k_celltype_well.yaml
PBS=core_util_data/splits/parse/bio/parse_bio_well_split/parse_bio_well_hbss100_pbs_donorholdout_es0.yaml
```

| # | Command | Writes, under the dataset root | Read by |
|---|---|---|---|
| 1 | `$B eval --config $MAIN --all-conditions` | `eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000` | every Tahoe driver of Figures 1-3 |
| 2 | `$B train --config $MAIN` | `hvg_cache/geneset_c0d6420ffc0fa861__n_2000` | copy_closest, Figure 3 main, ED_12cells |
| 3 | `$B stats --cache-dir $T/hvg_cache/geneset_c0d6420ffc0fa861__n_2000 --columns cell_name drug dose plate` | its `condition_stats/` | copy_closest |
| 4 | `$B eval --config $U5750 --all-conditions` | `eval_cache/…/geneset_e639c3a3c608d54a__n_5750` | ED_tahoe_hallmark_gsea |
| 5 | `$B eval --config $P3072 --all-conditions` | `eval_cache/…/geneset_12aead7e0e3f60fa__n_3072` | Figure 4 main, ED_program_summary |
| 6 | `$B train --config $P3072` | `hvg_cache/geneset_12aead7e0e3f60fa__n_3072` | Figure 4 main (the Additive fit), ED_program_summary |
| 7 | `$B eval --config $PBMC_EVAL --all-conditions --control-split $PBS --control-unit-column bc1_well` | `eval_cache/…/geneset_63a547306aa0e6cd__n_2000__cell_type__well` | every PBMC driver |
| 8 | `$B train --config $PBMC` | `hvg_cache/geneset_63a547306aa0e6cd__n_2000` | copy_closest |
| 9 | `$B stats --cache-dir $P/hvg_cache/geneset_63a547306aa0e6cd__n_2000 --columns donor cytokine cell_type bc1_well` | its `condition_stats/` | copy_closest |

Command 7's `--control-split` writes one control pool per PBS well, pools each donor and cell type from its training
wells only, and writes the held-out wells as test truth.

## The splits

The held-out splits ship in `core_util_data/splits/` ([core_util_data](../../core_util_data/README.md) explains their
format). `make_splits.py` regenerates one split family from its policy in
`core_util_data/splits/dataset_split_policies.yaml`; `--check` regenerates it into a temporary folder and compares it
with the shipped files, writing nothing:

```bash
uv run python scripts/data/make_splits.py --dataset tahoe --family tahoe_overlap_bio --check
```

The shipped files are canonical (their contents are hashed into the evaluation cache keys), so `--out-dir` refuses a
family's own folder unless given `--overwrite`.
