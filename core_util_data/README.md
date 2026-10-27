# `core_util_data/`

Inputs that training, evaluation and caching read, grouped by dataset. File and folder NAMES are
stable.

## Layout

```
core_util_data/
├── datasets/        Dataset registry, one YAML per dataset (tahoe, parse): data paths,
│                      conditioning columns, metadata joins. Its stems are the dataset names
│                      configs/<dataset>/ may use. Fixed location (src/ + the config-location rule).
├── splits/          Split definitions, splits/<dataset>/<bio|pert>/<family>/ (+ dataset_split_policies.yaml,
│                      fixed location). Shared with datasets/ as the cross-dataset "interface".
├── tahoe/
│   ├── metadata/       cell_line_metadata, drug_metadata, gene_metadata (Tahoe token map),
│   │                   quadruplet_summary, enhanced_quad, cell_line_cell_name_maps.json
│   ├── gene_subsets/   STATE_HVG*.txt (2k; +hallmark_5750 = union5750; +progeny_3066, pad3072; +readouts), tahoe_maxline10_*
│   └── eval_inputs/    frozen tables the level-3 drivers read as shipped (its README lists each and its rule)
├── parse/
│   ├── metadata/       pbmc_duplet.parquet (donor × cytokine counts)
│   └── gene_subsets/   pbmc_real_selected_genes.* (parse 2k), parse_biological_discovery_geneset.*,
│                       parse_geneset_63a547306aa0e6cd__n_2000_genes.txt
└── shared/
    ├── gene_subsets/   total_hallmark_genes* (hallmark universe, both datasets),
    │                   progeny_top100_genes_resolved1287.txt
    └── msigdb_v2026.1.Hs_json_files_to_download_locally/   not shipped: unpack MSigDB v2026.1.Hs's JSON bundle here
```

`splits/` mirrors `configs/` and `runs/` at the dataset/split level
(`<dataset>/<bio|pert>/`). Splits are shared across roles (main, scaling, ablation), so they
stop one level above the role folders.

## Two rules when touching splits

1. **Never rename a split file.** Run names embed a tag derived from the split's file stem
   (`src/training/utils.py`: `tahoe_legacy_bio_holdout` → `biohold`).
2. **Moving a split is safe for caches, a stale path is not an error you can miss.** Eval-cache
   and artifact keys hash the split's CONTENTS (`split_config_hash`), so a moved file keeps its
   key; a path that no longer exists raises `FileNotFoundError` instead of minting a new key.
   Saved run configs (`runs/**/config_resolved.yaml`) carry the split path — update them with
   the move (re-scoring reloads them).

## How splits work

A training config names **one** split YAML via `split_config:`. The split names a `dataset:`
registry entry and declares `validation:` / `test:` / `train:` blocks (schema:
`src/data/splits/data_class.py::SplitConfig`):

```yaml
dataset: tahoe
controls_to_train: true        # control rows always stay in train
test:
  cell_name: [A549, HepG2/C3A] # AND across keys; a list is OR within a key
validation:
  cell_name: false             # false / none -> force-empty
train:
  sample: all                  # or: sample: hbss + fraction: 0.05
```

Several AND-blocks OR'd together: give a list of dicts under `test:` / `validation:`.

## Adding a split

1. Generate or author it under `splits/<dataset>/<bio|pert>/<family>/` (generator:
   `scripts/data/make_splits.py`).
2. Point a config at it: `split_config: core_util_data/splits/<dataset>/<bio|pert>/<family>/<name>.yaml`.
3. When its runs are archived, move the family to `archive/core_util_data/splits/` and rewrite the
   references (configs, scripts, `config_resolved.yaml` of its runs).
