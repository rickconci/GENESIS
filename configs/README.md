# Training configs

One YAML per training run of the paper. Its header says what the run is and how it was launched; its keys carry the
run's name, split, gene set and every training setting. `src/main.py` reads the dataclass defaults
(`src/config/config.py`), then the YAML, then any `key=value` overrides after it:

```bash
uv run python -m src.main --config configs/tahoe/bio/main/mlp_h4096x8192_nrs256_vg05_5ep_state2k.yaml devices=1
uv run python -m src.main --config configs/tahoe/bio/main/mlp_h4096x8192_nrs256_vg05_5ep_state2k.yaml devices=1 use_wandb=false
```

`run_name_suffix` names the run's folder, and the folder mirrors the config's under `configs/`: the config above
trains into `runs/tahoe/bio/main/<run_name_suffix>/<timestamp>_…/`. `devices=2` trains on two GPUs (DDP). A stopped
run resumes from its own folder:

```bash
uv run python -m src.main --config <run>/config_resolved.yaml --workdir <run> resume_ckpt=<run>/checkpoints/last.ckpt
```

The variogram-weight ablation has one config for its four runs,
`configs/tahoe/bio/ablation/variogram/mlp_h4096x8192_nrs256_5ep_state2k.yaml`: each run sets `variogram_loss_weight`
and its `run_name_suffix` at launch, as its header lists (λ = 0.5 is the main config above).

## Which configs make which figure's runs

Paths are under `configs/`; a folder means every config in it. The scoring driver of each panel names the run it
reads ([scripts/figures](../scripts/figures/README.md)).

| Figure | Configs |
|---|---|
| Figure 2 main, ED_metrics, ED_split_coverage | Tahoe: `tahoe/bio/main/mlp_h4096x8192_nrs256_vg05_5ep_state2k.yaml`; PBMC: `parse/bio/main/mlp_h4096x8192_nrs256_vg05rs256_5ep_parse_bio_cytokines.yaml`, and `parse/bio/main/mlp_h4096x8192_nrs256_vg05rs256_5ep_parse_bio_donorholdout.yaml` for its per-well condition summary |
| ED_genesis_settings | the Tahoe main config, `tahoe/bio/ablation/variogram/` (λ = 0, 0.1, 0.3, 1), `tahoe/bio/ablation/n_repeats/` (m = 16, 64), `tahoe/bio/ablation/depth/` |
| ED_conditioning | `tahoe/bio/ablation/cond_input/` |
| ED_objective | `tahoe/bio/ablation/mmd_loss/` and its GENESIS partner, `tahoe/bio/scaling/n_cells_025/` |
| ED_pbmc_wells | the PBMC donor-holdout config (its condition summary) |
| ED_pert_generalization | `tahoe/pert/main/`, `parse/pert/main/`; the census reads `tahoe/pert/scaling/overlap_celln10_fixed/entity_seed0/…_r0_es0.yaml`'s run |
| Figure 3a | the Tahoe main config and its two model seeds, `tahoe/bio/main/…_mseed65.yaml` and `…_mseed66.yaml`; the `_es0`-`_es2` configs of `tahoe/bio/scaling/n_conds_025/` and `n_conds_005/`; `tahoe/bio/scaling/n_cells_025_keepctrl/`, `n_cells_005_keepctrl/` |
| Figure 3b | `tahoe/bio/scaling/overlap_celln10/` |
| ED_12cells_per_condition | `tahoe/bio/scaling/n_cells_001/`, `n_cells_005/`, `n_cells_025/`, `n_cells_percond25/`, the `mlp_h4096x8192_nrs256_vg05_20ep_state2k.yaml` of `n_conds_025/` and of `n_conds_005/`, and the Tahoe main config |
| ED_scaling_pert_split | `tahoe/pert/scaling/overlap_celln10_fixed/`; `parse/bio/scaling/` and the PBMC donor-holdout config (the 100% arm) |
| Figure 4 main, ED_program_summary, ED_celllines_per_model | `tahoe/bio/biology/progeny3072_cells005_keepctrl/` |
| ED_tahoe_hallmark_gsea | `tahoe/bio/biology/union5750_conds015/` |

Figure 1 main scores the measured cells only, with no checkpoint.
`parse/bio/eval_cache/eval_cache_parse_pbmc2k_celltype_well.yaml` trains nothing: it is the config that builds the
PBMC evaluation cache ([scripts/data](../scripts/data/README.md)).

## The other files

- `roots.yaml`: the one machine-specific file. `data_tahoe` and `data_parse` name the dataset folders (set them before
  training or scoring); the other keys locate model outputs made outside this repository (State's predictions under
  `ml_outputs`, the comparators' for rescoring them). It ships with placeholders; `PERTGEN_ROOTS` points at another
  copy.
- `eval/metrics.yaml`: the named metric sets the scorers compute (`core`: the three headline metrics; `extended`:
  every metric, what the published scores hold).
- `eval/arms.yaml`: every bar of a published metric-bar panel, the file its scores live in, and each panel's bars and
  references; Figure 2's extract reads it.
