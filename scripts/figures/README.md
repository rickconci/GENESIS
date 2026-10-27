# Reproduce the figures, level by level

Every published number rebuilds at four levels, each from the outputs of the level below it. Start at whichever
level you have the inputs for, and run every command from the repository root.

| Level | From → to | Every figure | One figure or panel |
|---|---|---|---|
| 1 | `core_deliverables/data/` → the 20 PNGs | `uv run python -m scripts.figures.draw_all` | `uv run python -m scripts.figures.figureN.draw.<panel>.draw` |
| 2 | `core_results/` → `core_deliverables/data/` | `uv run bash scripts/figures/extract_all.sh` | `uv run python -m scripts.figures.figureN.extract.extract` |
| 3 | GENESIS checkpoints → `core_results/` | `OUT=<dir> GPU=<i> uv run bash scripts/figures/score_all.sh` | `OUT=<dir> GPU=<i> uv run bash scripts/figures/figureN/score/<panel>/score.sh` |
| 4 | configs → GENESIS checkpoints | `uv run python -m src.main --config <cfg> devices=1` | the config of one run ([configs](../../configs/README.md)) |

Level 1 needs only this repository and `uv sync`; levels 2-4 need `uv sync --extra model`. Level 2 runs on CPU,
levels 3 and 4 need a CUDA GPU.

## Layout

Each figure's code for levels 1-3 lives in `scripts/figures/figureN/`, one folder per panel, where `<panel>` is `main`
(the main figure) or `ed_<name>` (one Extended Data figure):

```
figureN/
  draw/<panel>/draw.py      level 1: one figure from its tables (draw/data.py, draw/panels.py: shared by its panels)
  extract/extract.py        level 2: every table of the figure from core_results/
  score/<panel>/score.sh    level 3: the driver that rescores the panel (score/*.py: the producers it calls)
```

A panel with no `score/` folder draws from its figure's main results. The scoring engine the drivers call is
[scripts/eval](../eval/README.md).

## Level 1: draw the figures

`draw_all` draws all 20 PNGs (4 main and 12 Extended Data figures; one has a page per held-out cell line) into a
temporary folder in a few minutes, and compares every pixel with the committed copies in `core_deliverables/`. It
prints `20 of 20 PNGs pixel-identical` and exits non-zero on any difference; `--out DIR` keeps the drawings. One
panel's draw redraws in place, into `core_deliverables/figureN/`. Keep matplotlib at its pin: text rendering changes
between releases, and the check compares pixels.

## Level 2: rebuild the figure tables

`extract_all.sh` rebuilds every table from `core_results/` in memory and compares it with
`core_deliverables/data/figureN/MANIFEST.tsv` (bytes and sha256); it writes nothing unless given `--write`, which
overwrites the files that differ and their manifest rows.

## Level 3: rescore the checkpoints

Each driver's header names its inputs (`In:`), its outputs (`Out:`), the drivers it needs first, what it reads as-is,
and its cost. `score_all.sh` runs all 13 in dependency order (Figure 2 before Figures 3 and 4, which read its
results). The environment controls every driver:

- `OUT=<dir>` (required): a driver writes only there, and `$OUT/core_results/` mirrors `core_results/` path for path;
  `core_results/` itself is only read. A rerun into the same `OUT` skips the steps that finished. One `OUT` holds one
  mode: a full run refuses an `OUT` that holds a `SUBSET` run, and the reverse.
- `GPU=<i>` picks the GPU (`CUDA_VISIBLE_DEVICES`); every driver but the two CPU ones (ED_pbmc_wells,
  ED_program_summary) refuses to start without it, except under `DRY_RUN`.
  `PY=<python>` picks the interpreter (default: `python`, the environment's under `uv run`).
- `DRY_RUN=1` prints every command and lists every missing input, cache and run folder (`{data:…}` paths expanded):
  each path a command names, and each cache a producer reads by itself (the driver's `need_path` lines). It goes on
  past each, runs nothing and needs no GPU; a driver that found any exits 2. It does not look inside a run folder
  (the files [below](#what-the-data-release-holds)). Run
  `DRY_RUN=1 OUT=/tmp/genesis_dry uv run bash scripts/figures/score_all.sh` before spending GPU time.
- `SUBSET=<dir>` draws a seeded sample of test conditions into `<dir>` (once; about 50 per panel, fewer where a header
  says so), reruns only those, and compares each file the driver placed with `core_results/`
  (`scripts/eval/subset.py check --only`, exit 1 on any difference). A driver that places nothing, or only
  [new samples](#new-samples), passes with a note. Without `SUBSET` a driver reruns in full and prints the whole-tree
  compare, new samples already listed:
  `uv run python scripts/eval/subset.py check $OUT/core_results core_results --new-sample-file $OUT/.new_sample`.

`score_all.sh` runs every driver even when one fails, and ends with one line per driver: `ok`, `MISSING <n>` (a dry
run found n missing paths) or `FAIL <exit code>`; it exits non-zero unless every driver is `ok`.

### Exactness

The check is exact on the GPU model that made the published file: an H100 for GENESIS, copy_closest and the
references, an A100 for Figure 2's baselines (each GPU driver's header says which); another model differs around the
fifth significant digit. Figure 2 main holds both groups, so check it once on each model, or expect the other group
to differ. Figure 2's two no-response files, scored on CPU, are compared within 6e-6.

### New samples

The files a driver lists in `NEW_SAMPLE` were published from GENESIS draws without per-condition noise seeds, so a
rerun draws a new sample of the same model: the check prints their published and rerun means and does not compare
them. They are Figure 2a's GENESIS pair table and ED_objective's GENESIS pairs; every Figure 3b point and ED_12cells'
5%-cells and 25%-cells K = 3 arms; every ED_scaling_pert_split point; 5 of ED_genesis_settings' 9 runs (GENESIS and
λ = 0, 0.1, 0.3, 1); and ED_tahoe_hallmark_gsea.

### What the data release holds

`core_results/figureN/` holds every published score, the comparators' (STATE, CellFlow, PerturbDiff) among them; the
drivers read those as-is. Each GENESIS run under `runs/` holds the files the drivers read:

| File in the run folder | What it is | Read by |
|---|---|---|
| `checkpoints/last.ckpt` | the trained weights | every GENESIS score |
| `config_resolved.yaml`, `encoder_idx_to_name.json`, `output_gene_names.txt` | the run's config, condition vocabularies and gene order | loading the model |
| `eval/eval_manifest.json`, `eval/condition_summary.parquet` | the run's split and its cells per condition | `score.py`, `generate.py conditions`, `subset.py query`; Figure 2's PBMC panels |
| `eval/after_fit/test/metrics_summary.json` | the after-fit test evaluation | ED_pert_generalization (its census) |
| `run.log` | the training log (cells trained on) | Figure 3 main, ED_12cells, ED_scaling_pert_split |
| `split_cache/train_idx.npy` | the training rows | Figure 3 main, ED_12cells, Figure 4 main |
| `hbss_selected_triplets.parquet` (subsampled runs) | the conditions the run kept | ED_tahoe_hallmark_gsea |

`runs/parse/bio/scaling/full_100` is a link to `../main`: ED_scaling_pert_split's arm table names the PBMC 100% arm by
that path. Keep the link when copying `runs/`.

The drivers also read the dataset caches ([scripts/data](../data/README.md) builds them) and a few small tables as
shipped: [core_util_data/tahoe/eval_inputs](../../core_util_data/tahoe/eval_inputs/README.md) lists each one, the
rule that made it and the driver that reads it.

### From level 3 back to level 2

The extracts read only `core_results/`. To rebuild the tables from your own scores, back up the published files, copy
`$OUT/core_results/` over them, and run level 2:

```bash
cp -a core_results core_results.published && cp -a "$OUT/core_results/." core_results/
uv run bash scripts/figures/extract_all.sh
```

## Level 4: train

Each published run has its config under `configs/` ([configs](../../configs/README.md) says which config makes which
figure's runs). To score your own run at level 3, edit the driver that scores the published one: each names its runs
near the top, in a `need_run` line (`RUN=$(need_run '<run folder>' checkpoints/last.ckpt)`; `T_RUN=` and `P_RUN=` in
Figure 2's) or in a table of runs (`SPLIT_HALF=(…)` in Figure 3 main, `RUNS=(…)` in ED_genesis_settings). Replace the
published folder, or its timestamp, with yours; a `SUBSET` check then shows where your model's scores differ from the
published ones.

## Cost

Approximately, from the drivers' `Cost:` lines and the published runs (— = not recorded):

| Level | Hardware | Time | Disk, beyond the repository | RAM |
|---|---|---|---|---|
| 1 | any CPU | ~5 min | — | — |
| 2 | CPU | ~10 min for Figures 1-3, ~30 min for Figure 4 | `core_results/`, ~5.5 GB | — |
| 3, `SUBSET` | an H100 (an A100 to check Figure 2's baselines exactly) | a few hours | the datasets, their caches and `runs/` | — |
| 3, full | the same | ~3-4 GPU-days, plus 12-24 CPU-hours (ED_pbmc_wells) | the same, plus ~300 GB under `OUT` (Figure 4's two persisted predictions, ~190 and ~95 GB) | ~90 GB (Figure 4's CPU steps) |
| 4 | one 94 GB H100 per run (some recipes run out of memory on an 80 GB A100) | ~4-43 h per run (the main Tahoe GENESIS ~43 h); 67 configs | its caches; the main run's 15 checkpoints take ~10 GB | — |
