# The scoring engine

Four command-line tools and one shell environment. The level-3 drivers
(`scripts/figures/figureN/score/<panel>/score.sh`, see [scripts/figures](../figures/README.md)) call them, and are the
record of every published score's exact arguments: read a driver to see how a number was made. Each tool's `--help`
documents its options. Condition, replicate, control and control-match columns come from the dataset registry
(`core_util_data/datasets/<dataset>.yaml`), so one tool serves both datasets.

## `score.py`: one arm against the truth

Scores one arm against the evaluation truth cache and writes `<out-dir>/metrics_summary.json` (with
`--gene-sidecars`, also a per-gene parquet). The metrics are a named set from `configs/eval/metrics.yaml`
(`--metric-set`). One subcommand per kind of arm:

| Subcommand | The arm |
|---|---|
| `genesis` | a GENESIS checkpoint, generated and scored in one streaming pass; the run's config names the data |
| `npz` | an external model's predictions, one `.npz` per condition (STATE, CellFlow, PerturbDiff) |
| `baseline` | a fitted linear baseline (`scripts/baselines/fit.py` writes one `<kind>.npz` per kind) |
| `copy_neighbor` | copy_closest: the closest training context's cells for the test perturbation |
| `copy_lines` | no score: one closest training context per held-out context, as a TSV (Figure 4's copy baseline) |
| `no_response` | each condition's own matched controls as the prediction: how a condition an arm failed on is scored |

```bash
uv run python scripts/eval/score.py genesis --run-dir <run> --checkpoint last --eval-split test --out-dir <dir>
```

## `generate.py`: persist GENESIS cells

For arms scored outside the streaming pass. Both subcommands need a CUDA GPU.

- `conditions`: one `.npz` per test (or `--eval-split validation`) condition of the run's split, at the measured
  cell count, named by its condition values. Fields: `predictions` (cells x genes, float32, in the run's output
  space), `gene_names`, `normalization`, and one field per condition column. One seeded stream (`--seed`, default 0)
  is consumed in file-name order, so only the full ordered run reproduces the published cells; `--limit N` writes the
  first N files. It reads only the run folder.
- `replicates`: one `.npy` per (condition, replicate) of a scored pair table (`--counts-from`), or of a condition
  list with its truth cache (`--conditions` with `--eval-cache`), at that replicate's real size. Each file is one
  draw of its own seed, so it is reproducible alone (on the same GPU model) and a killed run resumes by skipping
  what exists.

```bash
uv run python scripts/eval/generate.py conditions --run-dir <run> --out-dir <dir>
```

## `reference.py`: how close two real measurements are

| Subcommand | What it scores | Writes |
|---|---|---|
| `split_half` | one condition's cells split into disjoint halves, scored against each other (the dashed line of every metric-bar panel) | `noise_ceiling.json` |
| `replicate_pairs` | a model against the experimental repeat: one row per pair of replicates of a condition | a pair table |
| `replicate_pairs_baselines` | the four linear baselines against the repeat, on the GENESIS pair list | pair tables, beside it |
| `ladder` | split-half against the repeat at matched cells per arm (Figure 1) | `reps.parquet` |
| `control_swap` | one condition's cells against two different control wells, real and shuffled (PBMC PBS wells) | `pairs.csv`, `summary.json` |
| `well_draws` | the same cells against equal-size control samples drawn within one well and between two | `rows.csv`, `energy.csv`, `strata.csv`, `summary.json` |

Every random draw is seeded per condition, so a rerun is bit for bit and a subset scores as it does in the full run.

## `subset.py`: a small, checkable rerun

A full level-3 run takes GPU-days. With `SUBSET=<dir>`, a driver scores only a seeded sample of conditions frozen in
`<dir>/<name>.json`, and `check` compares the rerun with the published files, so a regression shows up in minutes.

| Subcommand | What it does |
|---|---|
| `draw` | draws the sample from a split (`--n`, `--seed`; `--keys`/`--summary` for finer conditions, `--pairs` for replicate pairs) |
| `query` | turns a sample into `index in [...]` over a run's or a summary's rows, for `score.py --conditions-query` |
| `summary` | the sample's rows of the summary it was drawn from, for `--condition-summary` |
| `pairs` | the drawn replicate-pair rows, for `reference.py replicate_pairs --conditions` |
| `failed` | the summary's rows of the conditions some arm failed, for `score.py no_response` |
| `check` | compares a produced tree with the published one (`--only <rel>...`: just those files; `--new-sample`: means only); exit 1 on any difference |

## `_env.sh`: the drivers' shell environment

Every driver sources it. It finds the repository root, refuses an `OUT` that is inside `core_results/` or
`core_deliverables/` or would place files into them, records the mode of an `OUT` in `$OUT/.mode` (a full run refuses
a `SUBSET` run's `OUT`, and the reverse), and checks that `PY` imports torch and `src` before anything slow. Drivers
are written in its helpers:

| Helper | What it does |
|---|---|
| `run <done-file> <cmd...>` | runs a step, then marks it done; a rerun skips finished steps. Under `DRY_RUN`, prints the command and lists each path argument that does not exist |
| `place <produced> <rel>` | copies `$OUT/<produced>` to `$OUT/core_results/<rel>`, the same `<rel>` as under `core_results/` |
| `finish` | lists what was placed; under `SUBSET`, checks those files against `core_results/` (`subset.py check --only`); otherwise prints the whole-tree compare |
| `need_run <glob> [<inside>]` | the one run folder matching `<glob>` (that holds `<inside>`, e.g. `checkpoints/last.ckpt`); none or several stop the driver |
| `need_path <path>` | a cache or input a producer reads by itself, which no command line names; missing stops the driver |
| `need_gpu` | stops a driver that needs a GPU when `GPU` is unset (not under `DRY_RUN`) |

Under `DRY_RUN`, `need_run` and `need_path` report what is missing and go on, and `finish` exits 2 if anything was.
