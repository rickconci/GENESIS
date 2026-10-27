#!/usr/bin/env python3
"""Per-gene DEG calls for the gated conditions of one drug, every arm.

Goes from a scalar overlap score to the
*named genes underneath it*: for a pre-registered MEK-readout set, is each gene
called DE, with which sign, and at what log2FC in truth vs each model?

PRE-REGISTRATION.  ``READOUT`` below was fixed after looking at ONE slice
(Trametinib, dose 5.0 uM, plate 3) and at no other. Every other slice — and all
of Radotinib — is out-of-sample for that choice.

THE REPLICATE IS THE YARDSTICK.  Each condition is measured on two plates, so
truth has its own reproducibility. ``replicate`` below scores truth-plate-q as
if it were a prediction of truth-plate-p, through the identical code path as
every model. Without it "GENESIS recovers EGR1" has no scale: the question is
not whether a model matches truth, it is whether a model matches truth about as
well as a second measurement of the same condition does.

Contract mirrored from the eval scorers, not re-derived
-------------------------------------------------------
* The four neural arms are ``log1p``: STATE / PerturbDiff / CellFlow predictions are
  scored as ``ModelOutputSpace.LOG1P`` (src/eval/arms/npz.py), and GENESIS's
  ``metrics_summary.json`` declares ``log1p``.
* ⚠️ The four linear baselines are ``raw``: fit with ``normalization_id="raw_counts"``
  and emitted as ``ModelOutputSpace.RAW`` (src/eval/arms/baseline.py).
  Their per-cell cloud is NOT a tiled mean; it is ``emit_per_cell_delta``
  (imported, not reimplemented), which shifts real plate-matched control cells
  by the baseline's claimed delta in log1p-CPM. Tiling instead would change the
  DEG answer, because a Wilcoxon over identical rows is degenerate.
* DEG lives in ``raw_int_rgs``: predictions snapped onto the count lattice by
  ``to_raw_integer``, truth/control already integer.
* Gate = pooled-control detection >= 0.10 (q := 1 for genes below it), then
  ``q < 0.001 and |log2FC| >= 0.1``.  Identical to ``compute_deg_overlap_metrics``.

Two arms per (condition, truth plate)
-------------------------------------
``cross``  truth on plate p vs the model's cells for plate q != p, each with its
           OWN plate's DMSO pool.  This is the headline: it is what
           ``cross_mean`` scores, the honest comparison. ``replicate`` exists only here — on ``same`` it
           would be truth against itself.
``same``   truth and prediction both on plate p.  Easier; reported for reference
           and used as the numerical baseline against the eval run's own parquet.

Usage::

    python src/eval/readout/deg_readout.py --drug Trametinib --out-dir <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.eval.reference.pairs import (
    EvalNpzReader,
    read_control,
    read_perturbed,
)


# Read through the shared eval-cache reader, which also handles the plate-id variants.
def load_perturbed(ns: Path, *, cell_name: str, drug: str, dose: float, plate: str,
                   n_genes: int) -> np.ndarray:
    keys = {"cell_name": cell_name, "drug": drug, "dose": str(float(dose)), "plate": str(plate)}
    x = np.asarray(read_perturbed(EvalNpzReader(Path(ns), cache_size=4), keys, "plate"), np.float32)
    if x.shape[1] != n_genes:
        raise ValueError(f"{keys}: {x.shape[1]} genes, expected {n_genes}")
    return x


def load_control(ns: Path, *, cell_name: str, plate: str, n_genes: int) -> np.ndarray:
    keys = {"cell_name": cell_name, "plate": str(plate)}
    x = np.asarray(read_control(EvalNpzReader(Path(ns), cache_size=4), keys, "plate"), np.float32)
    if x.shape[1] != n_genes:
        raise ValueError(f"{keys}: {x.shape[1]} genes, expected {n_genes}")
    return x


# The baseline emission is imported, never reimplemented: it is the difference
# between scoring these baselines as they were published and inventing a new,
# more flattering (or more damning) version of them.
from src.eval.arms.baseline import (
    _stable_seed,
    emit_in_fit_space,
    emit_per_cell_delta,
    normalize_plate_for_eval,
)
from src.data.vocab import to_categorical_str
from src.data.data_utils import safe_filename_part
from src.paths import expand_path
from src.baselines.artifacts import Baseline
from src.baselines.roles import ArmKey
from src.baselines.space import RAW, checked_library, to_fit_space
from src.eval.metrics.deg import _benjamini_hochberg_torch, _compute_wilcoxon_vectorized
from src.eval.metrics.spaces import (
    _project_predictions_to_spaces,
    _project_raw_to_spaces,
    set_dataset_median_umi,
)

# ---------------------------------------------------------------------------
# The experiment: what we score, and what we pre-registered
# ---------------------------------------------------------------------------

EVAL_CACHE = Path(expand_path("{data:tahoe}/eval_cache/normalization_raw_counts/geneset_c0d6420ffc0fa861__n_2000"))
GENE_LIST = REPO / "core_util_data" / "tahoe" / "gene_subsets" / "STATE_HVG.txt"
# The canonical gene axis: every 2k arm's predictions and the eval cache use this exact
# file, so it defines their column order. Read at import so `load_pred` can assert against it.
GENES: list[str] = [ln.strip() for ln in GENE_LIST.read_text().splitlines() if ln.strip()]
ML = Path(expand_path("{root:ml_outputs}"))
PD_ROOT = Path(expand_path("{root:perturbdiff}/tahoe_sampling_from_scratch/per_plate"))

# The tahoe_bio baselines, fit on all training cells of the split (as arms.yaml's tahoe_bio arms)
BASELINE_ROOT = REPO / "results/tahoe/baselines/baseline_artifacts_bio_allcells"

# Pre-registered MEK/ERK-output readout. EGR1 + IER3 are immediate-early genes and
# direct ERK->ELK1/SRF targets; DUSP4/SPRY1 are ERK-feedback genes; TMEM158 (RIS1) is
# RAS-induced. These five are the ONLY MAPK-output genes the 2,000-gene panel carries.
READOUT = ("EGR1", "IER3", "DUSP4", "SPRY1", "TMEM158")

# The gated conditions, from batch_resolution's screen. Both drugs are held out on
# the same five cell lines; Trametinib is the only one with a dose series.
CELL_LINES = ("C32", "HOP62", "HepG2/C3A", "Hs 766T", "PANC-1")
DRUG_PLATES: dict[str, dict[float, tuple[str, str]]] = {
    "Trametinib": {0.05: ("1", "10"), 0.5: ("2", "11"), 5.0: ("3", "12")},
    "Radotinib": {5.0: ("6", "14")},
}

def gated_dose_plates(drug: str) -> dict[float, tuple[str, str]]:
    """dose -> (plate_i, plate_j) for a drug, from Figure 2a's published pair table (its truth columns).

    DRUG_PLATES above is a hardcoded two-drug map. Any other drug's plate pairs
    are read from the screen that defines them rather than re-typed, and only
    pairs that clear the 0.2 gate are returned -- scoring an ungated pair would
    put a condition in the readout that the screen excludes from every other number.
    """
    import pandas as _pd

    f = REPO / "core_results/figure2/plate_pairs/tahoe_bio/pair_metrics__genesis.parquet"
    d = _pd.read_parquet(f)
    d = d[(d.drug == drug) & (d.true_true__deg_adaptivek_truth >= 0.2)]
    if d.empty:
        raise SystemExit(f"{drug!r}: no gated plate pairs in {f.name}")
    return {float(r.dose): (str(r.plate_i), str(r.plate_j))
            for r in d.itertuples()}


MODEL_SPACE = "log1p"
BASELINE_SPACE = "raw"   # the baselines are fit with normalization_id="raw_counts"
BASELINE_SEED = 42       # score.py baseline --seed default
CTRL_DET_FLOOR = 0.10
FDR = 0.001
LFC_THR = 0.1

ARMS = {  # (label, prediction root, format) — the definitions in tables/arms.csv
    # ⚠️ The SAME GENESIS (this vg0.5 checkpoint) as batch_resolution, so every figure
    # quotes one model.
    "genesis": (
        REPO / "results/tahoe/split_half/pertgen"
        / ("260916_114938_cur-between_epoch-tp458752-bs1792xnrm256_nre25"
           "6_lr0.0009_wd0.0001_biohold_condmlp-D256-h4096x8192_cond-pla"
           "te_mlpout-relu_vg0.5p0.5np6k_log1p_s64_h4096x8192_nrs256_vg05rs256_5ep_state2k_tahoe_bio_100cond_seed0__last__test")
        / "predictions",
        "npy",
    ),
    "state": (ML / "tahoe_eval_state_v4_cp10k_regen", "npz"),
    "perturbdiff": (PD_ROOT, "npz"),
    "cellflow": (ML / "tahoe_eval_cellflow_bio_gs/final__test", "npz"),
}

# Fitted on the same split as every neural arm: the five cell lines x these drugs are
# test conditions, so no baseline ever saw the condition it is asked to predict. It
# HAS seen Trametinib in the ~45 training cell lines, which is the point of a
# perturbation-mean baseline — that is transfer, not leakage.
BASELINES = ("perturbation_mean", "linear_ridge", "context_mean", "additive")
TAHOE_AXES = {"held": ["cell_name"], "kept": [], "stratum": ["plate"], "pert": ["drug", "dose"]}


# ---------------------------------------------------------------------------
# Panels: the 2k contract, and the union gene axis beside it
# ---------------------------------------------------------------------------
# ⚠️ THE MODULE GLOBALS ABOVE DO NOT MOVE. Callers read ``ARMS``, ``BASELINES``,
# ``EVAL_CACHE`` and ``GENES`` as module attributes, so rebinding them for a
# ``--panel union5750`` run would silently re-point a 2k figure at 5,750-gene
# predictions -- read through a 2,000-long positional index, i.e. every gene
# mislabelled with no error raised anywhere. The union panel is
# therefore an ADDITIONAL entry that ``main`` resolves explicitly, never a mutation
# of the 2k one, and ``PANELS["2k"]`` is the globals themselves rather than a copy.


@dataclass(frozen=True)
class Panel:
    """One gene axis and the arms that can legally be decoded on it."""

    name: str
    cache: Path
    gene_list: Path
    genes: list[str]
    arms: dict[str, tuple[Path, str]]
    baselines: tuple[str, ...]
    # ⚠️ PER PANEL, because the fitted artifacts carry a gene axis. A single global
    # BASELINE_ROOT is what would let a union run load the 2,000-gene artifacts;
    # load_baselines' axis assertion would catch it, but only by raising mid-sweep.
    baseline_root: Path = BASELINE_ROOT


UNION_GENE_LIST = (REPO / "core_util_data" / "tahoe" / "gene_subsets" / "STATE_HVG_plus_hallmark_5750.txt")
UNION_EVAL_CACHE = Path(expand_path("{data:tahoe}/eval_cache/normalization_raw_counts/geneset_e639c3a3c608d54a__n_5750"))
# `last.ckpt`, test split, K=1, predictions persisted for all 4,211 conditions. A .npy carries no gene
# names, so load_pred CANNOT check the gene axis; scripts that read these persists assert it
# against the run's own `metrics_summary.json["gene_names"]`.
UNION_GENESIS = (REPO / "results/tahoe/split_half/pertgen"
                 / "union5750_last_test_k1/predictions")
# 15% HBSS conditions, 33 epochs, last.ckpt. K=1 test persist, the same protocol
# as UNION_GENESIS, so a stripe here is the same object as the full-training one.
UNION_GENESIS_HBSS015 = (REPO / "results/tahoe/split_half/pertgen"
                         / "union5750_hbss015_last_test_k1/predictions")
# Fit on the 5,750 axis from the SAME split YAML as every neural arm
# (tahoe_legacy_bio_holdout: 60,515 train / 4,211 test / 287 val conditions), so the
# four linear arms are comparable here. `perturbation_mean` is the one
# that matters most: it IS the drug's main effect over the training cell lines, i.e.
# the "universality" shortcut turned into a model, and `additive` is that plus a
# context term. A model that cannot beat them has not learned the interaction.
UNION_BASELINE_ROOT = REPO / "results/tahoe/baselines/baseline_artifacts_bio_union5750_fixed"

# STATE_HVG u PROGENy top-100 members, padded to 3,072 (the last 6 genes are pad: drop them
# before any readout). GENESIS = the Fig 3a 5% cells model with only the gene list changed
# (progeny3072_cells005_keepctrl), `last.ckpt`, K=1 test persist; STATE = our retrain on the SAME
# cells and genes (state/runs/st_tahoe_bio_cells005kc_r0_pad3072_t4000, step=200000: 100k was still
# improving, Pearson delta .371 -> .475 at 200k), one NPZ per condition with gene_names,
# which load_pred checks. No baselines: none were fitted on this axis.
PROGENY3072_GENE_LIST = REPO / "core_util_data" / "tahoe" / "gene_subsets" / "STATE_HVG_plus_progeny_pad3072.txt"
PROGENY3072_EVAL_CACHE = Path(expand_path("{data:tahoe}/eval_cache/normalization_raw_counts/geneset_12aead7e0e3f60fa__n_3072"))
PROGENY3072_GENESIS = (REPO / "results/tahoe/split_half/pertgen"
                       / "progeny3072_cells005kc_seed0_last_test_k1/predictions")
PROGENY3072_STATE = ML / "tahoe_bio_progeny3072_eval_state_st4k_step200000"

PANELS: dict[str, dict] = {
    # The 2k contract: every published 2k readout number comes from here, and it must
    # reproduce them byte-for-byte.
    "2k": {"cache": EVAL_CACHE, "gene_list": GENE_LIST, "genes": GENES,
           "arms": ARMS, "baselines": BASELINES, "n_genes": 2000},
    # STATE_HVG u hallmark. GENESIS is the ONLY arm here, on purpose:
    #   * STATE / PerturbDiff / CellFlow have predictions on the 2,000-gene axis only.
    #     There is no union prediction for them and reindexing a 2k dump onto 5,750
    #     genes would invent 3,750 columns.
    # The plate replicate is the yardstick and it needs no model at all.
    "union5750": {"cache": UNION_EVAL_CACHE, "gene_list": UNION_GENE_LIST,
                  "genes": None, "arms": {"genesis": (UNION_GENESIS, "npy")},
                  "baselines": BASELINES, "n_genes": 5750,
                  "baseline_root": UNION_BASELINE_ROOT},
    # The tahoe-bio HBSS 15 % training split: Additive refitted on exactly the 7,933
    # triplets that run kept (scripts/baselines/fit.py --train-subset its
    # hbss_selected_triplets.parquet; 9,073 conditions). GENESIS is that split's
    # 33-epoch run (UNION_GENESIS_HBSS015).
    "union5750_hbss015": {"cache": UNION_EVAL_CACHE, "gene_list": UNION_GENE_LIST,
                          "genes": None,
                          "arms": {"genesis": (UNION_GENESIS_HBSS015, "npy")},
                          "baselines": ("additive",),
                          "n_genes": 5750,
                          "baseline_root": REPO / "results/tahoe/baselines/union5750_hbss015_log1p_cp10k_full"},
    "progeny3072": {"cache": PROGENY3072_EVAL_CACHE, "gene_list": PROGENY3072_GENE_LIST,
                    "genes": None,
                    "arms": {"genesis": (PROGENY3072_GENESIS, "npy"), "state": (PROGENY3072_STATE, "npz")},
                    "baselines": (), "n_genes": 3072},
}


def resolve_panel(name: str) -> Panel:
    """Build one panel, asserting its gene list is the length the panel declares.

    The length check is not ceremony: ``main`` indexes every readout gene
    positionally into whatever this returns, so a gene file that grew or shrank
    since the predictions were written would mislabel genes rather than crash.
    """
    if name not in PANELS:
        raise ValueError(f"unknown panel {name!r}; choose from {sorted(PANELS)}")
    spec = PANELS[name]
    genes = spec["genes"]
    if genes is None:
        genes = [ln.strip() for ln in spec["gene_list"].read_text().splitlines()
                 if ln.strip()]
    if len(genes) != spec["n_genes"]:
        raise ValueError(f"panel {name!r}: {spec['gene_list']} has {len(genes)} "
                         f"genes, expected {spec['n_genes']}")
    return Panel(name=name, cache=spec["cache"], gene_list=spec["gene_list"],
                 genes=genes, arms=spec["arms"], baselines=spec["baselines"],
                 baseline_root=spec.get("baseline_root", BASELINE_ROOT))


# Truth's own second measurement, scored through the identical path as a model.
from src.eval.readout.ab_mix import draw_half_half_indices  # noqa: E402

REPLICATE = "replicate"
# GENESIS A+B: the N/2 + N/2 cell-level mix of the plate-A- and plate-B-conditioned
# generations, i.e. the model-side equivalent of averaging over plates. It answers a
# DIFFERENT question from the single-plate arms and has NO plate-replicate
# counterpart -- a two-plate replicate would need four plates -- so it must not be
# drawn against the replicate line. Mix size, salt (``_ab_salt``) and seed are the
# batch_resolution DEG-overlap draw's, so the per-gene arm and those numbers describe
# the same draw.
GENESIS_AB = "genesis_ab"


def _ab_salt(cell: str, drug: str, dose: float, pi, pj) -> int:
    raw = f"{cell}|{drug}|{float(dose):.8g}|{pi}|{pj}".encode()
    return int.from_bytes(hashlib.blake2s(raw, digest_size=4).digest(), "little")

_NPZ_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


# ---------------------------------------------------------------------------
# Loading: the plate-pair scorer's path builders and prediction guard (reference/pairs.py)
# ---------------------------------------------------------------------------

def pred_path(root: Path, fmt: str, drug: str, cell: str, dose: float, plate: str) -> Path:
    if fmt == "npy":
        keys = {"cell_name": cell, "drug": drug,
                "dose": to_categorical_str(dose), "plate": f"plate{plate}"}
        stem = "__".join(f"{k}={safe_filename_part(str(v))}" for k, v in keys.items())
        return root / f"{stem}.npy"
    # ⚠️ The DRUG is escaped the same way as the cell name. The npz writers ran
    # both through the same sanitiser -- on disk it is
    # 'C32__R_-Verapamil_hydrochloride__5.000000__plate14.npz' -- so interpolating
    # the raw drug name silently misses every punctuated compound. That is 20 of
    # the 76 gated drugs.
    cell_s = _NPZ_SAFE.sub("_", cell).strip("_")
    drug_s = _NPZ_SAFE.sub("_", drug).strip("_")
    return root / f"{cell_s}__{drug_s}__{float(dose):.6f}__plate{plate}.npz"


def load_pred(path: Path, genes: list[str] | None = None) -> np.ndarray:
    """Load predictions and assert they are log1p-scale, not raw counts.

    The npz ``normalization`` label is NOT trusted (PerturbDiff writes
    ``raw_counts`` over log1p values); the magnitude check is the real guard, and
    it is the same one the plate-pair scorer (src/eval/reference/pairs.py) uses.
    """
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        with np.load(path, allow_pickle=True) as z:
            if "predictions" not in z.files:
                raise KeyError(f"{path}: missing 'predictions'; keys={z.files}")
            arr = z["predictions"]
            # Gene axis is positional from here on, so a reordered dump would be
            # silently wrong biology. The npz arm (src/eval/arms/npz.py) reindexes;
            # the plate-pair scorer does not, so verify rather than assume.
            if "gene_names" in z.files:
                names = [str(g) for g in z["gene_names"]]
                expect = GENES if genes is None else genes
                if names != expect:
                    raise ValueError(
                        f"{path}: gene_names disagree with the panel's gene order; "
                        f"positional indexing would mislabel every gene."
                    )
    arr = np.asarray(arr, dtype=np.float32)
    if float(np.nanmax(arr)) > 30.0:
        raise ValueError(
            f"{path}: pred max={float(np.nanmax(arr)):.2f} looks like raw counts, "
            f"but MODEL_SPACE={MODEL_SPACE!r}."
        )
    return arr


def load_baselines(kinds: tuple[str, ...] = BASELINES,
                   genes: list[str] | None = None,
                   root: Path | None = None,
                   ) -> dict[str, Baseline]:
    """Load the fitted baseline artifacts, asserting the gene axis.

    ``kinds=()`` -- a panel with no baselines fitted on its gene axis -- returns ``{}``
    rather than raising. Returning an empty dict is the honest "this panel has no
    linear arms", as opposed to quietly decoding a 2k artifact on another axis.
    """
    expect = GENES if genes is None else genes
    out: dict[str, Baseline] = {}
    for kind in kinds:
        gen = Baseline.load((root or BASELINE_ROOT) / f"{kind}.npz")
        if list(gen.genes) != list(expect):
            raise ValueError(f"{kind}: gene axis differs from the panel's "
                             f"{len(expect)}-gene order")
        out[kind] = gen
    return out


def prediction_space(gen: Baseline) -> str:
    """How ``baseline_pred``'s rows read: ``raw`` counts for a raw fit, ``log1p`` counts for a fit-space one
    (snapped at the geometric cuts, as the split-half scorer reads them)."""
    return "raw" if gen.params.get("fit_space", RAW) == RAW else "log1p"


def baseline_pred(
    gen: Baseline,
    kind: str,
    *,
    drug: str,
    cell: str,
    dose: float,
    plate: str,
    ctrl_raw: np.ndarray,
    n_cells: int,
    ctrl_lib: np.ndarray | None = None,
) -> np.ndarray:
    """One baseline's ``(n_cells, G)`` prediction for one condition, in ``prediction_space(gen)``.

    The same path as the split-half getter (src/eval/arms/baseline.py): plate-matched control mean into
    ``predict``, then the emission the artifact's fit space calls for, with the same stable seed. A raw
    fit goes through ``emit_per_cell_delta``; a log1p_cp10k_full fit through ``emit_in_fit_space``, which
    needs the control cells' full-transcriptome libraries (``ctrl_lib``, row for row). The control pool is
    the same DMSO cells the DEG test uses as its reference, which is what makes these baselines
    comparable to the neural arms rather than to a differently-conditioned quantity.
    """
    if kind != gen.kind:
        raise ValueError(f"asked for {kind}, the artifact holds {gen.kind}")
    if gen.axes != TAHOE_AXES:
        raise ValueError(f"{kind}: fit with roles {gen.axes}; this readout keys Tahoe's {TAHOE_AXES}")
    plate_norm = normalize_plate_for_eval(f"plate{plate}")
    key = ArmKey(held=(str(cell),), kept=(), stratum=(plate_norm,), pert=(str(drug), round(float(dose), 6)))
    seed = _stable_seed((cell, drug, dose, plate_norm), BASELINE_SEED)
    if prediction_space(gen) != "raw":
        if ctrl_lib is None:
            raise ValueError(f"{kind}: a {gen.params['fit_space']} fit needs ctrl_lib, and its rows read as log1p")
        cells = to_fit_space(ctrl_raw, checked_library(ctrl_lib, len(ctrl_raw), f"control {cell} plate {plate}"))
        mu = cells.astype(np.float64).mean(axis=0)
        delta = np.asarray(gen.predict(key, mu), dtype=np.float64) - mu
        rows = emit_in_fit_space(delta=delta, ctrl_cells=cells, ctrl_lib=ctrl_lib, n_cells=n_cells, seed=seed)
        if not np.isfinite(rows).all():
            raise ValueError(f"{kind}: non-finite emission for {cell}/{drug}/{dose}/{plate}")
        return rows
    matched_ctrl = ctrl_raw.mean(axis=0, dtype=np.float64)
    vec = gen.predict(key, matched_ctrl)

    rows = emit_per_cell_delta(
        pred_vec_raw=np.asarray(vec, dtype=np.float32).ravel(),
        ctrl_cells_raw=ctrl_raw,
        ctrl_mean_raw=matched_ctrl,
        n_cells=n_cells,
        seed=_stable_seed((cell, drug, dose, plate_norm), BASELINE_SEED),
    )
    if not np.isfinite(rows).all():
        raise ValueError(f"{kind}: non-finite emission for {cell}/{drug}/{dose}/{plate}")
    return rows


# ---------------------------------------------------------------------------
# The DEG primitive, gated exactly as compute_deg_overlap_metrics gates it
# ---------------------------------------------------------------------------

def deg_stats(ctrl: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    """Wilcoxon + BH on one (control, target) pair in ``raw_int_rgs``.

    LFC is ``log2((mean_t + 1e-9) / (mean_c + 1e-9))`` on integer counts. ⚠️ When a group
    mean is exactly 0 that pseudocount produces ~±23, which is an artifact of the
    formula and not an effect size — ``*_mean`` below is returned so callers can
    detect it.
    """
    z, p, lfc = _compute_wilcoxon_vectorized(ctrl, target)
    q = _benjamini_hochberg_torch(p)
    return {
        "z": z.cpu().numpy(), "q": q.cpu().numpy(), "lfc": lfc.cpu().numpy(),
        "target_mean": target.to(torch.float64).mean(0).cpu().numpy(),
        "ctrl_mean": ctrl.to(torch.float64).mean(0).cpu().numpy(),
        "n_target": int(target.shape[0]),
    }


def call_de(st: dict[str, np.ndarray], keep: np.ndarray) -> np.ndarray:
    """The pipeline's DE call: floor-masked q, then FDR and |LFC| thresholds."""
    q = np.where(keep, st["q"], 1.0)
    return (q < FDR) & (np.abs(st["lfc"]) >= LFC_THR)


def pooled_det_frac(ctrl_a: torch.Tensor, ctrl_b: torch.Tensor) -> np.ndarray:
    """P(count > 0) pooled over both controls — the floor in compute_delta_deg_bundle."""
    n = float(ctrl_a.shape[0] + ctrl_b.shape[0])
    hits = ((ctrl_a > 0).to(torch.float64).sum(0)
            + (ctrl_b > 0).to(torch.float64).sum(0))
    return (hits / max(n, 1.0)).cpu().numpy()


def overlap_summary(t: dict, p: dict, keep: np.ndarray) -> dict[str, float]:
    """Whole-panel context for one (truth, pred) pairing.

    ``deg_adaptivek_truth`` mirrors ``_deg_family_from_stats``'s
    ``deg_overlap_top_p_ranked_k100_..._adaptivek_truth``: rank BOTH arms by q among
    their own significant genes, take the first ``k_eff = min(100, n_sig_truth)`` of
    each, and score the sign-matching intersection over ``k_eff``. Ranking both arms
    is what stops a model that calls the whole eligible universe DE from scoring 1.0.
    """
    sig_t, sig_p = call_de(t, keep), call_de(p, keep)
    n_t, n_p = int(sig_t.sum()), int(sig_p.sum())
    inter = int((sig_t & sig_p).sum())
    k_eff = min(100, n_t)

    def top_mask(sig: np.ndarray, q: np.ndarray) -> np.ndarray:
        ranked = np.argsort(np.where(sig, q, np.inf), kind="stable")[:k_eff]
        m = np.zeros_like(sig)
        m[ranked] = sig[ranked]
        return m

    sign_match = np.sign(t["z"]) == np.sign(p["z"])
    hit = top_mask(sig_t, t["q"]) & top_mask(sig_p, p["q"]) & sign_match
    both = sig_t & sig_p
    return {
        "n_sig_truth": n_t, "n_sig_pred": n_p,
        "jaccard": inter / max(n_t + n_p - inter, 1),
        "deg_adaptivek_truth": float(hit.sum()) / k_eff if k_eff else float("nan"),
        "sign_agree_on_truth_de": (
            float((np.sign(t["lfc"][both]) == np.sign(p["lfc"][both])).mean())
            if both.any() else float("nan")
        ),
        "lfc_pearson_on_truth_de": (
            float(np.corrcoef(t["lfc"][sig_t], p["lfc"][sig_t])[0, 1])
            if n_t > 2 else float("nan")
        ),
    }


def down_rank(lfc: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """1-based rank by log2FC ascending among floor-passing genes; 1 = most down.

    This is the specificity-aware readout. "Is EGR1 called DE" saturates because the
    gate's |log2FC| >= 0.1 is permissive and a model that calls most of the eligible
    universe passes it for free; "where does EGR1 sit in the model's own ranking of
    suppression" cannot be won that way. Genes below the detection floor get rank
    ``n_keep + 1`` so they can never appear to lead.
    """
    n_keep = int(keep.sum())
    rank = np.full(lfc.shape[0], n_keep + 1, dtype=np.int64)
    order = np.argsort(np.where(keep, lfc, np.inf), kind="stable")[:n_keep]
    rank[order] = np.arange(1, n_keep + 1)
    return rank


# ---------------------------------------------------------------------------
# Main: conditions x 2 truth plates x (4 models + 4 baselines + replicate) x 2 arms
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    # Both default to the pre-registered MEK readout and the hardcoded plate map.
    # They exist so OTHER drugs can be scored on OTHER gene sets through this same,
    # already-validated code path rather than forking it.
    ap.add_argument("--with-ab", action="store_true",
                    help="also score GENESIS A+B, the N/2+N/2 cell-level mix of the "
                         "two plate-conditioned generations. ⚠️ It averages over "
                         "plates, so it is NOT comparable to the plate replicate and "
                         "must not be plotted against that reference.")
    ap.add_argument("--ab-seed", type=int, default=0,
                    help="seed of the A+B cell mix (0 = the batch_resolution DEG-overlap draw)")
    ap.add_argument("--genes", default=",".join(READOUT),
                    help="comma-separated readout genes (default: the MEK readout)")
    # ⚠️ EXPLICIT, never inferred. The default is the 2k panel; a union run has to
    # ASK for the union axis. The
    # alternative -- sniffing the panel from the gene names -- is exactly the silent
    # retarget this argument exists to prevent.
    ap.add_argument("--panel", choices=sorted(PANELS), default="2k",
                    help="gene axis + the arms decodable on it. '2k' is the "
                         "published contract (8 arms); 'union5750' is GENESIS "
                         "plus the plate replicate and nothing else.")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():  # a bad GPU index: stop, never run on CPU
        raise SystemExit(f"--device {args.device}: CUDA is not available (pass --device cpu to run on the CPU)")
    dev = torch.device(args.device)
    drug = args.drug
    readout = tuple(g.strip() for g in args.genes.split(",") if g.strip())
    dose_plates = DRUG_PLATES.get(drug) or gated_dose_plates(drug)

    panel = resolve_panel(args.panel)
    panel_genes, panel_arms = panel.genes, panel.arms
    print(f"panel {panel.name}: {len(panel_genes)} genes  "
          f"cache={panel.cache.name}  arms={sorted(panel_arms)}"
          f"  baselines={list(panel.baselines) or 'none'}")
    gidx = {g: i for i, g in enumerate(panel_genes)}
    missing = [g for g in readout if g not in gidx]
    if missing:
        raise ValueError(f"pre-registered genes absent from the panel: {missing}")

    # *_rgs projections derive a dataset-level L_ref from process-global state; without
    # this every projection raises. Tahoe is the only dataset with plate replicates.
    print(f"L_ref={set_dataset_median_umi('tahoe')} (tahoe)")

    def raw_int(x: np.ndarray, *, is_pred: bool) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32, device=dev)
        proj = (_project_predictions_to_spaces(t, MODEL_SPACE) if is_pred
                else _project_raw_to_spaces(t))
        return proj["raw_int_rgs"]

    baselines = load_baselines(panel.baselines, panel_genes, panel.baseline_root)
    gene_rows: list[dict] = []
    cond_rows: list[dict] = []

    for dose, (pa, pb) in dose_plates.items():
        for cell in CELL_LINES:
            ctrl_raw = {p: load_control(panel.cache, cell_name=cell,
                                        plate=f"plate{p}", n_genes=len(panel_genes))
                        for p in (pa, pb)}
            ctrl = {p: raw_int(v, is_pred=False) for p, v in ctrl_raw.items()}
            truth_raw = {
                p: load_perturbed(panel.cache, cell_name=cell, drug=drug,
                                  dose=dose, plate=f"plate{p}",
                                  n_genes=len(panel_genes))
                for p in (pa, pb)
            }
            # Wilcoxon depends only on (control plate, arm cells), never on the model
            # or on the detection floor, so every arm's stats are computed exactly once.
            truth_st = {p: deg_stats(ctrl[p], raw_int(v, is_pred=False))
                        for p, v in truth_raw.items()}
            pred_st = {
                (m, p): deg_stats(ctrl[p], raw_int(
                    load_pred(pred_path(root, fmt, drug, cell, dose, p), panel_genes),
                    is_pred=True))
                for m, (root, fmt) in panel_arms.items() for p in (pa, pb)
            }
            # Baselines project from `raw`, not `log1p`, and are sized to the truth
            # cell count of the plate they predict — the same n the eval work queue
            # gives every arm.
            for kind, gen in baselines.items():
                for p in (pa, pb):
                    rows = baseline_pred(
                        gen, kind, drug=drug, cell=cell, dose=dose, plate=p,
                        ctrl_raw=ctrl_raw[p], n_cells=int(truth_raw[p].shape[0]))
                    t = torch.as_tensor(rows, dtype=torch.float32, device=dev)
                    pred_st[(kind, p)] = deg_stats(
                        ctrl[p],
                        _project_predictions_to_spaces(t, BASELINE_SPACE)["raw_int_rgs"])
            # ---- GENESIS A+B, scored against BOTH plates' controls like the
            # single-plate arms, so it slots into the same tables.
            if args.with_ab:
                try:
                    ga = raw_int(load_pred(pred_path(*panel_arms["genesis"], drug,
                                                     cell, dose, pa), panel_genes),
                                 is_pred=True)
                    gb = raw_int(load_pred(pred_path(*panel_arms["genesis"], drug,
                                                     cell, dose, pb), panel_genes),
                                 is_pred=True)
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [int(args.ab_seed), _ab_salt(cell, drug, dose, pa, pb)]))
                    ia, ib = draw_half_half_indices(int(ga.shape[0]),
                                                    int(gb.shape[0]), rng)
                    mix = torch.cat([ga[torch.as_tensor(ia, device=ga.device)],
                                     gb[torch.as_tensor(ib, device=gb.device)]], 0)
                    for p_ in (pa, pb):
                        pred_st[(GENESIS_AB, p_)] = deg_stats(ctrl[p_], mix)
                except Exception as e:
                    print(f"  no A+B for {drug}/{cell}/{dose}: {type(e).__name__}: {e}")

            det = {(x, y): pooled_det_frac(ctrl[x], ctrl[y])
                   for x in (pa, pb) for y in (pa, pb)}

            for tp, op in ((pa, pb), (pb, pa)):  # both plate directions
                base = {"cell_name": cell, "dose": dose, "drug": drug,
                        "truth_plate": tp, "other_plate": op}
                t_st = truth_st[tp]
                for arm, pp in (("cross", op), ("same", tp)):
                    # The floor pools the two controls this pairing actually uses:
                    # truth's plate and the prediction's plate (one pool for `same`).
                    keep = det[(tp, pp)] >= CTRL_DET_FLOOR
                    t_de = call_de(t_st, keep)
                    t_rank = down_rank(t_st["lfc"], keep)
                    # `replicate` only in `cross`: on `same` it is truth vs itself.
                    models = ([*panel_arms, *panel.baselines]
                              + ([REPLICATE] if arm == "cross" else []))
                    if args.with_ab and (GENESIS_AB, pp) in pred_st:
                        models = [*models, GENESIS_AB]
                    for model in models:
                        p_st = truth_st[pp] if model == REPLICATE else pred_st[(model, pp)]
                        p_de = call_de(p_st, keep)
                        p_rank = down_rank(p_st["lfc"], keep)
                        cond_rows.append({
                            **base, "arm": arm, "model": model, "pred_plate": pp,
                            "n_truth": t_st["n_target"], "n_pred": p_st["n_target"],
                            "n_ctrl_truth": int(ctrl[tp].shape[0]),
                            "n_genes_kept": int(keep.sum()),
                            **overlap_summary(t_st, p_st, keep),
                        })
                        for g in readout:
                            i = gidx[g]
                            gene_rows.append({
                                **base, "arm": arm, "model": model,
                                "pred_plate": pp, "gene": g,
                                "ctrl_det_frac": float(det[(tp, pp)][i]),
                                "passes_floor": bool(keep[i]),
                                "n_genes_kept": int(keep.sum()),
                                "truth_de": bool(t_de[i]),
                                "truth_lfc": float(t_st["lfc"][i]),
                                "truth_q": float(t_st["q"][i]),
                                "truth_z": float(t_st["z"][i]),
                                "truth_down_rank": int(t_rank[i]),
                                "truth_mean": float(t_st["target_mean"][i]),
                                "truth_ctrl_mean": float(t_st["ctrl_mean"][i]),
                                "pred_de": bool(p_de[i]),
                                "pred_lfc": float(p_st["lfc"][i]),
                                "pred_q": float(p_st["q"][i]),
                                "pred_z": float(p_st["z"][i]),
                                "pred_down_rank": int(p_rank[i]),
                                "pred_mean": float(p_st["target_mean"][i]),
                                "pred_ctrl_mean": float(p_st["ctrl_mean"][i]),
                                # Two sign conventions, because they genuinely differ
                                # on ~6% of genes: log2FC sign is what "downregulated"
                                # means biologically and is what this deliverable
                                # reports, while the pipeline's panel-level
                                # `direction_match` uses the Wilcoxon z. Both are
                                # emitted so no claim rests on an unstated choice.
                                "recovered": bool(
                                    t_de[i] and p_de[i]
                                    and np.sign(t_st["lfc"][i]) == np.sign(p_st["lfc"][i])
                                ),
                                "recovered_zsign": bool(
                                    t_de[i] and p_de[i]
                                    and np.sign(t_st["z"][i]) == np.sign(p_st["z"][i])
                                ),
                            })
            print(f"  done {drug} dose={dose} {cell}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("per_gene", gene_rows), ("per_condition", cond_rows)):
        df = pd.DataFrame(rows)
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
        print(f"wrote {args.out_dir / f'{name}.csv'} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
