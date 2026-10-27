"""Split-config dataclasses.

These mirror the structure of the YAMLs under ``core_util_data/splits/``.
Validation is done with explicit ``__post_init__`` rapid-fail checks so the
package does not depend on pydantic at runtime.

Example YAML (Tahoe)::

    dataset: tahoe100M
    controls_to_train: true

    validation:
      cell_name: false      # force-empty
      Organ:   [CNS/Brain]
      drug:    [Elimusertib hydrochloride, venetoclax, Binimetinib]
      dose:    [0.05, 0.5, 5.0]

    test:
      Organ: [Lung, Bowel, Pancreas]
      drug:  [...]
      dose:  [0.05, 0.5, 5.0]

    train:
      conditions:            # or flat train.sample / fraction / hbss (legacy)
        sample: hbss         # all | hbss
        fraction: 0.05        # required iff sample == hbss
        hbss:
          balance_axes: [cell_name, drug, dose]
          candidate_axis: cell_name
          diversity_axes:
            cell_name: Organ
            drug: moa-fine
          lam: 0.5
          seed: 42
      cells:                 # optional. Either target_n, or per_condition + group_by.
        per_condition: 25    # this many cells from each group; shorter groups kept whole
        group_by: [cell_name, drug, dose]
        seed: 0
        keep_controls: true  # controls are not part of the cap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

PredicateBlock = Union[Dict[str, Any], List[Dict[str, Any]]]


_VALID_SAMPLE_MODES = frozenset({"all", "hbss"})


@dataclass
class HBSSSpec:
    """Configuration for one Hierarchical Balanced Stratified Sampling run.

    Attributes:
        balance_axes: Obs columns that the IPF marginals will balance
            (e.g. ``["cell_name", "drug", "dose"]`` for Tahoe,
            ``["donor", "cytokine"]`` for Parse). At least one entry.
        candidate_axis: The single axis along which Phase-3 greedy
            candidates are enumerated within each slice of the remaining
            balance axes. Must be one of ``balance_axes`` — rapid-fail
            otherwise.
        diversity_axes: Mapping ``balance_col -> parent_col`` used to
            penalise within-slice group repetition during Phase 3
            (e.g. ``{"cell_name": "Organ", "drug": "moa-fine"}``). Only
            the entry whose key equals ``candidate_axis`` drives the
            penalty; other entries are reported in the balance diagnostic
            but do not affect selection. Empty dict disables diversity.
        lam: Diversity penalty coefficient. ``0`` = pure IPF-weighted
            multinomial. ``~0.5`` is the recommended default.
        seed: RNG seed for the multinomial draws in Phase 3.
        max_iter: Maximum IPF sweeps.
        tol: IPF convergence tolerance on ``max|ΔQ|``.
        ambiguous_diversity_to_singleton: If True, parent labels in
            ``{NaN, "unclear", "none", ""}`` are mapped to per-row
            singletons (so the penalty does not aggressively avoid every
            un-annotated row).
        tie_break: How exactly tied quota remainders are ordered. ``"stable"``:
            lowest index first, the same on every host. ``"legacy"`` (default, every
            shipped split): numpy's default sort, whose tie order depends
            on the CPU (AVX-512 Intel vs AVX2 AMD draw ~half different conditions),
            so a legacy draw reproduces only on the CPU family that made it.
    """

    balance_axes: List[str]
    candidate_axis: str
    diversity_axes: Dict[str, str] = field(default_factory=dict)
    lam: float = 0.5
    seed: int = 42
    max_iter: int = 300
    tol: float = 1e-4
    ambiguous_diversity_to_singleton: bool = True
    tie_break: str = "legacy"

    def __post_init__(self) -> None:
        if not isinstance(self.balance_axes, list) or not self.balance_axes:
            raise ValueError(
                f"HBSS: balance_axes must be a non-empty list; got "
                f"{self.balance_axes!r}."
            )
        if len(set(self.balance_axes)) != len(self.balance_axes):
            raise ValueError(
                f"HBSS: balance_axes entries must be unique; got "
                f"{self.balance_axes!r}."
            )
        if self.candidate_axis not in self.balance_axes:
            raise ValueError(
                f"HBSS: candidate_axis={self.candidate_axis!r} must be one of "
                f"balance_axes={self.balance_axes!r}."
            )
        unknown_div = [c for c in self.diversity_axes if c not in self.balance_axes]
        if unknown_div:
            raise ValueError(
                f"HBSS: diversity_axes keys must be a subset of balance_axes; "
                f"unknown keys {unknown_div!r} not in {self.balance_axes!r}."
            )
        if float(self.lam) < 0.0:
            raise ValueError(f"HBSS: lam must be >= 0; got {self.lam!r}.")
        if int(self.max_iter) <= 0:
            raise ValueError(f"HBSS: max_iter must be > 0; got {self.max_iter!r}.")
        if float(self.tol) <= 0.0:
            raise ValueError(f"HBSS: tol must be > 0; got {self.tol!r}.")
        if self.tie_break not in ("stable", "legacy"):
            raise ValueError(f"HBSS: tie_break must be 'stable' or 'legacy'; got {self.tie_break!r}.")


@dataclass
class TrainCellsSpec:
    """Post-condition cell budget on the final train index set.

    Attributes:
        target_n: Cap train rows via one uniform draw. ``None`` means no total cap.
        seed: RNG seed for the subsample draw.
        keep_controls: Leave every control row out of the cap.
        per_condition: Draw this many cells from each ``group_by`` condition.
            A condition with fewer cells is kept whole.
        group_by: Columns that name a condition for ``per_condition``.
    """

    target_n: Optional[int] = None
    seed: int = 0
    keep_controls: bool = False
    per_condition: Optional[int] = None
    group_by: tuple = ()

    def __post_init__(self) -> None:
        if self.target_n is not None and int(self.target_n) <= 0:
            raise ValueError(
                f"train.cells.target_n must be positive or null; got {self.target_n!r}."
            )
        if self.target_n is not None:
            self.target_n = int(self.target_n)
        if self.per_condition is not None and int(self.per_condition) <= 0:
            raise ValueError(
                f"train.cells.per_condition must be positive or null; got {self.per_condition!r}."
            )
        if self.per_condition is not None:
            self.per_condition = int(self.per_condition)
        if self.target_n is not None and self.per_condition is not None:
            raise ValueError("train.cells: set target_n or per_condition, not both.")
        self.seed = int(self.seed)
        self.group_by = tuple(str(c) for c in self.group_by)
        if not isinstance(self.keep_controls, bool):
            raise ValueError(f"train.cells.keep_controls must be true or false; got {self.keep_controls!r}.")
        if self.keep_controls and self.target_n is None and self.per_condition is None:
            raise ValueError("train.cells.keep_controls needs target_n or per_condition.")
        if self.per_condition is not None and not self.group_by:
            raise ValueError("train.cells.per_condition needs group_by.")
        if self.group_by and self.per_condition is None:
            raise ValueError("train.cells.group_by is only used with per_condition.")


@dataclass
class TrainSpec:
    """Train-sampling strategy (conditions, then optional cell budget).

    Attributes:
        sample: ``"all"`` keeps every train row; ``"hbss"`` additionally
            sub-samples the perturbed train rows via HBSS.
        fraction: Required iff ``sample == "hbss"``; must be in ``(0, 1)``.
        hbss: Required iff ``sample == "hbss"``; see :class:`HBSSSpec`.
        cells: Optional uniform cap on total train rows after conditions/HBSS.
    """

    sample: str = "all"
    fraction: Optional[float] = None
    hbss: Optional[HBSSSpec] = None
    cells: TrainCellsSpec = field(default_factory=TrainCellsSpec)

    def __post_init__(self) -> None:
        if self.sample not in _VALID_SAMPLE_MODES:
            raise ValueError(
                f"train.sample must be one of {sorted(_VALID_SAMPLE_MODES)!r}; "
                f"got {self.sample!r}."
            )
        if self.sample == "hbss":
            if self.fraction is None or not (0.0 < float(self.fraction) < 1.0):
                raise ValueError(
                    f"train.sample='hbss' requires train.fraction in (0, 1); "
                    f"got {self.fraction!r}."
                )
            if self.hbss is None:
                raise ValueError(
                    "train.sample='hbss' requires a train.hbss configuration block."
                )
        elif self.fraction is not None or self.hbss is not None:
            # Be explicit: ignored fields under sample='all' are a footgun.
            raise ValueError(
                "train.fraction / train.hbss must only be set when "
                f"train.sample='hbss'; got sample={self.sample!r}, "
                f"fraction={self.fraction!r}, hbss={'set' if self.hbss else 'unset'}."
            )


@dataclass
class SplitConfig:
    """Parsed split-config YAML.

    Attributes:
        dataset: Dataset identifier (informational; e.g. ``"tahoe100M"``,
            ``"pbmc"``). Not used by the engine — only for run naming /
            debugging.
        controls_to_train: When True (default), rows matching
            ``PertGenConfig.control_condition`` are forced into train
            regardless of any val/test predicate match.
        validation: Predicate selecting val rows. Dict = single AND-block;
            list of dicts = OR-of-ANDs. Empty dict = no rows.
        test: Same shape as ``validation``.
        train_exclusions: Optional OR-of-AND blocks forced **out** of train even
            when rows are not in val/test (graded-grid train-peek withhold).
        train: Train-sampling strategy (conditions + optional cell budget).
    """

    dataset: str
    controls_to_train: bool = True
    validation: PredicateBlock = field(default_factory=dict)
    test: PredicateBlock = field(default_factory=dict)
    train_exclusions: PredicateBlock = field(default_factory=dict)
    train: TrainSpec = field(default_factory=TrainSpec)

    @staticmethod
    def _parse_train_cells(raw: Optional[Mapping[str, Any]]) -> TrainCellsSpec:
        if not raw:
            return TrainCellsSpec()
        unknown = set(raw) - {"target_n", "seed", "keep_controls", "per_condition", "group_by"}
        if unknown:
            raise ValueError(
                f"Unknown keys under 'train.cells': {sorted(unknown)!r}. "
                "Allowed: 'target_n', 'per_condition', 'group_by', 'seed', 'keep_controls'."
            )
        return TrainCellsSpec(
            target_n=raw.get("target_n"),
            seed=int(raw.get("seed", 0)),
            keep_controls=raw.get("keep_controls", False),
            per_condition=raw.get("per_condition"),
            group_by=tuple(raw.get("group_by") or ()),
        )

    @staticmethod
    def _parse_train_conditions(
        cond_raw: Mapping[str, Any],
    ) -> TrainSpec:
        cond = dict(cond_raw)
        hbss_raw = cond.pop("hbss", None)
        hbss = HBSSSpec(**dict(hbss_raw)) if hbss_raw is not None else None
        unknown = set(cond) - {"sample", "fraction"}
        if unknown:
            raise ValueError(
                f"Unknown keys under train conditions: {sorted(unknown)!r}. "
                "Allowed: 'sample', 'fraction', 'hbss'."
            )
        return TrainSpec(
            sample=str(cond.get("sample", "all")),
            fraction=cond.get("fraction"),
            hbss=hbss,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitConfig":
        """Build a :class:`SplitConfig` from a YAML-loaded mapping.

        Raises:
            ValueError: On unknown top-level keys or invalid nested
                fields (see :class:`TrainSpec` / :class:`HBSSSpec`).
        """
        known = {"dataset", "controls_to_train", "validation", "test", "train", "train_exclusions"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"Unknown top-level keys in split config: {sorted(unknown)!r}. "
                f"Allowed keys: {sorted(known)!r}."
            )
        if "dataset" not in data:
            raise ValueError("Split config is missing the required 'dataset' field.")

        train_raw = dict(data.get("train") or {})
        cells = cls._parse_train_cells(train_raw.pop("cells", None))

        flat_keys = {"sample", "fraction", "hbss"}
        if "conditions" in train_raw:
            if flat_keys & set(train_raw):
                raise ValueError(
                    "train: use either nested train.conditions or flat "
                    "train.sample/fraction/hbss, not both."
                )
            cond_raw = dict(train_raw.pop("conditions"))
            unknown_train = set(train_raw)
            train = cls._parse_train_conditions(cond_raw)
        else:
            cond_raw = {k: train_raw.pop(k) for k in flat_keys if k in train_raw}
            unknown_train = set(train_raw)
            train = cls._parse_train_conditions(cond_raw)

        if unknown_train:
            raise ValueError(
                f"Unknown keys under 'train': {sorted(unknown_train)!r}. "
                "Allowed: 'conditions', 'cells', or legacy 'sample', 'fraction', 'hbss'."
            )
        train.cells = cells

        return cls(
            dataset=str(data["dataset"]),
            controls_to_train=bool(data.get("controls_to_train", True)),
            validation=data.get("validation") or {},
            test=data.get("test") or {},
            train_exclusions=data.get("train_exclusions") or {},
            train=train,
        )
