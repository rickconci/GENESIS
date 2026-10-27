"""The drug axis of the 3,072-panel Tahoe panels: which compounds are columns, on which 5 uM plates,
in which class, in which drawn order. The rules are here; scripts/figures/figure4/extract/extract.py applies them to
the scores and plates and writes data/drug_axis/drug_axis.tsv, which table() reads back.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The axis is one rule on the Tahoe-100M release's own annotation (drug_metadata.parquet,
# moa-fine): every compound withheld from the held-out lines at 5 uM whose class is one of MOA_GROUPS' 13. No
# compound is added or removed by hand, and nothing measured or modelled enters the choice. Unannotated
# compounds ("unclear", half the test set) and classes with no direct route to a PROGENy pathway (CDK, microtubule,
# COX, ...) stay off; so does every inert reference (Tahoe has no such class). The 67 drugs whose two 5 uM plates
# replicate are still scored (the cross-batch and context panels need two plates) but drawn only where the rule
# holds them.
EFFECT = 0.3             # the floor is measured where the plate-mean response is past this

# ----------------------------------------------------------------------------
# The groups: the drawn brackets, each a union of Tahoe classes
# ----------------------------------------------------------------------------
# A group is a list of steps, drawn in order; a step is a list of Tahoe moa-fine classes whose compounds are drawn
# together, alphabetical (casefold). RAS-MAPK and PI3K-AKT-mTOR step down the cascade; Other targeted keeps each
# class together. The bracket is "long|short": the short one is drawn where the long one does not fit (the
# cross-batch panels hold only the two-plate drugs).
MOA_GROUPS = [
    ("EGFR/ERBB|EGFR", [["EGFR/ERBB inhibitor"]]),
    ("RAS–MAPK|MAPK", [["RAS inhibitor"], ["RAF inhibitor"], ["MEK inhibitor"]]),
    ("PI3K–AKT–mTOR|PI3K", [["PI3K/AKT inhibitor"], ["MTOR inhibitor"]]),
    ("Translation|Transl.", [["Protein synthesis inhibitor"]]),
    ("DNA damage|DNA", [["DNA synthesis/repair inhibitor"]]),
    ("Other targeted|Other", [["GSK3 inhibitor"], ["HDAC inhibitor"], ["JAK/STAT inhibitor"], ["Proteasome inhibitor"]]),
    ("GC", [["Glucocorticoid receptor agonist"]]),
]
AXIS_MOA = [c for _g, steps in MOA_GROUPS for step in steps for c in step]   # the 13 classes, each once


def group_of(moa: str) -> str:
    """The drawn group of a Tahoe moa-fine class; raises for a class the axis does not draw."""
    for g, steps in MOA_GROUPS:
        if any(moa in step for step in steps):
            return g
    raise RuntimeError(f"{moa!r} is not an axis class")


def columns(drugs: set[str], moa: dict[str, str]) -> tuple[list[str], list[tuple[str, int, int]]]:
    """The drawn columns restricted to ``drugs`` and the bracket span of each group: MOA_GROUPS' order, step by
    step, alphabetical within a step. ``moa``: drug -> Tahoe moa-fine (the axis table's moa column). Raises if a
    drug has no class or one the axis does not draw: it would silently vanish from every panel."""
    if (bad := sorted(d for d in drugs if moa.get(d) not in AXIS_MOA)):
        raise RuntimeError(f"no axis class for {[(d, moa.get(d)) for d in bad]}")
    order, groups = [], []
    for name, steps in MOA_GROUPS:
        members = [d for step in steps for d in sorted((d for d in drugs if moa[d] in step), key=str.casefold)]
        if members:
            groups.append((name, len(order), len(order) + len(members) - 1))
            order += members
    return order, groups


# ----------------------------------------------------------------------------
# The floor: how far one plate's measured response moves on the replicate plate
# ----------------------------------------------------------------------------

def plate_floor(t: pd.DataFrame, pathways: list[str]) -> pd.Series:
    """Per pathway: mean |A - B| over the (line, drug, dose) measured on two plates at one dose, where the
    plate-mean response is past EFFECT, no gated gene reads a zero mean on either plate (floor_hits) and
    both keep more than half their plate's median cell count. Truth only, test conditions, every dose."""
    t = t[(t.arm == "truth") & t.is_test & t.readout.isin(pathways)]
    k = ["cell_name", "drug", "dose", "readout"]
    t = t[t.groupby(k).plate.transform("nunique") == 2]
    w = t.groupby(k).agg(A=("e", "first"), B=("e", "last"), fh=("floor_hits", "max"), rec=("recovery", "min"))
    w = w[(w.fh == 0) & (w.rec > 0.5) & (((w.A + w.B) / 2).abs() > EFFECT)]
    if (gone := sorted(set(pathways) - set(w.index.get_level_values("readout")))):
        raise SystemExit(f"no plate repeat past {EFFECT} for {gone}: their floor is undefined")
    return (w.A - w.B).abs().groupby(level="readout").mean().reindex(pathways)


# ----------------------------------------------------------------------------
# Read the table: data/drug_axis/drug_axis.tsv, as the axis extract wrote it
# ----------------------------------------------------------------------------

def table(path: Path) -> pd.DataFrame:
    """``path``: a bundle's data/drug_axis/drug_axis.tsv. The plates column stays text ('3;12')."""
    if not path.exists():
        raise SystemExit(f"{path} missing: python -m scripts.figures.figure4.extract.extract --write builds the drug axis")
    return pd.read_csv(path, sep="\t", dtype={"plates": str})


def conditions(t: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """{drug: its 5 uM plates} for every drug of the axis table ``t``: the extraction scores all 106, the
    panels choose (on_axis for the main panels, replicated for the cross-batch ones). Alphabetical, so the
    data files' row order never follows how the drugs are grouped for drawing."""
    return {d: tuple(p.split(";")) for d, p in sorted(zip(t.drug, t.plates), key=lambda x: x[0].casefold())}
