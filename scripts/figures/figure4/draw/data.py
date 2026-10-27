"""Figure 4's data: the drug axis (which compounds are columns, in which class and order) and one
dataset dict for the depth-adjusted PROGENy split, which every Figure 4 heatmap draws.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# The axis is one rule on the Tahoe-100M release's own annotation (drug_metadata.parquet, moa-fine): every
# compound withheld from the held-out lines at 5 uM whose class is one of MOA_GROUPS' 13. No compound is added or
# removed by hand, and nothing measured or modelled enters the choice.
#
# A group is a list of steps, drawn in order; a step is a list of Tahoe moa-fine classes whose compounds are drawn
# together, alphabetical (casefold). RAS-MAPK and PI3K-AKT-mTOR step down the cascade. The bracket is "long|short":
# the short one is drawn where the long one does not fit.
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


def axis_table(path: Path) -> pd.DataFrame:
    """drug_axis.tsv: one row per scored drug, ``on_axis`` marking the drawn ones. The plates column stays
    text ('3;12')."""
    return pd.read_csv(path, sep="\t", dtype={"plates": str})


def multi_plate(axis: pd.DataFrame) -> set[str]:
    """Drugs measured on more than one 5 uM plate. Their entries pool every plate (the mixture)."""
    return set(axis.drug[axis.plates.str.contains(";")])


def columns(drugs: set[str], moa: dict[str, str]) -> tuple[list[str], list[tuple[str, int, int]]]:
    """The drawn columns and the bracket span of each group: MOA_GROUPS' order, step by step, alphabetical within
    a step. ``moa``: drug -> Tahoe moa-fine. Raises if a drug has no class or one the axis does not draw: it would
    silently vanish from every panel."""
    if (bad := sorted(d for d in drugs if moa.get(d) not in AXIS_MOA)):
        raise RuntimeError(f"no axis class for {[(d, moa.get(d)) for d in bad]}")
    order, groups = [], []
    for name, steps in MOA_GROUPS:
        members = [d for step in steps for d in sorted((d for d in drugs if moa[d] in step), key=str.casefold)]
        if members:
            groups.append((name, len(order), len(order) + len(members) - 1))
            order += members
    return order, groups


# A dataset S is what every heatmap takes: the module values (mv, one row per pathway x line x drug x arm), the
# pathways, the held-out lines and their display spelling, the arms, the coverage behind the '*', and the columns
# in their drug classes.
def dataset(d: Path, axis: pd.DataFrame, arms: list[str]) -> dict:
    """``d``: a tahoe_progeny_depth/ folder. Rows: the 5 held-out lines; columns: the axis drugs in their classes;
    ``arms`` drawn against the measured A+B mixture."""
    drugs = set(axis.drug[axis.on_axis])
    disp = pd.read_csv(d / "heldout_lines.tsv", sep="\t").set_index("cell_name").display.to_dict()
    cov = pd.read_csv(d / "coverage.tsv", sep="\t")
    mv = pd.read_parquet(d / "module_values.parquet")
    mv = mv[mv.drug.isin(drugs)]
    if (absent := sorted(drugs - set(mv.drug))):
        raise RuntimeError(f"{d.name}: no values for {absent}")
    order, groups = columns(drugs, dict(zip(axis.drug, axis.moa)))
    return dict(mv=mv, progs=pd.read_csv(d / "programmes.tsv", sep="\t"), held=sorted(disp), disp=disp,
                cond_order=order, cond_groups=groups, arms=arms,
                cov=cov.set_index(["programme", "cell_name"]).mean_members)


def drawn_values(S: dict) -> pd.Series:
    """v as the heatmaps draw it, by (arm, line in display spelling, drug, pathway)."""
    m = S["mv"][S["mv"].set == "grid"]
    v = m.assign(cell_name=m.cell_name.map(S["disp"])).set_index(["arm", "cell_name", "drug", "programme"]).v
    if not v.index.is_unique or v.index.get_level_values("cell_name").isna().any() or v.isna().any():
        raise RuntimeError("a drawn value repeats, is missing, or its line has no display name")
    return v
