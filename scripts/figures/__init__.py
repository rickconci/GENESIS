"""One folder per figure: its draw scripts, and the modules only that figure imports."""

from pathlib import Path

CORE = Path(__file__).resolve().parents[2] / "core_deliverables"   # data/figureN in, figureN/ out
