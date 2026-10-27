"""The three metrics, named once for every figure.

A condition is scored by its top-100 DEG overlap (raw counts), and by Pearson Δ and energy distance in
depth-scaled log1p (slice_med), the space Figures 1-3 report. These are the score tables' column keys.
"""

DEG = "deg_raw_int_rgs/deg_overlap_top_p_ranked_k100_fdr0p001_lfc0p1_adaptivek_truth_detfloor10"
SPACE = "log1p_slice_med_rgs"
PEARSON, ENERGY = f"{SPACE}/pearson_r_diff", f"{SPACE}/energy_distance"
ARM_NAME = {"copy_closest": "Nearest-neighbor"}      # the per-condition tables spell this one arm differently
