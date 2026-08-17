#!/usr/bin/env python
"""Figure for the exposure-cue analysis (PLAN_EXPLAINABILITY_FOLLOWUPS Step 2).

Grouped horizontal bar chart of antonym cue loadings for the synthclic / cnnspot / combined
linear-probe directions (union of each probe's top-8 |loading| cues), from
``outputs/explain/exposure_cues/antonym_cue_profiles.csv`` (run ``analyze_exposure_cues.py`` first).

Append-only: writes outputs/explain/exposure_cues/cue-profiles.png.

Run (local, CPU):
    uv run python scripts/analyze/plot_exposure_cue_profiles.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

OUT = Path("outputs/explain/exposure_cues")
PROBES = ["synthclic", "cnnspot", "combined"]
TOP_K = 8


def main() -> None:
    prof = pd.read_csv(OUT / "antonym_cue_profiles.csv")
    cues: list[str] = []
    for p in PROBES:
        top = prof.reindex(prof[p].abs().sort_values(ascending=False).index)["cue"].head(TOP_K)
        cues += [c for c in top if c not in cues]
    sub = prof[prof["cue"].isin(cues)].copy()
    # order rows by the cue's strongest |loading| across the three probes
    sub["max_abs"] = sub[PROBES].abs().max(axis=1)
    sub = sub.sort_values("max_abs", ascending=False)
    long = sub.melt(id_vars="cue", value_vars=PROBES, var_name="probe", value_name="loading")

    fig, ax = plt.subplots(figsize=(7, 0.38 * len(cues) + 1.2))
    _ = sns.barplot(data=long, y="cue", x="loading", hue="probe", order=sub["cue"], ax=ax)
    _ = ax.axvline(0.0, color="0.3", lw=0.8)
    _ = ax.set_title(
        f"Linear-probe cue loadings (union of top-{TOP_K} cues per probe)\n"
        "antonym cue basis, least-squares preimage through CLIP's visual projection"
    )
    _ = ax.set_xlabel("loading of unit cue-space direction (+ = cue activation raises P(fake))")
    _ = ax.set_ylabel("")
    _ = ax.legend(title="training exposure", loc="lower right")
    _ = plt.tight_layout()
    fig.savefig(OUT / "cue-profiles.png", dpi=150, bbox_inches="tight")
    print("wrote", OUT / "cue-profiles.png")


if __name__ == "__main__":
    main()
