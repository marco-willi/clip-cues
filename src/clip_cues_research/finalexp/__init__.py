"""F1-F7: the final methodological consolidation.

Implements `.claude/plans/PLAN_FINAL_CONSOLIDATION.md`: collapse the model inventory to two CLIP
linear directions (canonical 1024-d ``D_h``, projected 768-d ``D_e``) by retraining everything
under **one matched recipe** on **one frozen, checksummed input snapshot**.

Modules:

- :mod:`~clip_cues_research.finalexp.data` — the ``reproduction/experiments/data/`` snapshot: manifest-mediated
  access with sha256 verification on load. All F-experiment inputs come from here.
- :mod:`~clip_cues_research.finalexp.features` — derived 768-d projected features and cue scores.
- :mod:`~clip_cues_research.finalexp.trainer` — the matched training recipe (one definition).
- :mod:`~clip_cues_research.finalexp.stability` — direction/score/profile stability metrics.
- :mod:`~clip_cues_research.finalexp.profiles` — cue-association profiles (the E12 estimators).
- :mod:`~clip_cues_research.finalexp.runner` — run folders + ``run_meta.json`` provenance.
"""

from __future__ import annotations

__all__ = ["data", "features", "profiles", "runner", "stability", "trainer"]
