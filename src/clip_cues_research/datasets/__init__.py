"""Revision-only dataset adapters (E7 external benchmarks). Never back-ported to clip_cues."""

from clip_cues_research.datasets.community_forensics import (
    CF_EVAL,
    CF_SMALL,
    is_community_forensics,
    load_community_forensics,
)

__all__ = ["CF_EVAL", "CF_SMALL", "is_community_forensics", "load_community_forensics"]
