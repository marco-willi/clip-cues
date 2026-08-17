#!/usr/bin/env python
"""TableB: regenerate the appendix per-generator table (`tab:clip:test_results_detail`, Table 13).

The manuscript's Table 13 (38 generator rows x 4 training sets x {ACC, AP}) does not reproduce the
authoritative cross-dataset matrix behind main Table 3: **6 of the 9 off-diagonal blocks disagree**,
by up to +0.166. It entered the paper repo hand-written and is derived from no tracked artifact.
This script regenerates all 12 cells from **one** detector definition so the appendix and the main
text describe the same model.

The detector is the matched, no-augmentation canonical 1024-d probe ``D_h`` (the F1 recipe: Adam
lr 1e-3 / wd 0.01, label smoothing 0.1, batch 64, <=200 epochs, early stop on val CE patience 5,
frozen cached pooler, seed 123). Eleven of the twelve cells already exist as per-image predictions
from the E3 cross-dataset runs, which used exactly that recipe — F1's own regression anchor *is* one
of them (F1 0.9230 vs E3 0.9239). Re-aggregating those predictions rather than refitting makes the
acceptance check against ``e3_cross_matrix_mAP.csv`` exact instead of approximate, and it means the
SynthBuster+ block requires **no new read of the closed SB+ test split** (see README section 3).

The twelfth cell (Combined -> CNNSpot) has no stored prediction file, so it is computed here by
scoring the matched combined head on the full CNNSpot test frame. The head is checked against the
combined predictions that *do* exist before it is used.

    make finalexp-data WITH_CFEVAL=1 WITH_APPENDIX=1
    uv run python scripts/finalexp/run_appendix_per_generator.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from clip_cues_research.analysis.metrics import (
    pairing_for_dataset,
    per_generator_accuracy,
    per_generator_ap,
)
from clip_cues_research.finalexp import data as D
from clip_cues_research.finalexp.runner import Run
from clip_cues_research.finalexp.trainer import RECIPE, train_head

EXPERIMENT = "TableB-per-generator"

TRAIN_SETS = ["cnnspot", "synthbuster-plus", "synthclic", "combined"]
TEST_CORPORA = ["cnnspot", "synthbuster-plus", "synthclic"]

# Which frame each test corpus is evaluated on. CNNSpot is the **full benchmark test set**
# (108,310 images / 21 generators), not the 8,000-image cnnspot-small frame F5/F6 train on: the
# manuscript's CNNSpot block has 21 rows, and 21 generators is what the authoritative matrix's
# CNNSpot column reports. `seeingdark` carries 180 synthetics here and none in cnnspot-small,
# which is the whole of the section-4.3 question.
EVAL_FRAME = {
    "cnnspot": "pooler/cnnspot_full",
    "synthbuster-plus": "pooler/synthbuster-plus",
    "synthclic": "pooler/synthclic",
}

# The cell with no stored predictions: the combined runs never evaluated on the full CNNSpot frame.
COMPUTED_CELL = ("combined", "cnnspot")
COMBINED_HEAD = "ckpt/linear_probe_combined_matched"

# The authoritative main-text matrix, rows=train x cols=eval (reproduction/revision_export/tables/
# e3_cross_matrix_mAP.csv). Hard-coded rather than read from `reproduction/revision_export/`, which is a build
# product and gitignored; the acceptance check must not silently pass because the target moved.
AUTHORITATIVE_MAP = {
    ("cnnspot", "cnnspot"): 0.9640,
    ("cnnspot", "synthbuster-plus"): 0.4962,
    ("cnnspot", "synthclic"): 0.5214,
    ("synthbuster-plus", "cnnspot"): 0.6692,
    ("synthbuster-plus", "synthbuster-plus"): 0.9952,
    ("synthbuster-plus", "synthclic"): 0.6112,
    ("synthclic", "cnnspot"): 0.4222,
    ("synthclic", "synthbuster-plus"): 0.7940,
    ("synthclic", "synthclic"): 0.9239,
    ("combined", "cnnspot"): 0.8982,
    ("combined", "synthbuster-plus"): 0.9761,
    ("combined", "synthclic"): 0.8670,
}

# `combined -> cnnspot` is the one authoritative cell measured on a *different* population: the
# combined E3 runs post-date the switch to the full CNNSpot frame's sibling runs and only ever
# scored cnnspot-small (4,000 imgs, 20 generators). Its 0.8982 is therefore not comparable with
# the other three CNNSpot cells, and the acceptance check reports it as a known exception rather
# than a failure. See README section 4.
KNOWN_FRAME_MISMATCH = {("combined", "cnnspot")}

# The manuscript's Table 13 column means, for the record of what is being replaced. Keys are
# (train set, test corpus). Values marked `~` in the comment are not quoted directly in the
# regeneration spec — they are reconstructed from the deltas it lists against the authoritative
# matrix, so they are good to +/-0.0005 and are reported for orientation only.
MANUSCRIPT_COLUMN_MEAN = {
    ("cnnspot", "cnnspot"): 0.9643,
    ("cnnspot", "synthbuster-plus"): 0.6623,
    ("cnnspot", "synthclic"): 0.5575,
    ("synthbuster-plus", "cnnspot"): 0.6752,  # ~ 0.6692 + 0.006
    ("synthbuster-plus", "synthbuster-plus"): 0.9942,  # ~ 0.9952 - 0.001
    ("synthbuster-plus", "synthclic"): 0.6350,
    ("synthclic", "cnnspot"): 0.3757,
    ("synthclic", "synthbuster-plus"): 0.7850,  # ~ 0.7940 - 0.009
    ("synthclic", "synthclic"): 0.9225,
    ("combined", "cnnspot"): 0.8410,
    ("combined", "synthbuster-plus"): 0.9608,
    ("combined", "synthclic"): 0.8700,  # ~ 0.8670 + 0.003
}


def head_from_checkpoint(artifact_id: str) -> tuple[np.ndarray, float]:
    """``(w, b)`` of a k=1 linear head stored as a Lightning checkpoint."""
    sd = D.get_checkpoint(artifact_id)
    w = np.asarray(sd["model.classification_head.fc.weight"], dtype=np.float64).ravel()
    b = float(np.asarray(sd["model.classification_head.fc.bias"]).ravel()[0])
    return w, b


def score_frame(artifact_id: str, split: str, w: np.ndarray, b: float) -> pd.DataFrame:
    """Predictions frame for one split of a snapshot pooler frame, scored by ``(w, b)``."""
    frame = D.get_frame(artifact_id, expected_space=D.SPACE_POOLER)
    x, _, sub = frame.split(split)
    z = np.asarray(x, dtype=np.float64) @ w + b
    return pd.DataFrame(
        {
            "image_id": sub["image_id"].astype(str).to_numpy(),
            "label": sub["label"].to_numpy().astype(int),
            "score": 1.0 / (1.0 + np.exp(-z)),
            "source": sub["source"].to_numpy(),
        }
    )


def combined_head_agrees(w: np.ndarray, b: float, tol: float = 1e-5) -> dict:
    """Check the combined checkpoint against the combined predictions that already exist.

    The 12th cell is the only one this script computes rather than re-aggregates, so the head it
    uses has to be shown to be the same head the other two combined cells came from. SynthCLIC is
    the reference (its test split is open; SB+'s is not).
    """
    stored = D.get_predictions("e3pred/combined__to__synthclic")
    fresh = score_frame("pooler/synthclic", "test", w, b)
    if not (stored["image_id"].astype(str).to_numpy() == fresh["image_id"].to_numpy()).all():
        raise AssertionError("combined head check: image_id order differs from the stored E3 frame")
    max_abs = float(np.abs(stored["score"].to_numpy() - fresh["score"].to_numpy()).max())
    if max_abs > tol:
        raise AssertionError(
            f"{COMBINED_HEAD} does not reproduce e3pred/combined__to__synthclic "
            f"(max |dscore| {max_abs:.2e} > {tol:.0e}). Refusing to use it for the "
            f"Combined -> CNNSpot cell."
        )
    return {"reference": "e3pred/combined__to__synthclic", "max_abs_score_delta": max_abs}


def predictions_for(train: str, test: str, inputs: list[str]) -> tuple[pd.DataFrame, str]:
    """``(predictions, provenance)`` for one table cell."""
    if (train, test) == COMPUTED_CELL:
        w, b = head_from_checkpoint(COMBINED_HEAD)
        inputs += [COMBINED_HEAD, EVAL_FRAME[test]]
        return score_frame(
            EVAL_FRAME[test], "test", w, b
        ), f"scored {COMBINED_HEAD} on {EVAL_FRAME[test]}"
    aid = f"e3pred/{train}__to__{test}"
    inputs.append(aid)
    return D.get_predictions(aid), aid


def cell_table(
    train: str, test: str, threshold: float, inputs: list[str]
) -> tuple[pd.DataFrame, str]:
    """Per-generator ACC + AP for one (train set, test corpus) cell."""
    pred, provenance = predictions_for(train, test, inputs)
    pairing = pairing_for_dataset(test)
    ap = per_generator_ap(pred, real_pairing=pairing)
    acc = per_generator_accuracy(pred, real_pairing=pairing, threshold=threshold)
    tbl = ap.merge(acc, on=["generator", "n_fake", "n_real"], validate="1:1")
    tbl.insert(0, "test_corpus", test)
    tbl["train_set"] = train
    tbl["real_pairing"] = pairing
    return tbl[
        ["test_corpus", "generator", "train_set", "acc", "ap", "n_fake", "n_real", "real_pairing"]
    ], provenance


def recipe_equivalence(detail: pd.DataFrame, seed: int, inputs: list[str]) -> pd.DataFrame:
    """Refit each training set under the F1 trainer and compare mAP against this table's own cells.

    The reused E3 predictions are only legitimate if the E3 estimator *is* the F1 estimator. F1's
    regression anchor already shows this for SynthCLIC in-domain; this widens it to every training
    set, on the two test corpora whose test splits are open.

    The comparator is the **generated** column mean rather than the authoritative matrix, so that
    the one cell the matrix measures on a different population (Combined -> CNNSpot) still tests
    the estimator rather than the frame difference.

    SynthBuster+ **test is excluded on purpose**: refitting and scoring it would be a new read of a
    split that this table otherwise never touches (README section 3).
    """
    rows = []
    for train in TRAIN_SETS:
        aid = f"pooler/{train}"
        inputs.append(aid)
        frame = D.get_frame(aid, expected_space=D.SPACE_POOLER)
        xtr, ytr, _ = frame.split("train")
        xva, yva, _ = frame.split("validation")
        head = train_head(xtr, ytr, xva, yva, seed=seed)
        w, b = head.weight, head.bias
        for test in ("cnnspot", "synthclic"):
            pred = score_frame(EVAL_FRAME[test], "test", w, b)
            refit = float(
                per_generator_ap(pred, real_pairing=pairing_for_dataset(test))["ap"].mean()
            )
            shipped = float(
                detail[(detail.train_set == train) & (detail.test_corpus == test)]["ap"].mean()
            )
            rows.append(
                {
                    "train_set": train,
                    "test_corpus": test,
                    "mAP_refit": round(refit, 4),
                    "mAP_table": round(shipped, 4),
                    "delta": round(refit - shipped, 4),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="ACC decision threshold on the sigmoid (0.5 == logit z > 0, the head's own rule)",
    )
    ap_.add_argument("--seed", type=int, default=123, help="seed for the recipe-equivalence refit")
    ap_.add_argument(
        "--skip-recipe-check", action="store_true", help="skip the refit equivalence check"
    )
    args = ap_.parse_args()

    missing = [i for i in ("pooler/cnnspot_full", "pooler/combined") if i not in D.load_manifest()]
    if missing:
        raise SystemExit(
            f"{', '.join(missing)} not in the snapshot.\n"
            f"Build it first:  make finalexp-data WITH_CFEVAL=1 WITH_APPENDIX=1"
        )

    inputs: list[str] = []
    run = Run(EXPERIMENT, "artifacts")

    head_check = combined_head_agrees(*head_from_checkpoint(COMBINED_HEAD))
    print(
        f"  combined head vs {head_check['reference']}: "
        f"max |dscore| {head_check['max_abs_score_delta']:.2e}  OK"
    )

    tables, provenance = [], {}
    for test in TEST_CORPORA:
        for train in TRAIN_SETS:
            tbl, prov = cell_table(train, test, args.threshold, inputs)
            tables.append(tbl)
            provenance[f"{train}->{test}"] = prov
    detail = pd.concat(tables, ignore_index=True)

    # Acceptance: every per-training-set column mean must reproduce the authoritative matrix.
    acc_rows = []
    for test in TEST_CORPORA:
        for train in TRAIN_SETS:
            block = detail[(detail.test_corpus == test) & (detail.train_set == train)]
            got, want = float(block["ap"].mean()), AUTHORITATIVE_MAP[(train, test)]
            acc_rows.append(
                {
                    "test_corpus": test,
                    "train_set": train,
                    "column_mean_ap": round(got, 4),
                    "authoritative_mAP": want,
                    "delta": round(got - want, 4),
                    "agrees_2dp": bool(round(got, 2) == round(want, 2)),
                    "known_frame_mismatch": (train, test) in KNOWN_FRAME_MISMATCH,
                    "manuscript_column_mean": MANUSCRIPT_COLUMN_MEAN[(train, test)],
                    "n_generators": int(len(block)),
                }
            )
    acceptance = pd.DataFrame(acc_rows)
    unexplained = acceptance[~acceptance.agrees_2dp & ~acceptance.known_frame_mismatch]

    recipe = None
    if not args.skip_recipe_check:
        recipe = recipe_equivalence(detail, args.seed, inputs)

    run.inputs = sorted(set(inputs))
    run.save_csv("per_generator_detail.csv", detail)
    run.save_csv("acceptance.csv", acceptance)
    if recipe is not None:
        run.save_csv("recipe_equivalence.csv", recipe)

    summary = {
        "experiment": EXPERIMENT,
        "target": "revision/main_revision.tex Table 13, \\label{tab:clip:test_results_detail}",
        "detector": "D_h — matched canonical 1024-d probe, F1 recipe, seed 123, NO augmentation",
        "recipe": RECIPE.as_dict() | {"seed": args.seed},
        "threshold_rule": (
            f"ACC at sigmoid > {args.threshold} (logit z > 0), applied uniformly to every cell and "
            f"computed on the same generator group as AP. State this in the caption."
        ),
        "real_pairing": {t: pairing_for_dataset(t) for t in TEST_CORPORA},
        "eval_frames": EVAL_FRAME,
        "eval_split": "test",
        "cell_provenance": provenance,
        "combined_head_check": head_check,
        "n_rows": int(len(detail)),
        "n_generators_per_corpus": {
            t: int(detail[detail.test_corpus == t]["generator"].nunique()) for t in TEST_CORPORA
        },
        "acceptance": {
            "cells": len(acceptance),
            "agree_2dp": int(acceptance.agrees_2dp.sum()),
            "known_frame_mismatch": sorted(f"{a}->{b}" for a, b in KNOWN_FRAME_MISMATCH),
            "unexplained_failures": unexplained.to_dict("records"),
            "passed": bool(unexplained.empty),
        },
        "recipe_equivalence": None if recipe is None else recipe.to_dict("records"),
        "sb_plus_test_access": (
            "NONE. The SynthBuster+ block re-aggregates per-image predictions written by the E3 "
            "cross-dataset runs on 2026-06-24, before EXTERNAL_VALIDATION_PROTOCOL.md was frozen "
            "(2026-07-18) and within the detection use it explicitly names. No model in this run "
            "is fitted on or scored against SB+ test; the recipe-equivalence refit skips it."
        ),
    }
    run.note(summary=summary)
    run.save_json("summary.json", summary)
    run.finish()

    print(f"\n  {len(detail)} rows, {detail.generator.nunique()} distinct generators")
    for t in TEST_CORPORA:
        n = detail[detail.test_corpus == t]["generator"].nunique()
        print(f"    {t:18s} {n:2d} generators, pairing={pairing_for_dataset(t)}")
    print("\n  acceptance vs e3_cross_matrix_mAP.csv:")
    print(acceptance.to_string(index=False))
    if recipe is not None:
        print("\n  recipe equivalence (refit under the F1 trainer vs the reused E3 cells):")
        print(recipe.to_string(index=False))
    if unexplained.empty:
        print("\n  ACCEPTANCE PASSED (known frame mismatch excepted, see README section 4)")
    else:
        raise SystemExit(f"\n  ACCEPTANCE FAILED for {len(unexplained)} cell(s):\n{unexplained}")


if __name__ == "__main__":
    main()
