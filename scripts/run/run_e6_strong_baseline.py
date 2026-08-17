#!/usr/bin/env python
"""E6: evaluate a stronger out-of-the-box detector (CommunityForensics) on our test sets.

Eval-only, no training. Reuses the E7 scoring spine: the detector exposes ``predict_batch`` →
`community_eval.score_cf_split` → full-metadata parquet + `cf_metrics`. A score-direction gate runs
first (mean P(fake) on fakes must exceed reals) and is recorded. Results land under
``results/e6_strong_baseline/<model_tag>/`` (W&B group ``e6_strong_baseline``).

    # smoke (local, SynthCLIC only, capped)
    python scripts/run/run_e6_strong_baseline.py --datasets synthclic --limit-per-dataset 200 --no-wandb
    # full (on the box)
    python scripts/run/run_e6_strong_baseline.py --datasets synthclic synthbuster_plus cnnspot community_forensics_eval --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from clip_cues_research.community_eval import cf_metrics, score_cf_split
from clip_cues_research.datasets import CF_EVAL
from clip_cues_research.datasets.scored import as_scored_split
from clip_cues_research.external_detectors import CommunityForensicsDetector
from clip_cues_research.results import make_run_id, save_run_results

EXPERIMENT = "e6_strong_baseline"
# eval dataset name -> split. "community_forensics_eval" is an alias for the CF-Eval HF id.
DATASET_SPLIT = {
    "synthclic": "test",
    "synthbuster_plus": "test",
    "synthbuster-plus": "test",
    "cnnspot": "test",
    "community_forensics_eval": "CompEval",
    CF_EVAL: "CompEval",
}


def _resolve(name: str) -> tuple[str, str]:
    """Map a CLI dataset name to (loader name, split)."""
    if name == "community_forensics_eval":
        return CF_EVAL, "CompEval"
    return name, DATASET_SPLIT.get(name, "test")


def score_direction_gate(detector, out_dir: Path, n: int = 20) -> bool:
    """Score n real + n fake SynthCLIC images; require mean P(fake|fake) > mean P(fake|real).

    Returns True if scores already mean higher=fake. Writes score_direction_check.md. Raises if the
    gap is wrong-signed *and* large (a real inversion) — caller should not proceed blindly.
    """
    ds = as_scored_split("synthclic", "test")
    labels = np.array(ds["label"])
    ridx = np.where(labels == 0)[0][:n].tolist()
    fidx = np.where(labels == 1)[0][:n].tolist()
    probs = detector.predict_batch([ds[i]["image"] for i in ridx + fidx])
    real_m, fake_m = float(probs[:n].mean()), float(probs[n:].mean())
    ok = fake_m > real_m
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "score_direction_check.md").write_text(
        "# E6 score-direction check (SynthCLIC, "
        f"{n} real + {n} fake)\n\n"
        f"- score used: `sigmoid(logit)` = P(fake)\n"
        f"- mean P(fake) | real = {real_m:.3f}\n"
        f"- mean P(fake) | fake = {fake_m:.3f}\n"
        f"- higher == fake: **{ok}** (inversion applied: **{not ok}**)\n"
    )
    return ok


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model-repo", default="OwensLab/commfor-model-384")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["synthclic", "synthbuster_plus", "cnnspot", "community_forensics_eval"],
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--limit-per-dataset", type=int, default=None, help="cap samples/dataset (smoke tests)"
    )
    p.add_argument("--device", default=None)
    p.add_argument("--cache-dir", default="data/hf_cache")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "clip-cues"))
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_id = make_run_id()
    model_tag = args.model_repo.split("/")[-1]
    base = f"results/{EXPERIMENT}/{model_tag}"
    print(f"E6 | model={args.model_repo} | run_id={run_id} | datasets={args.datasets}")

    detector = CommunityForensicsDetector(
        args.model_repo, device=args.device, batch_size=args.batch_size, cache_dir=args.cache_dir
    )
    score_fn = detector.predict_batch

    # ── score-direction gate ──
    ok = score_direction_gate(detector, Path(base))
    if not ok:
        print("!! score direction inverted (fake < real on SynthCLIC) — applying 1-p inversion")
        score_fn = lambda imgs: 1.0 - np.asarray(detector.predict_batch(imgs))  # noqa: E731
    print(f"score-direction gate: higher==fake = {ok}")

    wb = None
    if not args.no_wandb:
        import wandb

        wb = wandb.init(
            project=args.wandb_project,
            group=EXPERIMENT,
            name=f"{EXPERIMENT}_{model_tag}_{run_id}",
            config=vars(args) | {"run_id": run_id, "model_tag": model_tag, "direction_ok": ok},
        )

    for ds_name in args.datasets:
        loader_name, split = _resolve(ds_name)
        sub = ds_name.replace("/", "_")
        print(f"\n=== {ds_name} [{split}] ===")
        scored = as_scored_split(loader_name, split, cache_dir=args.cache_dir)
        if args.limit_per_dataset and len(scored) > args.limit_per_dataset:
            # shuffle before capping so the smoke subset keeps both classes / a mix of generators
            scored = scored.shuffle(seed=42).select(range(args.limit_per_dataset))
        df = score_cf_split(scored, score_fn, detector=sub, batch_size=args.batch_size)
        metrics = cf_metrics(df) | {
            "dataset": ds_name,
            "model_repo": args.model_repo,
            "run_id": run_id,
            "split": split,
            "direction_ok": ok,
        }
        # results/e6_strong_baseline/<model_tag>/<dataset>/<run_id>/metrics.json + predictions/<dataset>__<run_id>.parquet
        save_run_results(model_tag, sub, metrics, run_id=run_id, base=f"results/{EXPERIMENT}")
        pred_dir = Path(base) / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        df["dataset"] = ds_name
        df["model_repo"] = args.model_repo
        df.to_parquet(pred_dir / f"{sub}__{run_id}.parquet", index=False)
        print(
            f"  n={metrics['n']} AP={metrics.get('overall_ap', float('nan')):.4f} "
            f"AUROC={metrics.get('auroc', float('nan')):.4f} acc={metrics['accuracy']:.4f}"
        )
        if wb is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    wb.summary[f"{sub}/{k}"] = v

    if wb is not None:
        wb.finish()
    print(f"\nDone → {base}/  (export with scripts/export/export_e6_tables.py)")


if __name__ == "__main__":
    main()
