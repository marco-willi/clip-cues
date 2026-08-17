# F7 — Bridge from the matched heads to everything they replace (addition, not in the spec)

**Status:** ✅ done · **Driver:** `scripts/finalexp/run_f7_bridge.py` ·
**Artifacts:** [`artifacts/`](artifacts/)

```bash
uv run python scripts/finalexp/run_f7_bridge.py
```

## Why this exists

The spec does not ask for it, but the consolidation needs it. Retraining the detector inventory
risks **orphaning the existing interpretation record**, which targets either the deployed
checkpoints (E12/N23–N25) or the CV-tuned sklearn proxies `P768t`/`P1024t` (N7–N14, N18). Without a
bridge, adopting `D_h`/`D_e` would silently invalidate ~20 registered numbers.

**Decision rule, stated before the run:** Σ-cos ≥ 0.9 **and** cue-profile ρ ≥ 0.9 **and** decision
agreement ≥ 0.95 ⇒ downstream N-numbers transfer by citation and the consolidation is
documentation-only.

## Results (SynthCLIC test, matched seed 123)

| pair | Σ-cos | logit ρ | decision agree | error Jaccard | cue-profile ρ | passes |
|---|---|---|---|---|---|---|
| **`D_e` ~ `P768t`** (primary) | +0.974 | 0.972 | 0.973 | 0.836 | 0.982 | **YES** |
| **`D_h` ~ `P1024t`** (primary) | +0.965 | 0.962 | 0.973 | 0.822 | 0.981 | **YES** |
| `D_h` ~ deployed k=1 | +0.946 | 0.934 | 0.959 | 0.729 | 0.979 | **YES** |
| `D_h` ~ deployed k=8 | +0.935 | 0.920 | **0.949** | 0.661 | 0.978 | *no* (by 0.001) |
| `D_h` ~ `D_e` | n/a¹ | 0.930 | 0.963 | 0.771 | 0.989 | YES |
| deployed k=1 ~ deployed k=8 | +0.984 | 0.979 | 0.976 | 0.815 | 0.995 | YES |

¹ Cross-space pair (1024-d vs 768-d): there is no basis-free cosine between different spaces, so the
direction columns are NaN **by construction** and the pair is judged on its score-level statistics.

AUROC by target: `D_h` 0.9209 · `D_e` 0.8966 · deployed k=1 0.9279 · deployed k=8 0.9252 ·
`P1024t` 0.9063 · `P768t` 0.8771.

## Verdict

**Both primary pairs pass, so the downstream record transfers by citation.** N7–N14 and N18 target
`P768t`/`P1024t`, and the matched heads reproduce those probes at Σ-cos ≥ 0.965, decision agreement
0.973 and cue-profile ρ ≥ 0.981. The consolidation is therefore a **documentation change**: no
scoped re-run list is triggered, and the escalation contingency (5-seed end-to-end augmented
training on Lambda) is not needed.

The deployed-checkpoint bridge also passes (`D_h` ~ deployed k=1: Σ-cos 0.946, agreement 0.959,
cue ρ 0.979), which keeps E12/N23–N25 attached to the new inventory.

**One pair misses, and only just.** `D_h` ~ deployed k=8 lands at decision agreement **0.949**
against a 0.95 threshold — a 0.001 shortfall on a *cross-head* comparison that was never one of the
primary criteria (it relates a k=1 head to a k=8 head trained differently, so some disagreement is
expected). It passes the other two criteria comfortably. Reported as a miss rather than rounded up.

## The E12 correction, measured directly

**Augmentation effect = AUROC(deployed k=1) − AUROC(matched `D_h`) = +0.0069.**

E12 attributed the ~0.02–0.04 deployed-vs-proxy gap to the deployed models' "augmented re-encoding
training protocol". This measures that attribution directly and it does not hold: training the same
head on the same cached features with **no augmentation at all** loses only ~0.007. The gap to the
proxies (0.921 vs 0.906/0.877) is therefore a **regularization/estimator effect** — standardized
features with CV-tuned `C` versus a fixed `weight_decay` — not an augmentation effect. E12's
numbers are unaffected; only that causal sentence was wrong, and it now carries a correction box.

**Independent replication:** deployed k=1 ~ k=8 cue-profile ρ = **0.995** here reproduces E12/N23's
0.992 (v2-128) / 0.995 (antonyms) from a separate code path.

## Files

| file | contents |
|---|---|
| `artifacts/summary.json` | every pair, the pass rule, AUROC by target, the augmentation-effect measurement, the verdict |
| `artifacts/bridge.csv` | the same battery as a table |
