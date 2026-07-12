# Repository-wide strategy definition drift audit

Date: 2026-07-12

## High-risk drift

### `other-need`

Incorrect partner-acknowledgment semantics occur in the root/solutions Dataset descriptions, old single-label Judge, gold calibration, formal multilabel Judge, verbalizer generation, and D1 augmentation generator. See `other_need_definition_audit.md`.

### `uv-part`

Two incompatible semantics exist:

- Incorrect: “unique value of items / why the item matters to the current speaker” in `src/casino_dataset.py`, `solutions/src/casino_dataset.py`, `scripts/evaluate_strategy_control_llm.py`, `solutions/scripts/evaluate_strategy_control_llm.py`, `scripts/calibrate_evaluator.py`, and `scripts/generate_augmented_data.py`.
- Correct Undervalue-Partner semantics in `scripts/evaluate_strategy_multilabel_llm.py` and `scripts/generate_verbalizer_upper_bound.py`.

Risk: old single-label evaluation/calibration and D1 synthetic training supervision do not implement CaSiNo semantics. Historical files remain immutable; future protocol-v1 runs must use corrected definitions.

## Remaining boundary drift

- `elicit-pref`: several constants say preferences, priorities **or situation/reasons**, which is too broad. Protocol v1 restricts it to discovering item preference order/priority.
- `promote-coordination`: older definitions mention collaboration/compromise but do not clearly include explicit trade offers and mutual concessions.
- `showing-empathy`: older definitions permit generic understanding/support without requiring connection to partner personal context.
- `vouch-fair`: older definitions emphasize explicit fairness/balance and may miss implicit imbalance callouts; conversely a compromise alone must not qualify.
- `no-need`: older “signal flexibility” wording may incorrectly label concession-only responses.
- `self-need`: broad wording is directionally correct, but associated-third-party needs must be separated into `other-need`.
- `small-talk`: definitions are broadly consistent; protocol v1 clarifies exclusion of substantive negotiation/item allocation.

## Impact by experiment/report

| Artifact | Impact | Required response |
|---|---|---|
| Gold Judge calibration | Old single-label definitions are wrong for `other-need` and `uv-part`; aggregate accuracy/confusion are protocol-dependent. | Re-run calibration with corrected Judge on unchanged original annotations. |
| B3/B4/B9 multilabel reevaluation | Formal multilabel Judge has correct `uv-part` direction but incorrect `other-need`; aggregate Target Presence, Primary Accuracy, Macro-F1, off-target and prediction distributions can change. PPL/repetition/distinct metrics are unaffected. | Re-evaluate existing saved generations; no retraining or regeneration required. |
| Profile ablation | Both P0/P1 were evaluated by the incorrect multilabel `other-need` definition. Direction/magnitude, especially other-need and aggregate metrics, are provisional. | Re-evaluate existing P0/P1 generations with protocol-v1 Judge. No generation rerun required. |
| Verbalizer upper bound | V2–V4 generation prompts and examples are wrong for `other-need`; evaluation Judge is also wrong. V0/V1 generation did not receive the wrong other-need definition, although V1 receives the label name. | Re-evaluate V0/V1 saved generations; regenerate at least the `other-need` V2–V4 tasks (prefer a clean full-condition rerun for protocol consistency), then re-evaluate. |
| B9 Top-K pilot | Candidate generation uses Prefix IDs, not text definitions. Candidate Judge has wrong `other-need`. Oracle@K and rerank statistics can change. | Re-label existing 720 candidates; no candidate regeneration required. |
| Context/response classifier diagnostic | Learns original annotation label names without Judge definitions. | Model computation is not directly affected; interpretation of class semantics should use protocol v1. No automatic rerun required solely for this mismatch. |
| Historical B3/B4/B9 training | D1 augmentation generator used wrong `other-need` and wrong `uv-part` semantics, so adapters may encode mismatched synthetic behavior. | Do not retrain per current instruction. Preserve as historical baselines; disclose D1 protocol mismatch and evaluate saved outputs under protocol v1. Reconsider B9 initialization risk before M-series data/training. |

## Current research conclusions

- The overall control–fluency trade-off may still exist, because model outputs and PPL are unchanged, but its corrected magnitude is not yet known.
- Per-class claims for `other-need`, and aggregate multilabel metrics containing it, are provisional.
- The Top-K claim “other-need is generation-limited” is provisional until candidates are re-labeled.
- The verbalizer upper-bound magnitude is provisional because its other-need intervention used the wrong behavior.
- PPL, repetition, Distinct-1/2, raw generations, checkpoint hashes, and split-leakage results are unaffected by the Judge-definition correction.
