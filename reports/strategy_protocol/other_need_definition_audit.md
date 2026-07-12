# Other-Need definition audit

Date: 2026-07-12  
Canonical authority: CaSiNo Section 3 Strategy Annotations and Table 2, as verified by the project owner.

Canonical definition: **Other-Need is similar to Self-Need but is used when the current speaker discusses a need for someone else associated with the speaker rather than for themselves** (children, family, friends, group members, camping companions, etc.). It is not acknowledgment of the negotiation partner's need.

## Audited definition locations

“Current definition/behavior” below records the definition found at audit time. Active definition constants were corrected on `exp/counterfactual-sequence-ranking` after the mismatch was documented; historical outputs were not changed.

| File | Current definition/behavior | CaSiNo compliant? | Risk | Required action |
|---|---|---:|---|---|
| `src/casino_dataset.py` | “Acknowledge, discuss, or accommodate the other party's needs.” Used by optional strategy-text injection. | No | Critical if text injection or future generation uses it | Corrected on new branch to canonical third-party-associated need definition. |
| `solutions/src/casino_dataset.py` | Historical copied definition identical to root version. | No | High, historical/diagnostic branch | Preserve historical result files; mark code historical or update only for future reruns. |
| `scripts/evaluate_strategy_control_llm.py` | Single-label Judge defined partner acknowledgment as other-need. | No | Critical evaluation mismatch | Definition corrected; do not use old output as final evidence. Recalibrate/re-evaluate. |
| `solutions/scripts/evaluate_strategy_control_llm.py` | Same incorrect single-label Judge definition. | No | Critical evaluation mismatch | Treat historical solution-evaluation outputs as protocol-mismatched. |
| `scripts/calibrate_evaluator.py` | Gold calibration used the same incorrect partner-need definition. | No | Critical calibration mismatch | Definition corrected; re-run calibration on unchanged original annotations. |
| `scripts/evaluate_strategy_multilabel_llm.py` | “states or acknowledges the other participant's concrete need or reason.” | **No** | **CRITICAL_EVALUATION_PROTOCOL_MISMATCH** | Definition corrected; re-evaluate every saved generation assessed by the old Judge. |
| `scripts/evaluate_candidates_multilabel_batch.py` | Imports the multilabel Judge prompt above. | No | Critical for Top-K Oracle/Judge results | Re-label saved Top-K candidates; candidate generation need not rerun. |
| `scripts/generate_verbalizer_upper_bound.py` | Definition and example described acknowledging the negotiation partner's need. | No | Critical for V2–V4 generation; evaluation also inherited wrong multilabel Judge | Definition/example corrected; regenerate V2–V4 other-need tasks, then re-evaluate. V0/V1 generations can be reused. |
| `scripts/generate_augmented_data.py` | Definition/examples/guidance explicitly generated acknowledgment of the other party. | No | **Critical training-data semantic mismatch** | Future prompt corrected. Do not rewrite D1; record it as protocol-mismatched supervision. |
| `src/casino_dataset.py` raw CaSiNo parsing | Reads annotation names without redefining them. | Yes, assuming original annotation semantics | Low | Preserve original annotations unchanged. |
| `README.md` | Lists the strategy name but provides no behavioral definition. | Not contradictory | Low | Link future documentation to protocol v1. |
| `docs/strategy_protocol_conflicts.md` | Previously treated partner need acknowledgment as one possible definition. | Superseded | Documentation | Mark resolved and point to protocol v1. |
| `docs/strategy_protocol_v1.md` | Previously draft/blank pending decision. | Superseded | Documentation | Replaced with approved canonical protocol. |

## Critical finding

The formal multilabel Judge used for the 2026-07-11 phase report has an incorrect `other-need` definition. This is classified as:

`CRITICAL_EVALUATION_PROTOCOL_MISMATCH`

The previous `other-need` per-class metrics and any aggregate metrics incorporating that class are provisional. They must not be used as final conclusions until corrected re-evaluation is complete.

## Historical artifact policy

- Do not modify original CaSiNo annotations.
- Do not modify historical generations.
- Do not overwrite old Judge reports; write protocol-v1 reports to new paths.
- Do not describe corrected reruns as exact reproductions of the old evaluation protocol.
