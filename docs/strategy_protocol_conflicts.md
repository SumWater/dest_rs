# Strategy protocol conflict report

Status: resolved by CaSiNo Section 3/Table 2 review and project-owner decisions on 2026-07-12. The approved definitions are in `docs/strategy_protocol_v1.md`.

## Evidence inspected

- CaSiNo annotations in the repository's official/augmented splits;
- `src/casino_dataset.py` strategy-text descriptions;
- legacy single-label Judge prompts;
- `scripts/evaluate_strategy_multilabel_llm.py` definitions used for the 2026-07-11 phase report;
- verbalizer definitions and examples;
- evaluator calibration reports and phase report.

The raw CaSiNo files provide labels and utterances but do not include an authoritative definition document. Therefore annotation examples are behavioral evidence, not proof of the original taxonomy wording.

## Conflicts by class

| Strategy | Repository evidence | Conflict / ambiguity | Status |
|---|---|---|---|
| elicit-pref | Older prompts included broad “situation/reasons” questions. | Generic questions were too broad. | Resolved: only discovery of item preference order/priority. |
| no-need | Older text allowed generic flexibility. | Concession-only replies could be over-labeled. | Resolved: speaker must express own low/no need or sufficient supply. |
| other-need | Old Judges described partner-need acknowledgment. | This contradicted CaSiNo's Self-Need-like third-party semantics. | Resolved: speaker-associated third-party need. |
| promote-coordination | Older definitions mentioned collaboration/compromise. | Explicit trades and mutual concessions needed clearer inclusion. | Resolved in protocol v1. |
| self-need | Sources agreed on current speaker's own need/reason. | Needed separation from associated-third-party need. | Resolved in protocol v1. |
| showing-empathy | Older definitions permitted generic understanding/support. | Formulaic acknowledgment could be over-labeled. | Resolved: positive acknowledgment of partner personal context. |
| small-talk | Sources broadly agreed. | Needed exclusion of substantive negotiation/allocation. | Resolved in protocol v1. |
| uv-part | Old Dataset/Judge/augmentation used self-directed “unique value”; newer Judge used partner-undervaluation. | Materially incompatible behaviors. | Resolved: **Undervalue-Partner**. |
| vouch-fair | Older definitions emphasized explicit fairness. | Needed implicit imbalance callouts and exclusion of compromise-only replies. | Resolved in protocol v1. |

## Resolution

- `uv-part` is frozen as **Undervalue-Partner**, not unique value to self.
- `other-need` is frozen as a need established by the current speaker for an associated third party, not acknowledgment of the negotiation partner's need.
- The remaining boundaries are frozen in `docs/strategy_protocol_v1.md`.
- Historical artifacts are not rewritten; protocol-mismatched evaluations are re-run from saved generations where possible.
