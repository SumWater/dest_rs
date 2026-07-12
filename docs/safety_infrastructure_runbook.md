# Safety infrastructure runbook

Historical exact reproduction must use commit `4c10480bbdba29172ddcc51cc9c6cb750da16fef` or its frozen branch. This runbook is for the v2 safety infrastructure and migrated method-definition reruns.

## Local checks

```bash
python -m py_compile train.py src/config.py src/strategy_labels.py src/casino_dataset.py src/modeling.py src/losses.py src/evaluate.py scripts/remote_checks/inspect_b9_checkpoint.py scripts/remote_checks/verify_trainable_params.py scripts/remote_checks/verify_response_mask.py
python -m unittest discover -s tests -v
```

## Remote checks

Run from the repository root in the frozen experiment environment. Replace the checkpoint path and seed for each B9 seed.

```bash
python scripts/remote_checks/inspect_b9_checkpoint.py --config configs/b9_prefix_then_lora.json --checkpoint /ABSOLUTE/B9_CHECKPOINT --seed 42 --out reports/remote_checks/b9_seed42_checkpoint_check.json
python scripts/remote_checks/verify_trainable_params.py --config configs/b9_prefix_then_lora.json --checkpoint /ABSOLUTE/B9_CHECKPOINT --out reports/remote_checks/b9_seed42_trainable_params_check.json
python scripts/remote_checks/verify_response_mask.py --config configs/b9_prefix_then_lora.json --out reports/remote_checks/response_mask_check.json
```

Repeat checkpoint/trainable checks for seeds 43 and 44. Synchronize JSON reports back to the local repository before interpreting checkpoint state.
