# M-series multi-loader schedule and exposure control

This is the first-stage schedule design. It is infrastructure documentation only; M-01/M-02/M-03 must not start until checkpoint, label protocol, mask, and remote checks pass.

## Optimizer-step definition

One optimizer step consists of one forward/backward loss aggregation followed by one optimizer update. Gradient accumulation is `1` for the first-stage schedule. Each source loss is a token-normalized mean within its own batch before configured loss weights are applied.

## Fixed first-stage schedule

Assuming approximately 180 accepted synthetic positives and 180–300 accepted pairs:

| Experiment | max_train_steps | real batch | synthetic batch | pair batch | Per-step sources |
|---|---:|---:|---:|---:|---|
| M-01 | 180 | 0 | 4 | 0 | synthetic LM |
| M-02 | 180 | 4 | 4 | 0 | real LM + synthetic LM |
| M-03 | 180 | 4 | 4 | 2 | real LM + synthetic LM + ranking |

The step budget, real batch sequence, and synthetic batch sequence are identical for M-02 and M-03. M-03's only added training signal is the pair batch and ranking loss.

## Loader termination and cycling

- `max_train_steps=180` is the only training termination criterion; dataset “epoch” does not terminate a run.
- Every loader uses a separately seeded deterministic shuffled iterator.
- When a loader is exhausted, only that loader is re-created with `source_seed + completed_cycle` and continues.
- The run stops immediately after optimizer step 180; no loader is drained to finish a nominal epoch.
- M-02 and M-03 use identical source seeds and sampler implementation for real and synthetic loaders.
- Dataset fingerprints and the exact ordered sample IDs consumed per source must be stored in the run manifest. This permits verification that M-02 and M-03 saw the same real/synthetic sequence.

## Expected exposure

At 180 steps:

- M-01 synthetic samples: 720, approximately 4.0 equivalent epochs for 180 accepted samples.
- M-02 synthetic samples: 720, approximately 4.0 equivalent epochs; real samples: 720, approximately 0.14 equivalent epochs if the real set has 5,285 usable samples.
- M-03 synthetic samples: 720 and real samples: 720, exactly comparable to M-02; pair samples: 360, approximately 2.0 equivalent epochs for 180 pairs or 1.2 for 300 pairs.

Equivalent epochs are calculated from actual counters, not planned sizes:

```text
source_equivalent_epochs = source_samples_seen / source_dataset_size
```

If the accepted synthetic size differs materially from 180, keep the cap at four synthetic equivalent epochs and derive `max_train_steps = ceil(4 * synthetic_size / synthetic_batch_size)`. The same derived value must be used for M-01/M-02/M-03. A hard safety cap of 240 optimizer steps prevents accidental high-cycle training in the pilot.

## Required counters

Every run summary must contain:

- optimizer_steps and wall_clock_seconds;
- real/synthetic/pair batches_seen and samples_seen;
- real/synthetic/chosen/rejected response_tokens_seen;
- real/synthetic/pair equivalent_epochs;
- dataset sizes, fingerprints, sampler seeds, and consumed sample-ID trace hashes.

Training must fail if the final M-02 and M-03 real or synthetic exposure counts differ from plan, or if their pre-run manifests show different B9 hash, real/synthetic fingerprints, seed, batch size, sampler seed, or max_train_steps.
