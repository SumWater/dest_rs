# Backward-compatibility audit for the M-series branch

Date: 2026-07-12  
Branch: `exp/counterfactual-sequence-ranking`  
Historical frozen baseline commit: `4c10480bbdba29172ddcc51cc9c6cb750da16fef`

The historical commit/branch is the only source for exact B3/B4/B9 reproduction. Results produced by this branch must use new experiment IDs and must not be described as exact reproductions of the frozen results.

## 1. Can the new code read the old configurations?

`configs/b3_prefix_only.json`, `configs/b4_prefix_lora.json`, and `configs/b9_prefix_then_lora.json` all pass the new strict configuration parser. Missing new fields receive safe `TrainConfig` defaults.

Parsing a configuration does not guarantee that training starts:

- B3 and B4 have no requested warm start, so fail-closed checkpoint rules do not block model creation.
- B9 requests a Prefix warm start. Its configured Prefix checkpoint must exist, contain ordered labels, match the canonical order, and have the expected shape. The locally absent remote checkpoint therefore causes an intentional FAIL rather than random Prefix initialization.
- All three require the training data to contain all nine canonical strategies and no unknown strategies.

## 2. Old configurations/checkpoints that now fail

- Any requested Prefix/LoRA warm start whose directory or required files do not exist.
- A Prefix checkpoint with no `labels` list, a different order, unknown labels, or a different tensor shape.
- A serialized `label_map.json` whose order differs from the canonical label space.
- A training split missing any canonical strategy or containing an unknown strategy.
- A configuration containing an unknown/misspelled field.
- A batch with empty/all-masked response supervision, inconsistent explicit mask, shape mismatch, NaN, or Inf.
- M-series startup if Base, LoRA, or classifier is trainable, Prefix is not trainable, or optimizer membership differs from `requires_grad`.

The old B9 config also retains a stale, non-seed-specific path (`output/other/casino_augmented/b3_prefix_only`). It will FAIL unless that exact valid checkpoint exists. It should not be used for a new run.

## 3. Safety-only changes that preserve the valid-sample model definition

- Replacing dynamically sorted labels with the same frozen order used by the formal saved label maps.
- Failing on missing checkpoint files instead of silently creating random Prefix/new LoRA weights.
- Checking checkpoint label order and parameter shape before copying tensors.
- Checking trainable partitions and exact optimizer membership.
- Carrying an explicit response mask that is equivalent to `labels != -100` for valid historical batches.
- Rejecting unknown configuration keys and invalid numeric/mask states.

For a valid nonempty batch with the same tokenization, same canonical IDs, same checkpoint, and same trainable parameters, the intended Prefix/LoRA architecture and response-only causal-LM objective are unchanged.

## 4. Changes that can make a rerun non-comparable to frozen numbers

- Corpus PPL is now token-weighted rather than an unweighted mean of batch losses. Reported PPL can change, and best-epoch selection can change when validation response lengths/batch sizes vary.
- The new loss implementation computes response log-softmax in float32 and fails on invalid samples. Small numerical differences from the historical `cross_entropy` path are possible.
- Canonical validation rejects or exposes data/version problems that the old pipeline could silently filter/remap. A resulting cleaned dataset is not necessarily the historical dataset.
- A different Transformers/PEFT/PyTorch/CUDA version, quantization kernel, tokenizer, sampler order, or checkpoint path remains non-comparable even when the method name is unchanged.
- Migrated configs use new experiment IDs/output paths and must be accompanied by dataset/checkpoint hashes. They are method-definition reruns, not exact historical reproductions.

## 5. Migrated configs

- `configs/b3_prefix_only_v2.json`
- `configs/b4_prefix_lora_v2.json`
- `configs/b9_prefix_then_lora_v2.json`

They preserve the B3/B4/B9 trainable-module and loss definitions while making canonical labels, module enable/train flags, and B9 warm-start dependency explicit. B9 v2 resolves its Prefix from B3 v2 under the same dataset tag and fails if it is absent.

Before a migrated baseline run, record:

- frozen historical reference ID and commit;
- new branch commit;
- dataset split hashes;
- canonical label-space hash;
- base model/tokenizer identifier and hash;
- initialization checkpoint hashes;
- environment manifest;
- statement: “method-definition rerun under v2 safety infrastructure; not an exact reproduction.”

No `legacy_mode` is provided. Unsafe historical behavior is intentionally available only from the frozen historical commit.
