# 4. Experimental Setup

## 4.1 Dataset and Preprocessing

We use the CaSiNo negotiation dialogue dataset, which contains 1,030 two-party negotiation dialogues annotated with nine fine-grained strategies: *elicit-pref*, *self-need*, *other-need*, *no-need*, *promote-coordination*, *showing-empathy*, *small-talk*, *uv-part*, and *vouch-fair*.

**Data augmentation.** The original CaSiNo training set suffers from two limitations for our task: severe class imbalance (the most frequent strategy appears approximately 15 times more often than the least frequent) and low annotation coverage (only 396 of 1,030 dialogues carry strategy labels). To address these issues, we augment the training set using Qwen3.5-9B to generate strategy-driven negotiation dialogues. The augmentation procedure generates multi-turn dialogues where each turn is conditioned on a specific target strategy, using strategy definitions and few-shot examples as prompts. The augmented training set contains 610 dialogues with 5,285 single-label turns, achieving substantially better class balance (maximum-to-minimum ratio reduced to approximately 2:1).

**Data filtering.** The original CaSiNo annotations include multi-label samples (e.g., a single utterance annotated with both *self-need* and *other-need*) and *non-strategic* labels (utterances with no identifiable strategy). We exclude both categories from training: multi-label samples are dropped (22\% of original annotations), and *non-strategic* labels are filtered via the exclude-labels configuration (31\% of original annotations). This reduces the usable training data from the original set but ensures clean single-label supervision.

**Validation and test sets.** We use the original CaSiNo validation (41 dialogues, 230 labeled turns) and test (41 dialogues) splits without augmentation. This preserves comparability with prior work and ensures that evaluation reflects performance on human-annotated data.

## 4.2 Base Model and PEFT Configuration

**Base model.** We use Qwen3-8B as the base language model, loaded in 4-bit quantization with bfloat16 compute precision via the BitsAndBytes library. The base model parameters are frozen throughout all experiments.

**Strategy-specific Prefix.** Each of the nine strategies is associated with an independent prefix embedding consisting of $K = 20$ virtual tokens of dimension $d = 4096$. Prefix embeddings are initialized from $\mathcal{N}(0, 0.02^2)$. At training and inference time, the prefix corresponding to the target strategy is prepended to the input sequence before the dialogue context tokens.

**Shared LoRA.** LoRA is applied to the query, key, value, and output projection matrices ($q\_proj$, $k\_proj$, $v\_proj$, $o\_proj$) of all attention layers. We use rank $r = 16$, scaling factor $\alpha = 32$, and dropout $p = 0.05$. LoRA parameters are initialized following the standard Kaiming uniform scheme for $A$ and zeros for $B$. All LoRA parameters are shared across strategies.

## 4.3 Training Details

All models are trained using the AdamW optimizer with gradient clipping at 1.0. We use separate learning rates for Prefix and LoRA parameters: $\eta_{\text{prefix}} = 5 \times 10^{-4}$ and $\eta_{\text{LoRA}} = 1 \times 10^{-4}$. The classifier head (when used) shares the Prefix learning rate. Training batch size is 1 (single dialogue turn per step), and each experiment is trained for 2 epochs unless otherwise noted. All experiments use a single NVIDIA GPU.

**Loss weights across experiments.** Different experimental configurations use different loss weightings. Appendix~A (or a supplementary table) provides a complete list. The key configurations are:

| Experiment | $\lambda_{\text{orth}}$ | $\lambda_{\text{cls}}$ | $\lambda_{\text{contrastive}}$ | orth every N steps |
|---|---|---|---|---|
| B3 (Prefix-only) | 0 | 0 | 0 | -- |
| B4 (Prefix+LoRA) | 0 | 0 | 0 | -- |
| B5 (+ Orth) | 0.05 | 0 | 0 | 20 |
| B6$_{\text{fix}}$ (+ Orth + Cls) | 0.10 | 1.0 | 0 | 1 |
| B6$_{\text{cls}=5.0}$ | 0.10 | 5.0 | 0 | 1 |
| B6$_{\text{orth}=1.0}$ | 1.00 | 1.0 | 0 | 1 |
| B6$_{\text{grad}}$ (Grad. Routing) | 0.10 | 1.0 | 0 | 1 |
| B6$_{\text{contrastive}}$ | 0.10 | 0 | 1.0 | 1 |

The contrastive margin is $m = 0.1$. For gradient routing, the Prefix embeddings are detached during the generation forward pass so that $\mathcal{L}_{\text{gen}}$ gradients flow only to LoRA.

**Bug fix note.** In early versions of our code, the second forward pass in the delta-Prefix computation (used for orthogonal loss and auxiliary classification) lacked `torch.no_grad()`, causing unintended gradient flow from $\mathcal{L}_{\text{cls}}$ to the Prefix bank through an inconsistent computational path. This caused generation loss to explode from approximately 1.2 to 13.9 in the original B6 configuration. This bug was identified and fixed before all results reported in this paper. The fix ensures that all delta computations are properly detached, and we additionally set `orth_every_n_steps = 1` (instead of 20) to provide consistent supervision at every training step.

## 4.4 Evaluation Protocol

**Swap-sample strategy evaluation.** Strategy controllability is evaluated via swap-sample generation. For each of 30 contexts sampled from the validation set, we generate one response per strategy by swapping the active Prefix embedding while keeping all other conditions identical (prompt, generation parameters, LoRA state). This yields 270 generated utterances per experiment. Generation uses greedy decoding with a maximum of 40 new tokens. For the text-instruction baseline (B1), strategy identity is injected via a text prompt rather than a prefix swap.

**LLM-based strategy evaluator.** We classify each generated utterance using the same Qwen3-8B model in a separate inference pass. The evaluator receives a system prompt explaining the task, definitions of all nine strategies, 18 few-shot examples (2 per strategy, drawn from the training set), and the dialogue context. It outputs a single strategy name. We disable the model's chain-of-thought generation to prevent the strategy name from appearing in the reasoning trace and leaking into the classification. The evaluator reports overall 9-class accuracy and per-class accuracy.

**Evaluator calibration.** We calibrate the evaluator on 200 human-written utterances from the original CaSiNo validation set with ground-truth strategy labels. The same few-shot prompt and strategy definitions are used, but the utterances being classified are human-written rather than model-generated. We report per-class and overall calibration accuracy alongside all model results.

**Perplexity.** We report valid-set perplexity as a proxy for generation fluency and domain adaptation quality. For each experiment, we select the checkpoint with the lowest validation loss and report its valid PPL. In experiments where both Prefix and LoRA are active, PPL reflects the combined effect of both modules.

**Multi-seed evaluation.** For the four key configurations (B3, B4, B7, B9), we train across three independent random seeds (42, 43, 44) and report means, standard deviations, and bootstrap 95\% confidence intervals (10,000 resamples) on strategy accuracy. For PPL, we report the mean across seeds. Multi-seed confidence intervals use all 810 samples (3 seeds $\times$ 270 generations) for bootstrap estimation.

**Computational cost.** Each experiment (2 epochs, batch size 1) requires approximately 1.5 GPU-hours on an NVIDIA A100-80G. The complete set of experiments (including multi-seed and diagnostic variants) requires approximately 50 GPU-hours. LLM-based strategy evaluation adds approximately 0.5 GPU-hours per experiment.
