# 3. Diagnostic Framework

We formulate strategy-controlled negotiation dialogue generation as a conditional text generation task. Given a dialogue context $x$ (the conversation history up to the current turn) and a target negotiation strategy $s \in \mathcal{S}$ from a predefined set of nine strategies, the model must generate the next utterance $y$ that is both contextually appropriate and aligned with $s$.

To systematically study how different parameter-efficient fine-tuning (PEFT) modules contribute to strategy control and generation quality, we construct a diagnostic framework that spans a spectrum of configurations—from pure text instruction to full hybrid adaptation with multiple auxiliary objectives. Rather than proposing a single "best" method, our framework is designed to isolate the functional roles of each component through controlled comparisons.

## 3.1 Base Model and Strategy-Specific Prefix

We use Qwen3-8B as the base language model, loaded in 4-bit quantization (bfloat16). On top of the frozen base model, we introduce a **strategy-specific Prefix bank**: for each of the nine negotiation strategies, we maintain an independent set of $K = 20$ learnable virtual token embeddings. At inference time, the prefix corresponding to the target strategy $s$ is prepended to the input sequence, so the model receives:

$$\tilde{x} = [p_1^{(s)}, p_2^{(s)}, \ldots, p_K^{(s)}; x_1, x_2, \ldots, x_T]$$

where $p_i^{(s)} \in \mathbb{R}^d$ are the strategy-specific prefix embeddings and $x_i$ are the token embeddings of the dialogue context. During training, the prefix embeddings are optimized via the standard autoregressive language modeling loss:

$$\mathcal{L}_{\text{gen}} = -\frac{1}{T} \sum_{t=1}^{T} \log P(y_t \mid \tilde{x}, y_{<t})$$

Crucially, Prefix-only training (B3) uses only this loss—the strategy differentiation emerges purely from the model's need to reduce perplexity differently for different strategy-conditioned inputs.

## 3.2 Shared LoRA Adapter

In parallel, we introduce a shared Low-Rank Adaptation (LoRA) branch applied to the query, key, value, and output projection matrices of all attention layers. For each weight matrix $W \in \mathbb{R}^{d \times d}$, LoRA learns a low-rank decomposition:

$$W' = W + \Delta W = W + \frac{\alpha}{r} \cdot BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, with rank $r = 16$ and scaling factor $\alpha = 32$. Dropout of 0.05 is applied to the LoRA activations. The LoRA parameters are shared across all nine strategies, meaning they are trained to improve general language modeling and domain adaptation without receiving explicit strategy identity.

## 3.3 Joint Prefix-LoRA Configuration

In the joint configuration (B4), both the strategy-specific Prefix bank and the shared LoRA adapter are active and trained simultaneously. The intuition is that Prefix handles strategy steering while LoRA handles general generation quality—a natural functional division of labor. However, as our results show, this division does not emerge automatically from joint optimization.

## 3.4 Orthogonal Regularization

To explicitly encourage functional separation between Prefix and LoRA, we introduce a local-global orthogonal constraint. For a given input, we compute the hidden-state deltas contributed by each module:

$$\Delta_{\text{prefix}} = \mathbf{h}(\text{prefix=on, lora=off}) - \mathbf{h}(\text{prefix=off, lora=off})$$
$$\Delta_{\text{lora}} = \mathbf{h}(\text{prefix=off, lora=on}) - \mathbf{h}(\text{prefix=off, lora=off})$$

where $\mathbf{h}(\cdot)$ denotes the hidden states at the final transformer layer, computed under different module activation conditions. The orthogonal loss combines token-level and sequence-level constraints:

$$\mathcal{L}_{\text{orth}} = \frac{1}{2} \cdot \underbrace{\frac{1}{T}\sum_{t=1}^{T} \cos^2(\Delta_{\text{prefix}}[t], \Delta_{\text{lora}}[t])}_{\text{local (token-level)}} + \frac{1}{2} \cdot \underbrace{\cos^2\!\big(\overline{\Delta}_{\text{prefix}}, \overline{\Delta}_{\text{lora}}\big)}_{\text{global (sequence-level)}}$$

where $\overline{\Delta}$ denotes mean pooling over the sequence dimension, and $\cos(\cdot, \cdot)$ is cosine similarity. The total training loss becomes:

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + \lambda_{\text{orth}} \cdot \mathcal{L}_{\text{orth}}$$

## 3.5 Auxiliary Strategy Classification

To strengthen the strategy signal in the Prefix representations, we add an auxiliary strategy classification head on top of the Prefix-induced hidden delta $\Delta_{\text{prefix}}$. The classification loss is:

$$\mathcal{L}_{\text{cls}} = -\log P_{\text{cls}}(s \mid \Delta_{\text{prefix}})$$

where $P_{\text{cls}}$ is a linear classifier that predicts the target strategy $s$ from the delta representation. The full loss becomes:

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + \lambda_{\text{orth}} \cdot \mathcal{L}_{\text{orth}} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}}$$

This configuration (B6) represents the most complete version of our hybrid framework. An early implementation contained a gradient leak in the delta-Prefix computation that caused training instability; after fixing this bug and tuning the loss schedule, we refer to the stable full configuration as **B6$_{\text{fix}}$**. We note that $\mathcal{L}_{\text{cls}}$ provides only a single scalar signal per training sample, whereas $\mathcal{L}_{\text{gen}}$ distributes its signal across all output tokens (typically hundreds per turn). This inherent signal density asymmetry is a key factor in our analysis.

## 3.6 Diagnostic Variants

Beyond the base configurations, we introduce three diagnostic variants to isolate specific interference mechanisms:

**Gradient routing (B6$_{\text{grad}}$).** To test whether gradient conflict between the generation and classification objectives is the primary cause of Prefix-LoRA interference, we reroute gradients: the generation loss gradient flows only to LoRA parameters (Prefix embeddings are detached during the generation forward pass), while the classification loss gradient flows to the Prefix bank and classifier head. Orth loss gradients flow to both modules. This separation ensures that the Prefix bank is optimized solely for strategy classification, not for token-level generation.

**Contrastive generation loss (B6$_{\text{contrastive}}$).** Motivated by the sparsity of the classification signal, we introduce a dense, token-level alternative. For each training sample with target strategy $s$, we also compute the generation loss using a randomly sampled incorrect strategy $s' \neq s$ (with no gradient):

$$\mathcal{L}_{\text{contrastive}} = \max\!\big(0,\; \mathcal{L}_{\text{gen}}(s) - \mathcal{L}_{\text{gen}}(s') + m\big)$$

where $m = 0.1$ is a margin. The intuition is that the correct strategy Prefix should produce lower generation loss than an incorrect one. This signal is dense (per-token) and directly aligned with the evaluation objective.

**Frozen Prefix analysis (B9).** To determine whether Prefix-LoRA interference occurs during training (joint optimization) or at inference (environment change), we first train the Prefix bank independently (B3), then freeze it and train only the LoRA adapter. At inference time, both modules are active. If interference stems from joint optimization, freezing the Prefix should preserve B3-level strategy control. If interference stems from LoRA changing the attention environment, then even a frozen Prefix should degrade—as our results confirm.

## 3.7 Evaluation Overview

We evaluate strategy controllability through swap-sample generation: for each dialogue context, we generate one response per strategy by swapping the active Prefix embedding, then classify each generated utterance with an LLM-based strategy evaluator. The evaluator is calibrated on human-written validation responses with ground-truth labels, and we report per-class calibration accuracy alongside model results. For the four key configurations (B3, B4, B7, B9), we train across three random seeds and report means, standard deviations, and bootstrap confidence intervals. Detailed evaluation parameters are provided in Section~4.
