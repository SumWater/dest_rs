# Abstract, Related Work, Discussion, Conclusion

---

## Abstract

Negotiation dialogue generation requires responses that are both fluent and strategically aligned. Parameter-efficient fine-tuning (PEFT) offers a practical means of adapting large language models for controllable generation, yet the functional roles and interactions of different PEFT modules remain underexplored. In this paper, we conduct a diagnostic study of strategy-specific Prefix Tuning and shared LoRA adaptation for strategy-controlled negotiation dialogue generation on the CaSiNo dataset. Through systematic comparison of text prompting, Prefix-only, LoRA-only, joint Prefix-LoRA training, and six diagnostic variants (orthogonal regularization, auxiliary classification, gradient routing, contrastive generation, warm-start, and staged training), we find that Prefix-only tuning provides the strongest strategy controllability (55.9\% accuracy), while LoRA substantially improves generation fluency (24.8\% perplexity reduction). However, naive joint training introduces a robust control-quality trade-off, degrading strategy accuracy by approximately 19 percentage points. Diagnostic experiments reveal that orthogonal constraints, auxiliary classification, gradient routing, and contrastive losses do not reliably resolve this interference. Further analysis shows that LoRA disrupts Prefix control signals at the inference level—even frozen Prefix weights degrade when LoRA is introduced. We also calibrate the LLM-based strategy evaluator on human-written responses, finding only 52.0\% accuracy and substantial class-dependent shortcut risks. Our results provide a comprehensive diagnostic portrait of hybrid PEFT for controllable generation and highlight the gap between representation-space separation and functional strategy realization.

---

## 2. Related Work

### 2.1 Negotiation Dialogue and Strategy

Negotiation dialogue has been studied extensively in NLP, with datasets such as CaSiNo, Deal or No Deal, and CraigslistBargain enabling research on strategic communication. Unlike open-domain chit-chat, negotiation involves structured goals: parties exchange offers, express preferences, and employ persuasive tactics. Chawla et al. annotated the CaSiNo dataset with nine fine-grained negotiation strategies, including *elicit-pref* (eliciting preferences), *self-need* (expressing one's own needs), and *promote-coordination* (proposing collaboration). Generating utterances that faithfully reflect a target strategy—while maintaining fluency and contextual relevance—remains an open challenge.

Prior work on strategy-controlled generation has explored decoding-time constraints, reinforcement learning from strategy rewards, and conditional language modeling. However, these approaches often trade flexibility for controllability, or require careful reward design that may not generalize across strategies. The emergence of PEFT methods creates new opportunities: lightweight trainable modules can encode structured control signals without full model fine-tuning.

### 2.2 Parameter-Efficient Fine-Tuning

PEFT methods adapt large pre-trained models through a small number of additional parameters while keeping the base model frozen. The two approaches most relevant to our work are Prefix Tuning and LoRA.

**Prefix Tuning** prepends a sequence of learnable continuous vectors (virtual tokens) to the input. These prefix embeddings are optimized to steer the model's behavior in a task-specific or attribute-specific manner. Because different prefixes can be trained independently, Prefix Tuning naturally supports multi-attribute control: each strategy can be associated with its own prefix, and control is exerted by swapping the active prefix at inference time.

**LoRA** injects low-rank decomposition matrices into the weight matrices of attention projections (query, key, value, output). Empirical studies have shown that LoRA effectively captures task-specific knowledge while preserving the base model's general capabilities. However, because LoRA parameters are shared across inputs, they encode global adaptation patterns rather than instance-level control signals.

**Hybrid PEFT** methods combine multiple adapter types. Kim et al. (EMNLP 2024 Findings) analyzed the representational properties of Prefix Tuning and LoRA, finding through SVD analysis that LoRA reduces the effective rank of pre-trained representations while Prefix Tuning preserves the full representational space. They proposed sequential composition (Prefix $\rightarrow$ LoRA). Our work extends this line of inquiry by systematically diagnosing the interference mechanisms when Prefix and LoRA are combined, and by testing whether orthogonal constraints, auxiliary supervision, and gradient routing can resolve interference.

### 2.3 Controllable Text Generation

Controllable generation aims to steer model outputs toward desired attributes—sentiment, topic, formality, persona, or, in our case, negotiation strategy. Methods range from prompt engineering and controlled decoding to fine-tuning with attribute-conditioned architectures. In the PEFT paradigm, adapters and prefixes have been used for attribute control in tasks such as sentiment-controlled story generation and persona-conditioned dialogue. Our work focuses specifically on the challenge of fine-grained strategy control in negotiation, where strategy boundaries are inherently fuzzy and evaluator reliability is a first-order concern.

### 2.4 LLM-as-Judge Evaluation

Using LLMs to evaluate generated text has become common practice, particularly for semantic attributes that resist automated metrics. LLM-based evaluators have been applied to assess relevance, coherence, factual consistency, and dialogue quality. However, recent work has highlighted reliability concerns: evaluators may exhibit position bias, verbosity bias, and inconsistent calibration across categories. Our calibration study contributes to this growing literature by providing per-class LLM evaluator accuracy for nine negotiation strategies, demonstrating that evaluator reliability varies dramatically across classes and that model-generated outputs may exploit evaluator shortcuts.

---

## 6. Discussion

### 6.1 Why Does LoRA Degrade Prefix Control?

Our experiments narrow down the possible explanations for Prefix-LoRA interference. Gradient conflict explains part of the degradation (gradient routing recovers approximately 6 percentage points), but is not the primary mechanism. The failure of contrastive generation loss to activate (zero throughout training) indicates that standard autoregressive LM loss is largely insensitive to strategy identity—most generated tokens are strategy-agnostic, making it difficult for token-level signals to enforce strategy-level distinctions.

The frozen-prefix experiment (B9) provides the most direct clue: even when Prefix weights are identical to the high-performing B3 configuration (55.9\%), the introduction of LoRA at inference time reduces accuracy to 40.0\% across three seeds—a drop of approximately 16 percentage points with no change to the Prefix parameters themselves. This suggests that the interference is, at least in part, an *environment-level* phenomenon: Prefix embeddings learn to exert their influence in a specific attention space (the base model's Q/K/V/O projections), and when LoRA modifies these projections, the Prefix signals are partially disrupted. The effect is particularly severe for rare and semantically subtle strategies such as *other-need* and *uv-part*, whose per-class accuracy drops to near zero when LoRA is introduced (Figure~\ref{fig:perclass}).

This interpretation aligns with Kim et al.'s finding that LoRA reduces the effective rank of pre-trained representations. If LoRA compresses or rotates the attention subspace that Prefix was trained to operate in, the strategy-specific signals encoded in Prefix embeddings may no longer map effectively to output token distributions. This also explains why some strategies degrade more than others: rare and semantically subtle strategies (*other-need*, *uv-part*) may rely on more fragile representational patterns that are more easily disrupted.

### 6.2 The Gap Between Representation and Generation

A recurring theme in our diagnostic experiments is the gap between representation-space metrics and generation behavior. The auxiliary classifier achieves near-zero classification loss quickly, indicating that the Prefix-induced hidden delta contains strategy-discriminative information. Yet this internal separability does not translate into strategy-consistent generation. Similarly, orthogonal regularization drives the cosine similarity between Prefix and LoRA deltas to near-zero, achieving geometric orthogonality, but strategy control does not improve.

This gap highlights a fundamental challenge for representation-guided controllable generation: ensuring that internal representations encode the desired attribute is necessary but not sufficient. The pathway from hidden states to output token probabilities passes through multiple attention layers, and LoRA's modifications to these layers can alter how internal representations map to surface-level generation. Future methods may need to more directly constrain the *output* behavior—for instance, through strategy-aware decoding objectives or token-level strategy rewards—rather than relying solely on representation-space constraints.

### 6.3 Practical Implications

For practitioners building strategy-controlled negotiation systems, our results suggest that **Prefix-only tuning** is currently the most reliable approach when strategy fidelity is paramount, despite its higher perplexity. **LoRA with explicit strategy text** offers a reasonable alternative with better fluency but lower controllability. The hybrid approaches we tested do not, in their current form, convincingly outperform these simpler baselines.

Our evaluator calibration results carry an important methodological message: LLM-based strategy evaluation should always include calibration on human-annotated data and per-class analysis. Without calibration, high accuracy on certain strategies (e.g., model-generated *no-need* at 93\%) would appear to indicate successful control, when in reality it reflects shortcut learning that the evaluator fails to detect.

---

## 7. Conclusion

We presented a systematic diagnostic study of Prefix-LoRA hybrid PEFT for strategy-controlled negotiation dialogue generation. Through controlled comparison of twelve experimental configurations across three training seeds, we found that (1) Prefix-only tuning provides the strongest strategy controllability, (2) adding LoRA substantially improves perplexity but degrades strategy accuracy, revealing a robust control-quality trade-off, (3) orthogonal constraints, auxiliary classification, gradient routing, and contrastive losses do not reliably resolve this trade-off, and (4) LoRA disrupts Prefix control signals at the inference level rather than purely through training-time gradient conflict. We also calibrated the LLM-based strategy evaluator and identified substantial class-dependent reliability issues, including shortcut learning risks.

Our findings suggest that hybrid PEFT for controllable generation requires mechanisms that more directly align output behavior with control semantics, rather than relying on representation-space constraints alone. We release our evaluation framework, calibration data, and multi-seed results to facilitate further research on reliable strategy-controlled generation.

---

## Limitations

We acknowledge several limitations of our study.

**Single dataset and model.** Our experiments are conducted on the CaSiNo dataset with Qwen3-8B as the base model. While this allows controlled comparisons, the generalizability of our findings to other negotiation datasets (e.g., CraigslistBargain), other generation tasks, and other model architectures remains to be established.

**Evaluator reliability.** Our LLM-based strategy evaluator achieves only 52.0\% accuracy on human-written responses, and we did not conduct large-scale human evaluation to complement automatic metrics. While we report calibration results and perform per-class analysis, the absolute strategy accuracy values should be interpreted with caution. Small-scale human evaluation would strengthen confidence in our findings.

**Limited seeds and evaluation scale.** Our swap-sample evaluation uses 30 dialogue contexts (270 generations per experiment). While we report bootstrap confidence intervals and multi-seed results, larger evaluation sets would provide tighter confidence bounds and potentially reveal statistically significant differences among the moderate-performing variants (B4, B7, B9).

**Mechanism evidence.** Our claim that LoRA disrupts Prefix through environment-level attention changes is supported primarily by the frozen-prefix experiment (B9) and is consistent with prior representational analysis. However, we have not provided direct layer-wise evidence, such as comparing attention patterns or hidden-state distributions with LoRA on vs. off. Such analysis would strengthen the mechanistic interpretation.

**Hyperparameter scope.** We explored a reasonable range of hyperparameters ($\lambda_{\text{orth}} \in [0.05, 1.0]$, $\lambda_{\text{cls}} \in [0.2, 5.0]$, LoRA rank $r \in [8, 16]$), but a broader sweep or more sophisticated scheduling might yield different results. The failure of the mechanisms we tested does not prove that no variant of these mechanisms could work; it only establishes that simple, static implementations are insufficient.
