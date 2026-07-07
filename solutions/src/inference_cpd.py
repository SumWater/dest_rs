"""
Contrastive Prefix Decoding (Solution S4).

At inference time, amplifies strategy-specific signal by subtracting
a "neutral prefix" logit distribution from the "target prefix" distribution.

Idea:  P_final = softmax(logit_target + α · (logit_target - logit_neutral))

The neutral prefix removes the "fluency baseline" shared by all strategies,
leaving only the strategy-specific component.

Key advantage: zero training cost — works with any trained checkpoint.

Usage:
  from solutions.src.inference_cpd import generate_with_cpd
  outputs = generate_with_cpd(model, tokenizer, input_ids, target_strategy_id, ...)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, List


def _build_neutral_prefix(prefix_bank: torch.Tensor) -> torch.Tensor:
    """
    Build a neutral/dummy prefix by averaging all strategy prefixes,
    or by using zero embeddings.

    Args:
        prefix_bank: [num_strategies, num_virtual_tokens, hidden_size]

    Returns:
        neutral_prefix: [num_virtual_tokens, hidden_size]
    """
    # Option A: mean of all strategies (smooth, may still carry strategy signal)
    # return prefix_bank.mean(dim=0)

    # Option B: zero embedding (truly neutral, may cause distribution shift)
    # return torch.zeros_like(prefix_bank[0])

    # Option C: mean + slight random perturbation for diversity
    neutral = prefix_bank.mean(dim=0)
    return neutral


def _forward_with_prefix(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_embeds: torch.Tensor,  # [K, d]
) -> torch.Tensor:
    """
    Forward pass with a specific prefix, returning logits.

    Detects whether *model* is a HybridStrategyModel (which has its own
    prefix-prepending logic via prefix_override) or a raw HuggingFace model
    (where we manually prepend prefix embeddings).

    Args:
        model: HybridStrategyModel or HuggingFace causal LM
        input_ids: [1, seq_len]
        attention_mask: [1, seq_len]
        prefix_embeds: [K, hidden_size]

    Returns:
        logits: [1, seq_len, vocab_size]  (prefix positions stripped)
    """
    K = prefix_embeds.size(0)

    # ── HybridStrategyModel path ──
    # Its forward() signature is:
    #   forward(input_ids, attention_mask, strategy_ids, …, prefix_override=None, …)
    # and it internally prepends prefix embeddings + extends attention mask.
    # We pass prefix_override so it uses *our* prefix instead of prefix_bank.
    if hasattr(model, 'prefix_bank') and hasattr(model, 'peft_model'):
        batch_size = input_ids.size(0)
        # strategy_ids is required but irrelevant when prefix_override is set
        dummy_strategy_ids = torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        outputs, _, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            strategy_ids=dummy_strategy_ids,
            prefix_on=True,
            prefix_override=prefix_embeds,  # [K, d] → model handles unsqueeze/expand
            use_cache=False,
        )
        logits = outputs.logits  # [batch, K+seq_len, vocab_size]
        return logits[:, K:, :]  # strip prefix positions → [batch, seq_len, vocab]

    # ── Raw HuggingFace model path (fallback) ──
    embed_layer = model.get_input_embeddings()
    input_embeds = embed_layer(input_ids)  # [1, seq_len, d]

    prefix_expanded = prefix_embeds.unsqueeze(0)  # [1, K, d]
    full_embeds = torch.cat([prefix_expanded, input_embeds], dim=1)  # [1, K+seq_len, d]

    prefix_mask = torch.ones(
        attention_mask.size(0), K,
        dtype=attention_mask.dtype, device=attention_mask.device
    )
    full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    outputs = model(inputs_embeds=full_embeds, attention_mask=full_mask)
    logits = outputs.logits  # [1, K+seq_len, vocab_size]
    return logits[:, K:, :]  # [1, seq_len, vocab_size]


def generate_with_cpd(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    prefix_bank: torch.Tensor,
    target_strategy_id: int,
    alpha: float = 1.0,
    max_new_tokens: int = 40,
    temperature: float = 0.0,
    neutral_mode: str = "mean",
) -> List[int]:
    """
    Generate text using Contrastive Prefix Decoding.

    At each step:
      1. Compute logits with target strategy prefix
      2. Compute logits with neutral prefix (mean of all strategies)
      3. Final logits = logit_target + alpha * (logit_target - logit_neutral)

    Args:
        model: HuggingFace causal LM
        tokenizer: tokenizer
        input_ids: [1, context_len] tokenized input
        prefix_bank: [num_strategies, num_virtual_tokens, d]
        target_strategy_id: integer strategy index
        alpha: contrastive amplification factor (higher = stronger strategy control)
        max_new_tokens: maximum tokens to generate
        temperature: sampling temperature (1.0 = greedy-like, >1.0 = more random)
        neutral_mode: "mean" or "zero" for neutral prefix construction

    Returns:
        list of generated token IDs
    """
    if neutral_mode == "zero":
        neutral_prefix = torch.zeros_like(prefix_bank[0])
    else:
        neutral_prefix = prefix_bank.mean(dim=0)

    target_prefix = prefix_bank[target_strategy_id]  # [K, d]

    device = input_ids.device
    generated = []
    current_ids = input_ids

    for step in range(max_new_tokens):
        # Build attention mask
        attn_mask = torch.ones_like(current_ids)

        # Forward with target prefix
        logits_target = _forward_with_prefix(
            model, current_ids, attn_mask, target_prefix
        )  # [1, seq_len, vocab_size]
        last_logit_target = logits_target[0, -1, :]  # [vocab_size]

        # Forward with neutral prefix
        logits_neutral = _forward_with_prefix(
            model, current_ids, attn_mask, neutral_prefix
        )
        last_logit_neutral = logits_neutral[0, -1, :]  # [vocab_size]

        # Contrastive combination
        logit_diff = last_logit_target - last_logit_neutral
        logit_final = last_logit_target + alpha * logit_diff

        # Apply temperature
        if temperature != 1.0:
            logit_final = logit_final / temperature

        # Sample or greedy
        if temperature == 0 or temperature < 0.01:
            next_token = logit_final.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logit_final, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        next_token_id = next_token.item()

        # Stop at EOS
        if next_token_id == tokenizer.eos_token_id:
            break

        generated.append(next_token_id)
        current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)

    return generated


def compute_cpd_logit_overlap(
    model,
    input_ids: torch.Tensor,
    prefix_bank: torch.Tensor,
    target_strategy_id: int,
    wrong_strategy_id: int,
    top_k: int = 10,
) -> float:
    """
    Diagnostic: what fraction of top-k tokens overlap between correct
    and wrong strategy prefix logits?

    Low overlap → CPD has room to work.
    High overlap → CPD will produce noise.

    Args:
        model: HuggingFace causal LM
        input_ids: [1, seq_len]
        prefix_bank: [num_strategies, K, d]
        target_strategy_id: correct strategy
        wrong_strategy_id: incorrect strategy
        top_k: number of top tokens to compare

    Returns:
        overlap fraction (0.0 to 1.0)
    """
    attn_mask = torch.ones_like(input_ids)

    logits_target = _forward_with_prefix(
        model, input_ids, attn_mask, prefix_bank[target_strategy_id]
    )
    logits_wrong = _forward_with_prefix(
        model, input_ids, attn_mask, prefix_bank[wrong_strategy_id]
    )

    last_target = logits_target[0, -1, :]   # [vocab_size]
    last_wrong = logits_wrong[0, -1, :]

    top_target = set(last_target.topk(top_k).indices.tolist())
    top_wrong = set(last_wrong.topk(top_k).indices.tolist())

    overlap = len(top_target & top_wrong) / top_k
    return overlap
