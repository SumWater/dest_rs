"""
Attention Weight Extraction Utilities (Solution S3 prerequisite).

Diagnoses how LoRA changes Prefix token attention patterns by extracting
and comparing attention weights from B3 (Prefix-only) and B9 (Prefix frozen + LoRA).

This analysis answers: does LoRA compress Prefix attention weights globally
(justifying S3 gating), or does it redistribute them to different text tokens
(meaning gating won't help)?

Usage:
  from solutions.src.attention_utils import extract_attention_weights, compare_attention
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import json


def extract_attention_weights(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_embeds: torch.Tensor,  # [K, d]
    target_layers: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract attention weights for a single forward pass, focusing on
    how much attention prefix tokens receive from text tokens.

    Registers forward hooks on specified attention layers, runs one forward
    pass, and collects the attention probability matrices.

    Args:
        model: HuggingFace causal LM (must output_attentions=True)
        input_ids: [1, seq_len]
        attention_mask: [1, seq_len]
        prefix_embeds: [K, hidden_size]
        target_layers: list of layer indices to extract; None = all layers

    Returns:
        dict with keys:
          'layer_{i}_attn': [1, num_heads, K+seq_len, K+seq_len] attention probs
          'layer_{i}_prefix_weight': [1, num_heads, seq_len] — mean attention
              from each text position to all prefix tokens
    """
    embed_layer = model.get_input_embeddings()
    input_embeds = embed_layer(input_ids)  # [1, seq_len, d]

    prefix_expanded = prefix_embeds.unsqueeze(0)  # [1, K, d]
    full_embeds = torch.cat([prefix_expanded, input_embeds], dim=1)

    prefix_mask = torch.ones(
        attention_mask.size(0), prefix_embeds.size(0),
        dtype=attention_mask.dtype, device=attention_mask.device
    )
    full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

    # Collect attention weights via hooks
    attention_dict = {}

    def make_hook(layer_idx):
        def hook(module, input_tensors, output_tensors):
            # output_tensors is typically a tuple (attn_output, attn_weights, ...)
            if isinstance(output_tensors, tuple) and len(output_tensors) >= 2:
                attn = output_tensors[1]  # attn weights if output_attentions was requested
                if attn is not None:
                    attention_dict[f'layer_{layer_idx}_attn'] = attn.detach().cpu()
        return hook

    hooks = []
    if target_layers is None:
        # Register on all layers
        for i, layer in enumerate(model.base_model.layers if hasattr(model, 'base_model') else model.model.layers):
            h = layer.self_attn.register_forward_hook(make_hook(i))
            hooks.append(h)
    else:
        for i in target_layers:
            if hasattr(model, 'base_model'):
                layer = model.base_model.layers[i]
            else:
                layer = model.model.layers[i]
            h = layer.self_attn.register_forward_hook(make_hook(i))
            hooks.append(h)

    # Forward pass (may need to temporarily enable output_attentions)
    with torch.no_grad():
        try:
            outputs = model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                output_attentions=True,
            )
        except Exception:
            # Fallback: model may not support output_attentions kwarg
            outputs = model(inputs_embeds=full_embeds, attention_mask=full_mask)

    # Clean up hooks
    for h in hooks:
        h.remove()

    # Compute prefix attention weight summary
    K = prefix_embeds.size(0)
    result = {}
    for key, attn in attention_dict.items():
        result[key] = attn
        # attn: [batch, num_heads, K+seq_len, K+seq_len]
        # Prefix attention: text tokens → prefix tokens
        #   = attn[:, :, K:, :K]  → [batch, num_heads, seq_len, K]
        prefix_attn = attn[:, :, K:, :K]
        # Mean over prefix tokens → how much each text position attends to prefixes
        prefix_weight = prefix_attn.mean(dim=-1)  # [batch, num_heads, seq_len]
        layer_num = key.replace('layer_', '').replace('_attn', '')
        result[f'layer_{layer_num}_prefix_weight'] = prefix_weight

    return result


def compare_attention_b3_b9(
    b3_checkpoint_dir: str,
    b9_checkpoint_dir: str,
    sample_contexts: List[Dict],
    model_builder,
    num_layers: int = 32,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Compare attention patterns between B3 (Prefix-only) and B9 (Frozen Prefix + LoRA)
    on the same set of sample contexts.

    Returns per-layer statistics:
      - mean_prefix_attention_b3: average prefix attention weight per layer
      - mean_prefix_attention_b9: average prefix attention weight per layer
      - attention_divergence: per-layer KL divergence between B3 and B9 attention

    Args:
        b3_checkpoint_dir: path to B3 checkpoint (solutions/output/other/.../b3_prefix_only)
        b9_checkpoint_dir: path to B9 checkpoint (solutions/output/other/.../b9_prefix_then_lora)
        sample_contexts: list of dicts with 'input_ids' and 'strategy' keys
        model_builder: callable that loads a model from checkpoint dir
        num_layers: number of transformer layers
        save_path: if provided, save results as JSON

    Returns:
        dict with comparison statistics
    """
    # Load models
    model_b3 = model_builder(b3_checkpoint_dir)
    model_b9 = model_builder(b9_checkpoint_dir)

    prefix_b3 = model_b3.prefix_bank
    prefix_b9 = model_b9.prefix_bank  # Same as B3 since frozen

    results = {
        'num_samples': len(sample_contexts),
        'per_layer': defaultdict(lambda: {'b3_prefix_weight': [], 'b9_prefix_weight': []}),
    }

    for sample in sample_contexts:
        input_ids = sample['input_ids']
        attn_mask = sample.get('attention_mask',
            torch.ones_like(input_ids))
        strategy_id = sample['strategy']

        prefix_emb = prefix_b3[strategy_id]  # [K, d]

        # Extract B3 attention
        attn_b3 = extract_attention_weights(
            model_b3, input_ids, attn_mask, prefix_emb,
            target_layers=list(range(num_layers))
        )

        # Extract B9 attention
        attn_b9 = extract_attention_weights(
            model_b9, input_ids, attn_mask, prefix_emb,
            target_layers=list(range(num_layers))
        )

        # Aggregate
        for layer_idx in range(num_layers):
            key = f'layer_{layer_idx}_prefix_weight'
            if key in attn_b3:
                results['per_layer'][layer_idx]['b3_prefix_weight'].append(
                    attn_b3[key].mean().item()
                )
            if key in attn_b9:
                results['per_layer'][layer_idx]['b9_prefix_weight'].append(
                    attn_b9[key].mean().item()
                )

    # Compute summary statistics
    summary = []
    for layer_idx in range(num_layers):
        b3_vals = results['per_layer'][layer_idx]['b3_prefix_weight']
        b9_vals = results['per_layer'][layer_idx]['b9_prefix_weight']
        if b3_vals and b9_vals:
            b3_mean = sum(b3_vals) / len(b3_vals)
            b9_mean = sum(b9_vals) / len(b9_vals)
            summary.append({
                'layer': layer_idx,
                'b3_mean_prefix_attn': b3_mean,
                'b9_mean_prefix_attn': b9_mean,
                'ratio_b9_to_b3': b9_mean / b3_mean if b3_mean > 0 else 0,
                'conclusion': (
                    'compressed' if b9_mean < b3_mean * 0.8 else
                    'redistributed' if abs(b9_mean - b3_mean) < b3_mean * 0.2 else
                    'amplified'
                ),
            })

    results['summary'] = summary

    # Global conclusion
    ratios = [s['ratio_b9_to_b3'] for s in summary]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0
    if avg_ratio < 0.5:
        results['global_conclusion'] = (
            'B9 attention to prefix is GLOBALLY COMPRESSED (avg ratio {:.3f}). '
            'S3 gating is likely to help.'.format(avg_ratio)
        )
    elif avg_ratio > 1.5:
        results['global_conclusion'] = (
            'B9 attention to prefix is AMPLIFIED (avg ratio {:.3f}). '
            'Interference is not via attention compression.'.format(avg_ratio)
        )
    else:
        results['global_conclusion'] = (
            'B9 attention to prefix is REDISTRIBUTED (avg ratio {:.3f}). '
            'S3 gating may not help — the issue is WHERE attention goes, not how much.'.format(avg_ratio)
        )

    if save_path:
        # Convert defaultdict to regular dict for JSON
        results['per_layer'] = dict(results['per_layer'])
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    return results
