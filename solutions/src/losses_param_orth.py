"""
Parameter-Level Orthogonality Losses (Solution S2).

Constrains LoRA delta matrices to be orthogonal to prefix key/value subspaces,
preventing LoRA from interfering with prefix strategy signals in the forward pass.

Theory:
  In attention with prefix tokens:
    Q = X·(W_q + ΔW_q),  K_prefix = P·W_k^T
    The cross-term X·ΔW_q^T·W_k·P^T is where LoRA's Q delta "reads" prefix keys.
    We suppress it by constraining  ||(P·W_k^T)·ΔW_q||_F^2 → 0.

  Similarly for the output pathway:
    V_prefix = P·W_v^T,  O = attn·V·(W_o + ΔW_o)^T
    The cross-term attn·V_prefix·ΔW_o^T rewrites prefix value contributions.
    We suppress it by constraining  ||(P·W_v^T)·ΔW_o^T||_F^2 → 0.

  Gradients flow to LoRA parameters (A, B) but NOT to prefix_bank (detached),
  because the constraint targets LoRA's behaviour, not prefix embeddings.

Usage:
  from solutions.src.losses_param_orth import param_orth_losses
  orth_losses = param_orth_losses(model, cfg)
  total_loss += cfg.lambda_param_orth_qk * orth_losses['qk']
  total_loss += cfg.lambda_param_orth_vo * orth_losses['vo']
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple


def _as_dense_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return a usable 2D projection matrix from a normal or bitsandbytes weight."""
    if hasattr(weight, "dequantize"):
        try:
            weight = weight.dequantize()
            return weight.detach()
        except Exception:
            pass
    if hasattr(weight, "quant_state"):
        try:
            import bitsandbytes.functional as bnbF
            weight = bnbF.dequantize_4bit(weight.data, weight.quant_state)
            return weight.detach()
        except Exception:
            pass
    return weight.detach()


def _get_base_weight(module) -> Optional[torch.Tensor]:
    """Extract the underlying frozen base weight from PEFT-wrapped Linear/Linear4bit."""
    base_layer = getattr(module, "base_layer", module)
    weight = getattr(base_layer, "weight", None)
    if weight is None:
        weight = getattr(module, "weight", None)
    if weight is None:
        return None
    weight = _as_dense_weight(weight)
    return weight if weight.dim() == 2 else None


def _get_lora_deltas(peft_model) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA delta matrices (B @ A) from all LoRA layers.

    Returns dict mapping layer_name -> {
        'q': delta_W_q [d, d] or None,
        'k': delta_W_k [d, d] or None,
        'v': delta_W_v [d, d] or None,
        'o': delta_W_o [d, d] or None,
    }
    """
    deltas = {}
    layer_idx = 0

    for name, module in peft_model.named_modules():
        # PEFT LoRA layers have lora_A and lora_B as nn.ModuleDict with 'default' key
        if not hasattr(module, 'lora_A') or not hasattr(module, 'lora_B'):
            continue

        # Determine which projection this is
        proj_type = None
        if 'q_proj' in name:
            proj_type = 'q'
        elif 'k_proj' in name:
            proj_type = 'k'
        elif 'v_proj' in name:
            proj_type = 'v'
        elif 'o_proj' in name:
            proj_type = 'o'
        else:
            continue

        # Extract layer index from name (e.g., "layers.5.self_attn.q_proj")
        import re
        m = re.search(r'layers\.(\d+)', name)
        if m:
            layer_idx = int(m.group(1))
        else:
            layer_idx += 1  # fallback: increment

        if layer_idx not in deltas:
            deltas[layer_idx] = {'q': None, 'k': None, 'v': None, 'o': None}

        # NOTE: Do NOT detach — gradients must flow to lora_A/lora_B so the
        # orthogonality loss can actually steer LoRA updates.  The base
        # projection weights (W_k, W_v, …) are detached separately in
        # _get_base_projections since they are frozen parameters.
        lora_A = _get_lora_weight(module.lora_A)  # [r, d_in]
        lora_B = _get_lora_weight(module.lora_B)  # [d_out, r]
        if lora_A is not None and lora_B is not None:
            deltas[layer_idx][proj_type] = {
                'A': lora_A,
                'B': lora_B,
                'scaling': _get_lora_scaling(module),
            }

    return deltas


def _get_base_projections(peft_model) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Extract base (non-LoRA) Q/K/V/O weight matrices from all attention layers.

    Returns dict mapping layer_idx -> {'q': W_q [d, d], 'k': W_k [d, d], ...}
    """
    projs = {}

    for name, module in peft_model.named_modules():
        import re
        m = re.search(r'layers\.(\d+)', name)
        if not m:
            continue
        layer_idx = int(m.group(1))

        if layer_idx not in projs:
            projs[layer_idx] = {'q': None, 'k': None, 'v': None, 'o': None}

        if 'q_proj' in name and 'lora' not in name:
            projs[layer_idx]['q'] = _get_base_weight(module)
        elif 'k_proj' in name and 'lora' not in name:
            projs[layer_idx]['k'] = _get_base_weight(module)
        elif 'v_proj' in name and 'lora' not in name:
            projs[layer_idx]['v'] = _get_base_weight(module)
        elif 'o_proj' in name and 'lora' not in name:
            projs[layer_idx]['o'] = _get_base_weight(module)

    return projs


def _get_lora_weight(adapter_module):
    """
    Robustly extract LoRA A or B weight, handling different PEFT versions.
    PEFT <0.12: lora_A['default'] is nn.Linear, use .weight
    PEFT >=0.12: lora_A['default'] may be nn.Parameter (no .weight)
    """
    # Try dict access with .weight (older PEFT)
    try:
        return adapter_module['default'].weight
    except (KeyError, AttributeError):
        pass
    # Try dict access without .weight (newer PEFT, Parameter directly)
    try:
        return adapter_module['default']
    except (KeyError, TypeError, AttributeError):
        pass
    # Try attribute access with .weight
    try:
        return adapter_module.default.weight
    except (KeyError, AttributeError):
        pass
    # Try attribute access without .weight
    try:
        return adapter_module.default
    except (KeyError, AttributeError, TypeError):
        pass
    return None


def _get_lora_scaling(module) -> float:
    """Return PEFT LoRA scaling for the default adapter."""
    scaling = getattr(module, 'scaling', 1.0)
    if isinstance(scaling, dict):
        return float(scaling.get('default', 1.0))
    try:
        return float(scaling)
    except (TypeError, ValueError):
        return 1.0


def _get_hf_config(peft_model):
    """
    Traverse PEFT model structure to find the HF Transformers config
    that contains attention head information (num_key_value_heads, etc.).

    ``peft_model.config`` returns a ``PeftConfig`` (e.g. ``LoraConfig``) which
    is **not** None but does **not** contain ``num_key_value_heads``.  The real
    HF config is nested deeper in the model hierarchy:

      PeftModel.config                   → PeftConfig / LoraConfig  (no head info)
      PeftModel.base_model               → LoraModel (PEFT wrapper)
      PeftModel.base_model.config        → HF config (via LoraModel.__getattr__ delegation)
      PeftModel.base_model.model.config  → HF config (direct access)

    We try every candidate path and return the first config that actually has
    ``num_key_value_heads`` or at least ``num_attention_heads`` (from which we
    can infer the KV head count).
    """
    candidates = []

    # Path 1: peft_model.config (PeftConfig — won't have head info, but check anyway)
    candidates.append(getattr(peft_model, 'config', None))

    # Path 2 & 3: traverse base_model hierarchy
    base_model = getattr(peft_model, 'base_model', None)
    if base_model is not None:
        candidates.append(getattr(base_model, 'config', None))
        inner_model = getattr(base_model, 'model', None)
        if inner_model is not None:
            candidates.append(getattr(inner_model, 'config', None))

    # Path 4: PEFT's official unwrap method
    if hasattr(peft_model, 'get_base_model'):
        try:
            base = peft_model.get_base_model()
            if base is not None:
                candidates.append(getattr(base, 'config', None))
        except Exception:
            pass

    # Return the first config that actually has attention-head attributes
    for config in candidates:
        if config is None:
            continue
        n_kv = (getattr(config, 'num_key_value_heads', None) or
                getattr(config, 'num_kv_heads', None) or
                getattr(config, 'n_kv_heads', None))
        n_q = getattr(config, 'num_attention_heads', None)
        if n_kv or n_q:
            return config

    return None


def _maybe_expand_kv_for_gqa(
    tensor: torch.Tensor,
    d_out_kv: int,
    d_out_target: int,
    peft_model,
) -> Optional[torch.Tensor]:
    """
    Expand K/V tensor from d_out_kv to d_out_target for GQA models.

    In GQA (e.g. Qwen3-8B), K/V projections have fewer output dims than Q/O:
      d_out_q = 4096 (32 heads × 128),  d_out_k = 1024 (8 KV heads × 128)
    The forward pass repeats each KV head n_rep times to match Q heads.
    We replicate this expansion on the prefix-key/value tensor so the
    cross-term matmul with ΔW_q / ΔW_o is dimensionally consistent.

    Args:
        tensor: [S, K, d_out_kv] — prefix projected into K/V space
        d_out_kv: actual K/V output dim (e.g. 1024)
        d_out_target: desired dim (d_out_q for QK, d_in_o for VO) (e.g. 4096)
        peft_model: the model (to read config for head counts)

    Returns:
        [S, K, d_out_target] or None if expansion is not possible.
    """
    if d_out_kv == d_out_target:
        return tensor  # MHA, no expansion needed

    n_rep = d_out_target // d_out_kv
    if d_out_target % d_out_kv != 0:
        return None  # not a valid GQA ratio

    # Get the HF Transformers config (not PeftConfig) — see _get_hf_config docs
    config = _get_hf_config(peft_model)
    if config is None:
        return None

    n_kv_heads = getattr(config, 'num_key_value_heads', None)
    if n_kv_heads is None:
        n_kv_heads = getattr(config, 'num_kv_heads', None) or getattr(config, 'n_kv_heads', None)

    # Fallback: infer n_kv_heads from num_attention_heads and the GQA repeat ratio
    if not n_kv_heads or n_kv_heads == 0:
        n_q_heads = getattr(config, 'num_attention_heads', None)
        if n_q_heads and n_q_heads > 0:
            n_kv_heads = n_q_heads // n_rep

    if not n_kv_heads or n_kv_heads == 0:
        return None

    head_dim = d_out_kv // n_kv_heads
    if d_out_kv % n_kv_heads != 0:
        return None

    S, K, _ = tensor.shape
    # [S, K, d_out_kv] → [S, K, n_kv_heads, head_dim]
    # → repeat_interleave(n_rep, dim=2) → [S, K, n_q_heads, head_dim]
    # → reshape [S, K, d_out_target]
    tensor = tensor.view(S, K, n_kv_heads, head_dim)
    tensor = tensor.repeat_interleave(n_rep, dim=2)
    tensor = tensor.reshape(S, K, d_out_target)
    return tensor


def param_orth_losses(
    hybrid_model,
    cfg,
    prefix_bank: Optional[torch.Tensor] = None,
    enabled_qk: bool = True,
    enabled_vo: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Compute parameter-level orthogonality losses.

    Q-K orth (enabled_qk):
      Constrains  ||(P · W_k^T) · ΔW_q||_F^2 → 0
      where P is the prefix bank [S, K, d_in] and W_k is the base key
      projection [d_out_k, d_in].  This ensures LoRA's query delta does
      not "read" prefix key representations — only the base model's query
      can attend to prefix keys.

    V-O orth (enabled_vo):
      Constrains  ||(P · W_v^T) · ΔW_o^T||_F^2 → 0
      This ensures LoRA's output projection does not "rewrite" prefix
      value contributions at the attention output stage.

    Gradients flow to LoRA parameters (lora_A, lora_B) but NOT to
    prefix_bank (detached) — the constraint targets LoRA, not prefix.

    Args:
        hybrid_model: HybridStrategyModel instance
        cfg: TrainConfig (used for weight lookup)
        prefix_bank: [num_strategies, num_virtual_tokens, d_in] or None.
                     If None, uses hybrid_model.prefix_bank.
        enabled_qk: whether to compute Q-K orth loss
        enabled_vo: whether to compute V-O orth loss

    Returns:
        dict with 'qk' and 'vo' keys (scalar tensors with grad)
    """
    if prefix_bank is None:
        prefix_bank = hybrid_model.prefix_bank  # [S, K, d_in]

    # Detach prefix_bank: the orthogonality constraint targets LoRA params
    # only, not prefix embeddings (which are simultaneously trained).
    prefix_bank = prefix_bank.detach()  # [S, K, d_in]

    peft_model = hybrid_model.peft_model
    deltas = _get_lora_deltas(peft_model)   # grad-enabled LoRA deltas
    bases = _get_base_projections(peft_model)  # detached base weights

    qk_loss = torch.tensor(0.0, device=prefix_bank.device)
    vo_loss = torch.tensor(0.0, device=prefix_bank.device)
    n_qk = 0
    n_vo = 0

    for layer_idx in deltas:
        layer_deltas = deltas[layer_idx]
        layer_bases = bases.get(layer_idx, {})

        # --- Q-K orthogonality: ||(P · W_k^T) · ΔW_q||_F^2 → 0 ---
        # P @ W_k^T  → prefix keys  [S, K, d_out_k]
        # For GQA (Qwen3-8B): d_out_k=1024 ≠ d_out_q=4096, expand via repeat_interleave
        if enabled_qk and layer_deltas.get('q') is not None and layer_bases.get('k') is not None:
            delta_q = layer_deltas['q']
            delta_q_A = delta_q['A']          # [r, d_in]
            delta_q_B = delta_q['B']          # [d_out_q, r]
            delta_q_scaling = delta_q['scaling']
            base_k = layer_bases['k']          # [d_out_k, d_in]
            if base_k.dim() != 2 or base_k.size(1) != prefix_bank.size(-1):
                continue
            d_out_q = delta_q_B.size(0)
            d_out_k = base_k.size(0)
            prefix_keys = prefix_bank @ base_k.T  # [S, K, d_out_k]
            # Expand for GQA: [S, K, d_out_k] → [S, K, d_out_q]
            prefix_keys = _maybe_expand_kv_for_gqa(prefix_keys, d_out_k, d_out_q, peft_model)
            if prefix_keys is not None:
                # (P @ W_k^T)_expanded @ delta_q → [S, K, d_in]
                qk_mid = torch.einsum('sko,or->skr', prefix_keys, delta_q_B)
                qk_cross = torch.einsum('skr,ri->ski', qk_mid, delta_q_A)
                qk_cross = qk_cross * delta_q_scaling
                qk_loss = qk_loss + (qk_cross ** 2).mean()
                n_qk += 1

        # --- V-O orthogonality: ||(P · W_v^T) · ΔW_o^T||_F^2 → 0 ---
        # P @ W_v^T  → prefix values [S, K, d_out_v]
        # For GQA (Qwen3-8B): d_out_v=1024 ≠ d_in_o=4096, expand via repeat_interleave
        if enabled_vo and layer_deltas.get('o') is not None and layer_bases.get('v') is not None:
            delta_o = layer_deltas['o']
            delta_o_A = delta_o['A']          # [r, d_in_o]
            delta_o_B = delta_o['B']          # [d_out_o, r]
            delta_o_scaling = delta_o['scaling']
            base_v = layer_bases['v']          # [d_out_v, d_in_v]
            if base_v.dim() != 2 or base_v.size(1) != prefix_bank.size(-1):
                continue
            d_in_o = delta_o_A.size(1)
            d_out_v = base_v.size(0)
            prefix_values = prefix_bank @ base_v.T  # [S, K, d_out_v]
            # Expand for GQA: [S, K, d_out_v] → [S, K, d_in_o]
            prefix_values = _maybe_expand_kv_for_gqa(prefix_values, d_out_v, d_in_o, peft_model)
            if prefix_values is not None:
                # (P @ W_v^T)_expanded @ delta_o^T → [S, K, d_out_o]
                vo_mid = torch.einsum('sko,ro->skr', prefix_values, delta_o_A)
                vo_cross = torch.einsum('skr,ir->ski', vo_mid, delta_o_B)
                vo_cross = vo_cross * delta_o_scaling
                vo_loss = vo_loss + (vo_cross ** 2).mean()
                n_vo += 1

    # Average over layers
    if n_qk > 0:
        qk_loss = qk_loss / n_qk
    if n_vo > 0:
        vo_loss = vo_loss / n_vo

    return {'qk': qk_loss, 'vo': vo_loss}


def param_orth_losses_prefix_specific(
    hybrid_model,
    cfg,
    strategy_ids: torch.Tensor,  # [B]
) -> Dict[str, torch.Tensor]:
    """
    Deprecated. Use param_orth_losses() instead.

    The main param_orth_losses function now constrains LoRA deltas against
    ALL strategies' prefix subspaces (not just the batch's target strategy),
    which is a stronger and more uniform constraint.  This function is kept
    only for backward compatibility and simply delegates.
    """
    return param_orth_losses(
        hybrid_model, cfg,
        enabled_qk=True,
        enabled_vo=True,
    )
