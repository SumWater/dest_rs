from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .modeling import HybridStrategyModel, LayerCatcher, get_embed_device, lora_disabled_ctx

# S2: parameter-level orthogonality loss (solutions/src/losses_param_orth.py)
try:
    from solutions.src.losses_param_orth import param_orth_losses
    _PARAM_ORTH_AVAILABLE = True
except ImportError:
    param_orth_losses = None
    _PARAM_ORTH_AVAILABLE = False


def mode_uses_prefix(adapter_mode: str) -> bool:
    return adapter_mode in {"prefix_only", "prefix_lora", "prefix_lora_orth", "dest_rs"}


def mode_uses_lora(adapter_mode: str) -> bool:
    return adapter_mode in {"lora_only", "prefix_lora", "prefix_lora_orth", "dest_rs"}


def cfg_uses_prefix(cfg: TrainConfig) -> bool:
    if cfg.enable_prefix is not None:
        return bool(cfg.enable_prefix)
    return mode_uses_prefix(cfg.adapter_mode)


def cfg_uses_lora(cfg: TrainConfig) -> bool:
    if cfg.enable_lora is not None:
        return bool(cfg.enable_lora)
    return mode_uses_lora(cfg.adapter_mode)


def cfg_trains_classifier(cfg: TrainConfig) -> bool:
    if cfg.train_classifier is not None:
        return bool(cfg.train_classifier)
    return cfg.adapter_mode == "dest_rs"


def cfg_cls_target(cfg: TrainConfig) -> str:
    target = (cfg.cls_target or "h_real").strip().lower()
    if target not in {"h_real", "delta_prefix"}:
        raise ValueError(f"不支持的 cls_target={cfg.cls_target!r}，可选值为 h_real 或 delta_prefix")
    return target


def align_labels_and_response_mask(
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    num_virtual_tokens: int,
    logits_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 2 or response_mask.ndim != 2 or labels.shape != response_mask.shape:
        raise ValueError(f"labels/response_mask shape mismatch: {labels.shape} vs {response_mask.shape}")
    if num_virtual_tokens < 0:
        raise ValueError(f"num_virtual_tokens must be non-negative, got {num_virtual_tokens}")
    response_mask = response_mask.to(torch.bool)
    if num_virtual_tokens:
        labels = torch.cat([
            torch.full((labels.size(0), num_virtual_tokens), -100, dtype=labels.dtype, device=labels.device),
            labels,
        ], dim=1)
        response_mask = torch.cat([
            torch.zeros((response_mask.size(0), num_virtual_tokens), dtype=torch.bool, device=response_mask.device),
            response_mask,
        ], dim=1)
    if labels.size(1) != logits_sequence_length or response_mask.size(1) != logits_sequence_length:
        raise ValueError(
            f"Aligned length mismatch: labels={labels.shape}, mask={response_mask.shape}, "
            f"logits_sequence_length={logits_sequence_length}"
        )
    if torch.any(response_mask & labels.eq(-100)):
        raise ValueError("response_mask selects ignored labels")
    if torch.any(labels.ne(-100) & ~response_mask):
        raise ValueError("supervised labels exist outside response_mask")
    return labels, response_mask


def response_nll_stats(
    logits: torch.Tensor, aligned_labels: torch.Tensor, aligned_response_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 3 or aligned_labels.shape != aligned_response_mask.shape:
        raise ValueError("Invalid logits/labels/mask ranks or shapes")
    if aligned_labels.size(1) != logits.size(1):
        raise ValueError("Aligned labels and logits lengths differ")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = aligned_labels[:, 1:].contiguous()
    shift_mask = aligned_response_mask[:, 1:].to(torch.bool).contiguous()
    if shift_labels.shape != shift_mask.shape or shift_labels.size(1) != shift_logits.size(1):
        raise ValueError("Causal-shift alignment mismatch")
    counts = shift_mask.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError(f"response token count is zero after causal shift: {counts.tolist()}")
    safe_labels = shift_labels.masked_fill(~shift_mask, 0)
    selected = F.log_softmax(shift_logits.float(), dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    nll_sums = -(selected * shift_mask).sum(dim=1)
    if not torch.isfinite(nll_sums).all():
        raise FloatingPointError("NaN or Inf in response NLL")
    return nll_sums, counts


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor | None = None,
    *,
    num_virtual_tokens: int = 0,
) -> torch.Tensor:
    if response_mask is None:
        response_mask = labels.ne(-100)
    aligned_labels, aligned_mask = align_labels_and_response_mask(
        labels, response_mask,
        num_virtual_tokens=num_virtual_tokens,
        logits_sequence_length=logits.size(1),
    )
    nll_sums, counts = response_nll_stats(logits, aligned_labels, aligned_mask)
    return nll_sums.sum() / counts.sum()


def sample_wrong_strategies(strategy_ids: torch.Tensor, num_strategies: int) -> torch.Tensor:
    """每个样本随机选一个不同于当前策略的 ID。"""
    if num_strategies <= 1:
        return strategy_ids
    wrong = torch.randint(0, num_strategies - 1, strategy_ids.shape, device=strategy_ids.device)
    wrong = torch.where(wrong >= strategy_ids, wrong + 1, wrong)
    return wrong


def compute_contrastive_loss(
    hybrid: HybridStrategyModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy_ids: torch.Tensor,
    labels: torch.Tensor,
    extended_labels: torch.Tensor,
    gen_loss_right: torch.Tensor,
    cfg: TrainConfig,
    num_strategies: int,
) -> torch.Tensor:
    """对比生成损失：正确策略的 gen_loss 应低于错误策略。"""
    wrong_ids = sample_wrong_strategies(strategy_ids, num_strategies)
    with torch.no_grad():
        outputs_wrong, ext_labels_wrong, _ = hybrid(
            input_ids=input_ids,
            attention_mask=attention_mask,
            strategy_ids=wrong_ids,
            labels=labels,
            prefix_on=True,
            prefix_scale=cfg.prefix_scale_train,
            use_cache=False,
        )
        gen_loss_wrong = causal_lm_loss(
            outputs_wrong.logits, ext_labels_wrong.to(outputs_wrong.logits.device)
        )
    return F.relu(gen_loss_right - gen_loss_wrong + cfg.contrastive_margin)


def pool_target_tokens(hidden_states: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    mask = target_mask.to(hidden_states.device).unsqueeze(-1).to(hidden_states.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (hidden_states * mask).sum(dim=1) / denom


def classification_loss(
    hybrid: HybridStrategyModel,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    strategy_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hybrid.strategy_classifier.to(hidden_states.device)
    pooled = pool_target_tokens(hidden_states, labels.ne(-100))
    logits = hybrid.strategy_classifier(pooled)
    loss = F.cross_entropy(logits, strategy_ids.to(logits.device))
    return loss, logits


def local_global_orth_loss(
    delta_prefix: torch.Tensor,
    delta_lora: torch.Tensor,
    labels: torch.Tensor,
    cfg: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target_mask = labels.ne(-100).to(delta_prefix.device)
    if target_mask.sum().item() == 0:
        zero = delta_prefix.new_zeros(())
        return zero, zero, zero

    indices = torch.nonzero(target_mask, as_tuple=False)
    if indices.size(0) > cfg.orth_token_sample_size:
        permutation = torch.randperm(indices.size(0), device=indices.device)
        indices = indices[permutation[: cfg.orth_token_sample_size]]

    prefix_vectors = delta_prefix[indices[:, 0], indices[:, 1], :]
    lora_vectors = delta_lora[indices[:, 0], indices[:, 1], :]
    cosine = F.cosine_similarity(prefix_vectors, lora_vectors, dim=-1)
    local_loss = (cosine ** 2).mean()

    prefix_pooled = pool_target_tokens(delta_prefix, target_mask)
    lora_pooled = pool_target_tokens(delta_lora, target_mask)
    cosine_global = F.cosine_similarity(prefix_pooled, lora_pooled, dim=-1)
    global_loss = (cosine_global ** 2).mean()

    alpha = float(cfg.orth_alpha)
    orth_loss = alpha * local_loss + (1.0 - alpha) * global_loss
    return orth_loss, local_loss, global_loss


def compute_delta_prefix_hidden(
    hybrid: HybridStrategyModel,
    catcher: LayerCatcher,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy_ids: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    catcher.reset()
    with torch.no_grad():
        with lora_disabled_ctx(hybrid.peft_model):
            _ = hybrid(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_ids=strategy_ids,
                labels=None,
                prefix_on=False,
                prefix_scale=cfg.prefix_scale_train,
                use_cache=False,
            )
            h_base = catcher.pop().detach()

    catcher.reset()
    with torch.no_grad():
        with lora_disabled_ctx(hybrid.peft_model):
            _ = hybrid(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_ids=strategy_ids,
                labels=None,
                prefix_on=True,
                prefix_scale=cfg.prefix_scale_train,
                use_cache=False,
            )
            h_prefix = catcher.pop().detach()
    h_prefix = hybrid.slice_real_tokens(h_prefix, prefix_on=True)

    return h_prefix - h_base


def compute_delta_prefix_for_grad_routing(
    hybrid: HybridStrategyModel,
    catcher: LayerCatcher,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    strategy_ids: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    """Prefix-only 前向 + base-only (no_grad) → delta_prefix。

    与 compute_delta_prefix_hidden 的区别：prefix 前向**不包 no_grad**，
    使 cls_loss 梯度可以通过 delta_prefix 到达 prefix_bank。
    """
    catcher.reset()
    with torch.no_grad():
        with lora_disabled_ctx(hybrid.peft_model):
            _ = hybrid(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_ids=strategy_ids,
                labels=None,
                prefix_on=False,
                prefix_scale=cfg.prefix_scale_train,
                use_cache=False,
            )
            h_base = catcher.pop().detach()

    catcher.reset()
    # 注意：这里没有 torch.no_grad()，梯度会流向 prefix_bank
    with lora_disabled_ctx(hybrid.peft_model):
        _ = hybrid(
            input_ids=input_ids,
            attention_mask=attention_mask,
            strategy_ids=strategy_ids,
            labels=None,
            prefix_on=True,
            prefix_scale=cfg.prefix_scale_train,
            use_cache=False,
        )
        h_prefix = catcher.pop()
    h_prefix = hybrid.slice_real_tokens(h_prefix, prefix_on=True)

    return h_prefix - h_base


def compute_training_losses(
    hybrid: HybridStrategyModel,
    catcher: LayerCatcher,
    batch: dict,
    cfg: TrainConfig,
    global_step: int,
):
    device_in = get_embed_device(hybrid.peft_model)
    input_ids = batch["input_ids"].to(device_in)
    attention_mask = batch["attention_mask"].to(device_in)
    labels = batch["labels"].to(device_in)
    response_mask = batch["response_mask"].to(device_in)
    strategy_ids = batch["strategy_id"].to(device_in)

    prefix_on = cfg_uses_prefix(cfg)
    lora_on = cfg_uses_lora(cfg)
    should_compute_cls = cfg.lambda_cls > 0.0 and cfg_trains_classifier(cfg)
    cls_target = cfg_cls_target(cfg)
    should_compute_orth = (
        cfg.lambda_orth > 0.0
        and prefix_on
        and lora_on
        and global_step >= cfg.orth_start_step
        and (global_step % cfg.orth_every_n_steps == 0)
    )
    grad_routing = bool(cfg.grad_routing)

    if not should_compute_orth:
        catcher.reset()
        if lora_on:
            outputs, extended_labels, _ = hybrid(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_ids=strategy_ids,
                labels=labels,
                prefix_on=prefix_on,
                prefix_scale=cfg.prefix_scale_train,
                detach_prefix=grad_routing and prefix_on,
                use_cache=False,
            )
        else:
            with lora_disabled_ctx(hybrid.peft_model):
                outputs, extended_labels, _ = hybrid(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    strategy_ids=strategy_ids,
                    labels=labels,
                    prefix_on=prefix_on,
                    prefix_scale=cfg.prefix_scale_train,
                    detach_prefix=grad_routing and prefix_on,
                    use_cache=False,
                )
        h_both = catcher.pop()
        h_real = hybrid.slice_real_tokens(h_both, prefix_on=prefix_on)
        gen_loss = causal_lm_loss(
            outputs.logits,
            labels.to(outputs.logits.device),
            response_mask.to(outputs.logits.device),
            num_virtual_tokens=hybrid.num_virtual_tokens if prefix_on else 0,
        )
        cls_loss = torch.zeros((), device=gen_loss.device)
        cls_logits = None
        if should_compute_cls:
            cls_hidden = h_real
            if cls_target == "delta_prefix" and prefix_on:
                if grad_routing:
                    cls_hidden = compute_delta_prefix_for_grad_routing(
                        hybrid=hybrid,
                        catcher=catcher,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        strategy_ids=strategy_ids,
                        cfg=cfg,
                    )
                else:
                    cls_hidden = compute_delta_prefix_hidden(
                        hybrid=hybrid,
                        catcher=catcher,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        strategy_ids=strategy_ids,
                        cfg=cfg,
                    )
            cls_loss, cls_logits = classification_loss(hybrid, cls_hidden, labels, strategy_ids)
            cls_loss = cls_loss.to(gen_loss.device)
        zero = torch.zeros((), device=gen_loss.device)
        contrastive_loss = zero.clone()
        if cfg.contrastive_loss:
            contrastive_loss = compute_contrastive_loss(
                hybrid=hybrid,
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_ids=strategy_ids,
                labels=labels,
                extended_labels=extended_labels,
                gen_loss_right=gen_loss,
                cfg=cfg,
                num_strategies=hybrid.num_strategies,
            )
        total_loss = gen_loss + cfg.lambda_cls * cls_loss + cfg.lambda_contrastive * contrastive_loss

        # S2: parameter-level orthogonality (batch-independent, computed on all strategies)
        param_orth_qk_loss = zero.clone()
        param_orth_vo_loss = zero.clone()
        if _PARAM_ORTH_AVAILABLE and (cfg.lambda_param_orth_qk > 0 or cfg.lambda_param_orth_vo > 0):
            if global_step % cfg.param_orth_every_n_steps == 0:
                po_losses = param_orth_losses(
                    hybrid, cfg,
                    enabled_qk=cfg.lambda_param_orth_qk > 0,
                    enabled_vo=cfg.lambda_param_orth_vo > 0,
                )
                param_orth_qk_loss = po_losses['qk'] * cfg.lambda_param_orth_qk
                param_orth_vo_loss = po_losses['vo'] * cfg.lambda_param_orth_vo
                total_loss = total_loss + param_orth_qk_loss + param_orth_vo_loss

        return {
            "gen_loss": gen_loss,
            "orth_loss": zero,
            "orth_local_loss": zero,
            "orth_global_loss": zero,
            "param_orth_qk": param_orth_qk_loss,
            "param_orth_vo": param_orth_vo_loss,
            "cls_loss": cls_loss,
            "contrastive_loss": contrastive_loss,
            "total_loss": total_loss,
            "used_orth": False,
            "cls_logits": cls_logits,
        }

    was_training = hybrid.training
    hybrid.eval()

    # 第 1 次前向：base-only，关闭 Prefix 和 LoRA。
    catcher.reset()
    with torch.no_grad():
        with lora_disabled_ctx(hybrid.peft_model):
            _ = hybrid(
                input_ids=input_ids,
                attention_mask=attention_mask,
                strategy_ids=strategy_ids,
                labels=None,
                prefix_on=False,
                prefix_scale=cfg.prefix_scale_train,
                use_cache=False,
            )
            h_base = catcher.pop().detach()

    # 第 2 次前向：prefix-only，开启 Prefix、关闭 LoRA。
    # 无 torch.no_grad()：orth_loss 和 cls_loss 梯度可流向 prefix_bank。
    catcher.reset()
    with lora_disabled_ctx(hybrid.peft_model):
        _ = hybrid(
            input_ids=input_ids,
            attention_mask=attention_mask,
            strategy_ids=strategy_ids,
            labels=None,
            prefix_on=True,
            prefix_scale=cfg.prefix_scale_train,
            use_cache=False,
        )
        h_prefix = catcher.pop()
        h_prefix = hybrid.slice_real_tokens(h_prefix, prefix_on=True)

    # 第 3 次前向：LoRA-only，关闭 Prefix、开启 LoRA。
    # 无 torch.no_grad()：orth_loss 梯度可流向 LoRA。
    catcher.reset()
    _ = hybrid(
        input_ids=input_ids,
        attention_mask=attention_mask,
        strategy_ids=strategy_ids,
        labels=None,
        prefix_on=False,
        prefix_scale=cfg.prefix_scale_train,
        use_cache=False,
    )
    h_lora = catcher.pop()
    h_lora = hybrid.slice_real_tokens(h_lora, prefix_on=False)

    if was_training:
        hybrid.train()

    # 第 4 次前向：both-on，用于生成损失。
    # 梯度路由模式下 detach prefix，使 gen_loss 梯度只流向 LoRA。
    catcher.reset()
    outputs, extended_labels, _ = hybrid(
        input_ids=input_ids,
        attention_mask=attention_mask,
        strategy_ids=strategy_ids,
        labels=labels,
        prefix_on=True,
        prefix_scale=cfg.prefix_scale_train,
        detach_prefix=grad_routing,
        use_cache=False,
    )
    h_both = catcher.pop()
    logits = outputs.logits
    extended_labels = extended_labels.to(logits.device)
    gen_loss = causal_lm_loss(
        logits,
        labels.to(logits.device),
        response_mask.to(logits.device),
        num_virtual_tokens=hybrid.num_virtual_tokens,
    )
    h_both = hybrid.slice_real_tokens(h_both, prefix_on=True)

    delta_prefix = h_prefix - h_base
    delta_lora = h_lora - h_base

    orth_loss, local_loss, global_loss = local_global_orth_loss(delta_prefix, delta_lora, labels, cfg)
    orth_loss = orth_loss.to(gen_loss.device)
    local_loss = local_loss.to(gen_loss.device)
    global_loss = global_loss.to(gen_loss.device)

    cls_loss = torch.zeros((), device=gen_loss.device)
    cls_logits = None
    if should_compute_cls:
        cls_hidden = delta_prefix if cls_target == "delta_prefix" else h_both
        cls_loss, cls_logits = classification_loss(hybrid, cls_hidden, labels, strategy_ids)
        cls_loss = cls_loss.to(gen_loss.device)

    contrastive_loss = torch.zeros((), device=gen_loss.device)
    if cfg.contrastive_loss:
        contrastive_loss = compute_contrastive_loss(
            hybrid=hybrid,
            input_ids=input_ids,
            attention_mask=attention_mask,
            strategy_ids=strategy_ids,
            labels=labels,
            extended_labels=extended_labels,
            gen_loss_right=gen_loss,
            cfg=cfg,
            num_strategies=hybrid.num_strategies,
        )

    total_loss = gen_loss + cfg.lambda_orth * orth_loss + cfg.lambda_cls * cls_loss + cfg.lambda_contrastive * contrastive_loss

    # S2: parameter-level orthogonality (batch-independent, computed on all strategies)
    param_orth_qk_loss = torch.zeros((), device=gen_loss.device)
    param_orth_vo_loss = torch.zeros((), device=gen_loss.device)
    if _PARAM_ORTH_AVAILABLE and (cfg.lambda_param_orth_qk > 0 or cfg.lambda_param_orth_vo > 0):
        if global_step % cfg.param_orth_every_n_steps == 0:
            po_losses = param_orth_losses(
                hybrid, cfg,
                enabled_qk=cfg.lambda_param_orth_qk > 0,
                enabled_vo=cfg.lambda_param_orth_vo > 0,
            )
            param_orth_qk_loss = po_losses['qk'] * cfg.lambda_param_orth_qk
            param_orth_vo_loss = po_losses['vo'] * cfg.lambda_param_orth_vo
            total_loss = total_loss + param_orth_qk_loss + param_orth_vo_loss

    del h_base, h_prefix, h_lora, delta_prefix, delta_lora
    return {
        "gen_loss": gen_loss,
        "orth_loss": orth_loss,
        "orth_local_loss": local_loss,
        "orth_global_loss": global_loss,
        "param_orth_qk": param_orth_qk_loss,
        "param_orth_vo": param_orth_vo_loss,
        "cls_loss": cls_loss,
        "contrastive_loss": contrastive_loss,
        "total_loss": total_loss,
        "used_orth": True,
        "cls_logits": cls_logits,
    }
