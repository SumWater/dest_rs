from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .modeling import HybridStrategyModel, LayerCatcher, get_embed_device, lora_disabled_ctx


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


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    if valid.sum().item() == 0:
        return (shift_logits * 0.0).sum()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


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
        gen_loss = causal_lm_loss(outputs.logits, extended_labels.to(outputs.logits.device))
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
        total_loss = gen_loss + cfg.lambda_cls * cls_loss
        return {
            "gen_loss": gen_loss,
            "orth_loss": zero,
            "orth_local_loss": zero,
            "orth_global_loss": zero,
            "cls_loss": cls_loss,
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
    gen_loss = causal_lm_loss(logits, extended_labels)
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

    total_loss = gen_loss + cfg.lambda_orth * orth_loss + cfg.lambda_cls * cls_loss
    del h_base, h_prefix, h_lora, delta_prefix, delta_lora
    return {
        "gen_loss": gen_loss,
        "orth_loss": orth_loss,
        "orth_local_loss": local_loss,
        "orth_global_loss": global_loss,
        "cls_loss": cls_loss,
        "total_loss": total_loss,
        "used_orth": True,
        "cls_logits": cls_logits,
    }
