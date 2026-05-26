from __future__ import annotations

from contextlib import contextmanager
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from .config import TrainConfig


def get_embed_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


@contextmanager
def lora_disabled_ctx(peft_model):
    if hasattr(peft_model, "disable_adapter"):
        try:
            with peft_model.disable_adapter():
                yield
            return
        except Exception:
            pass

    if hasattr(peft_model, "disable_adapter_layers") and hasattr(peft_model, "enable_adapter_layers"):
        try:
            peft_model.disable_adapter_layers()
            yield
        finally:
            peft_model.enable_adapter_layers()
        return

    if hasattr(peft_model, "disable_adapters"):
        try:
            ctx = peft_model.disable_adapters()
            if ctx is not None and hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__"):
                with ctx:
                    yield
                return
            yield
            return
        except Exception:
            pass

    lora_layers = []
    old_values = []
    for module in peft_model.modules():
        if hasattr(module, "disable_adapters"):
            lora_layers.append(module)
            old_values.append(getattr(module, "disable_adapters"))
    try:
        for module in lora_layers:
            setattr(module, "disable_adapters", True)
        yield
    finally:
        for module, old_value in zip(lora_layers, old_values):
            setattr(module, "disable_adapters", old_value)


class HybridStrategyModel(nn.Module):
    def __init__(
        self,
        peft_model: nn.Module,
        hidden_size: int,
        num_strategies: int,
        num_virtual_tokens: int,
        init_std: float,
    ):
        super().__init__()
        self.peft_model = peft_model
        self.num_strategies = num_strategies
        self.num_virtual_tokens = num_virtual_tokens
        self.prefix_bank = nn.Parameter(
            torch.randn(num_strategies, num_virtual_tokens, hidden_size) * init_std
        )
        self.strategy_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_strategies),
        )

    def configure_classifier(self, hidden_size: int, cls_hidden_dim: int, cls_dropout: float) -> None:
        self.strategy_classifier = nn.Sequential(
            nn.Dropout(cls_dropout),
            nn.Linear(hidden_size, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, self.num_strategies),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        strategy_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        prefix_on: bool = True,
        prefix_scale: float = 1.0,
        prefix_override: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        embed_layer = self.peft_model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)

        extended_labels = labels
        extended_attention_mask = attention_mask

        if prefix_on:
            if prefix_override is not None:
                prefix = prefix_override.to(inputs_embeds.device, dtype=inputs_embeds.dtype) * prefix_scale
                if prefix.dim() == 2:
                    prefix = prefix.unsqueeze(0)
                if prefix.size(0) == 1 and input_ids.size(0) > 1:
                    prefix = prefix.expand(input_ids.size(0), -1, -1)
            else:
                if strategy_ids.dim() == 0:
                    strategy_ids = strategy_ids.unsqueeze(0)
                prefix = self.prefix_bank[strategy_ids] * prefix_scale
            inputs_embeds = torch.cat([prefix, inputs_embeds], dim=1)

            batch_size = attention_mask.size(0)
            prefix_mask = torch.ones(
                (batch_size, self.num_virtual_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            extended_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

            if labels is not None:
                prefix_labels = torch.full(
                    (batch_size, self.num_virtual_tokens),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                extended_labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.peft_model(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_attention_mask,
            labels=None,
            use_cache=use_cache,
            return_dict=True,
        )
        return outputs, extended_labels, extended_attention_mask

    def slice_real_tokens(self, hidden_states: torch.Tensor, prefix_on: bool) -> torch.Tensor:
        if prefix_on:
            return hidden_states[:, self.num_virtual_tokens :, :]
        return hidden_states


class LayerCatcher:
    def __init__(self, layer_module: nn.Module):
        self.layer_module = layer_module
        self.handle = None
        self.value = None

    def _hook(self, module, inputs, output):
        captured = output[0] if isinstance(output, tuple) else output
        self.value = captured

    def install(self) -> None:
        if self.handle is None:
            self.handle = self.layer_module.register_forward_hook(self._hook)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def reset(self) -> None:
        self.value = None

    def pop(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("LayerCatcher did not capture any hidden state.")
        value = self.value
        self.value = None
        return value


def unwrap_to_base_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "get_base_model"):
        try:
            return model.get_base_model()
        except Exception:
            pass
    if hasattr(model, "base_model"):
        base_model = getattr(model, "base_model")
        if hasattr(base_model, "model"):
            return base_model.model
    if hasattr(model, "model"):
        return model.model
    return model


def find_transformer_blocks(base_model: nn.Module) -> nn.ModuleList:
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        blocks = base_model.model.layers
        if isinstance(blocks, nn.ModuleList):
            return blocks
    if hasattr(base_model, "layers") and isinstance(base_model.layers, nn.ModuleList):
        return base_model.layers
    if hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
        blocks = base_model.transformer.h
        if isinstance(blocks, nn.ModuleList):
            return blocks
    raise RuntimeError("无法自动定位 transformer blocks。")


def build_layer_catcher(peft_model: nn.Module, layer_index: int) -> LayerCatcher:
    base = unwrap_to_base_model(peft_model)
    blocks = find_transformer_blocks(base)
    num_blocks = len(blocks)
    idx = layer_index if layer_index >= 0 else num_blocks + layer_index
    if idx < 0 or idx >= num_blocks:
        raise ValueError(f"orth_layer_index={layer_index} is out of range for {num_blocks} blocks")
    catcher = LayerCatcher(blocks[idx])
    catcher.install()
    print(f"[model] 已在 transformer block {idx}/{num_blocks - 1} 上安装层输出捕获器")
    return catcher


def detect_lora_targets(model, candidates: List[str]) -> List[str]:
    found = set()
    for name, _ in model.named_modules():
        for candidate in candidates:
            if name.endswith(candidate):
                found.add(candidate)
    return sorted(found) if found else list(candidates)


def load_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_backbone(cfg: TrainConfig):
    use_cuda = torch.cuda.is_available()
    use_4bit = bool(cfg.use_4bit and use_cuda)
    dtype = torch.bfloat16 if (cfg.use_bfloat16 and use_cuda) else torch.float32

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map="auto" if use_cuda else None,
    )
    model.config.use_cache = False
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    return model


def attach_lora(base_model, cfg: TrainConfig, warm_start_dir: str | None = None):
    lora_dir = os.path.join(warm_start_dir, "lora_adapter") if warm_start_dir else None
    if lora_dir and os.path.exists(lora_dir):
        peft_model = PeftModel.from_pretrained(base_model, lora_dir, is_trainable=True)
        print(f"[model] 已从 {lora_dir} 加载 LoRA warm start")
        return peft_model

    targets = detect_lora_targets(base_model, cfg.lora_target_candidates)
    print(f"[model] LoRA target modules: {targets}")
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    return peft_model


def build_hybrid_model(cfg: TrainConfig, num_strategies: int):
    tokenizer = load_tokenizer(cfg)
    base_model = load_backbone(cfg)
    peft_model = attach_lora(base_model, cfg, cfg.warm_start_dir)
    hidden_size = peft_model.get_input_embeddings().weight.shape[-1]

    hybrid = HybridStrategyModel(
        peft_model=peft_model,
        hidden_size=hidden_size,
        num_strategies=num_strategies,
        num_virtual_tokens=cfg.num_virtual_tokens,
        init_std=cfg.prefix_init_std,
    )
    hybrid.configure_classifier(hidden_size, cfg.cls_hidden_dim, cfg.cls_dropout)
    embed_device = get_embed_device(peft_model)
    hybrid.prefix_bank.data = hybrid.prefix_bank.data.to(embed_device)
    hybrid.strategy_classifier.to(embed_device)

    prefix_path = os.path.join(cfg.warm_start_dir, "prefix_bank.pt") if cfg.warm_start_dir else None
    if prefix_path and os.path.exists(prefix_path):
        payload = torch.load(prefix_path, map_location="cpu")
        saved_prefix = payload["prefix_bank"]
        if tuple(saved_prefix.shape) != tuple(hybrid.prefix_bank.shape):
            raise ValueError(
                f"Warm-start prefix 形状不匹配：期望 {tuple(hybrid.prefix_bank.shape)}，实际 {tuple(saved_prefix.shape)}"
            )
        hybrid.prefix_bank.data.copy_(saved_prefix.to(hybrid.prefix_bank.device, dtype=hybrid.prefix_bank.dtype))
        if "strategy_classifier" in payload:
            hybrid.strategy_classifier.load_state_dict(payload["strategy_classifier"], strict=False)
        print(f"[model] 已从 {prefix_path} 加载 prefix bank warm start")

    catcher = build_layer_catcher(peft_model, cfg.orth_layer_index)
    return hybrid, tokenizer, catcher


def freeze_for_adapter_mode(
    hybrid: HybridStrategyModel,
    adapter_mode: str,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
    for _, param in hybrid.peft_model.named_parameters():
        param.requires_grad = False
    for _, param in hybrid.strategy_classifier.named_parameters():
        param.requires_grad = False

    train_prefix = adapter_mode in {"prefix_only", "prefix_lora", "prefix_lora_orth", "dest_rs"}
    train_lora = adapter_mode in {"lora_only", "prefix_lora", "prefix_lora_orth", "dest_rs"}
    train_cls = adapter_mode == "dest_rs"

    if train_lora:
        for name, param in hybrid.peft_model.named_parameters():
            if "lora_" in name.lower():
                param.requires_grad = True
    hybrid.prefix_bank.requires_grad = train_prefix
    if train_cls:
        for _, param in hybrid.strategy_classifier.named_parameters():
            param.requires_grad = True

    prefix_params: List[nn.Parameter] = []
    lora_params: List[nn.Parameter] = []
    cls_params: List[nn.Parameter] = []
    for name, param in hybrid.named_parameters():
        if not param.requires_grad:
            continue
        if "prefix_bank" in name:
            prefix_params.append(param)
        elif "strategy_classifier" in name:
            cls_params.append(param)
        else:
            lora_params.append(param)
    return prefix_params, lora_params, cls_params


def print_trainable_parameters(hybrid: HybridStrategyModel) -> None:
    trainable = [(name, param.numel()) for name, param in hybrid.named_parameters() if param.requires_grad]
    total = sum(x[1] for x in trainable)
    print("=" * 88)
    print(f"[model] trainable parameter tensors: {len(trainable)} | total elements: {total:,}")
    for name, numel in trainable:
        print(f"  {name:<70} {numel:>12,}")
    print("=" * 88)
