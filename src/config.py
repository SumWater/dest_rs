from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class TrainConfig:
    # 路径配置
    model_name_or_path: str = "/replace/with/your/local/qwen/path"
    dataset_dir: str = "/replace/with/your/local/casino/data/or/split/path"
    train_file: Optional[str] = None
    valid_file: Optional[str] = None
    test_file: Optional[str] = None
    dataset_tag: str = "casino_original"
    experiment_name: str = "default"
    warm_start_dir: Optional[str] = None
    freeze_prefix: bool = False

    @property
    def need_dir(self) -> str:
        return os.path.join("output", "need", self.dataset_tag, self.experiment_name)

    @property
    def other_dir(self) -> str:
        return os.path.join("output", "other", self.dataset_tag, self.experiment_name)

    # 数据配置
    max_length: int = 512
    context_turns: int = 6
    max_reason_chars: int = 180
    include_profile: bool = True
    multi_label_policy: str = "drop"  # 可选值：first | drop | duplicate
    exclude_labels: List[str] = field(default_factory=lambda: ["non-strategic"])
    max_train_samples: Optional[int] = None
    max_valid_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    # 优化配置
    seed: int = 42
    batch_size: int = 1
    eval_batch_size: int = 1
    num_epochs: int = 2
    lr: float = 1e-4
    prefix_lr: float = 5e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_every_epoch: bool = True
    save_best_only: bool = True

    # 实验模式
    # 可选值：lora_only | prefix_only | prefix_lora | prefix_lora_orth | dest_rs
    adapter_mode: str = "dest_rs"

    # 骨干模型与量化配置
    trust_remote_code: bool = True
    use_4bit: bool = True
    use_bfloat16: bool = True
    gradient_checkpointing: bool = True

    # LoRA 配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_candidates: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # 策略前缀库配置
    num_virtual_tokens: int = 20
    prefix_init_std: float = 0.02
    prefix_scale_train: float = 1.0
    prefix_scale_eval: float = 1.0

    # 正交约束配置
    lambda_orth: float = 0.0
    orth_alpha: float = 0.5
    orth_start_step: int = 0
    orth_every_n_steps: int = 20
    orth_token_sample_size: int = 64
    orth_layer_index: int = -4

    # 策略分类辅助监督配置
    lambda_cls: float = 0.0
    cls_hidden_dim: int = 256
    cls_dropout: float = 0.1

    # 生成与样例配置
    demo_max_new_tokens: int = 40
    demo_temperature: float = 0.0
    demo_num_examples: int = 5


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    cfg = TrainConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def save_config(cfg: TrainConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
