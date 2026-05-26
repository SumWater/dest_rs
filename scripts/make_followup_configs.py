from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_value(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "")


def apply_common(cfg: dict, run_id: str, output_root: str, model_path: str | None, dataset_dir: str | None) -> dict:
    cfg = deepcopy(cfg)
    cfg["output_dir"] = str(Path(output_root) / run_id).replace("\\", "/")
    if model_path:
        cfg["model_name_or_path"] = model_path
    if dataset_dir:
        cfg["dataset_dir"] = dataset_dir
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="生成新蓝图对应的补充实验配置。")
    parser.add_argument("--b4-template", default="configs/b4_prefix_lora.json", help="B4 配置模板路径。")
    parser.add_argument("--b5-template", default="configs/b5_prefix_lora_orth.json", help="B5 配置模板路径。")
    parser.add_argument("--out-dir", default="configs/followup", help="生成配置输出目录。")
    parser.add_argument("--output-root", default="./outputs/followup", help="训练产物根目录。")
    parser.add_argument("--model-path", default=None, help="可选：覆盖模型路径。")
    parser.add_argument("--dataset-dir", default=None, help="可选：覆盖数据集路径。")
    args = parser.parse_args()

    b4 = read_json(Path(args.b4_template))
    b5 = read_json(Path(args.b5_template))
    out_dir = Path(args.out_dir)
    written: list[Path] = []

    # 一、lambda_orth 敏感性实验。
    for value in [0.01, 0.05, 0.10]:
        run_id = f"b5_lambda_{normalize_value(value)}"
        cfg = apply_common(b5, run_id, args.output_root, args.model_path, args.dataset_dir)
        cfg.update(
            {
                "adapter_mode": "prefix_lora_orth",
                "lambda_orth": value,
                "lambda_cls": 0.0,
            }
        )
        path = out_dir / "lambda_orth" / f"{run_id}.json"
        write_json(cfg, path)
        written.append(path)

    # 二、局部/全局正交权重 alpha 消融。
    for alpha, label in [(0.0, "global_only"), (0.5, "local_global"), (1.0, "local_only")]:
        run_id = f"b5_alpha_{label}"
        cfg = apply_common(b5, run_id, args.output_root, args.model_path, args.dataset_dir)
        cfg.update(
            {
                "adapter_mode": "prefix_lora_orth",
                "orth_alpha": alpha,
                "lambda_cls": 0.0,
            }
        )
        path = out_dir / "orth_alpha" / f"{run_id}.json"
        write_json(cfg, path)
        written.append(path)

    # 三、B4/B5 多随机种子稳定性实验。
    for seed in [42, 43, 44]:
        b4_run = f"b4_seed_{seed}"
        b4_cfg = apply_common(b4, b4_run, args.output_root, args.model_path, args.dataset_dir)
        b4_cfg.update({"adapter_mode": "prefix_lora", "lambda_orth": 0.0, "lambda_cls": 0.0, "seed": seed})
        b4_path = out_dir / "seeds" / f"{b4_run}.json"
        write_json(b4_cfg, b4_path)
        written.append(b4_path)

        b5_run = f"b5_seed_{seed}"
        b5_cfg = apply_common(b5, b5_run, args.output_root, args.model_path, args.dataset_dir)
        b5_cfg.update({"adapter_mode": "prefix_lora_orth", "lambda_cls": 0.0, "seed": seed})
        b5_path = out_dir / "seeds" / f"{b5_run}.json"
        write_json(b5_cfg, b5_path)
        written.append(b5_path)

    # 四、prefix length 与 LoRA rank 的小规模效率分析。
    for prefix_len in [8, 16, 20]:
        for rank in [8, 16]:
            run_id = f"b5_prefix_{prefix_len}_rank_{rank}"
            cfg = apply_common(b5, run_id, args.output_root, args.model_path, args.dataset_dir)
            cfg.update(
                {
                    "adapter_mode": "prefix_lora_orth",
                    "num_virtual_tokens": prefix_len,
                    "lora_r": rank,
                    "lora_alpha": rank * 2,
                    "lambda_cls": 0.0,
                }
            )
            path = out_dir / "efficiency" / f"{run_id}.json"
            write_json(cfg, path)
            written.append(path)

    print(f"已生成 {len(written)} 个配置文件：")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
