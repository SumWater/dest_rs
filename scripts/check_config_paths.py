from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: str | None, base_dir: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def check_dataset_dir(dataset_dir: Path) -> bool:
    direct = dataset_dir / "casino_train.json"
    nested = dataset_dir / "split" / "casino_train.json"
    return direct.exists() or nested.exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="检查配置文件中的模型、数据和输出路径。")
    parser.add_argument("configs", nargs="+", help="配置文件路径或 glob，例如 configs/*.json。")
    parser.add_argument("--base-dir", default=".", help="相对路径的解析基准，默认是当前目录。")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    paths: list[Path] = []
    for pattern in args.configs:
        matched = [Path(p) for p in glob.glob(pattern, recursive=True)]
        if matched:
            paths.extend(matched)
        else:
            paths.append(Path(pattern))

    ok = True
    for config_path in sorted(set(paths)):
        cfg = read_json(config_path)
        model_path = resolve_path(cfg.get("model_name_or_path"), base_dir)
        dataset_dir = resolve_path(cfg.get("dataset_dir"), base_dir)
        output_dir = resolve_path(cfg.get("output_dir"), base_dir)

        print("=" * 80)
        print(f"配置文件：{config_path}")
        print(f"模型路径：{model_path}")
        print(f"数据路径：{dataset_dir}")
        print(f"输出路径：{output_dir}")

        if model_path is None or "/replace/with/" in str(model_path) or not model_path.exists():
            ok = False
            print("  [错误] 模型路径不存在或仍是占位符。")
        else:
            print("  [通过] 模型路径存在。")

        if dataset_dir is None or not dataset_dir.exists() or not check_dataset_dir(dataset_dir):
            ok = False
            print("  [错误] 数据路径不存在，或缺少 casino_train.json / split/casino_train.json。")
        else:
            print("  [通过] 数据路径结构正常。")

        if output_dir is None:
            ok = False
            print("  [错误] output_dir 为空。")
        else:
            print("  [通过] 输出路径可由训练脚本自动创建。")

    print("=" * 80)
    if ok:
        print("所有配置路径检查通过。")
    else:
        raise SystemExit("存在配置路径问题，请先修改后再运行训练。")


if __name__ == "__main__":
    main()
