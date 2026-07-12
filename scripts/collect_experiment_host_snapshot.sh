#!/usr/bin/env bash
# Run from the project root on the Linux experiment host. Read-only except for reports/.
set -euo pipefail

mkdir -p reports/host_snapshot

git rev-parse HEAD > reports/host_snapshot/git_commit.txt
git status --short > reports/host_snapshot/git_status.txt
git tag --list > reports/host_snapshot/git_tags.txt
python --version > reports/host_snapshot/python_version.txt 2>&1
python -m pip freeze > reports/host_snapshot/pip_freeze.txt
nvidia-smi > reports/host_snapshot/nvidia_smi.txt

python - <<'PY'
import json
from pathlib import Path

versions = {}
for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
    try:
        module = __import__(name)
        versions[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        versions[name] = f"unavailable: {exc}"
try:
    import torch
    versions["torch_cuda_version"] = torch.version.cuda
    versions["cuda_available"] = torch.cuda.is_available()
    versions["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception:
    pass
Path("reports/host_snapshot/package_versions.json").write_text(
    json.dumps(versions, indent=2, ensure_ascii=False), encoding="utf-8"
)
PY

# Hash manifests avoid copying large checkpoints merely for registration.
find configs scripts src -type f -print0 2>/dev/null \
  | sort -z | xargs -0 sha256sum > reports/host_snapshot/code_config_sha256.txt
find output/other -type f \( -name '*.bin' -o -name '*.safetensors' -o -name '*.pt' -o -name '*.ckpt' -o -name 'adapter_config.json' \) -print0 2>/dev/null \
  | sort -z | xargs -0 -r sha256sum > reports/host_snapshot/checkpoint_sha256.txt
find output/need -type f \( -name '*.json' -o -name '*.jsonl' \) -print0 2>/dev/null \
  | sort -z | xargs -0 -r sha256sum > reports/host_snapshot/result_sha256.txt

echo "Snapshot written to reports/host_snapshot/"
