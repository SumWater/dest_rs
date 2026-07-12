from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.casino_dataset import StrategyDataCollator
from src.config import load_config
from src.losses import align_labels_and_response_mask
from src.modeling import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default="Dialogue history:\nPartner: What do you need?\nResponse:")
    parser.add_argument("--response", default="I need the water.")
    parser.add_argument("--out", default="reports/remote_checks/response_mask_check.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tokenizer = load_tokenizer(cfg)
    collator = StrategyDataCollator(tokenizer, cfg)
    item = {"dialogue_id": -1, "turn_index": 0, "speaker_id": "test", "primary_strategy": "self-need",
            "strategy_id": 4, "all_strategies": ["self-need"], "prompt": args.prompt, "target": args.response}
    batch = collator([item])
    logits_length = batch["labels"].size(1) + cfg.num_virtual_tokens
    labels, mask = align_labels_and_response_mask(
        batch["labels"], batch["response_mask"],
        num_virtual_tokens=cfg.num_virtual_tokens, logits_sequence_length=logits_length,
    )
    prefix_ids = [-1] * cfg.num_virtual_tokens
    ids = prefix_ids + batch["input_ids"][0].tolist()
    rows = []
    for index, (token_id, label, selected) in enumerate(zip(ids, labels[0].tolist(), mask[0].tolist())):
        rows.append({
            "input_token_index": index,
            "token_id": token_id,
            "decoded_token": "<PREFIX>" if token_id < 0 else tokenizer.decode([token_id]),
            "label": label,
            "response_mask": bool(selected),
            "shifted_response_mask": bool(mask[0, index + 1]) if index + 1 < mask.size(1) else False,
            "enters_lm_loss": bool(selected),
            "enters_sequence_score": bool(selected),
        })
    payload = {"status": "PASS", "num_virtual_tokens": cfg.num_virtual_tokens, "rows": rows}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
