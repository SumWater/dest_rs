"""Sample K=8 candidates from B9 for Oracle/Rerank diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.casino_dataset import StrategyLabelSpace
from src.config import load_config, resolve_warm_start_dir
from src.modeling import build_hybrid_model, freeze_for_adapter_mode, get_embed_device


@torch.inference_mode()
def sample_k(hybrid, tokenizer, prompt, strategy_id, cfg, k, seed, temperature, top_p, max_new_tokens):
    hybrid.eval(); device = get_embed_device(hybrid.peft_model)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device).repeat(k, 1)
    attention = enc.attention_mask.to(device).repeat(k, 1)
    strategy_ids = torch.full((k,), strategy_id, dtype=torch.long, device=device)
    generated = [[] for _ in range(k)]; logprob_sums = torch.zeros(k, device=device)
    finished = torch.zeros(k, dtype=torch.bool, device=device)
    generators = [torch.Generator(device=device).manual_seed(seed + i) for i in range(k)]
    for _ in range(max_new_tokens):
        outputs, _, _ = hybrid(input_ids=input_ids, attention_mask=attention,
                               strategy_ids=strategy_ids, labels=None, prefix_on=True,
                               prefix_scale=cfg.prefix_scale_eval, use_cache=False)
        raw_logits = outputs.logits[:, -1, :]
        raw_log_probs = torch.log_softmax(raw_logits, dim=-1)
        probs = torch.softmax(raw_logits / temperature, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = sorted_probs.cumsum(-1)
            remove = cumulative - sorted_probs > top_p
            sorted_probs[remove] = 0
            sorted_probs /= sorted_probs.sum(-1, keepdim=True)
            next_tokens = torch.stack([
                sorted_idx[i, torch.multinomial(sorted_probs[i], 1, generator=generators[i])].squeeze()
                for i in range(k)
            ])
        else:
            next_tokens = torch.stack([torch.multinomial(probs[i], 1, generator=generators[i]).squeeze() for i in range(k)])
        # Store the untempered B9 model log-probability, not the sampling-distribution score.
        chosen = raw_log_probs.gather(1, next_tokens[:, None]).squeeze(1)
        logprob_sums += torch.where(finished, 0, chosen)
        for i, token in enumerate(next_tokens.tolist()):
            if not finished[i]:
                if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
                    finished[i] = True
                else: generated[i].append(token)
        append_tokens = torch.where(finished, torch.full_like(next_tokens, tokenizer.pad_token_id), next_tokens)
        input_ids = torch.cat([input_ids, append_tokens[:, None]], dim=1)
        attention = torch.cat([attention, (~finished).to(attention.dtype)[:, None]], dim=1)
        if finished.all(): break
    return [{"utterance": tokenizer.decode(tokens, skip_special_tokens=True).strip(),
             "token_count": len(tokens),
             "sequence_logprob": float(logprob_sums[i] / max(1, len(tokens)))}
            for i, tokens in enumerate(generated)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--label-map", required=True); p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=42000); p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9); p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--num-contexts", type=int, default=None)
    args = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    cfg = load_config(args.config); cfg.warm_start_dir = args.checkpoint
    cfg.warm_start_prefix = True; cfg.warm_start_lora = True; resolve_warm_start_dir(cfg)
    checkpoint = Path(cfg.warm_start_dir)
    for required in (checkpoint / "prefix_bank.pt", checkpoint / "lora_adapter" / "adapter_model.safetensors"):
        if not required.exists(): raise FileNotFoundError(required)
    label_space = StrategyLabelSpace.from_json(json.loads(Path(args.label_map).read_text(encoding="utf-8")))
    reference = [json.loads(x) for x in args.reference.read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.num_contexts is not None:
        reference = reference[:args.num_contexts]
    hybrid, tokenizer, catcher = build_hybrid_model(cfg, len(label_space.labels)); catcher.remove()
    freeze_for_adapter_mode(hybrid, cfg); hybrid.eval(); args.out.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with args.out.open("w", encoding="utf-8") as out:
        for record in reference:
            for strategy_id, strategy in label_space.id_to_label.items():
                task_seed = args.seed + done * args.k
                candidates = sample_k(hybrid, tokenizer, record["prompt"], strategy_id, cfg, args.k,
                                      task_seed, args.temperature, args.top_p, args.max_new_tokens)
                for rank, candidate in enumerate(candidates, 1):
                    out.write(json.dumps({
                        "dialogue_id": record["dialogue_id"], "turn_index": record["turn_index"],
                        "prompt": record["prompt"], "target_strategy": strategy,
                        "candidate_rank": rank, "candidate_seed": task_seed + rank - 1,
                        **candidate,
                        "sampling": {"temperature": args.temperature, "top_p": args.top_p,
                                     "max_new_tokens": args.max_new_tokens},
                    }, ensure_ascii=False) + "\n")
                done += 1; print(f"tasks={done}/{len(reference)*len(label_space.labels)}", flush=True)
    print(f"saved={args.out} candidates={done*args.k}")


if __name__ == "__main__": main()
