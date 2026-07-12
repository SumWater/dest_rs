"""Batch version of the multi-label judge for flat Top-K candidate JSONL."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from evaluate_strategy_multilabel_llm import SYSTEM, compute_metrics, context_from_record, parse_answer, prompt

def chunks(xs,n):
    for i in range(0,len(xs),n): yield xs[i:i+n]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--model-path",required=True); p.add_argument("--jsonl",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-new-tokens",type=int,default=64); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    records=[json.loads(x) for x in a.jsonl.read_text(encoding="utf-8").splitlines() if x.strip()]
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(a.model_path,quantization_config=q,device_map="auto",dtype=torch.bfloat16,trust_remote_code=True).eval()
    tokenizer=AutoTokenizer.from_pretrained(a.model_path,trust_remote_code=True); tokenizer.padding_side="left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id=tokenizer.eos_token_id
    details=[]
    for batch in chunks(records,a.batch_size):
        texts=[]
        for r in batch:
            messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt(context_from_record(r,True),r["utterance"])}]
            texts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False))
        enc=tokenizer(texts,return_tensors="pt",padding=True).to(model.device)
        with torch.inference_mode(): out=model.generate(**enc,max_new_tokens=a.max_new_tokens,do_sample=False,pad_token_id=tokenizer.pad_token_id)
        decoded=tokenizer.batch_decode(out[:,enc.input_ids.shape[1]:],skip_special_tokens=True)
        for r,raw in zip(batch,decoded):
            raw=raw.strip(); present,primary=parse_answer(raw); target=r["target_strategy"]
            details.append({"dialogue_id":r.get("dialogue_id"),"turn_index":r.get("turn_index"),"target_strategy":target,
                "utterance":r["utterance"],"present_strategies":present,"primary_strategy":primary,
                "target_present":int(target in present),"primary_correct":int(primary==target),
                "off_target_strategies":[s for s in present if s!=target],"off_target_strategy_count":sum(s!=target for s in present),
                "parse_failed":int(not present and primary is None and "{}" not in raw),"raw_response":raw,
                "candidate_rank":r["candidate_rank"],"candidate_seed":r["candidate_seed"],
                "sequence_logprob":r["sequence_logprob"],"token_count":r["token_count"]})
        print(f"judged={len(details)}/{len(records)}",flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({"metrics":compute_metrics(details),"details":details},indent=2,ensure_ascii=False),encoding="utf-8")
if __name__=="__main__": main()
