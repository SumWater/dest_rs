"""Compute Oracle@K and target-aware judge reranking from candidate labels."""
import argparse, json
from collections import defaultdict
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument("--candidate-eval",type=Path,required=True)
    p.add_argument("--top1-eval",type=Path,required=True); p.add_argument("--out",type=Path,required=True); a=p.parse_args()
    candidates=json.loads(a.candidate_eval.read_text(encoding="utf-8"))["details"]
    top1=json.loads(a.top1_eval.read_text(encoding="utf-8"))["details"]
    groups=defaultdict(list)
    for d in candidates: groups[(d["dialogue_id"],d["turn_index"],d["target_strategy"])].append(d)
    for g in groups.values(): g.sort(key=lambda x:x["candidate_rank"])
    oracle4=oracle8=rerank=0; per_class=defaultdict(lambda:{"n":0,"oracle4":0,"oracle8":0,"rerank8":0})
    chosen=[]
    for key,g in groups.items():
        target=key[2]; o4=any(x["target_present"] for x in g[:4]); o8=any(x["target_present"] for x in g[:8])
        # Target-aware judge reranker: primary target, then target presence, fewer off-targets,
        # then B9 length-normalized sequence log-probability.
        best=max(g[:8],key=lambda x:(x["primary_strategy"]==target,x["target_present"],
                                     -x["off_target_strategy_count"],x.get("sequence_logprob",-1e9)))
        rr=bool(best["target_present"]); oracle4+=o4; oracle8+=o8; rerank+=rr
        q=per_class[target]; q["n"]+=1; q["oracle4"]+=o4; q["oracle8"]+=o8; q["rerank8"]+=rr
        chosen.append({"key":key,"candidate_rank":best["candidate_rank"],"utterance":best["utterance"],
                       "target_present":best["target_present"],"primary_strategy":best["primary_strategy"]})
    top1_by_key={(x["dialogue_id"],x["turn_index"],x["target_strategy"]):x for x in top1}
    matched_top1=[top1_by_key[key] for key in groups if key in top1_by_key]
    if len(matched_top1) != len(groups): raise RuntimeError("Top-1 and candidate task keys do not match")
    n=len(groups); top1_presence=sum(x["target_present"] for x in matched_top1)/len(matched_top1)
    metrics={"tasks":n,"top1_target_presence":top1_presence,"oracle_at_4":oracle4/n,
             "oracle_at_8":oracle8/n,"rerank_at_8":rerank/n,
             "per_class":{s:{k:(v if k=="n" else v/q["n"]) for k,v in q.items()} for s,q in per_class.items()}}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps({"metrics":metrics,"chosen":chosen},indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(metrics,indent=2,ensure_ascii=False))
if __name__=="__main__": main()
