from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


STRATEGY_DEFINITIONS = {
    "elicit-pref": "询问对方偏好、需求、优先级或约束，例如询问对方最需要什么、哪个物品更重要。",
    "self-need": "表达己方需求、困难、优先级或理由，例如说明自己为什么需要某个资源。",
    "other-need": "关注、回应或承认对方需求，并围绕对方需求调整表达或方案。",
    "promote-coordination": "推动合作、协调或共同达成方案，例如提出双方都能接受的安排。",
    "vouch-fair": "强调公平、公正、合理、均衡分配或互惠。",
    "showing-empathy": "表达理解、同情、安抚或认可对方处境。",
    "small-talk": "寒暄、闲聊或礼貌性交流，不直接推进资源谈判。",
    "no-need": "表达自己不需要某项资源，或某项资源对自己优先级较低。",
    "uv-part": "表达部分让步、部分接受或只愿意给出/接受一部分资源。",
}


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_prompt(row: dict) -> str:
    target_strategy = row.get("target_strategy", "")
    target_definition = STRATEGY_DEFINITIONS.get(target_strategy, "未提供定义，请根据策略名称和回复语义判断。")
    all_definitions = "\n".join(f"- {name}: {desc}" for name, desc in STRATEGY_DEFINITIONS.items())
    return f"""你是谈判对话策略标注员。请判断两个模型回复是否符合目标策略。

【策略定义】
{all_definitions}

【目标策略】
{target_strategy}: {target_definition}

【对话上下文】
{row.get("prompt", "")}

【参考真实回复】
{row.get("gold_target", "")}

【B4 回复：Prefix + LoRA】
{row.get("b4_response", "")}

【B5 回复：Prefix + LoRA + Orth】
{row.get("b5_response", "")}

请只根据“目标策略”判断 B4/B5 回复是否体现该策略。不要因为语言更流畅就判定符合策略；也不要要求回复必须和参考真实回复完全一致。

请严格输出一个 JSON 对象，不要输出 Markdown，不要输出额外解释。格式如下：
{{
  "b4_match": 0 或 1,
  "b5_match": 0 或 1,
  "which_better": "b4" 或 "b5" 或 "tie",
  "judge_note": "一句中文理由"
}}
"""


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"无法从模型输出中解析 JSON：{text[:300]}")


def call_deepseek(
    prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    timeout: int,
    max_retries: int,
) -> dict:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是严格、稳定的学术实验标注助手。你只输出合法 JSON。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
            raw = json.loads(body)
            content = raw["choices"][0]["message"]["content"]
            return extract_json(content)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            wait = min(2**attempt, 30)
            print(f"[重试] 第 {attempt}/{max_retries} 次调用失败：{exc}；{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"DeepSeek 调用失败：{last_error}")


def normalize_judgement(result: dict) -> dict:
    b4_match = int(result.get("b4_match", 0))
    b5_match = int(result.get("b5_match", 0))
    b4_match = 1 if b4_match else 0
    b5_match = 1 if b5_match else 0

    which = str(result.get("which_better", "tie")).strip().lower()
    if which not in {"b4", "b5", "tie"}:
        if b5_match > b4_match:
            which = "b5"
        elif b4_match > b5_match:
            which = "b4"
        else:
            which = "tie"

    return {
        "b4_match": str(b4_match),
        "b5_match": str(b5_match),
        "which_better": which,
        "judge_note": str(result.get("judge_note", "")).replace("\n", " ").strip(),
    }


def should_skip(row: dict, resume: bool) -> bool:
    if not resume:
        return False
    return bool((row.get("b4_match") or "").strip() and (row.get("b5_match") or "").strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 DeepSeek API 自动标注 B4/B5 judge 表格。")
    parser.add_argument("--input", required=True, help="build_judge_sheet.py 生成的 CSV。")
    parser.add_argument("--out", required=True, help="写入自动标注后的 CSV。")
    parser.add_argument("--api-key", default=None, help="DeepSeek API Key；默认读取环境变量 DEEPSEEK_API_KEY。")
    parser.add_argument("--base-url", default="https://api.deepseek.com", help="DeepSeek OpenAI-compatible API Base URL。")
    parser.add_argument("--model", default="deepseek-v4-pro", help="DeepSeek 模型 ID，默认 deepseek-v4-pro。")
    parser.add_argument("--temperature", type=float, default=0.0, help="judge 温度，默认 0。")
    parser.add_argument("--timeout", type=int, default=120, help="单次请求超时时间。")
    parser.add_argument("--max-retries", type=int, default=3, help="失败重试次数。")
    parser.add_argument("--sleep", type=float, default=0.5, help="每行标注之间的暂停秒数。")
    parser.add_argument("--limit", type=int, default=None, help="最多标注多少行，用于试跑。")
    parser.add_argument("--resume", action="store_true", help="跳过已经有 b4_match/b5_match 的行。")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("缺少 DeepSeek API Key。请设置环境变量 DEEPSEEK_API_KEY，或传入 --api-key。")

    input_path = Path(args.input)
    out_path = Path(args.out)
    rows = load_rows(input_path)
    if not rows:
        raise SystemExit(f"输入 CSV 为空：{input_path}")

    fieldnames = list(rows[0].keys())
    for name in ["b4_match", "b5_match", "which_better", "judge_note", "llm_judge_model"]:
        if name not in fieldnames:
            fieldnames.append(name)

    processed = 0
    for idx, row in enumerate(rows, start=1):
        if args.limit is not None and processed >= args.limit:
            break
        if should_skip(row, args.resume):
            continue

        print(f"[{idx}/{len(rows)}] 标注 dialogue={row.get('dialogue_id')} turn={row.get('turn_index')} strategy={row.get('target_strategy')}")
        prompt = build_prompt(row)
        result = call_deepseek(
            prompt=prompt,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
        )
        normalized = normalize_judgement(result)
        row.update(normalized)
        row["llm_judge_model"] = args.model
        processed += 1

        save_rows(out_path, rows, fieldnames)
        if args.sleep > 0:
            time.sleep(args.sleep)

    save_rows(out_path, rows, fieldnames)
    print(f"已完成标注 {processed} 行，结果写入：{out_path}")


if __name__ == "__main__":
    main()
