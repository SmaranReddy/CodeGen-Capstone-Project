"""Generate MongoDB queries with the fine-tuned model (Qwen2.5-Coder-3B-Instruct + LoRA adapter q3b_r16_q10).

Same settings as the scored MQL-0010 FT-A0 run: base model at the pinned revision in bfloat16, adapter applied on top,
greedy decoding, max 1024 new tokens, Qwen chat template (system + user message), output parsed as one JSON object.

Prompt input (pick one):
  --prompts FILE --id ID    a prompts.jsonl row with "system" and "messages" (e.g. MQL-0010 data/eval/prompts.jsonl,
                            the frozen nl-mql-spider1-v1 prompts the model was scored on)
  --system FILE --user FILE plain-text system prompt and user message (see PROMPT_FORMAT.md)

Optional:
  --merge-to DIR            also save base+adapter merged into one normal model folder (loads without peft)

Examples:
  python generate.py --adapter ./q3b_r16_q10 --prompts prompts.jsonl --id spider_train_00512
  python generate.py --adapter ./q3b_r16_q10 --system system.txt --user user.txt
Needs: torch, transformers, peft (a GPU is optional; CPU works but is slow).
"""
import argparse
import json
import sys
from pathlib import Path

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
BASE_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"


def parse_output(text):
    """Same rule as the project scorer (dev1034_core.parse_output): strip ``` fences, take the first JSON object."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t), None
    except Exception:
        pass
    start, depth = t.find("{"), 0
    if start == -1:
        return None, "no JSON object found"
    for i in range(start, len(t)):
        depth += {"{": 1, "}": -1}.get(t[i], 0)
        if depth == 0:
            try:
                return json.loads(t[start:i + 1]), None
            except Exception as e:
                return None, f"JSON parse error: {e}"
    return None, "unbalanced JSON braces"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="adapter folder (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--revision", default=BASE_REVISION)
    ap.add_argument("--prompts"); ap.add_argument("--id")
    ap.add_argument("--system"); ap.add_argument("--user")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--merge-to")
    a = ap.parse_args()

    if a.prompts and a.id:
        row = next((json.loads(l) for l in open(a.prompts, encoding="utf-8") if json.loads(l)["id"] == a.id), None)
        if row is None:
            sys.exit(f"id {a.id} not in {a.prompts}")
        system, messages = row["system"], row["messages"]
    elif a.system and a.user:
        system = Path(a.system).read_text(encoding="utf-8")
        messages = [{"role": "user", "content": Path(a.user).read_text(encoding="utf-8")}]
    else:
        sys.exit("give --prompts + --id, or --system + --user")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import peft
    from peft import PeftModel
    # some peft versions raise on an old torchao (Kaggle ships 0.10); a plain LoRA never uses torchao
    for m in list(sys.modules.values()):
        if getattr(m, "__name__", "").startswith("peft") and hasattr(m, "is_torchao_available"):
            m.is_torchao_available = lambda: False

    cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if cuda else torch.float32
    rev = None if Path(a.base).is_dir() else a.revision
    tok = AutoTokenizer.from_pretrained(a.base, revision=rev)
    base = AutoModelForCausalLM.from_pretrained(a.base, revision=rev, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload().eval()   # merged = same weights as the scored run
    if a.merge_to:
        model.save_pretrained(a.merge_to, safe_serialization=True)
        tok.save_pretrained(a.merge_to)
        print("merged model saved to", a.merge_to, file=sys.stderr)
    if cuda:
        model = model.to("cuda")

    text = tok.apply_chat_template([{"role": "system", "content": system}] + messages, tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id,
                             temperature=None, top_p=None, top_k=None)
    raw = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    query, err = parse_output(raw)
    print(json.dumps({"raw_text": raw, "query": query, "parse_error": err}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
