"""Token-length audit of the ft-v1 SFT records with the real Qwen2.5-Coder tokenizer.

    python scripts/finetune_v1/token_audit.py [--gguf models/qwen2.5-coder-7b-instruct-q4_k_m.gguf]

Loads only the GGUF vocabulary (vocab_only=True, no weights) and counts tokens of the
ChatML-rendered prompt (system + user) and of the completion (assistant target + <|im_end|>).
Writes data/finetune/ft_v1/token_audit.json (used to set max_seq_length in the plan).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nk_common as C  # noqa: E402


def chatml(msgs, add_generation=False):
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    return s + ("<|im_start|>assistant\n" if add_generation else "")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=str(C.ROOT / "models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"))
    args = ap.parse_args()
    from llama_cpp import Llama
    tok = Llama(model_path=args.gguf, vocab_only=True, verbose=False)
    out = {"tokenizer": Path(args.gguf).name, "template": "ChatML (Qwen2.5)"}
    for name in ("train.jsonl", "dev_internal.jsonl"):
        recs = C.read_jsonl(C.OUT_DIR / name)
        groups = {}
        for r in recs:
            p = len(tok.tokenize(chatml(r["messages"][:-1], True).encode("utf-8"), add_bos=False, special=True))
            c = len(tok.tokenize((r["messages"][-1]["content"] + "<|im_end|>").encode("utf-8"), add_bos=False, special=True))
            for g in ("all", r["source_dataset"]):
                groups.setdefault(g, {"prompt": [], "completion": [], "total": []})
                groups[g]["prompt"].append(p)
                groups[g]["completion"].append(c)
                groups[g]["total"].append(p + c)
        out[name] = {g: {k: {"mean": round(sum(v) / len(v), 1), "p50": pct(v, .5), "p95": pct(v, .95),
                             "p99": pct(v, .99), "max": max(v)} for k, v in d.items()} | {"n": len(d["total"])}
                     for g, d in groups.items()}
    (C.OUT_DIR / "token_audit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
