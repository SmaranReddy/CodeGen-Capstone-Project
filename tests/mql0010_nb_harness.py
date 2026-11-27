"""Execute the MQL-0010 Kaggle notebook cells locally with test hooks (fake model, fake T4 probe, pinned tokenizer).

Run with a Python that has `transformers` (tokenizer only; no torch), e.g. ~/codegen_data/venv_tok:
  python -I -B tests/mql0010_nb_harness.py --pkg PKG --tok TOKDIR --out OUT --session 1 [--resume PREV] [--stop-after N]
Prints one JSON line with the export status. No model, no GPU, no network, no database.
"""
import argparse
import hashlib
import json
import re
import sys


def fake_factory(cfg):
    def h(s):
        return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:8], 16)

    def run_batch(batch):
        out = []
        for p in batch:
            user = p["messages"][0]["content"]
            colls = re.findall(r"^- (\w+__\w+)\s*$", user, flags=re.M)
            coll = colls[0] if colls else "x__y"
            if len(p["messages"]) == 1:
                bad = h(p["system"] + user) % 100 < 25
                stage = {"$match": {"__missing_field__": 1}} if bad else {"$match": {}}
                txt = json.dumps({"collection": coll, "pipeline": [stage, {"$project": {"_id": 0}}]})
            else:
                fb = p["messages"][2]["content"]
                ok = h(user + fb) % 100 < (70 if "does not exist" in fb else 35)
                stage = {"$match": {}} if ok else {"$match": {"__missing_field__": 2}}
                txt = json.dumps({"collection": coll, "pipeline": [stage, {"$limit": 5}, {"$project": {"_id": 0}}]})
            out.append((txt, {"latency_s": 0.0, "input_tokens": len(user) // 4, "output_tokens": len(txt) // 4,
                              "finish_reason": "stop"}, None))
        return out
    info = {"dtype": "bfloat16", "gpu": "Tesla T4", "torch": "fake-2.11", "transformers": "fake-5.20"}
    return run_batch, info, (lambda e: False), (lambda: None)


FAKE_GPU = {"checked": True, "visible_devices": 1, "device0": "Tesla T4", "compute_capability": "7.5", "total_gib": 14.56,
            "bf16_supported_incl_emulation": True, "bf16_native": False, "free_gib": 14.4, "cuda": "fake", "torch": "fake"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--nb", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--resume")
    ap.add_argument("--stop-after", type=int)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tok)
    nb = json.loads(open(a.nb, encoding="utf-8").read())
    g = {"__name__": "__main__"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        exec("".join(cell["source"]), g)
        if cell["metadata"]["tags"] == ["config"]:
            g.update(SESSION=a.session, PKG_DIR=a.pkg, OUT_DIR=a.out, OUTPUT_ZIP=a.out + ".zip", INPUT_ROOT=a.out + "_input",
                     RESUME_FROM=a.resume, GENERATOR_FACTORY=fake_factory, FAKE_GPU=FAKE_GPU,
                     COUNT_TOKENS=lambda s: len(tok.encode(s, add_special_tokens=False)), STOP_AFTER_BATCHES=a.stop_after)
    print("HARNESS_RESULT " + json.dumps({"status": g["ST"]["status"], "passes": g["MAN"]["passes"], "zip": a.out + ".zip",
                                          "verify_exports": g["MAN"]["verify_exports"], "stopped": g["STOPPED"]}))


if __name__ == "__main__":
    sys.exit(main())
