"""S5: code freeze for sql2mql-v2.1 / verifier 3.1 / TC-1 (plan §E.1).

    python -I -B scripts/finetune_v2/freeze_v2_1_code.py freeze   -> data/interim/finetune_v2/v2_1_code_freeze.json (refuses to overwrite)
    python -I -B scripts/finetune_v2/freeze_v2_1_code.py check    -> exit 1 if any pinned file changed

Pins every new module plus the frozen v2 modules (the 19 of round3_code_freeze.json) they import, the TC-1
manifests, and the round-3 freeze file itself.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data/interim/finetune_v2/v2_1_code_freeze.json"
NEW = ["scripts/finetune_v2/sql2mql_v2_1.py", "scripts/finetune_v2/fixtures3_1.py", "scripts/finetune_v2/verifier3_1.py",
       "scripts/finetune_v2/type_contract.py", "scripts/nl2mql/atlas_type_contract_check.py",
       "scripts/finetune_v2/freeze_v2_1_code.py"]
DATA = ["data/interim/finetune_v2/type_contract_v1.json", "data/interim/finetune_v2/type_contract_eval_v1.json",
        "data/interim/finetune_v2/round3_code_freeze.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pinned():
    r3 = json.loads((ROOT / "data/interim/finetune_v2/round3_code_freeze.json").read_text(encoding="utf-8"))
    frozen = [k if "/" in k else f"scripts/finetune_v2/{k}" for k in r3["code_sha256"]]
    return {"new_modules": {p: sha(ROOT / p) for p in NEW},
            "frozen_v2_modules": {p: sha(ROOT / p) for p in frozen},
            "data": {p: sha(ROOT / p) for p in DATA}}


def main():
    cmd = sys.argv[1]
    if cmd == "freeze":
        if OUT.exists():
            sys.exit(f"refusing to overwrite {OUT}")
        r3 = json.loads((ROOT / "data/interim/finetune_v2/round3_code_freeze.json").read_text(encoding="utf-8"))
        cur = pinned()
        bad = [k for k, h in r3["code_sha256"].items()
               if cur["frozen_v2_modules"][k if "/" in k else f"scripts/finetune_v2/{k}"] != h]
        if bad:
            sys.exit(f"frozen v2 modules differ from round3_code_freeze.json: {bad}")
        doc = {"written_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "converter": "sql2mql-v2.1",
               "verifier": "sql2mql-v2-verifier-3.1", "contract": "TC-1", **cur}
        OUT.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)} sha256={sha(OUT)}")
        return
    doc = json.loads(OUT.read_text(encoding="utf-8"))
    cur = pinned()
    diff = [p for g in ("new_modules", "frozen_v2_modules", "data") for p, h in doc[g].items() if cur[g].get(p) != h]
    print(json.dumps({"ok": not diff, "changed": diff, "freeze_sha256": sha(OUT)}, indent=1))
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
