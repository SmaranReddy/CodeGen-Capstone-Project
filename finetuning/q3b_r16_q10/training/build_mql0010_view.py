"""Rebuild the fine-tuning dataset approved_v2_1_mql0010_view (MQL-0010-safe).

Inputs:
  --corpus   approved corpus v2.1 (corpus.jsonl, sha256 96552f06...)
  --sft-dir  approved_v2_1 SFT files (sft_train/sft_dev.prompt_completion.jsonl, 3,445 rows; built from the corpus
             with build_pilot_sft.py)
  --split    MQL-0010 registration/split_registration.json (from the MQL-0010 package)
Output: --out folder with sft_train (MQL-0010 build DBs), sft_dev (MQL-0010 registered dev DBs), sft_full (both),
manifest.json. Removes the 18 holdout databases (= MQL-0010's registered retrieval view, 2,912 records, ids hash checked)
and the 11 L3-L5 near-duplicate corpus records.
"""
import argparse, hashlib, json
from pathlib import Path

sha = lambda b: hashlib.sha256(b).hexdigest()
ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True); ap.add_argument("--sft-dir", required=True)
ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()

reg = json.loads(Path(a.split).read_text(encoding="utf-8"))
H, D, B = set(reg["holdout"]["dbs"]), set(reg["dev"]["dbs"]), set(reg["build_dbs"])
EXCL = sorted({p["corpus_id"] for p in reg["leakage"]["L3_L5_exclusion_pairs"]})
raw = Path(a.corpus).read_bytes()
assert sha(raw) == "96552f061afba13792d75240c18001e87e70f4aebf9d0934dfe27c0a7dd199bf", "not corpus v2.1"
corpus = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
view = [r for r in corpus if r["db_id"] not in H]
vh = sha("\n".join(r["id"] for r in view).encode())
assert len(view) == reg["holdout_retrieval_view"]["n_records"] and vh == reg["holdout_retrieval_view"]["ids_sha256"], \
    "holdout view does not match the MQL-0010 registration"

sft = {}
for f in ("sft_train", "sft_dev"):
    for l in (Path(a.sft_dir) / f"{f}.prompt_completion.jsonl").read_text(encoding="utf-8").splitlines(keepends=True):
        if l.strip():
            sft[json.loads(l)["id"]] = l if l.endswith("\n") else l + "\n"
dbof = {r["id"]: r["db_id"] for r in corpus}
assert set(sft) == set(dbof), "SFT rows don't match the corpus"

order = [r["id"] for r in view if r["id"] not in EXCL]            # corpus order
tr = [i for i in order if dbof[i] in B]
dv = [i for i in order if dbof[i] in D]
assert len(tr) + len(dv) == len(order) and not ({dbof[i] for i in order} & H)

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
files = {"sft_train.prompt_completion.jsonl": tr, "sft_dev.prompt_completion.jsonl": dv,
         "sft_full.prompt_completion.jsonl": tr + dv}
for n, ids in files.items():
    (out / n).write_text("".join(sft[i] for i in ids), encoding="utf-8")
print({n: len(ids) for n, ids in files.items()})
print({n: sha((out / n).read_bytes()) for n in files})
