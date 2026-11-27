"""Convert + verify a frozen sample of Spider-train items with sql2mql-v2 (local MongoDB only).

    python -B scripts/finetune_v2/run_v2.py dev            # the 500 DEV ids (development use)
    python -B scripts/finetune_v2/run_v2.py pilot --i-understand-this-is-the-pilot

`dev` refuses any pilot id. `pilot` is for Phase 2 only (after the Phase 1 gate) and refuses
to overwrite an existing pilot output. Output (per sample):
  outputs/sql2mql_v2/<sample>/records.jsonl   one evidence record per item
  outputs/sql2mql_v2/<sample>/summary.json    coverage / execution / equivalence, kept apart
The summary never counts execution success as correctness.
"""
import argparse
import json
import sys
import time
from collections import Counter

import v2_common as C
import schema_profile as SP
import sql2mql_v2 as M
import verifier as V


def load_sample(name):
    fr = json.loads(C.FREEZE.read_text(encoding="utf-8"))
    ids = set(fr[name]["ids"])
    if name == "dev" and ids & set(fr["pilot"]["ids"]):
        sys.exit("dev sample overlaps the pilot")
    rows = [r for r in C.read_jsonl(C.TRAIN) if r["id"] in ids]
    assert len(rows) == len(ids)
    return fr, rows


def process(rows, client, log_every=25):
    real = V.RealDbs(client)
    profs, out = {}, []
    t0 = time.time()
    for k, r in enumerate(sorted(rows, key=lambda r: (r["db_id"], r["id"])), 1):
        rec = {"id": r["id"], "db_id": r["db_id"], "hardness": r["hardness"], "question": r["question"],
               "sql": r["sql"], "converter": M.CONVERTER_VERSION}
        try:
            prof = profs.get(r["db_id"]) or profs.setdefault(r["db_id"], SP.Profile(r["db_id"]))
            mql, feats = M.convert(r["sql"], prof)
            rec.update(conversion="converted", features=feats, mql=mql)
        except M.Unsupported as e:
            rec.update(conversion="unsupported", code=e.code, detail=e.detail)
        except Exception as e:  # noqa: BLE001  (a converter crash is a bug: recorded, never accepted)
            rec.update(conversion="crash", code=type(e).__name__, detail=str(e)[:300])
        if rec["conversion"] == "converted":
            rec["verification"] = V.verify({"id": r["id"], "db_id": r["db_id"], "sql": r["sql"]}, mql, real)
        out.append(rec)
        if k % log_every == 0:
            print(f"{k}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    return out


def summarize(recs):
    n = len(recs)
    conv = [r for r in recs if r["conversion"] == "converted"]
    status = Counter(r["verification"]["status"] for r in conv)
    executed = sum(1 for r in conv if r["verification"]["checks"].get("l1_explicit_null") is not None
                   or r["verification"]["status"].startswith("l1_mismatch"))
    acc = [r for r in conv if r["verification"]["status"] == "accepted"]
    by_h = {}
    for h in sorted({r["hardness"] for r in recs}):
        hs = [r for r in recs if r["hardness"] == h]
        by_h[h] = {"items": len(hs), "converted": sum(r["conversion"] == "converted" for r in hs),
                   "accepted": sum(r.get("verification", {}).get("status") == "accepted" for r in hs)}
    feats = Counter(f for r in conv for f in r["features"])
    feat_acc = Counter(f for r in acc for f in r["features"])
    return {
        "items": n,
        "coverage": {"converted": len(conv), "rate": round(len(conv) / n, 4) if n else None,
                     "unsupported": dict(Counter(r["code"] for r in recs if r["conversion"] == "unsupported")
                                         .most_common()),
                     "crash": sum(r["conversion"] == "crash" for r in recs)},
        "execution": {"mql_executed_on_real_db": executed,
                      "note": "execution success is NOT correctness"},
        "verification": dict(status.most_common()),
        "accepted": {"n": len(acc), "of_items": round(len(acc) / n, 4) if n else None,
                     "of_converted": round(len(acc) / len(conv), 4) if conv else None},
        "by_hardness": by_h,
        "by_feature": {f: {"converted": feats[f], "accepted": feat_acc[f]} for f in sorted(feats)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample", choices=("dev", "pilot"))
    ap.add_argument("--i-understand-this-is-the-pilot", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.sample == "pilot" and not a.i_understand_this_is_the_pilot:
        sys.exit("the pilot runs only in Phase 2 (pass --i-understand-this-is-the-pilot)")
    out = C.OUT / (a.out or a.sample)
    if a.sample == "pilot" and (out / "records.jsonl").exists():
        sys.exit(f"{out} exists: refusing to overwrite pilot results")
    fr, rows = load_sample(a.sample)
    client = C.local_client()
    C.check_server(client)
    recs = process(rows, client)
    C.write_jsonl(out / "records.jsonl", recs)
    summ = summarize(recs)
    summ.update(sample=a.sample, freeze_sha256=fr["freeze_sha256"], converter=M.CONVERTER_VERSION,
                verifier=V.VERIFIER_VERSION, mongo=C.MONGO_IMAGE, built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                code_sha256={p: C.sha256_file(C.HERE / p) for p in
                             ("sql2mql_v2.py", "schema_profile.py", "verifier.py", "comparator.py",
                              "fixtures.py", "mongo_import.py", "common.py")})
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("coverage", "verification", "accepted")}, indent=1))


if __name__ == "__main__":
    main()
