"""Convert + verify with sql2mql-v2 (converter unchanged) and verifier-3 (local MongoDB only).

    python -B scripts/finetune_v2/run_v3.py dev3      re-verify DEV items with verifier-3:
        every DEV item the converter converts, EXCEPT the iteration-2 G4 holdout labels
        (never re-used). Measures verifier-3's extra rejections against verifier-2 on the
        same MQL (false-reject cost) and checks the superset property. -> outputs/sql2mql_v2/dev3/
    python -B scripts/finetune_v2/run_v3.py round3    the pre-registered round-3 sample (740
        records). Refuses to run unless stage 1 and stage 2 of the pre-registration exist and
        every frozen code hash matches. -> outputs/sql2mql_v2/round3/

Records are appended as they finish; a re-run resumes (a memory-pressure kill loses no work).
The pilot ids are refused in both modes.
"""
import argparse
import json
import sys
import time
from collections import Counter

import v2_common as C
import schema_profile as SP
import sql2mql_v2 as M
import verifier3 as V3
import round3_stats as S
import mutant_study_v3 as MS
import freeze_round3_code as FZ


def convert(r, profs):
    rec = {"id": r["id"], "db_id": r["db_id"], "hardness": r["hardness"], "question": r["question"],
           "sql": r["sql"], "stratum": S.stratum(r["sql"]), "converter": M.CONVERTER_VERSION}
    try:
        prof = profs.get(r["db_id"]) or profs.setdefault(r["db_id"], SP.Profile(r["db_id"]))
        mql, feats = M.convert(r["sql"], prof)
        rec.update(conversion="converted", features=feats, mql=mql)
    except M.Unsupported as e:
        rec.update(conversion="unsupported", code=e.code, detail=e.detail)
    except Exception as e:  # noqa: BLE001
        rec.update(conversion="crash", code=type(e).__name__, detail=str(e)[:300])
    return rec


def run(rows, out, real, profs, extra=None):
    path = out / "records.jsonl"
    prev, cut = MS.load_jsonl_tolerant(path)
    if cut:
        path.write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in prev), encoding="utf-8")
    if prev or cut:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "resume_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "records_kept": len(prev),
                                "cut_trailing_line_dropped": cut}) + "\n")
    done = {x["id"] for x in prev}
    todo = sorted((r for r in rows if r["id"] not in done), key=lambda r: (r["db_id"], r["id"]))
    t0 = time.time()
    for k, r in enumerate(todo, 1):
        rec = convert(r, profs)
        if rec["conversion"] == "converted":
            rec["verification"] = V3.verify({"id": r["id"], "db_id": r["db_id"], "sql": r["sql"]},
                                            rec["mql"], real)
        if extra:
            extra(rec)
        out.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        if k % 20 == 0 or k == len(todo):
            print(f"{k}/{len(todo)} ({len(done)} earlier)  {time.time() - t0:.0f}s", flush=True)
    return C.read_jsonl(path)


def summarize(recs, N=None, n=None):
    def block(rs):
        conv = [r for r in rs if r["conversion"] == "converted"]
        st = Counter(r["verification"]["status"] for r in conv)
        return {"records": len(rs), "converted": len(conv), "accepted": st.get("accepted", 0),
                "verification": dict(st.most_common()),
                "unsupported": dict(Counter(r["code"] for r in rs if r["conversion"] == "unsupported").most_common()),
                "crash": sum(r["conversion"] == "crash" for r in rs)}
    out = {"all": block(recs), "by_stratum": {h: block([r for r in recs if r["stratum"] == h]) for h in S.STRATA}}
    if N and n:
        # population estimates (stratified expansion); planning figures, not gate figures
        out["estimated_population_acceptance"] = round(
            sum(N[h] * out["by_stratum"][h]["accepted"] / n[h] for h in S.STRATA) / sum(N.values()), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("dev3", "round3"))
    a = ap.parse_args()
    fr = json.loads(C.FREEZE.read_text(encoding="utf-8"))
    pilot = set(fr["pilot"]["ids"])
    train = {r["id"]: r for r in C.read_jsonl(C.TRAIN)}
    client = C.local_client()
    C.check_server(client)
    real = V3.RealDbs(client)
    profs = {}
    if a.mode == "dev3":
        out = C.OUT / "dev3"
        v2 = {r["id"]: r for r in C.read_jsonl(C.OUT / "dev_iter2" / "records.jsonl")}
        hold = MS.holdout2_label_ids()
        ids = [i for i in fr["dev"]["ids"] if i not in hold and v2[i]["conversion"] == "converted"]
        assert not set(ids) & pilot

        def extra(rec):
            rec["verifier2_status"] = v2[rec["id"]]["verification"]["status"]
            rec["same_mql_as_verifier2_run"] = C.compact(rec.get("mql")) == C.compact(v2[rec["id"]].get("mql"))
        recs = run([train[i] for i in ids], out, real, profs, extra)
        both = Counter((r["verifier2_status"] == "accepted", r["verification"]["status"] == "accepted")
                       for r in recs)
        summ = summarize(recs)
        summ.update({
            "mode": "dev3", "excluded_iteration2_holdout_labels": len(hold),
            "verifier": V3.VERIFIER_VERSION, "items": len(recs),
            "same_mql_as_verifier2_run": sum(r["same_mql_as_verifier2_run"] for r in recs),
            "v2_accepted_v3_accepted": both[(True, True)],
            "v2_accepted_v3_rejected (extra rejections)": both[(True, False)],
            "v2_rejected_v3_accepted (superset violation)": both[(False, True)],
            "v2_rejected_v3_rejected": both[(False, False)],
            "extra_rejection_reasons": dict(Counter(r["verification"]["status"] for r in recs
                                                    if r["verifier2_status"] == "accepted"
                                                    and r["verification"]["status"] != "accepted")),
        })
    else:
        pre = FZ.check_frozen()
        out = C.OUT / "round3"
        ids = [i for h in S.STRATA for i in pre["sampling"]["ids"][h]]
        if set(ids) & pilot or set(ids) & set(fr["dev"]["ids"]):
            sys.exit("round-3 sample overlaps pilot/dev")
        recs = run([train[i] for i in ids], out, real, profs)
        if len(recs) != len(ids):
            sys.exit("incomplete")
        summ = summarize(recs, pre["strata"]["N_h"], pre["sampling"]["n_h_records"])
        summ.update(mode="round3", verifier=V3.VERIFIER_VERSION, prereg_sha256=pre["sha256_of_body_without_this_field"])
        acc = sorted(r["id"] for r in recs if r.get("verification", {}).get("status") == "accepted")
        (out / "accepted_labels.json").write_text(json.dumps({"ids": acc}, indent=0) + "\n", encoding="utf-8")
    summ.update(converter=M.CONVERTER_VERSION, mongo=C.MONGO_IMAGE, finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                code_sha256=FZ.code_hashes())
    (out / "summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "code_sha256"}, indent=1))


if __name__ == "__main__":
    main()
