"""Re-execute every log entry cited by a NO verdict or an adjudication (plan §E.7.5); read-only.

    python -I -B scripts/ai_review_panel_v1/respot.py <phase>   -> outputs/ai_review_panel_v1/<phase>/respot.json

A cited entry is 'identical' if the re-run result hash equals the logged hash, 'order_only' if the same rows come back
in another order (unsorted $group output), 'differs' otherwise, 'missing' if the cited id is not in the agent's log.
"""
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import ro_exec as RX  # noqa: E402

PANEL = ROOT / "outputs/ai_review_panel_v1"


def run(entry):
    k, q = entry["kind"], entry["query"]
    if k == "sql":
        con = sqlite3.connect(f"file:{(RX.SPIDER_DB / entry['target'] / (entry['target'] + '.sqlite')).as_posix()}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode("utf-8", errors="replace")
        return RX.sql_rows(con, q)
    if k == "sql_cx":
        src = sqlite3.connect(f"file:{(RX.SPIDER_DB / entry['target'] / (entry['target'] + '.sqlite')).as_posix()}?mode=ro", uri=True)
        mem = sqlite3.connect(":memory:")
        src.backup(mem)
        mem.text_factory = lambda b: b.decode("utf-8", errors="replace")
        mem.executescript(q["setup"])
        return RX.sql_rows(mem, q["query"])
    if k == "mql":
        RX.forbid_writes(q)
        docs = list(RX.mongo()[entry["target"]].aggregate(q))
    else:
        RX.forbid_writes(q)
        docs = list(RX.mongo().aggregate(q))
    cols = []
    for d in docs:
        for c in d:
            if c not in cols:
                cols.append(c)
    return {"columns": cols if k == "mql" else sorted({c for d in docs for c in d}), "rows": RX.jsonable(docs)}


def main(phase):
    res, detail = Counter(), []
    for b in sorted((PANEL / phase / "work").glob("batch_*")):
        for agent_dir in sorted(p for p in b.iterdir() if p.is_dir()):
            out = agent_dir / "output.jsonl"
            if not out.exists():
                continue
            log = {}
            lp = agent_dir / "exec_log.jsonl"
            if lp.exists():
                for x in lp.read_text(encoding="utf-8").splitlines():
                    if x.strip():
                        e = json.loads(x)
                        log[e["id"]] = e
            for line in out.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if agent_dir.name != "agent_4_adj" and "NO" not in (str(r.get("source_sql_correct")).upper(),
                                                                  str(r.get("mql_translation_correct")).upper()):
                    continue
                tag = next(iter(log)).split("-")[0] if log else None
                for cid in sorted(set(re.findall(r"[A-Za-z0-9_]+-\d{4}", json.dumps(r.get("execution_evidence", []))))):
                    if cid.startswith("T-") and tag:  # deviation DEV-1: placeholder form T-n == <agent tag>-n
                        cid = f"{tag}-{cid[2:]}"
                    e = log.get(cid)
                    if e is None:
                        res["missing"] += 1
                        detail.append({"batch": b.name, "agent": agent_dir.name, "id": cid, "status": "missing"})
                        continue
                    if e.get("error"):
                        res["logged_error"] += 1
                        continue
                    try:
                        got = run(e)
                    except Exception as ex:  # noqa: BLE001
                        res["rerun_error"] += 1
                        detail.append({"batch": b.name, "agent": agent_dir.name, "id": cid, "status": f"rerun_error {ex!r}"[:200]})
                        continue
                    h = hashlib.sha256(json.dumps(got, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
                    if h == e["result_sha256"]:
                        st = "identical"
                    elif len(got["rows"]) == e["row_count"] and all(x in got["rows"] for x in (e["rows_head"] or [])):
                        st = "order_only"
                    else:
                        st = "differs"
                        detail.append({"batch": b.name, "agent": agent_dir.name, "id": cid, "status": st})
                    res[st] += 1
    rep = {"phase": phase, "counts": dict(res), "problems": detail}
    (PANEL / phase / "respot.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep["counts"]), len(detail), "problems")


if __name__ == "__main__":
    main(sys.argv[1])
