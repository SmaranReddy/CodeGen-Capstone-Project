"""MQL-0010 post-hoc configuration check of a verified export (read-only, no DB access).

Recomputes every pass fingerprint from the FROZEN package (model id, revision, dtype, decoding, max_new_tokens, seed,
batch size, package manifest, config hash, registration index) and compares it with the export's pass_config.json and
with every prediction record. First-pass and delta payloads are recomputed from the frozen plans; retry payloads are
recomputed from the per-record retry prompt hashes. Also checks R5 generic text, retry-id lists and batch sizes.
Usage: python -I -B scripts/nl2mql_mql0010/analysis/posthoc_config_check.py <export_root> <out_json>
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "kaggle/nl2mql_mql0010_holdout"
sys.path[:0] = [str(PKG / "src"), str(ROOT / "scripts/nl2mql"), str(ROOT / "scripts")]
import mql0010_core as C
import recovery as RC
import scoring as SC

ex, outp = Path(sys.argv[1]), Path(sys.argv[2])
P = C.load_package(PKG)
plans = {a: C.read_jsonl(PKG / f"data/plans/plan_{a}.jsonl") for a in C.PLAN_ARMS}
recs = {a: {r["id"]: r for r in C.read_jsonl(ex / "predictions" / a / "predictions.jsonl")} for a in SC.ARMS}
ids = P["ids"]
a4 = json.loads((PKG / "registration/amendment_A4_2026-11-19.json").read_text(encoding="utf-8"))
res, bad = {}, []
def chk(name, ok):
    res[name] = bool(ok)
    if not ok: bad.append(name)

pf = json.loads((ex / "preflight/preflight_session1.json").read_text(encoding="utf-8"))
chk("preflight.config_sha256 == frozen cfg", pf["config_sha256"] == P["cfg_sha256"])
chk("preflight.registration_index == frozen", pf["registration_index_sha256"] == P["cfg"]["registration_index_sha256"])
m = P["cfg"]["model"]
chk("cfg model == A4 inference", m["id"] == a4["inference"]["model"] and m["revision"] == a4["inference"]["revision"]
    and m["max_new_tokens"] == a4["inference"]["max_new_tokens"] and m["seed"] == a4["inference"]["seed"]
    and m["decoding"].split(" ")[0] == a4["inference"]["decoding"])   # "greedy (do_sample=False)" vs "greedy"
chk("R5 generic text verbatim", RC.GENERIC_FEEDBACK == a4["R5_generic_retry_control"]["generic_feedback_text"])

passes = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in (ex / "passes").glob("*/pass_config.json")}
for arm in ("A0", "D0", "E"):
    pc = passes[f"first_{arm}"]
    payload = C.sha256_text("".join(r["prompt_sha256"] + "\n" for r in plans[arm]))
    chk(f"first:{arm} payload == frozen plan", pc["payload_sha256"] == payload)
    chk(f"first:{arm} fingerprint == frozen cfg", pc["fingerprint"] == C.pass_fingerprint(P, f"first:{arm}", payload))
pe, pff = ({r["id"]: r for r in plans[a]} for a in ("E", "EF"))
dids = [i for i in ids if pe[i]["prompt_sha256"] != pff[i]["prompt_sha256"]]
pc = passes["delta"]
payload = C.sha256_text("".join(f"{i}:{pe[i]['prompt_sha256']}:{pff[i]['prompt_sha256']}\n" for i in dids))
chk("delta ids == frozen-plan diff", pc["delta_ids"] == dids)
chk("delta payload", pc["payload_sha256"] == payload)
chk("delta fingerprint", pc["fingerprint"] == C.pass_fingerprint(P, "delta", payload))
fps = {f"first:{a}": passes[f"first_{a}"]["fingerprint"] for a in ("A0", "D0", "E")}
fps["delta"] = pc["fingerprint"]
for arm, (src, mode) in SC.RETRY_SOURCES.items():
    pc = passes[f"retry_{arm}"]
    ret = [i for i in ids if recs[arm][i]["recovery"]["retried"]]
    chk(f"retry:{arm} retry_ids == retried records", pc["retry_ids"] == ret or sorted(pc["retry_ids"]) == sorted(ret))
    chk(f"retry:{arm} feedback_mode == {mode}", pc["feedback_mode"] == mode and pc["source_arm"] == src)
    rg = [recs[arm][i].get("retry_generation") or {} for i in ret]
    if all("prompt_sha256" in g for g in rg):
        order = pc["retry_ids"]
        payload = C.sha256_text("".join(f"{i}:{recs[arm][i]['retry_generation']['prompt_sha256']}\n" for i in order))
        chk(f"retry:{arm} payload recomputed from records", payload == pc["payload_sha256"])
        chk(f"retry:{arm} record fingerprints == pass", all(recs[arm][i]["retry_generation"].get("fingerprint") == pc["fingerprint"] for i in ret))
    chk(f"retry:{arm} fingerprint == frozen cfg (batch 1)", pc["fingerprint"] == C.pass_fingerprint(P, f"retry:{arm}", pc["payload_sha256"]))
# first-answer records carry the frozen fingerprint, revision and config
for a in SC.ARMS:
    rs = recs[a].values()
    chk(f"{a} records: revision", all(r.get("model_revision") == m["revision"] for r in rs))
    chk(f"{a} records: config_sha256", all(r.get("config_sha256") == P["cfg_sha256"] for r in rs))
    chk(f"{a} records: fingerprint in pass fingerprints", all(r.get("fingerprint") in fps.values() for r in rs))
eb = {}
for a in ("A0", "D0", "E"):
    eb[a] = sorted({r.get("effective_batch_size") for r in recs[a].values()}, key=str)
for a in ("EP", "EF"):
    eb[a + "_delta"] = sorted({recs[a][i].get("effective_batch_size") for i in dids}, key=str)
out = {"ok": not bad, "failed": bad, "checks": res, "effective_batch_sizes": eb, "n_delta": len(dids),
       "cfg_sha256": P["cfg_sha256"]}
outp.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
