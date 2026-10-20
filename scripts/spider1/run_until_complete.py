"""Re-run run_eval.py until every dataset id has a non-provider_error record.

run_eval.py is already resumable (finished ids are skipped, provider errors are re-run) and
stops on a provider's per-day quota. This wrapper only waits and resumes, so a free-tier run
can finish across quota windows without manual restarts and without duplicate requests.
It never calls a model itself.

Usage:
  python scripts/spider1/run_until_complete.py --config configs/evaluation_spider1_expanded.yaml \
      --provider frontier_groq --label gpt-oss-120b [--wait-s 1800] [--max-rounds 200]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spider1_kit as K  # noqa: E402


def missing(cfg, label):
    ids = [r["id"] for r in K.read_jsonl(K.ROOT / cfg["dataset"])]
    p = K.ROOT / cfg["output_dir"] / label / "executions.jsonl"
    done = {r["id"] for r in K.read_jsonl(p) if r["result"]["category"] != "provider_error"} if p.exists() else set()
    return [i for i in ids if i not in done]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--provider", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--wait-s", type=int, default=1800)
    ap.add_argument("--max-rounds", type=int, default=200)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    for rnd in range(1, a.max_rounds + 1):
        todo = missing(cfg, a.label)
        print(f"round {rnd}: {len(todo)} ids without a final record", flush=True)
        if not todo:
            print("complete")
            return
        subprocess.run([sys.executable, str(HERE / "run_eval.py"), "--config", a.config,
                        "--provider", a.provider, "--label", a.label], check=False)
        left = missing(cfg, a.label)
        if not left:
            print("complete")
            return
        if len(left) < len(todo):
            continue  # progress was made (e.g. stopped on quota mid-way): retry at once, run_eval waits per request
        print(f"no progress ({len(left)} left); sleeping {a.wait_s}s before resuming", flush=True)
        time.sleep(a.wait_s)
    raise SystemExit(f"gave up after {a.max_rounds} rounds; {len(missing(cfg, a.label))} ids incomplete")


if __name__ == "__main__":
    main()
