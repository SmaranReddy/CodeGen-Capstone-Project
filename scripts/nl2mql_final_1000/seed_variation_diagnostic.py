"""MQL-0015 post-draw DIAGNOSTIC only (never used to choose the sample): structure-label deviation of the registered
seed vs 200 alternative seeds under the identical frozen method. Usage: python -I -B <this> <repo root>."""
import sys, json, importlib.util, contextlib, io
root = sys.argv[1]
spec = importlib.util.spec_from_file_location("s", root + "/scripts/nl2mql_final_1000/sample_final_1000.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
cfg = json.loads(S.CONFIG.read_text(encoding="utf-8"))
labs = ("having", "group_by", "count", "union", "intersect", "nested", "join")
dev = {l: [] for l in labs}; obs = None
for k in range(201):
    c = dict(cfg); c["seed"] = cfg["seed"] if k == 0 else 1000 + k
    recs, meta, *_r, sel, unused = S.draw(c)
    full = {l: sum(l in meta[i]["structure_labels"] for i in meta) for l in labs}
    d = {l: sum(l in meta[i]["structure_labels"] for i in sel) - full[l] * 1000 / 2147 for l in labs}
    if k == 0: obs = d
    else:
        for l in labs: dev[l].append(d[l])
for l in labs:
    xs = dev[l]; sd = (sum(x * x for x in xs) / len(xs)) ** .5
    print(f"{l:10s} registered {obs[l]:+6.1f}  | 200 alt seeds: rms dev {sd:5.1f}, share |dev|>=|obs| {sum(abs(x) >= abs(obs[l]) for x in xs) / len(xs):.3f}")
