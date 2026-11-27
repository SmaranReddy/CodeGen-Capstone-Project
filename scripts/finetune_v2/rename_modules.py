"""One-off, mechanical rename of the two sql2mql-v2 modules whose names collide with modules of
other script directories on a shared sys.path (pytest full suite):
    common.py   -> v2_common.py    (collides with scripts/rag, scripts/audit_sql2mql_v1)
    fixtures.py -> v2_fixtures.py  (collides with scripts/finetune_v1)
Only IMPORT LINES change (checked line by line); file contents are otherwise byte-identical.
Writes outputs/sql2mql_v2/rename_manifest.json with the before/after sha256 of every touched
file, so hashes recorded before the rename (pre-registration, summaries) stay traceable.
Run once, after the measurements that recorded the old hashes:
    python -B scripts/finetune_v2/rename_modules.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RENAMES = {"common": "v2_common", "fixtures": "v2_fixtures"}
IMPORT = re.compile(r"^\s*(import|from)\s")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def rewrite(line):
    if not IMPORT.match(line):
        return line
    for old, new in RENAMES.items():
        line = re.sub(rf"^(\s*import\s+){old}\b", rf"\g<1>{new}", line)
        line = re.sub(rf"^(\s*from\s+){old}(\s+import)\b", rf"\g<1>{new}\g<2>", line)
    return line


def main():
    if (HERE / "v2_common.py").exists():
        sys.exit("already renamed")
    files = sorted(HERE.glob("*.py")) + [ROOT / "tests/test_finetune_v2.py"]
    manifest = {"renames": {f"{o}.py": f"{n}.py" for o, n in RENAMES.items()}, "files": {}}
    for p in files:
        if p.name == "rename_modules.py":
            continue
        old_b = p.read_bytes()
        old = old_b.decode("utf-8").splitlines(keepends=True)
        new = [rewrite(x) for x in old]
        changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
        assert all(IMPORT.match(old[i]) for i in changed)
        new_b = "".join(new).encode("utf-8")
        target = p.with_name(RENAMES[p.stem] + ".py") if p.parent == HERE and p.stem in RENAMES else p
        manifest["files"][p.relative_to(ROOT).as_posix()] = {
            "renamed_to": target.relative_to(ROOT).as_posix() if target != p else None,
            "sha256_before": sha(old_b), "sha256_after": sha(new_b),
            "changed_lines": [old[i].strip() + "  ->  " + new[i].strip() for i in changed]}
        target.write_bytes(new_b)
        if target != p:
            p.unlink()
    out = ROOT / "outputs/sql2mql_v2/rename_manifest.json"
    out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["renamed_to"], len(v["changed_lines"])) for k, v in manifest["files"].items()
                      if v["changed_lines"] or v["renamed_to"]}, indent=1))


if __name__ == "__main__":
    main()
