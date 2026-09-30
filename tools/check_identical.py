#!/usr/bin/env python3
"""Compare the repository with the single-file setup.sh 2.0.3 (the Phase 1 acceptance check).

  A  every src/ file vs the matching heredoc body in the reference
  B  dist/setup.sh vs the reference file
  C  the trees written by `--code-only`: reference vs setup.sh (source form) vs dist/setup.sh,
     all into the same WS path, one after the other

Source form and dist must always produce the same tree (the bundle is correct). With --strict (Phase 1)
A, B and the reference tree must be identical too; later phases change files on purpose, and then this
tool just reports which files differ from 2.0.3.

    python3 tools/check_identical.py [--strict] [--ref-commit 3881587 | --ref-file PATH]
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle import ROOT  # noqa: E402
from extract import split  # noqa: E402

REF_COMMIT = "3881587"
REF_SHA256 = "75431821d8cf141795ccc39cafff26e0859aa6fc60db97ca36c0ff5cac043051"   # setup.sh 2.0.3


def reference(a):
    if a.ref_file:
        data = Path(a.ref_file).read_bytes()
    else:
        data = subprocess.run(["git", "-C", str(ROOT), "show", f"{a.ref_commit}:setup.sh"], check=True,
                              capture_output=True).stdout
    sha = hashlib.sha256(data).hexdigest()
    if a.ref_commit == REF_COMMIT and not a.ref_file and sha != REF_SHA256:
        raise SystemExit(f"reference {a.ref_commit}:setup.sh has sha256 {sha}, expected {REF_SHA256}")
    return data, sha


def code_only_tree(script, ws):
    """Run `bash script --code-only` with WS=ws and return {relative path: (sha256, executable)}."""
    if ws.exists():
        shutil.rmtree(ws)
    env = dict(os.environ, WS=str(ws))
    r = subprocess.run(["bash", str(script), "--code-only"], env=env, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"{script} --code-only failed ({r.returncode}):\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    tree = {}
    for p in sorted(ws.rglob("*")):
        rel = p.relative_to(ws)
        if p.is_file() and rel.parts[0] != "logs" and "__pycache__" not in rel.parts:
            tree[str(rel)] = (hashlib.sha256(p.read_bytes()).hexdigest(), os.access(p, os.X_OK))
    return tree


def diff_trees(a, b):
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k) != b.get(k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="everything must equal the reference (Phase 1)")
    ap.add_argument("--ref-commit", default=REF_COMMIT)
    ap.add_argument("--ref-file")
    a = ap.parse_args()
    ref, ref_sha = reference(a)
    fail, ref_diff = False, False

    # A: src/ files vs heredoc bodies
    _, bodies, order = split(ref.decode("utf-8"))
    same, changed, missing = [], [], []
    for _, rel in order:
        p = ROOT / "src" / rel
        if not p.exists():
            missing.append(rel)
        elif p.read_bytes() == bodies[rel].encode("utf-8"):
            same.append(rel)
        else:
            changed.append(rel)
    new = sorted(str(p.relative_to(ROOT / "src")) for p in (ROOT / "src").rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts and str(p.relative_to(ROOT / "src")) not in bodies)
    print(f"A  src/ vs heredoc bodies of the reference ({ref_sha[:12]}): {len(same)}/{len(order)} identical")
    for rel in changed:
        print(f"   changed: src/{rel}")
    for rel in missing:
        print(f"   missing: src/{rel}")
    for rel in new:
        print(f"   new:     src/{rel}")
    ref_diff |= bool(changed or missing or new)

    # B: dist/setup.sh vs reference
    dist = ROOT / "dist/setup.sh"
    dist_same = dist.exists() and dist.read_bytes() == ref
    print(f"B  dist/setup.sh vs reference: {'identical' if dist_same else 'DIFFERENT'}"
          f" (sha256 {hashlib.sha256(dist.read_bytes()).hexdigest()[:12] if dist.exists() else '-'})")
    ref_diff |= not dist_same

    # C: generated trees
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref_script = tmp / "setup-reference.sh"
        ref_script.write_bytes(ref)
        ws = tmp / "ws"
        trees = {name: code_only_tree(script, ws) for name, script in
                 (("reference", ref_script), ("source form", ROOT / "setup.sh"), ("dist", dist))}
    d_src_dist = diff_trees(trees["source form"], trees["dist"])
    d_ref_dist = diff_trees(trees["reference"], trees["dist"])
    print(f"C  --code-only trees: {len(trees['dist'])} files; source form vs dist: "
          f"{'identical' if not d_src_dist else f'{len(d_src_dist)} DIFFERENT'}; reference vs dist: "
          f"{'identical' if not d_ref_dist else f'{len(d_ref_dist)} different'}")
    for k in d_src_dist:
        print(f"   source form != dist: {k}")
    for k in d_ref_dist:
        print(f"   reference != dist:   {k}")
    fail |= bool(d_src_dist)
    ref_diff |= bool(d_ref_dist)

    if a.strict and ref_diff:
        fail = True
    print("RESULT:", "FAIL" if fail else "PASS", "(strict: everything identical to the reference)" if a.strict else "")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
