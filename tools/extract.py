#!/usr/bin/env python3
"""One-off (Phase 1): split a single-file setup.sh into src/ + a source-form setup.sh.

Every quoted heredoc line   <prefix><<'DELIM'   (body until a line equal to DELIM)
becomes                     <prefix>< "$SRC/<path>"  # bundle:heredoc DELIM
and its body is written to src/<path> byte for byte. A source-only SRC= line (+ guard) is inserted before
the logging section (before anything is written to WS). The result is verified: tools/bundle.py must rebuild the input byte for byte.

    python3 tools/extract.py <single-file setup.sh>      (writes setup.sh and src/ in the repo root)
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle import ROOT, bundle  # noqa: E402

HEREDOC = re.compile(r"^(?P<prefix>.*)<<'(?P<delim>[A-Za-z0-9_]+)'$")
STDIN_SCRIPTS = {                       # heredocs fed to "$PY -" on stdin (no target file in the line)
    "__END_OF_DL__": "installer/dl_models.py",
    "__END_OF_CFG__": "installer/write_config.py",
    "__END_OF_AGENT_DL__": "installer/dl_agent_model.py",
    "__END_OF_GEN3D_DL__": "installer/dl_gen3d.py",
}
HEADER_NOTE = (
    "# SOURCE FORM (development): needs src/ next to it. The file for the pod is the single-file dist/setup.sh,"
    "  # bundle:source-only\n"
    "# built by: python3 tools/bundle.py  (see docs/DEVELOPMENT.md)  # bundle:source-only\n"
)
SRC_LINES = (
    '# source form: the embedded files live in src/ next to this script; '
    'tools/bundle.py builds the single-file dist/setup.sh  # bundle:source-only\n'
    'SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src"  # bundle:source-only\n'
    '[[ -f "$SRC/app/common.py" ]] || { echo "src/ not found next to $0 - on a pod use the single-file '
    'dist/setup.sh"; exit 1; }  # bundle:source-only\n'
)


def target_path(prefix, delim):
    """Where a heredoc body goes inside src/."""
    if delim in STDIN_SCRIPTS:
        return STDIN_SCRIPTS[delim]
    m = re.search(r'cat > "\$(APP|WS)/([^"]+)" $', prefix)
    if not m:
        raise ValueError(f"unknown heredoc target for {delim}: {prefix!r}")
    base, name = m.groups()
    if base == "APP":
        return f"app/{name}"
    if name in ("run.sh", "start.sh", "agent.sh"):
        return f"scripts/{name}"
    if name == "cache/gen3d_build.sh":
        return "installer/gen3d_build.sh"
    raise ValueError(f"unknown heredoc target for {delim}: $WS/{name}")


def split(text):
    """-> (source-form text, {src path: body}, [(delim, path)])."""
    lines = text.splitlines(keepends=True)
    out, files, order, i = [], {}, [], 0
    while i < len(lines):
        line = lines[i]
        m = HEREDOC.match(line.rstrip("\n"))
        if line.startswith("# ---------- logging, error trap, folders"):
            out.append(SRC_LINES)
        if not m:
            out.append(line)
            if line.startswith("SETUP_VERSION="):
                out.append(HEADER_NOTE)
            i += 1
            continue
        delim = m["delim"]
        j = i + 1
        while lines[j].rstrip("\n") != delim:
            j += 1
        rel = target_path(m["prefix"], delim)
        if rel in files:
            raise ValueError(f"two heredocs write {rel}")
        files[rel] = "".join(lines[i + 1:j])
        order.append((delim, rel))
        out.append(f'{m["prefix"]}< "$SRC/{rel}"  # bundle:heredoc {delim}\n')
        i = j + 1
    if SRC_LINES not in out:
        raise ValueError("logging section not found - nowhere to put the SRC= line")
    return "".join(out), files, order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("single_file")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args()
    root = Path(a.root)
    original = Path(a.single_file).read_bytes()
    source, files, order = split(original.decode("utf-8"))
    for rel, body in files.items():
        p = root / "src" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body.encode("utf-8"))
    (root / "setup.sh").write_bytes(source.encode("utf-8"))
    if bundle(root / "setup.sh", root / "src") != original:
        print("ERROR: bundle(source form) differs from the input", file=sys.stderr)
        return 1
    print(f"extracted {len(order)} heredocs into {root / 'src'}; bundle round trip is byte-identical")
    for delim, rel in order:
        print(f"  {delim:<32} -> src/{rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
