#!/usr/bin/env python3
"""Build the single-file installer for the pod: setup.sh (source form) + src/  ->  dist/setup.sh.

In the source form every embedded file is a stdin redirect with a tag, for example
    cat > "$APP/common.py" < "$SRC/app/common.py"  # bundle:heredoc __END_OF_COMMON_PY__
and the bundler turns it back into the quoted heredoc the pod file has always used:
    cat > "$APP/common.py" <<'__END_OF_COMMON_PY__'
    <src/app/common.py byte for byte>
    __END_OF_COMMON_PY__
Lines ending in "# bundle:source-only" (the SRC= line and its guard) are left out of the bundle.

    python3 tools/bundle.py            write dist/setup.sh
    python3 tools/bundle.py --check    fail if dist/setup.sh is not what the sources produce
"""
import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAG = re.compile(r'^(?P<prefix>.*)< "\$SRC/(?P<path>[^"]+)"  # bundle:heredoc (?P<delim>[A-Za-z0-9_]+)$')
SOURCE_ONLY = "# bundle:source-only"


def tagged_files(setup_text):
    """[(delimiter, path relative to src/)] in file order."""
    out = []
    for line in setup_text.splitlines():
        m = TAG.match(line)
        if m:
            out.append((m["delim"], m["path"]))
    return out


def bundle(setup_path=ROOT / "setup.sh", src_dir=ROOT / "src"):
    """Return the bundled installer as bytes (UTF-8, LF line ends, exactly as written)."""
    text = Path(setup_path).read_bytes().decode("utf-8")
    out, seen = [], set()
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        if body.endswith(SOURCE_ONLY):
            continue
        m = TAG.match(body)
        if not m:
            if "$SRC" in body:
                raise ValueError(f"untagged $SRC reference would break the bundle: {body.strip()}")
            out.append(line)
            continue
        delim, rel = m["delim"], m["path"]
        if delim in seen:
            raise ValueError(f"heredoc delimiter used twice: {delim}")
        seen.add(delim)
        content = (Path(src_dir) / rel).read_bytes().decode("utf-8")
        if content and not content.endswith("\n"):
            raise ValueError(f"src/{rel} must end with a newline to be embedded as a heredoc")
        if any(ln == delim for ln in content.split("\n")):
            raise ValueError(f"src/{rel} contains its own delimiter line {delim}")
        out.append(f"{m['prefix']}<<'{delim}'\n{content}{delim}\n")
    return "".join(out).encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the existing output instead of writing it")
    ap.add_argument("--setup", default=str(ROOT / "setup.sh"))
    ap.add_argument("--src", default=str(ROOT / "src"))
    ap.add_argument("--out", default=str(ROOT / "dist/setup.sh"))
    a = ap.parse_args()
    data = bundle(a.setup, a.src)
    out = Path(a.out)
    if a.check:
        if not out.exists() or out.read_bytes() != data:
            print(f"{out} is out of date - run: python3 tools/bundle.py", file=sys.stderr)
            return 1
        print(f"{out} is up to date ({len(data)} bytes, "
              f"{len(tagged_files(Path(a.setup).read_text(encoding='utf-8')))} files embedded)")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.chmod(tmp, 0o755)
    os.replace(tmp, out)
    print(f"wrote {out} ({len(data)} bytes, {len(tagged_files(Path(a.setup).read_text(encoding='utf-8')))} files embedded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
