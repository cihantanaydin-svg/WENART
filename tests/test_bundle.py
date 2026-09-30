"""The single-file installer for the pod is generated from setup.sh + src/ and must stay in sync."""
import subprocess
import sys

import pytest

from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
from bundle import bundle, tagged_files  # noqa: E402
from extract import split  # noqa: E402


def test_dist_is_up_to_date():
    assert (ROOT / "dist/setup.sh").read_bytes() == bundle(), "run: python3 tools/bundle.py"


def test_every_tag_has_a_file_and_every_file_a_tag():
    tags = tagged_files((ROOT / "setup.sh").read_text(encoding="utf-8"))
    paths = [p for _, p in tags]
    assert len(set(paths)) == len(paths) == len({d for d, _ in tags})
    on_disk = {str(p.relative_to(ROOT / "src")) for p in (ROOT / "src").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    assert set(paths) == on_disk


def test_split_of_the_bundle_gives_back_src():
    _, files, _ = split(bundle().decode("utf-8"))
    for rel, body in files.items():
        assert (ROOT / "src" / rel).read_bytes() == body.encode("utf-8"), rel


def test_bundle_has_no_source_form_left():
    text = bundle().decode("utf-8")
    assert "$SRC" not in text and "bundle:" not in text


@pytest.mark.parametrize("content, error", [
    ("print(1)", "must end with a newline"),
    ("x = 1\n__END_OF_X__\n", "its own delimiter"),
])
def test_bundle_refuses_unsafe_content(tmp_path, content, error):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/x.py").write_text(content)
    (tmp_path / "setup.sh").write_text('cat > "$APP/x.py" < "$SRC/x.py"  # bundle:heredoc __END_OF_X__\n')
    with pytest.raises(ValueError, match=error):
        bundle(tmp_path / "setup.sh", tmp_path / "src")


def test_bundle_refuses_untagged_src(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "setup.sh").write_text('cp "$SRC/x.py" "$APP/"\n')
    with pytest.raises(ValueError, match="untagged"):
        bundle(tmp_path / "setup.sh", tmp_path / "src")


def test_bash_syntax():
    for f in ("setup.sh", "dist/setup.sh"):
        r = subprocess.run(["bash", "-n", str(ROOT / f)], capture_output=True, text=True)
        assert r.returncode == 0, f + r.stderr
