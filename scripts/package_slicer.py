"""Build an allowlisted source distribution for a future Hugging Face release.

No credentials, model weights, user cases, or remote writes are involved.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output_dir):
    output_dir = Path(output_dir).resolve()
    package = output_dir / "MemSeg-Agent-Slicer"
    archive_path = output_dir / "MemSeg-Agent-Slicer-source.zip"
    if package.exists() or archive_path.exists():
        raise FileExistsError("Use a new output directory; release files are not overwritten")
    files = {
        ROOT / "pyproject.toml",
        ROOT / "LICENSE",
        ROOT / "scripts/package_slicer.py",
        ROOT / "tests/test_interactive.py",
        ROOT / "tests/test_slicer_adapter.py",
    }
    files.update((ROOT / "src/memseg_agent").rglob("*.py"))
    files.update(
        path
        for path in (ROOT / "src/memseg_agent/interactive/web").iterdir()
        if path.suffix in (".html", ".css", ".js")
    )
    files.update(
        path
        for path in (ROOT / "plugins/slicer").rglob("*")
        if path.name == "CMakeLists.txt" or path.suffix in (".py", ".md")
    )
    files.add(ROOT / "release/huggingface/README.md")
    # Refuse symlinks so an allowlisted path cannot include an external file.
    for source in files:
        if not source.is_file() or source.resolve() != source.absolute():
            raise ValueError(f"Expected a regular source file: {source.relative_to(ROOT)}")
    package.mkdir(parents=True)
    for source in sorted(files):
        target = package / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.copyfile(ROOT / "release/huggingface/README.md", package / "README.md")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    hashes = {
        str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "format": "memseg-slicer-source-v1",
        "source_commit": revision.stdout.strip() or None,
        "source_dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "sha256": hashes,
    }
    (package / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(output_dir)))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    (output_dir / "SHA256SUMS").write_text(f"{digest}  {archive_path.name}\n", encoding="utf-8")
    return {
        "directory": str(package),
        "archive": str(archive_path),
        "files": len(hashes),
        "sha256": digest,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output_dir), indent=2))
