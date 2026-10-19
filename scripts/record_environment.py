"""Record the actual environment and SAM2 import location used for a run."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", help="Optionally record the SAM2 weight SHA256")
    args = p.parse_args()
    packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    spec = importlib.util.find_spec("sam2")
    result = {"python": sys.version, "platform": platform.platform(), "packages": packages, "sam2_import": spec.origin if spec else None}
    if args.checkpoint:
        digest = hashlib.sha256()
        with Path(args.checkpoint).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result["checkpoint_sha256"] = digest.hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
