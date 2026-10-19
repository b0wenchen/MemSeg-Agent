"""Run all folds with full, 10%, and 30% of TRAINING patients (not all patients)."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    p.add_argument("--fractions", nargs="+", type=float, default=[1.0, 0.1, 0.3])
    p.add_argument("--memory-mode", choices=["pseudo_observation", "direct_tokens"], default="pseudo_observation")
    p.add_argument("--split-dir", help="Directory produced by make_splits.py")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    for fold in args.folds:
        for fraction in args.fractions:
            cmd = [sys.executable, str(Path(__file__).with_name("train_static.py")), "--config", args.config, "--fold", str(fold), "--train-fraction", str(fraction), "--memory-mode", args.memory_mode]
            if args.split_dir:
                cmd += ["--split-manifest", str(Path(args.split_dir) / f"fold_{fold+1}.json")]
            print(cmd, flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
