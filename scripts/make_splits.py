"""Export fixed patient identities for the five outer folds."""
import argparse
import json
from pathlib import Path
from memseg_agent.data import NiftiSliceDataset
from memseg_agent.protocol import get_split, split_dict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    dataset = NiftiSliceDataset(args.data_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        path = out / f"fold_{fold + 1}.json"
        with path.open("x") as f:
            json.dump(split_dict(get_split(dataset, fold, args.seed)), f, indent=2)


if __name__ == "__main__":
    main()
