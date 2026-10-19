"""Dataset-specific preprocessing; refuses to overwrite an existing output."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["acdc", "camus", "chaos", "cardiacuda"], required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--spacing", nargs="+", type=float, help="Explicit spacing override; default preserves the archived script")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--raw-chaos", action="store_true", help="CHAOS MR/<id>/T2SPIR DICOM+PNG input; requires dcm2niix")
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        p.error("Use a new output directory to avoid overwriting existing data")
    if not Path(args.root).is_dir():
        p.error("Input root must be an existing directory")
    dims = 2 if args.dataset == "camus" else 3
    if args.spacing and (args.dataset == "cardiacuda" or len(args.spacing) != dims or min(args.spacing) <= 0):
        p.error("Spacing requires 2 values for CAMUS, 3 for ACDC/CHAOS; CardiacUDA uses pixel resizing")
    output.mkdir(parents=True)
    spacing = args.spacing or ([0.5, 0.5] if args.dataset == "camus" else [1.25, 1.25, 7.7])
    if args.dataset == "acdc":
        from memseg_agent.preprocessing.acdc import preprocess_acdc
        preprocess_acdc(args.root, str(output), crop_hw=(args.crop, args.crop), new_spacing=spacing)
    elif args.dataset == "camus":
        from memseg_agent.preprocessing.camus import preprocess_camus_2d
        preprocess_camus_2d(args.root, str(output), crop_hw=(args.crop, args.crop), new_spacing_xy=spacing)
    elif args.dataset == "chaos":
        from memseg_agent.preprocessing.chaos import preprocess_chaos
        preprocess_chaos(args.root, output, spacing, args.crop, args.raw_chaos)
    else:
        from memseg_agent.preprocessing.cardiacuda import preprocess_cardiacuda_allframe_video
        preprocess_cardiacuda_allframe_video(args.root, str(output), crop_hw=(args.crop, args.crop))
    metadata = vars(args) | {"effective_spacing": spacing if args.dataset != "cardiacuda" else "unit pixels", "annotation_centered_crop": args.dataset != "chaos"}
    (output / "preprocessing.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
