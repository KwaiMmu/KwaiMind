from __future__ import annotations

# Run both examples from the repository root after installing the package.
# examples/manifest.json contains:
#   0000: Lego style transfer using examples/images/input_Lego.png.
#   0001: Generate a model wearing the input T-shirt, using the original
#         Chinese prompt and examples/images/input.png as the reference.
# python examples/infer.py \
#   --checkpoint /path/to/KwaiMind/checkpoint \
#   --manifest examples/manifest.json \
#   --output-dir outputs/example
# Results are saved as outputs/example/0000.png and outputs/example/0001.png.

import argparse
from pathlib import Path

from PIL import Image

from kwaimind.data import load_manifest
from kwaimind.pipeline import KwaiMindPipeline


def is_complete_png(path: Path) -> bool:
    """True when `path` holds a PNG that Pillow can decode end to end.

    A run killed mid-save leaves a truncated or zero-byte file behind. Treating that as a
    finished result would silently drop the case from the benchmark, so verify the pixels
    decode rather than trusting that the file exists.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.load()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, action="append",
                        help="Manifest path; repeat to run several manifests from one model load")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory; with multiple manifests this is the parent and each "
                             "manifest writes into a subdirectory named after its file stem")
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pixels", type=int, default=1_440_000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--rounding", choices=["nearest", "floor"], default="nearest",
                        help="How output sizes are aligned to a multiple of 16 (default: nearest)")
    parser.add_argument("--single-image-mode", choices=["list", "pil"], default="list",
                        help="How a lone reference image is passed to DiffSynth (default: list)")
    parser.add_argument("--no-auto-resize", dest="auto_resize", action="store_false",
                        help="Disable edit_image_auto_resize (enabled by default)")
    parser.add_argument("--no-zero-cond-t", dest="zero_cond_t", action="store_false",
                        help="Disable zero_cond_t (enabled by default)")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world-size")

    multi = len(args.manifest) > 1
    jobs = []
    for manifest in args.manifest:
        records = load_manifest(manifest, args.data_root)[args.rank::args.world_size]
        directory = Path(args.output_dir) / Path(manifest).stem if multi else Path(args.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        jobs.append((manifest, records, directory))

    pipe = KwaiMindPipeline.from_pretrained(args.checkpoint, device=args.device)
    failures = []
    total = 0
    for manifest, records, directory in jobs:
        total += len(records)
        for record in records:
            output = directory / f"{record['id']}.png"
            if is_complete_png(output) and not args.overwrite:
                continue
            try:
                image = pipe(
                    record["prompt"],
                    record["images"],
                    seed=args.seed,
                    steps=args.steps,
                    width=record.get("width"),
                    height=record.get("height"),
                    max_pixels=args.max_pixels,
                    cfg_scale=args.cfg_scale,
                    negative_prompt=args.negative_prompt,
                    rounding=args.rounding,
                    single_image_mode=args.single_image_mode,
                    edit_image_auto_resize=args.auto_resize,
                    zero_cond_t=args.zero_cond_t,
                )
                # Save to a temporary name first so an interrupted write never leaves a
                # half-written PNG that a later run would mistake for a finished case.
                staging = output.with_suffix(".png.partial")
                image.save(staging, format="PNG")
                staging.replace(output)
                print(f"{Path(manifest).stem}/{record['id']}: {image.width}x{image.height}", flush=True)
            except Exception as error:
                failures.append((Path(manifest).stem, record["id"], str(error)))
                print(f"Failed {Path(manifest).stem}/{record['id']}: {error}", flush=True)
    if failures:
        for task, case_id, message in failures:
            print(f"FAILED {task}/{case_id}: {message}")
        raise RuntimeError(f"Inference failed for {len(failures)} of {total} samples")


if __name__ == "__main__":
    main()
