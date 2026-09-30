import argparse
import csv
from pathlib import Path
import numpy as np
import tifffile

from .config import PRIVATE_SPLITS, add_object_arg, get_paths
from .data import save_png
from .predict import CHECKER, load_calib, mask_for

def get_args():
    parser = argparse.ArgumentParser()
    add_object_arg(parser)
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--splits", nargs="*", default=list(PRIVATE_SPLITS), choices=list(PRIVATE_SPLITS))
    parser.add_argument("--pixel-threshold", type=float, default=None)
    parser.add_argument("--image-threshold", type=float, default=None)
    parser.add_argument("--min-anomalous-px", type=int, default=None)
    parser.add_argument("--no-gate", dest="gate", action="store_const", const=False, default=None)
    parser.add_argument("--gate", dest="gate", action="store_const", const=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()

def load_scores(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The image score is not recoverable from the tiff -- the "
            f"pixel map and the image scalar use different aggregation on purpose -- so "
            f"gating needs the csv that pipeline.predict wrote beside the submission. "
            f"Re-run pipeline.predict, or pass --no-gate to re-threshold without it.")
    with path.open(newline="") as f:
        return {(row["split"], Path(row["name"]).stem): row for row in csv.DictReader(f)}

def pick(value, fallback):
    return value if value is not None else fallback

def restamp():
    args = get_args()
    obj_dir, out_dir = get_paths(args)
    obj = args.object_name
    calib = load_calib(args.calibration or (out_dir / "calibration.npz"))

    gate = pick(args.gate, calib["gate"])
    min_px = pick(args.min_anomalous_px, calib["min_anomalous_px"])
    pixel_thr = pick(args.pixel_threshold, calib["pixel_threshold"])
    image_thr = pick(args.image_threshold, calib["image_threshold"])
    submission = args.submission or (out_dir / "submission")

    if not np.isfinite(pixel_thr):
        raise ValueError("pixel_threshold is not finite -- rerun pipeline.evaluate without --skip-pixel")

    print(f"{obj}: pixel threshold {pixel_thr:.6g} (rule {calib['pixel_rule']})")
    if gate:
        print(f"  gating ON  image threshold {image_thr:.6g} (rule {calib['image_rule']}), "
              f"min {min_px} px forced")
    else:
        print("  gating OFF")

    scores = load_scores(args.scores or (out_dir / "image_scores.csv")) if gate else {}

    for split in args.splits:
        tiff_dir = submission / "anomaly_images" / obj / split
        png_dir = submission / "anomaly_images_thresholded" / obj / split
        files = sorted(tiff_dir.glob("*.tiff"))
        if not files:
            print(f"  {split}: no tiffs in {tiff_dir}, skipped")
            continue
        png_dir.mkdir(parents=True, exist_ok=True)

        flagged = forced = empty = 0
        area = 0.0
        for path in files:
            amap = tifffile.imread(path).astype(np.float32)
            anomalous = True
            if gate:
                row = scores.get((split, path.stem))
                if row is None:
                    raise KeyError(f"{path.stem} is in {tiff_dir} but not in image_scores.csv for "
                                   f"split {split}; the csv is from a different run than the tiffs")
                anomalous = float(row["image_score"]) >= image_thr
            was_empty = not (amap >= pixel_thr).any()
            binary = mask_for(amap, pixel_thr, anomalous, gate, min_px)
            empty += int(was_empty)
            if anomalous and was_empty:
                forced += 1
            if binary.any():
                flagged += 1
            area += float(binary.mean())
            if not args.dry_run:
                save_png(binary, png_dir / f"{path.stem}.png")

        n = len(files)
        print(f"  {split}: {n} images, server will read {flagged} ({100 * flagged / n:.1f}%) "
              f"as anomalous; {empty} were empty at the raw threshold, {forced} forced on; "
              f"mean predicted area {100 * area / n:.4f}%")

    if args.dry_run:
        print("\ndry run: no .png written")
    else:
        print(f"\nrewrote masks under {submission / 'anomaly_images_thresholded' / obj}")
        print(f"The tiffs are untouched. Re-run MVTec's pre-upload checker before uploading:\n{CHECKER}{submission.resolve()}")

if __name__ == "__main__":
    restamp()
