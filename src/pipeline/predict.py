import argparse
import json
import time
from pathlib import Path
import numpy as np
import tifffile
import torch
from PIL import Image

from . import scoring
from .config import PRIVATE_SPLITS, get_paths
from .data import get_images, pick_files, read_images, save_png, write_csv
from .dump import add_run_args, get_geometry, run_tiles
from .metrics import make_masks
from .model import DTYPES, load_model
from .tiling import to_pixels

CHECKER = "python MVTecAD2_public_code_utils/check_and_prepare_data_for_upload.py"

def get_args():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument("--submission", type=Path, default=None)
    parser.add_argument("--splits", nargs="*", default=list(PRIVATE_SPLITS), choices=list(PRIVATE_SPLITS))
    parser.add_argument("--no-thresholded", action="store_true")
    return parser.parse_args()

def load_calib(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run pipeline.evaluate first. Without it there is no "
            f"calibrated threshold, and thresholding on the test distribution itself is "
            f"exactly the mistake that produced the original 265 false positives.")
    data = np.load(path, allow_pickle=False)

    def get(key, cast, default):
        return cast(data[key]) if key in data else default

    calib = {f"{k}_stats": {"mu": data[f"{k}_mu"], "sd": data[f"{k}_sd"]} for k in ("image", "pixel")}
    calib.update({f"{k}_spec": json.loads(str(data[f"{k}_spec"])) for k in ("image", "pixel")})
    calib.update({f"{k}_threshold": float(data[f"{k}_threshold"]) for k in ("image", "pixel")})
    calib.update({
        "upscale": get("upscale", int, 1),
        "image_rule": get("image_rule", str, "quantile"),
        "pixel_rule": get("pixel_rule", str, "mean+3std"),
        "gate": get("gate", bool, False),
        "min_anomalous_px": get("min_anomalous_px", int, 1),
    })
    calib["valid"] = tuple(int(v) for v in (data["valid_cells"] if "valid_cells" in data else data["image_mu"].shape))
    return calib

def mask_for(amap, thr, anomalous, gate, min_px):
    return make_masks(amap[None], thr, [anomalous] if gate else None, min_px)[0]

def check_outputs(submission, name, split, expected, native_hw, thresholded):
    tiff_dir = submission / "anomaly_images" / name / split
    files = sorted(tiff_dir.glob("*.tiff"))
    if len(files) != expected:
        print(f"  NOTE: {len(files)}/{expected} .tiff files in {tiff_dir} "
              f"(the official checker requires exactly {expected})")
    sample = tifffile.imread(files[0])
    assert sample.ndim == 2, f"{files[0]} is not single-channel (ndim={sample.ndim})"
    assert sample.dtype == np.float16, f"{files[0]} is {sample.dtype}, expected float16"
    assert sample.shape == tuple(native_hw), f"{files[0]} is {sample.shape}, expected {tuple(native_hw)}"
    print(f"  tiff ok: {sample.shape} {sample.dtype}")

    if thresholded:
        pngs = sorted((submission / "anomaly_images_thresholded" / name / split).glob("*.png"))
        arr = np.asarray(Image.open(pngs[0]))
        assert arr.ndim == 2, f"{pngs[0]} is not single-channel"
        assert set(np.unique(arr)).issubset({0, 255}), f"{pngs[0]} has values outside {{0, 255}}"
        print(f"  png ok:  {arr.shape} {arr.dtype}, {len(pngs)} files")

def predict():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj_dir, out_dir = get_paths(args)
    obj = args.object_name
    calib = load_calib(args.calibration or (out_dir / "calibration.npz"))
    pixel_thr = calib["pixel_threshold"]
    submission = args.submission or (out_dir / "submission")

    for k in ("image", "pixel"):
        print(f"{k} spec {calib[f'{k}_spec']}  threshold {calib[f'{k}_threshold']:.6g} (rule {calib[f'{k}_rule']})")
    if calib["gate"]:
        print(f"gating ON (image threshold {calib['image_threshold']:.6g}; "
              f"min {calib['min_anomalous_px']} px forced on anomalous images)")
    else:
        print("gating OFF (raw thresholded masks)")
    print(f"upscale {calib['upscale']} (from calibration.npz, not a flag)")
    if not np.isfinite(pixel_thr) and not args.no_thresholded:
        raise ValueError("pixel_threshold is not finite -- rerun pipeline.evaluate without --skip-pixel, "
                         "or pass --no-thresholded to submit anomaly maps only")

    model = load_model(ckpt=args.ckpt).to(device)
    plan = native = padded = None
    rows = []
    t0 = time.time()

    for split in args.splits:
        files = get_images(obj_dir / split)
        expected = len(files)
        files = pick_files(files, args.limit)
        tiff_dir = submission / "anomaly_images" / obj / split
        png_dir = submission / "anomaly_images_thresholded" / obj / split
        tiff_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_thresholded:
            png_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{split}: {len(files)} images -> {tiff_dir}")

        for i, img, name, condition in read_images(files, device):
            shape = (int(img.shape[1]), int(img.shape[2]))
            if plan is None:
                native = shape
                plan, padded, valid = get_geometry(native, obj, args.tile, calib["upscale"])
                if valid != calib["valid"]:
                    raise ValueError(f"grid geometry {valid} does not match calibration.npz {calib['valid']}; "
                                     f"--tile {args.tile} differs from the dump the thresholds came from")
            elif shape != native:
                raise ValueError(f"{name} is {shape}, inconsistent with the first image ({native})")

            mid, last = run_tiles(model, img, plan, args.tile, args.tile_batch,
                                  args.reduce, device, DTYPES[args.amp], calib["upscale"])
            mid, last = mid[None], last[None]
            score = float(scoring.get_scores(mid, last, calib["image_spec"], calib["image_stats"],
                                             valid=calib["valid"])[0])
            grid = scoring.make_map(mid, last, calib["pixel_spec"], calib["pixel_stats"])
            amap = to_pixels(grid, native, padded, calib["upscale"], div=1)[0]

            stem = Path(name).stem
            tifffile.imwrite(tiff_dir / f"{stem}.tiff", amap.astype(np.float16))
            anomalous = score >= calib["image_threshold"]
            if not args.no_thresholded:
                binary = mask_for(amap, pixel_thr, bool(anomalous), calib["gate"], calib["min_anomalous_px"])
                save_png(binary, png_dir / f"{stem}.png")

            rows.append({"split": split, "name": name, "condition": condition,
                         "image_score": score,
                         "predicted_anomalous": int(anomalous),
                         "map_max": float(amap.max()),
                         "anomalous_px": int((amap >= pixel_thr).sum()) if np.isfinite(pixel_thr) else -1})

            if i % 25 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  ({len(rows) / (time.time() - t0):.2f} img/s)", flush=True)

        flagged = sum(1 for r in rows if r["split"] == split and r["predicted_anomalous"])
        print(f"  flagged anomalous: {flagged}/{len(files)} ({100 * flagged / len(files):.1f}%)")
        check_outputs(submission, obj, split, expected, native, not args.no_thresholded)

    csv_path = out_dir / "image_scores.csv"
    write_csv(csv_path, rows)
    print(f"\nwrote {len(rows)} predictions to {submission} and {csv_path} "
          f"in {(time.time() - t0) / 60:.1f} min")
    print("The official checker needs all eight objects; run it once the other "
          f"objects are in place:\n{CHECKER}{submission.resolve()}")

if __name__ == "__main__":
    predict()
