import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
import tifffile
from PIL import Image
from pipeline import metrics as M

def server_labels(maps, thr, gate=None, min_px=1):
    counts = (maps >= thr).reshape(len(maps), -1).sum(1)
    if gate is not None:
        counts = np.where(np.asarray(gate, dtype=bool), np.maximum(counts, min_px), 0)
    return counts >= min_px

def synthetic(n_good=30, n_bad=30, size=24, seed=0):
    rng = np.random.default_rng(seed)
    total = n_good + n_bad
    labels = np.array([0] * n_good + [1] * n_bad)
    masks = np.zeros((total, size, size), bool)
    maps = rng.normal(0.2, 0.05, (total, size, size)).astype(np.float32)
    for i in range(n_good, total):
        masks[i, 8:12, 8:12] = True
        maps[i, 8:12, 8:12] += rng.normal(0.5, 0.1, (4, 4))
    for i in (2, 5, 9):
        maps[i, 0, 0] += 0.6
    return maps, masks, labels

def check_metrics():
    maps, masks, labels = synthetic()
    prevalence = float(labels.mean())
    floor = 2 * prevalence / (1 + prevalence)
    print(f"prevalence {prevalence:.3f}  all-positive ClassF1 floor {floor:.4f}")

    image_scores = maps.reshape(len(maps), -1).max(1)
    guarded = M.best_class_f1(image_scores, labels)
    assert not guarded["is_degenerate"], "guard let the all-positive endpoint win"
    assert abs(guarded["all_positive_f1"] - floor) < 1e-9
    print(f"image F1 {guarded['best_class_f1']:.4f} (non-degenerate), floor reported {floor:.4f}")

    threshold = float(np.quantile(maps[masks], 0.2))
    gate = image_scores >= guarded["best_image_threshold"]
    ungated = M.score_submission(maps, masks, labels, threshold, None, 1)
    gated = M.score_submission(maps, masks, labels, threshold, gate, 32)
    print(f"at thr {threshold:.4f}  ungated SegF1 {ungated['seg_f1']:.4f} "
          f"ClassF1 {ungated['class_f1']:.4f}   gated SegF1 {gated['seg_f1']:.4f} "
          f"ClassF1 {gated['class_f1']:.4f}")

    assert np.array_equal(server_labels(maps, threshold, gate, 32), gate)
    point = M.confusion(gate.astype(np.float64), labels, 0.5)
    assert abs(point["f1"] - gated["class_f1"]) < 1e-9
    assert gated["seg_f1"] >= ungated["seg_f1"] - 1e-9
    print("gated ClassF1 == image-classifier F1; gating did not reduce SegF1")

    joint = M.best_joint(maps, masks, labels, gate, 32, n_thresholds=48)
    assert np.isfinite(joint["best_threshold"])
    print(f"joint pick thr {joint['best_threshold']:.4f}  SegF1 {joint['seg_f1']:.4f}  "
          f"ClassF1 {joint['class_f1']:.4f}  joint {joint['joint']:.4f}")

    impossible = float(maps.max()) + 1.0
    empty = M.make_masks(maps, impossible, None, 32)
    assert not empty.any()
    assert M.score_submission(maps, masks, labels, impossible, None, 32)["class_f1"] == 0.0
    forced = M.make_masks(maps, impossible, gate, 32)
    counts = forced.reshape(len(forced), -1).sum(1)
    assert np.array_equal(counts > 0, gate), "forced pixels did not land on the gated images"
    assert set(counts[gate].tolist()) == {32}, f"expected exactly 32 forced px, got {set(counts.tolist())}"
    survived = M.score_submission(maps, masks, labels, impossible, gate, 32)
    assert abs(survived["class_f1"] - guarded["best_class_f1"]) < 1e-9
    print(f"forced-pixel path: ungated ClassF1 0.0000 -> gated {survived['class_f1']:.4f} "
          f"at a threshold above every score")

def check_restamp():
    root = Path(tempfile.mkdtemp())
    try:
        obj, split = "rice", "test_private"
        submission, out_dir = root / "submission", root / "out"
        out_dir.mkdir()
        tiff_dir = submission / "anomaly_images" / obj / split
        png_dir = submission / "anomaly_images_thresholded" / obj / split
        tiff_dir.mkdir(parents=True)
        png_dir.mkdir(parents=True)

        rng = np.random.default_rng(1)
        height = width = 40
        n_good = 5
        rows = []
        for index in range(10):
            anomaly_map = rng.normal(0.1, 0.02, (height, width)).astype(np.float16)
            if index >= n_good:
                anomaly_map[10:14, 10:14] += np.float16(0.4)
            stem = f"{index:03d}_regular"
            tifffile.imwrite(tiff_dir / f"{stem}.tiff", anomaly_map)
            Image.fromarray(np.zeros((height, width), np.uint8), "L").save(png_dir / f"{stem}.png")
            score = 0.05 if index < n_good else 0.5
            rows.append({"split": split, "name": f"{stem}.png", "condition": "regular",
                         "image_score": score, "predicted_anomalous": int(score >= 0.3),
                         "map_max": score, "anomalous_px": 0})

        with (out_dir / "image_scores.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        np.savez_compressed(
            out_dir / "calibration.npz",
            image_mu=np.zeros((5, 5), np.float32), image_sd=np.ones((5, 5), np.float32),
            pixel_mu=np.zeros((5, 5), np.float32), pixel_sd=np.ones((5, 5), np.float32),
            image_threshold=np.float64(0.3), pixel_threshold=np.float64(0.35),
            image_rule=np.asarray("f1_public"), pixel_rule=np.asarray("joint"),
            gate=np.bool_(True), min_anomalous_px=np.int64(8),
            image_spec=np.asarray('{"combine":"mul","norm":"none","sigma":0.0,"k":1}'),
            pixel_spec=np.asarray('{"combine":"mul","norm":"none","sigma":0.0}'),
            object=np.asarray(obj), upscale=np.int64(1),
            native_hw=np.asarray([height, width], np.int64),
            padded_hw=np.asarray([height, width], np.int64),
            valid_cells=np.asarray([5, 5], np.int64),
            pixel_eval=np.asarray([height, width], np.int64),
        )

        result = subprocess.run(
            [sys.executable, "-m", "pipeline.restamp", "--object", obj,
             "--submission", str(submission), "--out-dir", str(out_dir), "--splits", split],
            cwd=Path(__file__).parent, capture_output=True, text=True, check=False)
        print(result.stdout.rstrip())
        if result.returncode:
            print(result.stderr[-2000:])
            raise SystemExit("restamp failed")

        flagged = 0
        for index in range(10):
            arr = np.asarray(Image.open(png_dir / f"{index:03d}_regular.png"))
            assert set(np.unique(arr)).issubset({0, 255}), "png must be {0, 255}"
            positive = int((arr == 255).sum())
            if index < n_good:
                assert positive == 0, f"image {index} is gated normal but has {positive} px"
            else:
                assert positive >= 8, f"image {index} is gated anomalous but has {positive} px"
            flagged += positive > 0
        assert flagged == 10 - n_good
        print(f"restamp round-trip: {flagged}/10 read anomalous, as the image scores imply")
    finally:
        shutil.rmtree(root)

if __name__ == "__main__":
    check_metrics()
    print()
    check_restamp()
    print("\nOK")
