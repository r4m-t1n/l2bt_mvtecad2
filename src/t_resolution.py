import json
import sys
from pathlib import Path
import numpy as np
from pipeline import scoring
from pipeline.config import PATCH
from pipeline.tiling import read_geometry, to_pixels

run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/vial_tiled_l15_23_smoke")
data = np.load(run / "errors.npz", allow_pickle=False)
calib = np.load(run / "calibration.npz", allow_pickle=False)
threshold = float(calib["pixel_threshold"])
spec = json.loads(str(calib["pixel_spec"]))

mid, last, roles = data["mid"], data["last"], data["role"]
geo = read_geometry(data, mid.shape[1:], PATCH)
stats = scoring.get_pos_stats(scoring.combine_layers(mid[roles == "train"],
                                                  last[roles == "train"], spec["combine"]))
test = roles == "test"
grids = scoring.make_map(mid[test], last[test], spec, stats)

print(f"{run.name}: {int(test.sum())} test images, spec {spec}, threshold {threshold:.6g}")
print(f"  {'div':>4}{'resolution':>14}{'pred_frac':>12}{'mean':>12}{'p99.9':>12}")
reference = None
for div in (4, 2, 1):
    maps = to_pixels(grids, geo["native"], geo["padded"], geo["upscale"], div)
    frac = float((maps >= threshold).mean())
    reference = frac if reference is None else reference
    print(f"  {div:>4}{f'{maps.shape[1]}x{maps.shape[2]}':>14}{frac:>12.6f}"
          f"{maps.mean():>12.6f}{np.quantile(maps, 0.999):>12.6f}")
    del maps

maps4 = to_pixels(grids, geo["native"], geo["padded"], geo["upscale"], 4)
maps1 = to_pixels(grids, geo["native"], geo["padded"], geo["upscale"], 1)
f4, f1 = float((maps4 >= threshold).mean()), float((maps1 >= threshold).mean())
rel = abs(f1 - f4) / max(f4, 1e-12)
print(f"\n  native/4 -> native/1 predicted-area drift: {100 * rel:.3f}% relative "
      f"({f4:.6f} -> {f1:.6f})")
assert rel < 0.05, f"predicted area moved {100 * rel:.2f}% between resolutions"
print("OK: under 5% relative, so the calibrated threshold transfers")
