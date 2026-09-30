import numpy as np

from pipeline.metrics import MAX_FPR, aupro, best_seg_f1, seg_f1, seg_f1_curve

rng = np.random.default_rng(0)
N, H, W = 20, 40, 50
masks = np.zeros((N, H, W), dtype=bool)
for i in range(0, N, 2):
    r, c = rng.integers(0, H - 8), rng.integers(0, W - 8)
    masks[i, r:r + 6, c:c + 7] = True
maps = rng.normal(size=(N, H, W)).astype(np.float32)
maps[masks] += 2.5

th, f1 = seg_f1_curve(maps, masks, n_thresholds=40)
brute = np.array([seg_f1(maps, masks, t)["seg_f1"] for t in th])
assert np.allclose(f1, brute, atol=1e-9), np.abs(f1 - brute).max()
print(f"seg_f1_curve == brute force over {len(th)} thresholds (max |d| {np.abs(f1 - brute).max():.1e})")

b = best_seg_f1(maps, masks, n_thresholds=256)
assert b["best_seg_f1"] >= brute.max() - 1e-9
print(f"best_seg_f1 {b['best_seg_f1']:.4f} at thr {b['best_threshold']:.4f}  "
      f"prec {b['best_seg_precision']:.3f} rec {b['best_seg_recall']:.3f}  "
      f"pred_frac {b['best_predicted_fraction']:.5f} vs true {masks.mean():.5f}")

ratio = b["best_predicted_fraction"] / masks.mean()
assert 0.4 < ratio < 2.5, ratio
print(f"predicted/true area at the F1 optimum: {ratio:.2f}x")

assert np.isnan(best_seg_f1(maps, np.zeros_like(masks))["best_seg_f1"])

assert abs(best_seg_f1(masks.astype(np.float32), masks)["best_seg_f1"] - 1.0) < 1e-6
noise = best_seg_f1(rng.normal(size=masks.shape).astype(np.float32), masks)["best_seg_f1"]
print(f"perfect map F1 1.0000, pure-noise map F1 {noise:.4f}")
assert noise < 0.25

a05, a30 = aupro(maps, masks), aupro(maps, masks, max_fpr=0.3)
assert MAX_FPR == 0.05 and aupro(maps, masks, max_fpr=0.05) == a05
print(f"aupro @0.05 {a05:.4f}  @0.30 {a30:.4f}  (default {MAX_FPR})")
assert a05 < a30
print("OK")
