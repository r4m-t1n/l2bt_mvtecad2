import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

from . import scoring
from .config import MASK_DIR, PIXEL_SPEC, add_object_arg, get_paths
from .data import get_instance, load_masks, write_csv
from .metrics import MAX_FPR, calc_aupro, best_seg_f1, pixel_metrics, get_regions
from .tiling import load_dump, to_pixels

KS = (1, 2, 3, 5, 10, 20, 50, 100, 200)
SIGMAS = (0.0, 0.5, 1.0, 2.0)
PIXEL_DIV = 8

def get_args():
    parser = argparse.ArgumentParser()
    add_object_arg(parser)
    parser.add_argument("--errors", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--skip-pixel", action="store_true")
    return parser.parse_args()

def split_halves(labels, names):
    half_a, half_b = [], []
    for value in np.unique(labels):
        groups = {}
        for i in np.flatnonzero(labels == value):
            groups.setdefault(get_instance(str(names[i])), []).append(int(i))
        for n, key in enumerate(sorted(groups)):
            (half_a if n % 2 == 0 else half_b).extend(groups[key])
    return np.sort(np.asarray(half_a, dtype=np.int64)), np.sort(np.asarray(half_b, dtype=np.int64))

def auroc(scores, labels):
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) >= 2 else float("nan")

def run_sweep():
    args = get_args()
    obj_dir, out_dir = get_paths(args)
    errors_path = args.errors or (out_dir / "errors.npz")
    out_dir.mkdir(parents=True, exist_ok=True)

    data, mid_all, last_all, geo = load_dump(errors_path)
    labels_all, roles, names = data["label"], data["role"], data["name"]
    eval_size = (geo["native"][0] // PIXEL_DIV, geo["native"][1] // PIXEL_DIV)

    train = roles == "train"
    test = roles == "test"
    if not train.any():
        raise RuntimeError("no train/good images in the dump; per-position normalization needs them")

    pos_stats = {
        mode: scoring.get_pos_stats(scoring.combine_layers(mid_all[train], last_all[train], mode))
        for mode in scoring.COMBINES
    }

    mid, last = mid_all[test], last_all[test]
    labels = labels_all[test]
    test_names = names[test]
    half_a, half_b = split_halves(labels, test_names)
    n_cells = int(np.prod(geo["valid"]))
    stock_k = min(KS, key=lambda k: abs(k - 0.001 * n_cells))
    n_instances = len({get_instance(str(n)) for n in test_names})
    print(f"test: {int((labels == 0).sum())} good / {int((labels == 1).sum())} bad over "
          f"{n_instances} instances; halves {len(half_a)}/{len(half_b)} images; "
          f"stock topk_ratio=0.001 -> k={stock_k} patch cells")

    if len(np.unique(labels[half_a])) != 2 or len(np.unique(labels[half_b])) != 2:
        print("  WARNING: a selection half is single-class (too few distinct instances). "
              "Ranking on ALL test images instead; sel and hold columns are the same number "
              "and the holdout check is void.")
        half_a = half_b = np.arange(len(labels))

    mid_img, last_img = scoring.crop_grids(mid, geo["valid"]), scoring.crop_grids(last, geo["valid"])
    stats_img = {mode: scoring.crop_stats(stats, geo["valid"]) for mode, stats in pos_stats.items()}

    image_rows = []
    for mode in scoring.COMBINES:
        combined = scoring.combine_layers(mid_img, last_img, mode)
        for norm in scoring.NORMS:
            normed = scoring.norm_grids(combined, norm, stats_img[mode])
            for sigma in SIGMAS:
                blurred = scoring.blur(normed, sigma)
                for k in KS:
                    scores = scoring.topk_mean(blurred, k)
                    image_rows.append({
                        "combine": mode, "norm": norm, "sigma": sigma, "k": k,
                        "auroc_sel": auroc(scores[half_a], labels[half_a]),
                        "auroc_hold": auroc(scores[half_b], labels[half_b]),
                        "auroc_all": auroc(scores, labels),
                    })
    image_rows.sort(key=lambda r: -r["auroc_sel"])

    stock = next(r for r in image_rows if r["combine"] == "mul" and r["norm"] == "none"
                 and r["sigma"] == 0.0 and r["k"] == stock_k)
    best_image = image_rows[0]

    print("\nimage aggregation, top 10 by selection half:")
    print(f"  {'combine':<8}{'norm':<16}{'sigma':>6}{'k':>6}{'sel':>9}{'hold':>9}{'all':>9}")
    for row in image_rows[:10]:
        print(f"  {row['combine']:<8}{row['norm']:<16}{row['sigma']:>6.1f}{row['k']:>6}"
              f"{row['auroc_sel']:>9.4f}{row['auroc_hold']:>9.4f}{row['auroc_all']:>9.4f}")
    print(f"  {'-- stock L2BT equivalent --':<36}"
          f"{stock['auroc_sel']:>9.4f}{stock['auroc_hold']:>9.4f}{stock['auroc_all']:>9.4f}")
    gap = best_image["auroc_sel"] - best_image["auroc_hold"]
    if abs(gap) > 0.06:
        print(f"  WARNING: selection/holdout gap {gap:+.4f} -- the winner may be sweep noise; "
              f"prefer a configuration with a smaller gap.")

    write_csv(out_dir / "sweep_image.csv", image_rows)

    pixel_rows = []
    best_pixel = dict(PIXEL_SPEC)
    if not args.skip_pixel:
        bad = labels == 1
        masks = load_masks(list(test_names[bad]), eval_size, obj_dir / MASK_DIR)
        all_masks = np.zeros((len(labels), *eval_size), dtype=bool)
        all_masks[bad] = masks
        print(f"\nmasks: {int(masks.sum())} anomalous px of {all_masks.size} "
              f"({100 * masks.sum() / all_masks.size:.2f}%) at {eval_size[0]}x{eval_size[1]}")

        regions_a, regions_b = get_regions(all_masks[half_a]), get_regions(all_masks[half_b])
        print(f"ground-truth regions: {len(regions_a['regions'])} sel / "
              f"{len(regions_b['regions'])} hold")

        for mode in scoring.COMBINES:
            for norm in scoring.NORMS:
                for sigma in SIGMAS:
                    spec = {"combine": mode, "norm": norm, "sigma": sigma}
                    maps = to_pixels(scoring.make_map(mid, last, spec, pos_stats[mode]),
                                     geo["native"], geo["padded"], geo["upscale"], PIXEL_DIV)
                    row = {**spec,
                           "aupro_sel": calc_aupro(maps[half_a], regions_a, n_thresholds=48),
                           "aupro_hold": calc_aupro(maps[half_b], regions_b, n_thresholds=48)}
                    for k, v in pixel_metrics(maps[half_a], all_masks[half_a]).items():
                        row[f"{k}_sel"] = v
                    row["segf1_sel"] = best_seg_f1(maps[half_a], all_masks[half_a], n_thresholds=64)["best_seg_f1"]
                    row["segf1_hold"] = best_seg_f1(maps[half_b], all_masks[half_b], n_thresholds=64)["best_seg_f1"]
                    pixel_rows.append(row)
            print(f"  swept combine={mode}", flush=True)

        pixel_rows.sort(key=lambda r: -r["aupro_sel"])
        best_pixel = {k: pixel_rows[0][k] for k in ("combine", "norm", "sigma")}
        print("\npixel aggregation, top 10 by selection-half AUPRO "
              f"(fpr_limit {MAX_FPR}, matching the benchmark's AucPro_0.05):")
        print(f"  {'combine':<8}{'norm':<16}{'sigma':>6}{'aupro':>9}{'apro_ho':>9}"
              f"{'segF1':>8}{'segF1_ho':>9}{'pixAP':>9}")
        for row in pixel_rows[:10]:
            print(f"  {row['combine']:<8}{row['norm']:<16}{row['sigma']:>6.1f}"
                  f"{row['aupro_sel']:>9.4f}{row['aupro_hold']:>9.4f}"
                  f"{row['segf1_sel']:>8.4f}{row['segf1_hold']:>9.4f}"
                  f"{row['pixel_ap_sel']:>9.4f}")
        best_f1 = max(pixel_rows, key=lambda r: r["segf1_sel"])
        if best_f1 is not pixel_rows[0]:
            print(f"  note: best achievable SegF1 is {best_f1['segf1_sel']:.4f} at "
                  f"{best_f1['combine']}/{best_f1['norm']}/sigma {best_f1['sigma']:g} "
                  f"(AUPRO {best_f1['aupro_sel']:.4f}), vs {pixel_rows[0]['segf1_sel']:.4f} "
                  f"for the AUPRO winner")
        write_csv(out_dir / "sweep_pixel.csv", pixel_rows)

    chosen = {
        "object": args.object_name,
        "errors": str(errors_path),
        "ckpt": str(data["ckpt"]) if "ckpt" in data else None,
        "upscale": geo["upscale"],
        "image": {k: best_image[k] for k in ("combine", "norm", "sigma", "k")},
        "pixel": best_pixel,
        **{f"image_{k}": best_image[k] for k in ("auroc_sel", "auroc_hold", "auroc_all")},
        "stock_image_auroc_all": stock["auroc_all"],
        "pixel_selection": pixel_rows[0] if pixel_rows else None,
    }

    path = out_dir / "chosen_config.json"
    path.write_text(json.dumps(chosen, indent=2))
    print(f"\nwrote {path}")
    print(f"image score: {chosen['image']}  (AUROC {best_image['auroc_all']:.4f} "
          f"vs stock {stock['auroc_all']:.4f})")
    print(f"pixel map:   {best_pixel}")

if __name__ == "__main__":
    run_sweep()
