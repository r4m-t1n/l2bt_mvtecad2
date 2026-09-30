import argparse
import json
from pathlib import Path
import numpy as np

from . import scoring
from .config import (MASK_DIR, IMAGE_SPEC, PIXEL_SPEC, IMAGE_RULE, IMAGE_QUANTILE, PIXEL_RULE,
                     PIXEL_QUANTILE, AREA_FACTOR, GATE, MIN_PX, add_object_arg, get_paths)
from .data import load_masks
from .metrics import (aupro, best_class_f1, best_joint, best_seg_f1, confusion, image_metrics,
                      pixel_metrics, seg_f1, score_submission)
from .tiling import load_dump, to_pixels

PIXEL_DIV = 4
CHUNK = 32

STOCK = {"combine": "mul", "norm": "none", "sigma": 0.0}

def get_args():
    parser = argparse.ArgumentParser()
    add_object_arg(parser)
    parser.add_argument("--errors", type=Path, default=None)
    parser.add_argument("--chosen", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--calib-role", choices=("train", "val", "train+val"), default="train")
    parser.add_argument("--image-rule", choices=("f1_public", "quantile"), default=IMAGE_RULE)
    parser.add_argument("--image-quantile", type=float, default=IMAGE_QUANTILE)
    parser.add_argument("--pixel-rule", choices=("joint", "f1_public", "area", "quantile", "mean+3std"),
                        default=PIXEL_RULE)
    parser.add_argument("--pixel-quantile", type=float, default=PIXEL_QUANTILE)
    parser.add_argument("--area-factor", type=float, default=AREA_FACTOR)
    parser.add_argument("--no-gate", dest="gate", action="store_false")
    parser.add_argument("--min-anomalous-px", type=int, default=MIN_PX)
    parser.add_argument("--skip-pixel", action="store_true")
    parser.set_defaults(gate=GATE)
    return parser.parse_args()

def load_specs(path, out_dir):
    path = path or (out_dir / "chosen_config.json")
    if path.exists():
        chosen = json.loads(path.read_text())
        print(f"using aggregation from {path.name}")
        return chosen["image"], chosen["pixel"]
    print(f"{path.name} not found; falling back to the defaults in config.py")
    return dict(IMAGE_SPEC), dict(PIXEL_SPEC)

def normal_stats(mid, last, spec, stats, geo):
    total = count = total_sq = 0.0
    sample = []
    for start in range(0, len(mid), CHUNK):
        grids = scoring.make_map(mid[start:start + CHUNK], last[start:start + CHUNK], spec, stats)
        maps = to_pixels(grids, geo["native"], geo["padded"], geo["upscale"], PIXEL_DIV).astype(np.float64)
        total += maps.sum()
        total_sq += (maps**2).sum()
        count += maps.size
        sample.append(maps.reshape(-1)[::97].astype(np.float32))
    mean = total / count
    std = float(np.sqrt(max(0.0, total_sq / count - mean**2)))
    return float(mean), std, np.concatenate(sample)

def evaluate():
    args = get_args()
    obj_dir, out_dir = get_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = args.errors or (out_dir / "errors.npz")
    image_spec, pixel_spec = load_specs(args.chosen, out_dir)

    data, mid_all, last_all, geo = load_dump(errors_path)
    labels_all, roles, conditions, names = data["label"], data["role"], data["condition"], data["name"]
    eval_size = (geo["native"][0] // PIXEL_DIV, geo["native"][1] // PIXEL_DIV)

    train = roles == "train"
    test = roles == "test"
    calib = {"train": train, "val": roles == "val", "train+val": train | (roles == "val")}[args.calib_role]

    image_stats, pixel_stats, stock_stats = (
        scoring.get_pos_stats(scoring.combine_layers(mid_all[train], last_all[train], spec["combine"]))
        for spec in (image_spec, pixel_spec, STOCK))

    def image_scores(mask, spec, stats):
        return scoring.get_scores(mid_all[mask], last_all[mask], spec, stats, valid=geo["valid"])

    test_scores = image_scores(test, image_spec, image_stats)
    test_labels = labels_all[test]
    chosen_metrics = image_metrics(test_scores, test_labels)
    stock_image_spec = {**STOCK, "k": max(1, round(0.001 * int(np.prod(geo["valid"]))))}
    stock_metrics = image_metrics(image_scores(test, stock_image_spec, stock_stats), test_labels)

    calib_scores = image_scores(calib, image_spec, image_stats)
    image_thresholds = {
        "f1_public": best_class_f1(test_scores, test_labels)["best_image_threshold"],
        "quantile": float(np.quantile(calib_scores, args.image_quantile)),
    }
    image_thr = float(image_thresholds[args.image_rule])

    print("\n=== image level (test_public: "
          f"{int((test_labels == 0).sum())} good / {int((test_labels == 1).sum())} bad) ===")
    for tag, spec, found in (("chosen", image_spec, chosen_metrics), ("stock ", stock_image_spec, stock_metrics)):
        print(f"  {tag} {spec}")
        for key, value in found.items():
            print(f"    {key:<14}{value:.4f}")

    print(f"\n  image threshold rules ({int(calib.sum())} {args.calib_role}/good calibration images):")
    print(f"    {'rule':<12}{'threshold':>13}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}"
          f"{'recall':>9}{'FPR':>8}{'F1':>8}")
    image_by_rule = {}
    for rule, thr in image_thresholds.items():
        point = confusion(test_scores, test_labels, thr)
        image_by_rule[rule] = point
        marker = "  <-- applied" if rule == args.image_rule else ""
        print(f"    {rule:<12}{thr:>13.6g}{point['tp']:>5}{point['fp']:>5}"
              f"{point['fn']:>5}{point['tn']:>5}{point['recall']:>9.3f}"
              f"{point['fpr']:>8.3f}{point['f1']:>8.3f}{marker}")

    print(f"\n  normal-score quantile ladder (rule 'quantile' uses q={args.image_quantile}):")
    ladder = {}
    for q in (0.90, 0.95, 0.99, 0.995, 1.0):
        thr = float(np.quantile(calib_scores, q))
        point = confusion(test_scores, test_labels, thr)
        ladder[f"q{q}"] = point
        print(f"    q={q:<6} thr {thr:>11.6g}  TP {point['tp']:>3}  FP {point['fp']:>3}  "
              f"recall {point['recall']:.3f}  FPR {point['fpr']:.3f}  F1 {point['f1']:.3f}")

    print("\n  mean score on GOOD test images, by lighting condition:")
    good = test & (labels_all == 0)
    good_scores = image_scores(good, image_spec, image_stats)
    good_conds = conditions[good]
    by_condition = {}
    ref = float(np.mean(good_scores[good_conds == "regular"])) if (good_conds == "regular").any() else None
    for cond in sorted(set(good_conds)):
        subset = good_scores[good_conds == cond]
        ratio = float(np.mean(subset)) / ref if ref else float("nan")
        by_condition[cond] = {"n": int(len(subset)), "mean": float(np.mean(subset)), "ratio_to_regular": ratio}
        print(f"    {cond:<14}n={len(subset):>3}  mean {np.mean(subset):>11.6g}  "
              f"x{ratio:.2f} vs regular")

    results = {
        "object": args.object_name,
        "errors": str(errors_path),
        "geometry": {"native": list(geo["native"]), "padded": list(geo["padded"]),
                     "grid": list(geo["grid"]), "valid_cells": list(geo["valid"]),
                     "upscale": geo["upscale"], "pixel_eval": list(eval_size)},
        "image": {"spec": image_spec, "metrics": chosen_metrics,
                  "stock_spec": stock_image_spec, "stock_metrics": stock_metrics,
                  "threshold": image_thr, "rule": args.image_rule,
                  "candidates": image_thresholds, "by_rule": image_by_rule,
                  "calib_role": args.calib_role,
                  "calib_n": int(calib.sum()), "quantile": args.image_quantile,
                  "at_threshold": image_by_rule[args.image_rule], "quantile_sweep": ladder,
                  "good_by_condition": by_condition},
    }

    pixel_thr = float("nan")
    if not args.skip_pixel:
        labels = labels_all[test]
        is_bad = labels == 1
        test_names = [str(n) for n in names[test]]
        masks = np.zeros((len(test_names), *eval_size), dtype=bool)
        masks[is_bad] = load_masks([n for n, b in zip(test_names, is_bad) if b],
                                   eval_size, obj_dir / MASK_DIR)

        def pixel_maps(mask, spec, stats):
            grids = scoring.make_map(mid_all[mask], last_all[mask], spec, stats)
            return to_pixels(grids, geo["native"], geo["padded"], geo["upscale"], PIXEL_DIV)

        defect_frac = float(masks.mean())
        print(f"\n=== pixel level ({len(test_names)} test images at "
              f"{eval_size[0]}x{eval_size[1]}: {int(is_bad.sum())} defective, "
              f"{100 * defect_frac:.4f}% anomalous px) ===")

        pixel_results = {}
        for tag, spec, stats in (("chosen", pixel_spec, pixel_stats), ("stock", STOCK, stock_stats)):
            maps = pixel_maps(test, spec, stats)
            entry = pixel_metrics(maps, masks)
            entry["aupro"] = aupro(maps, masks)
            entry.update(best_seg_f1(maps, masks))
            pixel_results[tag] = {"spec": spec, **entry}
            print(f"  {tag} {spec}")
            for key in ("aupro", "pixel_ap", "pixel_auroc", "best_seg_f1"):
                print(f"    {key:<14}{entry[key]:.4f}")
            del maps

        mean, std, sample = normal_stats(mid_all[calib], last_all[calib], pixel_spec, pixel_stats, geo)
        maps = pixel_maps(test, pixel_spec, pixel_stats)
        area_q = float(np.clip(1.0 - defect_frac * args.area_factor, 0.0, 1.0))

        gate = (test_scores >= image_thr) if args.gate else None
        pixel_thresholds = {
            "joint": best_joint(maps, masks, labels, gate, args.min_anomalous_px)["best_threshold"],
            "f1_public": best_seg_f1(maps, masks)["best_threshold"],
            "area": float(np.quantile(sample, area_q)),
            "quantile": float(np.quantile(sample, args.pixel_quantile)),
            "mean+3std": mean + 3 * std,
        }
        print(f"\n  normal-pixel stats over {int(calib.sum())} {args.calib_role}/good images: "
              f"mean {mean:.6g}, std {std:.6g}")
        print(f"  area rule uses the {area_q:.6f} quantile "
              f"(public defect fraction {defect_frac:.6f} x factor {args.area_factor:g})")

        prevalence = float((labels == 1).mean())
        floor_f1 = 2 * prevalence / (1 + prevalence)
        gate_txt = f" (image threshold {image_thr:.6g}, min {args.min_anomalous_px} px forced)" if args.gate else ""
        print(f"\n  gating: {'ON' if args.gate else 'OFF'}{gate_txt}"
              f"   all-positive ClassF1 floor {floor_f1:.4f} at prevalence {prevalence:.3f}")
        print(f"\n  {'rule':<12}{'threshold':>13}{'pred_frac':>11}{'SegF1':>8}"
              f"{'prec':>8}{'recall':>8}{'|':>3}{'SegF1*':>8}{'ClassF1*':>10}{'flagged*':>10}{'joint*':>8}")
        by_rule = {}
        for rule, thr in pixel_thresholds.items():
            point = seg_f1(maps, masks, thr)
            sub = score_submission(maps, masks, labels, thr, gate, args.min_anomalous_px)
            point["submission"] = sub
            by_rule[rule] = point
            marker = "  <-- applied" if rule == args.pixel_rule else ""
            print(f"  {rule:<12}{thr:>13.6g}{point['predicted_fraction']:>11.6f}"
                  f"{point['seg_f1']:>8.4f}{point['seg_precision']:>8.4f}"
                  f"{point['seg_recall']:>8.4f}{'|':>3}{sub['seg_f1']:>8.4f}"
                  f"{sub['class_f1']:>10.4f}{sub['flagged_fraction']:>10.3f}"
                  f"{sub['joint']:>8.4f}{marker}")
        print("  * = as submitted: SegF1 on the written masks, ClassF1 as the server derives it")
        if args.gate:
            ungated = score_submission(maps, masks, labels, pixel_thresholds[args.pixel_rule],
                                       None, args.min_anomalous_px)
            print(f"  same threshold without gating: SegF1 {ungated['seg_f1']:.4f}  "
                  f"ClassF1 {ungated['class_f1']:.4f}  joint {ungated['joint']:.4f}  "
                  f"(flagged {ungated['flagged_fraction']:.3f})")

        pixel_thr = float(pixel_thresholds[args.pixel_rule])
        pixel_results["threshold"] = {
            "value": pixel_thr, "rule": args.pixel_rule,
            "normal_mean": mean, "normal_std": std,
            "public_defect_fraction": defect_frac,
            "area_quantile": area_q, "area_factor": args.area_factor,
            "gate": bool(args.gate), "min_anomalous_px": int(args.min_anomalous_px),
            "prevalence": prevalence, "all_positive_class_f1": floor_f1,
            "candidates": pixel_thresholds, "by_rule": by_rule,
            **by_rule[args.pixel_rule],
        }
        results["pixel"] = pixel_results
        del maps

    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    np.savez_compressed(
        out_dir / "calibration.npz",
        image_mu=image_stats["mu"], image_sd=image_stats["sd"],
        pixel_mu=pixel_stats["mu"], pixel_sd=pixel_stats["sd"],
        image_threshold=np.float64(image_thr),
        pixel_threshold=np.float64(pixel_thr),
        image_rule=np.asarray(args.image_rule),
        pixel_rule=np.asarray(args.pixel_rule),
        gate=np.bool_(args.gate),
        min_anomalous_px=np.int64(args.min_anomalous_px),
        image_spec=np.asarray(json.dumps(image_spec)),
        pixel_spec=np.asarray(json.dumps(pixel_spec)),
        object=np.asarray(args.object_name),
        upscale=np.int64(geo["upscale"]),
        native_hw=np.asarray(geo["native"], dtype=np.int64),
        padded_hw=np.asarray(geo["padded"], dtype=np.int64),
        valid_cells=np.asarray(geo["valid"], dtype=np.int64),
        pixel_eval=np.asarray(eval_size, dtype=np.int64),
    )
    print(f"\nwrote {out_dir / 'results.json'} and {out_dir / 'calibration.npz'}")

if __name__ == "__main__":
    evaluate()
