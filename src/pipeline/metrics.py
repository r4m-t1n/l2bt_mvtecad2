import numpy as np
from scipy.ndimage import label
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

MAX_FPR = 0.05

def image_metrics(scores, labels):
    if len(np.unique(labels)) < 2:
        return {"image_auroc": float("nan"), "image_ap": float("nan"), "image_f1max": float("nan")}
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_ap": float(average_precision_score(labels, scores)),
        "image_f1max": float(f1.max()),
    }

def pixel_metrics(maps, masks):
    truth = (masks.reshape(-1) > 0).astype(np.uint8)
    if truth.max() == 0:
        return {"pixel_auroc": float("nan"), "pixel_ap": float("nan")}
    scores = maps.reshape(-1)
    return {"pixel_auroc": float(roc_auc_score(truth, scores)), "pixel_ap": float(average_precision_score(truth, scores))}

def get_regions(masks):
    binary = masks > 0
    h, w = binary.shape[1:]
    regions = []
    for i in range(len(binary)):
        if not binary[i].any():
            continue
        labelled, count = label(binary[i])
        flat = labelled.reshape(-1)
        for region in range(1, count + 1):
            regions.append((np.flatnonzero(flat == region) + i * h * w).astype(np.int64))
    return {"regions": regions, "negatives": ~binary.reshape(-1), "shape": binary.shape}

def frac_above(sorted_scores, thresholds):
    return (len(sorted_scores) - np.searchsorted(sorted_scores, thresholds, side="left")) / len(sorted_scores)

def calc_aupro(maps, regions, max_fpr=MAX_FPR, n_thresholds=64):
    if maps.shape != regions["shape"]:
        raise ValueError(f"map shape {maps.shape} does not match ground truth {regions['shape']}")
    if not regions["regions"]:
        return float("nan")

    flat = maps.reshape(-1)
    negatives = np.sort(flat[regions["negatives"]])
    region_scores = [np.sort(flat[idx]) for idx in regions["regions"]]

    lo = 1.0 - min(1.0, max_fpr * 2)
    thresholds = np.unique(np.quantile(negatives, np.linspace(lo, 1.0, n_thresholds)))

    fprs = frac_above(negatives, thresholds)
    pros = np.stack([frac_above(r, thresholds) for r in region_scores], axis=1).mean(axis=1)

    order = np.argsort(fprs)
    fprs = np.concatenate([[0.0], fprs[order]])
    pros = np.concatenate([[0.0], pros[order]])

    keep = fprs <= max_fpr
    x, y = fprs[keep], pros[keep]
    if x[-1] < max_fpr:
        rest = np.flatnonzero(~keep)
        end = y[-1]
        if len(rest):
            nxt = rest[0]
            span = fprs[nxt] - x[-1]
            frac = (max_fpr - x[-1]) / span if span > 0 else 0.0
            end = y[-1] + frac * (pros[nxt] - y[-1])
        x, y = np.append(x, max_fpr), np.append(y, end)
    return float(np.trapezoid(y, x) / max_fpr)

def aupro(maps, masks, max_fpr=MAX_FPR, n_thresholds=64):
    return calc_aupro(maps, get_regions(masks), max_fpr, n_thresholds)

def seg_f1(maps, masks, thr):
    return {"threshold": float(thr), **mask_f1(maps >= thr, masks)}

def seg_f1_curve(maps, masks, n_thresholds=256):
    truth = masks > 0
    pos = np.sort(maps[truth].reshape(-1))
    neg = np.sort(maps[~truth].reshape(-1))
    thresholds = np.unique(np.quantile(pos, np.linspace(0.0, 1.0, n_thresholds)))
    tp = len(pos) - np.searchsorted(pos, thresholds, side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds, side="left")
    fn = len(pos) - tp
    return thresholds, 2 * tp / np.maximum(1e-12, 2 * tp + fp + fn)

def best_seg_f1(maps, masks, n_thresholds=256):
    if not (masks > 0).any():
        return {"best_seg_f1": float("nan"), "best_threshold": float("nan")}
    thresholds, f1 = seg_f1_curve(maps, masks, n_thresholds)
    best = seg_f1(maps, masks, float(thresholds[int(np.argmax(f1))]))
    return {f"best_{k}": best[k] for k in ("seg_f1", "threshold", "predicted_fraction", "seg_precision", "seg_recall")}

def confusion(scores, labels, thr):
    pred = scores >= thr
    truth = labels > 0
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    return {
        "threshold": float(thr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
        "fpr": fp / max(1, fp + tn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }

def best_class_f1(scores, labels, allow_degenerate=False):
    if len(np.unique(labels)) < 2:
        return {"best_class_f1": float("nan"), "best_image_threshold": float("nan")}
    unique = np.unique(scores)
    candidates = np.concatenate([[unique[0]], (unique[:-1] + unique[1:]) / 2, [unique[-1]]])
    points = [confusion(scores, labels, t) for t in candidates]

    total = len(labels)

    def all_or_none(p):
        return (p["tp"] + p["fp"]) in (0, total)

    usable = [p for p in points if allow_degenerate or not all_or_none(p)] or points
    best = max(usable, key=lambda p: p["f1"])
    floor = max((p["f1"] for p in points if all_or_none(p)), default=float("nan"))
    return {"best_class_f1": best["f1"], "best_image_threshold": best["threshold"],
            "best_class_recall": best["recall"], "best_class_precision": best["precision"],
            "best_class_fpr": best["fpr"],
            "all_positive_f1": floor, "is_degenerate": all_or_none(best)}

def make_masks(maps, thr, gate=None, min_px=1):
    binary = maps >= thr
    if gate is None:
        return binary
    keep = np.asarray(gate, dtype=bool)
    binary[~keep] = False
    for i in np.flatnonzero(keep):
        row = binary[i]
        if row.any():
            continue
        flat = maps[i].reshape(-1)
        top = np.argpartition(flat, -min_px)[-min_px:]
        row.reshape(-1)[top] = True
    return binary

def mask_f1(binary, masks):
    truth = masks > 0
    tp = int((binary & truth).sum())
    fp = int((binary & ~truth).sum())
    fn = int((~binary & truth).sum())
    return {
        "seg_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "seg_precision": tp / max(1, tp + fp),
        "seg_recall": tp / max(1, tp + fn),
        "predicted_fraction": float(binary.mean()),
    }

def score_submission(maps, masks, labels, thr, gate=None, min_px=1):
    binary = make_masks(maps, thr, gate, min_px)
    result = mask_f1(binary, masks)
    flagged = binary.reshape(len(binary), -1).any(1)
    point = confusion(flagged.astype(np.float64), labels, 0.5)
    result.update({
        "threshold": float(thr),
        "class_f1": point["f1"], "class_recall": point["recall"],
        "class_precision": point["precision"], "class_fpr": point["fpr"],
        "flagged_fraction": float(flagged.mean()),
        "joint": 0.5 * (result["seg_f1"] + point["f1"]),
    })
    return result

def get_thresholds(maps, masks, n):
    truth = masks > 0
    parts = [maps.reshape(len(maps), -1).max(1)]
    if truth.any():
        parts.append(np.quantile(maps[truth].reshape(-1), np.linspace(0.0, 1.0, n)))
    grid = np.unique(np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1) for p in parts]))
    if len(grid) > 2 * n:
        grid = grid[np.unique(np.linspace(0, len(grid) - 1, 2 * n).astype(int))]
    return grid

def best_joint(maps, masks, labels, gate=None, min_px=1, n_thresholds=96):
    best = max((score_submission(maps, masks, labels, t, gate, min_px) for t in get_thresholds(maps, masks, n_thresholds)),
               key=lambda r: r["joint"])
    return {"best_threshold": best["threshold"], **best}
