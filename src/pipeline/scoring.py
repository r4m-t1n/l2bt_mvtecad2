import numpy as np
from scipy.ndimage import gaussian_filter

EPS = 1e-6

COMBINES = ("mul", "add", "mid", "last")
NORMS = ("none", "img_med", "img_mad", "pos_z", "pos_z+img_mad")

def combine_layers(mid, last, mode):
    if mode == "mul":
        return mid * last
    if mode == "add":
        return mid + last
    if mode in ("mid", "last"):
        return mid if mode == "mid" else last
    raise ValueError(f"unknown combine mode {mode!r}; expected one of {COMBINES}")

def get_pos_stats(grids):
    return {"mu": grids.mean(axis=0), "sd": grids.std(axis=0)}

def norm_grids(grids, mode, stats=None):
    if mode not in NORMS:
        raise ValueError(f"unknown norm mode {mode!r}; expected one of {NORMS}")

    out = grids.astype(np.float32, copy=True)
    if mode.startswith("pos_z"):
        if stats is None:
            raise ValueError(f"norm mode {mode!r} needs position stats")
        out = (out - stats["mu"][None]) / (stats["sd"][None] + EPS)
    if mode == "img_med":
        out -= np.median(out, axis=(1, 2), keepdims=True)
    elif mode.endswith("img_mad"):
        med = np.median(out, axis=(1, 2), keepdims=True)
        mad = np.median(np.abs(out - med), axis=(1, 2), keepdims=True)
        out = (out - med) / (mad + EPS)
    return out

def blur(grids, sigma):
    return gaussian_filter(grids, sigma=(0, sigma, sigma), mode="nearest") if sigma > 0 else grids

def topk_mean(grids, k):
    flat = grids.reshape(len(grids), -1)
    k = int(np.clip(k, 1, flat.shape[1]))
    return np.partition(flat, -k, axis=1)[:, -k:].mean(axis=1)

def crop_grids(grids, valid):
    return grids if grids.shape[-2:] == tuple(valid) else grids[..., :valid[0], :valid[1]]

def crop_stats(stats, valid):
    return None if stats is None else {k: crop_grids(v, valid) for k, v in stats.items()}

def make_map(mid, last, spec, stats=None):
    grids = norm_grids(combine_layers(mid, last, spec["combine"]), spec["norm"], stats)
    return blur(grids, spec.get("sigma", 0.0))

def get_scores(mid, last, spec, stats=None, valid=None):
    if valid is not None:
        mid, last = crop_grids(mid, valid), crop_grids(last, valid)
        stats = crop_stats(stats, valid)
    return topk_mean(make_map(mid, last, spec, stats), spec.get("k", 1))
