import math
import numpy as np
import torch
import torch.nn.functional as F

from .config import PATCH

def round_up(n, patch):
    return -(-n // patch) * patch

def get_padded_size(native_hw, patch, upscale=1):
    return tuple(round_up(n * upscale, patch) for n in native_hw)

def get_valid_cells(native_hw, patch, upscale=1):
    return tuple(n * upscale // patch for n in native_hw)

def pad_image(img, patch, upscale=1):
    if upscale < 1:
        raise ValueError(f"upscale must be >= 1, got {upscale}")

    if upscale > 1:
        img = F.interpolate(img[None], scale_factor=upscale, mode="bilinear", align_corners=False)[0]

    h, w = int(img.shape[-2]), int(img.shape[-1])
    pad_h, pad_w = round_up(h, patch), round_up(w, patch)
    if (pad_h, pad_w) != (h, w):
        img = F.pad(img[None], (0, pad_w - w, 0, pad_h - h), mode="replicate")[0]
    return img, (pad_h, pad_w)

def get_offsets(total, tile, patch):
    if total % patch or tile % patch:
        raise ValueError(f"total ({total}) and tile ({tile}) must both be multiples of patch ({patch})")
    if tile > total:
        raise ValueError(f"tile ({tile}) exceeds the image extent ({total}); a tile is cut, never padded, "
                         f"so use --tile {total} or smaller")
    if tile == total:
        return [0]

    n = math.ceil(total / tile)
    span = total - tile
    offsets = sorted({(round(i * span / (n - 1)) // patch) * patch for i in range(n)})
    offsets[-1] = span

    for a, b in zip(offsets, offsets[1:]):
        if b > a + tile:
            raise ValueError(f"tiling leaves a gap between offsets {a} and {b} (tile={tile})")
    return offsets

def make_plan(h, w, tile, patch):
    return get_offsets(h, tile, patch), get_offsets(w, tile, patch), (h // patch, w // patch)

def cut_tiles(img, rows, cols, tile):
    return torch.stack([img[:, r:r + tile, c:c + tile] for r in rows for c in cols])

def stitch_tiles(tiles, rows, cols, patch, shape, reduce="max"):
    gh, gw = tiles.shape[1:]
    out = np.zeros(shape, dtype=np.float64)
    hits = np.zeros(shape, dtype=np.int32)
    if reduce == "max":
        out[:] = -np.inf

    for i, (r, c) in enumerate((r, c) for r in rows for c in cols):
        pr, pc = r // patch, c // patch
        view = out[pr:pr + gh, pc:pc + gw]
        if reduce == "max":
            np.maximum(view, tiles[i], out=view)
        else:
            view += tiles[i]
        hits[pr:pr + gh, pc:pc + gw] += 1

    if not hits.all():
        raise RuntimeError(f"{int((hits == 0).sum())} patch cells were never covered by a tile")
    if reduce == "mean":
        out /= hits
    return out.astype(np.float32)

def read_geometry(data, grid, patch):
    if "native_hw" in data:
        native, padded, valid = (tuple(int(v) for v in data[k]) for k in ("native_hw", "padded_hw", "valid_cells"))
        return {"native": native, "padded": padded, "valid": valid, "upscale": int(data["upscale"]), "grid": tuple(grid)}
    full = (grid[0] * patch, grid[1] * patch)
    return {"native": full, "padded": full, "valid": tuple(grid), "upscale": 1, "grid": tuple(grid)}

def upsample(grid, h, w):
    x = torch.from_numpy(np.ascontiguousarray(grid)).float()
    single = x.ndim == 2
    x = x[None, None] if single else x[:, None]
    out = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return out[0, 0].numpy() if single else out[:, 0].numpy()

def to_pixels(grids, native_hw, padded_hw, upscale=1, div=1):
    out_h, out_w = native_hw[0] // div, native_hw[1] // div
    up_h = math.ceil(padded_hw[0] / (upscale * div))
    up_w = math.ceil(padded_hw[1] / (upscale * div))
    if up_h < out_h or up_w < out_w:
        raise ValueError(f"padded extent {padded_hw} at upscale {upscale} is smaller than the "
                         f"requested output {(out_h, out_w)}; native_hw/padded_hw are inconsistent")

    maps = upsample(grids, up_h, up_w)
    if (up_h, up_w) == (out_h, out_w):
        return maps
    return np.ascontiguousarray(maps[..., :out_h, :out_w])

def load_dump(path):
    data = np.load(path, allow_pickle=False)
    mid, last = data["mid"], data["last"]
    geo = read_geometry(data, mid.shape[1:], PATCH)
    print(f"{path.name}: {len(data['label'])} images, patch grid "
          f"{geo['grid'][0]}x{geo['grid'][1]} ({geo['valid'][0]}x{geo['valid'][1]} unpadded), "
          f"native {geo['native'][0]}x{geo['native'][1]} at upscale {geo['upscale']}")
    return data, mid, last, geo
