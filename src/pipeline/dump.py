import argparse
import time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch

from .config import OBJECTS, SPLITS, PATCH, TILE, TILE_BATCH, OVERLAP, AMP, add_object_arg, get_paths
from .data import get_images, pick_files, read_images
from .model import DTYPES, load_model, get_errors
from .tiling import cut_tiles, pad_image, get_padded_size, stitch_tiles, make_plan, get_valid_cells

def add_run_args(parser):
    add_object_arg(parser)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--tile-batch", type=int, default=TILE_BATCH)
    parser.add_argument("--reduce", choices=("max", "mean"), default=OVERLAP)
    parser.add_argument("--amp", choices=sorted(DTYPES), default=AMP)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)

def get_args():
    parser = argparse.ArgumentParser()
    add_run_args(parser)
    parser.add_argument("--upscale", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS), choices=list(SPLITS))
    return parser.parse_args()

def autocast(device, amp_dtype):
    return torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()

@torch.no_grad()
def run_tiles(model, img, plan, tile, tile_batch, reduce, device, amp_dtype, upscale=1):
    rows, cols, shape = plan
    img, padded = pad_image(img, PATCH, upscale)
    if (padded[0] // PATCH, padded[1] // PATCH) != tuple(shape):
        raise ValueError(f"image prepares to {padded} -> grid {(padded[0] // PATCH, padded[1] // PATCH)}, "
                         f"but the tile plan expects {tuple(shape)}; the split has mixed image sizes")

    tiles = cut_tiles(img, rows, cols, tile)
    grid = (tile // PATCH, tile // PATCH)

    mids, lasts = [], []
    for start in range(0, len(tiles), tile_batch):
        batch = tiles[start:start + tile_batch].to(device, non_blocking=True)
        with autocast(device, amp_dtype):
            mid, last = get_errors(model, batch, grid)
        mids.append(mid.float().cpu().numpy())
        lasts.append(last.float().cpu().numpy())

    mid = stitch_tiles(np.concatenate(mids), rows, cols, PATCH, shape, reduce)
    last = stitch_tiles(np.concatenate(lasts), rows, cols, PATCH, shape, reduce)
    return mid, last

def get_geometry(native_hw, name, tile, upscale):
    recorded = OBJECTS.get(name, {}).get("native")
    if recorded and tuple(recorded) != tuple(native_hw):
        print(f"  NOTE: {name} images are {native_hw}, but config.OBJECTS records "
              f"{tuple(recorded)} -- update the registry")

    padded = get_padded_size(native_hw, PATCH, upscale)
    plan = make_plan(*padded, tile, PATCH)
    valid = get_valid_cells(native_hw, PATCH, upscale)
    rows, cols, shape = plan
    scaled = tuple(n * upscale for n in native_hw)
    scale_txt = f", upscaled {upscale}x to {scaled[0]}x{scaled[1]}" if upscale > 1 else ""
    pad_txt = "" if padded == scaled else f", edge-padded to {padded[0]}x{padded[1]}"
    print(f"  {name}: {native_hw[0]}x{native_hw[1]} native{scale_txt}{pad_txt}")
    print(f"  tiling: {len(rows)}x{len(cols)} = {len(rows) * len(cols)} tiles of {tile}px "
          f"-> patch grid {shape[0]}x{shape[1]}, {valid[0]}x{valid[1]} unpadded cells "
          f"({PATCH / upscale:g} native px per token)")
    return plan, padded, valid

def dump_errors():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj_dir, out_dir = get_paths(args)
    out_path = args.out or (out_dir / "errors.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(ckpt=args.ckpt).to(device)

    mids, lasts, labels, roles, conditions, names = [], [], [], [], [], []
    plan = native = padded = valid = None
    t0 = time.time()

    for split in args.splits:
        folder, label, role = SPLITS[split]
        files = pick_files(get_images(obj_dir / folder), args.limit)
        print(f"{split:<12} {len(files):>4} images", flush=True)

        for i, img, name, condition in read_images(files, device):
            shape = (int(img.shape[1]), int(img.shape[2]))
            if plan is None:
                native = shape
                plan, padded, valid = get_geometry(native, args.object_name, args.tile, args.upscale)
            elif shape != native:
                raise ValueError(f"{name} is {shape}, inconsistent with the first image ({native})")

            mid, last = run_tiles(model, img, plan, args.tile, args.tile_batch,
                                  args.reduce, device, DTYPES[args.amp], args.upscale)
            mids.append(mid)
            lasts.append(last)
            labels.append(label)
            roles.append(role)
            conditions.append(condition)
            names.append(name)

            if i % 25 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  ({len(names) / (time.time() - t0):.2f} img/s)", flush=True)

    np.savez_compressed(
        out_path,
        mid=np.stack(mids).astype(np.float32),
        last=np.stack(lasts).astype(np.float32),
        label=np.asarray(labels, dtype=np.int64),
        role=np.asarray(roles),
        condition=np.asarray(conditions),
        name=np.asarray(names),
        object=np.asarray(args.object_name),
        tile=np.int64(args.tile),
        patch=np.int64(PATCH),
        upscale=np.int64(args.upscale),
        native_hw=np.asarray(native, dtype=np.int64),
        padded_hw=np.asarray(padded, dtype=np.int64),
        valid_cells=np.asarray(valid, dtype=np.int64),
        reduce=np.asarray(args.reduce),
        ckpt=np.asarray(str(args.ckpt)),
    )
    print(f"wrote {len(names)} images to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) "
          f"in {(time.time() - t0) / 60:.1f} min")

if __name__ == "__main__":
    dump_errors()
