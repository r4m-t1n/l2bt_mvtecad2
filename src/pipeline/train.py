import os
import argparse
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .config import (SPLITS, BACKBONE, LAYERS, TILE, STEPS, BATCH_SIZE, LR, WORKERS, AMP,
                     COLOR_JITTER, CKPT_EVERY, LOG_EVERY, SEED, add_object_arg, get_paths)
from .data import CropDataset, get_images
from .dump import autocast
from .model import DTYPES, load_model, get_loss, get_student_params, is_student

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_args():
    parser = argparse.ArgumentParser()
    add_object_arg(parser)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--tile", type=int, default=TILE)
    parser.add_argument("--upscale", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=WORKERS)
    parser.add_argument("--amp", choices=sorted(DTYPES), default=AMP)
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--vflip", action="store_true")
    parser.add_argument("--color-jitter", type=float, default=COLOR_JITTER)
    parser.add_argument("--ckpt-every", type=int, default=CKPT_EVERY)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--full-ckpt", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()

def start_training():
    args = get_args()
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = DTYPES[args.amp]

    obj_dir, out_dir = get_paths(args)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    files = get_images(obj_dir / SPLITS["train_good"][0])
    if args.limit:
        files = files[:args.limit]

    dataset = CropDataset(files, args.tile, args.steps * args.batch_size, upscale=args.upscale,
                          hflip=args.hflip, vflip=args.vflip, jitter=args.color_jitter)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        drop_last=True, persistent_workers=args.num_workers > 0)

    model = load_model(ckpt=args.resume).to(device)
    model.model.backward_net.train()
    model.model.forward_net.train()

    optimizer = torch.optim.Adam(get_student_params(model), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    run_info = {"object": args.object_name, "tile": args.tile, "upscale": args.upscale, "train_images": len(files),
                "backbone": BACKBONE, "layers": list(LAYERS)}
    for key in ("steps", "batch_size", "lr", "num_workers", "amp", "hflip", "vflip", "color_jitter", "ckpt_every"):
        run_info[key] = getattr(args, key)
    run_info["log_every"] = LOG_EVERY
    run_info["seed"] = args.seed
    (out_dir / "train_config.json").write_text(json.dumps(run_info, indent=2))
    upscale_txt = f" upscaled {args.upscale}x to {args.tile}px" if args.upscale > 1 else ""
    print(f"training on {len(files)} {args.object_name} images, {args.steps} steps x batch "
          f"{args.batch_size} of {dataset.crop}px native crops{upscale_txt} -> {out_dir}")

    log_path = out_dir / "train_log.jsonl"
    log_file = log_path.open("a")
    running = 0.0
    t0 = time.time()

    def save(tag, step):
        path = ckpt_dir / f"{tag}.ckpt"
        state = model.state_dict()
        if not args.full_ckpt:
            state = {k: v for k, v in state.items() if is_student(k)}
        torch.save({"state_dict": state, "step": step, "config": run_info}, path)
        print(f"  saved {path.name} ({path.stat().st_size / 1e6:.0f} MB)")

    for step, images in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device, amp_dtype):
            loss, loss_mid, loss_last = get_loss(model, images)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item()
        if step % LOG_EVERY == 0:
            mean = running / LOG_EVERY
            running = 0.0
            rate = step * args.batch_size / (time.time() - t0)
            record = {"step": step, "loss": mean, "loss_mid": loss_mid.item(),
                      "loss_last": loss_last.item(), "img_per_s": round(rate, 2)}
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(f"step {step:>6}/{args.steps}  loss {mean:.5f}  "
                  f"(mid {loss_mid.item():.5f} / last {loss_last.item():.5f})  {rate:.1f} img/s")

        if step % args.ckpt_every == 0:
            save(f"step_{step:06d}", step)

        if step >= args.steps:
            break

    save("last", args.steps)
    log_file.close()
    print(f"done in {(time.time() - t0) / 60:.1f} min; log at {log_path}")

if __name__ == "__main__":
    start_training()
