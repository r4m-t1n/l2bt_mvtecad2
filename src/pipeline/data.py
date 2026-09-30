import re
import csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from .config import MEAN, STD

Image.MAX_IMAGE_PIXELS = None

def get_images(folder):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        raise FileNotFoundError(f"no images found in {folder}")
    return files

def get_condition(path):
    match = re.match(r"^\d+_(.+)$", path.stem)
    return match.group(1) if match else "unknown"

def get_instance(path):
    stem = Path(path).stem
    match = re.match(r"^(\d+)", stem)
    return match.group(1) if match else stem

def load_masks(names, size, mask_dir):
    h, w = size
    masks = np.zeros((len(names), h, w), dtype=bool)
    for i, name in enumerate(names):
        path = mask_dir / f"{Path(name).stem}_mask.png"
        if not path.exists():
            path = mask_dir / name
        if not path.exists():
            raise FileNotFoundError(f"no ground-truth mask for {name} in {mask_dir}")
        masks[i] = np.asarray(Image.open(path).convert("L").resize((w, h), Image.Resampling.BOX)) > 0
    return masks

def pick_files(files, limit):
    if not limit or limit >= len(files):
        return files
    step = len(files) / limit
    return [files[int(i * step)] for i in range(limit)]

def save_png(binary, path):
    Image.fromarray(np.where(binary, 255, 0).astype(np.uint8), mode="L").save(path)

def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

class CropDataset(Dataset):
    def __init__(self, files, tile, length, upscale=1, hflip=False, vflip=False, jitter=0.0):
        if tile % upscale:
            raise ValueError(f"tile ({tile}) must be divisible by upscale ({upscale})")
        self.files = files
        self.length = length
        self.crop = tile // upscale

        steps = [v2.ToImage(), v2.RandomCrop(self.crop)]
        if hflip:
            steps.append(v2.RandomHorizontalFlip())
        if vflip:
            steps.append(v2.RandomVerticalFlip())
        if jitter > 0:
            steps.append(v2.ColorJitter(brightness=jitter, contrast=jitter))
        steps.append(v2.ToDtype(torch.float32, scale=True))
        if upscale > 1:
            steps.append(v2.Resize((tile, tile), antialias=False))
        steps.append(v2.Normalize(mean=list(MEAN), std=list(STD)))
        self.transform = v2.Compose(steps)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.transform(Image.open(self.files[idx % len(self.files)]).convert("RGB"))

class ImageDataset(Dataset):
    def __init__(self, files):
        self.files = files
        self.transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                                     v2.Normalize(mean=list(MEAN), std=list(STD))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        return self.transform(Image.open(path).convert("RGB")), path.name, get_condition(path)

def read_images(files, device):
    loader = DataLoader(ImageDataset(files), batch_size=1, num_workers=4, pin_memory=device.type == "cuda")
    for i, (img, name, condition) in enumerate(loader, start=1):
        yield i, img[0], name[0], condition[0]
