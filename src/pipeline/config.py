import os
from pathlib import Path

DATASET_DIR = Path("data") / "mvtec_ad_2"
HERE = Path(__file__).resolve()

def find_root(rel, start):
    return next((p for p in [start, *start.parents] if (p / rel).is_dir()), None)

def env_path(name, fallback):
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else fallback

REPO_ROOT = env_path("L2BT_ROOT", find_root(DATASET_DIR, HERE.parent) or HERE.parents[3])
DATA_ROOT = env_path("MVTEC_ROOT", REPO_ROOT / DATASET_DIR)
RUNS_ROOT = env_path("L2BT_RUNS", HERE.parent.parent / "runs")
OBJECT = "fabric"

OBJECTS = {
    "can": {"native": (1024, 2232), "upscale": 2},
    "fabric": {"native": (2048, 2448), "upscale": 1},
    "fruit_jelly": {"native": (1520, 2100), "upscale": 1},
    "rice": {"native": (2048, 2448), "upscale": 1},
    "sheet_metal": {"native": (1056, 4224), "upscale": 1},
    "vial": {"native": (1900, 1400), "upscale": 1},
    "wallplugs": {"native": (2048, 2448), "upscale": 1},
    "walnuts": {"native": (2048, 2448), "upscale": 1},
}

SPLITS = {
    "train_good": ("train/good", 0, "train"),
    "val_good": ("validation/good", 0, "val"),
    "test_good": ("test_public/good", 0, "test"),
    "test_bad": ("test_public/bad", 1, "test"),
}
PRIVATE_SPLITS = ("test_private", "test_private_mixed")
MASK_DIR = "test_public/ground_truth/bad"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

BACKBONE = "vit_large_patch16_dinov3.lvd1689m"
LAYERS = (15, 23)
PATCH = 16

TILE = 512
TILE_BATCH = 8
OVERLAP = "max"
UPSCALE = 1

STEPS = 8000
BATCH_SIZE = 8
LR = 0.0005
WORKERS = 8
AMP = "bf16"
COLOR_JITTER = 0.0
CKPT_EVERY = 1000
LOG_EVERY = 50
SEED = 42

PIXEL_SPEC = {"combine": "mul", "norm": "none", "sigma": 0.0}
IMAGE_SPEC = {"combine": "mid", "norm": "pos_z+img_mad", "sigma": 0.0, "k": 1}

IMAGE_RULE = "f1_public"
IMAGE_QUANTILE = 0.99
PIXEL_RULE = "joint"
PIXEL_QUANTILE = 0.9995
AREA_FACTOR = 1.0
GATE = True
MIN_PX = 32

def add_object_arg(parser):
    parser.add_argument("--object", dest="object_name", default=OBJECT)

def get_paths(args):
    if getattr(args, "upscale", None) is None and hasattr(args, "upscale"):
        args.upscale = OBJECTS.get(args.object_name, {}).get("upscale", UPSCALE)
    out_dir = getattr(args, "out_dir", None)
    return DATA_ROOT / args.object_name, Path(out_dir) if out_dir else RUNS_ROOT / f"{args.object_name}_tiled_l15_23"
