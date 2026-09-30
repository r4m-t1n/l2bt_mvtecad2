from pathlib import Path
import torch
from torch.nn import functional as F
from anomalib.models.components.feature_extractors import TimmFeatureExtractor
from anomalib.models.image import L2BT
from anomalib.models.image.l2bt.torch_model import FeatureProjectionMLP

from .config import BACKBONE, LAYERS, PATCH

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}

def is_student(key):
    return "backward_net" in key or "forward_net" in key

def load_model(ckpt=None):
    model = L2BT(layers=LAYERS, topk_ratio=0.001, pre_processor=False)

    fe = TimmFeatureExtractor(
        backbone=BACKBONE,
        layers=[f"blocks.{i}" for i in LAYERS],
        pre_trained=True,
        requires_grad=False,
        output_fmt="NLC",
        return_class_token=False,
        norm=True,
        dynamic_img_size=True,
    )
    dim = fe.out_dims[0]
    teacher = model.model.teacher
    teacher.fe, teacher.patch_size, teacher.embed_dim = fe, fe.patch_size, dim
    model.model.anomaly_map_generator.patch_size = fe.patch_size
    model.model.backward_net = FeatureProjectionMLP(in_features=dim, out_features=dim)
    model.model.forward_net = FeatureProjectionMLP(in_features=dim, out_features=dim)

    if int(fe.patch_size) != PATCH:
        raise ValueError(f"backbone patch size {fe.patch_size} does not match config patch {PATCH}")

    if ckpt is not None:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        not_loaded = [k for k in missing if is_student(k)]
        if not_loaded:
            raise RuntimeError(f"student weights missing from {ckpt}: {not_loaded[:4]}")
        loaded = [k for k in state if is_student(k)]
        print(f"loaded {Path(ckpt).name}: {len(loaded)} student tensors "
              f"({len(unexpected)} unexpected keys)")

    teacher.requires_grad_(False)
    return model.eval()

def get_student_params(model):
    return list(model.model.backward_net.parameters()) + list(model.model.forward_net.parameters())

def get_loss(model, images):
    with torch.no_grad():
        mid_t, last_t = model.model.extract_teacher_features(images)
    mid_p, last_p = model.model.predict_student_features(mid_t, last_t)

    loss_mid = 1 - F.cosine_similarity(mid_p.float(), mid_t.float(), dim=-1).mean()
    loss_last = 1 - F.cosine_similarity(last_p.float(), last_t.float(), dim=-1).mean()
    return loss_mid + loss_last, loss_mid, loss_last

@torch.no_grad()
def get_errors(model, images, grid):
    mid_t, last_t = model.model.extract_teacher_features(images)
    mid_p, last_p = model.model.predict_student_features(mid_t, last_t)

    def dist(pred, target):
        diff = F.normalize(pred.float(), dim=-1) - F.normalize(target.float(), dim=-1)
        return diff.pow(2).sum(-1).sqrt().view(-1, *grid)

    return dist(mid_p, mid_t), dist(last_p, last_t)
