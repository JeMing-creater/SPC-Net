import os
import math
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor

from .loader import PathologySRSurvivalDataset


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _to_numpy_rgb(img_01: torch.Tensor) -> np.ndarray:
    """
    img_01: [3,H,W] float in [0,1] -> uint8 RGB [H,W,3]
    """
    x = img_01.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def _risk_from_time_event(time_val: float, event_val: int, eps: float = 1e-6) -> float:
    """
    r in [0,1]
    - event=0 -> r=0 (censored)
    - event=1 -> larger when time is shorter
    """
    if int(event_val) != 1:
        return 0.0
    inv = 1.0 / (float(time_val) + eps)
    r = inv / (inv + 0.01)  # stable squashing without dataset stats
    return float(np.clip(r, 0.0, 1.0))


def _schedule_by_risk(r: float) -> Tuple[int, float]:
    """
    survival-aware prompting schedule:
      - num_points: 4 -> 32
      - score_thresh: 0.90 (strict) -> 0.55 (loose) when risk increases
    """
    num_points = int(round(4 + r * 28))
    num_points = int(np.clip(num_points, 4, 32))

    score_thresh = float(0.90 - r * 0.35)
    score_thresh = float(np.clip(score_thresh, 0.55, 0.90))
    return num_points, score_thresh


def _make_grid_points(h: int, w: int, n: int) -> np.ndarray:
    """
    grid-like points in (x,y), shape [n,2]
    """
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    g = int(math.ceil(math.sqrt(n)))
    xs = np.linspace(0.1 * w, 0.9 * w, g)
    ys = np.linspace(0.1 * h, 0.9 * h, g)
    pts = np.array([(x, y) for y in ys for x in xs], dtype=np.float32)
    return pts[:n]


def _masks_from_points(
    predictor: SamPredictor,
    image_uint8: np.ndarray,
    points_xy: np.ndarray,
    score_thresh: float,
) -> List[np.ndarray]:
    """
    Returns list of binary masks (H,W) uint8 {0,1}.
    """
    predictor.set_image(image_uint8)

    point_labels = np.ones((points_xy.shape[0],), dtype=np.int32)  # all foreground

    masks, scores, _ = predictor.predict(
        point_coords=points_xy,
        point_labels=point_labels,
        multimask_output=True,
    )

    keep: List[np.ndarray] = []
    for m, s in zip(masks, scores):
        if float(s) >= float(score_thresh):
            keep.append((m > 0).astype(np.uint8))

    # ensure not empty
    if not keep and len(masks) > 0:
        j = int(np.argmax(scores))
        keep = [(masks[j] > 0).astype(np.uint8)]

    return keep




def _discrepancy_from_mask_sets(
    m_hr: List[np.ndarray],
    lr_up_01: torch.Tensor,
    hr_01: torch.Tensor,
    dilate: int = 5,
) -> torch.Tensor:
    """
    m_hr: HR masks list from SAM
    lr_up_01/hr_01: torch [3,H,W] in [0,1]
    return: D [1,H,W] in [0,1]
    """
    H, W = hr_01.shape[-2], hr_01.shape[-1]

    gate = _structure_gate_from_masks(m_hr, H, W, dilate=dilate)  # [1,H,W]
    gate = gate.to(dtype=torch.float32, device=hr_01.device)

    gd = _gradient_discrepancy_map(hr_01, lr_up_01)  # [1,H,W]
    gd = gd.to(dtype=torch.float32, device=hr_01.device)

    D = gate * gd

    # normalize again (optional but makes visualization consistent across patches)
    d_min = D.amin(dim=(1, 2), keepdim=True)
    d_max = D.amax(dim=(1, 2), keepdim=True)
    D = (D - d_min) / (d_max - d_min + 1e-6)

    return D.clamp(0, 1)


def generate_sam_disc_maps(
    out_img_dir: str,
    sam_ckpt: str,
    sam_model_type: str = "vit_b",
    device: str = "cuda",
    patch_num: int = 200,
    require_done: bool = True,
    disc_root: Optional[str] = None,
    max_items: int = -1,
    overwrite: bool = False,
    log_every: int = 50,
) -> str:
    """
    Offline generate discrepancy maps using SAM (v1).

    Output structure (default):
      {out_img_dir}/disc_maps/{slide_id}/{patch_name}.pt

    Returns:
      disc_root path
    """
    if sam_model_type not in ("vit_h", "vit_l", "vit_b"):
        raise ValueError(f"sam_model_type must be one of vit_h/vit_l/vit_b, got {sam_model_type}")

    if not os.path.isdir(out_img_dir):
        raise FileNotFoundError(f"out_img_dir not found: {out_img_dir}")

    if not os.path.isfile(sam_ckpt):
        raise FileNotFoundError(f"sam_ckpt not found: {sam_ckpt}")

    disc_root = disc_root or os.path.join(out_img_dir, "disc_maps")
    _ensure_dir(disc_root)

    # dataset (re-use your loader indexing & survival parsing)
    ds = PathologySRSurvivalDataset(
        out_img_dir=out_img_dir,
        require_done=bool(require_done),
        patch_num=int(patch_num),
    )

    # build SAM
    sam = sam_model_registry[sam_model_type](checkpoint=sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    n = len(ds) if int(max_items) < 0 else min(len(ds), int(max_items))
    print(f"[disc|SAM] items={n} | disc_root={disc_root} | model={sam_model_type} | device={device}")

    for i in range(n):
        batch: Dict[str, Any] = ds[i]
        lr = batch["lr"]     # [3,128,128] in [0,1]
        hr = batch["hr"]     # [3,512,512] in [0,1]
        time = float(batch["time"].item())
        event = int(batch["event"].item())
        meta = batch["meta"]

        slide_id = meta["slide_id"]
        hr_path = meta["hr_path"]
        patch_base = os.path.basename(hr_path)
        out_dir = os.path.join(disc_root, slide_id)
        _ensure_dir(out_dir)
        out_path = os.path.join(out_dir, patch_base.replace(".png", ".pt"))

        if (not overwrite) and os.path.isfile(out_path):
            continue

        # survival-aware schedule
        r = _risk_from_time_event(time, event)
        num_points, score_thresh = _schedule_by_risk(r)

        # LR_up to 512
        lr_up = F.interpolate(lr.unsqueeze(0), size=(512, 512), mode="bicubic", align_corners=False).squeeze(0)

        hr_u8 = _to_numpy_rgb(hr)
        lr_u8 = _to_numpy_rgb(lr_up)

        H, W = hr_u8.shape[0], hr_u8.shape[1]
        pts = _make_grid_points(H, W, num_points)

        # SAM masks
        m_hr = _masks_from_points(predictor, hr_u8, pts, score_thresh=score_thresh)

        # ✅ structure gating (SAM on HR) + gradient discrepancy (HR vs LR_up)
        # lr_up / hr are torch tensors in [0,1]
        D = _discrepancy_from_mask_sets(
            m_hr=m_hr,
            lr_up_01=lr_up,
            hr_01=hr,
            dilate=3,   # 可调：3~9都行，越大 gate 越“粗”
        ).float().cpu()

        torch.save(D, out_path)

        if log_every > 0 and (i + 1) % int(log_every) == 0:
            print(f"[disc|SAM] {i+1}/{n} saved: {out_path} | risk={r:.3f} pts={num_points} thr={score_thresh:.2f}")

    print("[disc|SAM] done.")
    return disc_root


def _gradient_discrepancy_map(a_01: torch.Tensor, b_01: torch.Tensor) -> torch.Tensor:
    """
    a_01, b_01: torch.Tensor [3,H,W] in [0,1]
    return: Dg [1,H,W] in [0,1], larger means larger edge/texture discrepancy
    """
    # to grayscale: [1,1,H,W]
    a = a_01.mean(dim=0, keepdim=True).unsqueeze(0)
    b = b_01.mean(dim=0, keepdim=True).unsqueeze(0)

    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=a.dtype, device=a.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=a.dtype, device=a.device).view(1, 1, 3, 3)

    ax = torch.nn.functional.conv2d(a, kx, padding=1)
    ay = torch.nn.functional.conv2d(a, ky, padding=1)
    bx = torch.nn.functional.conv2d(b, kx, padding=1)
    by = torch.nn.functional.conv2d(b, ky, padding=1)

    ga = torch.sqrt(ax * ax + ay * ay + 1e-12)
    gb = torch.sqrt(bx * bx + by * by + 1e-12)

    d = torch.abs(ga - gb)  # [1,1,H,W]

    # normalize to [0,1]
    d_min = d.amin(dim=(2, 3), keepdim=True)
    d_max = d.amax(dim=(2, 3), keepdim=True)
    d = (d - d_min) / (d_max - d_min + 1e-6)

    return d.squeeze(0)  # [1,H,W]


def _structure_gate_from_masks(
    masks_hr: List[np.ndarray],
    H: int,
    W: int,
    dilate: int = 5,
) -> torch.Tensor:
    """
    masks_hr: list of binary masks (H,W) uint8 {0,1} from HR
    return: gate [1,H,W] float in {0,1} (after optional dilation)
    """
    if len(masks_hr) == 0:
        return torch.ones((1, H, W), dtype=torch.float32)  # fallback: no gating

    gate = torch.zeros((H, W), dtype=torch.float32)
    for m in masks_hr:
        gate += torch.from_numpy((m > 0).astype(np.uint8)).float()
    gate = (gate > 0).float()  # union

    # optional dilation to avoid "too thin" structure coverage
    if dilate and dilate > 0:
        k = int(dilate)
        gate4d = gate.view(1, 1, H, W)
        gate4d = torch.nn.functional.max_pool2d(gate4d, kernel_size=2 * k + 1, stride=1, padding=k)
        gate = gate4d.view(H, W)

    return gate.unsqueeze(0)  # [1,H,W]
