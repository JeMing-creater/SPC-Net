# eval_all.py
from ast import Pass
import os
import glob
import json
import random
from dataclasses import asdict
from typing import Optional, Dict, Tuple, Callable, Any, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# --- your project imports ---
from src.sr_diffusion_trainer import SRTrainConfig, DiffusionSRControlNetTrainer


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)



# -----------------------------
# Loader builder (hyperparam-based)
# -----------------------------
def build_loaders_by_hparams(
    out_img_dir: str,
    batch_size: int = 4,
    patch_num: int = 200,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    split_seed: int = 2025,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    require_done: bool = True,
    shuffle_train: bool = True,
    disc_root: Optional[str] = None,
):
    """
    Wrapper around src.loader.build_case_split_dataloaders using the correct signature.

    IMPORTANT:
      - Use out_img_dir (not data_root).
      - out_img_dir should contain: hr_png/, lr_png/, clinical.tsv (and optionally disc_maps/).
    """
    from src import loader as loader_mod

    if not hasattr(loader_mod, "build_case_split_dataloaders"):
        raise RuntimeError("src.loader.build_case_split_dataloaders not found.")

    return loader_mod.build_case_split_dataloaders(
        out_img_dir=out_img_dir,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=split_seed,
        patch_num=patch_num,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        require_done=require_done,
        shuffle_train=shuffle_train,
        disc_root=disc_root,
    )


# -----------------------------
# Checkpoint helpers
# -----------------------------
def load_trainer_for_ours(
    cfg: SRTrainConfig,
    ckpt_path: str,
    device: str = "cuda",
    strict: bool = False,
) -> DiffusionSRControlNetTrainer:
    """
    Follow main.py-like flow: instantiate trainer, then load weights.
    """
    cfg.device = device
    trainer = DiffusionSRControlNetTrainer(cfg, token=getattr(cfg, "token", None))
    # your trainer should provide load_checkpoint
    if hasattr(trainer, "load_checkpoint"):
        trainer.load_checkpoint(ckpt_path, strict_full=strict)
    else:
        raise AttributeError("trainer has no load_checkpoint(). Please add/confirm it exists.")
    return trainer


# -----------------------------
# Validation (match trainer.validate)
# -----------------------------

def make_infer_fn_ours(
    trainer,
    sample_steps: int = 20,
    do_color_cc: bool = True,
    force_out_size: Optional[int] = 512,   # None: keep model output size
):
    """
    Unified infer_fn signature:
        sr = infer_fn(lr, hr, meta, device)

    - Works for validation (hr provided) and folder inference (hr may be None).
    - Encapsulates ALL logic: sample_sr + optional color correction + optional resize.
    """
    @torch.no_grad()
    def infer_fn(
        lr: torch.Tensor,
        hr: Optional[torch.Tensor],
        meta: Dict[str, Any],
        device: torch.device,
    ) -> torch.Tensor:
        lr = lr.to(device, non_blocking=True)

        steps = int(meta.get("sample_steps", sample_steps))

        # 1) diffusion SR
        sr = trainer.sample_sr(lr, num_steps=steps).clamp(0, 1)

        # 2) determine target size for alignment
        if hr is not None:
            target_hw = hr.shape[-2:]
        elif force_out_size is not None:
            target_hw = (int(force_out_size), int(force_out_size))
        else:
            target_hw = sr.shape[-2:]

        # 3) resize SR if needed
        if sr.shape[-2:] != target_hw:
            sr = F.interpolate(sr, size=target_hw, mode="bicubic", align_corners=False)

        # 4) color correction (reference from LR_up only, consistent with your current policy)
        if do_color_cc and hasattr(trainer, "_apply_color_correction_batch"):
            lr_up = F.interpolate(lr, size=target_hw, mode="bicubic", align_corners=False)
            sr = trainer._apply_color_correction_batch(sr, lr_up).clamp(0, 1)

        return sr

    return infer_fn


@torch.no_grad()
def eval_with_validate(
    loader,
    infer_fn: Callable[[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], torch.device], torch.Tensor],
    device: torch.device,
    max_batches: int = 10,
    split_name: str = "val",
    print_every: int = 10,
    meta_defaults: Optional[Dict[str, Any]] = None,
    method: str = "S3_Diff",
) -> Dict[str, float]:
    """
    Only:
      - iterate loader
      - sr = infer_fn(lr, hr, meta, device)
      - compute metrics vs hr
    """
    if meta_defaults is None:
        meta_defaults = {}

    try:
        from src.sr_metrics import SRMetrics
    except Exception:
        from sr_metrics import SRMetrics

    metrics = SRMetrics(device=device)

    psnr_list, ssim_list, lpips_list, stlpips_list = [], [], [], []
    n_batches = 0

    print(f"\n[eval] split={split_name} max_batches={max_batches}", flush=True)

    for bi, batch in enumerate(loader):
        print(f"[eval] processing batch {bi+1} ...", flush=True)
        if bi >= int(max_batches):
            break
        
        # unpack
        if isinstance(batch, dict):
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            meta = {k: v for k, v in batch.items() if k not in ["lr", "hr"]}
        else:
            lr, hr = batch
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)
            meta = {}

        # merge meta defaults
        meta = dict(meta_defaults, **meta)

        # infer SR
        if method == "S3_Diff":
            sr = infer_fn(lr, hr, meta, device)
        else:
            sr = infer_fn(lr, hr)
        
        
        sr = sr.clamp(0, 1)
        hr = hr.clamp(0, 1)

        if sr.shape != hr.shape:
            raise ValueError(f"sr/hr shape mismatch after infer_fn: sr={tuple(sr.shape)} hr={tuple(hr.shape)}")

        # metrics
        psnr_list.append(float(metrics.psnr(sr, hr)))
        ssim_list.append(float(metrics.ssim(sr, hr)))
        lpips_list.append(float(metrics.lpips(sr, hr)))
        stlpips_list.append(float(metrics.shift_tolerant_lpips(sr, hr)))

        n_batches += 1

        if print_every and (bi + 1) % int(print_every) == 0:
            print(f"[eval] {split_name} progress {bi+1}/{min(len(loader), max_batches)}", flush=True)

    out = {
        "PSNR": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "SSIM": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "LPIPS": float(np.mean(lpips_list)) if lpips_list else 0.0,
        "ST-LPIPS": float(np.mean(stlpips_list)) if stlpips_list else 0.0,
        "batches": int(n_batches),
    }
    print(f"[eval] {split_name} metrics: {out}", flush=True)
    return out

# -----------------------------
# Inference on a LR folder
# -----------------------------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def _list_imgs(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)
    return paths


def _img_to_tensor01(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    return t


def _tensor01_to_u8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.shape[0] == 3:
        x = x.permute(1, 2, 0)
    return (x.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def _infer_pair_hr_path(lr_path: str, lr_root: str, hr_root: str) -> str:
    """
    Keep relative path: hr_root/<relpath of lr under lr_root>.
    This is robust when lr_root contains nested folders.
    """
    rel = os.path.relpath(lr_path, lr_root)
    return os.path.join(hr_root, rel)



def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def _list_imgs(folder: str) -> List[str]:
    exts = ("*.png","*.jpg","*.jpeg","*.tif","*.tiff")
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(folder, e)))
    return sorted(out)

def _img_to_tensor01(path: str, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(device)

def _tensor01_to_u8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().clamp(0,1)
    if x.ndim == 4: x = x[0]
    if x.shape[0] == 3: x = x.permute(1,2,0)
    return (x.cpu().numpy()*255.0 + 0.5).astype(np.uint8)

def _infer_pair_hr_path(lr_path: str, lr_root: str, hr_root: str) -> str:
    rel = os.path.relpath(lr_path, lr_root)
    return os.path.join(hr_root, rel)

@torch.no_grad()
def infer_folder_lr_to_sr_generic(
    lr_dir: str,
    out_root: str,
    name: str,
    infer_fn: Callable[[torch.Tensor, Optional[torch.Tensor], Dict[str, Any], torch.device], torch.Tensor],
    device: torch.device,
    hr_dir: Optional[str] = None,
    lr_up_size: int = 512,
    limit: Optional[int] = None,
    meta_defaults: Optional[Dict[str, Any]] = None,
):
    """
    Save only: LR_up, SR, (optional HR) using the SAME infer_fn as validation.
    """
    if meta_defaults is None:
        meta_defaults = {}

    lr_dir = os.path.abspath(lr_dir)
    case_id = os.path.basename(os.path.normpath(lr_dir))

    base_out = os.path.join(out_root, "eval_image", name, case_id)
    out_lr_up = _ensure_dir(os.path.join(base_out, "LR_up"))
    out_sr = _ensure_dir(os.path.join(base_out, "SR"))
    out_hr = _ensure_dir(os.path.join(base_out, "HR")) if hr_dir else None

    lr_paths = _list_imgs(lr_dir)
    if limit is not None and limit > 0:
        lr_paths = lr_paths[: int(limit)]
    if len(lr_paths) == 0:
        raise RuntimeError(f"No images found under: {lr_dir}")

    print(f"[infer] lr_dir={lr_dir} num={len(lr_paths)} out={base_out}", flush=True)

    for idx, lr_path in enumerate(lr_paths):
        print(f"Processing {lr_path} ...", flush=True)
        rel = os.path.relpath(lr_path, lr_dir)
        stem = os.path.splitext(os.path.basename(lr_path))[0]
        subdir = os.path.dirname(rel)

        # make subdirs
        if subdir:
            os.makedirs(os.path.join(out_lr_up, subdir), exist_ok=True)
            os.makedirs(os.path.join(out_sr, subdir), exist_ok=True)
            if out_hr: os.makedirs(os.path.join(out_hr, subdir), exist_ok=True)

        # load LR
        lr = _img_to_tensor01(lr_path, device=device)

        # optional HR load (for saving only, not mandatory)
        hr = None
        if hr_dir:
            hr_path = _infer_pair_hr_path(lr_path, lr_dir, hr_dir)
            if os.path.isfile(hr_path):
                hr = _img_to_tensor01(hr_path, device=device)

        meta = dict(meta_defaults)
        meta["lr_path"] = lr_path
        if hr_dir:
            meta["hr_path"] = _infer_pair_hr_path(lr_path, lr_dir, hr_dir)

        # infer SR (same function as validation)
        if name == "S3_Diff":
            sr = infer_fn(lr, hr, meta, device).clamp(0,1)
        else:
            sr = infer_fn(lr, hr).clamp(0,1)

        # save LR_up (always size=lr_up_size for visualization)
        lr_up = F.interpolate(lr, size=(lr_up_size, lr_up_size), mode="bicubic", align_corners=False)
        Image.fromarray(_tensor01_to_u8(lr_up)).save(os.path.join(out_lr_up, subdir, f"{stem}.png"))

        # save SR (also resize to lr_up_size for consistent viewing)
        if sr.shape[-2:] != (lr_up_size, lr_up_size):
            sr_save = F.interpolate(sr, size=(lr_up_size, lr_up_size), mode="bicubic", align_corners=False)
        else:
            sr_save = sr
        Image.fromarray(_tensor01_to_u8(sr_save)).save(os.path.join(out_sr, subdir, f"{stem}.png"))

        # save HR if available
        if out_hr is not None and hr is not None:
            hr_save = hr
            if hr_save.shape[-2:] != (lr_up_size, lr_up_size):
                hr_save = F.interpolate(hr_save, size=(lr_up_size, lr_up_size), mode="bicubic", align_corners=False)
            Image.fromarray(_tensor01_to_u8(hr_save)).save(os.path.join(out_hr, subdir, f"{stem}.png"))

        if (idx + 1) % 20 == 0:
            print(f"[infer] {idx+1}/{len(lr_paths)}", flush=True)

    print(f"[infer] done. saved to {base_out}", flush=True)
    return base_out




# ----------models------------
# load (model here)
# ----------models------------
def S3_Diff(OUT_ROOT, METHOD_NAME, OURS_OUTPUT_DIR, CKPT_PREFER, DEVICE, SAMPLE_STEPS, MAX_BATCHES):
    # Important: cfg.output_dir controls where validate will write val_vis.
    # If you don't want to pollute training folder, set a separate output_dir here.
    
    print("[main] resolving checkpoint ...")
    ckpt_path = os.path.join(OURS_OUTPUT_DIR, CKPT_PREFER)
    print(f"[main] ckpt: {ckpt_path}")
    
    cfg = SRTrainConfig()
    cfg.output_dir = os.path.join(OUT_ROOT, "eval_runs", METHOD_NAME)  # redirect validate vis outputs
    cfg.device = DEVICE
    cfg.sample_steps = SAMPLE_STEPS
    cfg.val_batches = MAX_BATCHES
    cfg.val_vis_keep = 1  # keep minimal vis during eval (set 0 if you later modify validate to respect it)

    ensure_dir(cfg.output_dir)
    print(f"[main] eval output_dir (redirected): {cfg.output_dir}")

    print("[main] loading trainer ...")
    trainer = load_trainer_for_ours(cfg, ckpt_path, device=DEVICE, strict=False)
    print("[main] trainer ready.")
    return trainer




# -----------------------------
# MAIN (hyperparams live here)
# -----------------------------
if __name__ == "__main__":
    # -------------------------
    # 0) hyperparams (edit here)
    # -------------------------
    seed = 2025
    set_seed(seed)

    # data/loader
    # 说明：一些数据加载超参数，不用改
    OUT_IMG_DIR = "/mnt/liangjm/SpRR_data"  # <-- change to yours
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    PATCH_NUM = 300 # 只选择一个样本随机的前300个patch进行训练和验证

    # ours model
    # 说明：选择你训练好的OURS模型进行验证和推理
    OURS_OUTPUT_DIR = "./outputs/sr_controlnet/checkpoints/"  # this is training output_dir that contains checkpoints/
    CKPT_PREFER = "best"  # "best" or "latest"
    DEVICE = "cuda"

    # validate sampling
    MAX_BATCHES = 10
    SAMPLE_STEPS = 20  # override sampling steps for inference (optional)

    # inference single folder
    # 说明：选择一个LR和对应HR文件夹进行推理，生成SR结果，方法名可以写在METHOD_NAME
    LR_FOLDER_TO_INFER = "/mnt/liangjm/SpRR_data/lr_png/TCGA-ZP-A9D4-01Z-00-DX1/"
    HR_FOLDER_TO_INFER = "/mnt/liangjm/SpRR_data/hr_png/TCGA-ZP-A9D4-01Z-00-DX1/"
    OUT_ROOT = "./output"  # will save to ./output/eval_image/{name}/{case_id}/
    METHOD_NAME = "S3_Diff"

    # -------------------------
    # 1) loaders
    # -------------------------
    print("[main] building dataloaders ...")
    train_loader, val_loader, test_loader = build_loaders_by_hparams(
        out_img_dir=OUT_IMG_DIR,
        batch_size=BATCH_SIZE,
        patch_num=PATCH_NUM,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        split_seed=2025,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        require_done=True,
        disc_root=None,  # 或者 "/mnt/liangjm/SpRR_data/disc_maps"
    )
    print("[main] dataloaders ready.")

    # -------------------------
    # 2) load OURS trainer like main.py flow (init -> load weights)
    # -------------------------
    # 加载模型
    S3_Diff = S3_Diff(OUT_ROOT, METHOD_NAME, OURS_OUTPUT_DIR, CKPT_PREFER, DEVICE, SAMPLE_STEPS, MAX_BATCHES)
    # 构建输入输出函数。
    infer_fn_ours = make_infer_fn_ours(S3_Diff, sample_steps=SAMPLE_STEPS, do_color_cc=True, force_out_size=512)

    # # -------------------------
    # # 3) validate on val/test (same logic as training validate)
    # # -------------------------
    m_val = eval_with_validate(val_loader, infer_fn_ours, device=DEVICE, max_batches=10, split_name="val")
    m_test = eval_with_validate(test_loader, infer_fn_ours, device=DEVICE, max_batches=10, split_name="test")
    print(f"[main] {METHOD_NAME} val metrics: {json.dumps(m_val, indent=2)}")
    print(f"[main] {METHOD_NAME} test metrics: {json.dumps(m_test, indent=2)}")
    

    # save metrics
    output_dir = os.path.join(OUT_ROOT, "eval_runs", METHOD_NAME)
    ensure_dir(output_dir)
    with open(os.path.join(output_dir, "metrics_val.json"), "w") as f:
        json.dump(m_val, f, indent=2)
    with open(os.path.join(output_dir, "metrics_test.json"), "w") as f:
        json.dump(m_test, f, indent=2)
    print(f"[main] saved metrics to: {output_dir}")

    # -------------------------
    # 4) infer a LR folder -> SR (save SR_cc + CONTROL + optional DINO guidance)
    # -------------------------
    infer_folder_lr_to_sr_generic(
        lr_dir=LR_FOLDER_TO_INFER,
        hr_dir=HR_FOLDER_TO_INFER,
        out_root="./output",
        name=METHOD_NAME,
        infer_fn=infer_fn_ours,
        device=DEVICE,
        lr_up_size=512,
    )

    print("[main] eval_all done.")
