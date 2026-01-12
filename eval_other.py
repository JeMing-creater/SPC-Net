# eval_all.py
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


def to_u8(img_01: torch.Tensor) -> np.ndarray:
    """
    img_01: [3,H,W] or [H,W,3], float in [0,1]
    return uint8 HWC
    """
    if isinstance(img_01, torch.Tensor):
        x = img_01.detach().float().clamp(0, 1).cpu()
        if x.ndim == 3 and x.shape[0] == 3:
            x = x.permute(1, 2, 0)
        x = (x.numpy() * 255.0 + 0.5).astype(np.uint8)
        return x
    else:
        x = np.clip(img_01, 0, 1)
        return (x * 255.0 + 0.5).astype(np.uint8)


def load_image_as_tensor_01(path: str, device: torch.device) -> torch.Tensor:
    """
    Load RGB image -> torch float [1,3,H,W] in [0,1]
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def list_pngs(folder: str) -> List[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(paths)
    return paths


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
def resolve_ckpt_path(output_dir: str, prefer: str = "best") -> str:
    """
    Your trainer saves checkpoints under:
      {output_dir}/checkpoints/best/...
      {output_dir}/checkpoints/latest/...
    (based on your earlier structure)
    """
    ckpt_root = os.path.join(output_dir, "checkpoints")
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(f"checkpoint root not found: {ckpt_root}")

    cand_dirs = []
    if prefer.lower() == "best":
        cand_dirs = [os.path.join(ckpt_root, "best"), os.path.join(ckpt_root, "latest")]
    else:
        cand_dirs = [os.path.join(ckpt_root, "latest"), os.path.join(ckpt_root, "best")]

    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        # pick newest *.pt inside
        pts = sorted(glob.glob(os.path.join(d, "*.pt")))
        if len(pts) > 0:
            return pts[-1]

        # or pick newest folder
        sub = sorted(glob.glob(os.path.join(d, "step_*")))
        if len(sub) > 0:
            # common layout: step_xxxxxxxx/controlnet.pt (or something)
            # just pick latest folder and find *.pt inside
            last = sub[-1]
            pts2 = sorted(glob.glob(os.path.join(last, "*.pt")))
            if len(pts2) > 0:
                return pts2[-1]

    raise RuntimeError(f"No checkpoint found under: {ckpt_root} (prefer={prefer})")


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
        trainer.load_checkpoint(ckpt_path, strict=strict)
    else:
        raise AttributeError("trainer has no load_checkpoint(). Please add/confirm it exists.")
    return trainer


# -----------------------------
# Validation (match trainer.validate)
# -----------------------------
@torch.no_grad()
def eval_with_trainer_validate(
    trainer: DiffusionSRControlNetTrainer,
    loader,
    split_name: str = "val",
    max_batches: int = 10,
) -> Dict[str, float]:
    """
    Uses sr_diffusion_trainer.py::validate() directly to ensure identical protocol
    (including color correction and raw metrics).
    """
    print(f"\n[eval] running validate() on {split_name} ... max_batches={max_batches}")
    metrics = trainer.validate(loader, step=0, max_batches=max_batches)
    print(f"[eval] {split_name} metrics: {json.dumps(metrics, indent=2)}")
    return metrics


# -----------------------------
# Inference on a LR folder
# -----------------------------
@torch.no_grad()
def infer_folder_lr_to_sr(
    trainer: DiffusionSRControlNetTrainer,
    lr_dir: str,
    out_root: str,
    method_name: str,
    sample_steps: Optional[int] = None,
    save_control: bool = True,
    save_dino_guidance_if_possible: bool = True,
):
    """
    Generate SR for all LR images in lr_dir, store results in:
      {out_root}/{method_name}/{case_id}/

    Saved SR is SR_cc (color corrected using LR_up reference), consistent with your validate's main metric protocol
    (color corrected SR is the primary PSNR/SSIM/LPIPS/ST-LPIPS reported) :contentReference[oaicite:2]{index=2}
    """
    lr_dir = os.path.abspath(lr_dir)
    case_id = os.path.basename(os.path.normpath(lr_dir))
    out_dir = os.path.join(out_root, "eval_image", method_name, case_id)
    ensure_dir(out_dir)

    lr_paths = list_pngs(lr_dir)
    if len(lr_paths) == 0:
        raise RuntimeError(f"No images found under: {lr_dir}")

    print(f"\n[infer] LR folder: {lr_dir}")
    print(f"[infer] found {len(lr_paths)} LR images")
    print(f"[infer] output_dir: {out_dir}")

    device = trainer.device

    for idx, p in enumerate(lr_paths):
        # load LR
        lr = load_image_as_tensor_01(p, device=device)  # [1,3,H,W]
        # sample SR (raw)
        sr_raw = trainer.sample_sr(lr, num_steps=sample_steps)

        # LR_up (512)
        lr_up = F.interpolate(lr, size=(512, 512), mode="bicubic", align_corners=False)

        # color correction (SR_cc)
        if hasattr(trainer, "_apply_color_correction_batch"):
            sr_cc = trainer._apply_color_correction_batch(sr_raw, lr_up)
        else:
            sr_cc = sr_raw  # fallback

        # save SR_cc
        stem = os.path.splitext(os.path.basename(p))[0]
        Image.fromarray(to_u8(sr_cc[0])).save(os.path.join(out_dir, f"{stem}_SRcc.png"))

        # (optional) save SR_raw for debugging
        Image.fromarray(to_u8(sr_raw[0])).save(os.path.join(out_dir, f"{stem}_SRraw.png"))

        # save LR_up
        Image.fromarray(to_u8(lr_up[0])).save(os.path.join(out_dir, f"{stem}_LRup.png"))

        # For OUR method: try save CONTROL map + DINO guidance (if any API exists)
        if save_control:
            try:
                if hasattr(trainer, "_build_control_eval"):
                    control = trainer._build_control_eval(lr_up)  # [1,3,512,512] (best-effort)
                    c = control[0]
                    if c.ndim == 3 and c.shape[0] == 3:
                        Image.fromarray(to_u8(c)).save(os.path.join(out_dir, f"{stem}_CONTROL.png"))
            except Exception as e:
                print(f"[infer][WARN] save_control failed for {stem}: {e}")

        if save_dino_guidance_if_possible:
            # This is intentionally best-effort: only save if your ctrl_gen exposes something usable.
            try:
                cg = getattr(trainer, "ctrl_gen", None)
                if cg is not None:
                    # common patterns you might later implement:
                    # 1) cg.get_guidance_vis(lr_up) -> [B,3,H,W]
                    # 2) cg.last_guidance_vis (cached)
                    if hasattr(cg, "get_guidance_vis"):
                        g = cg.get_guidance_vis(lr_up)  # expected [1,3,*,*] in [0,1]
                        Image.fromarray(to_u8(g[0])).save(os.path.join(out_dir, f"{stem}_DINO_GUIDE.png"))
                    elif hasattr(cg, "last_guidance_vis"):
                        g = cg.last_guidance_vis
                        if isinstance(g, torch.Tensor) and g.ndim == 4:
                            Image.fromarray(to_u8(g[0])).save(os.path.join(out_dir, f"{stem}_DINO_GUIDE.png"))
            except Exception as e:
                # silent by default; you can print if you want
                pass

        if (idx + 1) % 20 == 0:
            print(f"[infer] done {idx+1}/{len(lr_paths)}")

    print(f"[infer] done. results saved to: {out_dir}")
    return out_dir


# -----------------------------
# Public API for other methods
# -----------------------------
def evaluate_any_model_on_loaders(
    model_name: str,
    sample_fn: Callable[[torch.Tensor], torch.Tensor],
    val_loader,
    device: torch.device,
    max_batches: int = 10,
) -> Dict[str, float]:
    """
    A generic entry you can reuse for other baselines:
    - sample_fn(lr_01[B,3,h,w]) -> sr_01[B,3,512,512]  (or same size as hr)
    - computes PSNR/SSIM/LPIPS/ST-LPIPS using SRMetrics (reuse from trainer codebase)
    """
    from src.sr_metrics import SRMetrics
    metrics = SRMetrics(device=device)

    psnr_list, ssim_list, lpips_list, stlpips_list = [], [], [], []

    print(f"\n[eval_any] evaluating {model_name} ... max_batches={max_batches}")

    for bi, batch in enumerate(val_loader):
        if bi >= max_batches:
            break

        # you project batch keys might be: lr, hr, ...
        # adapt here if your batch format changes
        if isinstance(batch, dict):
            lr = batch["lr"].to(device)
            hr = batch.get("hr", None)
            if hr is not None:
                hr = hr.to(device)
        else:
            # e.g., (lr, hr)
            lr, hr = batch
            lr = lr.to(device)
            hr = hr.to(device)

        sr = sample_fn(lr)

        # metrics assume sr/hr are [B,3,H,W] float [0,1]
        psnr_list.append(float(metrics.psnr(sr, hr)))
        ssim_list.append(float(metrics.ssim(sr, hr)))
        lpips_list.append(float(metrics.lpips(sr, hr)))
        stlpips_list.append(float(metrics.shift_tolerant_lpips(sr, hr)))

    out = {
        "PSNR": float(np.mean(psnr_list)) if psnr_list else 0.0,
        "SSIM": float(np.mean(ssim_list)) if ssim_list else 0.0,
        "LPIPS": float(np.mean(lpips_list)) if lpips_list else 0.0,
        "ST-LPIPS": float(np.mean(stlpips_list)) if stlpips_list else 0.0,
    }
    print(f"[eval_any] {model_name} metrics: {json.dumps(out, indent=2)}")
    return out


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
    OUT_IMG_DIR = "/mnt/liangjm/SpRR_data"  # <-- change to yours
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    PATCH_NUM = 300 # 只选择一个样本随机的前300个patch进行训练和验证

    # ours model
    OURS_OUTPUT_DIR = "./outputs/sr_controlnet"  # this is training output_dir that contains checkpoints/
    CKPT_PREFER = "best"  # "best" or "latest"
    DEVICE = "cuda"

    # validate sampling
    MAX_BATCHES = 10
    SAMPLE_STEPS = 20  # override sampling steps for inference (optional)

    # inference single folder
    LR_FOLDER_TO_INFER = "/mnt/liangjm/SpRR_data/lr_png/TCGA-ZP-A9D4-01Z-00-DX1/"
    OUT_ROOT = "./output"  # will save to ./output/eval_image/{name}/{case_id}/
    METHOD_NAME = "ours"

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

    # # -------------------------
    # # 2) load OURS trainer like main.py flow (init -> load weights)
    # # -------------------------
    # print("[main] resolving checkpoint ...")
    # ckpt_path = resolve_ckpt_path(OURS_OUTPUT_DIR, prefer=CKPT_PREFER)
    # print(f"[main] ckpt: {ckpt_path}")

    # # Important: cfg.output_dir controls where validate will write val_vis.
    # # If you don't want to pollute training folder, set a separate output_dir here.
    # cfg = SRTrainConfig()
    # cfg.output_dir = os.path.join(OUT_ROOT, "eval_runs", METHOD_NAME)  # redirect validate vis outputs
    # cfg.local_dir = getattr(cfg, "local_dir", "")  # make sure you set this properly if offline DINOv2
    # cfg.device = DEVICE
    # cfg.sample_steps = SAMPLE_STEPS
    # cfg.val_batches = MAX_BATCHES
    # cfg.val_vis_keep = 1  # keep minimal vis during eval (set 0 if you later modify validate to respect it)

    # ensure_dir(cfg.output_dir)
    # print(f"[main] eval output_dir (redirected): {cfg.output_dir}")

    # print("[main] loading trainer ...")
    # trainer = load_trainer_for_ours(cfg, ckpt_path, device=DEVICE, strict=False)
    # print("[main] trainer ready.")

    # # -------------------------
    # # 3) validate on val/test (same logic as training validate)
    # # -------------------------
    # m_val = eval_with_trainer_validate(trainer, val_loader, split_name="val", max_batches=MAX_BATCHES)
    # m_test = eval_with_trainer_validate(trainer, test_loader, split_name="test", max_batches=MAX_BATCHES)

    # # save metrics
    # ensure_dir(cfg.output_dir)
    # with open(os.path.join(cfg.output_dir, "metrics_val.json"), "w") as f:
    #     json.dump(m_val, f, indent=2)
    # with open(os.path.join(cfg.output_dir, "metrics_test.json"), "w") as f:
    #     json.dump(m_test, f, indent=2)
    # print(f"[main] saved metrics to: {cfg.output_dir}")

    # # -------------------------
    # # 4) infer a LR folder -> SR (save SR_cc + CONTROL + optional DINO guidance)
    # # -------------------------
    # infer_folder_lr_to_sr(
    #     trainer=trainer,
    #     lr_dir=LR_FOLDER_TO_INFER,
    #     out_root=OUT_ROOT,
    #     method_name=METHOD_NAME,
    #     sample_steps=SAMPLE_STEPS,
    #     save_control=True,
    #     save_dino_guidance_if_possible=True,
    # )

    # print("[main] eval_all done.")
