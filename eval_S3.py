# eval.py
import os
import re
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from dataclasses import asdict
from typing import Callable, Dict, Optional, Tuple, List

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _to_u8(x_3chw_01: torch.Tensor) -> np.ndarray:
    """
    x: [3,H,W] in [0,1]
    return: [H,W,3] uint8
    """
    x = x_3chw_01.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (x * 255.0 + 0.5).astype(np.uint8)


def _save_u8(path: str, rgb_u8: np.ndarray) -> None:
    Image.fromarray(rgb_u8, mode="RGB").save(path)


def _save_gray01(path: str, m_1hw_01: torch.Tensor) -> None:
    """
    m: [1,H,W] or [H,W] in [0,1]
    """
    if m_1hw_01.ndim == 3:
        m = m_1hw_01[0]
    else:
        m = m_1hw_01
    m = m.detach().clamp(0, 1).cpu().numpy()
    img = (m * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def _list_pngs(root: str) -> List[str]:
    """
    Recursively list png files under root.
    Keeps stable ordering for reproducibility.
    """
    if not os.path.isdir(root):
        return []
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".png"):
                out.append(os.path.join(r, fn))
    out.sort()
    return out


def _infer_pair_hr_path(lr_path: str, lr_root: str, hr_root: str) -> str:
    """
    Keep the relative path under lr_root (if any) and map to hr_root.
    """
    rel = os.path.relpath(lr_path, lr_root)
    return os.path.join(hr_root, rel)


# -----------------------------
# Checkpoint discovery
# -----------------------------
def find_best_or_latest_ckpt_dir(output_dir: str, ckpt_root_name: str = "checkpoints") -> Optional[str]:
    """
    Priority:
      1) <output_dir>/<ckpt_root_name>/best   (if exists and has controlnet/)
      2) latest step_<N> with controlnet/
    """
    ckpt_root = os.path.join(output_dir, ckpt_root_name)
    if not os.path.isdir(ckpt_root):
        return None

    # best dir convention
    best_dir = os.path.join(ckpt_root, "best")
    if os.path.isdir(os.path.join(best_dir, "controlnet")):
        return best_dir

    # fallback: latest step_*
    step_re = re.compile(r"^step_(\d+)$")
    best_step = -1
    best_path = None
    for name in os.listdir(ckpt_root):
        m = step_re.match(name)
        if not m:
            continue
        step = int(m.group(1))
        d = os.path.join(ckpt_root, name)
        if not os.path.isdir(os.path.join(d, "controlnet")):
            continue
        if step > best_step:
            best_step = step
            best_path = d

    return best_path


# -----------------------------
# Public: sampling evaluation on loaders
# -----------------------------
def eval_on_loaders_with_trainer_validate(
    trainer,
    val_loader=None,
    test_loader=None,
    out_dir: Optional[str] = None,
    max_batches: int = 10,
    sample_tag: str = "eval",
) -> Dict[str, Dict[str, float]]:
    """
    Use the trainer.validate() to keep EXACTLY the same validation strategy as training.
    Returns dict with "val"/"test" metrics.
    Optionally write metrics to disk.
    """
    results: Dict[str, Dict[str, float]] = {}

    if out_dir is not None:
        _ensure_dir(out_dir)

    if val_loader is not None:
        m = trainer.validate(val_loader, step=0, max_batches=max_batches)
        results["val"] = {k: float(v) for k, v in m.items()}
        if out_dir is not None:
            with open(os.path.join(out_dir, f"{sample_tag}_val_metrics.txt"), "w") as f:
                for k, v in results["val"].items():
                    f.write(f"{k}: {v}\n")

    if test_loader is not None:
        m = trainer.validate(test_loader, step=0, max_batches=max_batches)
        results["test"] = {k: float(v) for k, v in m.items()}
        if out_dir is not None:
            with open(os.path.join(out_dir, f"{sample_tag}_test_metrics.txt"), "w") as f:
                for k, v in results["test"].items():
                    f.write(f"{k}: {v}\n")

    return results


# -----------------------------
# Public: generic SR inference (for fair comparisons)
# -----------------------------
@torch.no_grad()
def sr_infer_directory_generic(
    sr_fn: Callable[[torch.Tensor], torch.Tensor],
    lr_root: str,
    out_root: str,
    hr_root: Optional[str] = None,
    device: str = "cuda",
    limit: Optional[int] = None,
) -> None:
    """
    Generic directory SR runner for method comparison.

    sr_fn:
      - input: lr_3chw_01 torch.Tensor on device (e.g., [3,128,128] in [0,1])
      - output: sr_3chw_01 torch.Tensor on device (e.g., [3,512,512] in [0,1])

    lr_root:
      - directory containing LR pngs (can be flat or nested; this minimal version supports flat)
    out_root:
      - output dir, will create:
          out_root/SR/*.png
          out_root/HR/*.png (if hr_root provided and file exists)

    Note:
      If your LR structure is nested (slide/patch.png), you can extend _list_pngs to recursive walk.
    """
    out_sr = _ensure_dir(os.path.join(out_root, "SR"))
    out_hr = _ensure_dir(os.path.join(out_root, "HR")) if hr_root else None

    lr_paths = _list_pngs(lr_root)
    if limit is not None:
        lr_paths = lr_paths[: int(limit)]

    for p in lr_paths:
        fn = os.path.basename(p)
        lr = Image.open(p).convert("RGB")
        lr_np = np.array(lr, dtype=np.uint8)
        lr_t = torch.from_numpy(lr_np).permute(2, 0, 1).float() / 255.0
        lr_t = lr_t.to(device)

        sr_t = sr_fn(lr_t)  # [3,H,W] in [0,1]
        sr_u8 = _to_u8(sr_t)
        _save_u8(os.path.join(out_sr, fn), sr_u8)

        if hr_root and out_hr is not None:
            hr_path = _infer_pair_hr_path(p, lr_root, hr_root)
            if os.path.isfile(hr_path):
                shutil.copy2(hr_path, os.path.join(out_hr, fn))


# -----------------------------
# Public: OUR method SR inference with extra maps
# -----------------------------
@torch.no_grad()
def sr_infer_directory_ours(
    trainer,
    lr_root: str,
    out_root: str,
    hr_root: Optional[str] = None,
    disc_map_root: Optional[str] = None,
    device: str = "cuda",
    num_steps: int = 20,
    limit: Optional[int] = None,
    save_control: bool = True,
    save_dino_energy: bool = True,
    save_disc_map: bool = True,
    save_sr_raw: bool = True,          # ✅ 新增：是否额外保存未校正SR
    save_sr_cc: bool = True,           # ✅ 新增：是否保存校正SR（建议保持 True）
) -> None:
    """
    Our method runner:
      - SR generation using trainer.sample_sr()
      - Apply color correction using ONLY LR_up as reference (trainer._apply_color_correction_batch)
      - Save SR_cc as default SR output (and optionally SR_raw)
      - Save CONTROL + DINO energy + SAM disc maps (if available)
      - Copy HR if hr_root provided
    """
    out_sr = _ensure_dir(os.path.join(out_root, "SR")) if save_sr_cc else None
    out_sr_raw = _ensure_dir(os.path.join(out_root, "SR_raw")) if save_sr_raw else None
    out_hr = _ensure_dir(os.path.join(out_root, "HR")) if hr_root else None
    out_ctrl = _ensure_dir(os.path.join(out_root, "CONTROL")) if save_control else None
    out_energy = _ensure_dir(os.path.join(out_root, "DINO_ENERGY")) if save_dino_energy else None
    out_disc = _ensure_dir(os.path.join(out_root, "SAM_DISC")) if save_disc_map else None

    lr_paths = _list_pngs(lr_root)
    if limit is not None:
        lr_paths = lr_paths[: int(limit)]

    ctrl_gen = getattr(trainer, "ctrl_gen", None)

    for idx, p in enumerate(lr_paths):
        if idx == 0 or (idx + 1) % 10 == 0:
            print(f"[infer] {idx+1}/{len(lr_paths)}: {p}", flush=True)

        fn = os.path.basename(p)
        stem = os.path.splitext(fn)[0]

        lr = Image.open(p).convert("RGB")
        lr_np = np.array(lr, dtype=np.uint8)
        lr_t = torch.from_numpy(lr_np).permute(2, 0, 1).float() / 255.0
        lr_t = lr_t.unsqueeze(0).to(device)  # [1,3,H,W]

        # bicubic upsample LR for control & color correction reference
        lr_up = F.interpolate(lr_t, size=(512, 512), mode="bicubic", align_corners=False)  # [1,3,512,512]

        # SR sampling (raw)
        sr = trainer.sample_sr(lr_t, num_steps=num_steps)  # [1,3,512,512] in [0,1]

        # ✅ Color correction: use ONLY LR_up as reference (matches your current policy)
        try:
            sr_cc = trainer._apply_color_correction_batch(sr, lr_up)
        except Exception as e:
            # fallback: if cc fails, still allow saving raw
            print(f"[WARN] color correction failed for {fn}: {e}", flush=True)
            sr_cc = sr

        # ---- save SR ----
        if out_sr_raw is not None:
            sr_u8 = _to_u8(sr[0])
            _save_u8(os.path.join(out_sr_raw, fn), sr_u8)

        if out_sr is not None:
            srcc_u8 = _to_u8(sr_cc[0])
            _save_u8(os.path.join(out_sr, fn), srcc_u8)

        # ---- HR copy (optional) ----
        if hr_root and out_hr is not None:
            hr_path = _infer_pair_hr_path(p, lr_root, hr_root)
            if os.path.isfile(hr_path):
                shutil.copy2(hr_path, os.path.join(out_hr, fn))

        # ---- CONTROL map (the actual controlnet cond) ----
        if save_control and out_ctrl is not None:
            try:
                ctrl = trainer._build_control_eval(lr_up)  # validation/eval build
                ctrl_u8 = _to_u8(ctrl[0])
                _save_u8(os.path.join(out_ctrl, fn), ctrl_u8)
            except Exception as e:
                print(f"[WARN] save CONTROL failed for {fn}: {e}", flush=True)

        # ---- DINO energy (guidance) map ----
        if save_dino_energy and out_energy is not None and ctrl_gen is not None:
            try:
                if hasattr(ctrl_gen, "_extract_feature_map") and hasattr(ctrl_gen, "_feat_to_energy"):
                    feat = ctrl_gen._extract_feature_map(lr_up)
                    energy = ctrl_gen._feat_to_energy(feat)
                    if energy.ndim == 4:
                        e = energy[0]  # [1,H,W]
                    elif energy.ndim == 3:
                        e = energy
                    else:
                        e = None
                    if e is not None:
                        _save_gray01(os.path.join(out_energy, f"{stem}.png"), e)
            except Exception as e:
                print(f"[WARN] save DINO energy failed for {fn}: {e}", flush=True)

        # ---- SAM discrepancy map (disc_map) if exists ----
        if save_disc_map and out_disc is not None and disc_map_root:
            pt_path = os.path.join(disc_map_root, f"{stem}.pt")
            if os.path.isfile(pt_path):
                try:
                    d = torch.load(pt_path, map_location="cpu")
                    if torch.is_tensor(d):
                        dd = d.float()
                        if dd.ndim == 2:
                            dd = dd.unsqueeze(0)
                        dd = dd.clamp(0, 1)
                        _save_gray01(os.path.join(out_disc, f"{stem}.png"), dd)
                except Exception as e:
                    print(f"[WARN] save SAM disc failed for {fn}: {e}", flush=True)
      
        


# -----------------------------
# Public: load our trainer for evaluation
# -----------------------------
def load_ours_trainer_from_ckpt(
    cfg,
    ckpt_dir: str,
    device: str = "cuda",
    token: Optional[str] = None,
    strict_full: bool = False,
):
    """
    Instantiate trainer and load checkpoint.
    cfg: SRTrainConfig instance
    """
    # local import to avoid circulars when used in other scripts
    from src.sr_diffusion_trainer import DiffusionSRControlNetTrainer

    # make sure device is set
    cfg.device = device

    trainer = DiffusionSRControlNetTrainer(cfg, token=token)
    trainer.load_checkpoint(ckpt_dir, strict_full=strict_full)

    return trainer


def load_ours_trainer_best_or_latest(
    cfg,
    output_dir: str,
    ckpt_root_name: str = "checkpoints",
    device: str = "cuda",
    token: Optional[str] = None,
    strict_full: bool = False,
):
    ckpt_dir = find_best_or_latest_ckpt_dir(output_dir, ckpt_root_name=ckpt_root_name)
    if ckpt_dir is None:
        raise RuntimeError(f"No checkpoint found under: {os.path.join(output_dir, ckpt_root_name)}")
    trainer = load_ours_trainer_from_ckpt(cfg, ckpt_dir, device=device, token=token, strict_full=strict_full)
    return trainer, ckpt_dir


if __name__ == "__main__":
    import os
    import time
    import yaml
    from easydict import EasyDict

    from src.loader import build_case_split_dataloaders
    from src.sr_diffusion_trainer import SRTrainConfig, DiffusionSRControlNetTrainer

    def _ts():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def _mark(msg: str):
        print(f"[{_ts()}][EVAL] {msg}", flush=True)

    _mark("eval.py start")

    # -----------------------------
    # 1) Load config.yml
    # -----------------------------
    t0 = time.time()
    _mark("Loading config.yml ...")
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    _mark(f"Loaded config.yml in {time.time() - t0:.2f}s")

    # -----------------------------
    # 2) Build dataloaders
    # -----------------------------
    t0 = time.time()
    _mark("Building dataloaders ... (this may take time if directory is large)")
    train_loader, val_loader, test_loader = build_case_split_dataloaders(
        out_img_dir=config.data_loader.out_img_dir,
        batch_size=config.trainer.batch_size,
        patch_num=getattr(config.data_loader, "patch_num", 200),
        train_ratio=config.data_loader.train_ratio,
        val_ratio=config.data_loader.val_ratio,
        test_ratio=config.data_loader.test_ratio,
        split_seed=getattr(config.data_loader, "seed", 2025),
        num_workers=config.data_loader.num_workers,
        pin_memory=config.data_loader.pin_memory,
    )
    _mark(f"Dataloaders ready in {time.time() - t0:.2f}s")
    # 这三行能立刻告诉你是不是 loader 构建后实际为空 / 极小
    _mark(f"len(train_loader)={len(train_loader)} len(val_loader)={len(val_loader)} len(test_loader)={len(test_loader)}")

    # 预取一个 batch，看是否卡在 DataLoader worker / IO
    try:
        _mark("Fetching 1 batch from val_loader to sanity check ...")
        _t = time.time()
        _b = next(iter(val_loader))
        _mark(f"Fetched 1 val batch in {time.time() - _t:.2f}s. keys={list(_b.keys())}")
    except Exception as e:
        _mark(f"[WARN] Failed to fetch val batch: {e}")

    # -----------------------------
    # 3) Build SRTrainConfig like main.py
    # -----------------------------
    train_out_dir = getattr(config.sr, "output_dir", "./outputs/sr_controlnet")
    _mark(f"train_out_dir={train_out_dir}")

    tcfg = SRTrainConfig(
        pretrained_sd15=getattr(config.sr, "pretrained_sd15", "runwayml/stable-diffusion-v1-5"),
        output_dir=train_out_dir,  # for ckpt discovery
        lr=float(getattr(config.sr, "lr", 1e-5)),
        train_steps=int(getattr(config.sr, "train_steps", 20000)),
        grad_accum=int(getattr(config.sr, "grad_accum", 1)),
        mixed_precision=getattr(config.sr, "mixed_precision", "fp16"),
        save_every=int(getattr(config.sr, "save_every", 2000)),
        device=getattr(config.sr, "device", "cuda"),
        val_every=int(getattr(config.sr, "val_every", 100)),
        val_batches=int(getattr(config.sr, "val_batches", 10)),
        sample_steps=int(getattr(config.sr, "sample_steps", 20)),
        local_dir=getattr(config.sr.vfm, "local_dir", "./src/models/dinov2_vitb14"),
        save_full_ckpt=getattr(config.sr, "save_full_ckpt", False),
        val_vis_keep=getattr(config.sr, "val_vis_keep", 5),
        vfm_unfreeze_last_blocks=int(getattr(config.sr.vfm, "vfm_unfreeze_last_blocks", 2)),
        vfm_warmup_steps=int(getattr(config.sr.vfm, "vfm_warmup_steps", 0)),
        vfm_lr=float(getattr(config.sr.vfm, "vfm_lr", 1e-6)),
        enable_color_correction=bool(getattr(config.sr, "enable_color_correction", False)),
        color_correction_ref=str(getattr(config.sr, "color_correction_ref", "lr")),
        color_lowpass_ksize=int(getattr(config.sr, "color_lowpass_ksize", 51)),
    )

    # -----------------------------
    # 4) Create trainer then load checkpoint
    # -----------------------------
    t0 = time.time()
    _mark("Initializing DiffusionSRControlNetTrainer (may load SD/DINO weights) ...")
    trainer = DiffusionSRControlNetTrainer(tcfg, token=getattr(config.sr, "token", None))
    _mark(f"Trainer initialized in {time.time() - t0:.2f}s")

    valer = getattr(config, "valer", EasyDict({}))
    ckpt_tag = str(getattr(valer, "ckpt_tag", "best")).lower()
    if ckpt_tag not in ["best", "latest"]:
        ckpt_tag = "best"

    ckpt_root = os.path.join(train_out_dir, "checkpoints")
    ckpt_dir = os.path.join(ckpt_root, ckpt_tag)
    _mark(f"Resolving checkpoint: ckpt_root={ckpt_root} ckpt_tag={ckpt_tag}")

    if not os.path.isdir(os.path.join(ckpt_dir, "controlnet")):
        alt = "latest" if ckpt_tag == "best" else "best"
        ckpt_dir_alt = os.path.join(ckpt_root, alt)
        if os.path.isdir(os.path.join(ckpt_dir_alt, "controlnet")):
            ckpt_dir = ckpt_dir_alt
            ckpt_tag = alt
        else:
            raise RuntimeError(f"No valid checkpoint found under: {ckpt_root} (need best/ or latest/ with controlnet/)")

    _mark(f"Loading checkpoint from: {ckpt_dir}")
    t0 = time.time()
    trainer.load_checkpoint(ckpt_dir, strict_full=False)
    _mark(f"Checkpoint loaded in {time.time() - t0:.2f}s")

    # -----------------------------
    # 4.5) Switch eval output dir
    # -----------------------------
    eval_out_dir = str(getattr(valer, "eval_output_dir", "")).strip()
    if not eval_out_dir:
        eval_out_dir = os.path.join(train_out_dir, "eval_runs", f"ckpt_{ckpt_tag}")
    os.makedirs(eval_out_dir, exist_ok=True)

    trainer.cfg.output_dir = eval_out_dir
    tcfg.output_dir = eval_out_dir
    os.makedirs(os.path.join(tcfg.output_dir, "val_vis"), exist_ok=True)
    os.makedirs(os.path.join(tcfg.output_dir, "eval_logs"), exist_ok=True)
    _mark(f"Eval outputs will be written to: {tcfg.output_dir}")

    # -----------------------------
    # 5) Loader evaluation
    # -----------------------------
    max_batches = int(getattr(valer, "max_batches", 10))
    do_val = bool(getattr(valer, "do_val", True))
    do_test = bool(getattr(valer, "do_test", True))
    eval_log_dir = os.path.join(tcfg.output_dir, "eval_logs")

    if do_val:
        _mark(f"Running validate() on val_loader: max_batches={max_batches}")
        t0 = time.time()
        m_val = trainer.validate(val_loader, step=0, max_batches=max_batches)
        _mark(f"validate(val) finished in {time.time() - t0:.2f}s. metrics={m_val}")
        with open(os.path.join(eval_log_dir, f"val_metrics_{ckpt_tag}.txt"), "w") as f:
            for k, v in m_val.items():
                f.write(f"{k}: {v}\n")

    if do_test:
        _mark(f"Running validate() on test_loader: max_batches={max_batches}")
        t0 = time.time()
        m_test = trainer.validate(test_loader, step=0, max_batches=max_batches)
        _mark(f"validate(test) finished in {time.time() - t0:.2f}s. metrics={m_test}")
        with open(os.path.join(eval_log_dir, f"test_metrics_{ckpt_tag}.txt"), "w") as f:
            for k, v in m_test.items():
                f.write(f"{k}: {v}\n")

    # -----------------------------
    # 6) Directory inference (optional)
    # -----------------------------
    do_dir_infer = bool(getattr(valer, "do_dir_infer", False))
    if do_dir_infer:
        lr_root = str(getattr(valer, "infer_lr_root", "")).strip()
        if not lr_root:
            raise RuntimeError("valer.do_dir_infer=True but valer.infer_lr_root is empty.")

        hr_root = str(getattr(valer, "infer_hr_root", "")).strip()
        hr_root = hr_root if hr_root else None

        disc_map_root = str(getattr(valer, "infer_disc_map_root", "")).strip()
        disc_map_root = disc_map_root if disc_map_root else None

        infer_out = str(getattr(valer, "infer_out", "")).strip()
        if not infer_out:
            infer_out = os.path.join(tcfg.output_dir, "eval_infer")

        infer_steps = int(getattr(valer, "infer_steps", getattr(tcfg, "sample_steps", 20)))
        infer_limit = int(getattr(valer, "infer_limit", 0))
        infer_limit = None if infer_limit <= 0 else infer_limit

        save_control = bool(getattr(valer, "save_control", True))
        save_dino_energy = bool(getattr(valer, "save_dino_energy", True))
        save_disc_map = bool(getattr(valer, "save_disc_map", True))

        os.makedirs(infer_out, exist_ok=True)
        _mark(f"Directory inference: lr_root={lr_root} -> infer_out={infer_out} steps={infer_steps} limit={infer_limit}")

        t0 = time.time()
        sr_infer_directory_ours(
            trainer=trainer,
            lr_root=lr_root,
            out_root=infer_out,
            hr_root=hr_root,
            disc_map_root=disc_map_root,
            device=getattr(tcfg, "device", "cuda"),
            num_steps=infer_steps,
            limit=infer_limit,
            save_control=save_control,
            save_dino_energy=save_dino_energy,
            save_disc_map=save_disc_map,
        )
        _mark(f"Directory inference done in {time.time() - t0:.2f}s: {infer_out}")

    _mark("eval.py done")
