# src/sr_diffusion_trainer.py
import os
import re
import json
import shutil
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# 可选：huggingface 镜像加速（如果 cfg.pretrained_sd15 指向本地目录，不影响）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from src.sr_metrics import SRMetrics

# ✅ M1: VFM control (DINOv2)
from src.vfm_control import VFMConfig, VFMControlGenerator


@dataclass
class SRTrainConfig:
    # SD1.5：repo_id（需联网）或本地目录（推荐离线）
    pretrained_sd15: str = "runwayml/stable-diffusion-v1-5"
    val_vis_keep: int = 5
    output_dir: str = "./outputs/sr_controlnet"
    lr: float = 1e-5
    train_steps: int = 20000
    grad_accum: int = 1
    mixed_precision: str = "fp16"  # "no" / "fp16" / "bf16"
    save_every: int = 2000
    device: str = "cuda"

    # VFM control
    local_dir: str = "./src/models/dinov2_vitb14"

    # checkpoint
    save_full_ckpt: bool = False  # True: 保存 optimizer/scaler/step；False: 仅保存 controlnet 权重
    ckpt_root: str = "checkpoints"  # 子目录名：{output_dir}/{ckpt_root}/step_XXXXXXXX

    # validation（step-based）
    val_every: int = 0
    val_batches: int = 10
    sample_steps: int = 20  # validate/sample 时 DDPM 采样步数（越大越慢）


def make_control_image_from_lr_fallback(lr_up: torch.Tensor) -> torch.Tensor:
    """
    兜底控制图（不依赖 SAM，不依赖 VFM）。
    lr_up: [B,3,512,512] in [0,1]
    return: [B,3,512,512] => (gray, edge, gray)
    """
    gray = lr_up.mean(dim=1, keepdim=True)  # [B,1,H,W]
    dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = torch.clamp(dx + dy, 0.0, 1.0)
    ctrl = torch.cat([gray, edge, gray], dim=1)
    return ctrl


class DiffusionSRControlNetTrainer:
    def __init__(self, cfg: SRTrainConfig, token: Optional[str] = None):
        self.cfg = cfg
        os.makedirs(cfg.output_dir, exist_ok=True)

        self.device = torch.device(cfg.device)
        self.metrics = SRMetrics(device=self.device)

        # ----------------------------
        # ✅ M1: init DINOv2 control generator (frozen)
        # ----------------------------
        self.vfm_cfg = VFMConfig(
            variant="dinov2_vitb14",
            local_dir=cfg.local_dir,
            image_size=518,
            patch_size=14,
            control_mode="energy_edge_gray",
            normalize=True,
        )
        self.ctrl_gen = VFMControlGenerator(self.vfm_cfg, device=self.device)
        self._warned_control_fallback = False
        print(f"[VFM] enabled: {self.vfm_cfg.variant} | local_dir={self.vfm_cfg.local_dir}")

        # ----------------------------
        # SD1.5 components
        # ----------------------------
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.pretrained_sd15, subfolder="tokenizer", token=token
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.pretrained_sd15, subfolder="text_encoder", token=token
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            cfg.pretrained_sd15, subfolder="vae", token=token
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_sd15, subfolder="unet", token=token
        ).to(self.device)

        # ControlNet init from UNet
        self.controlnet = ControlNetModel.from_unet(self.unet).to(self.device)

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.pretrained_sd15, subfolder="scheduler", token=token
        )

        # freeze base
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

        # train controlnet only
        self.controlnet.train()
        self.optimizer = AdamW(self.controlnet.parameters(), lr=float(cfg.lr))

        # AMP
        self.use_amp = (cfg.mixed_precision in ("fp16", "bf16"))
        self.autocast_dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16
        # ✅ torch.amp.GradScaler recommended in your env warning
        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision == "fp16"))

        # cache empty prompt
        self.empty_prompt_embeds = self._encode_text([""])

    # ----------------------------
    # helpers
    # ----------------------------
    @torch.no_grad()
    def _encode_text(self, prompts):
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(self.device)
        return self.text_encoder(**tokens).last_hidden_state  # [B,77,768]

    @torch.no_grad()
    def _vae_encode(self, x_01: torch.Tensor) -> torch.Tensor:
        x = x_01 * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.mode()
        return latents * 0.18215

    @torch.no_grad()
    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / 0.18215
        img = self.vae.decode(latents).sample
        img = (img + 1.0) / 2.0
        return img.clamp(0, 1)

    def _build_control(self, lr_up_512: torch.Tensor) -> torch.Tensor:
        """
        lr_up_512: [B,3,512,512] in [0,1]
        return: [B,3,512,512] in [0,1]
        """
        try:
            return self.ctrl_gen(lr_up_512)
        except Exception as e:
            if not self._warned_control_fallback:
                print(f"[WARN] VFM control failed once, fallback to lr-edge control. err={e}")
                self._warned_control_fallback = True
            return make_control_image_from_lr_fallback(lr_up_512)

    # ----------------------------
    # checkpoint utils
    # ----------------------------
    def _ckpt_root_dir(self) -> str:
        return os.path.join(self.cfg.output_dir, getattr(self.cfg, "ckpt_root", "checkpoints"))

    def _ckpt_dir(self, step: int) -> str:
        return os.path.join(self._ckpt_root_dir(), f"step_{step:08d}")

    def _find_latest_checkpoint(self) -> Optional[str]:
        ckpt_root = self._ckpt_root_dir()
        if not os.path.isdir(ckpt_root):
            return None

        # ✅ New: prefer "latest" dir (single latest checkpoint)
        latest_dir = os.path.join(ckpt_root, "latest")
        if os.path.isdir(os.path.join(latest_dir, "controlnet")):
            # if full ckpt required, check trainer_state.pt
            if getattr(self.cfg, "save_full_ckpt", False):
                if os.path.isfile(os.path.join(latest_dir, "trainer_state.pt")):
                    return latest_dir
            else:
                return latest_dir

        # Fallback: legacy step_XXXXXXXX dirs (keep for backward compatibility)
        step_re = re.compile(r"^step_(\d+)$")
        best_step = -1
        best_dir = None

        for name in os.listdir(ckpt_root):
            m = step_re.match(name)
            if not m:
                continue
            step = int(m.group(1))
            d = os.path.join(ckpt_root, name)

            if not os.path.isdir(os.path.join(d, "controlnet")):
                continue

            if getattr(self.cfg, "save_full_ckpt", False):
                if not os.path.isfile(os.path.join(d, "trainer_state.pt")):
                    continue

            if step > best_step:
                best_step = step
                best_dir = d

        return best_dir


    def _prune_val_vis(self):
        keep = int(getattr(self.cfg, "val_vis_keep", 0) or 0)
        if keep <= 0:
            return

        vis_base = os.path.join(self.cfg.output_dir, "val_vis")
        if not os.path.isdir(vis_base):
            return

        step_re = re.compile(r"^step_(\d+)$")
        items = []
        for name in os.listdir(vis_base):
            m = step_re.match(name)
            if not m:
                continue
            step = int(m.group(1))
            d = os.path.join(vis_base, name)
            if os.path.isdir(d):
                items.append((step, d))

        # 少于等于 keep，无需清理
        if len(items) <= keep:
            return

        # 按 step 从小到大排序，删最旧的
        items.sort(key=lambda x: x[0])
        to_delete = items[: max(0, len(items) - keep)]

        for step, d in to_delete:
            try:
                shutil.rmtree(d)
                print(f"[val_vis] pruned old vis: step={step} dir={d}")
            except Exception as e:
                print(f"[WARN] failed to prune val_vis dir={d}: {e}")
    
    def save_checkpoint(self, step: int, ckpt_name: str = "latest", extra_meta: Optional[Dict] = None):
        """
        ckpt_name:
        - "latest": ✅ always overwrite, keep only the newest checkpoint
        - "best"  : ✅ overwrite when best PSNR improves
        - legacy/other: still supported, will write to {ckpt_root}/{ckpt_name}
        """
        ckpt_root = self._ckpt_root_dir()
        os.makedirs(ckpt_root, exist_ok=True)

        ckpt_dir = os.path.join(ckpt_root, str(ckpt_name))
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1) controlnet
        self.controlnet.save_pretrained(os.path.join(ckpt_dir, "controlnet"))

        # 2) full state (optional)
        if getattr(self.cfg, "save_full_ckpt", False):
            payload = {
                "global_step": int(step),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            }
            torch.save(payload, os.path.join(ckpt_dir, "trainer_state.pt"))

        # 3) meta
        meta = {"global_step": int(step), "ckpt_name": str(ckpt_name)}
        if isinstance(extra_meta, dict):
            meta.update(extra_meta)

        with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[ckpt] saved ({ckpt_name}) checkpoint to {ckpt_dir}")

    def load_checkpoint(self, ckpt_dir: str, strict_full: bool = True) -> int:
        """
        ckpt_dir: .../step_XXXXXXXX
        strict_full:
          - True: 如果 cfg.save_full_ckpt=True，则要求 trainer_state.pt 存在
          - False: 只要 controlnet 存在就能加载（用于“加载权重继续训但不恢复优化器”）
        return: global_step
        """
        if not ckpt_dir or (not os.path.isdir(ckpt_dir)):
            raise FileNotFoundError(f"ckpt_dir not found: {ckpt_dir}")

        controlnet_dir = os.path.join(ckpt_dir, "controlnet")
        if not os.path.isdir(controlnet_dir):
            raise FileNotFoundError(f"controlnet dir missing: {controlnet_dir}")

        # 1) load controlnet weights (replace module)
        self.controlnet = ControlNetModel.from_pretrained(controlnet_dir).to(self.device)
        self.controlnet.train()

        # 2) IMPORTANT: rebuild optimizer to bind NEW controlnet params
        self.optimizer = AdamW(self.controlnet.parameters(), lr=float(self.cfg.lr))

        # default step from folder name
        m = re.search(r"step_(\d+)", os.path.basename(ckpt_dir))
        global_step = int(m.group(1)) if m else 0

        # 3) load full state if available/wanted
        state_path = os.path.join(ckpt_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            payload = torch.load(state_path, map_location="cpu")
            global_step = int(payload.get("global_step", global_step))

            if payload.get("optimizer", None) is not None:
                self.optimizer.load_state_dict(payload["optimizer"])

            if payload.get("scaler", None) is not None and self.scaler is not None:
                try:
                    self.scaler.load_state_dict(payload["scaler"])
                except Exception as e:
                    print(f"[WARN] scaler state load failed, will reset scaler. err={e}")
        else:
            if getattr(self.cfg, "save_full_ckpt", False) and strict_full:
                raise FileNotFoundError(f"trainer_state.pt missing: {state_path}")
            print("[ckpt] trainer_state.pt not found, loaded weights only (optimizer/scaler reset).")

        print(f"[ckpt] loaded from {ckpt_dir}, resume global_step={global_step}")
        return global_step

    # ----------------------------
    # training
    # ----------------------------
    def train(self, train_loader, val_loader=None, resume_ckpt: Optional[str] = None):
        cfg = self.cfg

        # -------------------------
        # resume (user-specified > auto-latest > none)
        # -------------------------
        global_step = 0
        ckpt_used = None

        if resume_ckpt:  # user specified
            ckpt_used = resume_ckpt
        else:
            ckpt_used = self._find_latest_checkpoint()

        if ckpt_used:
            try:
                strict_full = bool(getattr(cfg, "save_full_ckpt", False))
                global_step = int(self.load_checkpoint(ckpt_used, strict_full=strict_full))
            except Exception as e:
                print(f"[WARN] resume failed from {ckpt_used}. Start from scratch. err={e}")
                global_step = 0
        else:
            print("[ckpt] no checkpoint found. Start from scratch.")

        if global_step >= int(cfg.train_steps):
            print(f"[WARN] global_step({global_step}) >= train_steps({cfg.train_steps}). Nothing to train.")
            return

        # -------------------------
        # best PSNR state (persisted)
        # -------------------------
        ckpt_root = self._ckpt_root_dir()
        best_state_path = os.path.join(ckpt_root, "best_score.json")
        best_psnr = float("-inf")
        best_step = -1
        if os.path.isfile(best_state_path):
            try:
                with open(best_state_path, "r") as f:
                    d = json.load(f)
                best_psnr = float(d.get("best_psnr", best_psnr))
                best_step = int(d.get("best_step", best_step))
                print(f"[best] loaded best_psnr={best_psnr:.6f} @ step={best_step}")
            except Exception as e:
                print(f"[WARN] failed to read best_score.json, reset best. err={e}")

        self.best_psnr = best_psnr
        self.best_step = best_step

        # -------------------------
        # validation schedule
        # -------------------------
        val_every = int(getattr(cfg, "val_every", 0) or 0)
        val_batches = int(getattr(cfg, "val_batches", 10) or 10)
        do_val = (val_loader is not None) and (val_every > 0)

        # -------------------------
        # progress bar
        # -------------------------
        pbar = tqdm(
            total=int(cfg.train_steps),
            desc="train",
            unit="step",
            initial=int(global_step),
        )

        data_iter = iter(train_loader)

        # -------------------------
        # main loop
        # -------------------------
        while global_step < int(cfg.train_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            lr = batch["lr"].to(self.device, non_blocking=True)  # [B,3,128,128]
            hr = batch["hr"].to(self.device, non_blocking=True)  # [B,3,512,512]

            # control
            lr_up = F.interpolate(lr, size=(512, 512), mode="bicubic", align_corners=False)
            control_image = self._build_control(lr_up)  # [B,3,512,512]

            # target latents
            with torch.no_grad():
                latents = self._vae_encode(hr)

            bsz = latents.shape[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=self.device,
                dtype=torch.long,
            )
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            encoder_hidden_states = self.empty_prompt_embeds.repeat(bsz, 1, 1)

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.autocast_dtype):
                down_samples, mid_sample = self.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control_image,
                    return_dict=False,
                )

                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") / int(cfg.grad_accum)

            # backward
            if cfg.mixed_precision == "fp16":
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # optimizer step (grad accum)
            if (global_step + 1) % int(cfg.grad_accum) == 0:
                if cfg.mixed_precision == "fp16":
                    self.scaler.unscale_(self.optimizer)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            # step update
            global_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": float(loss.detach().cpu())})

            # validation (also handles best checkpoint inside validate)
            if do_val and (global_step % val_every == 0):
                try:
                    metrics = self.validate(val_loader, step=global_step, max_batches=val_batches)
                    if isinstance(metrics, dict) and metrics:
                        show = {f"val_{k}": float(v) for k, v in metrics.items()}
                        pbar.set_postfix({**pbar.postfix, **show})
                except Exception as e:
                    print(f"[WARN] validate failed at step={global_step}: {e}")

            # ✅ checkpoint: keep only latest (overwrite)
            if (global_step % int(cfg.save_every) == 0):
                self.save_checkpoint(global_step, ckpt_name="latest")

        pbar.close()
        # ✅ final save: overwrite latest
        self.save_checkpoint(global_step, ckpt_name="latest")

    # ----------------------------
    # sampling
    # ----------------------------
    @torch.no_grad()
    def sample_sr(self, lr: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        self.controlnet.eval()

        num_steps = int(num_steps or getattr(self.cfg, "sample_steps", 20))
        lr_up = F.interpolate(lr, size=(512, 512), mode="bicubic", align_corners=False)
        control = self._build_control(lr_up)

        B = lr.shape[0]
        latents = torch.randn((B, 4, 64, 64), device=self.device)

        self.noise_scheduler.set_timesteps(num_steps, device=self.device)

        for t in self.noise_scheduler.timesteps:
            encoder_hidden_states = self.empty_prompt_embeds.repeat(B, 1, 1)

            down, mid = self.controlnet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=control,
                return_dict=False,
            )

            noise_pred = self.unet(
                latents,
                t,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=down,
                mid_block_additional_residual=mid,
                return_dict=False,
            )[0]

            latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

        sr = self._vae_decode(latents)
        self.controlnet.train()
        return sr

    # ----------------------------
    # validation + save SR/GT/control
    # ----------------------------
    @torch.no_grad()
    def validate(self, val_loader, step: int = 0, max_batches: int = 10) -> Dict[str, float]:
        self.controlnet.eval()

        psnr_list, ssim_list, lpips_list, stlpips_list = [], [], [], []

        # save one random example
        save_vis = True
        vis_batch_idx = np.random.randint(0, max(1, int(max_batches)))
        vis_saved = False

        vis_root = os.path.join(self.cfg.output_dir, "val_vis", f"step_{step:07d}")
        os.makedirs(vis_root, exist_ok=True)

        def to_u8(img_3chw: torch.Tensor) -> np.ndarray:
            x = img_3chw.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            x = (x * 255.0 + 0.5).astype(np.uint8)
            return x

        num_steps = int(getattr(self.cfg, "sample_steps", 20))

        for i, batch in enumerate(val_loader):
            if i >= int(max_batches):
                break

            lr = batch["lr"].to(self.device, non_blocking=True)
            hr = batch["hr"].to(self.device, non_blocking=True)

            lr_up = F.interpolate(lr, size=(512, 512), mode="bicubic", align_corners=False)
            control = self._build_control(lr_up)

            # inline sampling (avoid state flips)
            B = lr.shape[0]
            latents = torch.randn((B, 4, 64, 64), device=self.device)

            self.noise_scheduler.set_timesteps(num_steps, device=self.device)

            for t in self.noise_scheduler.timesteps:
                encoder_hidden_states = self.empty_prompt_embeds.repeat(B, 1, 1)
                down, mid = self.controlnet(
                    latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control,
                    return_dict=False,
                )
                noise_pred = self.unet(
                    latents,
                    t,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid,
                    return_dict=False,
                )[0]
                latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample

            sr = self._vae_decode(latents)

            # metrics
            psnr_list.append(float(self.metrics.psnr(sr, hr)))
            ssim_list.append(float(self.metrics.ssim(sr, hr)))
            lpips_list.append(float(self.metrics.lpips(sr, hr)))
            stlpips_list.append(float(self.metrics.shift_tolerant_lpips(sr, hr)))

            # save one example
            if save_vis and (not vis_saved) and (i == vis_batch_idx):
                j = int(np.random.randint(0, B))

                Image.fromarray(to_u8(sr[j])).save(os.path.join(vis_root, "SR.png"))
                Image.fromarray(to_u8(hr[j])).save(os.path.join(vis_root, "GT.png"))
                Image.fromarray(to_u8(lr_up[j])).save(os.path.join(vis_root, "LR_up.png"))
                Image.fromarray(to_u8(control[j])).save(os.path.join(vis_root, "CONTROL.png"))

                with open(os.path.join(vis_root, "metrics.txt"), "w") as f:
                    f.write(f"step: {step}\n")
                    f.write(f"batch_idx: {i}\n")
                    f.write(f"sample_idx: {j}\n")
                    f.write(f"sample_steps: {num_steps}\n")
                    f.write(f"PSNR: {psnr_list[-1]:.6f}\n")
                    f.write(f"SSIM: {ssim_list[-1]:.6f}\n")
                    f.write(f"LPIPS: {lpips_list[-1]:.6f}\n")
                    f.write(f"ST-LPIPS: {stlpips_list[-1]:.6f}\n")

                vis_saved = True
                print(f"[val] saved SR/GT/LR_up/CONTROL to: {vis_root}")
                self._prune_val_vis()

        self.controlnet.train()

        results = {
            "PSNR": float(np.mean(psnr_list)) if psnr_list else 0.0,
            "SSIM": float(np.mean(ssim_list)) if ssim_list else 0.0,
            "LPIPS": float(np.mean(lpips_list)) if lpips_list else 0.0,
            "ST-LPIPS": float(np.mean(stlpips_list)) if stlpips_list else 0.0,
        }

        print(
            f"[val @ step {step}] "
            f"PSNR={results['PSNR']:.3f} | "
            f"SSIM={results['SSIM']:.4f} | "
            f"LPIPS={results['LPIPS']:.4f} | "
            f"ST-LPIPS={results['ST-LPIPS']:.4f}"
        )

        # ✅ Always overwrite "latest" checkpoint (keep only newest)
        try:
            self.save_checkpoint(step, ckpt_name="latest", extra_meta={"val_metrics": results})
        except Exception as e:
            print(f"[WARN] failed to save latest ckpt at val step={step}: {e}")

        # ✅ Save best by PSNR (overwrite "best" only when improved)
        try:
            cur_psnr = float(results.get("PSNR", 0.0))
            best_psnr = float(getattr(self, "best_psnr", float("-inf")))
            if cur_psnr > best_psnr:
                self.best_psnr = cur_psnr
                self.best_step = int(step)

                self.save_checkpoint(step, ckpt_name="best", extra_meta={"best_psnr": cur_psnr, "best_step": int(step)})

                best_state_path = os.path.join(self._ckpt_root_dir(), "best_score.json")
                with open(best_state_path, "w") as f:
                    json.dump({"best_psnr": float(cur_psnr), "best_step": int(step)}, f, indent=2)

                print(f"[best] updated best checkpoint: PSNR={cur_psnr:.6f} @ step={step}")
        except Exception as e:
            print(f"[WARN] best checkpoint update failed at step={step}: {e}")

        return results
