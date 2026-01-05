# src/sr_diffusion_trainer.py
import os
# 可选：huggingface 镜像加速（如果 cfg.pretrained_sd15 指向本地目录，不影响）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from PIL import Image

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDPMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer

from src.sr_metrics import SRMetrics

# ✅ M1: VFM control (DINOv2)
from src.vfm_control import VFMConfig, VFMControlGenerator


@dataclass
class SRTrainConfig:
    # SD1.5：可以是 repo_id（需联网）或本地目录（推荐离线）
    pretrained_sd15: str = "runwayml/stable-diffusion-v1-5"

    output_dir: str = "./outputs/sr_controlnet"
    lr: float = 1e-5
    train_steps: int = 20000
    grad_accum: int = 1
    mixed_precision: str = "fp16"  # "no" / "fp16" / "bf16"
    save_every: int = 2000
    device: str = "cuda"
    
    # VFM control 相关
    local_dir: str = "./src/models/dinov2_vitb14"  # DINOv2 本地目录（离线）
    control_scale: float = 0.5  # ControlNet 控制强度比例（越大受控制影响越大）
    
    
    # validation（step-based）
    val_every: int = 0        # 0 表示不验证；>0 每 val_every step 验证一次
    val_batches: int = 10     # 每次验证最多跑多少个 batch
    sample_steps: int = 20    # validate/sample 时 DDPM 采样步数（越大越慢）


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
        # ✅ M1: init DINOv2 control generator
        # ----------------------------
        self.vfm_cfg = VFMConfig(
            variant="dinov2_vitb14",
            local_dir=cfg.local_dir,  # 你指定的 base 权重目录
            image_size=518,
            patch_size=14,
            control_mode="energy_edge_gray",
            normalize=True,
        )
        self.ctrl_gen = VFMControlGenerator(self.vfm_cfg, device=self.device)
        self._warned_control_fallback = False
        print(f"[VFM] enabled: {self.vfm_cfg.variant} | local_dir={self.vfm_cfg.local_dir}")

        # ----------------------------
        # SD1.5 components (repo_id or local dir)
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

        # ControlNet initialized from UNet weights
        self.controlnet = ControlNetModel.from_unet(self.unet).to(self.device)

        self.noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.pretrained_sd15, subfolder="scheduler", token=token
        )

        # Freeze base
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.unet.eval()
        self.text_encoder.eval()

        # Train only ControlNet
        self.controlnet.train()
        self.optimizer = AdamW(self.controlnet.parameters(), lr=float(cfg.lr))

        # AMP
        self.use_amp = (cfg.mixed_precision in ("fp16", "bf16"))
        self.autocast_dtype = torch.float16 if cfg.mixed_precision == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision == "fp16"))

        # cache empty prompt embeds
        self.empty_prompt_embeds = self._encode_text([""])

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
        """
        x_01: [B,3,512,512] in [0,1]
        returns latents: [B,4,64,64]
        """
        x = x_01 * 2.0 - 1.0
        latents = self.vae.encode(x).latent_dist.mode()
        return latents * 0.18215

    @torch.no_grad()
    def _vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latents: [B,4,64,64] (scaled)
        returns img: [B,3,512,512] in [0,1]
        """
        latents = latents / 0.18215
        img = self.vae.decode(latents).sample
        img = (img + 1.0) / 2.0
        return img.clamp(0, 1)

    def _build_control(self, lr_up_512: torch.Tensor) -> torch.Tensor:
        """
        lr_up_512: [B,3,512,512] in [0,1]
        return: [B,3,512,512] in [0,1]
        """
        # ✅ 优先用 VFM control；失败则兜底
        try:
            ctrl = self.ctrl_gen(lr_up_512)
            # 安全极值
            ctrl = ctrl.clamp(0, 1)
            ctrl = ctrl * 0.9 + 0.05
            
            return ctrl
        except Exception as e:
            if not self._warned_control_fallback:
                print(f"[WARN] VFM control failed once, fallback to lr-edge control. err={e}")
                self._warned_control_fallback = True
            return make_control_image_from_lr_fallback(lr_up_512)

    def save(self, step: int):
        out = os.path.join(self.cfg.output_dir, f"controlnet_step_{step}")
        os.makedirs(out, exist_ok=True)
        self.controlnet.save_pretrained(out)
        print(f"[save] controlnet saved to {out}")

    def train(self, train_loader, val_loader=None):
        cfg = self.cfg
        global_step = 0

        do_val = (val_loader is not None) and (getattr(cfg, "val_every", 0) and int(cfg.val_every) > 0)
        val_every = int(getattr(cfg, "val_every", 0))
        val_batches = int(getattr(cfg, "val_batches", 10))

        pbar = tqdm(total=cfg.train_steps, desc="train", unit="step")
        data_iter = iter(train_loader)

        while global_step < cfg.train_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Expect batch keys: lr, hr
            lr = batch["lr"].to(self.device, non_blocking=True)  # [B,3,128,128]
            hr = batch["hr"].to(self.device, non_blocking=True)  # [B,3,512,512]

            # build control internally (M1: DINOv2)
            lr_up = F.interpolate(lr, size=(512, 512), mode="bicubic", align_corners=False)
            control_image = self._build_control(lr_up)  # [B,3,512,512]

            with torch.no_grad():
                latents = self._vae_encode(hr)  # x0 latents

            bsz = latents.shape[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps,
                (bsz,), device=self.device, dtype=torch.long
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
                
                scale = float(getattr(self.cfg, "control_scale", 1.0))
                down_samples = [d * scale for d in down_samples]
                mid_sample = mid_sample * scale

                noise_pred = self.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                    return_dict=False,
                )[0]

                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") / cfg.grad_accum

            # backward
            if cfg.mixed_precision == "fp16":
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % cfg.grad_accum == 0:
                if cfg.mixed_precision == "fp16":
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": float(loss.detach().cpu())})

            # periodic validation
            if do_val and (global_step % val_every == 0):
                try:
                    metrics = self.validate(val_loader, step=global_step, max_batches=val_batches)
                    if isinstance(metrics, dict) and metrics:
                        show = {f"val_{k}": float(v) for k, v in metrics.items()}
                        pbar.set_postfix({**pbar.postfix, **show})
                except Exception as e:
                    print(f"[WARN] validate failed at step={global_step}: {e}")

            # periodic checkpoint
            if (global_step % cfg.save_every) == 0:
                self.save(global_step)

        pbar.close()
        self.save(global_step)

    @torch.no_grad()
    def sample_sr(self, lr: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """
        lr: [B,3,128,128] in [0,1]
        return sr: [B,3,512,512] in [0,1]
        """
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
            
            scale = float(getattr(self.cfg, "control_scale", 1.0))
            down = [d * scale for d in down]
            mid = mid * scale

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

    
    @torch.no_grad()
    def validate(self, val_loader, step: int = 0, max_batches: int = 10):

        self.controlnet.eval()

        psnr_list, ssim_list, lpips_list, stlpips_list = [], [], [], []

        # ---- choose a random batch index for visualization ----
        save_vis = True
        vis_batch_idx = np.random.randint(0, max(1, int(max_batches)))
        vis_saved = False

        # ---- prepare output folder ----
        vis_root = os.path.join(self.cfg.output_dir, "val_vis", f"step_{step:07d}")
        os.makedirs(vis_root, exist_ok=True)

        # helper: tensor [3,H,W] -> uint8 HWC
        def to_u8(img_3chw: torch.Tensor) -> np.ndarray:
            x = img_3chw.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            x = (x * 255.0 + 0.5).astype(np.uint8)
            return x

        # sample steps
        num_steps = int(getattr(self.cfg, "sample_steps", 20))

        for i, batch in enumerate(val_loader):
            if i >= int(max_batches):
                break

            lr = batch["lr"].to(self.device, non_blocking=True)  # [B,3,128,128]
            hr = batch["hr"].to(self.device, non_blocking=True)  # [B,3,512,512]

            # ---- build lr_up + control (must be 512!) ----
            lr_up = torch.nn.functional.interpolate(
                lr, size=(512, 512), mode="bicubic", align_corners=False
            )
            control = self._build_control(lr_up)  # ✅ [B,3,512,512]

            # ---- diffusion sampling (inline, DO NOT call sample_sr to avoid state flips) ----
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

            sr = self._vae_decode(latents)  # [B,3,512,512] in [0,1]

            # ---- metrics ----
            psnr_list.append(float(self.metrics.psnr(sr, hr)))
            ssim_list.append(float(self.metrics.ssim(sr, hr)))
            lpips_list.append(float(self.metrics.lpips(sr, hr)))
            stlpips_list.append(float(self.metrics.shift_tolerant_lpips(sr, hr)))

            # ---- save one random example ----
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

        return results
