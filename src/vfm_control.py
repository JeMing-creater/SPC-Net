# src/vfm_control.py
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from transformers import Dinov2Model, Dinov2Config
except Exception:
    Dinov2Model = None
    Dinov2Config = None


@dataclass
class VFMConfig:
    variant: str = "dinov2_vitb14"   # 仅用于记录
    local_dir: str = "./src/models/dinov2_vitb14"   # 本地目录（离线）
    image_size: int = 518           # dinov2常见输入
    patch_size: int = 14            # vitb14=14
    control_mode: str = "energy_edge_gray"
    normalize: bool = True


def _make_edge_from_gray(gray_1ch: torch.Tensor) -> torch.Tensor:
    # gray_1ch: [B,1,H,W] in [0,1]
    dx = torch.abs(gray_1ch[:, :, :, 1:] - gray_1ch[:, :, :, :-1])
    dy = torch.abs(gray_1ch[:, :, 1:, :] - gray_1ch[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    edge = torch.clamp(dx + dy, 0.0, 1.0)
    return edge


class VFMControlGenerator(torch.nn.Module):
    """
    输入 lr_up [B,3,512,512] -> 输出 control [B,3,512,512]
    使用冻结 DINOv2 特征生成 energy map。
    """
    def __init__(self, cfg: VFMConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        if Dinov2Model is None:
            raise RuntimeError(
                "transformers.Dinov2Model not available. "
                "Please upgrade transformers or install a version that includes Dinov2Model."
            )

        if not cfg.local_dir or (not os.path.isdir(cfg.local_dir)):
            raise FileNotFoundError(f"DINOv2 local_dir not found: {cfg.local_dir}")

        # ✅ 离线加载
        self.vfm = Dinov2Model.from_pretrained(
            cfg.local_dir,
            local_files_only=True,
        ).to(device)
        self.vfm.eval()
        for p in self.vfm.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _extract_feature_map(self, lr_up: torch.Tensor) -> torch.Tensor:
        """
        lr_up: [B,3,512,512] in [0,1]
        return feat_map: [B,C,h,w]
        """
        B, C, H, W = lr_up.shape

        # DINOv2 期望 ImageNet 风格归一化（经验上对特征稳定有帮助）
        x = lr_up
        x = F.interpolate(x, size=(self.cfg.image_size, self.cfg.image_size), mode="bicubic", align_corners=False)

        # ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        out = self.vfm(pixel_values=x, output_hidden_states=False, return_dict=True)
        # last_hidden_state: [B, 1+N, C] (含 CLS token)
        tokens = out.last_hidden_state

        # 去掉 CLS
        tokens = tokens[:, 1:, :]  # [B, N, C]

        # 还原网格
        gh = self.cfg.image_size // self.cfg.patch_size
        gw = self.cfg.image_size // self.cfg.patch_size
        # tokens N 应该约等于 gh*gw（若有轻微不一致，做个安全处理）
        N = tokens.shape[1]
        if gh * gw != N:
            # 兜底：按 sqrt(N) 近似
            g = int((N) ** 0.5)
            gh, gw = g, g
            tokens = tokens[:, : gh * gw, :]

        feat = tokens.transpose(1, 2).contiguous().view(B, -1, gh, gw)  # [B,C,gh,gw]
        return feat

    @torch.no_grad()
    def _feat_to_energy(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        feat_map: [B,C,h,w]
        return energy: [B,1,h,w] in [0,1]
        """
        # L2 energy over channel
        energy = torch.sqrt(torch.clamp((feat_map ** 2).sum(dim=1, keepdim=True), min=1e-12))

        if self.cfg.normalize:
            # per-image min-max
            B = energy.shape[0]
            e = energy.view(B, -1)
            e_min = e.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
            e_max = e.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
            energy = (energy - e_min) / torch.clamp(e_max - e_min, min=1e-6)
            energy = energy.clamp(0, 1)

        return energy

    @torch.no_grad()
    def _compose_control(self, lr_up: torch.Tensor, energy_1ch: torch.Tensor) -> torch.Tensor:
        """
        lr_up: [B,3,512,512] in [0,1]
        energy_1ch: [B,1,h,w] in [0,1]
        return: control [B,3,512,512] in [0,1]
        """
        gray = lr_up.mean(dim=1, keepdim=True)
        edge = _make_edge_from_gray(gray)

        energy = F.interpolate(energy_1ch, size=(lr_up.shape[2], lr_up.shape[3]), mode="bilinear", align_corners=False)
        energy = energy.clamp(0, 1)

        mode = self.cfg.control_mode.lower()
        if mode == "energy_edge_gray":
            ctrl = torch.cat([energy, edge, gray], dim=1)
        elif mode == "energy_only":
            ctrl = energy.repeat(1, 3, 1, 1)
        else:
            # fallback
            ctrl = torch.cat([gray, edge, gray], dim=1)

        return ctrl.clamp(0, 1)

    @torch.no_grad()
    def forward(self, lr_up: torch.Tensor) -> torch.Tensor:
        feat = self._extract_feature_map(lr_up)
        energy = self._feat_to_energy(feat)
        ctrl = self._compose_control(lr_up, energy)
        return ctrl
