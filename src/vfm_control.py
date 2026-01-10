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
    默认：推理/验证时冻结且 no_grad
    训练时：可选择只解冻 DINOv2 最后若干个 block 参与反传
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

        # 默认：冻结
        self.freeze_all()

    def freeze_all(self):
        """冻结全部 DINOv2 参数，并置为 eval（用于 val / sample）。"""
        self.vfm.eval()
        for p in self.vfm.parameters():
            p.requires_grad_(False)

    def set_trainable(self, unfreeze_last_n_blocks: int = 2, train_ln: bool = True):
        """
        仅解冻最后 N 个 Transformer blocks（以及可选的 LayerNorm），用于参数高效微调。
        - unfreeze_last_n_blocks: 0 表示完全冻结（等价 freeze_all）
        """
        self.freeze_all()
        if unfreeze_last_n_blocks <= 0:
            return

        # 训练模式（注意：只要有 requires_grad=True 的参数，外部就能反传）
        self.vfm.train()

        # 兼容不同 transformers 版本/实现的层路径
        blocks = None
        if hasattr(self.vfm, "encoder") and hasattr(self.vfm.encoder, "layer"):
            blocks = self.vfm.encoder.layer
        elif hasattr(self.vfm, "backbone") and hasattr(self.vfm.backbone, "encoder") and hasattr(self.vfm.backbone.encoder, "layer"):
            blocks = self.vfm.backbone.encoder.layer
        elif hasattr(self.vfm, "encoder") and hasattr(self.vfm.encoder, "layers"):
            blocks = self.vfm.encoder.layers

        if blocks is not None:
            n_total = len(blocks)
            n_unfreeze = min(int(unfreeze_last_n_blocks), n_total)
            for blk in blocks[n_total - n_unfreeze:]:
                for p in blk.parameters():
                    p.requires_grad_(True)

            if train_ln:
                # 常见：同时解冻最终 norm / layernorm，提升稳定性
                for name, p in self.vfm.named_parameters():
                    lname = name.lower()
                    if ("layernorm" in lname) or (lname.endswith(".norm.weight")) or (lname.endswith(".norm.bias")):
                        p.requires_grad_(True)
        else:
            # 兜底：按名字解冻最后若干层（不完美，但可用）
            # 例如 encoder.layer.10 / encoder.layer.11
            # 这里不会报错，只是可能解冻范围略保守
            # 先收集所有 block index
            import re
            pat = re.compile(r"(encoder\.layer|encoder\.layers)\.(\d+)\.")
            idxs = []
            for n, _ in self.vfm.named_parameters():
                m = pat.search(n)
                if m:
                    idxs.append(int(m.group(2)))
            if idxs:
                max_idx = max(idxs)
                start_idx = max(0, max_idx - int(unfreeze_last_n_blocks) + 1)
                for n, p in self.vfm.named_parameters():
                    m = pat.search(n)
                    if m and int(m.group(2)) >= start_idx:
                        p.requires_grad_(True)
                if train_ln:
                    for n, p in self.vfm.named_parameters():
                        lname = n.lower()
                        if ("layernorm" in lname) or (lname.endswith(".norm.weight")) or (lname.endswith(".norm.bias")):
                            p.requires_grad_(True)

    def _extract_feature_map_impl(self, lr_up: torch.Tensor) -> torch.Tensor:
        """
        lr_up: [B,3,512,512] in [0,1]
        return feat_map: [B,C,h,w]
        """
        B, C, H, W = lr_up.shape

        x = lr_up
        x = F.interpolate(x, size=(self.cfg.image_size, self.cfg.image_size), mode="bicubic", align_corners=False)

        # ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        out = self.vfm(pixel_values=x, output_hidden_states=False, return_dict=True)
        tokens = out.last_hidden_state  # [B, 1+N, C]
        tokens = tokens[:, 1:, :]       # drop CLS -> [B, N, C]

        gh = self.cfg.image_size // self.cfg.patch_size
        gw = self.cfg.image_size // self.cfg.patch_size
        N = tokens.shape[1]
        if gh * gw != N:
            g = int((N) ** 0.5)
            gh, gw = g, g
            tokens = tokens[:, : gh * gw, :]

        feat = tokens.transpose(1, 2).contiguous().view(B, -1, gh, gw)  # [B,C,gh,gw]
        return feat

    def _feat_to_energy_impl(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        feat_map: [B,C,h,w]
        return energy: [B,1,h,w] in [0,1]
        """
        energy = torch.sqrt(torch.clamp((feat_map ** 2).sum(dim=1, keepdim=True), min=1e-12))

        if self.cfg.normalize:
            # per-image min-max（min/max 对梯度是次梯度；可用）
            B = energy.shape[0]
            e = energy.view(B, -1)
            e_min = e.min(dim=1, keepdim=True).values.view(B, 1, 1, 1)
            e_max = e.max(dim=1, keepdim=True).values.view(B, 1, 1, 1)
            energy = (energy - e_min) / torch.clamp(e_max - e_min, min=1e-6)
            energy = energy.clamp(0, 1)

        return energy

    def _compose_control_impl(self, lr_up: torch.Tensor, energy_1ch: torch.Tensor) -> torch.Tensor:
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
            ctrl = torch.cat([gray, edge, gray], dim=1)

        return ctrl.clamp(0, 1)

    def forward_train(self, lr_up: torch.Tensor) -> torch.Tensor:
        """
        训练用：允许梯度流入 DINOv2（取决于 set_trainable 解冻了哪些层）
        """
        feat = self._extract_feature_map_impl(lr_up)
        energy = self._feat_to_energy_impl(feat)
        ctrl = self._compose_control_impl(lr_up, energy)
        return ctrl

    @torch.no_grad()
    def forward(self, lr_up: torch.Tensor) -> torch.Tensor:
        """
        推理/验证用：强制 no_grad
        """
        feat = self._extract_feature_map_impl(lr_up)
        energy = self._feat_to_energy_impl(feat)
        ctrl = self._compose_control_impl(lr_up, energy)
        return ctrl

