# src/sr_metrics.py
import torch
import torch.nn.functional as F
import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips


class SRMetrics:
    def __init__(self, device="cuda"):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def psnr(self, sr, hr):
        # sr, hr: [B,3,H,W] in [0,1]
        sr = sr.clamp(0, 1).cpu().numpy()
        hr = hr.clamp(0, 1).cpu().numpy()

        vals = []
        for i in range(sr.shape[0]):
            vals.append(
                peak_signal_noise_ratio(
                    hr[i].transpose(1,2,0),
                    sr[i].transpose(1,2,0),
                    data_range=1.0
                )
            )
        return float(np.mean(vals))

    @torch.no_grad()
    def ssim(self, sr, hr):
        sr = sr.clamp(0, 1).cpu().numpy()
        hr = hr.clamp(0, 1).cpu().numpy()

        vals = []
        for i in range(sr.shape[0]):
            vals.append(
                structural_similarity(
                    hr[i].transpose(1,2,0),
                    sr[i].transpose(1,2,0),
                    channel_axis=2,
                    data_range=1.0
                )
            )
        return float(np.mean(vals))

    @torch.no_grad()
    def lpips(self, sr, hr):
        # LPIPS expects [-1,1]
        sr = sr * 2 - 1
        hr = hr * 2 - 1
        d = self.lpips_fn(sr, hr)
        return float(d.mean().item())

    @torch.no_grad()
    def shift_tolerant_lpips(self, sr, hr, max_shift=4):
        """
        Shift-Tolerant LPIPS:
        min LPIPS over small spatial shifts (±max_shift pixels)
        """
        sr = sr * 2 - 1
        hr = hr * 2 - 1

        B, C, H, W = sr.shape
        best = torch.full((B,), float("inf"), device=sr.device)

        for dx in range(-max_shift, max_shift + 1):
            for dy in range(-max_shift, max_shift + 1):
                shifted = torch.roll(sr, shifts=(dx, dy), dims=(2, 3))
                d = self.lpips_fn(shifted, hr).view(B)
                best = torch.minimum(best, d)

        return float(best.mean().item())
