from calendar import c
from math import e
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
from datetime import datetime
from typing import Dict

import monai
import torch
import yaml
from tqdm import tqdm
import torch.nn as nn
from dataclasses import dataclass, field
from accelerate import Accelerator
from easydict import EasyDict
from monai.utils import ensure_tuple_rep
from objprint import objstr
from timm.optim import optim_factory


import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# WSI 切片流程：40x HR png + 10x LR png
# from CLAM import *
CLAM_ROOT = "CLAM"  # ← 换成你 git clone CLAM 的地方
sys.path.append(CLAM_ROOT)
import src.preprocess_tcga_luad_with_clam as slic_tool


# 读取 clinical.tsv
from src.tcga_clinical import TCGAClinicalTable, slide_id_from_path, slide_to_case_id
from src.loader import build_case_split_dataloaders
from src.utils import  Logger, same_seeds
from src.sr_diffusion_trainer import SRTrainConfig, DiffusionSRControlNetTrainer
from src.gen_sam_disc_maps import generate_sam_disc_maps


if __name__ == "__main__":
    # 加载配置文件
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    
    same_seeds(config.data_loader.seed)
    
    logging_dir = (
        os.getcwd()
        + "/logs/"
        + config.checkpoint
        + str(datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .replace(".", "_")
    )
    
    accelerator = Accelerator(
        cpu=False, log_with=["tensorboard"], project_dir=logging_dir
    )
    
    Logger(logging_dir if accelerator.is_local_main_process else None)
    
    image_dict = {
        "TCGA-LUAD": [config.TCGA_LUAD.root, config.TCGA_LUAD.choose_WSI],
        "TCGA-KIRC": [config.TCGA_KIRC.root, config.TCGA_KIRC.choose_WSI],
        "TCGA-LIHC": [config.TCGA_LIHC.root, config.TCGA_LIHC.choose_WSI],
    }
    
    # 检查切片全面性
    slic_tool.run_clam_and_export(
        data_cfg = image_dict,
        out_img_dir=config.data_loader.out_img_dir,
        patch_size=config.data_loader.patch_size,
        step_size=config.data_loader.step_size,
        patch_level=config.data_loader.patch_level,
        down_scale=config.data_loader.down_scale,
        min_tissue_ratio = config.data_loader.min_tissue_ratio,
        seed=config.data_loader.seed,
        limit_samples = config.data_loader.patch_num
    )
    
    disc_cfg = getattr(config.sr, "disc", None)
    if disc_cfg and bool(getattr(disc_cfg, "generate_before_train", False)):
        generate_sam_disc_maps(
            out_img_dir=config.data_loader.out_img_dir,
            sam_ckpt=str(getattr(disc_cfg, "sam_ckpt")),
            sam_model_type=str(getattr(disc_cfg, "sam_model_type", "vit_b")),
            device=str(getattr(disc_cfg, "device", getattr(config.sr, "device", "cuda"))),
            patch_num=int(getattr(config.data_loader, "patch_num", 200)),
            require_done=bool(getattr(disc_cfg, "require_done", True)),
            disc_root=str(getattr(disc_cfg, "disc_root", "")) or None,
            max_items=int(getattr(disc_cfg, "max_items", -1)),
            overwrite=bool(getattr(disc_cfg, "overwrite", False)),
            log_every=int(getattr(disc_cfg, "log_every", 50)),
        )
    
    
    # 构建dataloaders
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
    
    # 4) 训练 ControlNet
    tcfg = SRTrainConfig(
        pretrained_sd15=getattr(config.sr, "pretrained_sd15", "runwayml/stable-diffusion-v1-5"),
        output_dir=getattr(config.sr, "output_dir", "./outputs/sr_controlnet"),
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
    )
    
    trainer = DiffusionSRControlNetTrainer(tcfg, token=getattr(config.sr, "token", None))
    trainer.train(train_loader, val_loader=val_loader, resume_ckpt=(config.sr.resume_ckpt if config.sr.resume_ckpt != "" else None))
