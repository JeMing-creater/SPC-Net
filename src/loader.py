import os
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, PngImagePlugin

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# --- Pillow robustness (your PNG issue) ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024


# Optional fallback reader
import cv2




def slide_to_case_id(slide_id: str) -> str:
    parts = slide_id.split("-")
    return "-".join(parts[:3])


def _safe_float(x) -> Optional[float]:
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


@dataclass
class SurvivalInfo:
    time: float
    event: int  # 1 if days_to_death used, else 0


class ClinicalSurvivalIndex:
    def __init__(
        self,
        clin_path: str,
        id_col: str = "case_submitter_id",
        death_col: str = "days_to_death",
        follow_col: str = "days_to_last_follow_up",
    ):
        if not os.path.exists(clin_path):
            raise FileNotFoundError(f"clinical.tsv not found: {clin_path}")

        self.id_col = id_col
        self.death_col = death_col
        self.follow_col = follow_col

        df = pd.read_csv(clin_path, sep="\t")
        if id_col not in df.columns:
            raise ValueError(f"clinical.tsv missing column: {id_col}")
        if death_col not in df.columns and follow_col not in df.columns:
            raise ValueError(f"clinical.tsv missing both: {death_col}, {follow_col}")

        df[id_col] = df[id_col].astype(str)
        self.df = df.set_index(id_col, drop=False)

    def get(self, case_id: str) -> Optional[SurvivalInfo]:
        if case_id not in self.df.index:
            return None

        row = self.df.loc[case_id]

        if self.death_col in self.df.columns:
            t = _safe_float(row.get(self.death_col, None))
            if t is not None:
                return SurvivalInfo(time=t, event=1)

        if self.follow_col in self.df.columns:
            t = _safe_float(row.get(self.follow_col, None))
            if t is not None:
                return SurvivalInfo(time=t, event=0)

        return None


def _list_pngs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.png")))


def _has_done_flag(hr_slide_dir: str) -> bool:
    return os.path.exists(os.path.join(hr_slide_dir, ".DONE"))


class PathologySRSurvivalDataset(Dataset):
    """
    Patch-level dataset for SR + survival:
      - Reads paired (lr, hr) PNG patches + cached tissue mask PNG
      - Filters cases without valid survival time
      - Filters slides without .DONE if require_done=True
      - Ensures lr/hr/mask are aligned by patch filename
      - patch_num: per-slide cap (take first N patches sorted by name)

    Return dict:
      {
        "lr": Tensor[3,128,128],
        "hr": Tensor[3,512,512],
        "time": Tensor[],
        "event": Tensor[],
        "meta": {...}
      }
    """

    def __init__(
        self,
        out_img_dir: str,
        id_col: str = "case_submitter_id",
        death_col: str = "days_to_death",
        follow_col: str = "days_to_last_follow_up",
        require_done: bool = True,
        patch_num: int = 200,
        transform_lr=None,
        transform_hr=None,
    ):
        self.out_img_dir = out_img_dir
        self.hr_root = os.path.join(out_img_dir, "hr_png")
        self.lr_root = os.path.join(out_img_dir, "lr_png")
        self.clin_path = os.path.join(out_img_dir, "clinical.tsv")

        if not os.path.isdir(self.hr_root):
            raise FileNotFoundError(f"hr_png not found: {self.hr_root}")
        if not os.path.isdir(self.lr_root):
            raise FileNotFoundError(f"lr_png not found: {self.lr_root}")
        if not os.path.exists(self.clin_path):
            raise FileNotFoundError(f"clinical.tsv not found: {self.clin_path}")

        self.surv_index = ClinicalSurvivalIndex(
            clin_path=self.clin_path,
            id_col=id_col,
            death_col=death_col,
            follow_col=follow_col,
        )

        self.require_done = require_done
        self.patch_num = int(patch_num) if patch_num is not None else None

        self.transform_lr = transform_lr or transforms.ToTensor()
        self.transform_hr = transform_hr or transforms.ToTensor()

        self.items: List[Dict[str, Any]] = []
        self._build_index()

    # -------- robust readers --------
    @staticmethod
    def _read_rgb_pil_or_cv2(path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read RGB image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)


    # -------- index builder --------
    def _build_index(self):
        slide_dirs = sorted(
            d for d in os.listdir(self.hr_root)
            if os.path.isdir(os.path.join(self.hr_root, d))
        )

        skipped_no_done = 0
        skipped_no_surv = 0
        skipped_no_lr_slide = 0
        skipped_missing_lr = 0
        kept = 0

        for slide_id in slide_dirs:
            hr_slide_dir = os.path.join(self.hr_root, slide_id)
            lr_slide_dir = os.path.join(self.lr_root, slide_id)

            # slide-level filters
            if self.require_done and (not _has_done_flag(hr_slide_dir)):
                skipped_no_done += 1
                continue

            if not os.path.isdir(lr_slide_dir):
                skipped_no_lr_slide += 1
                continue

            

            case_id = slide_to_case_id(slide_id)
            surv = self.surv_index.get(case_id)
            if surv is None:
                skipped_no_surv += 1
                continue

            # patch list from HR
            hr_pngs = sorted([f for f in os.listdir(hr_slide_dir) if f.endswith(".png")])
            if not hr_pngs:
                continue

            if self.patch_num is not None and self.patch_num > 0:
                hr_pngs = hr_pngs[: self.patch_num]

            for patch_name in hr_pngs:
                hr_path = os.path.join(hr_slide_dir, patch_name)
                lr_path = os.path.join(lr_slide_dir, patch_name)

                if not os.path.exists(lr_path):
                    skipped_missing_lr += 1
                    continue

                self.items.append(
                    {
                        "case_id": case_id,
                        "slide_id": slide_id,
                        "hr_path": hr_path,
                        "lr_path": lr_path,
                        "time": surv.time,
                        "event": surv.event,
                    }
                )
                kept += 1

        print(
            f"[loader] index built: {kept} paired patches | "
            f"skipped(no .DONE)={skipped_no_done} | "
            f"skipped(no survival)={skipped_no_surv} | "
            f"skipped(no lr slide)={skipped_no_lr_slide} | "
            f"skipped(missing lr)={skipped_missing_lr} | "
        )

    # -------- dataset API --------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]

        lr_img = self._read_rgb_pil_or_cv2(it["lr_path"])     # 128x128
        hr_img = self._read_rgb_pil_or_cv2(it["hr_path"])     # 512x512
        

        lr = self.transform_lr(lr_img)  # [3,128,128] in [0,1]
        hr = self.transform_hr(hr_img)  # [3,512,512] in [0,1]

        

        return {
            "lr": lr,
            "hr": hr,
            "time": torch.tensor(it["time"], dtype=torch.float32),
            "event": torch.tensor(it["event"], dtype=torch.long),
            "meta": {
                "case_id": it["case_id"],
                "slide_id": it["slide_id"],
                "lr_path": it["lr_path"],
                "hr_path": it["hr_path"],
            },
        }
        
def _split_cases(
    cases: List[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError("ratios must be non-negative")
    s = float(ratios.sum())
    if s <= 0:
        raise ValueError("sum of ratios must be > 0")
    ratios = ratios / s

    rng = np.random.default_rng(seed)
    cases = list(cases)
    rng.shuffle(cases)

    n = len(cases)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    # ensure total = n
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]

    return train_cases, val_cases, test_cases


def build_case_split_dataloaders(
    out_img_dir: str,
    batch_size: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    split_seed: int = 2025,
    patch_num: int = 200,
    num_workers: int = 8,
    pin_memory: bool = True,
    drop_last: bool = False,
    require_done: bool = True,
    id_col: str = "case_submitter_id",
    death_col: str = "days_to_death",
    follow_col: str = "days_to_last_follow_up",
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns: train_loader, val_loader, test_loader
    Split is by case_id to avoid leakage.
    """
    ds = PathologySRSurvivalDataset(
        out_img_dir=out_img_dir,
        id_col=id_col,
        death_col=death_col,
        follow_col=follow_col,
        require_done=require_done,
        patch_num=patch_num,
    )

    # collect cases that actually appear in dataset items
    cases = sorted({it["case_id"] for it in ds.items})
    if len(cases) == 0:
        raise RuntimeError("No valid cases found after filtering clinical + png pairs.")

    train_cases, val_cases, test_cases = _split_cases(
        cases=cases,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )

    train_set = set(train_cases)
    val_set = set(val_cases)
    test_set = set(test_cases)

    train_idx = [i for i, it in enumerate(ds.items) if it["case_id"] in train_set]
    val_idx   = [i for i, it in enumerate(ds.items) if it["case_id"] in val_set]
    test_idx  = [i for i, it in enumerate(ds.items) if it["case_id"] in test_set]

    print(
        f"[split] cases: total={len(cases)} | train={len(train_cases)} | val={len(val_cases)} | test={len(test_cases)}"
    )
    print(
        f"[split] patches: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}"
    )

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    test_ds = Subset(ds, test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import yaml
    from easydict import EasyDict
    
    config = EasyDict(
        yaml.load(open("/workspace/SuperR/config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    
    ds = PathologySRSurvivalDataset(out_img_dir="/mnt/liangjm/SpR_data/", patch_num=100)
    x = ds[0]
    print(x["lr"].shape, x["hr"].shape, x["time"], x["event"])
    
    # train_loader, val_loader, test_loader = build_case_split_dataloaders(
    #     out_img_dir=config.data_loader.out_img_dir,
    #     batch_size=config.trainer.batch_size,
    #     patch_num=getattr(config.data_loader, "patch_num", 200),
    #     train_ratio=config.data_loader.train_ratio,
    #     val_ratio=config.data_loader.val_ratio,
    #     test_ratio=config.data_loader.test_ratio,
    #     split_seed=getattr(config.data_loader, "split_seed", 2025),
    #     num_workers=config.data_loader.num_workers,
    #     pin_memory=config.data_loader.pin_memory,
    # )

    # train_num = 0
    # val_num = 0
    # teat_num = 0
    
    # for batch in train_loader:
    #     lr = batch["lr"]        # [B, 3, h, w]
    #     hr = batch["hr"]        # [B, 3, H, W]
    #     time = batch["time"]    # [B]
    #     event = batch["event"]  # [B]
    #     meta = batch["meta"]
    #     print(lr.shape, hr.shape, time[:3], event[:3])
    #     train_num += 1
    #     # break
    
    # for batch in val_loader:
    #     lr = batch["lr"]        # [B, 3, h, w]
    #     hr = batch["hr"]        # [B, 3, H, W]
    #     time = batch["time"]    # [B]
    #     event = batch["event"]  # [B]
    #     meta = batch["meta"]
    #     print(lr.shape, hr.shape, time[:3], event[:3])
    #     val_num += 1
    
    # for batch in test_loader:
    #     lr = batch["lr"]        # [B, 3, h, w]
    #     hr = batch["hr"]        # [B, 3, H, W]
    #     time = batch["time"]    # [B]
    #     event = batch["event"]  # [B]
    #     meta = batch["meta"]
    #     print(lr.shape, hr.shape, time[:3], event[:3])
    #     teat_num += 1
    
    # print(f"train batches: {train_num}, val batches: {val_num}, test batches: {teat_num}")
    