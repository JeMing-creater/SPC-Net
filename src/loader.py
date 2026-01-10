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

        # Prefer vital_status when available
        vital = None
        if "vital_status" in self.df.columns:
            vital = str(row.get("vital_status", "")).strip().lower()

        if vital in ("dead", "deceased"):
            t = _safe_float(row.get(self.death_col, None))
            if t is not None:
                return SurvivalInfo(time=t, event=1)
            return None

        if vital in ("alive", "living"):
            t = _safe_float(row.get(self.follow_col, None))
            if t is not None:
                return SurvivalInfo(time=t, event=0)
            return None

        # Fallback: old behavior
        t = _safe_float(row.get(self.death_col, None)) if self.death_col in self.df.columns else None
        if t is not None:
            return SurvivalInfo(time=t, event=1)

        t = _safe_float(row.get(self.follow_col, None)) if self.follow_col in self.df.columns else None
        if t is not None:
            return SurvivalInfo(time=t, event=0)

        return None



def build_survival_map(
    df: pd.DataFrame,
    id_col: str = "case_submitter_id",
    death_col: str = "days_to_death",
    follow_col: str = "days_to_last_follow_up",
) -> Dict[str, SurvivalInfo]:
    """
    Build {case_id -> SurvivalInfo(time, event)} from clinical.tsv DataFrame.

    - If vital_status exists:
        dead/deceased -> use days_to_death, event=1
        alive/living  -> use days_to_last_follow_up, event=0
    - Else fallback:
        if days_to_death valid -> event=1
        elif days_to_last_follow_up valid -> event=0
    """
    if id_col not in df.columns:
        raise ValueError(f"clinical.tsv missing column: {id_col}")
    if (death_col not in df.columns) and (follow_col not in df.columns):
        raise ValueError(f"clinical.tsv missing both: {death_col}, {follow_col}")

    out: Dict[str, SurvivalInfo] = {}

    has_vital = "vital_status" in df.columns
    for _, row in df.iterrows():
        case_id = str(row[id_col])

        vital = None
        if has_vital:
            vital = str(row.get("vital_status", "")).strip().lower()

        # preferred branch if vital_status available
        if vital in ("dead", "deceased"):
            t = _safe_float(row.get(death_col, None))
            if t is not None:
                out[case_id] = SurvivalInfo(time=t, event=1)
            continue

        if vital in ("alive", "living"):
            t = _safe_float(row.get(follow_col, None))
            if t is not None:
                out[case_id] = SurvivalInfo(time=t, event=0)
            continue

        # fallback
        t = _safe_float(row.get(death_col, None)) if death_col in df.columns else None
        if t is not None:
            out[case_id] = SurvivalInfo(time=t, event=1)
            continue

        t = _safe_float(row.get(follow_col, None)) if follow_col in df.columns else None
        if t is not None:
            out[case_id] = SurvivalInfo(time=t, event=0)
            continue

    return out



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
        disc_root: Optional[str] = None,  # discrepancy map root
    ):
        self.out_img_dir = out_img_dir
        self.hr_root = os.path.join(out_img_dir, "hr_png")
        self.lr_root = os.path.join(out_img_dir, "lr_png")
        self.clin_path = os.path.join(out_img_dir, "clinical.tsv")

        self.disc_root = disc_root or os.path.join(out_img_dir, "disc_maps")

        if not os.path.isdir(self.hr_root):
            raise FileNotFoundError(f"hr_png not found: {self.hr_root}")
        if not os.path.isdir(self.lr_root):
            raise FileNotFoundError(f"lr_png not found: {self.lr_root}")
        if not os.path.exists(self.clin_path):
            raise FileNotFoundError(f"clinical.tsv not found: {self.clin_path}")

        self.require_done = bool(require_done)
        self.patch_num = int(patch_num)

        if transform_lr is None:
            self.transform_lr = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform_lr = transform_lr

        if transform_hr is None:
            self.transform_hr = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform_hr = transform_hr

        # --- load survival table ---
        df = pd.read_csv(self.clin_path, sep="\t")
        surv_map = build_survival_map(df, id_col=id_col, death_col=death_col, follow_col=follow_col)

        # --- detect folder layout ---
        # Layout A: hr_png/<case>/<slide>/*.png
        # Layout B: hr_png/<slide>/*.png
        first_level = sorted([d for d in os.listdir(self.hr_root) if os.path.isdir(os.path.join(self.hr_root, d))])

        def _dir_has_pngs(d: str) -> bool:
            try:
                for f in os.listdir(d):
                    if f.lower().endswith(".png"):
                        return True
            except Exception:
                return False
            return False

        layout_is_slide_level = False
        # heuristic: if any first-level dir directly contains png => slide-level
        for name in first_level[: min(20, len(first_level))]:
            if _dir_has_pngs(os.path.join(self.hr_root, name)):
                layout_is_slide_level = True
                break

        self.items = []
        skipped_no_surv = 0
        skipped_no_hr_slide = 0
        skipped_no_lr_slide = 0
        skipped_missing_lr = 0
        skipped_no_done = 0

        if layout_is_slide_level:
            # ---------- Layout B: hr_png/<slide_id>/*.png ----------
            hr_slides = first_level
            if len(hr_slides) == 0:
                skipped_no_hr_slide = 1  # just a marker
            for slide_id in hr_slides:
                hr_slide_dir = os.path.join(self.hr_root, slide_id)
                lr_slide_dir = os.path.join(self.lr_root, slide_id)

                if not os.path.isdir(lr_slide_dir):
                    skipped_no_lr_slide += 1
                    continue

                if self.require_done and (not _has_done_flag(hr_slide_dir)):
                    skipped_no_done += 1
                    continue

                case_id = slide_to_case_id(slide_id)
                surv = surv_map.get(case_id, None)
                if surv is None or surv.time is None:
                    skipped_no_surv += 1
                    continue

                hr_pngs = sorted([p for p in os.listdir(hr_slide_dir) if p.lower().endswith(".png")])
                if not hr_pngs:
                    skipped_no_hr_slide += 1
                    continue
                if self.patch_num > 0:
                    hr_pngs = hr_pngs[: self.patch_num]

                for patch_name in hr_pngs:
                    hr_path = os.path.join(hr_slide_dir, patch_name)
                    lr_path = os.path.join(lr_slide_dir, patch_name)
                    if not os.path.exists(lr_path):
                        skipped_missing_lr += 1
                        continue

                    disc_path = os.path.join(self.disc_root, slide_id, patch_name.replace(".png", ".pt"))

                    self.items.append(
                        {
                            "case_id": case_id,
                            "slide_id": slide_id,
                            "hr_path": hr_path,
                            "lr_path": lr_path,
                            "disc_path": disc_path,
                            "time": surv.time,
                            "event": surv.event,
                        }
                    )
        else:
            # ---------- Layout A: hr_png/<case_id>/<slide_id>/*.png ----------
            hr_cases = first_level
            for case_id in hr_cases:
                case_dir = os.path.join(self.hr_root, case_id)
                if not os.path.isdir(case_dir):
                    continue

                surv = surv_map.get(case_id, None)
                if surv is None or surv.time is None:
                    skipped_no_surv += 1
                    continue

                slides = sorted([d for d in os.listdir(case_dir) if os.path.isdir(os.path.join(case_dir, d))])
                if len(slides) == 0:
                    skipped_no_hr_slide += 1
                    continue

                for slide_id in slides:
                    hr_slide_dir = os.path.join(self.hr_root, case_id, slide_id)
                    lr_slide_dir = os.path.join(self.lr_root, case_id, slide_id)

                    if not os.path.isdir(lr_slide_dir):
                        skipped_no_lr_slide += 1
                        continue

                    if self.require_done and (not _has_done_flag(hr_slide_dir)):
                        skipped_no_done += 1
                        continue

                    hr_pngs = sorted([p for p in os.listdir(hr_slide_dir) if p.lower().endswith(".png")])
                    if self.patch_num > 0:
                        hr_pngs = hr_pngs[: self.patch_num]

                    for patch_name in hr_pngs:
                        hr_path = os.path.join(hr_slide_dir, patch_name)
                        lr_path = os.path.join(lr_slide_dir, patch_name)
                        if not os.path.exists(lr_path):
                            skipped_missing_lr += 1
                            continue

                        disc_path = os.path.join(self.disc_root, slide_id, patch_name.replace(".png", ".pt"))

                        self.items.append(
                            {
                                "case_id": case_id,
                                "slide_id": slide_id,
                                "hr_path": hr_path,
                                "lr_path": lr_path,
                                "disc_path": disc_path,
                                "time": surv.time,
                                "event": surv.event,
                            }
                        )

        print(
            f"[dataset] layout={'slide_level' if layout_is_slide_level else 'case_level'} | "
            f"items={len(self.items)} | "
            f"skipped(no_surv)={skipped_no_surv} | "
            f"skipped(no_hr_slide)={skipped_no_hr_slide} | "
            f"skipped(no_lr_slide)={skipped_no_lr_slide} | "
            f"skipped(no_done)={skipped_no_done} | "
            f"skipped(missing lr)={skipped_missing_lr} | "
            f"disc_root={self.disc_root}"
        )


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


    # -------- dataset API --------
    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]

        lr_img = self._read_rgb_pil_or_cv2(it["lr_path"])  # 128x128 RGB
        hr_img = self._read_rgb_pil_or_cv2(it["hr_path"])  # 512x512 RGB

        lr = self.transform_lr(lr_img)  # [3,128,128] in [0,1]
        hr = self.transform_hr(hr_img)  # [3,512,512] in [0,1]

        disc_path = it.get("disc_path", None)
        disc_map = None
        if disc_path is not None and os.path.isfile(disc_path):
            try:
                disc_map = torch.load(disc_path, map_location="cpu")
                if not torch.is_tensor(disc_map):
                    disc_map = None
                else:
                    disc_map = disc_map.float()

                    # ensure [1,H,W]
                    if disc_map.dim() == 2:
                        disc_map = disc_map.unsqueeze(0)
                    elif disc_map.dim() == 3 and disc_map.shape[0] != 1:
                        disc_map = disc_map[:1]

                    # enforce size (512,512)
                    if disc_map is not None and disc_map.shape[-2:] != (512, 512):
                        disc_map = torch.nn.functional.interpolate(
                            disc_map.unsqueeze(0),
                            size=(512, 512),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)

                    # clamp and contiguous for safe GPU transfer/multiply
                    if disc_map is not None:
                        disc_map = disc_map.clamp(0, 1).contiguous()
            except Exception:
                disc_map = None

        if disc_map is None:
            disc_map = torch.zeros((1, 512, 512), dtype=torch.float32)

        out = {
            "lr": lr,
            "hr": hr,
            "disc_map": disc_map,
            "time": torch.tensor(it["time"], dtype=torch.float32),
            "event": torch.tensor(it["event"], dtype=torch.long),
            "meta": {
                "case_id": it["case_id"],
                "slide_id": it["slide_id"],
                "lr_path": it["lr_path"],
                "hr_path": it["hr_path"],
                "disc_path": disc_path,
            },
        }
        return out

        
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
    disc_root: Optional[str] = None,  # ✅ NEW
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
        disc_root=disc_root,  # ✅ NEW
    )

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

    print(f"[split] cases: total={len(cases)} | train={len(train_cases)} | val={len(val_cases)} | test={len(test_cases)}")
    print(f"[split] patches: train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")

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
    
    