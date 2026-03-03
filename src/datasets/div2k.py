import os
import glob
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


IMG_EXTS = ("png", "jpg", "jpeg", "bmp")


def list_images(folder: str):
    files = []
    for e in IMG_EXTS:
        files += glob.glob(os.path.join(folder, f"*.{e}"))
    return sorted(files)


def lr_to_hr_name(lr_name: str, scale: int) -> str:
    # DIV2K LR bicubic: 0001x4.png -> HR: 0001.png
    suffix = f"x{scale}"
    if lr_name.endswith(".png") and suffix in lr_name:
        return lr_name.replace(suffix, "")
    return lr_name


class DIV2KPairDataset(Dataset):
    def __init__(self, lr_dir: str, hr_dir: str, scale: int = 4, patch_size: int = 192, training: bool = True):
        super().__init__()
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.scale = scale
        self.patch = patch_size
        self.training = training

        self.lr_files = list_images(lr_dir)
        if len(self.lr_files) == 0:
            raise FileNotFoundError(f"No LR images found in: {lr_dir}")

        self.hr_files = []
        for p in self.lr_files:
            lr_base = os.path.basename(p)
            hr_base = lr_to_hr_name(lr_base, scale)
            self.hr_files.append(os.path.join(hr_dir, hr_base))

        missing = [p for p in self.hr_files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} HR files. Example missing: {missing[0]}\n"
                f"lr_dir={lr_dir}\nhr_dir={hr_dir}\n"
                "Likely LR names like 0001x4.png while HR is 0001.png."
            )

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx: int):
        lr = Image.open(self.lr_files[idx]).convert("RGB")
        hr = Image.open(self.hr_files[idx]).convert("RGB")

        if self.training:
            lr_patch = self.patch // self.scale
            w, h = lr.size
            if w < lr_patch or h < lr_patch:
                raise ValueError(
                    f"LR image too small for patch. LR size=({w},{h}) lr_patch={lr_patch}. "
                    f"Try smaller patch_size in config."
                )

            x = random.randint(0, w - lr_patch)
            y = random.randint(0, h - lr_patch)

            lr = lr.crop((x, y, x + lr_patch, y + lr_patch))
            hr = hr.crop((x * self.scale, y * self.scale,
                          (x + lr_patch) * self.scale, (y + lr_patch) * self.scale))

            # augment
            if random.random() < 0.5:
                lr = TF.hflip(lr); hr = TF.hflip(hr)
            if random.random() < 0.5:
                lr = TF.vflip(lr); hr = TF.vflip(hr)

        lr_t = TF.to_tensor(lr)
        hr_t = TF.to_tensor(hr)
        return lr_t, hr_t