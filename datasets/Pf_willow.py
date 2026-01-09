import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class PFWillowDataset(Dataset):
    """
    PF-Willow Dataset Loader WITHOUT relying on test_pairs.csv.
    It auto-generates image pairs within the same category.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        return_original_size: bool = True,
        num_pairs_per_category: int = 100  # you can adjust this
    ):
        self.root_dir = Path(root_dir)
        self.dataset_dir = self.root_dir / 'PF-dataset'
        self.transform = transform
        self.return_original_size = return_original_size

        self.categories = ['car', 'face', 'motorbike', 'duck', 'wineBottle']
        self.pairs = []

        # Generate pairs within each category
        for category in self.categories:
            img_files = list((self.dataset_dir / category).glob("*.jpg"))
            img_files += list((self.dataset_dir / category).glob("*.png"))
            img_names = [f.stem for f in img_files if (self.dataset_dir / category / f.stem).with_suffix(".txt").exists()]

            for _ in range(num_pairs_per_category):
                if len(img_names) < 2:
                    continue
                src, trg = random.sample(img_names, 2)
                self.pairs.append((category, src, trg))

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_keypoints(self, anno_path: Path) -> np.ndarray:
        with open(anno_path, 'r') as f:
            lines = f.readlines()
        keypoints = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                visibility = int(parts[2]) if len(parts) > 2 else 1
                keypoints.append([x, y, visibility])
        return np.array(keypoints, dtype=np.float32)

    def _get_image_and_anno_paths(self, category: str, image_name: str) -> Tuple[Path, Path]:
        category_dir = self.dataset_dir / category
        img_path = category_dir / f"{image_name}.jpg"
        if not img_path.exists():
            img_path = category_dir / f"{image_name}.png"
        anno_path = category_dir / f"{image_name}.txt"
        if not anno_path.exists():
            anno_path = category_dir / f"{image_name}.pts"
        return img_path, anno_path

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        category, src_name, trg_name = self.pairs[idx]

        src_img_path, src_anno_path = self._get_image_and_anno_paths(category, src_name)
        trg_img_path, trg_anno_path = self._get_image_and_anno_paths(category, trg_name)

        src_img = Image.open(src_img_path).convert('RGB')
        trg_img = Image.open(trg_img_path).convert('RGB')

        src_size = torch.tensor([src_img.height, src_img.width])
        trg_size = torch.tensor([trg_img.height, trg_img.width])

        src_kps = self._load_keypoints(src_anno_path)
        trg_kps = self._load_keypoints(trg_anno_path)

        if self.transform is not None:
            src_img = self.transform(src_img)
            trg_img = self.transform(trg_img)

            # Adjust keypoints for resized image
            if isinstance(src_img, torch.Tensor):
                _, new_h, new_w = src_img.shape
                scale_x = new_w / src_size[1].item()
                scale_y = new_h / src_size[0].item()
                src_kps[:, 0] *= scale_x
                src_kps[:, 1] *= scale_y

                _, new_h, new_w = trg_img.shape
                scale_x = new_w / trg_size[1].item()
                scale_y = new_h / trg_size[0].item()
                trg_kps[:, 0] *= scale_x
                trg_kps[:, 1] *= scale_y

        return {
            'src_img': src_img,
            'trg_img': trg_img,
            'src_kps': torch.from_numpy(src_kps),
            'trg_kps': torch.from_numpy(trg_kps),
            'src_size': src_size if self.return_original_size else None,
            'trg_size': trg_size if self.return_original_size else None,
            'category': category,
            'pair_idx': idx,
            'src_name': src_name,
            'trg_name': trg_name,
        }

    def get_num_keypoints(self) -> int:
        return 10  # Fixed for PF-Willow