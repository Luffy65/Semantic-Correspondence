import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import torchvision.transforms as transforms


class PFWillowDataset(Dataset):
    """
    PF-Willow Dataset for Semantic Correspondence.
    
    Dataset structure expected:
    pf-willow/
    ├── PF-dataset/
    │   ├── car/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── face/
    │   ├── motorbike/
    │   ├── duck/
    │   └── wineBottle/
    └── test_pairs.csv
    
    Args:
        root_dir: Path to pf-willow directory
        pairs_file: Name of CSV file containing image pairs (default: 'test_pairs.csv')
        transform: Optional transform to apply to images
        return_original_size: If True, returns original image sizes for PCK calculation
    """
    
    def __init__(
        self,
        root_dir: str,
        pairs_file: str = 'test_pairs.csv',
        transform: Optional[transforms.Compose] = None,
        return_original_size: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.dataset_dir = self.root_dir / 'PF-dataset'
        self.transform = transform
        self.return_original_size = return_original_size
        
        # Load pairs CSV
        pairs_path = self.root_dir / pairs_file
        if not pairs_path.exists():
            raise FileNotFoundError(
                f"Pairs file not found at {pairs_path}. "
                f"Download from the repository and place in {self.root_dir}"
            )
        
        self.pairs_df = pd.read_csv(pairs_path)
        
        # Validate dataset structure
        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found at {self.dataset_dir}. "
                f"Expected structure: {self.root_dir}/PF-dataset/"
            )
        
        self.categories = ['car', 'face', 'motorbike', 'duck', 'wineBottle']
        
    def __len__(self) -> int:
        return len(self.pairs_df)
    
    def _load_keypoints(self, anno_path: Path) -> np.ndarray:
        """
        Load keypoint annotations from file.
        
        Returns:
            numpy array of shape (N, 3) where N is number of keypoints
            Each row is [x, y, visibility] where visibility: 0=occluded, 1=visible
        """
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")
        
        with open(anno_path, 'r') as f:
            lines = f.readlines()
        
        keypoints = []
        for line in lines:
            # Parse keypoint format: typically "x y visibility" or "x y"
            parts = line.strip().split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                visibility = int(parts[2]) if len(parts) > 2 else 1
                keypoints.append([x, y, visibility])
        
        return np.array(keypoints, dtype=np.float32)
    
    def _get_image_and_anno_paths(
        self, 
        category: str, 
        image_name: str
    ) -> Tuple[Path, Path]:
        """Get paths to image and annotation files."""
        category_dir = self.dataset_dir / category
        
        # Image path
        img_path = category_dir / f"{image_name}.jpg"
        if not img_path.exists():
            img_path = category_dir / f"{image_name}.png"
        
        # Annotation path (typically .txt or .pts)
        anno_path = category_dir / f"{image_name}.txt"
        if not anno_path.exists():
            anno_path = category_dir / f"{image_name}.pts"
        
        return img_path, anno_path
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a pair of images with their keypoint annotations.
        
        Returns:
            Dictionary containing:
                - src_img: Source image tensor
                - trg_img: Target image tensor
                - src_kps: Source keypoints (N, 3)
                - trg_kps: Target keypoints (N, 3)
                - src_size: Original source image size (H, W)
                - trg_size: Original target image size (H, W)
                - category: Object category
                - pair_idx: Pair index
        """
        row = self.pairs_df.iloc[idx]
        
        # Extract pair information
        category = row['category']
        src_name = row['source']
        trg_name = row['target']
        
        # Get file paths
        src_img_path, src_anno_path = self._get_image_and_anno_paths(category, src_name)
        trg_img_path, trg_anno_path = self._get_image_and_anno_paths(category, trg_name)
        
        # Load images
        src_img = Image.open(src_img_path).convert('RGB')
        trg_img = Image.open(trg_img_path).convert('RGB')
        
        # Store original sizes for PCK calculation
        src_size = torch.tensor([src_img.height, src_img.width])
        trg_size = torch.tensor([trg_img.height, trg_img.width])
        
        # Load keypoints
        src_kps = self._load_keypoints(src_anno_path)
        trg_kps = self._load_keypoints(trg_anno_path)
        
        # Apply transforms if provided
        if self.transform is not None:
            # Store original keypoints
            orig_src_kps = src_kps.copy()
            orig_trg_kps = trg_kps.copy()
            
            src_img = self.transform(src_img)
            trg_img = self.transform(trg_img)
            
            # Adjust keypoints for resizing if transform includes resize
            # This assumes transform outputs tensors of shape (C, H, W)
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
    
    def get_category_indices(self, category: str):
        """Get all indices for a specific category."""
        return self.pairs_df[self.pairs_df['category'] == category].index.tolist()
    
    def get_num_keypoints(self) -> int:
        """Returns number of keypoints per image (10 for PF-Willow)."""
        return 10


# Example usage
if __name__ == "__main__":
    # Define transforms compatible with your vision models
    transform = transforms.Compose([
        transforms.Resize((518, 518)),  # DINOv2 typical size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create dataset
    dataset = PFWillowDataset(
        root_dir='/path/to/pf-willow',
        pairs_file='test_pairs.csv',
        transform=transform,
        return_original_size=True
    )
    
    # Create dataloader
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Test loading
    for batch in dataloader:
        print(f"Source images: {batch['src_img'].shape}")
        print(f"Target images: {batch['trg_img'].shape}")
        print(f"Source keypoints: {batch['src_kps'].shape}")
        print(f"Category: {batch['category']}")
        break