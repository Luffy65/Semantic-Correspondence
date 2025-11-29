#TODO: Correct this?

class SemanticCorrespondenceDataset(Dataset):
    def __init__(self, dataset_name, split='train', transforms=None):
        """
        Args:
            dataset_name: 'PF-PASCAL', 'PF-WILLOW', or 'SPair-71k'
            split: 'train', 'val', or 'test'
            transforms: image transformations
        """
        self.dataset_name = dataset_name
        self.split = split
        self.transforms = transforms
        
        # Load annotations
        self.pairs = self._load_pairs()
        
    def __getitem__(self, idx):
        pair_data = self.pairs[idx]
        
        # Load source and target images
        src_img = Image.open(pair_data['src_img_path'])
        tgt_img = Image.open(pair_data['tgt_img_path'])
        
        # Load keypoint annotations
        src_kps = pair_data['src_keypoints']  # (N, 2)
        tgt_kps = pair_data['tgt_keypoints']  # (N, 2)
        
        # Apply transformations
        if self.transforms:
            src_img, src_kps = self.transforms(src_img, src_kps)
            tgt_img, tgt_kps = self.transforms(tgt_img, tgt_kps)
        
        return {
            'src_img': src_img,
            'tgt_img': tgt_img,
            'src_kps': src_kps,
            'tgt_kps': tgt_kps,
            'n_pts': len(src_kps),  # number of valid keypoints
            'category': pair_data['category']
        }