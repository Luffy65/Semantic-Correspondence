from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import glob
import json
import os

"""SPair-71k Dataset class, old version commented out below."""
# class Normalize(object):
#     def __init__(self, image_keys):
#         self.image_keys = image_keys
#         self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     def __call__(self, image):
#         for key in self.image_keys:
#             image[key] /= 255.0
#             image[key] = self.normalize(image[key])
#         return image


# def read_img(path):
#     img = np.array(Image.open(path).convert('RGB'))

#     return torch.tensor(img.transpose(2, 0, 1).astype(np.float32))


# class SPairDataset(Dataset):
#     def __init__(self, pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype):

#         self.datatype = datatype
#         self.pck_alpha = pck_alpha
#         self.ann_files = open(os.path.join(layout_path, dataset_size, datatype + '.txt'), "r").read().split('\n')
#         self.ann_files = self.ann_files[:len(self.ann_files) - 1]
#         self.pair_ann_path = pair_ann_path
#         self.image_path = image_path
#         self.categories = list(map(lambda x: os.path.basename(x), glob.glob('%s/*' % image_path)))
#         self.categories.sort()
#         self.transform = Normalize(['src_img', 'trg_img'])

#     def __len__(self):
#         return len(self.ann_files)

#     def __getitem__(self, idx):
#         # get pre-processed images
#         ann_file = self.ann_files[idx] + '.json'
#         with open(os.path.join(self.pair_ann_path, self.datatype, ann_file)) as f:
#             annotation = json.load(f)

#         category = annotation['category']
#         src_img = read_img(os.path.join(self.image_path, category, annotation['src_imname']))
#         trg_img = read_img(os.path.join(self.image_path, category, annotation['trg_imname']))

#         trg_bbox = annotation['trg_bndbox']
#         pck_threshold = max(trg_bbox[2] - trg_bbox[0],  trg_bbox[3] - trg_bbox[1]) * self.pck_alpha

#         sample = {'pair_id': annotation['pair_id'],
#                   'filename': annotation['filename'],
#                   'src_imname': annotation['src_imname'],
#                   'trg_imname': annotation['trg_imname'],
#                   'src_imsize': src_img.size(),
#                   'trg_imsize': trg_img.size(),

#                   'src_bbox': annotation['src_bndbox'],
#                   'trg_bbox': annotation['trg_bndbox'],
#                   'category': annotation['category'],

#                   'src_pose': annotation['src_pose'],
#                   'trg_pose': annotation['trg_pose'],

#                   'src_img': src_img,
#                   'trg_img': trg_img,
#                   'src_kps': torch.tensor(annotation['src_kps']).float(),
#                   'trg_kps': torch.tensor(annotation['trg_kps']).float(),

#                   'mirror': annotation['mirror'],
#                   'vp_var': annotation['viewpoint_variation'],
#                   'sc_var': annotation['scale_variation'],
#                   'truncn': annotation['truncation'],
#                   'occlsn': annotation['occlusion'],

#                   'pck_threshold': pck_threshold}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample

"""SPair-71k Dataset class."""
def read_img(path):
    img = np.array(Image.open(path).convert('RGB'))
    return torch.tensor(img.transpose(2, 0, 1).astype(np.float32))

class SPairDataset(Dataset):
    def __init__(self, pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype, target_size=224):
        self.datatype = datatype
        self.pck_alpha = pck_alpha
        self.target_size = target_size  # NEW: allow configurable size
        ann_list = os.path.join(layout_path, dataset_size, datatype + '.txt')
        self.ann_files = [x.strip() for x in open(ann_list, "r").readlines() if x.strip()]
        self.pair_ann_path = pair_ann_path
        self.image_path = image_path

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx):
      ann_file = self.ann_files[idx] + '.json'
      with open(os.path.join(self.pair_ann_path, self.datatype, ann_file)) as f:
          annotation = json.load(f)

      category = annotation['category']
      src_img = read_img(os.path.join(self.image_path, category, annotation['src_imname']))
      trg_img = read_img(os.path.join(self.image_path, category, annotation['trg_imname']))

      # ========== RESIZE ==========
      TARGET_H, TARGET_W = self.target_size, self.target_size  # CHANGED

      # Save original dimensions
      orig_h_src, orig_w_src = src_img.shape[1], src_img.shape[2]
      orig_h_trg, orig_w_trg = trg_img.shape[1], trg_img.shape[2]

      # Calculate scale factors
      scale_x_src = TARGET_W / orig_w_src
      scale_y_src = TARGET_H / orig_h_src
      scale_x_trg = TARGET_W / orig_w_trg
      scale_y_trg = TARGET_H / orig_h_trg

      # Resize images
      src_img = F.interpolate(
          src_img.unsqueeze(0),
          size=(TARGET_H, TARGET_W),
          mode='bilinear',
          align_corners=False
      ).squeeze(0).clone()

      trg_img = F.interpolate(
          trg_img.unsqueeze(0),
          size=(TARGET_H, TARGET_W),
          mode='bilinear',
          align_corners=False
      ).squeeze(0).clone()

      # Normalize (0-255 -> 0-1 -> ImageNet normalization)
      mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
      std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
      src_img = (src_img / 255.0 - mean) / std
      trg_img = (trg_img / 255.0 - mean) / std

      # Scale keypoints
      src_kps = torch.tensor(annotation['src_kps'], dtype=torch.float32)
      src_kps[:, 0] *= scale_x_src
      src_kps[:, 1] *= scale_y_src

      trg_kps = torch.tensor(annotation['trg_kps'], dtype=torch.float32)
      trg_kps[:, 0] *= scale_x_trg
      trg_kps[:, 1] *= scale_y_trg

      # ========== BBOX AND PCK THRESHOLD ==========
      trg_bbox = annotation['trg_bndbox']  # [xmin, ymin, xmax, ymax]

      # Scale the bbox to the new dimensions (consistent with resized image)
      trg_bbox_scaled = [
          trg_bbox[0] * scale_x_trg,
          trg_bbox[1] * scale_y_trg,
          trg_bbox[2] * scale_x_trg,
          trg_bbox[3] * scale_y_trg
      ]

      # Calculate PCK threshold on the scaled bbox
      pck_threshold = max(
          trg_bbox_scaled[2] - trg_bbox_scaled[0],
          trg_bbox_scaled[3] - trg_bbox_scaled[1]
      ) * self.pck_alpha

      return {
          'src_img': src_img,              # (3, target_size, target_size)
          'trg_img': trg_img,              # (3, target_size, target_size)
          'src_kps': src_kps,              # (N, 2) scaled
          'trg_kps': trg_kps,              # (N, 2) scaled
          'trg_bbox': torch.tensor(trg_bbox_scaled, dtype=torch.float32),  # (4,)
          'pck_threshold': pck_threshold,   # Scalar float
          'category': annotation.get('category', ''),
          'src_imsize': (orig_h_src, orig_w_src),  # Original dimensions
          'trg_imsize': (orig_h_trg, orig_w_trg)
      }

if __name__ == '__main__':
    # Get current script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(script_dir, "SPair-71k") + os.sep
    pair_ann_path = relative_path + 'PairAnnotation'
    layout_path = relative_path + 'Layout'
    image_path = relative_path + 'JPEGImages'
    dataset_size = 'large'
    pck_alpha = 0.1

    trn_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='trn')
    val_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='val')
    test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='test')

    trn_dataloader = DataLoader(trn_dataset, num_workers=1)
    val_dataloader = DataLoader(val_dataset, num_workers=1)
    test_dataloader = DataLoader(test_dataset, num_workers=1)

    for data in trn_dataloader:
        print('Trn: %20s %20s %10.2f' % (data['src_imname'], data['trg_imname'], data['pck_threshold']))

    for data in val_dataloader:
        print('Val: %20s %20s %10.2f' % (data['src_imname'], data['trg_imname'], data['pck_threshold']))

    for data in test_dataloader:
        print('Test: %20s %20s %10.2f' % (data['src_imname'], data['trg_imname'], data['pck_threshold']))

    print('SPair-71k dataset implementation example finished.')
