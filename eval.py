# First task: evaluate models on SPair-71k dataset using the PCK metric.
import torch
from segment_anything import sam_model_registry

def computePCKatT(thresholds=[0.05, 0.1, 0.2]):
    ... #TODO

# Access Dataset
# ... TODO

# SAM model
sam = sam_model_registry["default"](checkpoint="checkpoints/sam_vit_b_01ec64.pth")

# Test SAM on SPair-71k
# computePCKatT(sam(Dataset)) TODO
