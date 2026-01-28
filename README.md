# Semantic Correspondence with Visual Foundation Models

**Project for Advanced Machine Learning Course**  
**Academic Year:** 2025-2026  
**Instructor:** Tatiana Tommasi  
**TA:** Claudia Cuttano  
**Institution:** Politecnico di Torino  
**Students:** Luca Amoroso, Antea Bonaccorso, Antonio Pio Caruso, Negar Hosseinali Pour

---

## Overview

This project explores semantic correspondence: the task of finding pixel-level matches between semantically similar parts across different images. For example, given a keypoint marking the left eye of a dog in one image, the goal is to predict the corresponding location of that same semantic part in another image, potentially with different viewpoints, scales, or even domains.

We leverage pretrained Visual Foundation Models to establish dense semantic correspondences without requiring task-specific training from scratch. Our approach systematically evaluates:

1. **Training-free baselines** using frozen feature representations
2. **Light fine-tuning** of the last transformer layers
3. **Prediction refinement** via windowed soft-argmax
4. **Generalization** across datasets (SPair-71k and PF-Willow)

---

## Problem Statement

Given a source image $I_s$ with an annotated keypoint $p_s$ and a target image $I_t$, the goal is to predict the location $\hat{p}_t$ in the target image that corresponds to the same semantic object part.

**Key Challenges:**
- Objects may appear in different viewpoints, scales, or contexts
- Images may come from different visual domains (e.g., photos vs. paintings)
- Models must distinguish semantically similar but geometrically different parts

---

## Methodology

### 1. Training-Free Baseline: Cosine Similarity Matching

We extract dense feature maps $F(I) \in \mathbb{R}^{H' \times W' \times D}$ from frozen pretrained backbones. For each source keypoint, we compute cosine similarity with all spatial locations in the target feature map and select the location with maximum similarity:

$$\hat{p}_t = \arg\max_{r} \cos(f_s, F(I_t)[r])$$

This simple baseline serves as a strong reference point to evaluate how well different foundation models encode semantic structure.

**Evaluated Backbones:**
- **DINOv2 ViT-B/14**: Self-supervised Vision Transformer trained with contrastive objectives
- **DINOv3 ViT-B/16**: Improved version with enhanced robustness
- **SAM ViT-B**: Segment Anything model with segmentation-oriented representations

### 2. Light Fine-Tuning

We adapt pretrained representations to the correspondence task by unfreezing only the last layers while keeping the remaining parameters frozen:

- **DINO models**: Unfreeze last 2 transformer layers + normalization layer
- **SAM**: Unfreeze last 2 layers + neck component

Fine-tuning uses a **contrastive loss** that encourages high similarity between matching features while discouraging similarity with non-matching locations:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\cos(f_s^i, f_t^i) / \tau)}{\sum_{j=1}^{N} \exp(\cos(f_s^i, f_t^j) / \tau)}$$

**Training Configuration:**
- Learning rate: $5 \times 10^{-6}$
- Temperature: $\tau = 0.07$
- Batch size: 16
- Optimizer: AdamW

### 3. Windowed Soft-Argmax Refinement

The global argmax produces discrete predictions and is sensitive to local noise. We refine predictions using a windowed soft-argmax that:

1. Obtains a coarse correspondence via global argmax
2. Applies softmax over similarity scores within a local window
3. Computes weighted average of spatial coordinates for sub-pixel localization

$$\tilde{p}_t = \sum_{r \in \mathcal{W}(\hat{p}_t)} r \cdot \frac{\exp(\alpha \cdot S(r))}{\sum_{k \in \mathcal{W}(\hat{p}_t)} \exp(\alpha \cdot S(k))}$$

**Parameters:**
- Window radius: $W = 5$
- Temperature: $\alpha = 0.01$

---

## Datasets and Evaluation

### Datasets

- **SPair-71k**: Primary benchmark with dense semantic keypoint annotations across 18 object categories
- **PF-Willow**: Extension dataset for evaluating cross-domain generalization without retraining

### Evaluation Metric: PCK (Percentage of Correct Keypoints)

A predicted correspondence is correct if its Euclidean distance from the ground truth is smaller than $\alpha$ times the image size (normalized by max dimension). We report PCK at multiple thresholds: $\alpha \in \{0.05, 0.1, 0.15, 0.2\}$

**Reporting Methods:**
- **Per-keypoint PCK**: Average correctness over all keypoints in the dataset
- **Per-category PCK**: Aggregated scores by object class to analyze category-specific performance
- **Per-image PCK**: Mean, median, std, min, max across all image pairs to measure consistency

---

## Results Summary

### Training-Free Baseline (Without Fine-Tuning)

| Model | PCK@0.05 | PCK@0.1 | PCK@0.15 | PCK@0.2 |
|-------|----------|---------|----------|---------|
| **DINOv2 ViT-B/14** | **36.75%** | **54.16%** | **64.07%** | **70.78%** |
| DINOv3 ViT-B/16 | 35.86% | 54.05% | 63.63% | 69.55% |
| SAM ViT-B | 13.15% | 23.24% | 31.03% | 37.76% |

**Key Finding:** Both DINO models significantly outperform SAM across all thresholds, indicating that self-supervised ViTs trained for representation learning are inherently well-suited for semantic correspondence.

### Fine-Tuned Models (Best Results)

| Model | PCK@0.05 | PCK@0.1 | PCK@0.15 | PCK@0.2 |
|-------|----------|---------|----------|---------|
| **DINOv2 Finetuned** | **60.02%** | 74.89% | 81.39% | 85.37% |
| **DINOv3 Finetuned** | 58.15% | **75.77%** | **82.34%** | **86.23%** |
| SAM Finetuned | 19.23% | 31.87% | 40.81% | 47.87% |

**Improvements from Fine-Tuning:**
- DINOv2: +20.18% at PCK@0.1
- DINOv3: +20.73% at PCK@0.1
- SAM: +8.63% at PCK@0.1

Even after fine-tuning, SAM remains significantly behind DINO models, highlighting architectural limitations of segmentation-oriented features for dense semantic correspondence.

### Per-Category Analysis

DINO models achieve consistently strong performance across nearly all categories:
- **Articulated/deformable objects** (bird, cat, dog, horse, person): 75-90% PCK@0.1
- **Rigid objects** (train, bus, car): 75-90% PCK@0.1
- **Challenging categories** (chair, boat): 50-60% PCK@0.1

SAM shows limited effectiveness, particularly on objects with high intra-class variability, typically achieving 20-50% PCK@0.1.

---

## Key Insights

### Why Do DINO Models Excel?

DINO models are trained with contrastive learning objectives that encourage semantic consistency across patches. This naturally aligns with the semantic correspondence task, where matching parts should have highly similar feature representations.

### Why Does SAM Underperform?

SAM features are optimized for segmentation boundaries rather than semantic similarity. They encode "what's an object edge" rather than "what's semantically similar." Additionally:
- SAM was designed to be prompted with visual inputs (points, bounding boxes, or masks)
- In our baseline, we don't leverage prompts
- Segmentation-oriented representations lack the semantic alignment needed for correspondence

### Layer Selection Matters

Intermediate layers perform poorly (Layer 6: ~11% PCK@0.1), while final layers achieve strong results (~36% PCK@0.1 for DINOv2). This confirms that **high-level semantic representations are crucial for accurate correspondence estimation**.

---

## Implementation Details

**Image Preprocessing:**
- DINOv2: Resized to $518 \times 518$
- DINOv3: Resized to $512 \times 512$
- SAM: Resized to $512 \times 512$

**Normalization:**
- DINO models: ImageNet normalization (mean: 0.485, 0.456, 0.406; std: 0.229, 0.224, 0.225)
- SAM: Model-specific normalization (mean: 123.675, 116.28, 103.53; std: 58.395, 57.12, 57.375)

**Framework:** PyTorch with AdamW optimizer

---

## Getting Started

### Prerequisites

See `requirements.txt` for all dependencies.

### Basic Usage

1. **Prepare data**: Download SPair-71k and PF-Willow datasets from the [Google Drive folder](https://drive.google.com/drive/folders/1fEWpONVft365O47IhEDLKZ2a0WkAuhyP?usp=sharing)

2. **Run training-free baseline**: See `main.ipynb` for computing correspondences with frozen features

3. **Fine-tune models**: Uncomment fine-tuning cells to adapt last layers

4. **Evaluate results**: Run `task4.ipynb` to compute PCK metrics and generate reports

---

## References

- SPair-71k: Min et al. (2019)
- DINOv2: Oquab et al. (2023)
- DINOv3: Simeoni et al. (2025)
- SAM: Kirillov et al. (2023)
- Soft-argmax refinement: Zhang et al. (2024)

---

## Resources

- [Google Drive - Data Folder](https://drive.google.com/drive/folders/1fEWpONVft365O47IhEDLKZ2a0WkAuhyP?usp=sharing)
- [Colab Wandb](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb)

