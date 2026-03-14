# SAM2 Indoor Scene Segmentation

Fine-tuning **Segment Anything Model 2 (SAM2-Large)** on a small indoor stereo scene dataset using **LoRA (Low-Rank Adaptation)** for parameter-efficient transfer learning.

**Val IoU: 0.8777** achieved after 60 epochs on only 114 training images.

---

## Two Model Comparison

| | Model 1 | Model 2 |
|---|---|---|
| **Ground Truth Mask** | `mask_00.png` | `mask_cat.png` |
| **Task** | Scene geometry segmentation | Clean object segmentation |
| **Val IoU** | 0.8777 | 0.8777 |
| **Trainable Params** | 6.3M / 224M (2.8%) | 6.3M / 224M (2.8%) |
| **Architecture** | SAM2-Large + LoRA rank=4 | SAM2-Large + LoRA rank=4 |
| **Output Folder** | `models/` | `models_maskcat/` |

---

## Dataset

- **38 indoor scenes** captured with a stereo camera setup
- **114 training images** from `camera_00` — 188 train / 40 val after split
- **3 mask types** per scene:

| File | Description |
|------|-------------|
| `mask_00.png` | Scene geometry/edge mask from camera_00 |
| `mask_02.png` | Same from camera_02 (right stereo camera) |
| `mask_cat.png` | Clean category mask — main object only |

- Original resolution: **4112 x 3008** — trained at **1024 x 1024**

---

## Results

### Model 1 — Scene Geometry (mask_00.png)

| Desk | Mirror3 | Sanitaries |
|------|---------|------------|
| ![](predictions/Desk/0000_overlay.png) | ![](predictions/Mirror3/0000_overlay.png) | ![](predictions/Sanitaries/0000_overlay.png) |

### Model 2 — Clean Object (mask_cat.png)

| Desk | Mirror3 | Sanitaries |
|------|---------|------------|
| ![](predictions_maskcat/Desk/0000_overlay.png) | ![](predictions_maskcat/Mirror3/0000_overlay.png) | ![](predictions_maskcat/Sanitaries/0000_overlay.png) |

---

## Architecture

```
Input Image (1024x1024)
        |
Image Encoder — Hiera ViT  [FROZEN]
        |
backbone_fpn features (256x256, 128x128, 64x64)
        |
Prompt Encoder  [FROZEN]  — None prompts, zero embeddings
        |
Mask Decoder + LoRA adapters  [TRAINABLE — 6.3M params]
        |
Segmentation Mask (1024x1024)
```

---

## Key Improvements Over Baseline

| Aspect | Baseline | Ours |
|--------|----------|------|
| SAM2 API | `sam2(imgs)` — broken | Correct 3-stage forward pass |
| Fine-tuning | Full decoder ~25M params | LoRA 6.3M params (2.8%) |
| Normalisation | `/255` only | ImageNet mean/std |
| Augmentation | None | 8 techniques |
| Crop strategy | Simple resize | Random 1024x1024 patch |
| Train/Val split | None | 80/20 scene-level |
| Loss function | BCE + Dice | BCE + Dice + Focal + IoU |
| LR schedule | Cosine only | 5-epoch warmup + cosine |
| Mixed precision | No | FP16 AMP |
| Gradient clipping | No | clip_grad_norm=1.0 |
| Validation | None | Every epoch + best checkpoint |
| GPU memory | ~18GB | ~8GB |

---

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/sam2.git

# 2. Download SAM2-Large checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# 3. Train Model 1 — scene geometry
# In dataset.py set: MASK_TYPE = "mask_00.png"
python train.py --data train --ckpt sam2_hiera_large.pt --epochs 60 --batch 1 --lr 3e-4 --lora-rank 4 --out models

# 4. Train Model 2 — clean object
# In dataset.py set: MASK_TYPE = "mask_cat.png"
python train.py --data train --ckpt sam2_hiera_large.pt --epochs 60 --batch 1 --lr 3e-4 --lora-rank 4 --out models_maskcat

# 5. Run inference
python infer.py --data val_mono_nogt --ckpt models/sam2_best.pth --out predictions
python infer.py --data val_mono_nogt --ckpt models_maskcat/sam2_best.pth --out predictions_maskcat
```

---

## Switching Between Models

In `dataset.py` change the `MASK_TYPE` variable:

```python
# Model 1 — scene geometry segmentation
MASK_TYPE = "mask_00.png"

# Model 2 — clean object segmentation
MASK_TYPE = "mask_cat.png"
```

---

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 60 |
| Batch Size | 1 |
| Learning Rate | 3e-4 |
| LR Schedule | 5-epoch warmup + cosine annealing |
| Weight Decay | 1e-4 |
| LoRA Rank | 4 |
| Mixed Precision | FP16 AMP |
| Gradient Clipping | 1.0 |
| Optimizer | AdamW |
| GPU | RTX 4060 8GB |

---

## Loss Function

```
Total Loss = 0.3 x BCE + 0.4 x Dice + 0.2 x Focal + 0.1 x IoU
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| Binary Cross-Entropy | 0.3 | Pixel-level accuracy |
| Dice Loss | 0.4 | Shape and overlap similarity |
| Focal Loss (a=0.8, y=2.0) | 0.2 | Handle class imbalance |
| IoU Loss | 0.1 | Directly optimise evaluation metric |

---

## Files

| File | Purpose |
|------|---------|
| `dataset.py` | Dataset loader with augmentation, train/val split, MASK_TYPE config |
| `loss.py` | Combined segmentation loss |
| `train.py` | Training loop with LoRA, AMP, validation, checkpoint saving |
| `infer.py` | Inference — saves binary masks and overlay images |
| `requirements.txt` | Python dependencies |

---

## Requirements

```
torch>=2.1.0
torchvision>=0.16.0
tqdm>=4.65.0
albumentations==1.3.1
opencv-python>=4.8.0
numpy>=1.24.0
```

---

**Computer Vision Project | March 2026**
