# Tag Corner Detection — Classical + Deep Learning

> Detect the 4 corner coordinates **(x1,y1, x2,y2, x3,y3, x4,y4)** of a tag in images using both **classical computer vision** and a **deep learning model** (ResNet-18 + PyTorch Lightning).  
> If no tag is present, the model rejects the prediction via a learned confidence score.

---

##  Repository Structure

```
 tag-corner-detection
 ┣  corner_detection.ipynb   ← Main notebook — open in Google Colab
 ┗  README.md
```

---

## Features

- **Classical pipeline** — Gaussian blur → Canny edges → morphological closing → quadrilateral contour fitting
- **Deep learning pipeline** — ResNet-18 backbone + upsampling decoder + dual heads (corner regression + confidence)
- **No-tag rejection** — confidence score thresholding suppresses false positives on empty images
- **Multi-task loss** — SmoothL1 (corners) + BCEWithLogits (confidence), combined with tunable weights
- **Training utilities** — ModelCheckpoint, EarlyStopping, LearningRateMonitor, resume-from-checkpoint
- **Portable** — works with Google Drive **or** direct file upload, no hardcoded paths

---

## Required Data Files

> ata files are **not included** in this repository (too large for GitHub). Provide them via Drive or direct upload as described above.

| File | Description |
|------|-------------|
| `flash_images.rar` | RAR archive containing all images |
| `train_corners.csv` | Training annotations |
| `val_corners.csv` | Validation annotations |

### CSV Format

Both CSV files must have exactly these two columns:

| Column | Description |
|--------|-------------|
| `filename` | Image filename (basename only, e.g. `img_001.jpg`) |
| `corners` | Space-separated 8 floats: `x1 y1 x2 y2 x3 y3 x4 y4` |

> If all 8 corner values are `0 0 0 0 0 0 0 0` → image has no tag (`has_tag = False`)

---
### Option A — Google Drive

1. Upload the 3 data files to a folder in your Google Drive:
   ```
   My Drive/tag_boundary_detection_training_data/
       flash_images.rar
       train_corners.csv
       val_corners.csv
   ```
2. Open `corner_detection.ipynb` in Google Colab
3. In **Section 1**, set:
   ```python
   USE_DRIVE     = True
   GDRIVE_FOLDER = "tag_boundary_detection_training_data"  # change if your folder is named differently
   ```
4. `Runtime → Run all`

>  Drive is **automatically unmounted** after copying. All subsequent cells use Colab local storage.

---

### Option B — Direct Upload *(recommended for reviewers)*

1. Open `corner_detection.ipynb` in Google Colab
2. In **Section 1**, set:
   ```python
   USE_DRIVE = False
   ```
3. Click the **Files** icon in the left Colab sidebar → **⬆️ Upload** → select all 3 files:
   - `flash_images.rar`
   - `train_corners.csv`
   - `val_corners.csv`
4. `Runtime → Run all`

---

## Model Architecture

```
Input Image (512×512)
       │
  ResNet-18 Backbone (pretrained ImageNet)
  stem → layer1 → layer2 → layer3 → layer4
       │
  Decoder (3× Upsample + Conv blocks)
  dec4 (512→256) → dec3 (256→128) → dec2 (128→64)
       │
  AdaptiveAvgPool2d → Flatten → Dropout
       │
  ┌────┴─────┐
  │          │
fc_coords  fc_conf
(8 outputs) (1 output)
  │          │
sigmoid    (raw logit)
  │          │
corners   confidence
```

## Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup | Install deps, mount Drive / upload files, extract RAR |
| 2 | CSV Loading | Parse `train_corners.csv` and `val_corners.csv` |
| 3 | Visualization | Ground truth corners overlaid on sample images |
| 4 | Classical — Preprocessing | Grayscale → Gaussian blur → Canny edges |
| 5 | Classical — Detection | Contour fitting → quadrilateral → ordered corners |
| 6 | DL — Dataset | `CornerDataset` with augmentation and normalization |
| 7 | DL — Model | `CornerNetLightning` definition |
| 8 | DL — Training | Trainer with checkpoint, early stopping, LR monitor |

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 512 × 512 |
| Batch size | 8 |
| Backbone LR | 1e-5 |
| Head LR | 1e-4 |
| Optimizer | AdamW (weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR (T_max=30) |
| Max epochs | 50 (EarlyStopping patience=7) |
| Precision | FP16 mixed |
| Val split | 15% |

---

## Dependencies

All installed automatically in the notebook:

```bash
apt-get install unrar
pip install rarfile patool opencv-python-headless albumentations pytorch-lightning
# torch / torchvision pre-installed in Colab
```
