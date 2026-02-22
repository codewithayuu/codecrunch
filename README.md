# Off-Road Semantic Segmentation — CodeCrunch

Pixel-level semantic segmentation of synthetic desert environments, built for the CodeCrunch hackathon by Duality AI on Unstop.

Segments off-road images into 7 terrain classes using DeepLabV3+ with ResNet-101. Trained on ~2,800 synthetic images from Duality's Falcon simulator.

---

## Results

| Metric | Value |
|--------|-------|
| Val mIoU | 0.7459 |
| Test mIoU | **0.4475** (multi-scale TTA) |
| Inference speed | 23.8 ms/image |
| Model parameters | 45.7M |
| Training time | ~3.5 hours on RTX 5050 |

The 0.30 gap between val and test comes from a domain shift — training and test images are from two different simulated deserts with different colors, rocks, and vegetation. Full analysis in the report.

---

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support
- ~2 GB disk space for model + predictions

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/codecrunch.git
cd codecrunch

conda create -n EDU python=3.10 -y
conda activate EDU

# PyTorch — pick one:
# RTX 50-series (Blackwell, sm_120) — nightly required:
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Older GPUs (RTX 20/30/40):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install -r requirements.txt
```

**RTX 5050 note:** Standard PyTorch 2.6 doesn't support Blackwell (sm_120). The nightly build is mandatory. Other GPUs work fine with the stable release.

---

## Usage

### Training

```bash
python train.py
```

Trains DeepLabV3+ ResNet-101 for 60 epochs at 512×896 resolution. Takes about 3.5 hours on an 8GB GPU. Best checkpoint saved automatically.

Key settings:
- 7-class output (3 absent test classes mapped to ignore)
- CE + Dice loss with Rocks weighted 5×
- 3-tier LR: encoder 3e-5, decoder 1e-4, seg head 3e-4
- Encoder frozen for first 5 epochs
- BFloat16 mixed precision

### Inference

Single image or folder:

```bash
python test.py --model model/best_model.pth --input path/to/images/ --output results/
python test.py --model model/best_model.pth --input single_image.png --output results/
```

Automatically applies multi-scale TTA (3 scales × hflip = 6 passes) and outputs color-coded segmentation maps at original resolution.

### Full benchmark run

```bash
python final_inference.py
```

Runs inference on all 1002 test images, computes per-class IoU, measures timing, and saves everything to the output directory.

---

## Project Structure

```
├── train.py                Training script
├── test.py                 Inference (single image or batch)
├── final_inference.py      Full benchmark + prediction generator
├── config.json             Training configuration
├── requirements.txt        Dependencies
├── report.pdf              8-page report
│
├── model/
│   └── best_model.pth      Trained weights (183 MB)
│
├── predictions/
│   ├── color/              1002 color-coded segmentation maps
│   └── raw/                1002 raw class-index masks
│
├── figures/
│   ├── training_curves.png
│   ├── per_class_iou.png
│   └── comparisons/        Input / GT / prediction side-by-side
│
└── benchmarks/
    ├── results.json         Machine-readable metrics
    └── results_summary.txt  Human-readable summary
```

---

## Classes

| Index | Class | Notes |
|-------|-------|-------|
| 0 | Trees | |
| 1 | Lush Bushes | Nearly absent in test (0.002%) |
| 2 | Dry Grass | |
| 3 | Dry Bushes | |
| 4 | Rocks | 18% of test — biggest challenge |
| 5 | Landscape | Largest test class (43%) |
| 6 | Sky | Easiest (IoU 0.98) |

Three original classes (Ground Clutter, Flowers, Logs) are absent from the test set and excluded from model output.

---

## Per-Class Performance

```
Class              Val IoU    Test IoU
───────────────────────────────────────
Sky                0.986      0.983
Landscape          0.687      0.625
Trees              0.874      0.505
Dry Bushes         0.520      0.498
Dry Grass          0.708      0.475
Rocks              0.566      0.046
Lush Bushes        0.724      0.001
───────────────────────────────────────
MEAN               0.746      0.448
```

Sky transfers perfectly. Rocks fails because the two deserts have completely different rock types and textures.

---

## The Domain Gap

The central challenge of this project. Training and test data come from different simulated environments:

- **Color:** Train RGB mean [149, 130, 107] vs test [123, 105, 94]
- **3 classes absent** from test entirely (Ground Clutter, Flowers, Logs)
- **Rocks:** 1.2% of train → 18.1% of test, and they look nothing alike
- **Key fix:** Suppressing absent classes at inference: test mIoU 0.31 → 0.44

We tried extreme augmentation, class merging, OHEM loss, and CutMix. None broke past 0.45. The gap is geological — you can't learn to recognize rocks you've never seen.

---

## Training Phases

| Phase | What changed | Val mIoU | Test mIoU |
|-------|-------------|----------|-----------|
| 2 | 512×512 baseline, 50 epochs | 0.649 | ~0.43 |
| 3 | 512×896, 100 epochs | 0.683 | 0.440 |
| 4 | Extreme augmentation | — | 0.439 |
| **5** | **7-class, 60 epochs** | **0.746** | **0.444** |
| 5+TTA | Multi-scale TTA | — | **0.448** |

---

## Hardware

| | |
|--|--|
| GPU | NVIDIA RTX 5050 Laptop, 8 GB VRAM |
| Architecture | Blackwell (sm_120) |
| PyTorch | 2.12.0.dev+cu128 (nightly) |
| Precision | BFloat16 |
| Peak VRAM | ~4.8 GB |
| Batch size | 4 at 512×896 |

---

## Mask Format Warning

If you work with this dataset: masks are UINT16 PNG with values up to 10,000. Default `cv2.imread()` converts to uint8 and silently destroys the data.

```python
# WRONG — corrupts mask values
mask = cv2.imread(path)

# CORRECT
mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
```

The starter code also had bugs: missing Flowers class (value 600) and a phantom Background class (value 0) that doesn't exist in any mask.

---

## License

Competition submission for CodeCrunch (Duality AI × Unstop).

