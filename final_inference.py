"""
FINAL INFERENCE & SUBMISSION ARTIFACT GENERATOR
=================================================
Generates ALL deliverables for CodeCrunch submission:
  1. Test mIoU (single-scale and MS-TTA)
  2. Per-class IoU breakdown
  3. 1002 color-coded prediction maps (original resolution)
  4. Inference time benchmarks
  5. Summary statistics file

Supports both 10-class (Phase 3) and 7-class (Phase 5) models.
Auto-detects model type from checkpoint.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore', category=UserWarning)

# ════════════════════════════════════════════════════════════════════
#  PATHS — EDIT THESE IF NEEDED
# ════════════════════════════════════════════════════════════════════

BASE_DIR = r"D:\project\hackathon\codecrunch"
OUTPUT_DIR = os.path.join(BASE_DIR, "final_submission")

# Model search order — tries Phase 5 first, then Phase 3
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "outputs_v5", "checkpoints", "best_model.pth"),
    os.path.join(BASE_DIR, "outputs_v3", "checkpoints", "best_model.pth"),
]

TEST_IMG_DIR = os.path.join(BASE_DIR, "Offroad_Segmentation_testImages", "Color_Images")
TEST_MSK_DIR = os.path.join(BASE_DIR, "Offroad_Segmentation_testImages", "Segmentation")

# Original image dimensions (predictions saved at this resolution)
ORIG_H, ORIG_W = 540, 960
# Model input dimensions
MODEL_H, MODEL_W = 512, 896

# ════════════════════════════════════════════════════════════════════
#  CLASS DEFINITIONS — BOTH 10-CLASS AND 7-CLASS
# ════════════════════════════════════════════════════════════════════

IGNORE_INDEX = 255

# 10-class (Phase 3)
RAW_TO_IDX_10 = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4,
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9,
}
CLASS_NAMES_10 = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
    "Flowers", "Logs", "Rocks", "Landscape", "Sky",
]
CLASS_COLORS_10 = np.array([
    [34,139,34], [0,255,0], [210,180,140], [139,90,43],
    [128,128,0], [255,0,255], [139,69,19], [128,128,128],
    [210,150,75], [135,206,235],
], dtype=np.uint8)
ABSENT_10 = [4, 5, 6]  # suppressed at inference

# 7-class (Phase 5)
RAW_TO_IDX_7 = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 255,
    600: 255, 700: 255, 800: 4, 7100: 5, 10000: 6,
}
CLASS_NAMES_7 = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Rocks", "Landscape", "Sky",
]
CLASS_COLORS_7 = np.array([
    [34,139,34], [0,255,0], [210,180,140], [139,90,43],
    [128,128,128], [210,150,75], [135,206,235],
], dtype=np.uint8)

# Raw pixel values for output masks (7-class to raw)
IDX7_TO_RAW = {0: 100, 1: 200, 2: 300, 3: 500, 4: 800, 5: 7100, 6: 10000}
IDX10_TO_RAW = {0:100, 1:200, 2:300, 3:500, 4:550, 5:600, 6:700, 7:800, 8:7100, 9:10000}


def build_lut(mapping):
    lut = np.full(10001, IGNORE_INDEX, dtype=np.uint8)
    for rv, ci in mapping.items():
        lut[rv] = ci
    return lut

LUT_10 = build_lut(RAW_TO_IDX_10)
LUT_7 = build_lut(RAW_TO_IDX_7)

# ════════════════════════════════════════════════════════════════════
#  DATASET
# ════════════════════════════════════════════════════════════════════

class TestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, model_h, model_w, lut):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.lut = lut
        self.files = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
        self.transform = A.Compose([
            A.Resize(model_h, model_w),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        print(f"  {len(self.files)} test images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, f)), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(os.path.join(self.mask_dir, f), cv2.IMREAD_UNCHANGED)
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        msk = self.lut[np.clip(msk.astype(np.int32), 0, 10000)]
        r = self.transform(image=img, mask=msk)
        return r['image'], r['mask'].long(), f

# ════════════════════════════════════════════════════════════════════
#  INFERENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_single_scale(model, imgs, num_classes, absent=None):
    """Single-scale prediction with optional class suppression."""
    logits = model(imgs)
    if absent:
        logits[:, absent, :, :] = -float('inf')
    return logits


@torch.no_grad()
def predict_multiscale(model, imgs, num_classes, scales=(0.75, 1.0, 1.25),
                       flip=True, absent=None):
    """Multi-scale TTA: average logits across scales and horizontal flip."""
    B, C, H, W = imgs.shape
    accum = torch.zeros(B, num_classes, H, W, device=imgs.device)

    for s in scales:
        sH = max(32, round(H * s / 32) * 32)
        sW = max(32, round(W * s / 32) * 32)
        scaled = F.interpolate(imgs, size=(sH, sW),
                               mode='bilinear', align_corners=False)

        logits = model(scaled)
        accum += F.interpolate(logits, size=(H, W),
                               mode='bilinear', align_corners=False)

        if flip:
            logits_f = model(torch.flip(scaled, [3]))
            logits_f = torch.flip(logits_f, [3])
            accum += F.interpolate(logits_f, size=(H, W),
                                   mode='bilinear', align_corners=False)

    if absent:
        accum[:, absent, :, :] = -float('inf')
    return accum


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, num_classes,
             absent=None, multiscale=False):
    """Compute mIoU with optional MS-TTA."""
    model.eval()
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)

    for imgs, msks, _ in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            if multiscale:
                logits = predict_multiscale(model, imgs, num_classes,
                                           absent=absent)
            else:
                logits = predict_single_scale(model, imgs, num_classes,
                                             absent=absent)
        pred = logits.argmax(1)
        valid = msks != IGNORE_INDEX
        for c in range(num_classes):
            pc = (pred == c) & valid
            mc = (msks == c) & valid
            inter[c] += (pc & mc).sum()
            union[c] += (pc | mc).sum()

    iou = inter / (union + 1e-6)
    vc = union > 0
    return iou[vc].mean().item(), iou.cpu().numpy(), vc.cpu().numpy()

# ════════════════════════════════════════════════════════════════════
#  PREDICTION GENERATION
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_predictions(model, loader, device, amp_dtype, num_classes,
                         absent, colors, idx_to_raw, out_color_dir,
                         out_raw_dir, multiscale=True):
    """Generate and save all test predictions at original resolution."""
    model.eval()
    os.makedirs(out_color_dir, exist_ok=True)
    os.makedirs(out_raw_dir, exist_ok=True)

    times = []
    count = 0

    for imgs, msks, fnames in tqdm(loader, desc="  Generating predictions"):
        imgs = imgs.to(device)
        t0 = time.time()

        with torch.amp.autocast('cuda', dtype=amp_dtype):
            if multiscale:
                logits = predict_multiscale(model, imgs, num_classes,
                                           absent=absent)
            else:
                logits = predict_single_scale(model, imgs, num_classes,
                                             absent=absent)

        # Resize to original resolution BEFORE argmax (better quality)
        logits_full = F.interpolate(logits, size=(ORIG_H, ORIG_W),
                                    mode='bilinear', align_corners=False)
        preds = logits_full.argmax(1).cpu().numpy()

        t1 = time.time()
        times.append((t1 - t0) / imgs.shape[0])

        for i in range(imgs.shape[0]):
            fname = fnames[i]
            pred = preds[i]

            # Color-coded prediction
            color_img = np.zeros((ORIG_H, ORIG_W, 3), dtype=np.uint8)
            for c in range(num_classes):
                color_img[pred == c] = colors[c]
            cv2.imwrite(os.path.join(out_color_dir, fname),
                        cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))

            # Raw prediction (original pixel values)
            raw_pred = np.zeros((ORIG_H, ORIG_W), dtype=np.uint16)
            for c in range(num_classes):
                if c in idx_to_raw:
                    raw_pred[pred == c] = idx_to_raw[c]
            cv2.imwrite(os.path.join(out_raw_dir, fname), raw_pred)

            count += 1

    avg_time = np.mean(times) * 1000  # ms per image
    return count, avg_time

# ════════════════════════════════════════════════════════════════════
#  VISUALIZATION HELPERS
# ════════════════════════════════════════════════════════════════════

def save_comparison_grid(model, loader, device, amp_dtype, num_classes,
                         absent, colors, class_names, out_dir, n=15):
    """Save side-by-side input/GT/prediction comparison images."""
    model.eval()
    os.makedirs(os.path.join(out_dir, 'comparisons'), exist_ok=True)
    mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(3, 1, 1)
    count = 0

    for imgs, msks, fnames in loader:
        if count >= n:
            break
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            logits = predict_multiscale(model, imgs.to(device), num_classes,
                                       absent=absent)
        preds = logits.argmax(1).cpu().numpy()

        for i in range(min(imgs.shape[0], n - count)):
            img_np = ((imgs[i] * std + mean).clamp(0, 1)
                      .permute(1, 2, 0).numpy() * 255).astype(np.uint8)

            gt_color = np.zeros((*msks[i].shape, 3), dtype=np.uint8)
            pred_color = np.zeros((*preds[i].shape, 3), dtype=np.uint8)
            for c in range(num_classes):
                gt_color[msks[i].numpy() == c] = colors[c]
                pred_color[preds[i] == c] = colors[c]

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            axes[0].imshow(img_np)
            axes[0].set_title('Input', fontsize=14)
            axes[0].axis('off')

            axes[1].imshow(gt_color)
            axes[1].set_title('Ground Truth', fontsize=14)
            axes[1].axis('off')

            axes[2].imshow(pred_color)
            axes[2].set_title('Prediction (MS-TTA)', fontsize=14)
            axes[2].axis('off')

            # Legend
            patches = [mpatches.Patch(color=colors[c]/255., label=class_names[c])
                       for c in range(num_classes)]
            fig.legend(handles=patches, loc='lower center', ncol=min(7, num_classes),
                       fontsize=10, framealpha=0.9)
            plt.tight_layout(rect=[0, 0.06, 1, 1])
            plt.savefig(os.path.join(out_dir, 'comparisons', fnames[i]),
                        dpi=120, bbox_inches='tight')
            plt.close()
            count += 1


def save_iou_chart(class_names, iou_ss, iou_ms, vc, out_dir):
    """Save per-class IoU comparison bar chart."""
    valid_names = [class_names[c] for c in range(len(class_names)) if vc[c]]
    valid_ss = [iou_ss[c] for c in range(len(class_names)) if vc[c]]
    valid_ms = [iou_ms[c] for c in range(len(class_names)) if vc[c]]

    x = np.arange(len(valid_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - w/2, valid_ss, w, label='Single-Scale', color='steelblue')
    bars2 = ax.bar(x + w/2, valid_ms, w, label='MS-TTA', color='coral')

    ax.set_ylabel('IoU', fontsize=13)
    ax.set_title('Per-Class IoU on Test Set', fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_names, rotation=30, ha='right', fontsize=11)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_class_iou.png'), dpi=150,
                bbox_inches='tight')
    plt.close()

# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  FINAL INFERENCE & SUBMISSION GENERATOR")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.bfloat16 if (device.type == 'cuda' and
                torch.cuda.is_bf16_supported()) else torch.float16
    print(f"\n  Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Find best model ──
    model_path = None
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            model_path = p
            break

    if model_path is None:
        print("  ❌ No model found! Check MODEL_CANDIDATES paths.")
        sys.exit(1)

    print(f"\n  Loading: {model_path}")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Auto-detect model type
    num_classes = ckpt.get('num_classes', None)
    if num_classes is None:
        # Infer from checkpoint — check segmentation head output channels
        for k, v in ckpt['model_state_dict'].items():
            if 'segmentation_head' in k and v.dim() == 4:
                num_classes = v.shape[0]
                break

    if num_classes == 7:
        print(f"  Model type: 7-class (Phase 5)")
        class_names = CLASS_NAMES_7
        colors = CLASS_COLORS_7
        lut = LUT_7
        absent = None  # no suppression needed
        idx_to_raw = IDX7_TO_RAW
    else:
        num_classes = 10
        print(f"  Model type: 10-class (Phase 3) — will suppress classes 4,5,6")
        class_names = CLASS_NAMES_10
        colors = CLASS_COLORS_10
        lut = LUT_10
        absent = ABSENT_10
        idx_to_raw = IDX10_TO_RAW

    # Build model
    encoder = ckpt.get('config', {}).get('encoder_name', 'resnet101')
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,  # loading from checkpoint
        in_channels=3,
        classes=num_classes,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"  Val mIoU: {ckpt.get('val_miou', '?')}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # ── Dataset ──
    print(f"\n  Loading test data...")
    test_ds = TestDataset(TEST_IMG_DIR, TEST_MSK_DIR, MODEL_H, MODEL_W, lut)
    test_ld = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)

    # ══════════════════════════════════════════════════════════════
    #  EVALUATION
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  EVALUATION")
    print(f"{'─'*60}")

    # Single-scale
    print(f"\n  Single-scale evaluation...")
    t0 = time.time()
    ss_miou, ss_pc, ss_vc = evaluate(
        model, test_ld, device, amp_dtype, num_classes,
        absent=absent, multiscale=False)
    ss_time = time.time() - t0
    print(f"  Single-scale mIoU: {ss_miou:.4f} ({ss_time:.1f}s)")
    for c in range(num_classes):
        if ss_vc[c]:
            print(f"    {class_names[c]:<18} {ss_pc[c]:.4f}")

    # Multi-scale TTA
    print(f"\n  Multi-scale TTA (×0.75, ×1.0, ×1.25 + hflip = 6 passes)...")
    t0 = time.time()
    ms_miou, ms_pc, ms_vc = evaluate(
        model, test_ld, device, amp_dtype, num_classes,
        absent=absent, multiscale=True)
    ms_time = time.time() - t0
    print(f"  MS-TTA mIoU: {ms_miou:.4f} ({ms_time:.1f}s)")
    for c in range(num_classes):
        if ms_vc[c]:
            tta_gain = ms_pc[c] - ss_pc[c]
            print(f"    {class_names[c]:<18} {ms_pc[c]:.4f}  (TTA: {tta_gain:+.4f})")

    # ══════════════════════════════════════════════════════════════
    #  GENERATE PREDICTIONS
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  GENERATING 1002 PREDICTIONS (MS-TTA, original resolution)")
    print(f"{'─'*60}")

    color_dir = os.path.join(OUTPUT_DIR, "predictions_color")
    raw_dir = os.path.join(OUTPUT_DIR, "predictions_raw")

    n_saved, avg_ms = generate_predictions(
        model, test_ld, device, amp_dtype, num_classes,
        absent, colors, idx_to_raw,
        color_dir, raw_dir, multiscale=True)
    print(f"  Saved {n_saved} predictions")
    print(f"  Average inference time: {avg_ms:.1f}ms per image (MS-TTA)")

    # Single-scale timing (for benchmark)
    print(f"\n  Benchmarking single-scale inference time...")
    ss_times = []
    with torch.no_grad():
        for imgs, _, _ in test_ld:
            imgs = imgs.to(device)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                _ = model(imgs)
            torch.cuda.synchronize()
            ss_times.append((time.time() - t0) / imgs.shape[0] * 1000)
    ss_avg_ms = np.mean(ss_times)
    print(f"  Single-scale: {ss_avg_ms:.1f}ms per image")

    # ══════════════════════════════════════════════════════════════
    #  VISUALIZATIONS FOR REPORT
    # ══════════════════════════════════════════════════════════════

    print(f"\n{'─'*60}")
    print(f"  GENERATING REPORT FIGURES")
    print(f"{'─'*60}")

    # Comparison grid
    print(f"  Saving comparison images...")
    save_comparison_grid(model, test_ld, device, amp_dtype, num_classes,
                         absent, colors, class_names, OUTPUT_DIR, n=15)

    # IoU bar chart
    print(f"  Saving IoU chart...")
    save_iou_chart(class_names, ss_pc, ms_pc, ms_vc, OUTPUT_DIR)

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════

    results = {
        "model_path": model_path,
        "model_type": f"{num_classes}-class",
        "encoder": encoder,
        "parameters": params,
        "epoch": ckpt.get('epoch', None),
        "val_miou": float(ckpt.get('val_miou', 0)),
        "test_miou_single_scale": float(ss_miou),
        "test_miou_ms_tta": float(ms_miou),
        "tta_gain": float(ms_miou - ss_miou),
        "inference_time_ms_single": round(ss_avg_ms, 1),
        "inference_time_ms_tta": round(avg_ms, 1),
        "num_test_images": n_saved,
        "image_resolution": f"{ORIG_H}×{ORIG_W}",
        "model_input_resolution": f"{MODEL_H}×{MODEL_W}",
        "per_class_iou_single": {},
        "per_class_iou_ms_tta": {},
    }

    for c in range(num_classes):
        if ms_vc[c]:
            results["per_class_iou_single"][class_names[c]] = round(float(ss_pc[c]), 4)
            results["per_class_iou_ms_tta"][class_names[c]] = round(float(ms_pc[c]), 4)

    with open(os.path.join(OUTPUT_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Human-readable summary
    summary = []
    summary.append("=" * 60)
    summary.append("CODECRUNCH — FINAL SUBMISSION RESULTS")
    summary.append("=" * 60)
    summary.append("")
    summary.append(f"Model:          DeepLabV3+ {encoder}")
    summary.append(f"Parameters:     {params:,}")
    summary.append(f"Classes:        {num_classes}")
    summary.append(f"Input size:     {MODEL_H}×{MODEL_W}")
    summary.append(f"Output size:    {ORIG_H}×{ORIG_W}")
    summary.append("")
    summary.append(f"Val mIoU:              {ckpt.get('val_miou', 0):.4f}")
    summary.append(f"Test mIoU (SS):        {ss_miou:.4f}")
    summary.append(f"Test mIoU (MS-TTA):    {ms_miou:.4f}  (+{ms_miou-ss_miou:.4f})")
    summary.append("")
    summary.append(f"Inference time (SS):   {ss_avg_ms:.1f}ms/image")
    summary.append(f"Inference time (TTA):  {avg_ms:.1f}ms/image")
    summary.append("")
    summary.append("Per-Class IoU (MS-TTA):")
    summary.append("-" * 40)
    for c in range(num_classes):
        if ms_vc[c]:
            summary.append(f"  {class_names[c]:<18} {ms_pc[c]:.4f}")
    summary.append("-" * 40)
    summary.append(f"  {'MEAN':<18} {ms_miou:.4f}")
    summary.append("")
    if num_classes == 10:
        summary.append("Note: Classes 4 (Ground Clutter), 5 (Flowers), 6 (Logs)")
        summary.append("are absent from test set and suppressed at inference.")
    summary.append("")
    summary.append(f"Predictions saved to: {OUTPUT_DIR}/")
    summary.append(f"  predictions_color/  — {n_saved} color-coded PNG files")
    summary.append(f"  predictions_raw/    — {n_saved} raw class-value PNG files")
    summary.append(f"  comparisons/        — 15 input/GT/pred comparison images")
    summary.append(f"  per_class_iou.png   — IoU bar chart")
    summary.append(f"  results.json        — machine-readable results")

    summary_text = "\n".join(summary)
    print(f"\n{summary_text}")

    with open(os.path.join(OUTPUT_DIR, "results_summary.txt"), 'w') as f:
        f.write(summary_text)

    print(f"\n  ✅ All submission artifacts generated in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
