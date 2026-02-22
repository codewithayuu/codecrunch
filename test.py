"""
CodeCrunch — Off-Road Semantic Segmentation
Test Inference Script
============================================
Usage:
    python test.py --model best_model.pth --input test_images/ --output predictions/

Generates color-coded segmentation maps for all input images.
Supports DeepLabV3+ with ResNet-101 encoder (10 or 7 class).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm

# ── Class definitions ──
CLASS_COLORS_10 = np.array([
    [34,139,34],[0,255,0],[210,180,140],[139,90,43],[128,128,0],
    [255,0,255],[139,69,19],[128,128,128],[210,150,75],[135,206,235]], dtype=np.uint8)
CLASS_COLORS_7 = np.array([
    [34,139,34],[0,255,0],[210,180,140],[139,90,43],
    [128,128,128],[210,150,75],[135,206,235]], dtype=np.uint8)
ABSENT_10 = [4, 5, 6]


def load_model(model_path, device):
    """Load model from checkpoint, auto-detect class count."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Detect number of classes
    num_classes = ckpt.get('num_classes', None)
    if num_classes is None:
        for k, v in ckpt['model_state_dict'].items():
            if 'segmentation_head' in k and v.dim() == 4:
                num_classes = v.shape[0]; break
    if num_classes is None:
        num_classes = 10  # default

    encoder = ckpt.get('config', {}).get('encoder_name', 'resnet101')
    model = smp.DeepLabV3Plus(
        encoder_name=encoder, encoder_weights=None,
        in_channels=3, classes=num_classes).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, num_classes, encoder


def predict(model, img_path, device, num_classes, model_h=512, model_w=896):
    """Run inference on a single image with MS-TTA."""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    transform = A.Compose([
        A.Resize(model_h, model_w),
        A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ToTensorV2()])
    tensor = transform(image=img)['image'].unsqueeze(0).to(device)

    # Multi-scale TTA
    B, C, H, W = tensor.shape
    accum = torch.zeros(B, num_classes, H, W, device=device)

    with torch.no_grad():
        for s in [0.75, 1.0, 1.25]:
            sH = max(32, round(H*s/32)*32)
            sW = max(32, round(W*s/32)*32)
            scaled = F.interpolate(tensor, (sH,sW), mode='bilinear', align_corners=False)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                lo = model(scaled)
            accum += F.interpolate(lo, (H,W), mode='bilinear', align_corners=False)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                lf = model(torch.flip(scaled,[3]))
            accum += F.interpolate(torch.flip(lf,[3]), (H,W), mode='bilinear', align_corners=False)

    # Suppress absent classes (10-class model only)
    if num_classes == 10:
        accum[:, ABSENT_10, :, :] = -float('inf')

    # Resize to original and argmax
    logits_full = F.interpolate(accum, (orig_h, orig_w),
                                mode='bilinear', align_corners=False)
    pred = logits_full.argmax(1).squeeze().cpu().numpy()
    return pred


def colorize(pred, num_classes):
    """Convert class index map to RGB color image."""
    colors = CLASS_COLORS_7 if num_classes == 7 else CLASS_COLORS_10
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for c in range(num_classes):
        out[pred == c] = colors[c]
    return out


def main():
    parser = argparse.ArgumentParser(description='CodeCrunch Test Inference')
    parser.add_argument('--model', required=True, help='Path to model .pth')
    parser.add_argument('--input', required=True, help='Input image or directory')
    parser.add_argument('--output', default='predictions', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, num_classes, encoder = load_model(args.model, device)
    print(f"Model: DeepLabV3+ {encoder}, {num_classes} classes")

    os.makedirs(args.output, exist_ok=True)

    if os.path.isfile(args.input):
        files = [args.input]
    else:
        files = sorted(os.path.join(args.input, f)
                       for f in os.listdir(args.input)
                       if f.lower().endswith(('.png', '.jpg')))

    print(f"Processing {len(files)} images...")
    for fpath in tqdm(files):
        pred = predict(model, fpath, device, num_classes)
        color = colorize(pred, num_classes)
        fname = os.path.basename(fpath)
        cv2.imwrite(os.path.join(args.output, fname),
                    cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(files)} predictions to {args.output}/")


if __name__ == '__main__':
    main()
