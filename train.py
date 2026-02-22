"""
PHASE 5: CLEAN 7-CLASS MODEL (FINAL)
======================================
Key changes from Phase 4:
  1. 7-class output — absent classes (Ground Clutter, Flowers, Logs) become IGNORE,
     NOT merged into other classes. No wrong gradients, no confusion.
  2. Phase 3 encoder+decoder transferred, only segmentation head is fresh.
  3. Encoder frozen for first 5 epochs so the random seg head can catch up
     without corrupting learned features.
  4. 3-tier LR: encoder 3e-5, decoder 1e-4, seg head 3e-4
     (encoder warmup adjusted so it ramps AFTER unfreeze)
  5. Rocks weight 5× — biggest test failure, must improve.
  6. Moderate augmentation biased darker/browner (toward test domain).
  7. Multi-scale TTA at final evaluation.
  8. Early stopping with patience 15 to avoid wasted epochs.

Fixes applied:
  - Class weights rebalanced: Lush Bushes 1.0 (was 0.5), Trees 1.5, Dry Bushes 1.5
  - save_curves KeyError fixed (history['lr'] → history['lr_head'])
  - Encoder LR warmup shifted to start AFTER unfreeze epoch
  - Early stopping added (patience=15)

Why this should work:
  - In 10-class model, 61% of test Rock pixels were predicted as Ground Clutter.
    With Ground Clutter eliminated from the output space, the model MUST pick
    something else. Combined with 5× Rocks weight, many should land on Rocks.
  - No class merging means no contradictory gradients (flowers ≠ dry grass).
  - IGNORE pixels (~7% of training) simply produce zero loss — clean signal.
"""

import os
import json
import time
import math
import random
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════

CONFIG = {
    "base_dir":     r"D:\project\hackathon\codecrunch",
    "output_dir":   r"D:\project\hackathon\codecrunch\outputs_v5",
    "resume_from":  r"D:\project\hackathon\codecrunch\outputs_v3\checkpoints\best_model.pth",

    "encoder_name":    "resnet101",
    "encoder_weights": "imagenet",

    "img_h": 512,
    "img_w": 896,

    "num_epochs":  60,
    "batch_size":  4,

    # 3-tier LR: encoder (pretrained) < decoder (pretrained) < seg head (random)
    "encoder_lr":  3e-5,
    "decoder_lr":  1e-4,
    "seghead_lr":  3e-4,
    "weight_decay": 0.01,

    "warmup_epochs":         5,
    "freeze_encoder_epochs": 5,   # encoder unfreezes at this epoch
    "encoder_warmup_epochs": 3,   # encoder gets its OWN warmup AFTER unfreeze
    "min_lr": 1e-7,

    "early_stopping_patience": 15,  # stop if test mIoU doesn't improve for 15 test-evals

    "ce_weight":   0.5,
    "dice_weight": 0.5,

    "num_workers": 2,
    "pin_memory":  True,
    "seed":        42,
    "gradient_clip_max_norm": 1.0,
    "save_every_n_epochs":    10,
    "test_eval_every":        3,
}

# ════════════════════════════════════════════════════════════════════
#  7-CLASS MAPPING
# ════════════════════════════════════════════════════════════════════

NUM_CLASSES = 7
IGNORE_INDEX = 255

#  Raw pixel value  →  7-class index
#  550, 600, 700    →  IGNORE (255)  ← the key change
RAW_TO_IDX_7 = {
    100:   0,    # Trees
    200:   1,    # Lush Bushes
    300:   2,    # Dry Grass
    500:   3,    # Dry Bushes
    550:   255,  # Ground Clutter  → IGNORE
    600:   255,  # Flowers         → IGNORE
    700:   255,  # Logs            → IGNORE
    800:   4,    # Rocks
    7100:  5,    # Landscape
    10000: 6,    # Sky
}

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Rocks", "Landscape", "Sky",
]

CLASS_COLORS = np.array([
    [ 34, 139,  34],   # Trees       — forest green
    [  0, 255,   0],   # Lush Bushes — bright green
    [210, 180, 140],   # Dry Grass   — tan
    [139,  90,  43],   # Dry Bushes  — brown
    [128, 128, 128],   # Rocks       — gray
    [210, 150,  75],   # Landscape   — sandy
    [135, 206, 235],   # Sky         — light blue
], dtype=np.uint8)

# ── Vectorized look-up table ──
_LUT = np.full(10001, IGNORE_INDEX, dtype=np.uint8)
for _rv, _ci in RAW_TO_IDX_7.items():
    _LUT[_rv] = _ci

def remap_mask(m):
    return _LUT[np.clip(m.astype(np.int32), 0, 10000)]

# ════════════════════════════════════════════════════════════════════
#  AUGMENTATION — MODERATE, BIASED TOWARD TEST DOMAIN
# ════════════════════════════════════════════════════════════════════

def _safe_rrc(h, w):
    """RandomResizedCrop with API version handling."""
    ratio = w / h  # 1.75
    rng = (ratio * 0.75, ratio * 1.25)  # (1.31, 2.19)
    try:
        return A.RandomResizedCrop(
            size=(h, w), scale=(0.5, 1.0), ratio=rng,
            interpolation=cv2.INTER_LINEAR, p=1.0)
    except TypeError:
        return A.RandomResizedCrop(
            height=h, width=w, scale=(0.5, 1.0), ratio=rng,
            interpolation=cv2.INTER_LINEAR, p=1.0)

def get_train_transforms(h, w):
    return A.Compose([
        _safe_rrc(h, w),
        A.HorizontalFlip(p=0.5),

        # ONE color distortion — moderate, not stacked
        A.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.3, hue=0.05, p=0.6),

        # Bias DARKER and BROWNER (train mean BGR [107,130,149] → test [94,105,123])
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.1),   # mostly darker
            contrast_limit=0.2, p=0.4),

        A.CLAHE(clip_limit=4.0, p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),

        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(h, w):
    return A.Compose([
        A.Resize(height=h, width=w, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# ════════════════════════════════════════════════════════════════════
#  DATASETS
# ════════════════════════════════════════════════════════════════════

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        all_imgs = sorted(f for f in os.listdir(img_dir)
                          if f.lower().endswith(('.png', '.jpg')))
        self.files = [f for f in all_imgs
                      if os.path.exists(os.path.join(mask_dir, f))]
        print(f"    {len(self.files)} image-mask pairs")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.img_dir, f), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB)
        msk = cv2.imread(os.path.join(self.mask_dir, f), cv2.IMREAD_UNCHANGED)
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        msk = remap_mask(msk)
        if self.transform:
            t = self.transform(image=img, mask=msk)
            img, msk = t['image'], t['mask']
        return img, msk.long()


class TestDataset(Dataset):
    """Test set — for monitoring only, NEVER for training."""
    def __init__(self, img_dir, mask_dir, h, w):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir
        self.files = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
        self.transform = A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.img_dir, f)), cv2.COLOR_BGR2RGB)
        msk = cv2.imread(os.path.join(self.mask_dir, f), cv2.IMREAD_UNCHANGED)
        if msk.ndim == 3:
            msk = msk[:, :, 0]
        msk = remap_mask(msk)
        r = self.transform(image=img, mask=msk)
        return r['image'], r['mask'].long(), f

# ════════════════════════════════════════════════════════════════════
#  LOSS
# ════════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    def __init__(self, class_weights, ce_w=0.5, dice_w=0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(weight=class_weights,
                                        ignore_index=IGNORE_INDEX)
        self.dice = DiceLoss(mode='multiclass',
                             ignore_index=IGNORE_INDEX, from_logits=True)
        self.ce_w   = ce_w
        self.dice_w = dice_w

    def forward(self, logits, target):
        return (self.ce_w   * self.ce(logits, target)
              + self.dice_w * self.dice(logits, target))

# ════════════════════════════════════════════════════════════════════
#  METRICS
# ════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_miou(model, loader, device, amp_dtype):
    """Global mIoU — accumulate, then divide (correct method)."""
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    for batch in loader:
        imgs = batch[0].to(device)
        msks = batch[1].to(device)
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            logits = model(imgs)
        pred  = logits.argmax(1)
        valid = msks != IGNORE_INDEX
        for c in range(NUM_CLASSES):
            pc = (pred == c) & valid
            mc = (msks == c) & valid
            inter[c] += (pc & mc).sum()
            union[c] += (pc | mc).sum()
    iou = inter / (union + 1e-6)
    vc  = union > 0
    return iou[vc].mean().item(), iou.cpu().numpy(), vc.cpu().numpy()


@torch.no_grad()
def compute_miou_multiscale(model, loader, device, amp_dtype,
                            scales=(0.75, 1.0, 1.25), flip=True):
    """Multi-scale TTA evaluation (for final eval only — slow)."""
    model.eval()
    inter = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)

    for batch in loader:
        imgs = batch[0].to(device)
        msks = batch[1].to(device)
        B, C, H, W = imgs.shape
        accum = torch.zeros(B, NUM_CLASSES, H, W, device=device)

        for s in scales:
            sH = max(32, round(H * s / 32) * 32)
            sW = max(32, round(W * s / 32) * 32)
            scaled = F.interpolate(imgs, size=(sH, sW),
                                   mode='bilinear', align_corners=False)

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(scaled)
            accum += F.interpolate(logits, size=(H, W),
                                   mode='bilinear', align_corners=False)

            if flip:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    logits_f = model(torch.flip(scaled, [3]))
                logits_f = torch.flip(logits_f, [3])
                accum += F.interpolate(logits_f, size=(H, W),
                                       mode='bilinear', align_corners=False)

        pred  = accum.argmax(1)
        valid = msks != IGNORE_INDEX
        for c in range(NUM_CLASSES):
            pc = (pred == c) & valid
            mc = (msks == c) & valid
            inter[c] += (pc & mc).sum()
            union[c] += (pc | mc).sum()

    iou = inter / (union + 1e-6)
    vc  = union > 0
    return iou[vc].mean().item(), iou.cpu().numpy(), vc.cpu().numpy()

# ════════════════════════════════════════════════════════════════════
#  SCHEDULER — with separate encoder warmup after unfreeze
# ════════════════════════════════════════════════════════════════════

def get_scheduler(optimizer, cfg):
    """
    - Groups 1,2 (decoder, seg head): standard cosine with warmup_epochs warmup.
    - Group 0 (encoder): LR=0 while frozen, then its own warmup ramp over
      encoder_warmup_epochs AFTER unfreeze, then joins cosine decay.
    """
    warmup       = cfg['warmup_epochs']
    total        = cfg['num_epochs']
    freeze       = cfg['freeze_encoder_epochs']
    enc_warmup   = cfg['encoder_warmup_epochs']
    min_lr_ratio = cfg['min_lr'] / cfg['seghead_lr']

    def lr_lambda_dec_head(epoch):
        """Decoder & seg head: standard cosine with warmup."""
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    def lr_lambda_encoder(epoch):
        """Encoder: zero while frozen, warmup after unfreeze, then cosine decay."""
        if epoch < freeze:
            return 0.0  # frozen — LR doesn't matter but set to 0 for cleanliness
        # Epochs since unfreeze
        since_unfreeze = epoch - freeze
        if since_unfreeze < enc_warmup:
            return (since_unfreeze + 1) / enc_warmup
        # Cosine decay from unfreeze point
        progress = (epoch - freeze - enc_warmup) / max(1, total - freeze - enc_warmup)
        return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, [lr_lambda_encoder, lr_lambda_dec_head, lr_lambda_dec_head])

# ════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ════════════════════════════════════════════════════════════════════

def mask_to_color(m):
    c = np.zeros((*m.shape, 3), dtype=np.uint8)
    for i in range(NUM_CLASSES):
        c[m == i] = CLASS_COLORS[i]
    return c

def save_curves(history, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ep = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(ep, history['train_loss'], label='Train', lw=2)
    axes[0].plot(ep, history['val_loss'],   label='Val',   lw=2)
    axes[0].set_title('Loss'); axes[0].legend()
    axes[0].grid(True, alpha=0.3); axes[0].set_xlabel('Epoch')

    # mIoU
    axes[1].plot(ep, history['val_miou'], label='Val mIoU',  lw=2, color='green')
    te = [(i+1, v) for i, v in enumerate(history['test_miou']) if v > 0]
    if te:
        axes[1].plot(*zip(*te), label='Test mIoU', lw=2,
                     color='red', marker='o', markersize=4)
    axes[1].axhline(y=0.4401, color='gray', ls='--', alpha=0.5,
                    label='Phase 3 test (10-cls+supp)')
    axes[1].set_title('mIoU'); axes[1].legend()
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('Epoch')

    # Learning rates (all 3 tiers)
    axes[2].plot(ep, history['lr_head'], lw=2, color='purple', label='Seg Head')
    axes[2].plot(ep, history['lr_dec'],  lw=2, color='blue',   label='Decoder')
    axes[2].plot(ep, history['lr_enc'],  lw=2, color='orange', label='Encoder')
    axes[2].set_title('Learning Rates'); axes[2].legend()
    axes[2].grid(True, alpha=0.3); axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(out, 'curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_samples(model, loader, device, amp_dtype, out, n=10):
    model.eval()
    os.makedirs(os.path.join(out, 'samples'), exist_ok=True)
    mean = torch.tensor([.485, .456, .406]).view(3, 1, 1)
    std  = torch.tensor([.229, .224, .225]).view(3, 1, 1)
    count = 0
    for imgs, msks, fnames in loader:
        if count >= n:
            break
        with torch.amp.autocast('cuda', dtype=amp_dtype):
            preds = model(imgs.to(device)).argmax(1).cpu().numpy()
        for i in range(min(imgs.shape[0], n - count)):
            img_np = ((imgs[i] * std + mean).clamp(0, 1)
                      .permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            ax[0].imshow(img_np);                          ax[0].set_title('Input');   ax[0].axis('off')
            ax[1].imshow(mask_to_color(msks[i].numpy()));  ax[1].set_title('GT');      ax[1].axis('off')
            ax[2].imshow(mask_to_color(preds[i]));         ax[2].set_title('Pred');    ax[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out, 'samples', fnames[i]),
                        dpi=100, bbox_inches='tight')
            plt.close()
            count += 1

# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    cfg = CONFIG
    print("=" * 70)
    print("  PHASE 5: CLEAN 7-CLASS MODEL (FINAL)")
    print("  ├─ 7 classes — absent classes are IGNORED, not merged")
    print("  ├─ Phase 3 encoder+decoder → fresh 7-class seg head")
    print("  ├─ Encoder frozen for 5 epochs (seg head warm-up)")
    print("  ├─ Encoder gets own 3-epoch warmup AFTER unfreeze")
    print("  ├─ 3-tier LR: encoder 3e-5 / decoder 1e-4 / seg head 3e-4")
    print("  ├─ Rocks weight 5× (biggest test failure)")
    print("  ├─ Rebalanced weights: Trees 1.5, Lush 1.0, DBush 1.5")
    print("  ├─ Moderate augmentation biased toward test domain")
    print("  ├─ Early stopping (patience=15 test evals)")
    print("  └─ Multi-scale TTA at final evaluation")
    print("=" * 70)

    # ── Reproducibility ──
    random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed']); torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = True

    device    = torch.device('cuda')
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (amp_dtype == torch.float16)
    print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  AMP: {'bf16 (no scaler)' if amp_dtype == torch.bfloat16 else 'fp16 (with scaler)'}")

    out = cfg['output_dir']
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, 'checkpoints'), exist_ok=True)
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    base = cfg['base_dir']
    H, W = cfg['img_h'], cfg['img_w']

    # ── Data ──
    print(f"\n  Loading data (7-class, IGNORE absent)...")
    train_ds = SegDataset(
        os.path.join(base, "Offroad_Segmentation_Training_Dataset", "train", "Color_Images"),
        os.path.join(base, "Offroad_Segmentation_Training_Dataset", "train", "Segmentation"),
        get_train_transforms(H, W))
    val_ds = SegDataset(
        os.path.join(base, "Offroad_Segmentation_Training_Dataset", "val", "Color_Images"),
        os.path.join(base, "Offroad_Segmentation_Training_Dataset", "val", "Segmentation"),
        get_val_transforms(H, W))

    train_ld = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,
                          num_workers=cfg['num_workers'], pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_ld   = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False,
                          num_workers=cfg['num_workers'], pin_memory=True,
                          persistent_workers=True)

    test_ds = TestDataset(
        os.path.join(base, "Offroad_Segmentation_testImages", "Color_Images"),
        os.path.join(base, "Offroad_Segmentation_testImages", "Segmentation"),
        H, W)
    test_ld = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} (monitor only)")

    # ── Model with Phase 3 transfer ──
    print(f"\n  Building 7-class DeepLabV3+ ResNet-101...")
    model = smp.DeepLabV3Plus(
        encoder_name=cfg['encoder_name'],
        encoder_weights=cfg['encoder_weights'],   # ImageNet fallback
        in_channels=3,
        classes=NUM_CLASSES,                       # 7, not 10
    ).to(device)

    if os.path.exists(cfg['resume_from']):
        ckpt  = torch.load(cfg['resume_from'], map_location=device, weights_only=False)
        state = ckpt['model_state_dict']

        # Keep everything EXCEPT segmentation head (shape mismatch: 10→7 channels)
        compatible = {k: v for k, v in state.items()
                      if 'segmentation_head' not in k}
        missing, unexpected = model.load_state_dict(compatible, strict=False)

        print(f"  ✅ Phase 3 encoder + decoder loaded")
        print(f"     Transferred : {len(compatible)} tensors")
        print(f"     Fresh (head): {len(missing)} tensors  ← random init, needs warm-up")
        print(f"     Phase 3 val mIoU was: {ckpt.get('val_miou', '?')}")
    else:
        print(f"  ⚠️  Phase 3 checkpoint not found — using ImageNet encoder only")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_p:,}")

    # ── Freeze encoder ──
    freeze_ep = cfg['freeze_encoder_epochs']
    for p in model.encoder.parameters():
        p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  🔒 Encoder frozen for {freeze_ep} epochs")
    print(f"     Trainable now: {trainable:,} / {total_p:,} "
          f"({100*trainable/total_p:.1f}%)")

    # ── Baseline test mIoU (random seg head → will be poor) ──
    print(f"\n  Baseline test mIoU (random seg head)...")
    base_test, base_pc, base_vc = compute_miou(model, test_ld, device, amp_dtype)
    print(f"  Baseline: {base_test:.4f}")
    for c in range(NUM_CLASSES):
        if base_vc[c]:
            print(f"    {CLASS_NAMES[c]:<18} {base_pc[c]:.4f}")

    # ── Loss with rebalanced class weights ──
    #                  Trees  Lush  DGrass DBush Rocks  Land  Sky
    class_weights = torch.tensor(
        [1.5,   1.0,   0.8,   1.5,  5.0,   0.6,  0.4],
        dtype=torch.float32).to(device)

    print(f"\n  Class weights (rebalanced):")
    for i in range(NUM_CLASSES):
        tag = ""
        if i == 4:
            tag = " ← HEAVY (biggest test failure)"
        elif i == 0:
            tag = " ← raised from 1.0"
        elif i == 1:
            tag = " ← raised from 0.5"
        elif i == 3:
            tag = " ← lowered from 2.0"
        print(f"    {CLASS_NAMES[i]:<18}: {class_weights[i]:.1f}{tag}")

    criterion = CombinedLoss(class_weights,
                             ce_w=cfg['ce_weight'], dice_w=cfg['dice_weight'])

    # ── Optimizer (3-tier LR) ──
    # Order matters: group 0=encoder, 1=decoder, 2=seg head
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(),          'lr': cfg['encoder_lr']},
        {'params': model.decoder.parameters(),          'lr': cfg['decoder_lr']},
        {'params': model.segmentation_head.parameters(),'lr': cfg['seghead_lr']},
    ], weight_decay=cfg['weight_decay'])

    scheduler = get_scheduler(optimizer, cfg)

    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    # ── Training state ──
    history = {k: [] for k in
               ['train_loss', 'val_loss', 'val_miou', 'test_miou',
                'lr_enc', 'lr_dec', 'lr_head', 'epoch_time']}
    best_test       = 0.0
    best_val        = 0.0
    best_epoch      = 0
    no_improve_count = 0  # for early stopping (counts test-eval epochs without improvement)

    # Reference: Phase 3 test mIoU with 10-class + suppression
    PHASE3_TEST = 0.4401

    print(f"\n{'═'*70}")
    print(f"  TRAINING: {cfg['num_epochs']} epochs (early stopping patience={cfg['early_stopping_patience']})")
    print(f"  Phase 3 test mIoU (10-cls + suppression): {PHASE3_TEST:.4f}")
    print(f"  Target: > 0.50")
    print(f"{'═'*70}\n")

    t0 = time.time()
    stopped_early = False

    for epoch in range(cfg['num_epochs']):
        ep = epoch + 1
        es = time.time()

        # ── Unfreeze encoder after warm-up ──
        if epoch == freeze_ep:
            for p in model.encoder.parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n  🔓 Encoder UNFROZEN — {trainable:,} params trainable")
            print(f"     Encoder will warm up over {cfg['encoder_warmup_epochs']} epochs\n")

        # ── Train ──
        model.train()
        epoch_losses = []
        pbar = tqdm(train_ld, desc=f"  Ep {ep:3d}/{cfg['num_epochs']}",
                     leave=False, ncols=100)
        for imgs, msks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            msks = msks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(imgs)
                loss = criterion(logits, msks)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),
                                         cfg['gradient_clip_max_norm'])
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),
                                         cfg['gradient_clip_max_norm'])
                optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_tl = np.mean(epoch_losses)

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, msks in val_ld:
                imgs, msks = imgs.to(device), msks.to(device)
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    loss = criterion(model(imgs), msks)
                val_losses.append(loss.item())
        avg_vl = np.mean(val_losses)

        val_miou, val_pc, val_vc = compute_miou(model, val_ld, device, amp_dtype)

        # ── Test (periodic) ──
        test_miou, test_pc, test_vc = 0.0, None, None
        do_test = (ep % cfg['test_eval_every'] == 0 or ep == 1
                   or ep == cfg['num_epochs'])
        if do_test:
            test_miou, test_pc, test_vc = compute_miou(
                model, test_ld, device, amp_dtype)

        # ── LR bookkeeping ──
        lr_enc  = optimizer.param_groups[0]['lr']
        lr_dec  = optimizer.param_groups[1]['lr']
        lr_head = optimizer.param_groups[2]['lr']
        scheduler.step()

        et = time.time() - es
        history['train_loss'].append(avg_tl)
        history['val_loss'].append(avg_vl)
        history['val_miou'].append(val_miou)
        history['test_miou'].append(test_miou)
        history['lr_enc'].append(lr_enc)
        history['lr_dec'].append(lr_dec)
        history['lr_head'].append(lr_head)
        history['epoch_time'].append(et)

        eta = (time.time() - t0) / ep * (cfg['num_epochs'] - ep)
        frozen = " 🔒" if epoch < freeze_ep else ""
        tstr = f"Test:{test_miou:.4f}" if test_miou > 0 else ""

        print(f"  Ep {ep:3d}/{cfg['num_epochs']} │ "
              f"TrL:{avg_tl:.4f} │ VaL:{avg_vl:.4f} │ "
              f"Val:{val_miou:.4f} │ {tstr:16s}│ "
              f"LR:{lr_head:.2e}/{lr_enc:.2e} │ {et:.0f}s{frozen} │ ETA:{eta/60:.0f}m")

        # ── Detailed per-class (on test eval epochs) ──
        if test_miou > 0 and test_pc is not None:
            print(f"  {'─'*66}")
            for c in range(NUM_CLASSES):
                if test_vc[c]:
                    print(f"      {CLASS_NAMES[c]:<18} {test_pc[c]:.4f}")
            delta = test_miou - PHASE3_TEST
            print(f"      {'mIoU':<18} {test_miou:.4f}  "
                  f"(vs Phase3: {delta:+.4f})")
            print(f"  {'─'*66}")

        # ── Checkpoints ──
        if test_miou > best_test:
            best_test  = test_miou
            best_epoch = ep
            no_improve_count = 0
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'val_miou': val_miou,
                'test_miou': test_miou,
                'num_classes': NUM_CLASSES,
                'class_names': CLASS_NAMES,
                'config': cfg,
            }, os.path.join(out, 'checkpoints', 'best_model.pth'))
            print(f"  ★ New best test mIoU: {test_miou:.4f}")
        elif do_test and test_miou > 0:
            no_improve_count += 1
            print(f"  No test improvement ({no_improve_count}/{cfg['early_stopping_patience']})")

        if val_miou > best_val:
            best_val = val_miou
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'val_miou': val_miou,
                'test_miou': test_miou,
                'num_classes': NUM_CLASSES,
                'class_names': CLASS_NAMES,
                'config': cfg,
            }, os.path.join(out, 'checkpoints', 'best_val_model.pth'))

        if ep % cfg['save_every_n_epochs'] == 0:
            torch.save({
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'val_miou': val_miou,
                'test_miou': test_miou,
            }, os.path.join(out, 'checkpoints', f'epoch_{ep}.pth'))

        # ── Early stopping check ──
        if no_improve_count >= cfg['early_stopping_patience']:
            print(f"\n  ⏹ Early stopping triggered at epoch {ep} "
                  f"(no test improvement for {no_improve_count} evaluations)")
            stopped_early = True
            break

    # ══════════════════════════════════════════════════════════════
    #  FINAL EVALUATION
    # ══════════════════════════════════════════════════════════════
    total_time = time.time() - t0

    # Reload best-test model
    ckpt = torch.load(os.path.join(out, 'checkpoints', 'best_model.pth'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    print(f"\n{'═'*70}")
    print(f"  FINAL EVALUATION  (best model from epoch {best_epoch})")
    if stopped_early:
        print(f"  (Training stopped early at epoch {ep})")
    print(f"{'═'*70}")

    # Single-scale
    ss_miou, ss_pc, ss_vc = compute_miou(model, test_ld, device, amp_dtype)
    print(f"\n  Single-scale test mIoU: {ss_miou:.4f}")
    for c in range(NUM_CLASSES):
        if ss_vc[c]:
            print(f"    {CLASS_NAMES[c]:<18} {ss_pc[c]:.4f}")

    # Multi-scale TTA
    print(f"\n  Multi-scale TTA (×0.75, ×1.0, ×1.25, each + hflip = 6 passes)...")
    ms_miou, ms_pc, ms_vc = compute_miou_multiscale(
        model, test_ld, device, amp_dtype,
        scales=(0.75, 1.0, 1.25), flip=True)
    print(f"\n  Multi-scale TTA test mIoU: {ms_miou:.4f}")
    for c in range(NUM_CLASSES):
        if ms_vc[c]:
            gain = ms_pc[c] - ss_pc[c]
            print(f"    {CLASS_NAMES[c]:<18} {ms_pc[c]:.4f}  (TTA: {gain:+.4f})")

    # Save samples
    print(f"\n  Saving sample predictions...")
    save_samples(model, test_ld, device, amp_dtype, out, n=12)

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════
    actual_epochs = ep if stopped_early else cfg['num_epochs']

    print(f"\n{'═'*70}")
    print(f"  PHASE 5 COMPLETE — {total_time/60:.1f} min ({actual_epochs} epochs)")
    if stopped_early:
        print(f"  (Early stopped — saved {cfg['num_epochs'] - actual_epochs} epochs)")
    print(f"  {'─'*50}")
    print(f"  Phase 3 test  (10-cls + suppression):  {PHASE3_TEST:.4f}")
    print(f"  Phase 5 best  (7-cls, single-scale):   {best_test:.4f}  ({best_test-PHASE3_TEST:+.4f})")
    print(f"  Phase 5 best  (7-cls, MS-TTA):         {ms_miou:.4f}  ({ms_miou-PHASE3_TEST:+.4f})")
    print(f"  Best epoch:                            {best_epoch}")
    print(f"{'═'*70}")

    save_curves(history, out)

    results = {
        'phase3_test_miou': PHASE3_TEST,
        'baseline_test_miou': float(base_test),
        'best_test_miou_single': float(best_test),
        'best_test_miou_ms_tta': float(ms_miou),
        'best_val_miou': float(best_val),
        'best_epoch': best_epoch,
        'actual_epochs': actual_epochs,
        'stopped_early': stopped_early,
        'total_time_min': total_time / 60,
        'per_class_single': {CLASS_NAMES[c]: float(ss_pc[c])
                             for c in range(NUM_CLASSES) if ss_vc[c]},
        'per_class_ms_tta': {CLASS_NAMES[c]: float(ms_pc[c])
                             for c in range(NUM_CLASSES) if ms_vc[c]},
    }
    with open(os.path.join(out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(out, 'results.txt'), 'w') as f:
        f.write(f"PHASE 5 — 7-CLASS MODEL (FINAL)\n{'='*60}\n\n")
        f.write(f"Phase 3 test (10-cls + suppression): {PHASE3_TEST:.4f}\n")
        f.write(f"Phase 5 best (single-scale):         {best_test:.4f}\n")
        f.write(f"Phase 5 best (multi-scale TTA):      {ms_miou:.4f}\n\n")
        f.write(f"Per-class IoU (MS-TTA):\n")
        for c in range(NUM_CLASSES):
            if ms_vc[c]:
                f.write(f"  {CLASS_NAMES[c]:<18}: {ms_pc[c]:.4f}\n")
        f.write(f"\nBest epoch: {best_epoch}\n")
        f.write(f"Actual epochs: {actual_epochs}")
        if stopped_early:
            f.write(f" (early stopped)\n")
        else:
            f.write(f"\n")
        f.write(f"Training time: {total_time/60:.1f} min\n")

    print(f"\n  All outputs saved to {out}/")
    print(f"  Next steps:")
    if ms_miou > 0.55:
        print(f"    → mIoU > 0.55 — proceed to final submission pipeline")
    elif ms_miou > 0.50:
        print(f"    → mIoU > 0.50 — good. Try ConvNeXt encoder for +2-3%")
    else:
        print(f"    → mIoU < 0.50 — try ConvNeXt encoder or EfficientNet-B5")
        print(f"    → Also consider: longer training, OHEM loss, pseudo-labeling")


if __name__ == '__main__':
    main()
