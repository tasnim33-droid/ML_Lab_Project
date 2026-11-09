"""
BUSI Breast Ultrasound - INFERENCE-ONLY Gradio App
===================================================

This script does NOT train anything. It only:
  1) Loads pre-trained weights from checkpoint (.pth) files that the
     original notebook already saved to disk.
  2) Launches the Gradio app directly.

Architecture and weights are copied exactly from the notebook
(BUSI_v1_3.ipynb) - no retraining, no preprocessing, no dataset split is
performed here.

Required files (must already exist under PROJECT_ROOT/checkpoints/ -
these are created automatically the first time the full notebook is run):
  - joint_best.pth              (UNet + Classifier weights, from Stage 3)
  - ood_centroids.pth           (specialised OOD detector calibration)
  - generic_ood_detector.pth    (generic modality-check OOD detector)

If the two OOD checkpoints are missing, the script still runs - the OOD
checks are simply skipped for that stage (same try/except behaviour as
the original notebook's predict()).

Before running, set PROJECT_ROOT below to the same path you used in the
notebook on this machine.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision import transforms as T
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
import gradio as gr

# =============================================================================
# 1) CONFIG - set this to match the path used in the notebook on this machine
# =============================================================================

PROJECT_ROOT = r"./"   # <- folder that contains the checkpoints\ subfolder
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

IMG_SIZE = 256
MC_T = 20            # same as Part 7 of the notebook; lower (e.g. 10) for faster inference
CLASS_NAMES = ["normal", "benign", "malignant"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

JOINT_CKPT = os.path.join(CKPT_DIR, "joint_best.pth")
OOD_CENTROIDS_CKPT = os.path.join(CKPT_DIR, "ood_centroids.pth")
GENERIC_OOD_CKPT = os.path.join(CKPT_DIR, "generic_ood_detector.pth")

if not os.path.exists(JOINT_CKPT):
    raise FileNotFoundError(
        f"Checkpoint not found: {JOINT_CKPT}\n"
        f"This file is required to load the model. Run the full notebook "
        f"through Stage 3 training at least once to generate it."
    )
for _p in [OOD_CENTROIDS_CKPT, GENERIC_OOD_CKPT]:
    if not os.path.exists(_p):
        print(f"WARNING: {os.path.basename(_p)} not found - that OOD stage "
              f"will be skipped (the app still runs otherwise).")


# =============================================================================
# 2) PREPROCESSING - copied from notebook Part 2.1 / 8.1
# =============================================================================

def preprocess_image(image):
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    return image


def preprocess_for_inference(image_array):
    if image_array.ndim == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

    processed = preprocess_image(image_array.astype(np.uint8))

    img_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return img_tf(PILImage.fromarray(processed)).unsqueeze(0)


# =============================================================================
# 3) MODEL ARCHITECTURE - copied from notebook Part 4.1 / 4.2
#    (must stay identical to the notebook or the saved state_dict won't load)
# =============================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class ResDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.shortcut = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                          if in_ch != out_ch else nn.Identity())
        self.se = SEBlock(out_ch)

    def forward(self, x):
        return self.se(self.conv(x) + self.shortcut(x))


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResDoubleConv(in_ch, out_ch, dropout)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.block = ResDoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        b = base_ch
        self.enc1 = ResDoubleConv(in_ch, b)
        self.enc2 = Down(b, b * 2)
        self.enc3 = Down(b * 2, b * 4)
        self.enc4 = Down(b * 4, b * 8)
        self.bottleneck = Down(b * 8, b * 16, dropout=0.3)
        self.up1 = Up(b * 16, b * 8, b * 8)
        self.up2 = Up(b * 8, b * 4, b * 4)
        self.up3 = Up(b * 4, b * 2, b * 2)
        self.up4 = Up(b * 2, b, b)
        self.out_conv = nn.Conv2d(b, 1, kernel_size=1)
        if deep_supervision:
            self.ds1 = nn.Conv2d(b * 8, 1, 1)
            self.ds2 = nn.Conv2d(b * 4, 1, 1)
            self.ds3 = nn.Conv2d(b * 2, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3)
        b_ = self.bottleneck(e4)
        d1 = self.up1(b_, e4); d2 = self.up2(d1, e3); d3 = self.up3(d2, e2); d4 = self.up4(d3, e1)
        logit = self.out_conv(d4)
        if self.deep_supervision and self.training:
            H, W = x.shape[2], x.shape[3]
            aux1 = F.interpolate(self.ds1(d1), (H, W), mode="bilinear", align_corners=False)
            aux2 = F.interpolate(self.ds2(d2), (H, W), mode="bilinear", align_corners=False)
            aux3 = F.interpolate(self.ds3(d3), (H, W), mode="bilinear", align_corners=False)
            return logit, aux1, aux2, aux3
        return logit


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        )
        self.se = SEBlock(out_ch)
        self.skip = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch))
            if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x):
        return self.se(self.conv(x)) + self.skip(x)


class CNNBackbone(nn.Module):
    def __init__(self, in_ch=3, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = ResBlock(32, 64, stride=2, dropout=dropout)
        self.layer2 = ResBlock(64, 128, stride=2, dropout=dropout)
        self.layer3 = ResBlock(128, 256, stride=2, dropout=dropout)
        self.layer4 = ResBlock(256, 512, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
        return self.pool(x).flatten(1)


class TumorClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()
        self.global_branch = CNNBackbone(in_ch=3, dropout=0.1)
        self.masked_branch = CNNBackbone(in_ch=3, dropout=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, mask, return_embedding=False):
        masked_img = image * (mask + 0.1).clamp(0, 1)
        feat_global = self.global_branch(image)
        feat_masked = self.masked_branch(masked_img)
        embedding = torch.cat([feat_global, feat_masked], dim=1)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


# =============================================================================
# 4) MC-DROPOUT UTILITIES - copied from notebook Part 4.4
# =============================================================================

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


@torch.no_grad()
def mc_dropout_predict(unet_model, classifier_model, image_tensor, T=20, device="cuda"):
    unet_model.eval(); enable_dropout(unet_model)
    classifier_model.eval(); enable_dropout(classifier_model)

    image_tensor = image_tensor.to(device)
    seg_preds, cls_preds = [], []
    for t in range(T):
        inp = image_tensor if t == 0 else image_tensor + torch.randn_like(image_tensor) * 0.03
        seg_out = unet_model(inp)
        seg_out = seg_out[0] if isinstance(seg_out, tuple) else seg_out
        prob = torch.sigmoid(seg_out)
        logits = classifier_model(inp, (prob > 0.5).float())
        seg_preds.append(prob.cpu())
        cls_preds.append(torch.softmax(logits, dim=1).cpu())

    seg_stack = torch.stack(seg_preds)
    cls_stack = torch.stack(cls_preds)
    return seg_stack.mean(0), seg_stack.var(0), cls_stack.mean(0), cls_stack.std(0)


def mahalanobis_distance(embedding_reduced, centroids, cov_inv):
    dists = []
    for c in range(centroids.shape[0]):
        diff = embedding_reduced - centroids[c]
        d2 = torch.einsum('bi,ij,bj->b', diff, cov_inv, diff)
        dists.append(d2)
    dists = torch.stack(dists, dim=1)
    min_dist, nearest_class = dists.min(dim=1)
    return min_dist.sqrt(), nearest_class


# =============================================================================
# 5) MODEL LOAD - weight loading only, no training
# =============================================================================

print("Building UNet + TumorClassifier architecture...")
unet = UNet(in_ch=3, base_ch=64, deep_supervision=True).to(DEVICE)
classifier = TumorClassifier(num_classes=3, dropout=0.4).to(DEVICE)

print(f"Loading weights from {JOINT_CKPT} ...")
joint_ckpt = torch.load(JOINT_CKPT, map_location=DEVICE, weights_only=False)
unet.load_state_dict(joint_ckpt["unet"])
classifier.load_state_dict(joint_ckpt["classifier"])
unet.eval()
classifier.eval()
print(f"Weights loaded (val_acc at save time = {joint_ckpt.get('best_acc', 'N/A')})")

# -- Generic (modality) OOD detector - ImageNet-pretrained backbone ---------
# IMPORTANT: this must use the SAME ImageNet-pretrained weights the notebook
# used when it calibrated generic_ood_detector.pth (Part 6.4 / cell with
# "TWO-STAGE OOD DETECTION -- Stage 0"), otherwise the extracted features
# will not match the saved mean/covariance and the Stage-0 check will be
# meaningless. Needs internet access the first time to download the weights
# (cached afterwards by torchvision).
print("Loading generic ImageNet-pretrained ResNet50 backbone...")
generic_backbone = tvm.resnet50(weights="DEFAULT")
generic_backbone.fc = nn.Identity()
generic_backbone.eval().to(DEVICE)
for p in generic_backbone.parameters():
    p.requires_grad = False

_imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
_imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


@torch.no_grad()
def extract_generic_features(image_tensor):
    x = image_tensor.to(DEVICE)
    x = (x - _imagenet_mean) / _imagenet_std
    feat = generic_backbone(x)
    return feat.cpu().numpy()


# -- Trust-score normalisation ranges ----------------------------------------
# VALIDATION-set-calibrated values printed by the "[Point 22 fix]" cell in
# BUSI_v1_3.ipynb (Section 8.2) after training:
#   unc=[0.0043,0.3024]  dist=[3.3099,22.7894]
# If the model is retrained or recalibrated later, re-run that cell and
# update the two lines below with the new printed values.
TRUST_UNC_MIN, TRUST_UNC_MAX = 0.0043, 0.3024
TRUST_DIST_MIN, TRUST_DIST_MAX = 3.3099, 22.7894
print(f"Trust-score ranges: unc=[{TRUST_UNC_MIN},{TRUST_UNC_MAX}]  "
      f"dist=[{TRUST_DIST_MIN},{TRUST_DIST_MAX}] (validation-set calibrated)")


def composite_trust_score(confidence, unc_norm, dist_norm, w_conf=0.4, w_unc=0.3, w_dist=0.3):
    return 100.0 * (w_conf * confidence + w_unc * (1 - unc_norm) + w_dist * (1 - dist_norm))


# -- Simple OOD buffer (hard-cap only). The notebook's full drift-statistics
#    trigger needs a live val_loader; this simplified version is for
#    inference-only use and just tracks a capped in-memory buffer. ----------
HARD_CAP_SAMPLES = 1000


class OODBuffer:
    def __init__(self, hard_cap=HARD_CAP_SAMPLES):
        self.hard_cap = hard_cap
        self.images, self.masks, self.distances = [], [], []

    def add(self, image_tensor, mask_tensor, distance):
        self.images.append(image_tensor.cpu())
        self.masks.append(mask_tensor.cpu())
        self.distances.append(distance)

    def __len__(self):
        return len(self.images)

    def ready_for_retrain(self):
        return len(self) >= self.hard_cap


ood_buffer = OODBuffer()


# =============================================================================
# 6) PREDICT - copied from notebook Part 8.2 (logic unchanged)
# =============================================================================

def predict(input_image):
    if input_image is None:
        empty = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return empty, empty, empty, empty, "No image uploaded."

    orig = input_image.astype(np.uint8)
    if orig.ndim == 2:
        orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    elif orig.shape[2] == 4:
        orig = cv2.cvtColor(orig, cv2.COLOR_RGBA2RGB)

    tensor = preprocess_for_inference(orig)
    orig_show = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))

    # Stage 0 - generic modality check
    generic_dist, generic_thresh, is_generic_ood_flag = None, None, False
    try:
        generic_ckpt = torch.load(GENERIC_OOD_CKPT, map_location="cpu", weights_only=False)
        gen_mean = generic_ckpt["mean"].numpy()
        gen_cov_inv = generic_ckpt["cov_inv"].numpy()
        gen_thresh = generic_ckpt["threshold"]
        gen_pca = generic_ckpt["pca"]

        imgs_01 = (tensor.to(DEVICE) * _imagenet_std + _imagenet_mean).clamp(0, 1)
        gen_feat = extract_generic_features(imgs_01)
        gen_feat_reduced = gen_pca.transform(gen_feat)[0]
        diff = gen_feat_reduced - gen_mean
        generic_dist = float(np.sqrt(diff @ gen_cov_inv @ diff))
        generic_thresh = float(gen_thresh)
        is_generic_ood_flag = generic_dist > generic_thresh
    except Exception:
        pass

    # Stage 1 - specialised scanner/domain check
    ood_dist, ood_thresh_g, is_ood_flag = None, None, False
    pred_mask_ood = None
    try:
        with torch.no_grad():
            seg_out = unet(tensor.to(DEVICE))
            seg_out = seg_out[0] if isinstance(seg_out, tuple) else seg_out
            # IMPORTANT: use the soft/continuous probability mask here, NOT a
            # binary threshold -- this must match compute_class_centroids()
            # and calibrate_ood_threshold() in the notebook exactly, or the
            # embeddings (and therefore OOD distances) will not line up with
            # the saved centroids/covariance.
            pred_mask_ood = torch.sigmoid(seg_out)
            _, emb = classifier(tensor.to(DEVICE), pred_mask_ood, return_embedding=True)
        ood_ckpt = torch.load(OOD_CENTROIDS_CKPT, map_location="cpu", weights_only=False)
        centroids_g = ood_ckpt["centroids"]
        cov_inv_g = ood_ckpt["cov_inv"]
        ood_thresh_g = ood_ckpt["threshold"]
        pca_g = ood_ckpt["pca"]

        emb_reduced = torch.from_numpy(pca_g.transform(emb.cpu().numpy())).float()
        dists = []
        for c in range(centroids_g.shape[0]):
            diff = emb_reduced - centroids_g[c]
            d2 = torch.einsum('bi,ij,bj->b', diff, cov_inv_g, diff)
            dists.append(d2)
        ood_dist = torch.stack(dists, dim=1).min(dim=1).values.sqrt().item()
        is_ood_flag = ood_dist > ood_thresh_g
    except Exception:
        pass

    combined_flag = bool(is_generic_ood_flag or is_ood_flag)
    if is_generic_ood_flag and not is_ood_flag:
        flag_reason = "modality/content mismatch (Stage 0)"
    elif is_ood_flag and not is_generic_ood_flag:
        flag_reason = "scanner/domain shift (Stage 1)"
    elif is_generic_ood_flag and is_ood_flag:
        flag_reason = "both modality mismatch AND domain shift"
    else:
        flag_reason = None

    # MC-Dropout inference
    mean_seg, unc_map, mean_cls, cls_std = mc_dropout_predict(unet, classifier, tensor, T=MC_T, device=DEVICE)
    pred_cls = mean_cls.argmax(dim=1).item()
    confidence = mean_cls[0, pred_cls].item()
    probs = mean_cls[0].numpy()
    stds = cls_std[0].numpy()

    unc_score = float(cls_std[0].mean().item())
    unc_weight = 1.0 + min(unc_score / 0.3, 1.0) * 2.0
    difficulty = "EASY" if unc_weight < 1.5 else "MEDIUM" if unc_weight < 2.2 else "HARD"

    unc_tilde = float(np.clip((unc_score - TRUST_UNC_MIN) / max(TRUST_UNC_MAX - TRUST_UNC_MIN, 1e-8), 0, 1))
    dist_tilde = float(np.clip((ood_dist - TRUST_DIST_MIN) / max(TRUST_DIST_MAX - TRUST_DIST_MIN, 1e-8), 0, 1)) \
        if ood_dist is not None else 0.5
    trust = composite_trust_score(confidence, unc_tilde, dist_tilde)
    if is_generic_ood_flag:
        trust = min(trust, 20.0)
    trust_pct = int(np.clip(trust, 0, 100))

    out_original = orig_show.copy()
    mask_np = mean_seg[0, 0].numpy()
    out_mask = cv2.cvtColor((mask_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    unc_np = unc_map[0, 0].numpy()
    unc_norm = (unc_np - unc_np.min()) / (unc_np.max() - unc_np.min() + 1e-8)
    out_unc = (cm.hot(unc_norm)[:, :, :3] * 255).astype(np.uint8)

    overlay = orig_show.astype(float) / 255.0
    mask_bin = (mask_np > 0.5).astype(float)
    unc_bin = (unc_norm > 0.6).astype(float)
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.45 * mask_bin, 0, 1)
    overlay[..., 1] = np.clip(overlay[..., 1] + 0.35 * unc_bin, 0, 1)
    out_overlay = (overlay * 255).astype(np.uint8)

    SEP = "-" * 35
    icons = {"normal": "[N]", "benign": "[B]", "malignant": "[M]"}
    conf_level = "HIGH" if confidence > 0.80 else "MEDIUM" if confidence > 0.60 else "LOW"
    unc_level = "HIGH" if unc_np.mean() > 0.02 else "MEDIUM" if unc_np.mean() > 0.005 else "LOW"

    if combined_flag:
        try:
            if pred_mask_ood is None:
                raise ValueError("segmentation mask unavailable")
            ood_buffer.add(tensor.squeeze(0), pred_mask_ood.squeeze(0).cpu(),
                            ood_dist if ood_dist is not None else -1.0)
            buf_line = f"OOD Buffer  : {len(ood_buffer)}/{HARD_CAP_SAMPLES}"
            if ood_buffer.ready_for_retrain():
                buf_line += "  RETRAIN TRIGGER REACHED"
        except Exception as e:
            buf_line = f"OOD Buffer  : N/A ({e})"

        gen_line = (f"Stage 0 (modality check) : dist={generic_dist:.2f} / thresh={generic_thresh:.2f}  "
                    f"{'FLAGGED' if is_generic_ood_flag else 'ok'}"
                    if generic_dist is not None else "Stage 0 (modality check) : unavailable")
        spec_line = (f"Stage 1 (scanner check)   : dist={ood_dist:.2f} / thresh={ood_thresh_g:.2f}  "
                     f"{'FLAGGED' if is_ood_flag else 'ok'}"
                     if ood_dist is not None else "Stage 1 (scanner check)   : unavailable")

        text = (
            f"{SEP}\nDIAGNOSIS: FLAGGED ({flag_reason})\nTrustworthiness: {trust_pct}%\n{SEP}\n"
            f"{gen_line}\n{spec_line}\nPlease consult a clinician.\nFLAGGED - added to OOD buffer\n"
            f"{buf_line}\n{SEP}\n[Debug info - raw internal output, NOT reliable]\n"
            f"  Raw Diagnosis   : {CLASS_NAMES[pred_cls].upper()}\n"
            f"  Raw Confidence  : {confidence*100:.1f}% [{conf_level}]\n"
            f"  Raw Uncertainty : {unc_np.mean():.5f} [{unc_level}]\n"
            f"  Raw Difficulty  : {difficulty} (weight: {unc_weight:.2f}x)\n"
            f"  Raw Probabilities - Normal: {probs[0]*100:.1f}%  Benign: {probs[1]*100:.1f}%  "
            f"Malignant: {probs[2]*100:.1f}%\n{SEP}\nFor research use only. Consult a clinician."
        )
    else:
        icon = icons[CLASS_NAMES[pred_cls]]
        if ood_dist is not None:
            ood_line = f"IN-DISTRIBUTION (Stage 0 dist: {generic_dist:.2f}, Stage 1 dist: {ood_dist:.2f})"
            flag_line = "NOT FLAGGED (both stages passed)"
            try:
                buf_line = f"OOD Buffer  : {len(ood_buffer)}/{HARD_CAP_SAMPLES}"
            except Exception:
                buf_line = ""
        else:
            ood_line = "OOD check unavailable (centroids not found)"
            flag_line, buf_line = "", ""

        text = (
            f"{SEP}\n{icon} DIAGNOSIS: {CLASS_NAMES[pred_cls].upper()}\n"
            f"Confidence      : {confidence*100:.1f}% [{conf_level}]\n"
            f"Uncertainty     : {unc_np.mean():.5f} [{unc_level}]\n{SEP}\n"
            f"{ood_line}\nTrustworthiness: {trust_pct}%\n"
            f"Sample Difficulty: {difficulty} (weight: {unc_weight:.2f}x)\n{flag_line}\n{SEP}\n"
            f"Class Probabilities (mean +/- std):\n"
            f"  Normal    : {probs[0]*100:.1f}% +/- {stds[0]*100:.1f}%\n"
            f"  Benign    : {probs[1]*100:.1f}% +/- {stds[1]*100:.1f}%\n"
            f"  Malignant : {probs[2]*100:.1f}% +/- {stds[2]*100:.1f}%\n{buf_line}\n{SEP}\n"
            f"For research use only. Consult a clinician."
        )

    return out_original, out_mask, out_unc, out_overlay, text


# =============================================================================
# 7) GRADIO UI - copied from notebook Part 8.3
# =============================================================================

with gr.Blocks(title="BUSI Tumour Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Breast Ultrasound Tumour Detection
        ### UNet Segmentation + Dual-Branch Classifier + MC Dropout Uncertainty
        **Inference-only build - weights loaded directly, no training.**
        Upload a breast ultrasound image then click Analyse.
        """
    )
    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Ultrasound Image", type="numpy", height=280)
            analyse_btn = gr.Button("Analyse", variant="primary", size="lg")
        with gr.Column(scale=1):
            cls_output = gr.Textbox(label="Diagnosis", lines=18)
    gr.Markdown("---")
    gr.Markdown("### Visual Outputs")
    with gr.Row():
        out1 = gr.Image(label="Original", height=240)
        out2 = gr.Image(label="Tumour Mask", height=240)
        out3 = gr.Image(label="Uncertainty Map", height=240)
        out4 = gr.Image(label="Overlay", height=240)

    analyse_btn.click(fn=predict, inputs=[input_img], outputs=[out1, out2, out3, out4, cls_output])

    gr.Markdown(
        """
        ---
        Red = tumour region | Yellow = uncertain boundary

        For research use only. Not a clinical diagnostic tool.
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
