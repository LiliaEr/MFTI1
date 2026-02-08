# pipeline_core.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


Metric = Literal["cosine", "l2"]
EmbedMode = Literal["arcface", "ce"]


# Types


@dataclass
class FaceDet:
    bbox_xyxy: np.ndarray  # float32 [x1,y1,x2,y2] in original image coords
    score: float


@dataclass
class FaceResult:
    bbox_xyxy: np.ndarray               # (4,)
    score: float
    landmarks_xy: np.ndarray            # (5,2) in original image coords
    aligned_112: Optional[np.ndarray]   # (112,112,3) RGB uint8
    embedding: np.ndarray               # (D,) float32 (L2-normalized)


@dataclass
class PipelineConfig:
    # Paths
    hourglass_ckpt: str = "stacked_hourglass_best.pt"
    ce_ckpt: Optional[str] = None
    arcface_ckpt: Optional[str] = None

    # Detector (RetinaFace via insightface FaceAnalysis)
    det_conf_th: float = 0.80
    bbox_padding: float = 0.15      # bbox expand fraction (helps landmarks)

    # Hourglass I/O
    lm_in_size: int = 128
    lm_num_points: int = 5
    softargmax_beta: float = 100.0

    # Alignment
    aligned_size: int = 112

    # Embedding preprocess: ImageNet normalize
    emb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    emb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Runtime
    device: str = "cpu"  # по умолчанию CPU
    fp16: bool = False

    # Debug output (save intermediate images if set)
    debug_dir: Optional[str] = None
    debug_max_faces: int = 10       # save only first N faces to avoid spam


# Small image helpers

def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)
    x2 = np.clip(x2, 0, w - 1)
    y2 = np.clip(y2, 0, h - 1)
    x2 = max(x2, x1 + 1.0)
    y2 = max(y2, y1 + 1.0)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def pad_xyxy(xyxy: np.ndarray, pad_frac: float, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = xyxy.astype(np.float32)
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * pad_frac
    pad_y = bh * pad_frac
    out = np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32)
    return clip_xyxy(out, w, h)


def crop_rgb(image_rgb: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    return image_rgb[y1:y2, x1:x2].copy()


def ensure_dir(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_rgb(path: Path, img_rgb_u8: np.ndarray) -> None:
    cv2.imwrite(str(path), rgb_to_bgr(img_rgb_u8))


def draw_bbox(img_rgb: np.ndarray, bbox_xyxy: np.ndarray, score: float, color=(0, 255, 0)) -> np.ndarray:
    out = img_rgb.copy()
    x1, y1, x2, y2 = bbox_xyxy.astype(int)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out


def draw_points(img_rgb: np.ndarray, pts_xy: np.ndarray, color=(255, 0, 0)) -> np.ndarray:
    out = img_rgb.copy()
    for (x, y) in pts_xy.astype(int):
        cv2.circle(out, (int(x), int(y)), 3, color, -1)
    return out



# RetinaFace detector via insightface FaceAnalysis


class RetinaFaceDetector:
    def __init__(self, det_size=(640, 640), conf_th: float = 0.8, providers=None):
        from insightface.app import FaceAnalysis

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.conf_th = float(conf_th)
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)

        # ctx_id: -1 => CPU
        ctx_id = -1
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    @torch.inference_mode()
    def detect_faces(self, image_rgb: np.ndarray, pad_frac: float = 0.15) -> List[FaceDet]:
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        img_bgr = rgb_to_bgr(image_rgb)
        faces = self.app.get(img_bgr)

        h, w = image_rgb.shape[:2]
        out: List[FaceDet] = []
        for f in faces:
            score = float(getattr(f, "det_score", 0.0))
            if score < self.conf_th:
                continue
            bbox = np.array(f.bbox, dtype=np.float32)  # xyxy
            bbox = clip_xyxy(bbox, w=w, h=h)
            bbox = pad_xyxy(bbox, pad_frac=pad_frac, w=w, h=h)
            out.append(FaceDet(bbox_xyxy=bbox, score=score))

        out.sort(key=lambda x: x.score, reverse=True)
        return out



# Stacked Hourglass
class Residual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch // 2)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch // 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch // 2)
        self.conv3 = nn.Conv2d(out_ch // 2, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.skip is not None:
            identity = self.skip(identity)
        out = F.relu(out + identity, inplace=True)
        return out


class Hourglass(nn.Module):
    def __init__(self, depth: int, channels: int):
        super().__init__()
        self.up1 = Residual(channels, channels)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = Residual(channels, channels)

        if depth > 1:
            self.low2 = Hourglass(depth - 1, channels)
        else:
            self.low2 = Residual(channels, channels)

        self.low3 = Residual(channels, channels)

    def forward(self, x):
        up1 = self.up1(x)
        low = self.pool(x)
        low = self.low1(low)
        low = self.low2(low)
        low = self.low3(low)
        up2 = F.interpolate(low, scale_factor=2, mode="nearest")
        return up1 + up2


class StackedHourglass(nn.Module):
    def __init__(self, num_stacks: int = 2, num_keypoints: int = 5, hg_depth: int = 4, channels: int = 256):
        super().__init__()
        self.num_stacks = num_stacks
        self.num_keypoints = num_keypoints

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, channels),
        )

        self.hgs = nn.ModuleList([Hourglass(hg_depth, channels) for _ in range(num_stacks)])
        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(channels, channels),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ) for _ in range(num_stacks)
        ])
        self.heads = nn.ModuleList([nn.Conv2d(channels, num_keypoints, 1) for _ in range(num_stacks)])

        self.merge_features = nn.ModuleList([nn.Conv2d(channels, channels, 1, bias=False) for _ in range(num_stacks - 1)])
        self.merge_preds = nn.ModuleList([nn.Conv2d(num_keypoints, channels, 1, bias=False) for _ in range(num_stacks - 1)])

    def forward(self, x):
        x = self.stem(x)
        preds: List[torch.Tensor] = []
        for i in range(self.num_stacks):
            hg = self.hgs[i](x)
            feat = self.features[i](hg)
            pred = self.heads[i](feat)
            preds.append(pred)
            if i < self.num_stacks - 1:
                x = x + self.merge_features[i](feat) + self.merge_preds[i](pred)
        return preds


def load_hourglass(cfg: PipelineConfig) -> nn.Module:
    model = StackedHourglass(num_stacks=2, num_keypoints=cfg.lm_num_points, hg_depth=4, channels=256)
    ckpt = torch.load(cfg.hourglass_ckpt, map_location="cpu")
    sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    return model



# Soft-argmax


@torch.no_grad()
def heatmaps_to_points_softargmax(hm: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
    B, K, H, W = hm.shape
    hm_flat = hm.view(B, K, -1)
    prob = torch.softmax(hm_flat * beta, dim=-1)

    ys = torch.arange(H, device=hm.device, dtype=hm.dtype)
    xs = torch.arange(W, device=hm.device, dtype=hm.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)  # (HW,2)

    pts = torch.matmul(prob, coords)  # (B,K,2)
    return pts


# Alignment (ArcFace template + Umeyama)

ARC_FACE_112_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> np.ndarray:
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    N = src.shape[0]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_d = src - src_mean
    dst_d = dst - dst_mean

    cov = (dst_d.T @ src_d) / N
    U, S, Vt = np.linalg.svd(cov)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        S[-1] *= -1
        R = U @ Vt

    if estimate_scale:
        var_src = (src_d ** 2).sum() / N
        scale = S.sum() / var_src
    else:
        scale = 1.0

    t = dst_mean - scale * (R @ src_mean)

    M = np.zeros((2, 3), dtype=np.float32)
    M[:, :2] = (scale * R).astype(np.float32)
    M[:, 2] = t.astype(np.float32)
    return M


# Embedders


class ResNet34FaceArc(nn.Module):
    def __init__(self, emb_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        m = models.resnet34(weights=weights)

        old_conv = m.conv1
        new_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight[:, :, 2:5, 2:5])
        m.conv1 = new_conv
        m.maxpool = nn.Identity()

        in_features = m.fc.in_features
        m.fc = nn.Identity()

        self.backbone = m
        self.emb = nn.Sequential(
            nn.Linear(in_features, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.emb(feat)
        return emb


class ResNet34FaceCEIdeal(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 512, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        m = models.resnet34(weights=weights)

        old_conv = m.conv1
        new_conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if pretrained:
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight[:, :, 2:5, 2:5])
        m.conv1 = new_conv
        m.maxpool = nn.Identity()

        in_features = m.fc.in_features
        m.fc = nn.Identity()

        self.backbone = m
        self.emb = nn.Sequential(
            nn.Linear(in_features, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.emb(feat)
        logits = self.classifier(emb)
        return logits


class CEEmbedderWrapper(nn.Module):
    def __init__(self, ce_model: ResNet34FaceCEIdeal):
        super().__init__()
        self.backbone = ce_model.backbone
        self.emb = ce_model.emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb = self.emb(feat)
        return emb


def load_embedder(cfg: PipelineConfig, mode: EmbedMode) -> nn.Module:
    if mode == "arcface":
        if not cfg.arcface_ckpt:
            raise ValueError("cfg.arcface_ckpt is None, but embed_mode='arcface'")
        model = ResNet34FaceArc(emb_dim=512, pretrained=False)
        ckpt = torch.load(cfg.arcface_ckpt, map_location="cpu")
        sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(sd, strict=True)
        return model

    if not cfg.ce_ckpt:
        raise ValueError("cfg.ce_ckpt is None, but embed_mode='ce'")
    ckpt = torch.load(cfg.ce_ckpt, map_location="cpu")
    sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    if isinstance(ckpt, dict) and "num_classes" in ckpt:
        num_classes = int(ckpt["num_classes"])
    else:
        num_classes = 1000

    ce_model = ResNet34FaceCEIdeal(num_classes=num_classes, emb_dim=512, pretrained=False)
    ce_model.load_state_dict(sd, strict=True)
    return CEEmbedderWrapper(ce_model)



# Main pipeline


class FacePipeline:
    def __init__(self, cfg: PipelineConfig, embed_mode: EmbedMode = "arcface"):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.embed_mode = embed_mode

        self.debug_dir = ensure_dir(cfg.debug_dir)
        self.use_amp = bool(cfg.fp16 and self.device.type == "cuda")

        # CPU-only providers
        providers = ["CPUExecutionProvider"]
        self.detector = RetinaFaceDetector(conf_th=cfg.det_conf_th, det_size=(640, 640), providers=providers)

        self.lm_model = load_hourglass(cfg).to(self.device).eval()
        self.embedder = load_embedder(cfg, mode=embed_mode).to(self.device).eval()

    # ----------------------
    # Input conversion
    # ----------------------
    def _to_rgb(self, image: Any) -> np.ndarray:
        if isinstance(image, Image.Image):
            return pil_to_rgb_np(image)

        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("Expected HxWx3 image.")
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            return image  # assume RGB

        if isinstance(image, (str, Path)):
            bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Cannot read image from path: {image}")
            return bgr_to_rgb(bgr)

        raise TypeError(f"Unsupported image type: {type(image)}")

    # ----------------------
    # Detector
    # ----------------------
    @torch.inference_mode()
    def detect(self, image_rgb: np.ndarray) -> List[FaceDet]:
        return self.detector.detect_faces(image_rgb, pad_frac=self.cfg.bbox_padding)

    # ----------------------
    # Landmarks
    # ----------------------
    def _preprocess_for_landmarks(self, crop_rgb_u8: np.ndarray) -> torch.Tensor:
        img = cv2.resize(crop_rgb_u8, (self.cfg.lm_in_size, self.cfg.lm_in_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return x.unsqueeze(0)

    @torch.inference_mode()
    def predict_landmarks(self, image_rgb: np.ndarray, det: FaceDet) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        bbox = det.bbox_xyxy
        crop = crop_rgb(image_rgb, bbox)
        crop_h, crop_w = crop.shape[:2]

        x = self._preprocess_for_landmarks(crop).to(self.device)
        preds_list = self.lm_model(x)
        hm_small = preds_list[-1]
        hm = F.interpolate(hm_small, size=(self.cfg.lm_in_size, self.cfg.lm_in_size), mode="bilinear", align_corners=False)
        pts_128 = heatmaps_to_points_softargmax(hm, beta=self.cfg.softargmax_beta)[0]

        scale_x = crop_w / float(self.cfg.lm_in_size)
        scale_y = crop_h / float(self.cfg.lm_in_size)
        pts_crop = pts_128.clone()
        pts_crop[:, 0] *= scale_x
        pts_crop[:, 1] *= scale_y

        x1, y1, _, _ = bbox
        pts_orig = pts_crop.cpu().numpy().astype(np.float32)
        pts_orig[:, 0] += float(x1)
        pts_orig[:, 1] += float(y1)

        pts_orig[:, 0] = np.clip(pts_orig[:, 0], 0, w - 1)
        pts_orig[:, 1] = np.clip(pts_orig[:, 1], 0, h - 1)
        return pts_orig

    # ----------------------
    # Alignment
    # ----------------------
    def align(self, image_rgb: np.ndarray, landmarks_xy: np.ndarray) -> np.ndarray:
        dst = ARC_FACE_112_TEMPLATE.copy()
        if self.cfg.aligned_size != 112:
            s = self.cfg.aligned_size / 112.0
            dst *= s

        M = umeyama_similarity(landmarks_xy.astype(np.float32), dst.astype(np.float32), estimate_scale=True)
        aligned = cv2.warpAffine(
            image_rgb,
            M,
            (self.cfg.aligned_size, self.cfg.aligned_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        return aligned.astype(np.uint8)

    # ----------------------
    # Embedding
    # ----------------------
    def _preprocess_for_embedder(self, aligned_rgb_u8: np.ndarray) -> torch.Tensor:
        img = aligned_rgb_u8.astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1)
        mean = torch.tensor(self.cfg.emb_mean, dtype=x.dtype).view(3, 1, 1)
        std = torch.tensor(self.cfg.emb_std, dtype=x.dtype).view(3, 1, 1)
        x = (x - mean) / std
        return x.unsqueeze(0)

    @torch.inference_mode()
    def embed(self, aligned_rgb_u8: np.ndarray) -> np.ndarray:
        x = self._preprocess_for_embedder(aligned_rgb_u8).to(self.device)
        emb = self.embedder(x)
        emb = emb.float()
        emb = F.normalize(emb, p=2, dim=1)
        return emb[0].detach().cpu().numpy().astype(np.float32)

    # ----------------------
    # Full inference
    # ----------------------
    @torch.inference_mode()
    def infer(self, image: Any) -> List[FaceResult]:
        image_rgb = self._to_rgb(image)

        if self.debug_dir is not None:
            save_rgb(self.debug_dir / "00_input.jpg", image_rgb)

        dets = self.detect(image_rgb)

        if self.debug_dir is not None:
            det_vis = image_rgb.copy()
            for d in dets:
                det_vis = draw_bbox(det_vis, d.bbox_xyxy, d.score)
            save_rgb(self.debug_dir / "01_dets.jpg", det_vis)

        results: List[FaceResult] = []
        for i, det in enumerate(dets[: self.cfg.debug_max_faces]):
            crop = crop_rgb(image_rgb, det.bbox_xyxy)
            if self.debug_dir is not None:
                save_rgb(self.debug_dir / f"02_face_{i:02d}_crop.jpg", crop)

            lm = self.predict_landmarks(image_rgb, det)
            if self.debug_dir is not None:
                lm_vis = draw_points(draw_bbox(image_rgb, det.bbox_xyxy, det.score), lm)
                save_rgb(self.debug_dir / f"03_face_{i:02d}_landmarks.jpg", lm_vis)

            aligned = self.align(image_rgb, lm)
            if self.debug_dir is not None:
                save_rgb(self.debug_dir / f"04_face_{i:02d}_aligned.jpg", aligned)

            emb = self.embed(aligned)

            results.append(
                FaceResult(
                    bbox_xyxy=det.bbox_xyxy,
                    score=det.score,
                    landmarks_xy=lm,
                    aligned_112=aligned,
                    embedding=emb,
                )
            )

        return results

    # ----------------------
    # Distance utilities
    # ----------------------
    @staticmethod
    def distance(emb1: np.ndarray, emb2: np.ndarray, metric: Metric = "cosine") -> float:
        emb1 = emb1.astype(np.float32)
        emb2 = emb2.astype(np.float32)
        if metric == "cosine":
            sim = float(np.dot(emb1, emb2))  # L2-normalized => cosine similarity
            return 1.0 - sim
        if metric == "l2":
            return float(np.linalg.norm(emb1 - emb2))
        raise ValueError(f"Unknown metric: {metric}")
