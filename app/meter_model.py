"""Two-stage meter reader trained on the Kaggle Water Meters dataset (training/train_meter_models.py).

Stage 1, meter_seg.onnx: marks the counter window pixel by pixel, which gives a rotated box. A second, zoomed pass
    sharpens the box for small or distant counters.
Stage 2, meter_rec.onnx: reads the straightened counter (one digit class per column, CTC decoding) and says whether
    the crop is upright or upside down, so sideways and upside-down photos work.

Only numpy, OpenCV and onnxruntime are needed at runtime. The geometry helpers are shared with the training script
so a box is straightened exactly the same way in training and in use.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SEG_SIZE = 256
REC_H, REC_W = 40, 192
REC_PAD_X, REC_PAD_Y = 0.04, 0.15
BOX_VARIANTS = ((1.0, 1.0), (1.06, 1.2), (0.96, 0.9))  # width, height scales read and averaged (tuned on validation)
ZOOM_LONG, ZOOM_SHORT = 1.7, 4.0  # zoomed pass: square side = max(long side * 1.7, short side * 4)
SEG_FILE, REC_FILE, META_FILE = "meter_seg.onnx", "meter_rec.onnx", "meter_model.json"


# ---------------------------------------------------------------- geometry

def normalize_rect(rect):
    """cv2 rotated rect with the long side as width and the angle in (-90, 90]."""
    (cx, cy), (w, h), angle = rect
    if w < h:
        w, h, angle = h, w, angle + 90
    while angle > 90:
        angle -= 180
    while angle <= -90:
        angle += 180
    return (float(cx), float(cy)), (float(w), float(h)), float(angle)


def rectify(image, rect, out_w=REC_W, out_h=REC_H, pad_x=REC_PAD_X, pad_y=REC_PAD_Y, flip=False):
    """Cuts the rotated box out of the image as a horizontal strip (padded, stretched to out_w x out_h)."""
    (cx, cy), (w, h), angle = rect
    if flip:
        angle += 180
    a = np.deg2rad(angle)
    ux, uy = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    half_w, half_h = w * (1 + 2 * pad_x) / 2, h * (1 + 2 * pad_y) / 2
    c = np.array([cx, cy])
    tl, tr, bl = c - ux * half_w - uy * half_h, c + ux * half_w - uy * half_h, c - ux * half_w + uy * half_h
    src = np.array([tl, tr, bl])
    # Work on a small region around the box, area-downscaled first when the strip is much smaller than the photo.
    corners = np.vstack([src, tr + bl - tl])
    x0, y0 = np.floor(corners.min(axis=0)).astype(int) - 2
    x1, y1 = np.ceil(corners.max(axis=0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, image.shape[1]), min(y1, image.shape[0])
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.zeros((out_h, out_w) + image.shape[2:], dtype=image.dtype)
    region, src = image[y0:y1, x0:x1], src - [x0, y0]
    scale = min(out_w / (2 * half_w), out_h / (2 * half_h)) * 1.25
    if scale < 1:
        region = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        src = src * scale
    dst = np.float32([[0, 0], [out_w, 0], [0, out_h]])
    matrix = cv2.getAffineTransform(src.astype(np.float32), dst)
    return cv2.warpAffine(region, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def view_matrix(center, size, angle=0.0, out=SEG_SIZE):
    """Affine matrix mapping a (rotated) source window of `size` around `center` onto an out x out square."""
    sx, sy = out / size[0], out / size[1]
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    linear = np.diag([sx, sy]) @ rot
    offset = np.array([out / 2, out / 2]) - linear @ np.asarray(center, dtype=np.float64)
    return np.hstack([linear, offset[:, None]]).astype(np.float32)


def render_view(image, matrix, out=SEG_SIZE, border=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_LINEAR):
    """warpAffine with an area-downscale first, so shrinking a 1333px photo to 256px does not alias."""
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2])))) * 1.25
    if scale < 1 and interpolation != cv2.INTER_NEAREST:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        matrix = np.hstack([matrix[:, :2] / scale, matrix[:, 2:]]).astype(np.float32)
    return cv2.warpAffine(image, matrix, (out, out), flags=interpolation, borderMode=border)


def mask_rect(prob, matrix, threshold=0.5):
    """Largest blob of the predicted mask as a rotated box in original image coordinates, with its mean score."""
    binary = (prob > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count < 2:
        return None, 0.0
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < 12:
        return None, 0.0
    ys, xs = np.nonzero(labels == best)
    points = np.column_stack([xs + 0.5, ys + 0.5]).astype(np.float32)
    inverse = cv2.invertAffineTransform(matrix)
    original = points @ inverse[:, :2].T + inverse[:, 2]
    return normalize_rect(cv2.minAreaRect(original.astype(np.float32))), float(prob[labels == best].mean())


def ctc_best_digits(log_probs, length):
    """Most likely CTC path that spells exactly `length` digits. log_probs: (T, 11), class 0 = blank.

    Viterbi over states (digits emitted so far, last symbol); a repeated digit needs a blank in between.
    Returns the digits and the product of each digit's best frame probability.
    """
    log_probs = np.asarray(log_probs, dtype=np.float64)
    steps, classes = log_probs.shape
    neg, rows = -1e9, np.arange(length + 1)
    digits_idx = np.arange(1, classes)
    score = np.full((length + 1, classes), neg)
    score[0, 0] = log_probs[0, 0]
    if length:
        score[1, 1:] = log_probs[0, 1:]
    back_k = np.zeros((steps, length + 1, classes), dtype=np.int16)
    back_c = np.zeros((steps, length + 1, classes), dtype=np.int16)
    for t in range(1, steps):
        lp = log_probs[t]
        new = np.full_like(score, neg)
        bk = np.repeat(rows[:, None], classes, axis=1).astype(np.int16)
        bc = np.zeros((length + 1, classes), dtype=np.int16)
        best_any = score.argmax(axis=1)                      # blank: keep the count, any previous symbol
        new[:, 0], bc[:, 0] = score[rows, best_any] + lp[0], best_any
        new[:, 1:], bc[:, 1:] = score[:, 1:] + lp[1:], digits_idx  # same digit continues (collapses)
        if length:
            prev = score[:-1]                                # emit a new digit after a different symbol
            order = np.argsort(prev, axis=1)
            top1, top2 = order[:, -1:], order[:, -2:-1]
            source = np.where(top1 == digits_idx[None], top2, top1)
            emit = np.take_along_axis(prev, source, axis=1) + lp[1:]
            better = emit > new[1:, 1:]
            new[1:, 1:] = np.where(better, emit, new[1:, 1:])
            bk[1:, 1:] = np.where(better, rows[:-1, None], bk[1:, 1:])
            bc[1:, 1:] = np.where(better, source, bc[1:, 1:])
        score, back_k[t], back_c[t] = new, bk, bc
    last = int(np.argmax(score[length]))
    if score[length, last] <= neg / 2:
        return "", 0.0
    path, k, c = [], length, last
    for t in range(steps - 1, -1, -1):
        path.append(c)
        if t:
            k, c = int(back_k[t, k, c]), int(back_c[t, k, c])
    path.reverse()
    digits, probs, previous = [], [], 0
    for t, c in enumerate(path):
        if c and c != previous:
            digits.append(str(c - 1)); probs.append(np.exp(log_probs[t, c]))
        elif c:
            probs[-1] = max(probs[-1], np.exp(log_probs[t, c]))
        previous = c
    return "".join(digits), float(np.prod(probs))


# ---------------------------------------------------------------- runtime

def _seg_input(view):
    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return rgb.transpose(2, 0, 1)[None]


def rec_input(crops):
    batch = np.stack([cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]).astype(np.float32) / 127.5 - 1.0
    return batch.transpose(0, 3, 1, 2)


class MeterModel:
    def __init__(self, models_dir: Path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.seg = ort.InferenceSession(str(models_dir / SEG_FILE), options, providers=["CPUExecutionProvider"])
        self.rec = ort.InferenceSession(str(models_dir / REC_FILE), options, providers=["CPUExecutionProvider"])
        meta_path = models_dir / META_FILE
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        self.digits = int(self.meta.get("digits", 8))

    @staticmethod
    def available(models_dir: Path) -> bool:
        return (models_dir / SEG_FILE).exists() and (models_dir / REC_FILE).exists()

    def _segment(self, image, matrix):
        view = render_view(image, matrix)
        logits = self.seg.run(None, {"image": _seg_input(view)})[0][0, 0]
        return mask_rect(1 / (1 + np.exp(-logits)), matrix)

    def locate(self, image, region=None):
        """Rotated counter box. `region` (x, y, w, h) limits the search to a calibrated window."""
        x, y, w, h = region or (0, 0, image.shape[1], image.shape[0])
        rect, score = self._segment(image, view_matrix((x + w / 2, y + h / 2), (w, h)))
        if rect is None:
            return None, 0.0
        (cx, cy), (long_side, short_side), _ = rect
        side = max(long_side * ZOOM_LONG, short_side * ZOOM_SHORT, 48.0)
        zoomed, zoom_score = self._segment(image, view_matrix((cx, cy), (side, side)))
        if zoomed is not None and zoomed[1][0] > long_side * 0.5:
            return zoomed, zoom_score
        return rect, score

    def recognize(self, image, rect):
        """Reads the counter from three box sizes and both orientations, then keeps the more confident decoder."""
        (cx, cy), (w, h), angle = rect
        crops = []
        for scale_w, scale_h in BOX_VARIANTS:
            variant = ((cx, cy), (w * scale_w, h * scale_h), angle)
            crops += [rectify(image, variant), rectify(image, variant, flip=True)]
        outputs = self.rec.run(None, {"crop": rec_input(crops)})
        log_probs, orient = outputs[0], outputs[1]
        upright = np.exp(orient[:, 0]) / np.exp(orient).sum(axis=1)
        choice = 0 if upright[0::2].mean() >= upright[1::2].mean() else 1
        digits, confidence = ctc_best_digits(np.log(np.exp(log_probs[choice::2]).mean(axis=0)), self.digits)
        if len(outputs) > 2:
            probs = np.exp(outputs[2][choice::2]).mean(axis=0)  # (digits, 10)
            position_confidence = float(np.prod(probs.max(axis=1)))
            if position_confidence >= confidence:
                digits, confidence = "".join(str(d) for d in probs.argmax(axis=1)), position_confidence
        return digits, confidence, ("upright" if choice == 0 else "flipped")

    def read(self, image, region=None):
        rect, locate_score = self.locate(image, region)
        if rect is None:
            return None
        digits, confidence, orientation = self.recognize(image, rect)
        return {"digits": digits, "confidence": confidence * min(1.0, locate_score / 0.6), "rect": rect, "orientation": orientation}
