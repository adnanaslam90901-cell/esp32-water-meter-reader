"""Train the counter-locator and digit-reader models on the Kaggle Water Meters dataset and export them to models/.

    python -m venv .trainenv
    .trainenv/bin/pip install -r requirements-train.txt
    .trainenv/bin/python training/train_meter_models.py ~/Downloads/water-meters-dataset/WaterMeters

Dataset layout (https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset):
    data.csv (photo_name, value, location), images/*.jpg, masks/*.jpg

Photos are split by file name: 75% train, 10% validation, 15% test. The test photos are never used for training,
so the accuracy printed at the end is what to expect on new photos of similar meters. The split is saved to
models/meter_split.json and exports mark each photo's split.

The app never needs PyTorch: both models are exported to ONNX and run with onnxruntime.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.meter_model import (  # noqa: E402
    META_FILE, REC_FILE, REC_H, REC_PAD_X, REC_PAD_Y, REC_W, SEG_FILE, SEG_SIZE, MeterModel,
    normalize_rect, rectify, render_view, view_matrix,
)

DIGITS = 8
SOURCE_MAX = 640                       # stored photo size for the locator's training views
CTX_PAD_X, CTX_PAD_Y = 0.30, 0.90      # stored strip around the counter, room for position jitter
CTX_W = round(REC_W * (1 + 2 * CTX_PAD_X) / (1 + 2 * REC_PAD_X))
CTX_H = round(REC_H * (1 + 2 * CTX_PAD_Y) / (1 + 2 * REC_PAD_Y))


def split_of(name: str) -> str:
    bucket = int(hashlib.md5(name.encode()).hexdigest(), 16) % 100
    return "test" if bucket < 15 else "val" if bucket < 25 else "train"


def label_of(value: str) -> str | None:
    try:
        text = f"{round(float(value) * 1000):0{DIGITS}d}"
    except ValueError:
        return None
    return text if len(text) == DIGITS else None


# ---------------------------------------------------------------- data

def load_one(root: Path, row: dict):
    name, label = row["photo_name"], label_of(row["value"])
    image = cv2.imread(str(root / "images" / name))
    mask = cv2.imread(str(root / "masks" / name), cv2.IMREAD_GRAYSCALE)
    if label is None or image is None or mask is None:
        return None
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(mask > 127)
    if len(xs) < 50:
        return None
    rect = normalize_rect(cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32)))
    scale = min(1.0, SOURCE_MAX / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    (cx, cy), (w, h), angle = rect
    return {
        "name": name, "split": split_of(name), "label": label, "rect": rect, "flip": False,
        "jpeg": cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 94])[1],
        "mask": cv2.imencode(".png", (small_mask > 127).astype(np.uint8) * 255)[1],
        "small_rect": ((cx * scale, cy * scale), (w * scale, h * scale), angle),
        "ctx": rectify(image, rect, CTX_W, CTX_H, CTX_PAD_X, CTX_PAD_Y),
    }


def load_samples(root: Path, limit: int | None = None) -> list[dict]:
    with open(root / "data.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[:limit]
    with ThreadPoolExecutor(4) as pool:
        samples = [s for s in pool.map(lambda r: load_one(root, r), rows) if s]
    print(f"Loaded {len(samples)} of {len(rows)} photos "
          f"({sum(s['split'] == 'train' for s in samples)} train / {sum(s['split'] == 'val' for s in samples)} val / "
          f"{sum(s['split'] == 'test' for s in samples)} test)", flush=True)
    return samples


def to_tensor(bgr) -> torch.Tensor:
    rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0)


def color_augment(img):
    img = img.astype(np.float32)
    if random.random() < 0.8:
        img = img * random.uniform(0.6, 1.4) + random.uniform(-40, 40)
    if random.random() < 0.3:
        img = img * np.array([random.uniform(0.8, 1.2) for _ in range(3)], dtype=np.float32)
    if random.random() < 0.25:
        img = 255 * (np.clip(img, 0, 255) / 255) ** random.uniform(0.6, 1.6)
    if random.random() < 0.12:
        img = np.repeat(img.mean(axis=2, keepdims=True), 3, axis=2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.5, 1.6))
    if random.random() < 0.2:
        img = np.clip(img.astype(np.int16) + np.random.randint(-12, 13, img.shape), 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(25, 70)])[1], cv2.IMREAD_COLOR)
    return img


class SegData(Dataset):
    def __init__(self, samples, train):
        self.samples, self.train = samples, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        image, mask = cv2.imdecode(s["jpeg"], cv2.IMREAD_COLOR), cv2.imdecode(s["mask"], cv2.IMREAD_GRAYSCALE)
        h, w = image.shape[:2]
        center, size, angle, border = (w / 2, h / 2), (w, h), 0.0, cv2.BORDER_CONSTANT
        if self.train:
            angle = (random.choice([90, 180, 270]) if random.random() < 0.3 else 0) + random.uniform(-12, 12)
            border = random.choice([cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT])
            if random.random() < 0.55:  # whole photo, like the first pass at runtime
                size = (w * random.uniform(0.75, 1.08), h * random.uniform(0.75, 1.08))
                center = (w / 2 + random.uniform(-0.1, 0.1) * w, h / 2 + random.uniform(-0.1, 0.1) * h)
            else:  # zoomed on the counter, like the second pass
                (cx, cy), (long_side, short_side), _ = s["small_rect"]
                side = max(long_side * random.uniform(1.3, 2.4), short_side * random.uniform(3.0, 5.5), 24)
                center, size = (cx + random.uniform(-0.25, 0.25) * side, cy + random.uniform(-0.25, 0.25) * side), (side, side)
        matrix = view_matrix(center, size, angle)
        view = render_view(image, matrix, border=border)
        target = render_view(mask, matrix, interpolation=cv2.INTER_NEAREST)
        if self.train:
            view = color_augment(view)
        return to_tensor(view), torch.from_numpy((target > 127).astype(np.float32))[None]


class RecData(Dataset):
    def __init__(self, samples, train):
        self.samples, self.train = samples, train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        bw, bh = CTX_W / (1 + 2 * CTX_PAD_X), CTX_H / (1 + 2 * CTX_PAD_Y)
        cx, cy, angle = CTX_W / 2, CTX_H / 2, 0.0
        if self.train:  # imitate an imperfect box from the locator
            cx += random.uniform(-0.05, 0.05) * bw
            cy += random.uniform(-0.12, 0.12) * bh
            bw *= random.uniform(0.9, 1.12)
            bh *= random.uniform(0.85, 1.3)
            angle = random.uniform(-4, 4)
        crop = rectify(s["ctx"], ((cx, cy), (bw, bh), angle), flip=s["flip"])
        if self.train:
            crop = color_augment(crop)
        target = torch.tensor([int(d) + 1 for d in s["label"]], dtype=torch.long)
        return to_tensor(crop), target


# ---------------------------------------------------------------- models

def cbr(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SegNet(nn.Module):
    """Light U-Net: 256x256 photo -> counter-window logits (predicted at 128x128, upsampled). Sized for laptop CPUs."""

    def __init__(self, c=(16, 24, 48, 64, 96)):
        super().__init__()
        self.e1 = cbr(3, c[0], 2)                                   # 128
        self.e2 = nn.Sequential(cbr(c[0], c[1], 2), cbr(c[1], c[1]))  # 64
        self.e3 = nn.Sequential(cbr(c[1], c[2], 2), cbr(c[2], c[2]))  # 32
        self.e4 = nn.Sequential(cbr(c[2], c[3], 2), cbr(c[3], c[3]))  # 16
        self.e5 = nn.Sequential(cbr(c[3], c[4], 2), cbr(c[4], c[4]), cbr(c[4], c[4]))  # 8
        self.d4, self.d3 = cbr(c[4] + c[3], c[3]), cbr(c[3] + c[2], c[2])
        self.d2, self.d1 = cbr(c[2] + c[1], c[1]), cbr(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    @staticmethod
    def up(x, skip):
        return torch.cat([F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False), skip], 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2); e4 = self.e4(e3); e5 = self.e5(e4)
        d4 = self.d4(self.up(e5, e4)); d3 = self.d3(self.up(d4, e3)); d2 = self.d2(self.up(d3, e2)); d1 = self.d1(self.up(d2, e1))
        return F.interpolate(self.head(d1), scale_factor=2.0, mode="bilinear", align_corners=False)


class RecNet(nn.Module):
    """40x192 straightened counter -> per-column digit log-probabilities (CTC), upright/flipped logits and
    per-position digit log-probabilities (8 x 10). The position head learns fast and pulls CTC off its early plateau."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            cbr(3, 24), nn.MaxPool2d(2),
            cbr(24, 48), nn.MaxPool2d(2),
            cbr(48, 96), cbr(96, 96), nn.MaxPool2d((2, 1)),
            cbr(96, 128), cbr(128, 128), nn.MaxPool2d((2, 1)),
        )
        self.seq = nn.Sequential(
            nn.Conv1d(128 * (REC_H // 16), 192, 3, padding=1), nn.BatchNorm1d(192), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Conv1d(192, 192, 3, padding=2, dilation=2), nn.BatchNorm1d(192), nn.ReLU(inplace=True),
        )
        self.ctc_head = nn.Conv1d(192, 11, 1)
        self.position_head = nn.Conv1d(192, 10, 1)
        self.orient = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, x):
        features = self.body(x)
        hidden = self.seq(features.reshape(features.shape[0], 128 * (REC_H // 16), REC_W // 4))
        log_probs = F.log_softmax(self.ctc_head(hidden), dim=1).permute(0, 2, 1)
        pooled = F.avg_pool1d(hidden, (REC_W // 4) // DIGITS)
        positions = F.log_softmax(self.position_head(pooled), dim=1).permute(0, 2, 1)
        return log_probs, self.orient(features.mean(dim=(2, 3))), positions


# ---------------------------------------------------------------- training

def seed_worker(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def loader(dataset, batch, shuffle):
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=2, drop_last=shuffle, worker_init_fn=seed_worker)


def dice_loss(logits, target):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (prob.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)).mean()


def train_seg(samples, epochs, checkpoint: Path) -> SegNet:
    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "val"]
    model = SegNet()
    train_loader, val_loader = loader(SegData(train, True), 12, True), loader(SegData(val, False), 16, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-3, total_steps=epochs * len(train_loader), pct_start=0.15)
    best = -1.0
    for epoch in range(1, epochs + 1):
        started, total = time.time(), 0.0
        model.train()
        for x, y in train_loader:
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); schedule.step()
            total += loss.item()
        model.eval()
        inter = union = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred = (model(x) > 0).float()
                inter += (pred * y).sum().item(); union += ((pred + y) > 0).float().sum().item()
        iou = inter / max(union, 1)
        if iou > best:
            best = iou
            torch.save(model.state_dict(), checkpoint)
        print(f"  locator epoch {epoch}/{epochs}: loss {total / len(train_loader):.3f}, val IoU {iou:.3f} (best {best:.3f}), {time.time() - started:.0f}s", flush=True)
    model.load_state_dict(torch.load(checkpoint))
    return model.eval()


def ctc(log_probs, targets):
    steps = log_probs.shape[1]
    return F.ctc_loss(log_probs.permute(1, 0, 2), targets, torch.full((len(targets),), steps, dtype=torch.long),
                      torch.full((len(targets),), DIGITS, dtype=torch.long), blank=0, zero_infinity=True)


def greedy(log_probs) -> list[str]:
    out = []
    for row in log_probs.argmax(-1).tolist():
        digits, previous = [], 0
        for c in row:
            if c and c != previous:
                digits.append(str(c - 1))
            previous = c
        out.append("".join(digits))
    return out


def fix_orientation(model, samples):
    """Photos taken sideways or upside down get a guessed orientation first; the partly trained reader corrects it."""
    model.eval()
    changed = 0
    with torch.no_grad():
        for start in range(0, len(samples), 64):
            chunk = samples[start:start + 64]
            targets = torch.tensor([[int(d) + 1 for d in s["label"]] for s in chunk])
            losses = []
            for flip in (False, True):
                x = torch.stack([to_tensor(rectify(s["ctx"], ((CTX_W / 2, CTX_H / 2), (CTX_W / (1 + 2 * CTX_PAD_X), CTX_H / (1 + 2 * CTX_PAD_Y)), 0.0), flip=flip)) for s in chunk])
                log_probs, _, positions = model(x)
                steps = log_probs.shape[1]
                ctc_loss = F.ctc_loss(log_probs.permute(1, 0, 2), targets, torch.full((len(chunk),), steps), torch.full((len(chunk),), DIGITS), reduction="none", zero_infinity=True)
                position_loss = F.nll_loss(positions.reshape(-1, 10), (targets - 1).reshape(-1), reduction="none").reshape(len(chunk), DIGITS).sum(1)
                losses.append(ctc_loss + position_loss)
            for s, keep, flipped in zip(chunk, losses[0].tolist(), losses[1].tolist()):
                should_flip = flipped + 1.0 < keep
                if should_flip != s["flip"] and (should_flip or keep + 1.0 < flipped):
                    s["flip"], changed = should_flip, changed + 1
    return changed


def position_digits(positions) -> list[str]:
    return ["".join(str(int(d)) for d in row) for row in positions.argmax(-1)]


def train_rec(samples, epochs, checkpoint: Path) -> RecNet:
    train = [s for s in samples if s["split"] == "train"]
    val = [s for s in samples if s["split"] == "val"]
    model = RecNet()
    batch = 16
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=5e-4)
    steps_per_epoch = len(train) // batch
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=epochs * steps_per_epoch, pct_start=0.15)
    best = -1.0
    for epoch in range(1, epochs + 1):
        if epoch in (max(2, epochs // 4), epochs // 2):
            changed = fix_orientation(model, train + val)
            print(f"  reader: corrected orientation of {changed} crops", flush=True)
        started, total = time.time(), 0.0
        model.train()
        for x, targets in loader(RecData(train, True), batch, True):
            flipped = torch.flip(x[: len(x) // 3], dims=(2, 3))
            log_probs, orient, positions = model(torch.cat([x, flipped]))
            orient_target = torch.cat([torch.zeros(len(x), dtype=torch.long), torch.ones(len(flipped), dtype=torch.long)])
            # Full-weight CTC stalls on its uniform-output plateau and drags the shared layers with it; a small
            # weight lets the position and orientation heads build the features first.
            loss = (F.nll_loss(positions[: len(x)].reshape(-1, 10), (targets - 1).reshape(-1))
                    + 0.5 * F.cross_entropy(orient, orient_target, weight=torch.tensor([1.0, 3.0]))
                    + 0.2 * ctc(log_probs[: len(x)], targets))
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); schedule.step()
            total += loss.item()
        model.eval()
        ctc_hits = position_hits = orient_hits = count = 0
        with torch.no_grad():
            for x, targets in loader(RecData(val, False), 64, False):
                log_probs, orient, positions = model(x)
                truth = ["".join(str(int(t) - 1) for t in row) for row in targets]
                ctc_hits += sum(p == t for p, t in zip(greedy(log_probs), truth))
                position_hits += sum(p == t for p, t in zip(position_digits(positions), truth))
                _, orient_flipped, _ = model(torch.flip(x, dims=(2, 3)))
                orient_hits += int((orient.argmax(1) == 0).sum()) + int((orient_flipped.argmax(1) == 1).sum())
                count += len(x)
        accuracy = max(ctc_hits, position_hits) / max(count, 1)
        if accuracy >= best:
            best = accuracy
            torch.save(model.state_dict(), checkpoint)
        print(f"  reader epoch {epoch}/{epochs}: loss {total / steps_per_epoch:.3f}, val exact CTC {100 * ctc_hits / max(count, 1):.1f}% / "
              f"positions {100 * position_hits / max(count, 1):.1f}% (best {100 * best:.1f}%), "
              f"orientation {100 * orient_hits / max(2 * count, 1):.1f}%, {time.time() - started:.0f}s", flush=True)
    model.load_state_dict(torch.load(checkpoint))
    return model.eval()


# ---------------------------------------------------------------- export + evaluation

def export(model, example, path: Path, input_name: str, output_names: list[str]):
    torch.onnx.export(model, (example,), str(path), input_names=[input_name], output_names=output_names,
                      dynamic_axes={input_name: {0: "batch"}, **{n: {0: "batch"} for n in output_names}}, opset_version=17, dynamo=False)


def evaluate(root: Path, samples, models_dir: Path) -> dict:
    model = MeterModel(models_dir)
    results = {}
    for split in ("test", "val"):
        subset = [s for s in samples if s["split"] == split]
        exact = located = black = 0
        started = time.time()
        for s in subset:
            image = cv2.imread(str(root / "images" / s["name"]))
            out = model.read(image)
            if out is None:
                continue
            located += 1
            exact += out["digits"] == s["label"]
            black += out["digits"][:5] == s["label"][:5]
        n = max(len(subset), 1)
        results[split] = {"photos": len(subset), "exact_percent": round(100 * exact / n, 1),
                          "whole_cubic_metres_percent": round(100 * black / n, 1), "counter_found_percent": round(100 * located / n, 1),
                          "seconds_per_photo": round((time.time() - started) / n, 3)}
        print(f"  {split}: {results[split]}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="folder with data.csv, images/ and masks/")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    parser.add_argument("--seg-epochs", type=int, default=25)
    parser.add_argument("--rec-epochs", type=int, default=40)
    parser.add_argument("--stage", choices=["all", "seg", "rec", "export"], default="all",
                        help="seg/rec train one model (they can run in parallel), export converts saved checkpoints and evaluates")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="use only the first N photos (quick smoke test)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    checkpoints = ROOT / "training" / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.dataset, args.limit)

    if args.stage in ("all", "seg"):
        print("Training the counter locator", flush=True)
        train_seg(samples, args.seg_epochs, checkpoints / "seg.pt")
    if args.stage in ("all", "rec"):
        print("Training the digit reader", flush=True)
        train_rec(samples, args.rec_epochs, checkpoints / "rec.pt")
    if args.stage in ("seg", "rec"):
        return
    seg, rec = SegNet(), RecNet()
    seg.load_state_dict(torch.load(checkpoints / "seg.pt")); rec.load_state_dict(torch.load(checkpoints / "rec.pt"))
    seg.eval(); rec.eval()

    export(seg, torch.zeros(1, 3, SEG_SIZE, SEG_SIZE), args.out / SEG_FILE, "image", ["logits"])
    export(rec, torch.zeros(2, 3, REC_H, REC_W), args.out / REC_FILE, "crop", ["log_probs", "orientation", "positions"])
    split = {name: sorted((s["name"] for s in samples if s["split"] == name), key=str) for name in ("train", "val", "test")}
    (args.out / "meter_split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    meta = {"digits": DIGITS, "decimal_places": 3, "dataset": "Kaggle tapakah68/yandextoloka-water-meters-dataset",
            "trained_at": time.strftime("%Y-%m-%d %H:%M")}
    (args.out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Checking the exported models on photos they never saw (end to end, full-size photos)", flush=True)
    meta["accuracy"] = evaluate(args.dataset, samples, args.out)
    (args.out / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved models to {args.out}")


if __name__ == "__main__":
    main()
