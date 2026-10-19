from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from .meter_model import META_FILE, MeterModel

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class MeterReader:
    """Water-meter reader: trained counter models (models/*.onnx) when present, RapidOCR multi-pass as the fallback."""

    def __init__(self, config_path: Path, overrides: dict | None = None, models_dir: Path = MODELS_DIR):
        self.config_path = config_path
        self.overrides = overrides or {}
        self.models_dir = models_dir
        self._engine = None
        self._model = None
        self._lock = Lock()

    def _meter_model(self):
        if self._model is None and MeterModel.available(self.models_dir):
            self._model = MeterModel(self.models_dir)
        return self._model

    def _model_decimals(self) -> int:
        try:
            return int(json.loads((self.models_dir / META_FILE).read_text(encoding="utf-8")).get("decimal_places", 0))
        except (OSError, ValueError, TypeError):
            return 0

    def config(self) -> dict:
        # Until calibration is saved, decimals follow the trained model (3 red wheels on the dataset's meters).
        default = {"roi": None, "decimal_places": self._model_decimals(), "min_digits": 4, "max_digits": 10}
        if self.config_path.exists():
            try:
                default.update(json.loads(self.config_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        default.update(self.overrides)
        return default

    def save_config(self, values: dict) -> dict:
        config = self.config()
        config.update(values)
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config

    def _ocr(self):
        if self._engine is None:
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
        return self._engine

    @staticmethod
    def decode(data: bytes) -> np.ndarray:
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("The uploaded bytes are not a valid image")
        return image

    @staticmethod
    def _fixed_roi(image: np.ndarray, roi: dict | None):
        h, w = image.shape[:2]
        if not roi:
            return None
        x = int(np.clip(float(roi.get("x", 0)), 0, 1) * w)
        y = int(np.clip(float(roi.get("y", 0)), 0, 1) * h)
        rw = int(np.clip(float(roi.get("w", 1)), 0.01, 1) * w)
        rh = int(np.clip(float(roi.get("h", 1)), 0.01, 1) * h)
        x2, y2 = min(w, x + rw), min(h, y + rh)
        if x2 - x < 20 or y2 - y < 10:
            return None
        return image[y:y2, x:x2], (x, y, x2 - x, y2 - y), "fixed"

    @staticmethod
    def _auto_roi(image: np.ndarray):
        h, w = image.shape[:2]
        scaled = image
        factor = 1.0
        if max(h, w) > 1400:
            factor = 1400 / max(h, w)
            scaled = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 135)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        sh, sw = scaled.shape[:2]
        candidates = []
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area_ratio = (cw * ch) / (sw * sh)
            aspect = cw / max(ch, 1)
            if .012 <= area_ratio <= .42 and 2.0 <= aspect <= 9.5 and cw > 100:
                center_penalty = abs((x + cw / 2) / sw - .5) + abs((y + ch / 2) / sh - .5)
                score = area_ratio * 4 + min(aspect, 6) / 10 - center_penalty * .28
                candidates.append((score, x, y, cw, ch))
        if not candidates:
            x, y, cw, ch = int(sw * .08), int(sh * .30), int(sw * .84), int(sh * .40)
        else:
            _, x, y, cw, ch = max(candidates)
            pad_x, pad_y = int(cw * .05), int(ch * .18)
            x, y = max(0, x - pad_x), max(0, y - pad_y)
            cw, ch = min(sw - x, cw + 2 * pad_x), min(sh - y, ch + 2 * pad_y)
        x, y, cw, ch = [int(v / factor) for v in (x, y, cw, ch)]
        return image[y:y + ch, x:x + cw], (x, y, cw, ch), "automatic"

    @staticmethod
    def _variants(crop: np.ndarray):
        if crop.shape[1] < 900:
            scale = 900 / max(crop.shape[1], 1)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(2.2, (8, 8)).apply(gray)
        return [
            ("color", crop),
            ("contrast", clahe),
            ("otsu", cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
            ("adaptive", cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 9)),
        ]

    @staticmethod
    def _unpack_output(output):
        # RapidOCR 3.x object output.
        if hasattr(output, "txts"):
            texts = list(output.txts or [])
            scores = list(output.scores or [])
            return texts, scores
        # Compatibility with older tuple/list output.
        raw = output[0] if isinstance(output, tuple) else output
        texts, scores = [], []
        for item in raw or []:
            if len(item) >= 3:
                texts.append(str(item[1])); scores.append(float(item[2]))
        return texts, scores

    @staticmethod
    def normalize(text: str, decimal_places: int = 0) -> str:
        substitutions = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8", "Z": "2"})
        digits = re.sub(r"\D", "", text.translate(substitutions))
        if decimal_places > 0 and len(digits) > decimal_places:
            return f"{digits[:-decimal_places]}.{digits[-decimal_places:]}"
        return digits

    def read(self, image: np.ndarray) -> dict:
        config = self.config()
        fixed = self._fixed_roi(image, config.get("roi"))
        model = self._meter_model()
        if model is not None:
            with self._lock:
                found = model.read(image, fixed[1] if fixed else None)
            if found is not None:
                return self._model_result(image, found, int(config.get("decimal_places", 0)), "fixed" if fixed else "model")
        selected = fixed or self._auto_roi(image)
        crop, (x, y, w, h), roi_source = selected
        candidates = []
        with self._lock:
            engine = self._ocr()
            for variant, prepared in self._variants(crop):
                try:
                    texts, scores = self._unpack_output(engine(prepared))
                except Exception:
                    continue
                raw = "".join(texts)
                reading = self.normalize(raw, int(config.get("decimal_places", 0)))
                conf = float(np.mean(scores) * 100) if scores else 0.0
                valid_len = int(config.get("min_digits", 4)) <= len(reading.replace(".", "")) <= int(config.get("max_digits", 10))
                rank = conf + (20 if valid_len else -20) + len(reading) * .4
                candidates.append({"reading": reading, "confidence": round(conf, 1), "variant": variant, "rank": rank, "raw": raw})
        best = max(candidates, key=lambda c: c["rank"], default={"reading": "", "confidence": 0.0, "variant": "none", "raw": ""})
        best.pop("rank", None)
        return {**best, "roi": {"x": x, "y": y, "w": w, "h": h, "source": roi_source}, "image_width": image.shape[1], "image_height": image.shape[0]}

    def _model_result(self, image: np.ndarray, found: dict, decimal_places: int, roi_source: str) -> dict:
        h, w = image.shape[:2]
        box = cv2.boxPoints(found["rect"])
        x, y = np.clip(box.min(axis=0), 0, [w, h]).astype(int)
        x2, y2 = np.clip(box.max(axis=0), 0, [w, h]).astype(int)
        digits = found["digits"]
        return {"reading": self.normalize(digits, decimal_places) if digits else "", "confidence": round(found["confidence"] * 100, 1),
                "variant": f"model-{found['orientation']}", "raw": digits,
                "roi": {"x": int(x), "y": int(y), "w": int(x2 - x), "h": int(y2 - y), "source": roi_source, "angle": round(found["rect"][2], 1)},
                "image_width": w, "image_height": h}
