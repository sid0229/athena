"""OCR engines for reading scanned notes and handwritten prescriptions.

Supported engines:
- VisionEngine: Apple Vision via ocrmac (macOS built-in, fast, MIT licence). Default on macOS.
- TesseractEngine: Tesseract via pytesseract (Apache-2.0 licence). Portable fallback for printed text.
- QwenVLEngine: qwen2.5vl:3b via Ollama (Qwen research licence). Optional VLM for hard handwriting.

OCR only produces text and confidence data; it never makes clinical decisions.
All processing runs locally on this device.
"""

from __future__ import annotations

import base64
import io
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

# Avoid matplotlib font cache build delays/warnings in ocrmac
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")


@dataclass
class WordConfidence:
    word: str
    confidence: float
    start: int
    end: int
    page: int = 1
    bbox: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "confidence": round(self.confidence, 3),
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": [round(b, 2) for b in self.bbox] if self.bbox else None,
        }


@dataclass
class PageOCRResult:
    page_num: int
    text: str
    words: list[WordConfidence]
    low_confidence_words: list[WordConfidence]
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_num": self.page_num,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "low_confidence_words": [w.to_dict() for w in self.low_confidence_words],
            "seconds": round(self.seconds, 3),
        }


@dataclass
class OCRResult:
    text: str
    pages: list[PageOCRResult]
    low_confidence_words: list[WordConfidence]
    engine: str
    page_count: int
    seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "pages": [p.to_dict() for p in self.pages],
            "low_confidence_words": [w.to_dict() for w in self.low_confidence_words],
            "engine": self.engine,
            "page_count": self.page_count,
            "seconds": round(self.seconds, 3),
        }


class BaseOCREngine:
    name: str = "base"

    def __init__(self, low_confidence_threshold: float = 0.70):
        self.threshold = low_confidence_threshold

    def is_available(self) -> bool:
        raise NotImplementedError

    def ocr_page(self, image: Image.Image, page_num: int = 1) -> PageOCRResult:
        raise NotImplementedError

    def ocr_document(self, images: list[Image.Image]) -> OCRResult:
        t0 = time.perf_counter()
        pages: list[PageOCRResult] = []
        doc_text_parts: list[str] = []
        all_low_conf: list[WordConfidence] = []
        global_char_offset = 0

        for i, img in enumerate(images, start=1):
            page_res = self.ocr_page(img, page_num=i)
            # Recompute character offsets relative to full document
            page_low_conf: list[WordConfidence] = []
            for w in page_res.low_confidence_words:
                adjusted = WordConfidence(
                    word=w.word,
                    confidence=w.confidence,
                    start=w.start + global_char_offset,
                    end=w.end + global_char_offset,
                    page=w.page,
                    bbox=w.bbox,
                )
                page_low_conf.append(adjusted)
                all_low_conf.append(adjusted)

            pages.append(PageOCRResult(
                page_num=page_res.page_num,
                text=page_res.text,
                words=page_res.words,
                low_confidence_words=page_low_conf,
                seconds=page_res.seconds,
            ))
            doc_text_parts.append(page_res.text)
            # Add newline separator length between pages
            global_char_offset += len(page_res.text)
            if i < len(images):
                global_char_offset += 2  # for "\n\n"

        full_text = "\n\n".join(doc_text_parts).strip()
        elapsed = time.perf_counter() - t0
        return OCRResult(
            text=full_text,
            pages=pages,
            low_confidence_words=all_low_conf,
            engine=self.name,
            page_count=len(images),
            seconds=elapsed,
        )


class VisionEngine(BaseOCREngine):
    """Apple Vision via ocrmac (macOS native, MIT licence)."""
    name = "vision"

    def is_available(self) -> bool:
        try:
            from ocrmac import ocrmac
            from PIL import ImageDraw
            # Test text recognition on a small sample to verify Vision subsystem access
            img = Image.new("RGB", (400, 100), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((20, 30), "Athena Medical OCR", fill=(0, 0, 0))
            res = ocrmac.OCR(img, framework="vision").recognize()
            return len(res) > 0
        except Exception:
            return False

    def ocr_page(self, image: Image.Image, page_num: int = 1) -> PageOCRResult:
        t0 = time.perf_counter()
        from ocrmac import ocrmac

        if image.mode != "RGB":
            image = image.convert("RGB")

        # ocrmac returns list of (line_text, confidence, (x1, y1, x2, y2))
        try:
            results = ocrmac.OCR(image, framework="vision", recognition_level="accurate").recognize(px=True)
        except Exception as e:
            raise RuntimeError(f"Apple Vision OCR failed: {e}") from e

        lines: list[str] = []
        words: list[WordConfidence] = []
        low_conf: list[WordConfidence] = []
        char_pos = 0

        for line_text, conf, bbox in results:
            line_str = line_text.strip()
            if not line_str:
                continue
            line_start = char_pos
            for match in re.finditer(r"\S+", line_str):
                w = match.group(0)
                w_start = line_start + match.start()
                w_end = line_start + match.end()
                wc = WordConfidence(
                    word=w,
                    confidence=float(conf),
                    start=w_start,
                    end=w_end,
                    page=page_num,
                    bbox=tuple(bbox) if bbox else None,
                )
                words.append(wc)
                if wc.confidence < self.threshold:
                    low_conf.append(wc)

            lines.append(line_str)
            char_pos += len(line_str) + 1  # for "\n"

        page_text = "\n".join(lines)
        return PageOCRResult(
            page_num=page_num,
            text=page_text,
            words=words,
            low_confidence_words=low_conf,
            seconds=time.perf_counter() - t0,
        )


class TesseractEngine(BaseOCREngine):
    """Tesseract via pytesseract (Apache-2.0 licence)."""
    name = "tesseract"

    def is_available(self) -> bool:
        try:
            import pytesseract
            img = Image.new("RGB", (100, 30), color=(255, 255, 255))
            pytesseract.image_to_string(img)
            return True
        except Exception:
            return False

    def ocr_page(self, image: Image.Image, page_num: int = 1) -> PageOCRResult:
        t0 = time.perf_counter()
        import pytesseract

        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except Exception as e:
            raise RuntimeError(f"Tesseract OCR failed: {e}") from e

        # Group words by (block_num, par_num, line_num)
        n_items = len(data["text"])
        line_map: dict[tuple[int, int, int], list[int]] = {}
        for i in range(n_items):
            word = data["text"][i].strip()
            if not word:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            line_map.setdefault(key, []).append(i)

        lines: list[str] = []
        words: list[WordConfidence] = []
        low_conf: list[WordConfidence] = []
        char_pos = 0

        for key, indices in line_map.items():
            line_words = []
            for idx in indices:
                w_str = data["text"][idx].strip()
                raw_conf = float(data["conf"][idx])
                # Tesseract conf is 0..100 (-1 for unclassified)
                conf = max(0.0, raw_conf) / 100.0 if raw_conf >= 0 else 0.5
                bbox = (
                    float(data["left"][idx]),
                    float(data["top"][idx]),
                    float(data["left"][idx] + data["width"][idx]),
                    float(data["top"][idx] + data["height"][idx]),
                )
                line_words.append((w_str, conf, bbox))

            line_text = " ".join(w for w, _, _ in line_words)
            line_start = char_pos
            w_offset = 0
            for w_str, conf, bbox in line_words:
                w_start = line_start + w_offset
                w_end = w_start + len(w_str)
                wc = WordConfidence(
                    word=w_str,
                    confidence=conf,
                    start=w_start,
                    end=w_end,
                    page=page_num,
                    bbox=bbox,
                )
                words.append(wc)
                if conf < self.threshold:
                    low_conf.append(wc)
                w_offset += len(w_str) + 1

            lines.append(line_text)
            char_pos += len(line_text) + 1

        page_text = "\n".join(lines)
        return PageOCRResult(
            page_num=page_num,
            text=page_text,
            words=words,
            low_confidence_words=low_conf,
            seconds=time.perf_counter() - t0,
        )


class QwenVLEngine(BaseOCREngine):
    """Qwen2.5-VL via local Ollama (Qwen research licence)."""
    name = "qwen_vl"

    def __init__(self, low_confidence_threshold: float = 0.70, model: str = "qwen2.5vl:3b",
                 host: str = "http://localhost:11434"):
        super().__init__(low_confidence_threshold)
        self.model = model
        self.host = host.rstrip("/")

    def is_available(self) -> bool:
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                import json
                data = json.loads(resp.read().decode())
                models = [m.get("name", "") for m in data.get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False

    def ocr_page(self, image: Image.Image, page_num: int = 1) -> PageOCRResult:
        t0 = time.perf_counter()
        import json
        import urllib.request

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Encode image to JPEG base64
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        prompt = (
            "Transcribe the text in this image verbatim. "
            "Do not interpret, summarize, or correct spelling. "
            "Return only the transcription, preserving line breaks."
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.0,
            },
        }

        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                transcribed = res_data.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(f"Qwen2.5-VL OCR failed via Ollama: {e}") from e

        # Parse words and locate any uncertain cues like [?], (?), [unclear]
        uncertain_pattern = re.compile(r"(\[?\?\]?|\[unclear\]|\[illegible\])", re.I)
        words: list[WordConfidence] = []
        low_conf: list[WordConfidence] = []
        char_pos = 0

        for line in transcribed.split("\n"):
            line_str = line.strip()
            line_start = char_pos
            for match in re.finditer(r"\S+", line_str):
                w = match.group(0)
                w_start = line_start + match.start()
                w_end = line_start + match.end()
                is_uncertain = bool(uncertain_pattern.search(w))
                conf = 0.40 if is_uncertain else 0.95
                wc = WordConfidence(
                    word=w,
                    confidence=conf,
                    start=w_start,
                    end=w_end,
                    page=page_num,
                    bbox=None,
                )
                words.append(wc)
                if conf < self.threshold:
                    low_conf.append(wc)
            char_pos += len(line_str) + 1

        return PageOCRResult(
            page_num=page_num,
            text=transcribed,
            words=words,
            low_confidence_words=low_conf,
            seconds=time.perf_counter() - t0,
        )


def get_ocr_engine(engine_name: str | None = None, threshold: float | None = None) -> BaseOCREngine:
    """Factory to get the requested or default OCR engine."""
    from athena.config import load_config
    cfg = load_config().get("ocr", {})

    name = engine_name or cfg.get("engine", "vision")
    thresh = threshold if threshold is not None else float(cfg.get("low_confidence_threshold", 0.70))

    if name == "vision":
        engine = VisionEngine(low_confidence_threshold=thresh)
        if engine.is_available():
            return engine
        # Fallback to Tesseract if Apple Vision is unavailable (e.g. CI or non-macOS)
        tess = TesseractEngine(low_confidence_threshold=thresh)
        if tess.is_available():
            return tess
        return engine  # Return vision engine anyway so calling ocr_page raises clear error

    elif name == "qwen_vl":
        model = cfg.get("qwen_vl_model", "qwen2.5vl:3b")
        return QwenVLEngine(low_confidence_threshold=thresh, model=model)

    elif name == "tesseract":
        return TesseractEngine(low_confidence_threshold=thresh)

    raise ValueError(f"Unknown OCR engine '{name}'. Expected 'vision', 'qwen_vl', or 'tesseract'.")
