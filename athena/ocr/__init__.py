"""Athena OCR package for reading handwritten prescriptions and scanned clinical notes."""

from athena.ocr.engine import (
    BaseOCREngine,
    OCRResult,
    PageOCRResult,
    QwenVLEngine,
    TesseractEngine,
    VisionEngine,
    WordConfidence,
    get_ocr_engine,
)

__all__ = [
    "BaseOCREngine",
    "OCRResult",
    "PageOCRResult",
    "QwenVLEngine",
    "TesseractEngine",
    "VisionEngine",
    "WordConfidence",
    "get_ocr_engine",
]
