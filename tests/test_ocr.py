"""Unit tests for OCR engines, scanned PDFs, image ingest, and page limit guards."""

from __future__ import annotations

import io
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageDraw, ImageFont

from athena.ingest import IngestError, extract_document, extract_text
from athena.ocr.engine import (
    QwenVLEngine,
    TesseractEngine,
    VisionEngine,
    get_ocr_engine,
)


def _draw_test_image(text: str, width: int = 700, height: int = 200) -> Image.Image:
    """Create a high-contrast test image with text for OCR."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Try using system font or default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size=32)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 40), text, fill=(0, 0, 0), font=font)
    return img


def _make_scanned_pdf(text: str) -> bytes:
    """Create a minimal PDF that contains an image with text (no PDF text layer)."""
    img = _draw_test_image(text, width=800, height=300)
    pdf = pdfium.PdfDocument.new()
    # A standard page
    page = pdf.new_page(width=800, height=300)
    # Convert image to raw bytes and insert into page
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    image_obj = pdfium.PdfImage.new(pdf)
    image_obj.load_jpeg(buf)
    image_obj.set_matrix(pdfium.PdfMatrix(800, 0, 0, 300, 0, 0))
    page.insert_obj(image_obj)
    page.gen_content()
    
    out = io.BytesIO()
    pdf.save(out)
    return out.getvalue()


def _make_multi_page_pdf(n_pages: int) -> bytes:
    """Create a minimal empty PDF with n_pages pages."""
    pdf = pdfium.PdfDocument.new()
    for _ in range(n_pages):
        pdf.new_page(width=400, height=400)
    out = io.BytesIO()
    pdf.save(out)
    return out.getvalue()


def _make_digital_pdf(lines: list[str]) -> bytes:
    """Minimal one-page PDF with a Helvetica digital text layer."""
    content = "BT /F1 12 Tf 72 720 Td " + " ".join(f"({t}) Tj 0 -16 Td" for t in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


# ---------------------------------------------------------------- Tests

def test_ocr_factory_and_selection():
    eng_tess = get_ocr_engine("tesseract")
    assert isinstance(eng_tess, TesseractEngine)
    assert eng_tess.name == "tesseract"

    with pytest.raises(ValueError, match="Unknown OCR engine"):
        get_ocr_engine("non_existent_engine")


def test_tesseract_engine_word_confidence():
    engine = TesseractEngine(low_confidence_threshold=0.70)
    if not engine.is_available():
        pytest.skip("Tesseract is not available on this system")

    img = _draw_test_image("Aspirin 81 mg PO daily\nWarfarin 5 mg PO daily")
    res = engine.ocr_page(img, page_num=1)
    assert "Aspirin" in res.text or "aspirin" in res.text.lower()
    assert len(res.words) > 0
    # Check that word positions and confidences are well-formed
    for w in res.words:
        assert 0.0 <= w.confidence <= 1.0
        assert 0 <= w.start < w.end <= len(res.text)


def test_vision_engine_availability():
    engine = VisionEngine(low_confidence_threshold=0.70)
    if not engine.is_available():
        pytest.skip("Apple Vision is not available on this system or in this environment")

    img = _draw_test_image("Lisinopril 20 mg PO daily")
    res = engine.ocr_page(img, page_num=1)
    assert "Lisinopril" in res.text or "lisinopril" in res.text.lower()
    assert len(res.words) > 0


def test_qwen_vl_engine_availability():
    engine = QwenVLEngine()
    if not engine.is_available():
        pytest.skip("Qwen2.5-VL via Ollama is not available")

    img = _draw_test_image("Metformin 500 mg PO BID")
    res = engine.ocr_page(img, page_num=1)
    assert "Metformin" in res.text


def test_extract_document_image():
    engine = get_ocr_engine("tesseract")
    if not engine.is_available():
        pytest.skip("Tesseract is not available")

    img = _draw_test_image("Metoprolol 25 mg daily")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    res = extract_document("prescription.png", data, ocr_engine="tesseract")
    assert res.source == "ocr"
    assert res.ocr_engine == "tesseract"
    assert res.page_count == 1
    assert "Metoprolol" in res.text or "metoprolol" in res.text.lower()
    assert len(res.preview_images) == 1
    assert res.preview_images[0].startswith("data:image/png;base64,")


def test_digital_pdf_skips_ocr():
    """A PDF with a digital text layer must use digital extraction and skip OCR."""
    pdf_bytes = _make_digital_pdf(["Discharge Summary", "Warfarin 5 mg PO daily"])
    res = extract_document("discharge.pdf", pdf_bytes)
    assert res.source == "digital"
    assert res.ocr_engine is None
    assert "Warfarin 5 mg PO daily" in res.text
    assert len(res.preview_images) == 0


def test_scanned_pdf_renders_and_ocrs():
    """A scanned PDF with no text layer renders pages and runs OCR."""
    engine = get_ocr_engine("tesseract")
    if not engine.is_available():
        pytest.skip("Tesseract is not available")

    pdf_bytes = _make_scanned_pdf("Furosemide 40 mg PO daily")
    res = extract_document("scan.pdf", pdf_bytes, ocr_engine="tesseract")
    assert res.source == "ocr"
    assert res.ocr_engine == "tesseract"
    assert "Furosemide" in res.text or "furosemide" in res.text.lower()


def test_page_limit_guard():
    """Documents exceeding max_pages (default 10) must be rejected with IngestError."""
    long_pdf = _make_multi_page_pdf(11)
    with pytest.raises(IngestError, match="exceeds the maximum limit of 10 pages"):
        extract_document("too_long.pdf", long_pdf)


def test_unsupported_file_type():
    with pytest.raises(IngestError, match="Unsupported file type"):
        extract_document("note.xyz", b"some binary data")


def test_empty_scanned_pdf_error():
    blank_pdf = _make_multi_page_pdf(1)
    with pytest.raises(IngestError, match="No text found in this scanned PDF"):
        extract_document("blank_scan.pdf", blank_pdf)
