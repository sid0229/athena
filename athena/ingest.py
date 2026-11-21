"""Plain text extraction from uploaded notes and images.

Supported formats:
- Plain text: .txt
- Word documents: .docx (standard library only)
- Digital PDFs: .pdf with text layer (pypdf) — skips OCR
- Scanned PDFs: .pdf with no text layer rendered via pypdfium2 (Apache-2.0/BSD-3) and OCR'd
- Images: .png, .jpg, .jpeg, .heic (via OCR)

OCR only produces text and confidence annotations; it never makes clinical decisions.
The text is returned for the reviewer to check before analysis; nothing leaves the device.
"""

from __future__ import annotations

import base64
import io
import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any
from xml.etree import ElementTree

from PIL import Image

from athena.config import load_config
from athena.ocr.engine import get_ocr_engine

MAX_BYTES = 20 * 1024 * 1024          # 20 MB upload size limit
MAX_XML_BYTES = 50 * 1024 * 1024      # uncompressed document.xml limit (zip bombs)
DEFAULT_MAX_PAGES = 10                # maximum allowed pages
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class IngestError(ValueError):
    """The file cannot be turned into note text; the message is shown to the reviewer."""


@dataclass
class IngestResult:
    text: str
    source: str                      # "digital" | "ocr"
    ocr_engine: str | None = None    # "vision" | "qwen_vl" | "tesseract" | None
    low_confidence_words: list[dict[str, Any]] = field(default_factory=list)
    page_count: int = 1
    seconds: float = 0.0
    preview_images: list[str] = field(default_factory=list)  # Base64 data URLs for UI preview

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "ocr_engine": self.ocr_engine,
            "low_confidence_words": self.low_confidence_words,
            "page_count": self.page_count,
            "seconds": round(self.seconds, 3),
            "preview_images": self.preview_images,
        }


def extract_document(name: str, data: bytes, ocr_engine: str | None = None) -> IngestResult:
    """Extract document text, metadata, and OCR low-confidence words if applicable."""
    if len(data) > MAX_BYTES:
        raise IngestError("File is larger than 20 MB.")

    cfg = load_config().get("ocr", {})
    max_pages = int(cfg.get("max_pages", DEFAULT_MAX_PAGES))

    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""

    if ext == "txt":
        text = _tidy(_txt(data))
        if not text.strip():
            raise IngestError("No text found in this file.")
        return IngestResult(text=text, source="digital")

    elif ext == "docx":
        text = _tidy(_docx(data))
        if not text.strip():
            raise IngestError("No text found in this file.")
        return IngestResult(text=text, source="digital")

    elif ext == "pdf":
        return _pdf(data, max_pages=max_pages, ocr_engine=ocr_engine)

    elif ext in ("png", "jpg", "jpeg", "heic"):
        return _image(data, ext=ext, ocr_engine=ocr_engine)

    else:
        raise IngestError(
            "Unsupported file type: upload a .txt, .docx, .pdf, .png, .jpg, .jpeg or .heic file."
        )


def extract_text(name: str, data: bytes, ocr_engine: str | None = None) -> str:
    """Backwards-compatible text extraction."""
    return extract_document(name, data, ocr_engine=ocr_engine).text


def _txt(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _docx(data: bytes) -> str:
    """Body paragraphs and table rows in document order."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > MAX_XML_BYTES:
                raise IngestError("Document is too large to read.")
            root = ElementTree.fromstring(z.read(info))
    except (zipfile.BadZipFile, KeyError, ElementTree.ParseError):
        raise IngestError("This is not a readable .docx file (old .doc files are not supported).")
    body = root.find(f"{_W}body")
    lines: list[str] = []
    for el in (body if body is not None else []):
        if el.tag == f"{_W}p":
            lines.append(_para(el))
        elif el.tag == f"{_W}tbl":
            for row in el.iter(f"{_W}tr"):
                cells = [" ".join(_para(p) for p in tc.iter(f"{_W}p")).strip() for tc in row.iter(f"{_W}tc")]
                lines.append("\t".join(cells))
    return "\n".join(lines)


def _para(p: Any) -> str:
    out = []
    for el in p.iter():
        if el.tag == f"{_W}t" and el.text:
            out.append(el.text)
        elif el.tag == f"{_W}tab":
            out.append("\t")
        elif el.tag in (f"{_W}br", f"{_W}cr"):
            out.append("\n")
    return "".join(out)


def _image_to_data_url(img: Image.Image, max_dim: int = 1200) -> str:
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _load_single_image(data: bytes, ext: str) -> Image.Image:
    if ext == "heic":
        # macOS AppKit loader
        try:
            from AppKit import NSBitmapImageRep, NSData, NSImage, NSPNGFileType
            nsdata = NSData.dataWithBytes_length_(data, len(data))
            img = NSImage.alloc().initWithData_(nsdata)
            if img is not None:
                rep = NSBitmapImageRep.imageRepWithData_(nsdata)
                if rep is not None:
                    png_data = rep.representationUsingType_properties_(NSPNGFileType, None)
                    if png_data is not None:
                        return Image.open(io.BytesIO(png_data.bytes()))
        except Exception:
            pass

        # CLI fallback via macOS sips
        try:
            import subprocess
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".heic", delete=False) as f_in:
                f_in.write(data)
                in_path = f_in.name
            out_path = in_path + ".png"
            res = subprocess.run(["sips", "-s", "format", "png", in_path, "--out", out_path], capture_output=True)
            if res.returncode == 0 and os.path.exists(out_path):
                with open(out_path, "rb") as f_out:
                    png_bytes = f_out.read()
                try:
                    os.remove(in_path)
                    os.remove(out_path)
                except OSError:
                    pass
                return Image.open(io.BytesIO(png_bytes))
        except Exception:
            pass

        raise IngestError("Could not decode .heic image.")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except Exception as e:
        raise IngestError("This image file could not be read.") from e


def _image(data: bytes, ext: str, ocr_engine: str | None = None) -> IngestResult:
    img = _load_single_image(data, ext)
    engine = get_ocr_engine(ocr_engine)
    ocr_res = engine.ocr_document([img])

    text = _tidy(ocr_res.text)
    if not text.strip():
        raise IngestError("No text found in this image.")

    preview = [_image_to_data_url(img)]
    return IngestResult(
        text=text,
        source="ocr",
        ocr_engine=engine.name,
        low_confidence_words=[w.to_dict() for w in ocr_res.low_confidence_words],
        page_count=1,
        seconds=ocr_res.seconds,
        preview_images=preview,
    )


def _pdf(data: bytes, max_pages: int = DEFAULT_MAX_PAGES, ocr_engine: str | None = None) -> IngestResult:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestError("This PDF is password-protected.")
        if len(reader.pages) > max_pages:
            raise IngestError(f"PDF exceeds the maximum limit of {max_pages} pages (has {len(reader.pages)}).")
        pages = [p.extract_text() or "" for p in reader.pages]
    except (PdfReadError, ValueError, KeyError) as e:
        if isinstance(e, IngestError):
            raise
        raise IngestError("This PDF could not be read.")

    digital_text = "".join(pages).strip()
    if digital_text:
        # Digital PDF with valid text layer: skip OCR
        return IngestResult(
            text=_tidy("\n\n".join(pages)),
            source="digital",
            page_count=len(reader.pages),
        )

    # Scanned PDF: render pages via pypdfium2 and run OCR
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(data)
    except Exception as e:
        raise IngestError("Scanned PDF could not be rendered for OCR.") from e

    if len(pdf) > max_pages:
        raise IngestError(f"Scanned PDF exceeds the maximum limit of {max_pages} pages (has {len(pdf)}).")

    images = [page.render(scale=2.0).to_pil() for page in pdf]
    if not images:
        raise IngestError("No pages found in this PDF.")

    engine = get_ocr_engine(ocr_engine)
    ocr_res = engine.ocr_document(images)

    text = _tidy(ocr_res.text)
    if not text.strip():
        raise IngestError("No text found in this scanned PDF.")

    previews = [_image_to_data_url(img) for img in images[:3]]  # preview up to first 3 pages
    return IngestResult(
        text=text,
        source="ocr",
        ocr_engine=engine.name,
        low_confidence_words=[w.to_dict() for w in ocr_res.low_confidence_words],
        page_count=len(images),
        seconds=ocr_res.seconds,
        preview_images=previews,
    )


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
