"""Plain text from uploaded notes: .txt, .docx (standard library only) and digital .pdf (pypdf).

Scanned PDFs have no text layer; they are rejected with a clear message (OCR is not supported yet).
The text is returned for the reviewer to check before analysis; nothing leaves the device.
"""

from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree

MAX_BYTES = 20 * 1024 * 1024          # upload size limit
MAX_XML_BYTES = 50 * 1024 * 1024      # uncompressed document.xml limit (zip bombs)
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class IngestError(ValueError):
    """The file cannot be turned into note text; the message is shown to the reviewer."""


def extract_text(name: str, data: bytes) -> str:
    if len(data) > MAX_BYTES:
        raise IngestError("File is larger than 20 MB.")
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    if ext == "txt":
        text = _txt(data)
    elif ext == "docx":
        text = _docx(data)
    elif ext == "pdf":
        text = _pdf(data)
    else:
        raise IngestError("Unsupported file type: upload a .txt, .docx or .pdf file.")
    text = _tidy(text)
    if not text.strip():
        raise IngestError("No text found in this file.")
    return text


def _txt(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _docx(data: bytes) -> str:
    """Body paragraphs and table rows in document order (a table row becomes one tab-separated line)."""
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


def _para(p) -> str:
    out = []
    for el in p.iter():
        if el.tag == f"{_W}t" and el.text:
            out.append(el.text)
        elif el.tag == f"{_W}tab":
            out.append("\t")
        elif el.tag in (f"{_W}br", f"{_W}cr"):
            out.append("\n")
    return "".join(out)


def _pdf(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestError("This PDF is password-protected.")
        pages = [p.extract_text() or "" for p in reader.pages]
    except (PdfReadError, ValueError, KeyError) as e:
        if isinstance(e, IngestError):
            raise
        raise IngestError("This PDF could not be read.")
    if not "".join(pages).strip():
        raise IngestError("This PDF has no text layer (a scan or photo). Scanned documents are not supported yet.")
    return "\n\n".join(pages)


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
