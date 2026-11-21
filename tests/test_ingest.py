"""Uploaded notes: .txt, .docx (stdlib) and digital .pdf (pypdf) become plain text; bad files get clear errors."""

import io
import zipfile

import pytest

from athena.ingest import IngestError, extract_text

W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def _docx(body: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f'<w:document {W}><w:body>{body}</w:body></w:document>')
    return buf.getvalue()


def _pdf(lines: list[str]) -> bytes:
    """Minimal one-page PDF with a Helvetica text layer."""
    content = "BT /F1 12 Tf 72 720 Td " + " ".join(f"({t}) Tj 0 -16 Td" for t in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offs).encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_txt_with_bom():
    assert extract_text("n.txt", "﻿Metoprolol 25 mg\r\n".encode()) == "Metoprolol 25 mg\n"


def test_docx_paragraphs_and_table_rows_in_order():
    body = ('<w:p><w:r><w:t>Discharge Medications:</w:t></w:r></w:p>'
            '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Warfarin</w:t></w:r></w:p></w:tc>'
            '<w:tc><w:p><w:r><w:t>5 mg daily</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
            '<w:p><w:r><w:t>Aspirin</w:t><w:tab/><w:t>81 mg</w:t></w:r></w:p>')
    assert extract_text("n.docx", _docx(body)) == "Discharge Medications:\nWarfarin\t5 mg daily\nAspirin\t81 mg\n"


def test_digital_pdf():
    text = extract_text("n.pdf", _pdf(["Discharge Medications:", "Lisinopril 10 mg daily"]))
    assert "Discharge Medications:" in text and "Lisinopril 10 mg daily" in text


def test_scanned_pdf_and_bad_files_get_clear_errors():
    with pytest.raises(IngestError, match="No text found in this scanned PDF"):
        extract_text("scan.pdf", _pdf([]))
    with pytest.raises(IngestError, match="not a readable .docx"):
        extract_text("old.docx", b"\xd0\xcf\x11\xe0 legacy doc")
    with pytest.raises(IngestError, match="could not be read"):
        extract_text("x.pdf", b"not a pdf")
    with pytest.raises(IngestError, match="Unsupported"):
        extract_text("x.rtf", b"{\\rtf1}")


def test_extract_endpoint(tmp_path, monkeypatch):
    import importlib
    import sys

    from fastapi.testclient import TestClient

    from athena.config import ROOT
    monkeypatch.setenv("ATHENA_AUDIT_DB", str(tmp_path / "audit.db"))
    sys.path.insert(0, str(ROOT / "app"))
    c = TestClient(importlib.import_module("server").app)
    r = c.post("/api/extract?filename=n.pdf", content=_pdf(["Simvastatin 40 mg"]))
    assert r.status_code == 200 and "Simvastatin 40 mg" in r.json()["text"]
    r = c.post("/api/extract?filename=n.pdf", content=b"junk")
    assert r.status_code == 400 and "could not be read" in r.json()["detail"]
