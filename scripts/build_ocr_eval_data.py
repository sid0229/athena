"""Generate synthetic evaluation dataset for OCR benchmarking.

Builds 50 synthetic documents (no real patient data):
- 30 handwritten pages (15 photos, 15 scans)
- 20 printed pages (10 photos, 10 scans)
- Exact ground-truth .txt transcripts
- manifest.json with sample metadata and target drug names.

All content is based on Athena's synthetic demo patients.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "ocr_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Available fonts on macOS
HW_FONTS = [
    "/System/Library/Fonts/Noteworthy.ttc",
    "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",
    "/System/Library/Fonts/Supplemental/ChalkboardSE.ttc",
    "/System/Library/Fonts/MarkerFelt.ttc",
]
# Fallback if any font is missing
HW_FONTS = [f for f in HW_FONTS if os.path.exists(f)]

PR_FONTS = [
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
]
PR_FONTS = [f for f in PR_FONTS if os.path.exists(f)]

# Base medication regimens derived from demo patients
REGIMENS = [
    {
        "title": "Discharge Medication Order",
        "lines": [
            "Rx: Warfarin 5 mg PO daily",
            "Aspirin 81 mg PO daily",
            "Furosemide 40 mg PO daily",
            "Clarithromycin 500 mg PO BID x 5 days",
            "Simvastatin 40 mg PO qhs (HELD)",
        ],
        "drugs": ["warfarin", "aspirin", "furosemide", "clarithromycin", "simvastatin"],
    },
    {
        "title": "Outpatient Prescription",
        "lines": [
            "1. Metoprolol succinate 50 mg PO daily",
            "2. Lasix 80 mg PO once daily in morning",
            "3. Tylenol 500 mg PO q6h prn pain",
            "4. Clarithromycin 500 mg PO BID",
        ],
        "drugs": ["metoprolol", "furosemide", "acetaminophen", "clarithromycin"],
    },
    {
        "title": "Orthopaedic Discharge Orders",
        "lines": [
            "Sertraline 100 mg PO daily",
            "Tramadol 50 mg PO q6h prn moderate pain",
            "Sumatriptan 50 mg PO prn migraine",
            "Ondansetron 4 mg PO q8h prn nausea",
            "Enoxaparin 40 mg SubQ daily x 28 days",
        ],
        "drugs": ["sertraline", "tramadol", "sumatriptan", "ondansetron", "enoxaparin"],
    },
    {
        "title": "Pain & Sedation Plan",
        "lines": [
            "Lorazepam 0.5 mg PO at bedtime prn",
            "Oxycodone 5 mg PO q4h prn severe pain",
            "Omeprazole 20 mg PO daily before breakfast",
            "Sertraline 100 mg PO daily",
        ],
        "drugs": ["lorazepam", "oxycodone", "omeprazole", "sertraline"],
    },
    {
        "title": "Endocrine & Renal Regimen",
        "lines": [
            "Insulin glargine 20 units SubQ at bedtime",
            "Lisinopril 20 mg PO daily",
            "Spironolactone 25 mg PO daily",
            "Potassium chloride 20 mEq PO daily",
            "Amlodipine 5 mg PO daily",
            "Metformin 1000 mg PO BID (DISCONTINUED)",
        ],
        "drugs": ["insulin glargine", "lisinopril", "spironolactone", "potassium chloride", "amlodipine", "metformin"],
    },
    {
        "title": "Clinic Follow-Up Note",
        "lines": [
            "Patient seen for medication review.",
            "Current medications:",
            "- Lisinopril 20 mg PO daily",
            "- Ibuprofen 400 mg PO TID prn joint pain",
            "- Amlodipine 10 mg PO daily",
            "- Metformin was stopped due to renal impairment",
        ],
        "drugs": ["lisinopril", "ibuprofen", "amlodipine", "metformin"],
    },
    {
        "title": "Discharge Medication Reconciliation",
        "lines": [
            "Medications on Discharge:",
            "1. Coumadin 5 mg PO every evening",
            "2. Percocet 5-325 mg PO q6h prn severe pain",
            "3. Acetaminophen 500 mg PO q6h prn fever",
            "4. Guaifenesin 100 mg/5 mL PO q4h prn cough",
            "5. Clarithromycin 500 mg PO BID",
        ],
        "drugs": ["warfarin", "oxycodone", "acetaminophen", "guaifenesin", "clarithromycin"],
    },
    {
        "title": "Cardiology Consultation Note",
        "lines": [
            "Recommend continuing Warfarin 5 mg daily, goal INR 2.0-3.0.",
            "Add Aspirin 81 mg PO daily.",
            "Furosemide 40 mg PO daily for CHF maintenance.",
            "Hold Simvastatin while completing Clarithromycin course.",
        ],
        "drugs": ["warfarin", "aspirin", "furosemide", "simvastatin", "clarithromycin"],
    },
    {
        "title": "Prescription - Dr. Sharma",
        "lines": [
            "Patient: Demo Patient (Synthetic)",
            "Date: 2026-10-12",
            "Rx 1: Sertraline HCl 100 mg tabs #30 Sig: 1 tab PO qd",
            "Rx 2: Tramadol 50 mg tabs #20 Sig: 1 tab PO q6h prn",
            "Rx 3: Ondansetron 4 mg ODT #10 Sig: 1 tab PO q8h prn nausea",
        ],
        "drugs": ["sertraline", "tramadol", "ondansetron"],
    },
    {
        "title": "Nephrology Transfer Summary",
        "lines": [
            "Diagnosis: Type 2 DM, CKD 3b, hypertension.",
            "Discharge Meds:",
            "Insulin glargine 20 units SC qhs",
            "Lisinopril 20 mg PO daily",
            "Amlodipine 5 mg PO daily",
            "Spironolactone 25 mg PO daily",
            "Avoid NSAIDs like Ibuprofen.",
        ],
        "drugs": ["insulin glargine", "lisinopril", "amlodipine", "spironolactone", "ibuprofen"],
    },
]


def create_paper_background(width: int, height: int, is_handwritten: bool, is_scan: bool) -> Image.Image:
    """Create a realistic paper background with slight tint or ruled lines."""
    # Base paper color
    if is_handwritten:
        base_color = (252, 250, 242) if is_scan else (248, 244, 232)
    else:
        base_color = (255, 255, 255) if is_scan else (248, 248, 246)

    img = Image.new("RGB", (width, height), color=base_color)
    draw = ImageDraw.Draw(img)

    # Ruled lines for handwritten prescription pad
    if is_handwritten:
        line_color = (210, 225, 240) if is_scan else (195, 210, 230)
        margin_x = 120
        # Margin line
        draw.line([(margin_x, 0), (margin_x, height)], fill=(240, 190, 190), width=2)
        # Horizontal lines
        for y in range(160, height - 60, 48):
            draw.line([(0, y), (width, y)], fill=line_color, width=1)

    return img


def apply_photo_effects(img: Image.Image, seed: int) -> Image.Image:
    """Simulate realistic mobile photo capture: lighting gradient, subtle shadow, lens tilt."""
    rng = random.Random(seed)
    w, h = img.size

    # 1. Lighting gradient (vignette / uneven ambient light)
    gradient = Image.new("L", (w, h), color=255)
    g_draw = ImageDraw.Draw(gradient)
    corner_dark = rng.randint(180, 220)
    for y in range(h):
        factor = y / h
        val = int(255 - (255 - corner_dark) * factor * 0.7)
        g_draw.line([(0, y), (w, y)], fill=val)

    # Apply lighting gradient to image
    img = ImageEnhance.Brightness(img).enhance(0.98)
    shadow_mask = gradient.filter(ImageFilter.GaussianBlur(radius=40))

    # Convert to numpy-free Pillow blend
    shadow_img = Image.new("RGB", (w, h), color=(rng.randint(30, 50), rng.randint(30, 50), rng.randint(35, 55)))
    img = Image.composite(img, Image.blend(img, shadow_img, 0.08), shadow_mask)

    # 2. Subtle rotation/tilt (-1.2 to +1.2 degrees)
    angle = rng.uniform(-1.2, 1.2)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(240, 238, 232))

    # 3. Slight camera blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.45))
    return img


def apply_scan_effects(img: Image.Image, seed: int) -> Image.Image:
    """Simulate flatbed scanner capture: high contrast, slight 0.3-deg deskew, subtle scanner noise."""
    rng = random.Random(seed)
    angle = rng.uniform(-0.4, 0.4)
    img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    # Slight contrast boost
    img = ImageEnhance.Contrast(img).enhance(1.05)
    img = ImageEnhance.Sharpness(img).enhance(1.1)
    return img


def render_document(
    regimen: dict,
    is_handwritten: bool,
    is_scan: bool,
    sample_id: str,
    font_path: str,
    seed: int,
) -> tuple[str, Image.Image]:
    """Render document and return (transcript_text, image)."""
    rng = random.Random(seed)
    w, h = 900, 1200
    img = create_paper_background(w, h, is_handwritten, is_scan)
    draw = ImageDraw.Draw(img)

    font_size = rng.randint(24, 30) if is_handwritten else rng.randint(22, 26)
    font = ImageFont.truetype(font_path, size=font_size)
    header_font = ImageFont.truetype(font_path, size=font_size + 6)

    # Ink colors
    if is_handwritten:
        ink_choices = [
            (20, 35, 110),    # Blue ballpoint
            (15, 25, 75),     # Dark blue
            (25, 25, 28),     # Black ink
            (35, 45, 95),     # Faded blue
        ]
        ink = rng.choice(ink_choices)
    else:
        ink = (20, 20, 22) if is_scan else (35, 35, 38)

    # Build text transcript
    full_lines = [regimen["title"], ""] + regimen["lines"]
    transcript = "\n".join(full_lines).strip()

    # Draw text
    start_x = 150 if is_handwritten else 80
    y = 90
    for idx, line in enumerate(full_lines):
        if not line:
            y += 24
            continue
        cur_font = header_font if idx == 0 else font
        # Add slight natural handwriting jitter
        x_jitter = rng.randint(-3, 3) if is_handwritten else 0
        y_jitter = rng.randint(-2, 2) if is_handwritten else 0

        draw.text((start_x + x_jitter, y + y_jitter), line, font=cur_font, fill=ink)
        y += 48 if is_handwritten else 36

    # Post-process
    if is_scan:
        final_img = apply_scan_effects(img, seed)
    else:
        final_img = apply_photo_effects(img, seed)

    return transcript, final_img


def main():
    print(f"Building synthetic OCR dataset in {OUT_DIR}...")
    manifest = []
    sample_idx = 1

    # 1. Handwritten pages: 15 photos, 15 scans (total 30)
    for is_scan, count, mode_str in [(False, 15, "photo"), (True, 15, "scan")]:
        for i in range(1, count + 1):
            sid = f"hw_{mode_str}_{i:02d}"
            regimen = REGIMENS[(sample_idx - 1) % len(REGIMENS)]
            font_path = HW_FONTS[(i - 1) % len(HW_FONTS)]
            transcript, img = render_document(
                regimen=regimen,
                is_handwritten=True,
                is_scan=is_scan,
                sample_id=sid,
                font_path=font_path,
                seed=sample_idx * 101,
            )
            img_path = OUT_DIR / f"{sid}.png"
            txt_path = OUT_DIR / f"{sid}.txt"
            img.save(img_path)
            txt_path.write_text(transcript, encoding="utf-8")

            manifest.append({
                "id": sid,
                "image_file": f"{sid}.png",
                "transcript_file": f"{sid}.txt",
                "doc_type": "handwritten",
                "capture": mode_str,
                "font": Path(font_path).name,
                "expected_drugs": regimen["drugs"],
            })
            sample_idx += 1

    # 2. Printed pages: 10 photos, 10 scans (total 20)
    for is_scan, count, mode_str in [(False, 10, "photo"), (True, 10, "scan")]:
        for i in range(1, count + 1):
            sid = f"pr_{mode_str}_{i:02d}"
            regimen = REGIMENS[(sample_idx - 1) % len(REGIMENS)]
            font_path = PR_FONTS[(i - 1) % len(PR_FONTS)]
            transcript, img = render_document(
                regimen=regimen,
                is_handwritten=False,
                is_scan=is_scan,
                sample_id=sid,
                font_path=font_path,
                seed=sample_idx * 202,
            )
            img_path = OUT_DIR / f"{sid}.png"
            txt_path = OUT_DIR / f"{sid}.txt"
            img.save(img_path)
            txt_path.write_text(transcript, encoding="utf-8")

            manifest.append({
                "id": sid,
                "image_file": f"{sid}.png",
                "transcript_file": f"{sid}.txt",
                "doc_type": "printed",
                "capture": mode_str,
                "font": Path(font_path).name,
                "expected_drugs": regimen["drugs"],
            })
            sample_idx += 1

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} samples:")
    print(f"  - Handwritten: {sum(1 for m in manifest if m['doc_type'] == 'handwritten')}")
    print(f"  - Printed:     {sum(1 for m in manifest if m['doc_type'] == 'printed')}")
    print(f"  - Scans:       {sum(1 for m in manifest if m['capture'] == 'scan')}")
    print(f"  - Photos:      {sum(1 for m in manifest if m['capture'] == 'photo')}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
