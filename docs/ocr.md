# Athena OCR — Reading Handwritten Prescriptions and Scanned Clinical Notes

Athena allows clinical pharmacists and reviewers to upload photos and scans of handwritten prescriptions, clinic orders, and scanned discharge summaries. Athena converts the document into text, presents it to the reviewer in a **Check-the-text** verification step alongside the original image, and then executes its standard neuro-symbolic reconciliation and interaction analysis on the confirmed text.

> [!IMPORTANT]
> **Safety Principle**: OCR only produces text and confidence metadata; **it never makes clinical decisions**. Every finding and drug interaction flag originates strictly from verified medical databases (DrugBank, DDInter) matched against the reviewer-confirmed medication text.

---

## 1. Supported OCR Engines & Licensing

Athena supports three local OCR engines, strictly adhering to permissive licensing (zero GPL or AGPL dependencies):

| Engine | Primary Use Case | License | Footprint / Requirements | Default |
| :--- | :--- | :--- | :--- | :--- |
| **Apple Vision (`vision`)** | Handwritten prescriptions & scans on macOS | **MIT** (`ocrmac`) | Built into macOS Neural Engine / Metal; 0 MB download; ~74 ms/page | **Yes** (Default) |
| **Qwen2.5-VL 3B (`qwen_vl`)** | Hard cursive handwriting & degraded photos | **Qwen Research License** | Local Ollama (`qwen2.5vl:3b`); ~3.2 GB disk; ~5.4 s/page | Optional |
| **Tesseract (`tesseract`)** | Printed documents on non-Mac systems | **Apache-2.0** (`pytesseract`) | System Tesseract binary; portable Linux/Windows fallback | Optional fallback |

### Licensing Compliance
- **No GPL / AGPL libraries**: Tools such as Surya or PyMuPDF (AGPL) were strictly rejected because Athena is an open, publicly distributable clinical repository.
- **PDF Rendering**: Scanned PDFs are rendered page-by-page using `pypdfium2` (licensed under Apache-2.0 and 3-clause BSD), ensuring permissive compliance.
- **Qwen Research License**: Qwen2.5-VL is used locally via Ollama for non-commercial research and academic validation in line with the Qwen Community License.

Engine choice is configured in `configs/default.yaml`:
```yaml
ocr:
  engine: vision                   # vision (Apple Vision, default) | qwen_vl (Ollama) | tesseract (portable fallback)
  low_confidence_threshold: 0.70   # words below this are highlighted for reviewer verification
  max_pages: 10                    # maximum pages allowed for document uploads
  qwen_vl_model: qwen2.5vl:3b
```

---

## 2. Ingest Pipeline & Document Handling

The document ingest pipeline (`athena/ingest.py` and `/api/extract`) accepts:
- Plain text (`.txt`)
- Word documents (`.docx`, standard library XML extraction)
- Digital PDFs (`.pdf` with digital text layer)
- Scanned PDFs (`.pdf` without text layer)
- Photos & scans (`.png`, `.jpg`, `.jpeg`, `.heic`)

### Digital PDF vs Scanned PDF Bypass
When a PDF is uploaded:
1. `pypdf.PdfReader` inspects the document. If a non-empty text layer exists, Athena extracts digital text directly and **skips OCR entirely**, eliminating latency and potential OCR distortion.
2. If the PDF lacks a text layer (a scanned document or photographed page), Athena automatically switches to the scanned-PDF pipeline:
   - Renders each page via `pypdfium2` at high resolution (`scale=2.0`, ~150–200 DPI).
   - Runs the selected OCR engine across rendered pages.
   - Extracts verbatim text and word-level character bounding boxes and confidences.

### Guardrails
- **File size limit**: Enforces a strict 20 MB cap (`MAX_BYTES = 20 * 1024 * 1024`).
- **Page limit**: Rejects documents exceeding 10 pages (`DEFAULT_MAX_PAGES = 10`), preventing accidental denial-of-service on local hardware.

---

## 3. Check-the-Text Step in the UI

To ensure clinical safety, OCR documents undergo an interactive verification step before analysis:

1. **Safety Banner**: Displays an alert:
   > *"Text read by OCR. Check it against the original before analysis."*
2. **Side-by-Side Comparison**:
   - Left column displays the original uploaded prescription/note image preview.
   - Right column displays an editable text area containing the OCR transcription.
3. **Low-Confidence Word Highlighting**:
   - Any word whose OCR confidence falls below `low_confidence_threshold` (0.70) is flagged with its character position and confidence score (e.g. `Tylend (54%)`), prompting the reviewer to verify ambiguous drug names or dosages.
4. **Provenance Metadata**:
   - Reports generated from OCR input record `meta.input: "ocr"` and `meta.ocr_engine: "vision"`.
   - The OCR badge is permanently displayed on the Med Review dashboard header, in run details, and on the printable sign-off summary (`CaseSummary.tsx`).

---

## 4. Evaluation Benchmark

### Test Dataset (`data/ocr_eval/`)
To evaluate OCR performance without risking patient data, we constructed a synthetic benchmark dataset of **50 documents** derived directly from Athena's demo patients (Warfarin, Aspirin, Lasix, Clarithromycin, Percocet, Simvastatin, Sertraline, Tramadol, Lisinopril, Insulin glargine, etc.):
- **30 Handwritten Pages**: 15 phone photos (`hw_photo_01`–`15`) and 15 scans (`hw_scan_01`–`15`), utilizing diverse handwriting styles, ballpoint/faded pen inks, ruled prescription pads, and natural perspective/tilt.
- **20 Printed Pages**: 10 phone photos (`pr_photo_01`–`10`) and 10 scans (`pr_scan_01`–`10`) with scanner noise and mobile lighting gradients.
- **Ground Truth**: Each sample has a character-exact ground-truth `.txt` transcript and JSON metadata entry in `manifest.json`.

### Benchmark Results

All 50 samples were scored across Character Error Rate (CER), Word Error Rate (WER), Drug Name Recall (surface matching), Downstream Extracted Medication Recall (Athena pipeline), Downstream Critical/Review Alert Recall, Latency, and Peak Memory:

```
==========================================================================================
ATHENA OCR ENGINE BENCHMARK RESULTS
==========================================================================================
Engine       | CER    | WER    | Drug Rec  | Med Rec  | Alert Rec  | Sec/Page  | Peak MB 
------------------------------------------------------------------------------------------
vision       | 0.014  | 0.108  | 97.5    % | 97.5   % | 96.0     % | 0.074     | 2299.4  
qwen_vl      | 0.005  | 0.063  | 100.0   % | 100.0  % | 100.0    % | 5.422     | 2299.4  
tesseract    | 0.044  | 0.167  | 88.5    % | 88.5   % | 87.8     % | 0.128     | 2299.4  
==========================================================================================
```

#### Slices: Handwritten vs Printed
```
---------------------------------------------------------------------------
Engine       | HW CER  | HW DrugRec  | PR CER  | PR DrugRec 
---------------------------------------------------------------------------
vision       | 0.018   | 96.3      % | 0.007   | 99.2      %
qwen_vl      | 0.005   | 100.0     % | 0.006   | 100.0     %
tesseract    | 0.067   | 80.8      % | 0.010   | 100.0     %
---------------------------------------------------------------------------
```

#### Slices: Mobile Photo vs Flatbed Scan
```
---------------------------------------------------------------------------
Engine       | Photo CER  | Photo DrugRec  | Scan CER  | Scan DrugRec 
---------------------------------------------------------------------------
vision       | 0.014      | 97.2         % | 0.014     | 97.7        %
qwen_vl      | 0.005      | 100.0        % | 0.006     | 100.0       %
tesseract    | 0.033      | 90.1         % | 0.056     | 86.8        %
---------------------------------------------------------------------------
```

---

## 5. Engine Selection Rationale

**Apple Vision (`vision`) is chosen as the default engine:**
1. **Downstream Drug Recall**: Delivers **97.5% drug recall** and **96.0% actionable alert recall** across all test pages (99.2% on printed, 96.3% on cursive handwriting).
2. **Speed**: At **0.074 seconds per page** (74 ms), Vision is **73× faster** than Qwen2.5-VL (5.42 s/page) and 1.7× faster than Tesseract (128 ms/page).
3. **Zero Overhead**: Built into macOS Neural Engine/Metal; requires no external model downloads, keeping Athena responsive on 8 GB unified memory devices.

**When to choose Qwen2.5-VL (`qwen_vl`)**:
For severe cursive handwriting or damaged prescription slips where traditional OCR degrades, Qwen2.5-VL achieved **100% drug recall** and **0.5% CER**. Reviewers with difficult handwritten slips can switch to `ocr.engine: qwen_vl` in `configs/default.yaml`.

**When to choose Tesseract (`tesseract`)**:
Tesseract achieved **100.0% drug recall** and **1.0% CER** on printed notes. It serves as an open-source, portable fallback for non-macOS environments (Linux/Windows).

---

## 6. Observed Failure Modes & Error Analysis

Inspection of OCR error transcripts revealed systematic failure patterns:

1. **Letter Substitution in Brand/Generic Names**:
   - `Tylenol` was misread as `Tylend` in script cursive (`hw_photo_02`).
   - `Tramadol` was misread as `Tramadal` (`hw_photo_03`).
   - `Amlodipine` was misread as `Arnlodipine` when the "m" had an open arch.
2. **Medical Abbreviations & Numbers**:
   - `q6h` misread as `96h` or `gou`.
   - `prn` misread as `pru`.
   - `SubQ` misread as `Sub 2`.
   - `mg` vs `ng`: Faint loops or cursive descenders occasionally cause "mg" to resemble "ng" (nanograms vs milligrams).
3. **Decimals and Punctuation**:
   - Leading or trailing decimal points (e.g. `5.0 mg` vs `50 mg` or `.5 mg` vs `5 mg`) can be obscured by paper noise or ruled lines.

### Mitigation in Athena
- Athena's **Normalizer** uses fuzzy lexicon matching (rapidfuzz threshold 92 with salt stripping) which successfully resolves many minor OCR typos (e.g. `Tramadal` -> `tramadol`).
- The **Check-the-text editable UI** and low-confidence word flags immediately surface uncertain tokens to the pharmacist before analysis is executed.
- The **Grounding Guard** ensures that invented or ungrounded drug names cannot be hallucinated into findings.
