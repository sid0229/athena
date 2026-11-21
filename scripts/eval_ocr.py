"""Benchmark OCR engines on the synthetic evaluation dataset.

Evaluates:
- CER (Character Error Rate) via jiwer
- WER (Word Error Rate) via jiwer
- Drug-name surface recall in raw text
- Downstream medication extraction recall (Athena normalizer/pipeline)
- Downstream Critical/Review interaction alert recall
- Latency (seconds per page) and peak memory (MB)

Breaks results down by:
- Overall
- Document type (Handwritten vs Printed)
- Capture mode (Photo vs Scan)

Saves summary and detailed results to data/ocr_eval/eval_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jiwer
from PIL import Image

from athena.extraction.pipeline import MedicationExtractor
from athena.fusion.report import analyze
from athena.ocr.engine import (
    BaseOCREngine,
    QwenVLEngine,
    TesseractEngine,
    VisionEngine,
)
from athena.verification.checker import InteractionChecker

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "data" / "ocr_eval"
MANIFEST_PATH = EVAL_DIR / "manifest.json"


# Drug name synonyms for text matching
DRUG_SYNONYMS = {
    "warfarin": ["warfarin", "coumadin"],
    "aspirin": ["aspirin", "asa", "acetylsalicylic acid"],
    "furosemide": ["furosemide", "lasix"],
    "clarithromycin": ["clarithromycin", "biaxin"],
    "simvastatin": ["simvastatin", "zocor"],
    "metoprolol": ["metoprolol", "toprol", "lopressor"],
    "acetaminophen": ["acetaminophen", "tylenol", "paracetamol", "apap"],
    "sertraline": ["sertraline", "zoloft"],
    "tramadol": ["tramadol", "ultram"],
    "sumatriptan": ["sumatriptan", "imitrex"],
    "ondansetron": ["ondansetron", "zofran"],
    "enoxaparin": ["enoxaparin", "lovenox"],
    "lorazepam": ["lorazepam", "ativan"],
    "oxycodone": ["oxycodone", "percocet", "oxycontin", "roxicodone"],
    "omeprazole": ["omeprazole", "prilosec"],
    "insulin glargine": ["insulin glargine", "glargine", "lantus", "basaglar"],
    "lisinopril": ["lisinopril", "prinivil", "zestril"],
    "spironolactone": ["spironolactone", "aldactone"],
    "potassium chloride": ["potassium chloride", "kcl", "k-dur", "potassium"],
    "amlodipine": ["amlodipine", "norvasc"],
    "metformin": ["metformin", "glucophage"],
    "ibuprofen": ["ibuprofen", "motrin", "advil"],
    "guaifenesin": ["guaifenesin", "mucinex", "robitussin"],
}


def _check_drug_in_text(drug: str, text: str) -> bool:
    t_lower = text.lower()
    aliases = DRUG_SYNONYMS.get(drug.lower(), [drug.lower()])
    return any(a in t_lower for a in aliases)


def _get_peak_memory_mb() -> float:
    # ru_maxrss is in bytes on macOS, kilobytes on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


def evaluate_engine(
    engine: BaseOCREngine,
    samples: list[dict],
    checker: InteractionChecker,
    extractor: MedicationExtractor,
    gt_cache: dict[str, dict],
) -> dict[str, Any]:
    print(f"\nEvaluating engine: {engine.name} ({len(samples)} samples)...")
    results = []
    t_start = time.perf_counter()

    for idx, s in enumerate(samples, start=1):
        img_path = EVAL_DIR / s["image_file"]
        gt_data = gt_cache[s["id"]]
        gt_text = gt_data["text"]
        img = Image.open(img_path)

        t0 = time.perf_counter()
        try:
            page_res = engine.ocr_page(img, page_num=1)
            pred_text = page_res.text.strip()
        except Exception as e:
            print(f"  Error on {s['id']}: {e}")
            pred_text = ""
        elapsed = time.perf_counter() - t0

        # CER & WER
        if not pred_text:
            cer = 1.0
            wer = 1.0
        else:
            try:
                cer = float(jiwer.cer(gt_text, pred_text))
                wer = float(jiwer.wer(gt_text, pred_text))
            except Exception:
                cer, wer = 1.0, 1.0

        # Drug name surface recall
        expected_drugs = s["expected_drugs"]
        found_drugs = [d for d in expected_drugs if _check_drug_in_text(d, pred_text)]
        drug_recall = len(found_drugs) / len(expected_drugs) if expected_drugs else 1.0

        # Downstream extraction & alerts
        gt_med_names = gt_data["meds"]
        gt_actionable_alerts = gt_data["alerts"]

        # Analyze OCR text with fast rules extractor
        if pred_text:
            ocr_report = analyze(pred_text, extractor=extractor, checker=checker).to_dict()
            ocr_med_names = {m["name"].lower() for m in ocr_report["medications"]}
            ocr_actionable_alerts = {
                (f["type"], f["title"])
                for f in ocr_report["findings"]
                if f["tier"] in ("critical", "review")
            }
        else:
            ocr_med_names = set()
            ocr_actionable_alerts = set()

        # Med recall
        med_recall = (
            len(gt_med_names & ocr_med_names) / len(gt_med_names)
            if gt_med_names
            else 1.0
        )

        # Alert recall
        if gt_actionable_alerts:
            alert_recall = len(gt_actionable_alerts & ocr_actionable_alerts) / len(gt_actionable_alerts)
        else:
            alert_recall = 1.0

        sample_res = {
            "id": s["id"],
            "doc_type": s["doc_type"],
            "capture": s["capture"],
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "drug_recall": round(drug_recall, 4),
            "med_recall": round(med_recall, 4),
            "alert_recall": round(alert_recall, 4),
            "seconds": round(elapsed, 3),
            "expected_drugs": expected_drugs,
            "found_drugs": found_drugs,
            "gt_meds_count": len(gt_med_names),
            "ocr_meds_count": len(ocr_med_names),
            "gt_alerts_count": len(gt_actionable_alerts),
            "ocr_alerts_count": len(ocr_actionable_alerts),
        }
        results.append(sample_res)

        if idx % 10 == 0 or idx == len(samples):
            print(f"  Processed {idx}/{len(samples)} (last: {elapsed:.2f}s, CER: {cer:.2f}, DrugRec: {drug_recall:.2f})")

    total_time = time.perf_counter() - t_start
    peak_mem = _get_peak_memory_mb()

    # Aggregate slices
    def aggregate(subset: list[dict]) -> dict[str, float]:
        if not subset:
            return {}
        n = len(subset)
        return {
            "count": n,
            "cer": round(sum(x["cer"] for x in subset) / n, 4),
            "wer": round(sum(x["wer"] for x in subset) / n, 4),
            "drug_recall": round(sum(x["drug_recall"] for x in subset) / n, 4),
            "med_recall": round(sum(x["med_recall"] for x in subset) / n, 4),
            "alert_recall": round(sum(x["alert_recall"] for x in subset) / n, 4),
            "sec_per_page": round(sum(x["seconds"] for x in subset) / n, 3),
        }

    hw = [x for x in results if x["doc_type"] == "handwritten"]
    pr = [x for x in results if x["doc_type"] == "printed"]
    photo = [x for x in results if x["capture"] == "photo"]
    scan = [x for x in results if x["capture"] == "scan"]

    summary = {
        "engine": engine.name,
        "total_samples": len(samples),
        "total_seconds": round(total_time, 2),
        "sec_per_page": round(total_time / len(samples), 3) if samples else 0,
        "peak_memory_mb": round(peak_mem, 1),
        "overall": aggregate(results),
        "by_type": {
            "handwritten": aggregate(hw),
            "printed": aggregate(pr),
        },
        "by_capture": {
            "photo": aggregate(photo),
            "scan": aggregate(scan),
        },
        "detailed": results,
    }
    return summary


def print_summary_table(summaries: list[dict]):
    print("\n" + "=" * 90)
    print("ATHENA OCR ENGINE BENCHMARK RESULTS")
    print("=" * 90)
    print(f"{'Engine':<12} | {'CER':<6} | {'WER':<6} | {'Drug Rec':<9} | {'Med Rec':<8} | {'Alert Rec':<10} | {'Sec/Page':<9} | {'Peak MB':<8}")
    print("-" * 90)
    for s in summaries:
        ov = s["overall"]
        print(
            f"{s['engine']:<12} | "
            f"{ov['cer']:<6.3f} | "
            f"{ov['wer']:<6.3f} | "
            f"{ov['drug_recall']*100:<8.1f}% | "
            f"{ov['med_recall']*100:<7.1f}% | "
            f"{ov['alert_recall']*100:<9.1f}% | "
            f"{ov['sec_per_page']:<9.3f} | "
            f"{s['peak_memory_mb']:<8.1f}"
        )
    print("=" * 90)

    print("\nBREAKDOWN: HANDWRITTEN vs PRINTED")
    print("-" * 75)
    print(f"{'Engine':<12} | {'HW CER':<7} | {'HW DrugRec':<11} | {'PR CER':<7} | {'PR DrugRec':<11}")
    print("-" * 75)
    for s in summaries:
        hw = s["by_type"]["handwritten"]
        pr = s["by_type"]["printed"]
        print(
            f"{s['engine']:<12} | "
            f"{hw['cer']:<7.3f} | "
            f"{hw['drug_recall']*100:<10.1f}% | "
            f"{pr['cer']:<7.3f} | "
            f"{pr['drug_recall']*100:<10.1f}%"
        )
    print("-" * 75)

    print("\nBREAKDOWN: PHOTO vs SCAN")
    print("-" * 75)
    print(f"{'Engine':<12} | {'Photo CER':<10} | {'Photo DrugRec':<14} | {'Scan CER':<9} | {'Scan DrugRec':<13}")
    print("-" * 75)
    for s in summaries:
        ph = s["by_capture"]["photo"]
        sc = s["by_capture"]["scan"]
        print(
            f"{s['engine']:<12} | "
            f"{ph['cer']:<10.3f} | "
            f"{ph['drug_recall']*100:<13.1f}% | "
            f"{sc['cer']:<9.3f} | "
            f"{sc['drug_recall']*100:<12.1f}%"
        )
    print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR engines on Athena synthetic eval set")
    parser.add_argument("--engines", default="vision,qwen_vl,tesseract", help="Comma-separated engine list")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Run scripts/build_ocr_eval_data.py first.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if args.limit:
        manifest = manifest[:args.limit]

    engine_names = [e.strip() for e in args.engines.split(",") if e.strip()]
    checker = InteractionChecker()
    extractor = MedicationExtractor("rules", normalizer=checker.normalizer)
    summaries = []

    print("Pre-computing ground truth reports for downstream comparison...")
    gt_cache: dict[str, dict] = {}
    for s in manifest:
        txt_path = EVAL_DIR / s["transcript_file"]
        gt_text = txt_path.read_text(encoding="utf-8").strip()
        gt_rep = analyze(gt_text, extractor=extractor, checker=checker).to_dict()
        gt_cache[s["id"]] = {
            "text": gt_text,
            "meds": {m["name"].lower() for m in gt_rep["medications"]},
            "alerts": {(f["type"], f["title"]) for f in gt_rep["findings"] if f["tier"] in ("critical", "review")},
        }

    for name in engine_names:
        if name == "vision":
            eng = VisionEngine()
        elif name == "qwen_vl":
            eng = QwenVLEngine()
        elif name == "tesseract":
            eng = TesseractEngine()
        else:
            print(f"Unknown engine '{name}', skipping.")
            continue

        if not eng.is_available():
            print(f"Engine '{name}' is not available on this system. Skipping.")
            continue

        res = evaluate_engine(eng, manifest, checker, extractor, gt_cache)
        summaries.append(res)

    if summaries:
        print_summary_table(summaries)
        out_json = EVAL_DIR / "eval_results.json"
        out_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nSaved detailed evaluation results to {out_json}")


if __name__ == "__main__":
    main()
