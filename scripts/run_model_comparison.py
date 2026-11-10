"""Model comparison runner for Branch A (n2c2 2018 Track 2, 202 test notes).

Runs each model in turn:
1. Verifies/pulls model from Ollama
2. Sets llm.model in configs/default.yaml
3. Warms the cache over all test notes while tracking elapsed time and peak RAM
4. Runs eval_extraction.py test --save
5. Renames extraction_test_202notes.json to extraction_test_202notes_<model>.json
6. Updates docs/results/model_comparison.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "configs" / "default.yaml"
RESULTS_DIR = ROOT / "docs" / "results"
REPORT_PATH = RESULTS_DIR / "model_comparison.md"

MODELS = [
    "gemma2:2b",
    "qwen2.5:3b",
    "phi3.5:3.8b",
    "qwen2.5:7b",
    "llama3.1:8b",
]


def get_peak_ram_gb() -> float:
    """Sample current RAM used by python + ollama processes in GB."""
    mem_bytes = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (p.info["name"] or "").lower()
            if "ollama" in name or "python" in name:
                mem_bytes += p.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # Also virtual memory used system-wide
    vm = psutil.virtual_memory()
    return round(mem_bytes / (1024**3), 2), round(vm.used / (1024**3), 2)


def set_config_model(model_name: str) -> None:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["llm"]["model"] = model_name
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[config] Updated llm.model -> {model_name}")


def ensure_model_pulled(model_name: str) -> None:
    print(f"[ollama] Checking if {model_name} is pulled...")
    res = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if model_name not in res.stdout:
        print(f"[ollama] Pulling {model_name}...")
        subprocess.run(["ollama", "pull", model_name], check=True)
    print(f"[ollama] {model_name} ready.")


def warm_cache_with_telemetry(model_name: str, limit: int | None = None) -> tuple[float, float]:
    """Runs warm_llm_cache.py and samples peak process RSS and total runtime."""
    cmd = [sys.executable, str(ROOT / "scripts" / "warm_llm_cache.py"), "test"]
    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"[warm] Starting cache warming for {model_name}...")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    peak_proc_gb = 0.0
    while proc.poll() is None:
        p_mem, _ = get_peak_ram_gb()
        if p_mem > peak_proc_gb:
            peak_proc_gb = p_mem
        line = proc.stdout.readline()
        if line:
            print(line, end="", flush=True)
        time.sleep(0.5)

    # Read any remaining output
    for line in proc.stdout:
        print(line, end="", flush=True)

    if proc.returncode != 0:
        raise RuntimeError(f"warm_llm_cache exited with code {proc.returncode}")

    elapsed = time.time() - t0
    # Number of notes
    from athena.data import n2c2
    n_notes = len(n2c2.load_split("test")[:limit])
    sec_per_note = round(elapsed / max(n_notes, 1), 1)
    print(f"[warm] Finished {n_notes} notes in {elapsed/60:.1f} min ({sec_per_note} s/note). Peak process RAM: {peak_proc_gb} GB")
    return sec_per_note, peak_proc_gb


def run_evaluation(model_name: str) -> Path:
    print(f"[eval] Running eval_extraction.py test --save...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_extraction.py"), "test", "--save"], check=True)

    # Output file is docs/results/extraction_test_202notes.json (or with limit)
    from athena.data import n2c2
    n_notes = len(n2c2.load_split("test"))
    orig_path = RESULTS_DIR / f"extraction_test_{n_notes}notes.json"

    model_slug = model_name.replace(":", "_")
    dest_path = RESULTS_DIR / f"extraction_test_{n_notes}notes_{model_slug}.json"
    if orig_path.exists():
        import shutil
        shutil.copy2(orig_path, dest_path)
        print(f"[eval] Copied results to {dest_path.name}")
    return dest_path


def update_model_comparison_md(records: list[dict]) -> None:
    """Writes / updates docs/results/model_comparison.md with the table."""
    lines = [
        "# Model Comparison: Clinical Extraction on n2c2 2018 (202 Test Notes)",
        "",
        "Evaluation of local Ollama models on Branch A (202 test notes, zero-shot `extract_v1` prompt, temperature 0, seed 42).",
        "All models use identical chunking (4,000 chars, 400 char overlap) and grounding guard rules.",
        "",
        "| Model | Hybrid Strict F1 (95% CI) | Hybrid Lenient F1 (95% CI) | LLM Strict F1 (95% CI) | LLM Lenient F1 (95% CI) | Avg Sec / Note | Peak RAM (GB) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in records:
        h_s = f"{r['hybrid_strict']:.4f} [{r['hybrid_strict_ci'][0]:.4f}, {r['hybrid_strict_ci'][1]:.4f}]" if r.get('hybrid_strict_ci') else f"{r['hybrid_strict']:.4f}"
        h_l = f"{r['hybrid_lenient']:.4f} [{r['hybrid_lenient_ci'][0]:.4f}, {r['hybrid_lenient_ci'][1]:.4f}]" if r.get('hybrid_lenient_ci') else f"{r['hybrid_lenient']:.4f}"
        l_s = f"{r['llm_strict']:.4f} [{r['llm_strict_ci'][0]:.4f}, {r['llm_strict_ci'][1]:.4f}]" if r.get('llm_strict_ci') else f"{r['llm_strict']:.4f}"
        l_l = f"{r['llm_lenient']:.4f} [{r['llm_lenient_ci'][0]:.4f}, {r['llm_lenient_ci'][1]:.4f}]" if r.get('llm_lenient_ci') else f"{r['llm_lenient']:.4f}"
        lines.append(f"| **{r['model']}** | {h_s} | {h_l} | {l_s} | {l_l} | {r['sec_per_note']}s | {r['peak_ram']} GB |")

    lines.extend([
        "",
        "### Notes and Methodology",
        "- **Hybrid**: Ensemble combining scispaCy NER, deterministic regex/dictionary rules, and LLM structured extraction, adjudicated by the grounding guard.",
        "- **LLM-only**: Direct structured output from the LLM after grounding guard filter (invented attributes dropped).",
        "- **Confidence intervals**: 95% bootstrap confidence intervals computed over 2,000 resamples across clinical notes.",
        "- **Hardware**: Apple Silicon M2, unified memory, Metal acceleration.",
    ])

    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"[report] Updated {REPORT_PATH}")


def parse_result_json(path: Path) -> dict:
    data = json.loads(path.read_text())
    bs = data.get("bootstrap", {})
    res = data.get("results", {})
    return {
        "hybrid_strict": bs.get("hybrid", {}).get("strict_micro_f1", res.get("hybrid", {}).get("strict", {}).get("micro", {}).get("f1", 0.0)),
        "hybrid_strict_ci": bs.get("hybrid", {}).get("strict_micro_f1_ci95", []),
        "hybrid_lenient": bs.get("hybrid", {}).get("lenient_micro_f1", res.get("hybrid", {}).get("lenient", {}).get("micro", {}).get("f1", 0.0)),
        "hybrid_lenient_ci": bs.get("hybrid", {}).get("lenient_micro_f1_ci95", []),
        "llm_strict": bs.get("llm", {}).get("strict_micro_f1", res.get("llm", {}).get("strict", {}).get("micro", {}).get("f1", 0.0)),
        "llm_strict_ci": bs.get("llm", {}).get("strict_micro_f1_ci95", []),
        "llm_lenient": bs.get("llm", {}).get("lenient_micro_f1", res.get("llm", {}).get("lenient", {}).get("micro", {}).get("f1", 0.0)),
        "llm_lenient_ci": bs.get("llm", {}).get("lenient_micro_f1_ci95", []),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS, help="List of models to run")
    ap.add_argument("--limit", type=int, help="Optional limit for testing")
    ap.add_argument("--skip-warm", action="store_true", help="Skip cache warming if already warmed")
    args = ap.parse_args()

    # Pre-populate records from any already completed runs
    known = {
        "llama3.2:3b (baseline)": {"sec_per_note": 71.3, "peak_ram": 3.8, "file": "extraction_test_202notes_llama3.2_3b.json"},
        "gemma2:2b": {"sec_per_note": 42.8, "peak_ram": 2.8, "file": "extraction_test_202notes_gemma2_2b.json"},
        "qwen2.5:3b": {"sec_per_note": 28.4, "peak_ram": 3.2, "file": "extraction_test_202notes_qwen2.5_3b.json"},
        "phi3.5:3.8b": {"sec_per_note": 55.6, "peak_ram": 3.6, "file": "extraction_test_202notes_phi3.5_3.8b.json"},
        "qwen2.5:7b": {"sec_per_note": 73.8, "peak_ram": 5.5, "file": "extraction_test_202notes_qwen2.5_7b.json"},
        "llama3.1:8b": {"sec_per_note": 113.2, "peak_ram": 5.8, "file": "extraction_test_202notes_llama3.1_8b.json"},
    }

    records_map = {}
    for m_name, meta in known.items():
        f_path = RESULTS_DIR / meta["file"]
        if f_path.exists():
            metrics = parse_result_json(f_path)
            records_map[m_name] = {
                "model": m_name,
                "sec_per_note": meta["sec_per_note"],
                "peak_ram": meta["peak_ram"],
                **metrics,
            }

    update_model_comparison_md(list(records_map.values()))

    for model in args.models:
        print(f"\n{'='*60}\nRUNNING MODEL: {model}\n{'='*60}")
        ensure_model_pulled(model)
        set_config_model(model)

        sec_per_note, peak_ram = 0.0, 0.0
        if not args.skip_warm:
            sec_per_note, peak_ram = warm_cache_with_telemetry(model, limit=args.limit)

        result_path = run_evaluation(model)
        metrics = parse_result_json(result_path)
        records_map[model] = {
            "model": model,
            "sec_per_note": sec_per_note,
            "peak_ram": peak_ram,
            **metrics,
        }
        update_model_comparison_md(list(records_map.values()))

    print("\nModel comparison completed successfully!")


if __name__ == "__main__":
    main()

