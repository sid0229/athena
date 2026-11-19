"""Branch A, LLM part: local Ollama model -> raw medication rows per chunk.

The LLM only *extracts*. It never decides whether an interaction exists or how
severe it is (README §1). Every value it returns is later grounded in the note
text by `align.py`; values that cannot be found there are dropped.

Output is a compact JSON-schema-constrained list of 8-string rows (no field
names), which is ~3x faster to generate than named fields on llama3.2:3b.
Results are cached on disk keyed by (model, prompt version, chunk text).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import ollama

from athena.config import data_path, load_config

PROMPT_VERSION = "extract_v1"
FIELDS = ("drug", "strength", "dosage", "form", "route", "frequency", "duration", "status")
_SCHEMA = {
    "type": "object",
    "properties": {
        "meds": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"},
                      "minItems": len(FIELDS), "maxItems": len(FIELDS)},
        }
    },
    "required": ["meds"],
}


@dataclass(frozen=True)
class RawMed:
    drug: str
    strength: str = ""
    dosage: str = ""
    form: str = ""
    route: str = ""
    frequency: str = ""
    duration: str = ""
    status: str = ""

    def attributes(self) -> dict[str, str]:
        return {f: getattr(self, f) for f in FIELDS[1:-1] if getattr(self, f).strip()}


_ROW = re.compile(r'\[\s*"(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")*\s*\]')


def _salvage_rows(content: str) -> list[list[str]]:
    """Complete rows from truncated output (e.g. cut off by num_predict), de-duplicated."""
    rows, seen = [], set()
    for m in _ROW.finditer(content or ""):
        try:
            row = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if len(row) == len(FIELDS) and tuple(row) not in seen:
            seen.add(tuple(row))
            rows.append(row)
    return rows


def _prompt() -> str:
    return (Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.txt").read_text()


class OllamaExtractor:
    def __init__(
        self,
        model: str | None = None,
        use_cache: bool = True,
        use_corrections: bool | None = None,
        retriever: Any = None,
    ):
        cfg = load_config()
        llm_cfg = cfg["llm"]
        ext_cfg = cfg.get("extraction", {})
        self.model = model or llm_cfg["model"]
        self.options = {"temperature": llm_cfg["temperature"], "seed": llm_cfg["seed"], "num_ctx": llm_cfg["num_ctx"],
                        # Cap output: a stuck model repeating rows would otherwise run to num_ctx.
                        "num_predict": llm_cfg.get("num_predict", 1500)}
        self.client = ollama.Client(host=llm_cfg["host"], timeout=llm_cfg["timeout_s"] * 3)
        self.system = _prompt()
        self.use_cache = use_cache

        self.use_corrections = ext_cfg.get("use_corrections", False) if use_corrections is None else use_corrections
        self.retriever = None
        self.correction_bank_version = "v0_none"
        if self.use_corrections:
            from athena.extraction.corrections_fewshot import CorrectionRetriever
            self.retriever = retriever or CorrectionRetriever()
            self.correction_bank_version = self.retriever.version
            cache_subdir = f"{self.model.replace(':', '_')}_{PROMPT_VERSION}_corr_{self.correction_bank_version}"
        else:
            cache_subdir = f"{self.model.replace(':', '_')}_{PROMPT_VERSION}"

        self.cache_dir = data_path("processed") / "llm_cache" / cache_subdir
        self.stats = {"calls": 0, "cache_hits": 0, "parse_failures": 0, "seconds": 0.0, "gen_tokens": 0}

    def _cache_signature(self) -> dict:
        """What the cached outputs depend on besides (model, prompt version, chunk text)."""
        sig = {
            "prompt_sha256": hashlib.sha256(self.system.encode()).hexdigest(),
            "options": {k: self.options[k] for k in ("temperature", "num_ctx", "num_predict")},
        }
        if self.use_corrections:
            sig["use_corrections"] = True
            sig["correction_bank_version"] = self.correction_bank_version
        return sig

    def check_cache(self) -> None:
        """Refuse stale cache: editing the prompt file or decoding options without bumping
        PROMPT_VERSION would otherwise silently reuse old outputs (code review, Nov 2026)."""
        if not self.use_cache:
            return
        meta = self.cache_dir / "_signature.json"
        sig = self._cache_signature()
        if meta.exists():
            old = json.loads(meta.read_text())
            if old != sig:
                raise RuntimeError(f"LLM cache {self.cache_dir.name} was made with a different prompt or options "
                                   f"— bump PROMPT_VERSION (or delete the cache). {old} != {sig}")
        else:
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps(sig, indent=1))

    def _cache_path(self, text: str) -> Path:
        token = f"{self.model}|{PROMPT_VERSION}|{text}"
        if self.use_corrections:
            token = f"{token}|corr_{self.correction_bank_version}"
        h = hashlib.sha256(token.encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def extract_chunk(self, text: str) -> list[RawMed]:
        if not getattr(self, "_cache_checked", False):
            self.check_cache()
            self._cache_checked = True
        path = self._cache_path(text)
        if self.use_cache and path.exists():
            self.stats["cache_hits"] += 1
            rows = json.loads(path.read_text())
        else:
            rows = self._call(text)
            if self.use_cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows if rows is not None else []))
            rows = rows or []
        meds = []
        for row in rows:
            if len(row) != len(FIELDS) or not str(row[0]).strip():
                continue
            meds.append(RawMed(*[str(v).strip() for v in row]))
        return meds

    def _call(self, text: str) -> list[list[str]] | None:
        messages = [{"role": "system", "content": self.system}]
        if self.use_corrections and self.retriever:
            examples = self.retriever.retrieve(text, k=2)
            for ex_text, rows in examples:
                messages.append({"role": "user", "content": ex_text})
                messages.append({"role": "assistant", "content": json.dumps({"meds": rows})})
        messages.append({"role": "user", "content": text})

        # A retry with another seed only differs when sampling; at temperature 0 it repeats itself.
        for attempt in range(2 if self.options["temperature"] > 0 else 1):
            resp = self.client.chat(
                model=self.model,
                messages=messages,
                format=_SCHEMA,
                options=self.options if attempt == 0 else {**self.options, "seed": self.options["seed"] + 1},
            )
            self.stats["calls"] += 1
            self.stats["seconds"] += (resp.total_duration or 0) / 1e9
            self.stats["gen_tokens"] += resp.eval_count or 0
            try:
                return json.loads(resp.message.content)["meds"]
            except (json.JSONDecodeError, KeyError, TypeError):
                self.stats["parse_failures"] += 1
                rows = _salvage_rows(resp.message.content)
                if rows:
                    self.stats["salvaged"] = self.stats.get("salvaged", 0) + 1
                    return rows
        return None

    def available(self) -> bool:
        try:
            return self.model in {m.model for m in self.client.list().models}
        except Exception:
            return False
