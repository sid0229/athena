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
    def __init__(self, model: str | None = None, use_cache: bool = True):
        cfg = load_config()["llm"]
        self.model = model or cfg["model"]
        self.options = {"temperature": cfg["temperature"], "seed": cfg["seed"], "num_ctx": cfg["num_ctx"],
                        # Cap output: a stuck model repeating rows would otherwise run to num_ctx.
                        "num_predict": cfg.get("num_predict", 1500)}
        self.client = ollama.Client(host=cfg["host"], timeout=cfg["timeout_s"] * 3)
        self.system = _prompt()
        self.cache_dir = data_path("processed") / "llm_cache" / f"{self.model.replace(':', '_')}_{PROMPT_VERSION}"
        self.use_cache = use_cache
        self.stats = {"calls": 0, "cache_hits": 0, "parse_failures": 0, "seconds": 0.0, "gen_tokens": 0}

    def _cache_path(self, text: str) -> Path:
        h = hashlib.sha256(f"{self.model}|{PROMPT_VERSION}|{text}".encode()).hexdigest()[:24]
        return self.cache_dir / f"{h}.json"

    def extract_chunk(self, text: str) -> list[RawMed]:
        path = self._cache_path(text)
        if self.use_cache and path.exists():
            self.stats["cache_hits"] += 1
            rows = json.loads(path.read_text())
        else:
            rows = self._call(text)
            if self.use_cache and rows is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows))
            rows = rows or []
        meds = []
        for row in rows:
            if len(row) != len(FIELDS) or not str(row[0]).strip():
                continue
            meds.append(RawMed(*[str(v).strip() for v in row]))
        return meds

    def _call(self, text: str) -> list[list[str]] | None:
        for attempt in range(2):
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "system", "content": self.system}, {"role": "user", "content": text}],
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
