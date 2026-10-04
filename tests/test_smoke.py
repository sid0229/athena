"""Batch 0 smoke tests: environment, data presence, scispaCy, local LLM."""

import json

import pytest

from athena.config import data_path, load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["llm"]["host"].startswith("http://localhost")
    assert cfg["llm"]["temperature"] == 0.0


@pytest.mark.parametrize(
    "key",
    ["n2c2", "mimic_demo", "drugbank_kaggle", "drugbank_benchmark", "ddinter", "faers"],
)
def test_raw_data_present(key):
    assert data_path(key).exists(), f"missing dataset: {data_path(key)}"


def test_scispacy_finds_drugs():
    import spacy

    nlp = spacy.load(load_config()["extraction"]["scispacy_model"])
    doc = nlp("Discharged on warfarin 5 mg PO daily; Lasix 40 mg BID.")
    drugs = {e.text.lower() for e in doc.ents if e.label_ == "CHEMICAL"}
    assert {"warfarin", "lasix"} <= drugs


def _ollama_client():
    ollama = pytest.importorskip("ollama")
    cfg = load_config()["llm"]
    client = ollama.Client(host=cfg["host"], timeout=cfg["timeout_s"])
    try:
        names = {m.model for m in client.list().models}
    except Exception:
        pytest.skip("Ollama server not running")
    if cfg["model"] not in names:
        pytest.skip(f"model {cfg['model']} not pulled")
    return client, cfg


@pytest.mark.ollama
def test_ollama_returns_schema_json():
    from pydantic import BaseModel

    class Med(BaseModel):
        drug: str
        strength: str | None
        frequency: str | None

    class Meds(BaseModel):
        medications: list[Med]

    client, cfg = _ollama_client()
    resp = client.chat(
        model=cfg["model"],
        messages=[
            {"role": "system", "content": "Extract medications from the text. Copy values verbatim."},
            {"role": "user", "content": "Continue metoprolol 25 mg twice daily and aspirin 81 mg daily."},
        ],
        format=Meds.model_json_schema(),
        options={"temperature": cfg["temperature"], "seed": cfg["seed"], "num_ctx": cfg["num_ctx"]},
    )
    meds = Meds.model_validate(json.loads(resp.message.content))
    found = {m.drug.lower() for m in meds.medications}
    assert {"metoprolol", "aspirin"} <= found
