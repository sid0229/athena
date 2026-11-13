"""Map a raw drug string (from a note or a prescription table) to canonical KB drug(s).

Deterministic, ordered stages; the first stage that matches wins. Every result
carries the stage name and a match score so that the fusion layer can use
normalisation quality as part of extraction confidence (README §2.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz, process

from athena.config import load_config
from athena.normalize.lexicon import load_lexicon, norm

# Common clinical abbreviations and brand names -> KB name (checked before cleaning).
# Targets were checked to exist in the lexicon (Batch 2). Ambiguous abbreviations
# (e.g. "ctx": ceftriaxone vs cyclophosphamide) are deliberately left out.
ABBREVIATIONS = {
    "asa": "aspirin", "apap": "acetaminophen", "hctz": "hydrochlorothiazide",
    "ntg": "nitroglycerin", "mso4": "morphine", "ms contin": "morphine",
    "mgso4": "magnesium sulfate", "kcl": "potassium chloride", "nacl": "sodium chloride",
    "pcn": "penicillin", "tpa": "alteplase", "ivig": "immune globulin",
    "coumadin": "warfarin", "lasix": "furosemide", "tylenol": "acetaminophen",
    "zosyn": "piperacillin + tazobactam", "vanc": "vancomycin", "vanco": "vancomycin",
    "dilaudid": "hydromorphone", "ativan": "lorazepam",
    "protonix": "pantoprazole", "zofran": "ondansetron", "reglan": "metoclopramide",
    "colace": "docusate", "percocet": "oxycodone + acetaminophen",
    "vicodin": "hydrocodone + acetaminophen", "bactrim": "sulfamethoxazole + trimethoprim",
    "lovenox": "enoxaparin", "plavix": "clopidogrel", "lipitor": "atorvastatin",
    "lithium": "lithium carbonate", "ferrous sulfate": "ferrous sulfate anhydrous",
    "fe sulfate": "ferrous sulfate anhydrous",
    "insulin": "insulin human", "regular insulin": "insulin human (regular)",
    "insulin regular": "insulin human (regular)", "insulin human regular": "insulin human (regular)",
    "insulin regular human": "insulin human (regular)", "insulin - sliding scale": "insulin human",
    "nph": "insulin human (isophane)", "insulin nph": "insulin human (isophane)",
    "nph insulin": "insulin human (isophane)", "insulin human nph": "insulin human (isophane)",
    "neutra-phos": "potassium phosphate",
    # From the most frequent unmapped n2c2 *train* mentions (test split not inspected).
    "glargine": "insulin glargine", "lantus": "insulin glargine", "iss": "insulin human",
    "ssi": "insulin human", "zantac": "ranitidine", "amio": "amiodarone",
    "azithro": "azithromycin", "epi": "epinephrine", "neosynephrine": "phenylephrine",
    "neo-synephrine": "phenylephrine", "cytoxan": "cyclophosphamide",
    "vit k": "phylloquinone", "vitamin k": "phylloquinone", "folate": "folic acid",
    "duoneb": "ipratropium + albuterol", "trazadone": "trazodone",
    "penicillin": "benzylpenicillin", "trimethoprim/sulfa": "sulfamethoxazole + trimethoprim",
    "tmp/smx": "sulfamethoxazole + trimethoprim", "piperacillin/tazo": "piperacillin + tazobactam",
    "pip/tazo": "piperacillin + tazobactam",
    # Code review (Nov 2026). "levo" removed: on wards it also means Levophed / levothyroxine.
    "insulin aspart": "insulin aspart (aspart)", "novolog": "insulin aspart (aspart)",
    "novolog mix": "insulin aspart (aspart)", "potassium": "potassium chloride", "k": "potassium chloride",
    "magnesium": "magnesium cation", "bactrim ds": "sulfamethoxazole + trimethoprim",
    "tylenol #3": "acetaminophen + codeine", "tylenol with codeine": "acetaminophen + codeine",
}

# Drug classes / generic references. n2c2 annotates these as Drug, but they name
# no specific agent, so they cannot be checked for interactions. Surfaced as
# status "drug_class" so the pharmacist still sees them.
DRUG_CLASSES = {
    "antibiotic", "antibiotics", "abx", "steroid", "steroids", "ppi", "ppis", "statin", "statins",
    "beta blocker", "beta blockers", "beta-blocker", "beta-blockers", "bb", "acei", "ace inhibitor",
    "ace inhibitors", "arb", "arbs", "nsaid", "nsaids", "narcotic", "narcotics", "opioid", "opioids",
    "opiates", "pressors", "pressor", "vasopressors", "anticoagulation", "anticoagulant",
    "anticoagulants", "antiplatelet", "chemotherapy", "chemo", "diuretic", "diuretics",
    "penicillins", "cephalosporins", "antihypertensives", "antihypertensive", "pain medication",
    "pain medications", "pain meds", "laxative", "laxatives", "bowel regimen", "sedation",
    "benzodiazepines", "benzos", "antiemetics", "antifungals", "antivirals", "immunosuppression",
    "iv antibiotics", "home medications", "medications", "meds", "vitamins",
    "multivitamin", "multivitamins", "mvi", "calcium", "electrolytes",
    "nebs", "nebulizers", "anesthesia", "analgesics", "analgesia", "anti-hypertensives",
    "phosphorus", "repletion", "lytes", "sliding scale",
}

# IV fluids, blood products, gases, contrast and supplies: real orders, but not
# drugs for DDI purposes.
NON_DRUG_PATTERNS = [
    r"^(d5w|d5 ?1/2 ?ns|d5ns|d5lr|d10w|lr|sw|ns|1/2 ?ns|ivf|ivfs|o2|oxygen)\b",
    r"\b(dextrose|sodium chloride|sterile water|lactated ringer|normocarb|iso-osmotic|normal saline|saline)\b",
    r"^(vial|bag|syringe|soln|flush|excel bag|mini bag|amp)\.?$",
    r"^(fluids?|iv fluids?|blood|blood products?|ffp|prbcs?|prbc's|rbcs?|platelets?|cryo|cryoprecipitate|contrast|iv contrast|d50|d50w)$",
    r"\bflush\b",
]
_NON_DRUG = [re.compile(p) for p in NON_DRUG_PATTERNS]

SALT_AND_FORM_WORDS = (
    "hcl|hydrochloride|hydrobromide|sodium|na|potassium|calcium|magnesium|sulfate|sulphate|"
    "tartrate|succinate|succ|citrate|maleate|mesylate|besylate|fumarate|acetate|phosphate|sod|"
    "bromide|chloride|hyclate|lactate|gluconate|disodium|dihydrate|monohydrate|"
    "syringe|neb|nebulizer|soln|solution|inj|injection|tab|tablet|tablets|cap|capsule|capsules|"
    "powder|liquid|oral|susp|suspension|ointment|cream|gel|lotion|patch|drops|enema|inhaler|hfa|"
    "spray|nasal|ophthalmic|ophth|topical|jelly|mdi|diskus|discus|respimat|flexpen|kwikpen|solostar|preserv|free|generic|rectal|er|sr|xl|xr|cr|dr|ec|odt|disintegrating|"
    "extended|release|immediate|delayed|chewable|pf|p\\.f|sl|iv|po|pr|im|sc|subq|pca|"
    "premix|desensitization|vial|bag|flush|mini|plus|excel|"
    # frequency / sig words that follow the drug name in medication lists
    "bid|tid|qid|qd|qod|qhs|qam|qpm|daily|prn|q\\d+h|q\\d+hr|ds|ir|po/ng|ng|gt|x"
)
# IV vehicle after a drug ("KCl 20 mEq in sodium chloride 0.9%"): the drug is the first part.
_VEHICLE_RE = re.compile(r"\s+in\s+(?:\d[\d.]*\s*%?\s*)?(?:sodium chloride|ns|normal saline|d5w|d5 ?1/2 ?ns|"
                         r"dextrose|sterile water|lr|lactated ringer)\b.*$")
_SALT_RE = re.compile(rf"\b({SALT_AND_FORM_WORDS})\b\.?")
_DOSE_RE = re.compile(r"(?<![a-z])\d[\d.,/-]*\s*(?:%|mg|mcg|meq|ml|g|gm|units?|l)?(?![a-z])")
_COMBO_SPLIT = re.compile(r"\s*(?:-|/|\+|\band\b|\bwith\b)\s*")


@dataclass(frozen=True)
class Normalized:
    raw: str
    drugs: tuple[str, ...]   # canonical KB names; empty if unmapped / non-drug
    stage: str               # which rule matched
    score: float             # 1.0 exact ... lower for fuzzy; 0 if unmapped
    status: str              # "mapped" | "ambiguous" | "drug_class" | "non_drug" | "unmapped"


def _clean(s: str) -> str:
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = s.replace(",", " ")
    s = _DOSE_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" -,.;:")


def _strip_salts(s: str) -> str:
    """Drop salt / formulation words after the first token.

    The first token is kept so that drugs whose name starts with a salt word
    ("sodium polystyrene sulfonate", "potassium chloride") are not mangled.
    """
    head, _, tail = s.partition(" ")
    tail = _SALT_RE.sub(" ", tail)
    return re.sub(r"\s+", " ", f"{head} {tail}").strip(" -,.;:")


class DrugNormalizer:
    def __init__(self, lexicon: dict | None = None, cfg: dict | None = None):
        lex = lexicon or load_lexicon()
        self.index: dict[str, list[str]] = lex["index"]
        self.canonical: list[str] = lex["canonical"]
        self.drugbank_id: dict[str, str] = lex["drugbank_id"]
        self.source: dict[str, str] = lex.get("source", {})
        full_cfg = cfg or load_config()
        self.fuzzy_threshold = full_cfg["normalize"]["fuzzy_match_threshold"]
        self.ablation = full_cfg.get("ablation", {})
        # Restricting typo matching to curated (non-Wikidata) names was tried in the code review:
        # it lost more correct matches (Keflex, Versed, Thorazine) than wrong ones, so it stays open.
        self._fuzzy_keys = [k for k in self.index if len(k) >= 5]

    def _lookup(self, s: str) -> list[str] | None:
        if " + " in s:  # abbreviation targets such as "piperacillin + tazobactam"
            return self._lookup_combo(s)
        return self.index.get(s)

    def _lookup_combo(self, s: str) -> list[str] | None:
        parts = [p for p in _COMBO_SPLIT.split(s) if p]
        if len(parts) < 2:
            return None
        found = []
        for p in parts:
            hit = self._lookup(p) or self._lookup(_strip_salts(p))
            if not hit or len(hit) != 1:
                return None
            found.append(hit[0])
        return found

    def normalize(self, raw: str) -> Normalized:
        s = norm(raw)
        if not s:
            return Normalized(raw, (), "empty", 0.0, "unmapped")
        s = _VEHICLE_RE.sub("", s) or s  # "kcl 20 meq in ns" -> "kcl 20 meq"

        if s in DRUG_CLASSES:
            return Normalized(raw, (), "drug_class", 1.0, "drug_class")
        if any(p.search(s) for p in _NON_DRUG):
            return Normalized(raw, (), "non_drug", 1.0, "non_drug")
        use_abbrev = self.ablation.get("normalizer_abbreviations", True)
        use_salt = self.ablation.get("normalizer_salt_stripping", True)
        use_fuzzy = self.ablation.get("normalizer_fuzzy", True)

        if use_abbrev:
            s = ABBREVIATIONS.get(s, s)
        if any(p.search(s) for p in _NON_DRUG):
            return Normalized(raw, (), "non_drug", 1.0, "non_drug")

        cleaned = _clean(s)
        stripped = _strip_salts(cleaned) if use_salt else cleaned
        # Abbreviations/brands before the index: "percocet 5/325" must reach the curated entry,
        # not an ambiguous index hit for the cleaned string.
        stages = [("exact", s, 1.0)]
        if use_abbrev:
            stages.append(("abbreviation", ABBREVIATIONS.get(cleaned) or "", 0.95))
        stages.append(("cleaned", cleaned, 0.95))
        if use_abbrev and use_salt:
            stages.append(("abbreviation", ABBREVIATIONS.get(stripped) or "", 0.9))
        if use_salt:
            stages.append(("salt_stripped", stripped, 0.9))

        for stage, cand, score in stages:
            if cand and (hit := self._lookup(cand)):
                return self._result(raw, hit, stage, score, self.source.get(cand))
        for stage, cand in ([("combination", cleaned)] + ([("combination", stripped)] if use_salt else [])):
            if hit := self._lookup_combo(cand):
                return Normalized(raw, tuple(sorted(set(hit))), stage, 0.9, "mapped")

        if use_fuzzy and len(stripped) >= 5:
            # Typo tolerance only: same first 4 letters, similar length.
            keys = [k for k in self._fuzzy_keys
                    if k[:4] == stripped[:4] and abs(len(k) - len(stripped)) <= 3]
            m = process.extractOne(stripped, keys, scorer=fuzz.ratio,
                                   score_cutoff=self.fuzzy_threshold)
            if m:
                return self._result(raw, self.index[m[0]], "fuzzy", round(m[1] / 100 * 0.85, 3), self.source.get(m[0]))

        return Normalized(raw, (), "unmapped", 0.0, "unmapped")

    @staticmethod
    def _result(raw: str, hit: list[str], stage: str, score: float, source: str | None = None) -> Normalized:
        if len(hit) == 1:
            return Normalized(raw, tuple(hit), stage, score, "mapped")
        if source in ("rxnorm", None):
            # RxNorm multi-ingredient entries are combination products (Entresto, Percocet,
            # Kaletra): every ingredient is checked. Code review found all of them were combos.
            return Normalized(raw, tuple(sorted(set(hit))), "combination_brand" if source else stage, score, "mapped")
        # Wikidata aliases pointing at several different drugs (e.g. "floxin") stay ambiguous.
        return Normalized(raw, tuple(hit), stage, score * 0.5, "ambiguous")
