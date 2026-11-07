"""High-alert drug combinations: a small, reviewed rule table that always grades a pair Major.

Why: the KB grades severity from DDInter, else derives it from DrugBank sentence templates, and that
derivation is nearly always "Moderate" (code review, Nov 2026: held-out Major recall 8%). Classic
dangerous pairs therefore landed in Moderate/Unknown (fluoxetine + phenelzine, sildenafil +
nitroglycerin, lithium + hydrochlorothiazide). Each rule names its clinical effect and source, and is
shown to the reviewer as the reason. Rules only RAISE severity; they never lower or hide a record.

Names are KB concept names (as in data/processed/interactions.db).
"""

from __future__ import annotations

from dataclasses import dataclass

MAOI = {"phenelzine", "tranylcypromine", "isocarboxazid", "selegiline", "rasagiline", "moclobemide",
        "linezolid", "methylene blue"}
SEROTONERGIC = {"fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram", "fluvoxamine",
                "venlafaxine", "desvenlafaxine", "duloxetine", "milnacipran", "amitriptyline", "nortriptyline",
                "imipramine", "clomipramine", "desipramine", "doxepin", "tramadol", "meperidine", "methadone",
                "fentanyl", "tapentadol", "dextromethorphan", "sumatriptan", "rizatriptan", "zolmitriptan",
                "buspirone", "mirtazapine", "trazodone", "lithium carbonate", "st. john's wort"}
SSRI_SNRI = {"fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram", "fluvoxamine",
             "venlafaxine", "desvenlafaxine", "duloxetine", "milnacipran"}
NITRATES = {"nitroglycerin", "isosorbide mononitrate", "isosorbide dinitrate", "isosorbide", "amyl nitrite"}
PDE5 = {"sildenafil", "tadalafil", "vardenafil", "avanafil", "riociguat"}
OPIOIDS = {"morphine", "hydromorphone", "oxycodone", "hydrocodone", "codeine", "oxymorphone", "buprenorphine",
           "fentanyl", "methadone", "tramadol", "tapentadol", "meperidine"}
BENZODIAZEPINES = {"alprazolam", "lorazepam", "diazepam", "clonazepam", "midazolam", "temazepam",
                   "chlordiazepoxide", "oxazepam", "triazolam", "clobazam"}
DIURETICS = {"hydrochlorothiazide", "chlorthalidone", "indapamide", "metolazone", "furosemide", "bumetanide",
             "torasemide"}
ACEI_ARB = {"lisinopril", "enalapril", "captopril", "ramipril", "benazepril", "quinapril", "fosinopril",
            "perindopril", "trandolapril", "losartan", "valsartan", "irbesartan", "candesartan cilexetil",
            "olmesartan", "telmisartan", "sacubitril"}
NSAIDS = {"ibuprofen", "naproxen", "diclofenac", "ketorolac", "indomethacin", "meloxicam", "celecoxib"}
POTASSIUM_RAISING = {"spironolactone", "eplerenone", "amiloride", "triamterene", "potassium chloride",
                     "trimethoprim"}
STRONG_CYP3A4_INHIBITORS = {"clarithromycin", "itraconazole", "ketoconazole", "posaconazole", "voriconazole",
                            "ritonavir", "cobicistat"}
CYP3A4_NARROW = {"simvastatin", "lovastatin", "colchicine"}
QT_KNOWN_RISK = {"amiodarone", "sotalol", "dofetilide", "dronedarone", "haloperidol", "ziprasidone", "pimozide",
                 "thioridazine", "methadone", "citalopram", "escitalopram", "ondansetron", "azithromycin",
                 "clarithromycin", "erythromycin", "levofloxacin", "moxifloxacin", "ciprofloxacin",
                 "hydroxychloroquine", "chloroquine"}


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    a: frozenset[str]
    b: frozenset[str]
    effect: str
    source: str

    def matches(self, x: str, y: str) -> bool:
        return (x in self.a and y in self.b) or (y in self.a and x in self.b)


def _r(id_, name, a, b, effect, source):
    return Rule(id_, name, frozenset(a), frozenset(b), effect, source)


RULES: tuple[Rule, ...] = (
    _r("maoi_serotonergic", "MAOI + serotonergic drug", MAOI, SEROTONERGIC - MAOI,
       "serotonin syndrome / hypertensive crisis; combination contraindicated",
       "FDA labelling (MAOIs); ONC high-priority DDI list (Phansalkar et al., JAMIA 2012)"),
    _r("nitrate_pde5", "Nitrate + PDE5 inhibitor / riociguat", NITRATES, PDE5,
       "severe hypotension; combination contraindicated", "FDA labelling (sildenafil, tadalafil, riociguat)"),
    _r("opioid_benzo", "Opioid + benzodiazepine", OPIOIDS, BENZODIAZEPINES,
       "profound sedation, respiratory depression, death", "FDA boxed warning (2016)"),
    _r("lithium_renal", "Lithium + diuretic / ACE inhibitor / ARB / NSAID", {"lithium carbonate"},
       DIURETICS | ACEI_ARB | NSAIDS, "reduced lithium clearance and lithium toxicity", "FDA labelling (lithium)"),
    _r("hyperkalaemia", "Potassium-raising drug + ACE inhibitor / ARB", POTASSIUM_RAISING, ACEI_ARB,
       "hyperkalaemia", "FDA labelling (spironolactone, eplerenone, ACE inhibitors)"),
    _r("mtx_tmp", "Methotrexate + trimethoprim", {"methotrexate"}, {"trimethoprim"},
       "bone-marrow suppression", "ONC high-priority DDI list (Phansalkar et al., JAMIA 2012)"),
    _r("cyp3a4_statin_colchicine", "Strong CYP3A4 inhibitor + simvastatin / lovastatin / colchicine",
       STRONG_CYP3A4_INHIBITORS, CYP3A4_NARROW, "rhabdomyolysis (statins) / fatal colchicine toxicity",
       "FDA labelling (simvastatin, lovastatin, colchicine); ONC high-priority DDI list"),
    _r("tramadol_ssri", "Tramadol + SSRI / SNRI", {"tramadol"}, SSRI_SNRI,
       "serotonin syndrome and lowered seizure threshold", "FDA labelling (tramadol)"),
    _r("qt_qt", "Two drugs with known risk of torsades de pointes", QT_KNOWN_RISK, QT_KNOWN_RISK,
       "additive QT prolongation, torsades de pointes", "CredibleMeds 'known risk of TdP' list"),
    _r("clopidogrel_ppi", "Clopidogrel + omeprazole / esomeprazole", {"clopidogrel"}, {"omeprazole", "esomeprazole"},
       "reduced antiplatelet effect (CYP2C19 inhibition)", "FDA labelling (clopidogrel)"),
    _r("allopurinol_thiopurine", "Allopurinol / febuxostat + azathioprine / mercaptopurine",
       {"allopurinol", "febuxostat"}, {"azathioprine", "mercaptopurine"},
       "thiopurine toxicity (xanthine oxidase inhibition)", "FDA labelling (azathioprine, febuxostat)"),
)


def match(name_a: str, name_b: str) -> Rule | None:
    """First rule that matches a pair of KB concept names (route-specific names are ignored)."""
    if name_a == name_b:
        return None
    return next((r for r in RULES if r.matches(name_a, name_b)), None)
