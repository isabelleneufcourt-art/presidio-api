from typing import List, Optional

import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

LANG = "fr"

# Modèle français installé dans l'image (md = meilleure détection des noms que sm)
MODEL_NAME = next(
    (m for m in ("fr_core_news_md", "fr_core_news_sm") if spacy.util.is_package(m)), None
)
if MODEL_NAME is None:
    raise RuntimeError("Aucun modèle spaCy français installé (fr_core_news_md ou fr_core_news_sm)")

# Configuration du moteur NLP pour utiliser la langue française
nlp_configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": LANG, "model_name": MODEL_NAME}],
}

provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
nlp_engine = provider.create_engine()
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[LANG])


def _pattern(entity: str, regexes: List[str], score: float) -> PatternRecognizer:
    return PatternRecognizer(
        supported_entity=entity,
        supported_language=LANG,
        patterns=[Pattern(f"{entity.lower()}_{i}", rx, score) for i, rx in enumerate(regexes)],
    )


# Recognizers par expressions régulières (France / Belgique), absents du modèle NER
CUSTOM_RECOGNIZERS = [
    _pattern("EMAIL_ADDRESS", [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"], 0.9),
    _pattern(
        "PHONE_NUMBER",
        [
            r"(?<!\w)(?:\+|00)\d{2}[\s.]?\(?0?\)?\s?\d(?:[\s./-]?\d){7,9}(?!\d)",
            r"(?<!\d)0\d{1,2}(?:[\s./-]?\d{2}){3,4}(?!\d)",
        ],
        0.7,
    ),
    _pattern("IBAN_CODE", [r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,3})?\b"], 0.85),
    _pattern("CREDIT_CARD", [r"\b(?:\d{4}[\s-]?){3}\d{4}\b"], 0.6),
    _pattern(
        "FR_SSN",
        [r"(?<!\d)[12]\s?\d{2}\s?(?:0[1-9]|1[0-2])\s?\d{2}\s?\d{3}\s?\d{3}(?:\s?\d{2})?(?!\d)"],
        0.7,
    ),
    _pattern("BE_NATIONAL_NUMBER", [r"\b\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}\b"], 0.9),
    _pattern("VAT_NUMBER", [r"\b(?:BE|FR)\s?0?\d{3}[\s.]?\d{3}[\s.]?\d{3}\b"], 0.8),
    _pattern("URL", [r"https?://[^\s<>\"]+", r"\bwww\.[^\s<>\"]+"], 0.8),
    _pattern(
        "IP_ADDRESS",
        [r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"],
        0.8,
    ),
]
for recognizer in CUSTOM_RECOGNIZERS:
    analyzer.registry.add_recognizer(recognizer)

DEFAULT_ENTITIES = ["PERSON", "LOCATION", "ORGANIZATION"] + [
    r.supported_entities[0] for r in CUSTOM_RECOGNIZERS
]

app = FastAPI()


class TextRequest(BaseModel):
    text: str
    language: str = LANG
    entities: Optional[List[str]] = None
    score_threshold: float = 0.4


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_text(req: TextRequest):
    if req.language != LANG:
        raise HTTPException(400, f"Langue non supportée : '{req.language}' (seulement '{LANG}')")

    results = analyzer.analyze(
        text=req.text,
        entities=req.entities or DEFAULT_ENTITIES,
        language=LANG,
        score_threshold=req.score_threshold,
    )

    entities = []
    for res in results:
        entities.append({
            "type": res.entity_type,
            "entity_type": res.entity_type,  # compatible avec le format Presidio standard
            "start": res.start,
            "end": res.end,
            "text": req.text[res.start:res.end],
            "score": res.score,
        })

    return {"entities": entities}
