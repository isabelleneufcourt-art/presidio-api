from fastapi import FastAPI
from pydantic import BaseModel
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

# Configuration du moteur NLP pour utiliser la langue française
nlp_configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "fr", "model_name": "fr_core_news_sm"}]
}

provider = NlpEngineProvider(nlp_config=nlp_configuration)
nlp_engine = provider.create_engine()
analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["fr"])

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(req: TextRequest):
    # Détection des noms de personnes, lieux et organisations
    results = analyzer.analyze(
        text=req.text,
        entities=["PERSON", "LOCATION", "ORGANIZATION"],
        language="fr",
        score_threshold=0.6
    )
    
    entities = []
    for res in results:
        entities.append({
            "type": res.entity_type,
            "start": res.start,
            "end": res.end,
            "text": req.text[res.start:res.end],
            "score": res.score
        })
        
    return {"entities": entities}