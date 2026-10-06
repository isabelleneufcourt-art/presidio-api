FROM python:3.10-slim

WORKDIR /app

# Installation de Presidio, FastAPI et uvicorn
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    presidio-analyzer \
    spacy

# Téléchargement du modèle de langue français pour SpaCy
RUN python -m spacy download fr_core_news_sm

COPY main.py /app/main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
