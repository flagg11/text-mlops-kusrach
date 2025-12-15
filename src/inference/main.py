from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from ..text_prep.labeling import hybrid_sentiment_label, LEXICON

app = FastAPI(
    title="VK Comments Sentiment API",
    description="API для анализа тональности комментариев",
    version="0.2.0"
)

class PredictRequest(BaseModel):
    comments: list[str]

class PredictResponse(BaseModel):
    predictions: list[str]

model = joblib.load("models/lr_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    preds = []
    for c in req.comments:
        pred = hybrid_sentiment_label(c, model=model, vectorizer=vectorizer)
        preds.append(pred)
    return {"predictions": preds}
