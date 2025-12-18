from fastapi import FastAPI
import joblib

from src.inference.schemas import PredictRequest, PredictResponse
from src.utils.config import load_config
from src.text_prep import labeling

config = load_config()
model_path = config["train"]["model_path"]
vectorizer_path = config["train"]["vectorizer_path"]

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

app = FastAPI(title="Анализ комментов ВК")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    predictions = [
        labeling.hybrid_sentiment_label(comment, model=model, vectorizer=vectorizer)
        for comment in request.comments
    ]
    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config['inference']['host'], port=config['inference']['port'])
