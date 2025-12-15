from fastapi import FastAPI
import pickle

from src.inference.schemas import PredictRequest, PredictResponse
from src.utils.config import load_config

app = FastAPI()

config = load_config()
model_path = config["train"]["model_path"]
vectorizer_path = config["train"]["vectorizer_path"]

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    X = vectorizer.transform(request.comments)
    preds = model.predict(X)
    return {"predictions": preds.tolist()}
