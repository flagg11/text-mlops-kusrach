from fastapi import FastAPI
import joblib

from src.inference.schemas import PredictRequest, PredictResponse
from src.utils.config import load_config

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
    X = vectorizer.transform(request.comments)
    preds = model.predict(X)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        class_to_index = {c: idx for idx, c in enumerate(model.classes_)}
        confidences = [round(float(probs[i][class_to_index[preds[i]]]), 3) for i in range(len(preds))]
    else:
        confidences = [None] * len(preds)



    return {
        "predictions": preds.tolist(),
        "confidences": confidences
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config['inference']['host'], port=config['inference']['port'])
