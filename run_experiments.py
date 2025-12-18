import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from src.utils.config import load_config
import os
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"


def run_experiments():
    cfg = load_config()
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    experiments = cfg.get("experiments", [])
    output_dir = Path(cfg.get("output_dir", "models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        print("=" * 80)
        print(f"Running experiment: {exp['name']}")

        df = pd.read_csv(exp["dataset_path"])
        text_col = "processed_text"
        X = df[text_col]
        y = df["sentiment"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if exp["vectorizer"] == "tfidf":
            vectorizer = TfidfVectorizer(
                max_features=exp["max_features"],
                ngram_range=tuple(exp["ngram_range"])
            )
        else:
            vectorizer = CountVectorizer(
                max_features=exp["max_features"],
                ngram_range=tuple(exp["ngram_range"])
            )

        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        if exp["model_type"] == "logreg":
            model = LogisticRegression(max_iter=1000)
        else:
            model = MultinomialNB()

        with mlflow.start_run(run_name=exp["name"]):
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred))

            mlflow.log_param("model", exp["model_type"])
            mlflow.log_param("vectorizer", exp["vectorizer"])
            mlflow.log_param("max_features", exp["max_features"])
            mlflow.log_param("ngram_range", exp["ngram_range"])
            mlflow.log_param("dataset", exp["dataset_path"])
            mlflow.log_param("preprocessing", exp["preprocessing"])

            mlflow.log_metric("accuracy", acc)

            mlflow.sklearn.log_model(model, "model")

            model_path = output_dir / f"{exp['name']}_model.pkl"
            vec_path = output_dir / f"{exp['name']}_vectorizer.pkl"
            joblib.dump(model, model_path)
            joblib.dump(vectorizer, vec_path)

            print(f"Saved model to {model_path}")
            print(f"Saved vectorizer to {vec_path}")


if __name__ == "__main__":
    run_experiments()
