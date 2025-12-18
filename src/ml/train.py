import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score


def train_model(cfg: dict):
    train_cfg = cfg["train"]

    df = pd.read_csv(cfg["dataset"]["output_path"])

    text_col = "processed_text"
    
    X = df[text_col].astype(str)
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_cfg.get("test_size", 0.2),
        random_state=train_cfg.get("random_state", 42),
        stratify=y
    )

    if train_cfg.get("vectorizer", "tfidf") == "tfidf":
        vectorizer = TfidfVectorizer(
            max_features=train_cfg.get("max_features", 20000),
            ngram_range=tuple(train_cfg.get("ngram_range", [1, 2]))
        )
    elif train_cfg.get("vectorizer") == "count":
        vectorizer = CountVectorizer(
            max_features=train_cfg.get("max_features", 10000),
            ngram_range=tuple(train_cfg.get("ngram_range", [1, 2]))
        )
    else:
        raise ValueError(f"Unknown vectorizer type: {train_cfg.get('vectorizer')}")

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    if train_cfg.get("model_type", "logreg") == "logreg":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif train_cfg.get("model_type") == "nb":
        model = MultinomialNB(class_prior=[0.2, 0.6, 0.2])
    else:
        raise ValueError(f"Unknown model type: {train_cfg.get('model_type')}")

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        mlflow.log_param("model", train_cfg.get("model_type", "logreg"))
        mlflow.log_param("max_features", train_cfg.get("max_features"))
        mlflow.log_param("ngram_range", train_cfg.get("ngram_range"))

        mlflow.log_metric("accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        print("Accuracy:", acc)
        print(report)

    joblib.dump(model, train_cfg["model_path"])
    joblib.dump(vectorizer, train_cfg["vectorizer_path"])

    print("Model saved to:", train_cfg["model_path"])
    print("Vectorizer saved to:", train_cfg["vectorizer_path"])


if __name__ == "__main__":
    from src.utils.config import load_config
    cfg = load_config()
    train_model(cfg)
