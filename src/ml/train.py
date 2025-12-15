import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def train_model(cfg: dict):
    train_cfg = cfg["train"]

    df = pd.read_csv(train_cfg["data_path"])

    X = df["lemmatized"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=train_cfg["max_features"],
        ngram_range=tuple(train_cfg["ngram_range"])
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)

    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        model.fit(X_train_vec, y_train)

        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_features", train_cfg["max_features"])
        mlflow.log_param("ngram_range", train_cfg["ngram_range"])

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
