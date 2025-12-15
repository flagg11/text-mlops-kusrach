import pickle
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from src.utils.config import load_config


def main():
    config = load_config()

    train_cfg = config["train"]
    mlflow_cfg = config["mlflow"]

    df = pd.read_csv(train_cfg["data_path"])
    X = df[train_cfg["text_column"]]
    y = df[train_cfg["target_column"]]

    with open(train_cfg["vectorizer_path"], "rb") as f:
        vectorizer = pickle.load(f)

    X_vec = vectorizer.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"]
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with open(train_cfg["model_path"], "wb") as f:
        pickle.dump(model, f)

    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run():
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("test_size", train_cfg["test_size"])
        mlflow.log_metric("accuracy", acc)

        mlflow.log_text(report, "classification_report.txt")
        mlflow.sklearn.log_model(model, "model")

    print(f"Training finished, accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
