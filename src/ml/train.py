import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import joblib
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/labeled/comments.csv")
X = df["lemmatized"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

mlflow.set_experiment("sentiment_analysis")
with mlflow.start_run(run_name="logreg_tfidf"):

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    lr = LogisticRegression(max_iter=500, class_weight='balanced')
    lr.fit(X_train_tfidf, y_train)
    preds = lr.predict(X_test_tfidf)

    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))

    joblib.dump(lr, "models/lr_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

    mlflow.sklearn.log_model(lr, "lr_model")
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("ngram_range", "(1,3)")
    mlflow.log_param("max_features", 10000)
    mlflow.log_metric("accuracy", acc)
