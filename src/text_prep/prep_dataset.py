import pandas as pd

from src.text_prep.lemmatization import lemmatize_text
from src.text_prep.labeling import hybrid_sentiment_label


def prep_dataset(
    input_path: str = "data/clean/clean_comments.csv",
    output_path: str = "data/labeled/comments.csv"
):
    df = pd.read_csv(input_path)

    df["clean_text"] = df["clean_text"].astype(str)
    df["lemmatized"] = df["clean_text"].apply(lemmatize_text)
    df["sentiment"] = df["lemmatized"].apply(hybrid_sentiment_label)

    df = df[["lemmatized", "sentiment"]]

    df.to_csv(output_path, index=False)

    print("Dataset prepared:")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    prep_dataset()
