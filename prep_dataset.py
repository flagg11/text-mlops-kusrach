import pandas as pd
from labeling import sentiment_label

df = pd.read_csv("data/clean/clean_comments.csv")

df["sentiment"] = df["clean_text"].astype(str).apply(sentiment_label)

df = df[["clean_text", "sentiment"]]

df.to_csv("data/labeled/comments.csv", index=False)

print(df["sentiment"].value_counts())
