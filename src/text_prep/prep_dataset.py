import pandas as pd

'''import sys
print("SYS.PATH =", sys.path)
'''

from labeling import sentiment_label
from lemmatization import lemmatize_text


df = pd.read_csv("data/clean/clean_comments.csv")

df["lemmatized"] = df["clean_text"].astype(str).apply(lemmatize_text)
df["sentiment"] = df["lemmatized"].apply(sentiment_label)

df = df[["lemmatized", "sentiment"]]

df.to_csv("data/labeled/comments.csv", index=False)
print(df["sentiment"].value_counts())

